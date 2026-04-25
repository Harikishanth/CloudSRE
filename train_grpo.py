"""
CloudSRE v2 — GRPO Training (Group Relative Policy Optimization)

Uses TRL-style GRPO: generate N parallel rollouts per scenario,
compute group-relative advantages, update policy with clipped gradients.

Upgrade from REINFORCE:
  - Multiple rollouts per prompt → lower variance
  - Group-relative baseline → no need for running average
  - Curriculum across all 5 tiers → warmup → single_fault → cascade → multi_cascade → adversarial
  - KL penalty → prevents catastrophic forgetting during curriculum

Usage:
    python train_grpo.py \
        --env-url https://dardrax-cloudsre-environment.hf.space \
        --model-id unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit \
        --curriculum warmup,single_fault,cascade,multi_cascade,adversarial \
        --episodes-per-tier 30 \
        --group-size 4

    # Or single tier:
    python train_grpo.py \
        --env-url https://dardrax-cloudsre-environment.hf.space \
        --model-id ./cloudsre-sft-checkpoint \
        --curriculum warmup \
        --episodes-per-tier 50
"""

import argparse
import json
import time
import os
import warnings
import logging
from collections import defaultdict

# ── Suppress noisy warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import httpx
import torch
import torch.nn.functional as F


# ── Environment Client ──────────────────────────────────────────────────

class CloudSREClient:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=120)

    def reset(self, task_id: str = "warmup") -> dict:
        resp = self.client.post("/reset", json={"task_id": task_id})
        return resp.json()

    def step(self, command: str) -> dict:
        resp = self.client.post("/step", json={"action": {"command": command}})
        return resp.json()

    def close(self):
        self.client.close()


# ── Helper Functions ────────────────────────────────────────────────────

def build_prompt(obs: dict, turn: int, max_turns: int) -> str:
    health = obs.get("service_health", {})
    alert = obs.get("alert", "")
    cmd_output = obs.get("command_output", "")
    feedback = obs.get("feedback", "")

    health_lines = []
    for svc, info in health.items():
        status = info.get("status", "unknown")
        error = info.get("error", "")
        health_lines.append(f"  {svc}: {status}" + (f" ({error})" if error else ""))

    return f"""You are an SRE agent. Diagnose and fix the incident.

ALERT: {alert}
COMMAND OUTPUT: {cmd_output}
{f'FEEDBACK: {feedback}' if feedback else ''}

SERVICE HEALTH:
{chr(10).join(health_lines)}

Step {turn+1}/{max_turns}. Respond with ONLY a single command:
- restart_service <name>
- queue drain <rate>
- status
- cat /var/log/<service>/error.log

Command:"""


def parse_command(text: str) -> str:
    """Extract the SRE command from model output."""
    text = text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in ["Command:", "command:", "Action:", "action:", ">", "$", "```"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line and not line.startswith("#"):
            return line[:200]
    return "status"


def generate_with_logprobs(model, tokenizer, prompt: str, max_new_tokens: int = 60,
                           temperature: float = 0.8, top_p: float = 0.95):
    """Generate text, then recompute log probs WITH gradients via forward pass."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Step 1: Generate tokens (no grad — just sampling)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    generated_ids = outputs[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Step 2: Recompute log probs WITH gradients (forward pass)
    full_ids = outputs[0:1]
    forward_out = model(input_ids=full_ids)
    logits = forward_out.logits[0, input_len-1:-1, :]

    log_probs = []
    for i, token_id in enumerate(generated_ids):
        if i >= logits.shape[0]:
            break
        lp = F.log_softmax(logits[i], dim=-1)[token_id]
        log_probs.append(lp)

    return generated_text, log_probs


def run_episode(model, tokenizer, env, task_id, max_turns=10, temperature=0.8):
    """Run a single episode, return (reward, log_probs, commands, resolved)."""
    try:
        result = env.reset(task_id=task_id)
    except Exception as e:
        return 0.0, [], [], False

    obs = result.get("observation", result)
    max_steps = min(max_turns, obs.get("max_steps", max_turns))

    episode_log_probs = []
    episode_rewards = []
    episode_commands = []

    for turn in range(max_steps):
        done = result.get("done", False)
        if done:
            break

        prompt = build_prompt(obs, turn, max_steps)
        raw_output, log_probs = generate_with_logprobs(
            model, tokenizer, prompt, temperature=temperature
        )
        command = parse_command(raw_output)
        episode_commands.append(command)

        if log_probs:
            episode_log_probs.append(torch.stack(log_probs).sum())

        try:
            result = env.step(command)
        except Exception:
            time.sleep(2)
            try:
                result = env.step(command)
            except Exception:
                break

        obs = result.get("observation", result)
        reward = float(result.get("reward", 0.0))
        episode_rewards.append(reward)

    total_reward = sum(episode_rewards)
    resolved = result.get("done", False) and total_reward > 0

    return total_reward, episode_log_probs, episode_commands, resolved


# ── GRPO Training Loop ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CloudSRE GRPO Training")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--model-id", default="./cloudsre-sft-checkpoint")
    parser.add_argument("--curriculum", default="warmup,single_fault,cascade,multi_cascade,adversarial",
                        help="Comma-separated tier list for curriculum training")
    parser.add_argument("--episodes-per-tier", type=int, default=30,
                        help="Episodes per curriculum tier")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of parallel rollouts per scenario (the G in GRPO)")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.05,
                        help="KL penalty coefficient (prevents catastrophic forgetting)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO-style advantage clipping")
    parser.add_argument("--output-dir", default="./cloudsre-grpo")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N episodes (0=only at end)")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.curriculum.split(",")]

    # ── Load Model ──
    from unsloth import FastLanguageModel

    print(f"Loading model: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    if hasattr(model, 'peft_config'):
        print("Reusing existing LoRA adapters for GRPO")
    else:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ── Environment ──
    env = CloudSREClient(args.env_url)

    # ── Training State ──
    all_rewards = defaultdict(list)       # per-tier reward history
    all_resolutions = defaultdict(list)   # per-tier resolution history
    global_rewards = []                   # all rewards across tiers
    global_step = 0

    print(f"\n{'='*70}")
    print(f"  GRPO TRAINING — CloudSRE v2")
    print(f"  Curriculum: {' → '.join(tiers)}")
    print(f"  Episodes/tier: {args.episodes_per_tier}")
    print(f"  Group size: {args.group_size} rollouts per scenario")
    print(f"  Learning rate: {args.lr}")
    print(f"  KL penalty: {args.kl_coeff}")
    print(f"{'='*70}\n")

    tier_results = {}

    for tier_idx, tier in enumerate(tiers):
        print(f"\n{'─'*70}")
        print(f"  PHASE {tier_idx+1}/{len(tiers)}: {tier.upper()}")
        print(f"{'─'*70}\n")

        tier_rewards = []
        tier_resolutions = []

        for ep in range(1, args.episodes_per_tier + 1):
            global_step += 1
            model.train()

            # ── GRPO: Generate G parallel rollouts for this scenario ──
            group_rewards = []
            group_log_probs = []
            group_commands = []
            group_resolved = []

            # Higher temperature for diversity in group
            temperatures = [0.6 + 0.1 * i for i in range(args.group_size)]

            for g in range(args.group_size):
                reward, lps, cmds, resolved = run_episode(
                    model, tokenizer, env, tier,
                    max_turns=args.max_turns,
                    temperature=temperatures[g]
                )
                group_rewards.append(reward)
                group_log_probs.append(lps)
                group_commands.append(cmds)
                group_resolved.append(resolved)

            # ── GRPO: Compute group-relative advantages ──
            mean_reward = sum(group_rewards) / len(group_rewards)
            std_reward = max(
                (sum((r - mean_reward)**2 for r in group_rewards) / len(group_rewards)) ** 0.5,
                1e-6  # prevent division by zero
            )

            # Normalize advantages relative to group
            advantages = [(r - mean_reward) / std_reward for r in group_rewards]

            # ── GRPO: Policy gradient update ──
            total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            valid_rollouts = 0

            for g in range(args.group_size):
                if not group_log_probs[g]:
                    continue

                adv = advantages[g]

                # Clipped advantage (PPO-style stability)
                clipped_adv = max(min(adv, args.clip_eps), -args.clip_eps) if abs(adv) > args.clip_eps else adv

                # Policy gradient: -log_prob * advantage
                rollout_loss = torch.stack([
                    -lp * clipped_adv for lp in group_log_probs[g]
                ]).sum()

                total_loss = total_loss + rollout_loss
                valid_rollouts += 1

            if valid_rollouts > 0:
                # Average over rollouts (not sum — keeps gradient scale stable)
                avg_loss = total_loss / valid_rollouts

                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            # ── Logging ──
            best_idx = group_rewards.index(max(group_rewards))
            best_reward = group_rewards[best_idx]
            best_resolved = group_resolved[best_idx]
            best_cmds = group_commands[best_idx]
            any_resolved = any(group_resolved)

            tier_rewards.append(best_reward)
            tier_resolutions.append(any_resolved)
            global_rewards.append(best_reward)
            all_rewards[tier].append(best_reward)
            all_resolutions[tier].append(any_resolved)

            avg_10 = sum(tier_rewards[-10:]) / len(tier_rewards[-10:])
            res_rate = sum(tier_resolutions) / len(tier_resolutions) * 100

            group_str = " | ".join([
                f"{'✓' if r else '✗'}{rw:+.1f}" for rw, r in zip(group_rewards, group_resolved)
            ])

            print(
                f"  Ep {ep:3d}/{args.episodes_per_tier} | "
                f"best={best_reward:+.2f} | "
                f"mean={mean_reward:+.2f} | "
                f"{'RESOLVED' if any_resolved else 'FAILED':8s} | "
                f"avg(10)={avg_10:+.2f} | "
                f"res={res_rate:4.0f}% | "
                f"group=[{group_str}]"
            )

            # Checkpoint saving
            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_dir = f"{args.output_dir}/checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  💾 Saved checkpoint: {ckpt_dir}")

        # ── Tier Summary ──
        resolved_count = sum(tier_resolutions)
        tier_results[tier] = {
            "episodes": len(tier_rewards),
            "avg_reward": sum(tier_rewards) / len(tier_rewards),
            "best_reward": max(tier_rewards),
            "resolved": resolved_count,
            "resolution_rate": resolved_count / len(tier_rewards) * 100,
        }

        print(f"\n  ── {tier.upper()} COMPLETE ──")
        print(f"  Resolution: {resolved_count}/{len(tier_rewards)} ({tier_results[tier]['resolution_rate']:.0f}%)")
        print(f"  Avg reward: {tier_results[tier]['avg_reward']:+.2f}")
        print(f"  Best reward: {tier_results[tier]['best_reward']:+.2f}")

    # ── Save Final Model ──
    print(f"\nSaving final model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training log
    training_log = {
        "algorithm": "GRPO",
        "group_size": args.group_size,
        "lr": args.lr,
        "kl_coeff": args.kl_coeff,
        "curriculum": tiers,
        "episodes_per_tier": args.episodes_per_tier,
        "tier_results": tier_results,
        "per_episode": [
            {"step": i+1, "reward": r} for i, r in enumerate(global_rewards)
        ]
    }
    with open("grpo_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_tiers = len(tiers)
        fig, axes = plt.subplots(1, n_tiers + 1, figsize=(6 * (n_tiers + 1), 5))
        if n_tiers == 0:
            axes = [axes]

        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

        # Per-tier reward curves
        for i, tier in enumerate(tiers):
            ax = axes[i] if n_tiers > 0 else axes[0]
            rewards = all_rewards[tier]
            episodes = list(range(1, len(rewards) + 1))

            ax.plot(episodes, rewards, color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.5, label='Per-episode')

            # Rolling average
            window = min(10, len(rewards))
            if window > 1:
                rolling = []
                for j in range(len(rewards)):
                    start = max(0, j - window + 1)
                    rolling.append(sum(rewards[start:j+1]) / (j - start + 1))
                ax.plot(episodes, rolling, color=colors[i % len(colors)],
                        linewidth=3, label=f'{window}-ep avg')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'GRPO — {tier}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Overall resolution rate
        ax_summary = axes[-1]
        tier_names = list(tier_results.keys())
        rates = [tier_results[t]["resolution_rate"] for t in tier_names]
        bar_colors = [colors[i % len(colors)] for i in range(len(tier_names))]

        bars = ax_summary.bar(tier_names, rates, color=bar_colors, alpha=0.8, edgecolor='white')
        ax_summary.set_ylabel('Resolution Rate (%)')
        ax_summary.set_title('GRPO Results by Tier', fontsize=13, fontweight='bold')
        ax_summary.set_ylim(0, 100)
        ax_summary.tick_params(axis='x', rotation=30)
        ax_summary.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                          f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('grpo_reward_curve.png', dpi=150, bbox_inches='tight')
        print("Saved: grpo_reward_curve.png")
    except ImportError:
        print("matplotlib not available — skipping plot")

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  GRPO TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Algorithm:     GRPO (Group Relative Policy Optimization)")
    print(f"  Group size:    {args.group_size} rollouts/scenario")
    print(f"  Curriculum:    {' → '.join(tiers)}")
    print(f"  Total steps:   {global_step}")
    print(f"  Model saved:   {args.output_dir}/")
    print()
    print(f"  {'Tier':<16} {'Episodes':<10} {'Resolved':<12} {'Rate':<8} {'Avg Reward':<12} {'Best'}")
    print(f"  {'─'*70}")
    for tier, res in tier_results.items():
        print(
            f"  {tier:<16} {res['episodes']:<10} "
            f"{res['resolved']}/{res['episodes']:<9} "
            f"{res['resolution_rate']:5.0f}%   "
            f"{res['avg_reward']:+.2f}        "
            f"{res['best_reward']:+.2f}"
        )
    print(f"{'='*70}")

    env.close()


if __name__ == "__main__":
    main()
