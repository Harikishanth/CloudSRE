"""
CloudSRE v2 — REINFORCE Training (Real Gradient Updates)

Unlike train_colab.py (rollout-only), this script ACTUALLY updates model weights
using the REINFORCE policy gradient algorithm.

Usage:
    !python train_reinforce.py \
        --env-url https://dardrax-cloudsre-environment.hf.space \
        --model-id ./cloudsre-sft-checkpoint \
        --task-id warmup \
        --episodes 50
"""

import argparse
import json
import time
import os
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

def get_broken_services(health: dict) -> list:
    return [svc for svc, info in health.items() if info.get("status") != "healthy"]


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


def generate_with_logprobs(model, tokenizer, prompt: str, max_new_tokens: int = 60):
    """Generate text AND return log probabilities for REINFORCE."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    scores = outputs.scores  # list of logits at each generation step

    # Compute log probabilities of the actually generated tokens
    log_probs = []
    for i, score in enumerate(scores):
        if i >= len(generated_ids):
            break
        token_id = generated_ids[i]
        log_prob = F.log_softmax(score[0], dim=-1)[token_id]
        log_probs.append(log_prob)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text, log_probs


def parse_command(text: str) -> str:
    """Extract the SRE command from model output."""
    text = text.strip()
    # Take the first line that looks like a command
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove common prefixes
        for prefix in ["Command:", "command:", "Action:", "action:", ">", "$", "```"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        if line and not line.startswith("#"):
            return line[:200]
    return "status"


# ── REINFORCE Training Loop ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CloudSRE REINFORCE Training")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--model-id", default="./cloudsre-sft-checkpoint")
    parser.add_argument("--task-id", default="warmup")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="./cloudsre-reinforce")
    args = parser.parse_args()

    # ── Load Model ──
    from unsloth import FastLanguageModel

    print(f"Loading model: {args.model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Enable training mode
    if hasattr(model, 'peft_config'):
        print("Reusing existing LoRA adapters for REINFORCE")
    else:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

    # Set up optimizer on trainable (LoRA) parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ── Environment ──
    env = CloudSREClient(args.env_url)

    # ── Training ──
    all_rewards = []
    all_resolutions = []
    baseline_reward = 0.0  # Running average baseline for variance reduction

    print(f"\n{'='*60}")
    print(f"REINFORCE Training: {args.episodes} episodes")
    print(f"{'='*60}\n")

    for ep in range(1, args.episodes + 1):
        # Reset environment
        try:
            result = env.reset(task_id=args.task_id)
        except Exception as e:
            print(f"  Ep {ep}: Reset failed ({e}), skipping")
            continue

        obs = result.get("observation", result)
        max_steps = min(args.max_turns, obs.get("max_steps", args.max_turns))

        episode_log_probs = []
        episode_rewards = []
        episode_commands = []

        model.train()  # Enable gradient computation

        for turn in range(max_steps):
            done = result.get("done", False)
            if done:
                break

            prompt = build_prompt(obs, turn, max_steps)

            # Generate WITH log probabilities (this is the key difference!)
            raw_output, log_probs = generate_with_logprobs(model, tokenizer, prompt)
            command = parse_command(raw_output)
            episode_commands.append(command)

            # Collect log probs for this turn
            if log_probs:
                episode_log_probs.append(torch.stack(log_probs).sum())

            # Step environment
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

        # ── Compute Episode Return ──
        total_reward = sum(episode_rewards)
        resolved = result.get("done", False) and total_reward > 0

        all_rewards.append(total_reward)
        all_resolutions.append(resolved)

        # ── REINFORCE Policy Gradient Update ──
        if episode_log_probs:
            # Advantage = reward - baseline (variance reduction)
            advantage = total_reward - baseline_reward

            # Policy gradient loss: -log_prob * advantage
            # We want to INCREASE probability of actions that got high reward
            policy_loss = torch.stack([
                -lp * advantage for lp in episode_log_probs
            ]).sum()

            # Backward pass — THIS IS WHAT WAS MISSING
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Update baseline (exponential moving average)
            baseline_reward = 0.9 * baseline_reward + 0.1 * total_reward

        # ── Logging ──
        avg_10 = sum(all_rewards[-10:]) / len(all_rewards[-10:])
        best = max(all_rewards)
        status = "RESOLVED ✓" if resolved else "FAILED"

        print(
            f"Ep {ep:3d}/{args.episodes} | "
            f"r={total_reward:+.2f} | "
            f"steps={len(episode_commands):2d} | "
            f"{status:11s} | "
            f"avg(10)={avg_10:+.2f} | "
            f"best={best:+.2f} | "
            f"cmds: {', '.join(episode_commands[:3])}"
        )

    # ── Save ──
    print(f"\nSaving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save per-episode rewards
    episode_log = [{"episode": i+1, "reward": r, "resolved": res}
                   for i, (r, res) in enumerate(zip(all_rewards, all_resolutions))]
    with open("training_rewards.json", "w") as f:
        json.dump(episode_log, f, indent=2)

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        episodes = list(range(1, len(all_rewards) + 1))

        # Reward curve
        ax1.plot(episodes, all_rewards, color='#3498db', linewidth=1.5, alpha=0.6, label='Per-episode')
        window = min(10, len(all_rewards))
        if window > 1:
            rolling = []
            for i in range(len(all_rewards)):
                start = max(0, i - window + 1)
                rolling.append(sum(all_rewards[start:i+1]) / (i - start + 1))
            ax1.plot(episodes, rolling, color='#e74c3c', linewidth=3, label=f'{window}-ep rolling avg')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.set_title(f'REINFORCE Reward — {args.task_id}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Resolution rate
        cum_rate = [sum(all_resolutions[:i+1])/(i+1)*100 for i in range(len(all_resolutions))]
        ax2.plot(episodes, cum_rate, color='#2ecc71', linewidth=2)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Cumulative Resolution Rate (%)', fontsize=12)
        ax2.set_title('Resolution Rate Over Training', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('reward_curve.png', dpi=150)
        print("Saved: reward_curve.png")
    except ImportError:
        print("matplotlib not available")

    # ── Summary ──
    resolved_count = sum(all_resolutions)
    print(f"\n{'='*60}")
    print(f"REINFORCE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Episodes:     {len(all_rewards)}")
    print(f"  Avg reward:   {sum(all_rewards)/len(all_rewards):+.2f}")
    print(f"  Best reward:  {max(all_rewards):+.2f}")
    print(f"  Resolved:     {resolved_count}/{len(all_rewards)} ({resolved_count/len(all_rewards)*100:.0f}%)")
    print(f"  Model saved:  {args.output_dir}/")

    env.close()


if __name__ == "__main__":
    main()
