"""
CloudSRE v2 — Model Evaluation ("Final Exam")

Loads the trained GRPO model and evaluates it on fresh (unseen) scenarios
across all tiers. Produces:
  1. Per-tier resolution rate + average reward + average steps
  2. Before/after comparison (base model vs trained model)
  3. evaluation_results.json for submission evidence
  4. evaluation_table.png visualization

Usage:
    python evaluate_model.py \
        --env-url https://dardrax-cloudsre-environment.hf.space \
        --model-id ./cloudsre-grpo \
        --episodes 20

    # Compare base vs trained:
    python evaluate_model.py \
        --env-url https://dardrax-cloudsre-environment.hf.space \
        --model-id ./cloudsre-grpo \
        --base-model-id unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit \
        --episodes 10
"""

import argparse
import json
import time
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import httpx
import torch


class CloudSREClient:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=120)

    def reset(self, task_id: str = "warmup") -> dict:
        return self.client.post("/reset", json={"task_id": task_id}).json()

    def step(self, command: str) -> dict:
        return self.client.post("/step", json={"action": {"command": command}}).json()

    def close(self):
        self.client.close()


def build_prompt(obs, turn, max_turns):
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


def parse_command(text):
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


def evaluate_model(model, tokenizer, env, tiers, episodes_per_tier, max_turns=10):
    """Run the model through fresh scenarios and collect metrics."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    results = {}

    for tier in tiers:
        resolved = 0
        total_reward = 0.0
        total_steps = 0
        episode_details = []

        for ep in range(episodes_per_tier):
            try:
                result = env.reset(task_id=tier)
            except Exception:
                continue

            obs = result.get("observation", result)
            max_steps = min(max_turns, obs.get("max_steps", max_turns))
            ep_reward = 0.0
            steps = 0
            commands = []

            for turn in range(max_steps):
                done = result.get("done", False)
                if done:
                    break

                prompt = build_prompt(obs, turn, max_steps)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=60,
                        do_sample=False,  # Greedy for evaluation (deterministic)
                        temperature=1.0,
                    )

                gen_text = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                command = parse_command(gen_text)
                commands.append(command)

                try:
                    result = env.step(command)
                except Exception:
                    time.sleep(2)
                    try:
                        result = env.step(command)
                    except Exception:
                        break

                obs = result.get("observation", result)
                ep_reward += float(result.get("reward", 0.0))
                steps += 1

            ep_resolved = result.get("done", False) and ep_reward > 0

            if ep_resolved:
                resolved += 1
            total_reward += ep_reward
            total_steps += steps

            episode_details.append({
                "episode": ep + 1,
                "reward": ep_reward,
                "steps": steps,
                "resolved": ep_resolved,
                "commands": commands[:3],  # First 3 commands for logs
            })

            status = "✓" if ep_resolved else "✗"
            print(f"  [{tier}] Ep {ep+1:2d}/{episodes_per_tier} | "
                  f"r={ep_reward:+.2f} | steps={steps} | {status} | "
                  f"cmds: {', '.join(commands[:2])}")

        results[tier] = {
            "episodes": episodes_per_tier,
            "resolved": resolved,
            "resolution_rate": resolved / max(episodes_per_tier, 1) * 100,
            "avg_reward": total_reward / max(episodes_per_tier, 1),
            "avg_steps": total_steps / max(episodes_per_tier, 1),
            "details": episode_details,
        }

    return results


def print_results_table(results, label=""):
    header = f"EVALUATION RESULTS{' — ' + label if label else ''}"
    print(f"\n{'='*75}")
    print(f"  {header}")
    print(f"{'='*75}")
    print(f"  {'Tier':<16} | {'Resolved':<10} | {'Rate':<8} | {'Avg Reward':<12} | {'Avg Steps'}")
    print(f"  {'─'*70}")
    for tier, r in results.items():
        rate_str = f"{r['resolved']}/{r['episodes']}"
        pct = f"{r['resolution_rate']:.0f}%"
        print(f"  {tier:<16} | {rate_str:<10} | {pct:<8} | "
              f"{r['avg_reward']:+.2f}         | {r['avg_steps']:.1f}")
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="CloudSRE Model Evaluation")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--model-id", required=True, help="Trained model to evaluate")
    parser.add_argument("--base-model-id", default="",
                        help="Optional base model for before/after comparison")
    parser.add_argument("--tiers", default="warmup,single_fault,cascade",
                        help="Comma-separated tiers to evaluate")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per tier")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    env = CloudSREClient(args.env_url)

    from unsloth import FastLanguageModel

    # ── Evaluate Trained Model ──
    print(f"\n{'='*75}")
    print(f"  Loading trained model: {args.model_id}")
    print(f"{'='*75}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id, max_seq_length=2048, load_in_4bit=True,
    )

    trained_results = evaluate_model(model, tokenizer, env, tiers, args.episodes, args.max_turns)
    print_results_table(trained_results, "Trained Agent (GRPO)")

    # ── Optional: Evaluate Base Model for Comparison ──
    base_results = None
    if args.base_model_id:
        print(f"\n{'='*75}")
        print(f"  Loading base model: {args.base_model_id}")
        print(f"{'='*75}")

        del model
        torch.cuda.empty_cache()

        base_model, base_tok = FastLanguageModel.from_pretrained(
            model_name=args.base_model_id, max_seq_length=2048, load_in_4bit=True,
        )

        base_results = evaluate_model(base_model, base_tok, env, tiers, args.episodes, args.max_turns)
        print_results_table(base_results, "Base Model (Untrained)")

        # ── Before/After Comparison ──
        print(f"\n{'='*75}")
        print(f"  BEFORE vs AFTER COMPARISON")
        print(f"{'='*75}")
        print(f"  {'Tier':<16} | {'Base Rate':<12} | {'Trained Rate':<14} | {'Improvement'}")
        print(f"  {'─'*70}")
        for tier in tiers:
            base_rate = base_results[tier]["resolution_rate"]
            trained_rate = trained_results[tier]["resolution_rate"]
            delta = trained_rate - base_rate
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {tier:<16} | {base_rate:5.0f}%       | {trained_rate:5.0f}%         | {arrow} {abs(delta):.0f}%")
        print(f"{'='*75}")

    # ── Save Results ──
    output = {
        "model": args.model_id,
        "tiers_evaluated": tiers,
        "episodes_per_tier": args.episodes,
        "trained_results": {t: {k: v for k, v in r.items() if k != "details"}
                            for t, r in trained_results.items()},
    }
    if base_results:
        output["base_model"] = args.base_model_id
        output["base_results"] = {t: {k: v for k, v in r.items() if k != "details"}
                                   for t, r in base_results.items()}

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: evaluation_results.json")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Resolution Rate comparison
        x = range(len(tiers))
        width = 0.35

        trained_rates = [trained_results[t]["resolution_rate"] for t in tiers]
        bars1 = ax1.bar([i + width/2 for i in x], trained_rates, width,
                        label='Trained (GRPO)', color='#2ecc71', edgecolor='white')

        if base_results:
            base_rates = [base_results[t]["resolution_rate"] for t in tiers]
            bars0 = ax1.bar([i - width/2 for i in x], base_rates, width,
                            label='Base (Untrained)', color='#e74c3c', alpha=0.6, edgecolor='white')

        ax1.set_xlabel('Tier')
        ax1.set_ylabel('Resolution Rate (%)')
        ax1.set_title('CloudSRE v2 — Evaluation Results', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tiers, rotation=20)
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars1, trained_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Average Reward comparison
        trained_rewards = [trained_results[t]["avg_reward"] for t in tiers]
        ax2.bar(tiers, trained_rewards, color='#3498db', edgecolor='white', alpha=0.8)
        ax2.set_xlabel('Tier')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward by Tier', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=20)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
        print("Saved: evaluation_results.png")
    except ImportError:
        print("matplotlib not available — skipping plot")

    env.close()


if __name__ == "__main__":
    main()
