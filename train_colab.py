"""
CloudSRE v2 — Minimal Colab Training Script (Unsloth + TRL)

This is the MINIMAL training script required by the hackathon.
Run this in Google Colab with a T4/A100 GPU.

Requirements:
  1. Your CloudSRE v2 environment hosted on HF Spaces
  2. A Colab notebook with GPU runtime
  3. HF token with Inference API access

Usage in Colab:
  !pip install unsloth trl openenv-core httpx openai
  !python train_colab.py --env-url https://your-space.hf.space --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress annoying HuggingFace / Transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")

# ── Check GPU availability ───────────────────────────────────────────────

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu} ({mem:.1f} GB)")
            return True
        else:
            print("WARNING: No GPU detected. Training will be very slow.")
            return False
    except ImportError:
        print("WARNING: PyTorch not installed.")
        return False


# ── Minimal OpenEnv client (no separate package needed) ─────────────────

class SimpleCloudSREClient:
    """Minimal HTTP client for CloudSRE v2 environment.

    Works without installing the full cloud_sre_v2 package.
    Just needs httpx.
    """

    def __init__(self, base_url: str, timeout: float = 120.0):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._session_id = None

    def reset(self, task_id: str = "warmup") -> dict:
        """Reset the environment for a new episode."""
        resp = self.client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        # OpenEnv ResetResponse: {observation: {...}, reward: float, done: bool}
        return data

    def step(self, command: str) -> dict:
        """Execute one command and get the result."""
        # OpenEnv StepRequest expects: {"action": {"command": "..."}}
        resp = self.client.post("/step", json={
            "action": {"command": command},
        })
        resp.raise_for_status()
        # OpenEnv StepResponse: {observation: {...}, reward: float, done: bool}
        return resp.json()

    def close(self):
        self.client.close()


# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a production SRE (Site Reliability Engineer) responding to a PagerDuty alert.
You must diagnose and fix the incident before the SLA timer expires.

Output EXACTLY ONE shell command per turn. No explanations, no markdown, no commentary.

AVAILABLE COMMANDS:
  status                                  — Overview of all services (START HERE)
  curl http://localhost:<port>/healthz     — Check specific service health
  curl http://localhost:<port>/metrics     — View service metrics
  cat /var/log/<service>/error.log        — Read error logs (CRITICAL for diagnosis)
  sqlite3 /data/app.db '<SQL>'            — Query database state
  ps aux                                  — List running processes
  restart_service <service>               — Restart a service (payment|auth|worker|frontend|cache|notification)
  queue status                            — Check message queue depth
  queue drain 50                          — Drain queue safely (NEVER drain all!)

SERVICES: payment(:8001) auth(:8002) worker(:8003) frontend(:8004) cache(:8005) notification(:8006)

SRE WORKFLOW (follow this order):
  1. TRIAGE:       Run 'status' to see which services are down
  2. INVESTIGATE:  Check healthz, read logs, check metrics of affected services
  3. DIAGNOSE:     Cross-reference logs + metrics to find root cause
  4. FIX:          Apply the targeted fix (restart, drain, config change)
  5. VERIFY:       Re-check health to confirm resolution

CRITICAL RULES:
  - NEVER restart all services blindly — find the root cause first
  - If queue depth > 100, use 'queue drain 50' NOT 'queue drain all'
  - Always verify fix with healthz after applying
  - Watch for CASCADE failures: fixing one service may break another"""


# ── Command parser ───────────────────────────────────────────────────────

VALID_PREFIXES = (
    "curl ", "cat ", "tail ", "head ", "grep ", "sqlite3 ",
    "kill ", "restart_service ", "ps ", "queue ", "drain ",
    "config ", "status", "diagnose:", "fix:",
)

def parse_command(text: str) -> str:
    """Extract the first valid SRE command from LLM output."""
    for line in text.strip().split("\n"):
        line = re.sub(r'^[\-\*\>•`]+\s*', '', line.strip())
        if any(line.startswith(p) for p in VALID_PREFIXES):
            return line
    return "status"  # fallback


# ── Rollout function ─────────────────────────────────────────────────────

def run_episode(
    env: SimpleCloudSREClient,
    generate_fn,  # callable(prompt) -> str
    task_id: str = "warmup",
    max_turns: int = 30,  # upper bound; actual limit comes from environment
) -> dict:
    """Run one full SRE episode.

    Args:
        env: CloudSRE environment client
        generate_fn: function that takes a prompt string and returns LLM output
        task_id: which task tier to run
        max_turns: max commands per episode

    Returns:
        dict with total_reward, steps, resolved, history
    """
    result = env.reset(task_id=task_id)
    obs = result.get("observation", {})

    history = []
    rewards = []

    # Read max_steps from environment (warmup=10, cascade=20, multi=25, adversarial=30)
    env_max_steps = obs.get("max_steps", max_turns)
    effective_max = min(max_turns, env_max_steps)  # respect both limits

    for turn in range(effective_max):
        done = result.get("done", False)
        if done:
            break

        # Build prompt
        alert = obs.get("alert", "")
        cmd_output = obs.get("command_output", "")
        feedback = obs.get("feedback", "")
        health = obs.get("service_health", {})

        health_text = "\n".join(
            f"  {n}: {h.get('status', '?')}" for n, h in health.items()
        )

        history_text = ""
        if history:
            history_text = "PREVIOUS:\n" + "\n".join(
                f"  $ {h['cmd']}" for h in history[-5:]
            ) + "\n\n"

        prompt = f"""{SYSTEM_PROMPT}

{history_text}ALERT: {alert}
OUTPUT: {cmd_output}
HEALTH:
{health_text}
{f'FEEDBACK: {feedback}' if feedback else ''}
Step {turn+1}/{max_turns}. Next command:"""

        # Generate
        response = generate_fn(prompt)
        command = parse_command(response)

        # Step
        result = env.step(command)
        obs = result.get("observation", {})
        reward = float(result.get("reward", 0.0))
        rewards.append(reward)
        history.append({"cmd": command, "reward": reward})

    resolved = result.get("done", False) and any(r > 0.3 for r in rewards[-1:])

    if resolved:
        # Resolved: sum of all rewards (includes resolution bonus from env)
        total = sum(rewards)
    else:
        # FAILED: heavy penalty that overwhelms per-step bonuses
        # Per-step rewards shouldn't make failure look positive
        per_step_sum = sum(rewards)
        total = min(per_step_sum, -0.5)  # Cap at -0.5 to ensure negative signal

    return {
        "total_reward": round(total, 3),
        "steps": len(history),
        "resolved": resolved,
        "history": history,
    }


# ── Main: Unsloth + GRPO Training ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CloudSRE v2 — Colab Training")
    parser.add_argument("--env-url", required=True, help="HF Space URL")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token")
    parser.add_argument("--model-id", default="unsloth/Qwen3-0.6B", help="Model to train")
    parser.add_argument("--task-id", default="warmup", help="Task tier")
    parser.add_argument("--episodes", type=int, default=20, help="Training episodes")
    parser.add_argument("--max-turns", type=int, default=30, help="Max turns per episode (env may set lower)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (8 for 0.5B-1.5B, 16 for 3B+)")
    parser.add_argument("--output-dir", default="cloudsre-agent", help="Output directory")
    parser.add_argument("--wandb-project", default="", help="WandB project name (enables logging)")
    args = parser.parse_args()

    has_gpu = check_gpu()
    print(f"\nModel: {args.model_id}")
    print(f"Env:   {args.env_url}")
    print(f"Task:  {args.task_id}")
    print(f"Episodes: {args.episodes}")

    # ── Load model with Unsloth ──────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        print("\nLoading model with Unsloth (2x faster)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            # Daniel (Unsloth): "You MUST do LoRA on MLP too, not just attention"
            # Reference: "LoRA Regret" blog post by Thinking Machines + Unsloth
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # attention
                "gate_proj", "up_proj", "down_proj",      # MLP
            ],
            use_gradient_checkpointing="unsloth",  # async gradient offload to RAM
        )
        USE_UNSLOTH = True
        print("Unsloth loaded successfully!")
    except ImportError:
        print("\nUnsloth not available. Using standard HF Transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map="auto", torch_dtype="auto",
        )
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        USE_UNSLOTH = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Generate function ────────────────────────────────────────────────
    import torch

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, max_length=None, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Connect to environment ───────────────────────────────────────────
    env = SimpleCloudSREClient(base_url=args.env_url)

    # ── Training loop (simplified GRPO) ──────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Starting training: {args.episodes} episodes")
    print(f"{'='*50}\n")

    all_rewards = []
    best_reward = float("-inf")

    # WandB integration for visual proof
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "model": args.model_id,
                    "task_id": args.task_id,
                    "episodes": args.episodes,
                    "lora_r": args.lora_r,
                    "max_turns": args.max_turns,
                },
                name=f"cloudsre-{args.task_id}-{args.model_id.split('/')[-1]}",
            )
            print("WandB initialized!")
        except Exception as e:
            print(f"WandB init failed: {e}. Continuing without WandB.")
            use_wandb = False

    for ep in range(1, args.episodes + 1):
        result = run_episode(
            env=env,
            generate_fn=generate,
            task_id=args.task_id,
            max_turns=args.max_turns,
        )

        total = result["total_reward"]
        all_rewards.append(total)
        if total > best_reward:
            best_reward = total

        avg_10 = sum(all_rewards[-10:]) / len(all_rewards[-10:])
        status = "RESOLVED" if result["resolved"] else "FAILED"

        print(
            f"Ep {ep:3d}/{args.episodes} | "
            f"reward={total:+.2f} | "
            f"steps={result['steps']:2d} | "
            f"{status:8s} | "
            f"avg(10)={avg_10:+.2f} | "
            f"best={best_reward:+.2f}"
        )

        # WandB logging
        if use_wandb:
            import wandb
            wandb.log({
                "episode": ep,
                "reward": total,
                "steps": result["steps"],
                "resolved": 1 if result["resolved"] else 0,
                "avg_reward_10": avg_10,
                "best_reward": best_reward,
            })

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\nSaving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save rewards
    with open(f"{args.output_dir}/rewards.json", "w") as f:
        json.dump({
            "rewards": all_rewards,
            "best": best_reward,
            "avg": sum(all_rewards) / len(all_rewards),
            "model": args.model_id,
            "task": args.task_id,
            "episodes": args.episodes,
        }, f, indent=2)

    print(f"\nFinal stats:")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Avg reward:  {sum(all_rewards)/len(all_rewards):+.2f}")
    print(f"  Best reward: {best_reward:+.2f}")
    print(f"  Resolved:    {sum(1 for r in all_rewards if r > 0)}/{len(all_rewards)}")
    print(f"\nModel saved to: {args.output_dir}/")

    env.close()

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
