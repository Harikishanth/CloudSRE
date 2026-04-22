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
from datetime import datetime
from pathlib import Path

# ── Check GPU availability ───────────────────────────────────────────────

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
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
        self._session_id = data.get("session_id")
        return data

    def step(self, command: str) -> dict:
        """Execute one command and get the result."""
        resp = self.client.post("/step", json={
            "command": command,
            "session_id": self._session_id,
        })
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a production SRE on-call. Diagnose and fix broken services.

Output ONE command per turn. No explanations, no markdown.

COMMANDS:
  status                                  — overview of all services
  curl http://localhost:<port>/healthz     — check service health
  cat /var/log/<service>/error.log        — read error logs
  restart_service <service>               — restart a service
  queue drain 10                          — drain queue safely (NOT drain all!)

WORKFLOW: status → healthz → logs → fix → verify"""


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
    max_turns: int = 10,
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

    for turn in range(max_turns):
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

    total = sum(rewards) if rewards else -1.0
    return {
        "total_reward": total,
        "steps": len(history),
        "resolved": result.get("done", False) and total > 0,
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
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per episode")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output-dir", default="cloudsre-agent", help="Output directory")
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
                **inputs, max_new_tokens=128, temperature=0.7,
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


if __name__ == "__main__":
    main()
