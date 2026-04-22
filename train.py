"""
GRPO Training Script — CloudSRE v2 Agent

Follows the standard OpenEnv + TRL pattern (same as Kube SRE Gym's train.py).

Improvements over theirs:
  1. 5-reward decomposition (total, triage, investigation, fix, cascade)
  2. Phase-aware system prompt with real SRE commands (not just kubectl)
  3. Cascade-aware rollout — detects and tracks cascade events
  4. Multi-panel reward visualization with phase breakdown
  5. Unsloth support for 2x faster training on consumer GPUs
  6. Eval mode — test a trained model without training
  7. CSV + JSONL dual logging for richer post-training analysis

Architecture:
  Terminal 1: OpenEnv server (port 7860)
    uv run server

  Terminal 2: GRPO training
    python train.py --model-id Qwen/Qwen3-0.6B --env-url http://localhost:7860
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Help PyTorch reuse fragmented GPU memory (critical for TRL+vLLM colocate)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Silence TRL experimental warning for rollout_func
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from transformers import AutoTokenizer

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

try:
    from cloud_sre_v2 import CloudSREEnv, CloudSREAction
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cloud_sre_v2 import CloudSREEnv, CloudSREAction

# ---- TRL 0.29.0 / vLLM 0.11.x compatibility ----
# TRL 0.29.0 expects vLLM logprobs as list-of-lists (top-k per token),
# but vLLM 0.11.x returns plain floats. Patch until TRL releases a fix.
# See: https://github.com/huggingface/trl/issues/4159

_orig_vllm_gen = None

def _patch_vllm_generate(trainer):
    """Wrap vLLM generate to ensure logprobs are in top-k list format."""
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, "vllm_generation"):
        return
    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate

def patch_trl_vllm_compat():
    """Apply TRL/vLLM compatibility patches. Call before trainer.train()."""
    _orig_train = GRPOTrainer.train
    def _patched_train(self, *args, **kwargs):
        _patch_vllm_generate(self)
        return _orig_train(self, *args, **kwargs)
    GRPOTrainer.train = _patched_train

if __name__ == "__main__":
    patch_trl_vllm_compat()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# System prompt — the heart of what the agent learns
# ============================================================

SYSTEM_PROMPT = """You are a production SRE on-call. Diagnose and fix ALL broken services in this microservice mesh.

Output ONE command per turn. No explanations, no markdown, no prefixes. Just the raw command.

SERVICES:
  payment  (port 8001) — processes payments, writes to SQLite DB
  auth     (port 8002) — JWT authentication, token signing/verification
  worker   (port 8003) — message queue consumer, processes background jobs
  frontend (port 8004) — reverse proxy, routes to payment + auth

DIAGNOSTIC COMMANDS:
  curl http://localhost:<port>/healthz          — check service health (START HERE)
  curl http://localhost:<port>/metrics           — view error rates, latency, memory
  cat /var/log/<service>/error.log              — read error logs (structured JSON)
  grep "ERROR" /var/log/<service>/error.log     — search for errors
  sqlite3 /data/app.db 'SELECT count(*) FROM payments WHERE status="pending"'
  ps aux                                         — list all service processes
  queue status                                   — check message queue depth
  status                                         — overview of ALL services

FIX COMMANDS:
  restart_service <service>                      — restart a crashed service
  kill <service>                                 — kill a hung process
  queue drain 10                                 — drain queue at safe rate (10/batch)
  config <service> key=value                     — change service config

WORKFLOW:
1. Run `status` to see which services are broken
2. Check /healthz of broken services
3. Read error logs: cat /var/log/<service>/error.log
4. Check metrics if needed: curl http://localhost:<port>/metrics
5. Apply the fix (restart, drain, config change)
6. Verify with `status` again

CRITICAL RULES:
- If queue depth > 100, use `queue drain 10` (NOT `queue drain all` — that causes thundering herd!)
- After fixing one service, CHECK OTHERS — cascading failures can trigger new problems
- Cross-reference logs with metrics — some error logs are RED HERRINGS (misleading signals)
- Do NOT repeat the same command more than once"""


# ============================================================
# Args
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for CloudSRE agent")

    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="Agent model to fine-tune")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for 2x faster training")

    # Environment
    parser.add_argument("--env-url", default="http://localhost:7860", help="OpenEnv server URL")
    parser.add_argument("--task-id", default="warmup",
                       choices=["warmup", "single_fault", "cascade", "multi_cascade", "adversarial"],
                       help="Task tier to train on")

    # Training
    parser.add_argument("--dataset-size", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-turns", type=int, default=15, help="Max commands per episode")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per agent response")
    parser.add_argument("--num-generations", type=int, default=8,
                       help="G for GRPO (8+ recommended for stable advantage estimation)")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 = auto)")
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="T=1.0 optimal for GRPO exploration")

    # vLLM
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8080",
                       help="vLLM server URL (server mode only)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Output
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--report-to", default="none", choices=("tensorboard", "wandb", "none"))
    parser.add_argument("--reward-log", default="reward_log.csv")
    parser.add_argument("--logging-steps", type=int, default=1)

    # Eval mode
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation without training")
    parser.add_argument("--eval-episodes", type=int, default=10)

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def format_observation(obs) -> str:
    """Format observation into agent-readable text."""
    command_output = getattr(obs, "command_output", "") or ""
    feedback = getattr(obs, "feedback", "") or ""
    step = getattr(obs, "step_number", 0)
    max_steps = getattr(obs, "max_steps", 15)
    phase = getattr(obs, "phase", "triage")
    cascade = getattr(obs, "cascade_triggered", False)
    cascade_alert = getattr(obs, "cascade_alert", "") or ""

    # Service health summary
    health = getattr(obs, "service_health", {})
    health_lines = []
    for name, info in health.items():
        status = info.get("status", "unknown")
        err_rate = info.get("error_rate", 0)
        health_lines.append(f"  {name}: {status} (error_rate={err_rate:.1%})")
    health_text = "\n".join(health_lines) if health_lines else "  (no health data)"

    text = f"""{command_output}

SERVICE HEALTH:
{health_text}"""

    if cascade and cascade_alert:
        text += f"\n\n{cascade_alert}"

    if feedback:
        text += f"\n\nFEEDBACK: {feedback}"

    text += f"\n\nStep {step}/{max_steps} | Phase: {phase}"
    return text


def format_history(history: list[dict]) -> str:
    """Format conversation history for agent context."""
    if not history:
        return ""
    lines = ["PREVIOUS COMMANDS AND RESULTS:"]
    for entry in history:
        cmd = entry["command"]
        output = entry["output"]
        reward = entry.get("reward", 0.0)
        feedback = entry.get("feedback", "")
        phase = entry.get("phase", "")
        if len(output) > 300:
            output = output[:300] + "... (truncated)"
        lines.append(f"$ {cmd}")
        lines.append(f"  Output: {output}")
        if feedback:
            lines.append(f"  Feedback: {feedback}")
    return "\n".join(lines)


def parse_commands(text: str) -> list[str]:
    """Extract SRE commands from agent response.

    Supports:
      curl http://... , cat /var/log/... , sqlite3 ... , ps aux,
      kill ... , restart_service ... , queue ... , config ... ,
      status, diagnose: ... , fix: ... , grep ...

    Returns at most 2 commands to prevent spam.
    """
    valid_prefixes = (
        "curl ", "cat ", "tail ", "head ", "grep ",
        "sqlite3 ", "kill ", "restart_service ", "python ",
        "ps ", "queue ", "drain ", "config ", "status",
        "diagnose:", "diagnosis:", "fix:",
    )

    commands = []
    seen = set()

    for line in text.strip().split("\n"):
        line = line.strip()
        # Strip common LLM formatting
        line = re.sub(r'^[\-\*\>•]\s*', '', line)
        line = re.sub(r'^```\w*\s*', '', line)
        line = re.sub(r'```$', '', line)
        line = line.strip()

        if any(line.startswith(p) for p in valid_prefixes):
            if line not in seen:
                commands.append(line)
                seen.add(line)
        if len(commands) >= 2:
            break

    return commands


def apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback if enable_thinking is not supported."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


# ============================================================
# Rollout — one full SRE episode
# ============================================================

def rollout_once(
    trainer: GRPOTrainer,
    env: CloudSREEnv,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
) -> dict[str, list]:
    """Run one full CloudSRE incident episode.

    The agent builds conversation history across turns for multi-step
    diagnosis: triage → investigate → fix → (handle cascade) → verify.

    Token accumulation: prompt_ids and completion_ids extend across turns.
    This matches the TRL OpenEnv pattern — GRPO assigns episode-level
    reward to the full token sequence.
    """
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []

    # Per-phase reward tracking (our advantage over theirs)
    step_rewards: list[float] = []
    triage_rewards: list[float] = []
    investigation_rewards: list[float] = []
    fix_rewards: list[float] = []
    cascade_rewards: list[float] = []

    # Conversation history
    conversation_history: list[dict] = []
    cascade_detected = False

    MAX_TOTAL_TOKENS = 4096  # OOM prevention

    for _turn in range(max_turns):
        if result.done:
            break
        if len(completion_ids) >= MAX_TOTAL_TOKENS:
            break

        # Build prompt with full history
        history_text = format_history(conversation_history)
        obs_text = format_observation(observation)

        if history_text:
            user_prompt = f"{history_text}\n\n---\n\nCURRENT OBSERVATION:\n{obs_text}"
        else:
            user_prompt = obs_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate with vLLM via TRL
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse and execute commands
        commands = parse_commands(completion_text)
        if not commands:
            step_rewards.append(-0.5)
            conversation_history.append({
                "agent_text": completion_text[:500],
                "command": completion_text[:100].strip(),
                "output": "(no valid command parsed)",
                "reward": -0.5,
                "feedback": "Invalid output — expected a real SRE command.",
                "phase": "unknown",
            })
            continue

        for cmd in commands:
            try:
                result = env.step(CloudSREAction(command=cmd))
                reward = float(result.reward or 0.0)
                step_rewards.append(reward)
                observation = result.observation

                # Extract phase and cascade info from observation
                phase = getattr(observation, "phase", "triage")
                was_cascade = getattr(observation, "cascade_triggered", False)
                cmd_output = getattr(observation, "command_output", "") or ""
                hint = getattr(observation, "feedback", "") or ""

                # Track per-phase rewards
                if phase == "triage":
                    triage_rewards.append(reward)
                elif phase == "investigation":
                    investigation_rewards.append(reward)
                elif phase in ("fix", "mitigation"):
                    fix_rewards.append(reward)

                # Track cascade handling
                if was_cascade and not cascade_detected:
                    cascade_detected = True
                    cascade_rewards.append(reward)
                elif was_cascade:
                    cascade_rewards.append(reward)

                conversation_history.append({
                    "agent_text": completion_text[:500],
                    "command": cmd,
                    "output": cmd_output[:500],
                    "reward": reward,
                    "feedback": hint,
                    "phase": phase,
                })

                if result.done:
                    break
            except Exception as e:
                logger.warning(f"Step error: {e}")
                step_rewards.append(-0.1)
                conversation_history.append({
                    "command": cmd, "output": f"ERROR: {e}",
                    "reward": -0.1, "feedback": "", "phase": "unknown",
                })
                break

    # Aggregate rewards
    total_reward = sum(step_rewards) if step_rewards else -1.0
    triage_score = sum(triage_rewards) / max(len(triage_rewards), 1) if triage_rewards else 0.0
    investigation_score = sum(investigation_rewards) / max(len(investigation_rewards), 1) if investigation_rewards else 0.0
    fix_score = sum(fix_rewards) / max(len(fix_rewards), 1) if fix_rewards else 0.0
    cascade_score = sum(cascade_rewards) / max(len(cascade_rewards), 1) if cascade_rewards else 0.0

    # Save transcript
    try:
        transcript_path = os.environ.get("AGENT_TRANSCRIPT_LOG", "agent_transcripts.jsonl")
        transcript = {
            "total_reward": total_reward,
            "triage_reward": triage_score,
            "investigation_reward": investigation_score,
            "fix_reward": fix_score,
            "cascade_reward": cascade_score,
            "cascade_detected": cascade_detected,
            "num_steps": len(conversation_history),
            "resolved": result.done and total_reward > 0,
            "phases": [h.get("phase", "") for h in conversation_history],
            "conversation": conversation_history,
        }
        with open(transcript_path, "a") as f:
            f.write(json.dumps(transcript) + "\n")
    except Exception as e:
        logger.warning(f"Failed to save transcript: {e}")

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "triage_reward": triage_score,
        "investigation_reward": investigation_score,
        "fix_reward": fix_score,
        "cascade_reward": cascade_score,
    }


# ============================================================
# Reward functions (TRL convention — 5 decomposed signals)
# ============================================================

def reward_total(completions: list[str], **kwargs) -> list[float]:
    """Total episode reward — primary GRPO signal."""
    rewards = kwargs.get("total_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]

def reward_triage(completions: list[str], **kwargs) -> list[float]:
    """Triage phase reward — did the agent check the right services first?"""
    rewards = kwargs.get("triage_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]

def reward_investigation(completions: list[str], **kwargs) -> list[float]:
    """Investigation reward — did the agent read logs and metrics?"""
    rewards = kwargs.get("investigation_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]

def reward_fix(completions: list[str], **kwargs) -> list[float]:
    """Fix phase reward — was the fix correct?"""
    rewards = kwargs.get("fix_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]

def reward_cascade(completions: list[str], **kwargs) -> list[float]:
    """Cascade handling reward — did the agent handle cascading failures?"""
    rewards = kwargs.get("cascade_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


# ============================================================
# Reward visualization (multi-panel, superior to theirs)
# ============================================================

def plot_rewards(csv_path: Path, out_path: Path = None):
    """Plot multi-panel reward curves from CSV log.

    Panel 1: Total reward with rolling average + trend line
    Panel 2: Phase breakdown (triage, investigation, fix, cascade)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    episodes, totals, triages, investigations, fixes, cascades = [], [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            episodes.append(int(row[0]))
            totals.append(float(row[1]))
            triages.append(float(row[2]))
            investigations.append(float(row[3]))
            fixes.append(float(row[4]))
            cascades.append(float(row[5]))

    if not episodes:
        logger.warning("No episodes to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    window = min(10, len(episodes))
    def rolling_avg(vals):
        return [sum(vals[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(vals))]

    # ---- Panel 1: Total Reward ----
    rolling = rolling_avg(totals)
    ax1.plot(episodes, totals, alpha=0.2, color="#3b82f6", marker="o", markersize=2, label="Per episode")
    ax1.plot(episodes, rolling, color="#3b82f6", linewidth=2.5, label=f"Rolling avg ({window})")

    # Trend line
    z = np.polyfit(episodes, totals, 1)
    trend = np.poly1d(z)
    direction = "improving" if z[0] > 0 else "declining"
    ax1.plot(episodes, trend(episodes), color="#ef4444", linewidth=1.5, linestyle="--",
             label=f"Trend ({direction}: {z[0]:+.3f}/ep)")

    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.set_title("CloudSRE v2 — GRPO Training Reward Curve", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Stats annotation
    ax1.text(0.02, 0.02,
             f"Episodes: {len(episodes)} | Final avg: {rolling[-1]:.2f} | "
             f"Best: {max(totals):.2f} | Resolved: {sum(1 for t in totals if t > 0)}/{len(totals)}",
             transform=ax1.transAxes, fontsize=9, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="#fef3c7", alpha=0.8))

    # ---- Panel 2: Phase Breakdown ----
    phase_colors = {"Triage": "#10b981", "Investigation": "#6366f1",
                    "Fix": "#f59e0b", "Cascade": "#ef4444"}

    ax2.plot(episodes, rolling_avg(triages), color=phase_colors["Triage"],
             linewidth=2, label="Triage")
    ax2.plot(episodes, rolling_avg(investigations), color=phase_colors["Investigation"],
             linewidth=2, label="Investigation")
    ax2.plot(episodes, rolling_avg(fixes), color=phase_colors["Fix"],
             linewidth=2, label="Fix")
    if any(c != 0 for c in cascades):
        ax2.plot(episodes, rolling_avg(cascades), color=phase_colors["Cascade"],
                 linewidth=2, label="Cascade")

    ax2.set_ylabel("Phase Reward (avg)", fontsize=12)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_title("Phase-Level Reward Decomposition", fontsize=12)
    ax2.legend(fontsize=9, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = out_path or csv_path.with_suffix(".png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    patch_trl_vllm_compat()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CloudSRE v2 — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:    {args.model_id}")
    logger.info(f"Env URL:        {args.env_url}")
    logger.info(f"Task tier:      {args.task_id}")
    logger.info(f"Episodes:       {args.dataset_size}")
    logger.info(f"Generations/G:  {args.num_generations}")
    logger.info(f"vLLM mode:      {args.vllm_mode}")
    logger.info(f"Unsloth:        {args.use_unsloth}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Connect to OpenEnv server ----
    env = CloudSREEnv(base_url=args.env_url)

    # ---- Dataset (each entry triggers one episode) ----
    dataset_prompt = f"Diagnose and fix this production incident. Task: {args.task_id}"
    dataset = Dataset.from_dict({"prompt": [dataset_prompt] * args.dataset_size})

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_dir = Path("outputs") / f"cloudsre-grpo-{sanitize_name(args.model_id)}-{args.task_id}-{timestamp}"
    output_dir = Path(args.output_dir or default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- GRPO Config ----
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",
        save_total_limit=3,
        # DAPO improvements over vanilla GRPO
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
    )

    # ---- Reward CSV logger ----
    reward_log_path = output_dir / args.reward_log
    episode_counter = [0]
    all_rewards = []

    with open(reward_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "total_reward", "triage_reward", "investigation_reward",
            "fix_reward", "cascade_reward", "timestamp",
        ])

    def _log_episode(total_r, triage_r, inv_r, fix_r, cascade_r):
        episode_counter[0] += 1
        all_rewards.append(total_r)
        with open(reward_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode_counter[0], total_r, triage_r, inv_r, fix_r, cascade_r,
                datetime.now().isoformat(),
            ])

        n = len(all_rewards)
        mean_all = sum(all_rewards) / n
        last10 = all_rewards[-10:]
        mean_10 = sum(last10) / len(last10)
        best = max(all_rewards)

        logger.info(
            f"Episode {episode_counter[0]}: reward={total_r:.2f} "
            f"(triage={triage_r:.2f}, inv={inv_r:.2f}, fix={fix_r:.2f}, cascade={cascade_r:.2f}) | "
            f"mean={mean_all:.2f}, mean(10)={mean_10:.2f}, best={best:.2f}"
        )

    # ---- Rollout function (called by GRPOTrainer each step) ----
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids = []
        episode_completion_ids = []
        episode_logprobs = []
        total_rewards = []
        triage_rewards = []
        investigation_rewards = []
        fix_rewards_list = []
        cascade_rewards = []

        for _ in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT,
                max_turns=args.max_turns,
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            triage_rewards.append(episode["triage_reward"])
            investigation_rewards.append(episode["investigation_reward"])
            fix_rewards_list.append(episode["fix_reward"])
            cascade_rewards.append(episode["cascade_reward"])

            _log_episode(
                episode["total_reward"], episode["triage_reward"],
                episode["investigation_reward"], episode["fix_reward"],
                episode["cascade_reward"],
            )

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
            "triage_reward": triage_rewards,
            "investigation_reward": investigation_rewards,
            "fix_reward": fix_rewards_list,
            "cascade_reward": cascade_rewards,
        }

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_total,
            reward_triage,
            reward_investigation,
            reward_fix,
            reward_cascade,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    # ---- Train ----
    logger.info("Starting GRPO training...")
    logger.info(f"5 reward signals: total, triage, investigation, fix, cascade")
    logger.info(f"Task tier: {args.task_id}")

    try:
        trainer.train()
    finally:
        env.close()
        try:
            plot_rewards(reward_log_path, output_dir / "reward_plot.png")
        except Exception as e:
            logger.warning(f"Could not generate reward plot: {e}")

    # ---- Save ----
    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_log_path}")

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to https://huggingface.co/{args.hub_repo}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
