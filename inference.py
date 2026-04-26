"""
CloudSRE v2 — Rich Inference Script

Produces detailed, publication-quality episode transcripts showing:
  - Per-step phase labels (TRIAGE/INVESTIGATE/FIX/VERIFY)
  - Command + truncated output
  - Per-step reward with explanation
  - Service health delta
  - Episode-level score breakdown

MANDATORY environment variables:
    HF_TOKEN       Your Hugging Face API key
    API_BASE_URL   LLM API endpoint (default: HF router)
    MODEL_NAME     Model identifier (default: Qwen2.5-72B-Instruct)
    ENV_URL        Live environment URL (default: HF Space)

STDOUT FORMAT (required by hackathon):
    [START] task=<name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import sys
import time
import textwrap
from typing import List, Optional, Dict

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import CloudSREEnv
from models import CloudSREAction

# ── Environment variables ────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://dardrax-cloud-sre-v2.hf.space")

# ── Constants ────────────────────────────────────────────────────────────
BENCHMARK = "cloud_sre_v2"
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 256
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Cloud SRE on-call. You diagnose and fix production incidents
    in a multi-region microservice architecture (AWS-style).

    Regions and Services:
      us-east-1: payment, auth, billing, gateway, loadbalancer, config
      eu-west-1: worker, scheduler, search, storage, metrics_collector
      ap-south-1: frontend, cache, notification, email, dns

    Output ONE command per turn. No explanations, no markdown. Just the command.

    Available commands:
      curl http://<svc>.<region>.internal/healthz  -- check service health
      curl http://<svc>.<region>.internal/metrics   -- view metrics
      cat /var/log/<service>/error.log              -- read error logs
      sqlite3 /data/app.db 'SQL'                    -- query RDS
      queue status                                  -- check SQS
      queue drain 10                                -- drain SQS safely
      restart_service <service>                     -- restart service
      status                                        -- cloud dashboard

    Workflow: status -> healthz -> logs -> fix -> verify
    IMPORTANT: Use `queue drain 10` NOT `queue drain all` to avoid thundering herd.
""").strip()


# ── Phase Classification ─────────────────────────────────────────────────

def classify_phase(cmd: str) -> str:
    """Classify a command into SRE workflow phase."""
    cmd_lower = cmd.lower()
    if any(k in cmd_lower for k in ["status", "healthz", "health"]):
        return "TRIAGE"
    elif any(k in cmd_lower for k in ["cat ", "log", "sqlite3", "metrics", "queue status"]):
        return "INVESTIGATE"
    elif any(k in cmd_lower for k in ["restart", "drain", "kill", "fix:", "scale"]):
        return "FIX"
    elif any(k in cmd_lower for k in ["verify", "curl"]) and cmd_lower.count("curl") == 1:
        return "VERIFY"
    return "DIAGNOSE"


def reward_explanation(reward: float, cmd: str, phase: str) -> str:
    """Generate human-readable explanation for reward signal."""
    if reward >= 0.2:
        return "correct fix applied"
    elif reward >= 0.1:
        return "useful investigation" if "INVESTIGATE" in phase else "good triage step"
    elif reward >= 0.05:
        return "relevant diagnostic"
    elif reward >= 0:
        return "neutral action"
    elif reward >= -0.1:
        return "inefficient — repeated or unnecessary"
    else:
        return "wrong approach — penalty"


def health_delta(prev: Dict, curr: Dict) -> str:
    """Show what changed in service health between steps."""
    if not prev or not curr:
        return ""
    changes = []
    for svc in curr:
        prev_status = prev.get(svc, {}).get("status", "unknown")
        curr_status = curr.get(svc, {}).get("status", "unknown")
        if prev_status != curr_status:
            emoji = "✅" if curr_status == "healthy" else "🔴"
            changes.append(f"{emoji} {svc}: {prev_status}→{curr_status}")
    if changes:
        return " | ".join(changes)
    return "no change"


# ── Logging (hackathon format + rich transcript) ─────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clean_action = action.replace("\n", " ").replace("\r", " ")[:200]
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Rich Transcript ─────────────────────────────────────────────────────

def print_episode_header(task_id: str, scenario_id: str, alert: str):
    """Print beautiful episode header."""
    print(f"\n╔{'═'*68}╗", flush=True)
    print(f"║  EPISODE  │  Tier: {task_id:<12s}  │  Model: {MODEL_NAME.split('/')[-1]:<20s}  ║", flush=True)
    print(f"╠{'═'*68}╣", flush=True)
    print(f"║  Scenario: {scenario_id[:56]:<56s}  ║", flush=True)
    print(f"╠{'═'*68}╣", flush=True)
    alert_line = alert[:64] if alert else "No alert"
    print(f"║  ALERT: {alert_line:<60s}║", flush=True)
    print(f"╠{'═'*68}╣", flush=True)


def print_step_rich(step: int, phase: str, cmd: str, output: str,
                    reward: float, explanation: str, health_change: str):
    """Print one step with full context."""
    phase_colors = {
        "TRIAGE": "🔍", "INVESTIGATE": "🔬", "FIX": "🔧",
        "VERIFY": "✅", "DIAGNOSE": "📋"
    }
    emoji = phase_colors.get(phase, "▶")

    print(f"\n{'='*72}", flush=True)
    print(f" Step {step:2d} | [{phase:11s}] {emoji} ", flush=True)
    print(f"{'-'*72}", flush=True)
    
    # Full command
    print(f"  $ {cmd.strip()}", flush=True)
    print(f"", flush=True)
    
    # Full output (indented to look like terminal output)
    if output:
        out_lines = output.strip().split('\n')
        # Cap to 15 lines max so it doesn't flood entirely, but much better than 54 chars
        if len(out_lines) > 15:
            out_lines = out_lines[:15] + ["  ... (output truncated) ..."]
        for line in out_lines:
            print(f"    {line}", flush=True)
    else:
        print(f"    (no output)", flush=True)
    print(f"", flush=True)

    # Reward + explanation
    sign = "+" if reward >= 0 else ""
    print(f"  Reward: {sign}{reward:.3f} ({explanation})", flush=True)

    # Health delta
    if health_change and health_change != "no change":
        print(f"  Health: {health_change}", flush=True)
    else:
        print(f"  Health: no change", flush=True)
    print(f"{'='*72}\n", flush=True)


def print_episode_footer(resolved: bool, steps: int, total_reward: float,
                         rewards: List[float], max_steps: int):
    """Print episode summary."""
    print(f"║                                                                    ║", flush=True)
    print(f"╠{'═'*68}╣", flush=True)
    status = "✅ RESOLVED" if resolved else "❌ FAILED"
    print(f"║  RESULT: {status} in {steps} steps (max: {max_steps})                    ║", flush=True)
    print(f"║  TOTAL REWARD: {total_reward:+.3f}                                        ║", flush=True)

    # Show reward per step
    if rewards:
        rw_str = " ".join(f"{r:+.2f}" for r in rewards[:12])
        print(f"║  PER-STEP:  [{rw_str}]", flush=True)

    print(f"╚{'═'*68}╝", flush=True)
    print(flush=True)


# ── LLM call ────────────────────────────────────────────────────────────

def get_model_response(
    client: OpenAI,
    observation_text: str,
    history: List[str],
) -> str:
    """Get next SRE command from the LLM."""
    history_text = ""
    if history:
        history_text = "PREVIOUS COMMANDS:\n" + "\n".join(f"  $ {h}" for h in history[-5:]) + "\n\n"

    user_prompt = f"{history_text}CURRENT OBSERVATION:\n{observation_text}\n\nOutput your next command:"

    if not HF_TOKEN:
        return "status"  # Fallback for dry-run

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            # Extract just the command (first line)
            for line in text.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
            return text.split("\n")[0].strip() if text else "status"
        except Exception:
            if attempt < 2:
                time.sleep((attempt + 1) * 2)
    return "status"


# ── Run one task ─────────────────────────────────────────────────────────

def run_task(env_client, llm_client, task_id: str):
    """Run a single task episode with multi-step interaction + rich transcript."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    success = False
    score = 0.0
    step = 0

    try:
        result = env_client.reset(task_id=task_id)
        obs = result.observation
        history = []
        prev_health = {}

        # Rich header
        scenario_id = getattr(obs, "scenario_id", "unknown")
        alert = getattr(obs, "alert", "")
        print_episode_header(task_id, scenario_id, alert)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Build observation text
            obs_text = format_obs(obs)

            # Get next command from LLM
            command = get_model_response(llm_client, obs_text, history)
            history.append(command)

            # Classify phase
            phase = classify_phase(command)

            # Execute command
            result = env_client.step(CloudSREAction(command=command))
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)

            # Rich step output
            cmd_output = getattr(obs, "command_output", "")
            curr_health = getattr(obs, "service_health", {})
            h_delta = health_delta(prev_health, curr_health)
            explanation = reward_explanation(reward, command, phase)

            print_step_rich(step, phase, command, cmd_output,
                          reward, explanation, h_delta)

            # Hackathon format
            log_step(step=step, action=command, reward=reward,
                    done=result.done, error=None)

            prev_health = curr_health

        score = round(min(max(sum(rewards), 0.01), 0.99), 2) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

        # Rich footer
        resolved = getattr(obs, "done", False) if obs else False
        print_episode_footer(success, len(rewards), sum(rewards), rewards, MAX_STEPS)

    except Exception as e:
        log_step(step=step + 1, action="ERROR", reward=0.0, done=True, error=str(e))
        score = 0.0
    finally:
        log_end(task=task_id, success=success, steps=len(rewards), score=score, rewards=rewards)


def format_obs(obs) -> str:
    """Format observation for LLM consumption."""
    parts = []
    alert = getattr(obs, "alert", "")
    if alert:
        parts.append(f"ALERT:\n{alert}")

    cmd_output = getattr(obs, "command_output", "")
    if cmd_output:
        parts.append(f"COMMAND OUTPUT:\n{cmd_output}")

    health = getattr(obs, "service_health", {})
    if health:
        lines = []
        for name, info in health.items():
            status = info.get("status", "unknown")
            lines.append(f"  {name}: {status}")
        parts.append("SERVICE HEALTH:\n" + "\n".join(lines))

    feedback = getattr(obs, "feedback", "")
    if feedback:
        parts.append(f"FEEDBACK: {feedback}")

    cascade = getattr(obs, "cascade_triggered", False)
    cascade_alert = getattr(obs, "cascade_alert", "")
    if cascade and cascade_alert:
        parts.append(f"CASCADE ALERT: {cascade_alert}")

    step = getattr(obs, "step_number", 0)
    max_s = getattr(obs, "max_steps", 15)
    parts.append(f"Step {step}/{max_s}")

    return "\n\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    target_task = os.getenv("TASK_NAME")
    tasks_to_run = [target_task] if target_task else ["warmup", "single_fault", "cascade"]

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    print(f"\n{'='*70}")
    print(f"  CloudSRE v2 — INFERENCE")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Env:    {ENV_URL}")
    print(f"  Tasks:  {', '.join(tasks_to_run)}")
    print(f"{'='*70}\n")

    max_env_retries = 10
    for attempt in range(max_env_retries):
        try:
            with CloudSREEnv(base_url=ENV_URL).sync() as env:
                for t in tasks_to_run:
                    run_task(env, llm_client, t)
            break
        except Exception as e:
            print(
                f"Retry ({attempt+1}/{max_env_retries}) — Waiting for HF Space: {e}",
                flush=True,
            )
            if attempt < max_env_retries - 1:
                time.sleep(10)
            else:
                print("Fatal: Could not connect to environment.", flush=True)
                sys.exit(0)


if __name__ == "__main__":
    main()
"""
