"""
CloudSRE v2 — Inference Script

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
from typing import List, Optional

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


# ── Logging (hackathon format) ───────────────────────────────────────────

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
    """Run a single task episode with multi-step interaction."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    success = False
    score = 0.0
    step = 0

    try:
        result = env_client.reset(task_id=task_id)
        obs = result.observation
        history = []

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Build observation text
            obs_text = format_obs(obs)

            # Get next command from LLM
            command = get_model_response(llm_client, obs_text, history)
            history.append(command)

            # Execute command
            result = env_client.step(CloudSREAction(command=command))
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)

            log_step(step=step, action=command, reward=reward,
                    done=result.done, error=None)

        score = round(min(max(sum(rewards), 0.01), 0.99), 2) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

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
