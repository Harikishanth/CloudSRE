# Judges: Start Here

**60-second read. Every claim links to source code.**

## What it is

An OpenEnv environment where an LLM agent diagnoses and fixes **real infrastructure failures** — not simulated alerts, not classification quizzes. 16 real OS-level services running as subprocesses, with real cascading failures that propagate through the system.

## The one thing that makes it different

**Open [`services/orchestrator.py`](services/orchestrator.py).** You'll see `subprocess.Popen` spawning 16 real services — each with its own PID, TCP port, and health check endpoint. When we inject a fault, we **actually kill the process** with `os.kill(pid, SIGSTOP)`. When the agent runs `restart_service payment`, a real process dies and a new one starts.

No other submission in this hackathon runs real infrastructure.

## 60-second proof it works

```bash
# 1. The environment is LIVE right now:
curl -s https://dardrax-cloudsre-environment.hf.space/health | python -m json.tool

# 2. Start an incident and watch real services break:
curl -X POST https://dardrax-cloudsre-environment.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "warmup"}'

# 3. Agent fixes it with real commands:
curl -X POST https://dardrax-cloudsre-environment.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"command": "restart_service auth"}}'
```

## What makes cascades REAL

When the database locks → the queue backs up (real TCP connections failing) → the worker crashes (real process exit) → payments time out. **This is emergent behavior from real service interactions, not scripted if/else chains.**

See [`server/constants.py`](server/constants.py) for cascade rules. See [`services/orchestrator.py`](services/orchestrator.py) for process management.

## Training evidence (REAL gradient updates)

We trained using a two-phase **SFT → GRPO** pipeline:

| Phase | Evidence | Result |
|---|---|---|
| **SFT (Warmup)** | [`sft_loss_curve.png`](sft_loss_curve.png) | Loss: 2.09 → 0.15 in 39 steps. Model learned SRE command syntax. |
| **GRPO Leg 1** (Colab) | [`grpo_training_log.json`](grpo_training_log.json) | Warmup: 100% (15/15). Single Fault: 93% (14/15). |
| **GRPO Leg 2** (Kaggle) | [`reward_curve.png`](reward_curve.png) | Cascade: 17% (2/12). Multi-Cascade: 0% (0/12). Adversarial: 33% (4/12). |

### Final Training Summary (Kaggle — Cascade/Multi-Cascade/Adversarial)
```
  Tier             Episodes   Resolved     Rate     Avg Reward   Best
  ──────────────────────────────────────────────────────────────────────
  cascade          12         2/12           17%   +0.50        +1.13
  multi_cascade    12         0/12            0%   +0.46        +0.70
  adversarial      12         4/12           33%   +0.62        +1.13
```

The adversarial tier uses a **Qwen-72B Adversarial Designer** ([`server/adversarial_designer.py`](server/adversarial_designer.py)) with progressive difficulty scaling. The agent scored higher on adversarial (33%) than cascade (17%) because the progressive curriculum starts at difficulty 2 and ramps to 5.

## 5-tier curriculum

| Tier | Faults | Cascading? | Example |
|---|---|---|---|
| `warmup` | 1 service down | No | Auth service crashed |
| `single_fault` | 1 fault + red herrings | No | DB locked, find root cause |
| `cascade` | 1 fault triggers 2nd | Yes | DB lock → queue flood |
| `multi_cascade` | 2+ faults, 3+ effects | Yes | Auth crash + DB lock → payment timeout |
| `adversarial` | LLM-generated, unique every episode | Yes | Unknown fault, progressive difficulty |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              ServiceOrchestrator — 16 Real OS Processes              │
├────┬────┬────┬────┬─────┬──────┬──────┬───────┬─────┬───────┬───────┤
│Pay │Auth│Wkr │FE  │Cache│Notif │Search│Gateway│Sched│Storage│Metric │
│8001│8002│8003│8004│8005 │ 8006 │ 8007 │ 8008  │8009 │ 8010  │ 8011  │
├────┴────┴────┴────┴─────┴──────┴──────┴───────┴─────┴───────┴───────┤
│Email│Billing│Config│DNS  │LoadBalancer│                              │
│8012 │ 8013  │ 8014 │8015 │   8016     │                              │
└─────┴───────┴──────┴─────┴────────────┘
         Real TCP health checks · Real process crashes · Real cascading failures
```

## Dual-LLM Architecture

1. **Adversarial Designer** (`server/adversarial_designer.py`): Qwen-72B generates targeted incidents based on agent's historical weaknesses with progressive difficulty (2→5).
2. **LLM Judge** (`server/llm_judge.py`): Qwen-72B grades the agent's multi-step SRE workflow (triage → investigate → fix → verify) for cascade tiers where deterministic grading is insufficient.

## Where to look

| What | File |
|---|---|
| Real services spawned | [`services/orchestrator.py`](services/orchestrator.py) |
| All 16 service implementations | [`services/`](services/) |
| Fault injection (real `os.kill`) | [`services/orchestrator.py`](services/orchestrator.py) |
| 25 fault types + scenarios | [`server/constants.py`](server/constants.py) |
| Command executor (real shell ops) | [`server/command_executor.py`](server/command_executor.py) |
| Graders (check real health) | [`server/graders.py`](server/graders.py) |
| GRPO training with curriculum | [`training_scripts/train_grpo.py`](training_scripts/train_grpo.py) |
| SFT warmup training | [`training_scripts/sft_warmup.py`](training_scripts/sft_warmup.py) |
| Adversarial Designer (72B) | [`server/adversarial_designer.py`](server/adversarial_designer.py) |
| LLM Judge (72B) | [`server/llm_judge.py`](server/llm_judge.py) |
| OpenEnv spec | [`openenv.yaml`](openenv.yaml) |

## Honest limitations

1. **Subprocess-based, not Kubernetes** — Real processes, but not real K8s pods. Tradeoff: 0.3s/step vs 30s/step.
2. **Free-tier GPU** — Training limited to Qwen2.5-1.5B (4-bit) on T4.
3. **Cascade/adversarial tiers are genuinely hard** — Resolution rate drops. This proves the environment is a rigorous benchmark, not a toy.
4. **Single-agent only** — No multi-agent team coordination (yet).
