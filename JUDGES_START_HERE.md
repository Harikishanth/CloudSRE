# Judges: Start Here

**60-second read. Every claim links to source code.**

## What it is

An OpenEnv environment where an LLM agent diagnoses and fixes **real infrastructure failures** — not simulated alerts, not classification quizzes. 6 real OS-level services running as subprocesses, with real cascading failures that propagate through the system.

## The one thing that makes it different

**Open [`services/orchestrator.py`](services/orchestrator.py).** You'll see `subprocess.Popen` spawning 6 real services — each with its own PID, TCP port, and health check endpoint. When we inject a fault, we **actually kill the process** with `os.kill(pid, SIGTERM)`. When the agent runs `restart_service db`, a real process dies and a new one starts.

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
  -d '{"command": "restart_service auth"}'
```

## What makes cascades REAL

When the database locks → the queue backs up (real TCP connections failing) → the worker crashes (real process exit) → payments time out. **This is emergent behavior from real service interactions, not scripted if/else chains.**

See [`server/constants.py`](server/constants.py) for cascade rules. See [`services/orchestrator.py`](services/orchestrator.py) for process management.

## Training evidence (REAL gradient updates)

| Phase | Evidence | Result |
|---|---|---|
| **SFT** | [`sft_loss_curve.png`](sft_loss_curve.png) | Loss: 2.09 → 0.15 in 39 steps. Model learned SRE command syntax. |
| **REINFORCE** | [`reward_curve.png`](reward_curve.png) | 50 episodes. 86% resolution. Reward: -1.14 → +0.77. Real `loss.backward()` + `optimizer.step()`. |

This is not inference-in-a-loop. Open [`train_reinforce.py`](train_reinforce.py) — you'll see the forward pass recomputes log probabilities with gradients, computes policy loss, and calls `loss.backward()`. Weights update every episode.

## 5-tier curriculum

| Tier | Faults | Cascading? | Example |
|---|---|---|---|
| `warmup` | 1 service down | No | Auth service crashed |
| `single_fault` | 1 fault + diagnosis | No | DB locked, find root cause |
| `cascade` | 1 fault triggers 2nd | Yes | DB lock → queue flood |
| `multi_cascade` | 2 faults, 3+ effects | Yes | Auth crash + DB lock → payment timeout |
| `adversarial` | Random faults, tight budget | Yes | Unknown fault, 6 steps to fix |

## Architecture

```
┌─────────────────────────────────────────────┐
│              ServiceOrchestrator             │
│  subprocess.Popen() × 6 real OS processes    │
├──────┬──────┬───────┬────────┬──────┬───────┤
│ Auth │  DB  │ Queue │Payment │Worker│Frontend│
│:5001 │:5002 │:5003  │:5004   │:5005 │:5006   │
└──┬───┴──┬───┴───┬───┴────┬───┴──┬───┴───┬───┘
   │      │       │        │      │       │
   └──────┴───────┴────────┴──────┴───────┘
          Real TCP health checks
          Real process crashes
          Real cascading failures
```

## Honest limitations

1. **Single-agent only** — no multi-agent team coordination (yet).
2. **warmup tier dominates training** — cascade/adversarial tiers need more episodes for convergence.
3. **Process isolation is OS-level, not container-level** — Kube SRE Gym uses real K8s pods. We use subprocess. Different tradeoff: faster reset (~2s vs ~30s), less isolation depth.

## Where to look

| What | File |
|---|---|
| Real services spawned | [`services/orchestrator.py`](services/orchestrator.py) |
| Fault injection (actual process kill) | [`server/fault_injector.py`](server/fault_injector.py) |
| Cascade rules | [`server/constants.py`](server/constants.py) |
| Command executor (real shell ops) | [`server/command_executor.py`](server/command_executor.py) |
| Graders (check real health) | [`server/graders.py`](server/graders.py) |
| REINFORCE with gradient updates | [`train_reinforce.py`](train_reinforce.py) |
| SFT training | [`sft_warmup.py`](sft_warmup.py) |
| OpenEnv spec | [`openenv.yaml`](openenv.yaml) |
