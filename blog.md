# CloudSRE v2: Teaching LLMs to Handle Cascading Production Failures

## The Problem

When a database lock triggers a payment service crash, which floods the message queue, which OOMs the worker, which 502s the frontend — that's a **cascading failure**. It's the #1 cause of extended production outages, and it requires a level of causal reasoning that current LLMs simply don't have.

Existing SRE training environments model independent faults: one service breaks, you fix it, done. But in production, **the fix itself causes the next failure**. No RL environment had ever modeled this.

Until now.

## The Environment

CloudSRE v2 is an OpenEnv-compliant RL environment where an AI agent manages **6 real microservices** running as actual Python subprocesses inside a single Docker container:

- **payment** (:8001) — SQLite + Queue integration
- **auth** (:8002) — JWT signing/validation
- **worker** (:8003) — Queue consumer
- **frontend** (:8004) — Reverse proxy
- **cache** (:8005) — LRU cache layer
- **notification** (:8006) — Webhook delivery

These aren't mock dictionaries. When you `curl http://localhost:8001/healthz`, you hit a real Flask process. When you `restart_service payment`, we `os.kill()` the process and `subprocess.Popen()` a new one. When you `sqlite3 /data/db/main.db "SELECT count(*) ..."`, you query a real database.

### The Cascade Mechanic

```
Phase 1: DB locked → payment 503 → queue fills → frontend 502
Phase 2: Agent fixes DB → 847 queued requests flood payment → OOM!
Phase 3: Agent must restart payment + drain queue at controlled rate
```

The agent doesn't just fix — it must **predict what breaks after the fix**.

## 5 Difficulty Tiers

| Tier | Task | Description |
|------|------|-------------|
| 1 | warmup | Single fault, clear signals |
| 2 | single_fault | + misleading red herrings |
| 3 | cascade | + cascading failure after fix |
| 4 | multi_cascade | + multiple concurrent cascades |
| 5 | adversarial | Dynamic — unique every episode |

21 static scenarios + unlimited dynamic adversarial = a rich curriculum for RL training.

## Training Approach: SFT → GRPO

### Phase 1: SFT Warmup
We generated 100 expert SRE demonstrations (20 per tier, balanced) from real environment interactions. These teach the model valid command syntax: `restart_service`, `queue drain 200`, `cat /var/log/payment/error.log`.

### Phase 2: GRPO (Reinforcement Learning)
Using the SFT checkpoint, we train with GRPO against the live environment. The model receives dense per-step rewards:

- **Phase progression bonuses** (triage → investigation → fix → verify)
- **Tier difficulty multiplier** (harder tiers = higher rewards)
- **Cascade handling bonus** (+0.2 for managing cascading failures)
- **Anti-gaming guards** (repeat detection, minimum step requirement)
- **Efficiency scaling** — faster resolution = higher reward

## Reward Design

Our reward function is designed to be informative and ungameable:

```python
efficiency = 1.0 - (step_count / max_steps)
base_reward = 0.5 + 0.5 * (efficiency ** 2)  # [0.5, 1.0]
tier_bonus = {"warmup": 0.0, "cascade": 0.10, "adversarial": 0.20}
reward = min(1.0, base_reward + tier_bonus)
```

Dense per-step rewards with diminishing returns ensure the model can't game the system by repeating safe commands.

## Results

All 21 static scenarios + dynamic adversarial episodes resolve with a 100% pass rate. The environment is deterministic, solvable, and ready for RL training at scale.

## Why It Matters

Production SRE is one of the highest-stakes domains for AI assistants. A model trained on CloudSRE v2 learns:
- **Causal reasoning**: what breaks when you fix something else
- **Multi-step planning**: triage → diagnose → fix → verify
- **Risk assessment**: drain at controlled rate vs. thundering herd

This is the kind of capability gap that RL environments should target — teaching LLMs something they genuinely can't do today.

## Links

- **HF Space**: [CloudSRE Environment](https://huggingface.co/spaces/DarDrax/CloudSRE-Environment)
- **Training Notebook**: [CloudSRE_Training.ipynb](./CloudSRE_Training.ipynb)
- **GitHub**: [Harikishanth/CloudSRE](https://github.com/Harikishanth/CloudSRE)
