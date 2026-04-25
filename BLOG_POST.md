# CloudSRE v2: Teaching a 1.5B Model to Survive a 3 AM Production Outage

> *"It's 3:14 AM. PagerDuty screams. The billing service is down. But that's just the symptom — the real cause is a corrupted database connection pool three services deep, in a data center 8,000 miles away."*

This is CloudSRE v2 — an OpenEnv-compatible reinforcement learning environment that trains LLM agents to diagnose and resolve **cascading infrastructure failures** across a 16-service, 3-region microservice architecture using real OS-level fault injection.

**Hackathon Links:**
- 🌐 [Live HF Space](https://dardrax-cloudsre-environment.hf.space)
- 🏋️ [Training Notebook](CloudSRE_Training_Colab.py)
- 🤗 [Trained Model](https://huggingface.co/DarDrax/cloudsre-1.5B-FINAL)
- 📊 [WandB Dashboard](https://wandb.ai/-dardrax-/CloudSRE-Hackathon-Run)

---

## The Problem Nobody Else Is Solving

Most SRE training environments simulate **one fault on one server**. *"nginx crashed. Restart it."* A junior engineer could solve that.

Real production incidents at scale look nothing like that:

```
3:14 AM  — Alert: billing-service returning 500s
3:15 AM  — You check billing. It's healthy??
3:17 AM  — Wait — payment-service is timing out on DB writes
3:19 AM  — The DB connection pool is exhausted in us-east-1
3:22 AM  — Root cause: a config drift in the cache layer caused
           a thundering herd that saturated the connection pool,
           which cascaded upstream through payment → billing → frontend
3:31 AM  — You fix the cache config. Pool drains. Services recover.
3:45 AM  — Post-mortem reveals the cache drift was caused by a
           silent deployment 6 hours ago in eu-west-1.
```

**That's** what CloudSRE v2 simulates. Not toy problems — the real thing.

---

## Architecture: 16 Services × 3 Regions × 25 Fault Types

We built the most complex microservice topology in the competition:

```
                        ┌─────────────────────────────────────────┐
                        │           LOAD BALANCER                 │
                        └──────────┬──────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
     ┌──────▼──────┐       ┌──────▼──────┐       ┌──────▼──────┐
     │  us-east-1  │       │  eu-west-1  │       │  ap-south-1 │
     └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
            │                     │                      │
    ┌───┬───┼───┬───┐    ┌───┬───┼───┬───┐     ┌───┬───┼───┬───┐
    │   │   │   │   │    │   │   │   │   │     │   │   │   │   │
   FE  API AUTH PAY DB  FE  API AUTH PAY DB   FE  API AUTH PAY DB
        │         │           │         │          │         │
      SRCH  BILL CACHE     SRCH  BILL CACHE    SRCH  BILL CACHE
        │                     │                     │
     WORKER              WORKER                 WORKER
        │                     │                     │
     NOTIFY              NOTIFY                 NOTIFY
```

**16 distinct services**, each running as an isolated process with its own:
- Health endpoint (`/healthz`)
- Error logs (`/var/log/<service>/error.log`)
- Metrics (CPU, memory, latency, error rate)
- Dependency graph (services depend on each other)

### Why This Matters for RL

A single-server environment has ~9 possible states. Our environment has:

```
16 services × 3 regions × 4 health states × 25 fault types = ~4,800 unique states
```

This forces the agent to develop **genuine reasoning** — it can't memorize its way to victory.

---

## The Secret Sauce: OS-Level Fault Injection

Most environments set a `is_broken = True` flag and call it a day. We inject **real operating system failures**:

| Fault Category | Implementation | Why It Matters |
|---|---|---|
| Process Freeze | `SIGSTOP` signal | Service appears "alive" but unresponsive — mimics GC pauses |
| Process Kill | `SIGKILL` signal | Immediate crash — mimics OOM kills |
| Disk Full | `fallocate` / manual write | Log rotation failures, DB write stalls |
| File Corruption | Direct file modification | Config drift, corrupted state files |
| Connection Pool Exhaustion | Thread-based request flooding | The #1 cause of cascading failures in production |
| Rate Limiting | `SIGSTOP` + HTTP state degradation | API throttling that propagates upstream |

The agent doesn't just read "service X is unhealthy" — it has to **investigate**, **diagnose root cause**, and **fix the actual OS-level issue**.

---

## Training: 5-Tier Curriculum via GRPO

We don't throw the agent into the deep end. We use a **5-tier curriculum** that progressively increases difficulty:

### Tier 1: Warmup (Orientation)
*"Here's your environment. Learn to check health endpoints."*

The agent learns basic commands: `curl /healthz`, `cat /var/log/*/error.log`, `restart_service`.

### Tier 2: Single Fault
*"One service is broken. Find it and fix it."*

The agent learns the triage → investigate → fix → verify loop.

### Tier 3: Cascade
*"One fault has caused a chain reaction across 3 services."*

The agent must trace dependency chains backwards to find root cause.

### Tier 4: Multi-Cascade
*"Multiple independent faults are happening simultaneously."*

The agent must prioritize and handle concurrent incidents.

### Tier 5: Adversarial
*"We're actively trying to fool you with misleading symptoms."*

Red herring alerts, services that look broken but aren't, and faults that shift mid-episode.

### Why GRPO Over SFT?

We deliberately chose **not** to use Supervised Fine-Tuning. Here's why:

SFT teaches the model to mimic expert trajectories — but incident response isn't about memorization. Two identical symptoms can have completely different root causes depending on the system state. SFT would produce a model that pattern-matches; GRPO produces a model that **reasons**.

With Group Relative Policy Optimization, the model tries 4 different approaches per scenario and learns from comparing outcomes. It discovers strategies we never programmed.

---

## Reward Engineering: Not Just +1/-1

Our reward function is a **dense, multi-component signal** designed to prevent reward hacking:

```python
reward = (
    0.3 × health_delta          # Did the system get healthier?
  + 0.2 × diagnostic_quality    # Did you investigate before fixing?
  + 0.2 × command_efficiency    # Are you using diverse, relevant commands?
  + 0.2 × root_cause_accuracy   # Did you find the ACTUAL root cause?
  + 0.1 × resolution_speed      # How fast did you resolve it?
  - penalties                    # Blind restarts, repeated commands, wrong fixes
)
```

**Key anti-hacking measures:**
- Restarting without investigating = penalty
- Repeating the same command = diminishing returns
- Fixing symptoms instead of root cause = no resolution bonus
- Random commands = efficiency penalty

---

## Results

### Training Progression

Starting from Qwen2.5-1.5B-Instruct (base), our agent progressed through:

| Tier | Episodes | Starting Reward | Final Reward | Resolution Rate |
|------|----------|----------------|--------------|-----------------|
| Warmup | 25 | -0.50 | +0.30 | 45% |
| Single Fault | 25 | -0.20 | +0.55 | 72% |
| Cascade | 25 | -0.40 | — | Training in progress |
| Multi-Cascade | 25 | — | — | Training in progress |
| Adversarial | 25 | — | — | Training in progress |

*Live training dashboard: [WandB →](https://wandb.ai/-dardrax-/CloudSRE-Hackathon-Run)*

### What the Agent Learned

**Before training (base model):**
```
Alert: billing-service returning 500s
Agent: restart_service billing     ← blind restart, doesn't investigate
Agent: restart_service billing     ← repeats same command
Agent: restart_service payment     ← random guessing
Result: FAILED (no diagnosis, no root cause)
```

**After training (Leg 1 — single fault):**
```
Alert: billing-service returning 500s
Agent: curl http://billing.us-east-1.internal/healthz     ← triage first
Agent: cat /var/log/billing/error.log                      ← investigate
Agent: curl http://payment.us-east-1.internal/healthz      ← check dependencies
Agent: cat /var/log/payment/error.log                      ← found the real issue!
Agent: restart_service payment                             ← targeted fix
Agent: curl http://billing.us-east-1.internal/healthz      ← verify resolution
Result: RESOLVED ✅ (correct root cause, efficient, verified)
```

---

## Technical Stack

| Component | Technology |
|---|---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit via Unsloth |
| RL Algorithm | GRPO (Group Relative Policy Optimization) |
| Training Framework | HuggingFace TRL + Unsloth |
| Environment | 16-service Docker orchestrator on HF Space |
| Fault Injection | OS-level (SIGSTOP, SIGKILL, fallocate, file I/O) |
| Monitoring | Weights & Biases |
| Regions | 3 (us-east-1, eu-west-1, ap-south-1) |

---

## What Makes CloudSRE v2 Different

| Feature | CloudSRE v2 | Typical SRE Env |
|---------|------------|-----------------|
| Services | 16 | 1-6 |
| Regions | 3 | 1 |
| Fault types | 25 OS-level | 5-12 flag-based |
| Cascading failures | ✅ Multi-hop | ❌ or single-hop |
| Fault injection | OS signals | HTTP flags |
| Curriculum | 5-tier adaptive | Fixed difficulty |
| Anti-reward-hacking | Dense multi-component | Sparse +1/-1 |

---

## Future Work

1. **Incident Postmortem Generator** — LLM-based narrative report from episode transcripts
2. **Live Observability Dashboard** — Gradio UI visualizing real-time service health
3. **72B Judge Model** — Using Qwen2.5-72B as an automated evaluator for agent performance
4. **Multi-Agent SRE** — Training a "Commander + Responder" pair that coordinates incident response

---

## Team

Built by **Team DarDrax** for the Meta PyTorch OpenEnv Hackathon India 2026.

*"We didn't build a toy. We built the gym where the next generation of autonomous SRE agents will train."*
