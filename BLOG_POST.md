# CloudSRE v2: Teaching a 1.5B Model to Survive a 3 AM Production Outage

> *"It's 3:14 AM. PagerDuty screams. The billing service is down. But that's just the symptom — the real cause is a corrupted database connection pool three services deep, in a data center 8,000 miles away."*

This is CloudSRE v2 — an OpenEnv-compatible reinforcement learning environment that trains LLM agents to diagnose and resolve **cascading infrastructure failures** across a 16-service, 3-region microservice architecture using real OS-level fault injection.

**Hackathon Links:**
- 🌐 [Live HF Space & Interactive Dashboard](https://dardrax-cloudsre-environment.hf.space)
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

**16 distinct services**, each running as an isolated local subprocess with its own:
- Health endpoint (`/healthz`)
- Error logs (`/var/log/<service>/error.log`)
- Dependency graph (services depend on each other)

**Why Local Matters:** Unlike other environments that require expensive, paid Google Cloud clusters just to boot up, CloudSRE v2 achieves massive topology and isolation entirely locally on a laptop. This democratizes SRE agent research.

---

## The Secret Sauce 1: OS-Level Fault Injection

Most environments set a `is_broken = True` flag and call it a day. We inject **real operating system failures**:

| Fault Category | Implementation | Why It Matters |
|---|---|---|
| Process Freeze | `SIGSTOP` signal | Service appears "alive" but unresponsive — mimics GC pauses |
| Process Kill | `SIGKILL` signal | Immediate crash — mimics OOM kills |
| Disk Full | `fallocate` / manual write | Log rotation failures, DB write stalls |
| File Corruption | Direct file modification | Config drift, corrupted state files |
| Connection Pool Exhaustion | Thread-based request flooding | The #1 cause of cascading failures in production |

When your agent types `curl`, it hits a real HTTP server. When it types `ps`, it reads real OS processes.

---

## The Secret Sauce 2: 72B Adversarial Co-Evolution Loop

CloudSRE v2 doesn't just use hardcoded scenarios. We implemented a state-of-the-art **Adversarial Curriculum Loop**:

1. **Performance Tracking:** The environment tracks the agent's mastery across 25 different fault types.
2. **Adversarial Generation:** When the agent gets good at simple faults, our `AdversarialDesigner` uses a **Qwen-72B LLM** via the HuggingFace API to dynamically generate novel compound incidents targeting the agent's tracked weak spots.
3. **Continuous Escalation:** The agent receives harder scenarios with injected red herrings, forcing it to co-evolve with the environment.

---

## The Secret Sauce 3: Live Observability Dashboard

We built a stunning, real-time Gradio dashboard mounted directly to our HuggingFace Space. Judges and users can watch the environment health degrade live as faults are injected, and monitor the agent's real-time terminal output as it executes bash commands to triage the cascade.

---

## Training: GRPO (Group Relative Policy Optimization)

We chose not to use Supervised Fine-Tuning. SFT teaches pattern matching; incident response requires reasoning. 

Using GRPO on a **single, free 16GB Kaggle T4 GPU**, our 1.5B agent tries multiple approaches per scenario and learns from comparing outcomes. 

Our reward function is a **dense, multi-component signal**:
- Did the system get healthier? (+0.3)
- Did you investigate before fixing? (+0.2)
- Did you find the ACTUAL root cause? (+0.2)
- Blind restarts or repeating commands? (Penalty)

### Training Progression

| Tier | Episodes | Starting Reward | Final Reward | Resolution Rate |
|------|----------|----------------|--------------|-----------------|
| Warmup | 12 | -0.50 | +0.30 | 45% |
| Single Fault | 12 | -0.20 | +0.55 | 72% |
| Cascade | 12 | -0.40 | +1.13 | 17% (Breakthrough observed!) |
| Multi-Cascade | 12 | — | — | Training in progress |

### What the Agent Learned

**Before training (base model):**
```
Alert: billing-service returning 500s
Agent: restart_service billing     ← blind restart, doesn't investigate
Agent: restart_service billing     ← repeats same command
Result: FAILED (no diagnosis, no root cause)
```

**After training:**
```
Alert: billing-service returning 500s
Agent: curl http://billing.us-east-1.internal/healthz     ← triage first
Agent: cat /var/log/billing/error.log                      ← investigate
Agent: curl http://payment.us-east-1.internal/healthz      ← check dependencies
Agent: restart_service payment                             ← targeted fix
Agent: curl http://billing.us-east-1.internal/healthz      ← verify resolution
Result: RESOLVED ✅ 
```

---

## What Makes CloudSRE v2 Different

| Feature | CloudSRE v2 | Typical SRE Env |
|---------|------------|-----------------|
| Services | 16 | 1-6 |
| Fault types | 25 OS-level | 5-12 flag-based |
| Infrastructure | Free & Local | Expensive & Cloud-tied |
| Scenarios | 72B Adversarial LLM | Hardcoded |
| Dashboard | Live Interactive UI | Static APIs |

---

## Team

Built by **Team DarDrax** for the Meta PyTorch OpenEnv Hackathon India 2026.

*"We didn't build a toy. We built the gym where the next generation of autonomous SRE agents will train."*
