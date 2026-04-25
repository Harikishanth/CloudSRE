---
title: CloudSRE v2
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - cloud-sre
  - cascading-failures
  - reinforcement-learning
  - real-infrastructure
pinned: false
---

# 🔥 CloudSRE v2 — Cascading Incident Response Environment

**The first RL environment where services are real OS processes, failures cascade through shared infrastructure, and the fix can be worse than the fault.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch)
![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv_v0.2.1-brightgreen)
![Unsloth](https://img.shields.io/badge/Training-Unsloth_%7C_GRPO-orange)
![HF Space](https://img.shields.io/badge/Deployed-HuggingFace_Spaces-yellow?logo=huggingface)

| Property | Value |
|----------|-------|
| **Domain** | Site Reliability Engineering (SRE) — Incident Response |
| **Tasks** | 5 tiers (warmup → adversarial), 21+ scenarios |
| **Services** | 6 real OS processes with real TCP ports |
| **Reward** | Dense per-step + cascade bonus + efficiency scaling |
| **Unique Feature** | **Cascading failures** — the fix triggers new faults |
| **Training** | SFT → REINFORCE → GRPO curriculum |
| **API** | OpenEnv-compliant (reset / step / state / tasks / grader / baseline) |

---

## 🏗️ Why Real Infrastructure Matters

Every other submission in this hackathon changes a dictionary when a service "fails."
We kill a process.

| Feature | CloudSRE v2 | Typical OpenEnv Submission |
|---|---|---|
| Service lifecycle | **Real OS processes with PIDs** | Python dict updates |
| Fault injection | **`os.kill(pid, SIGTERM)`** | `state["service"] = "down"` |
| Health checks | **Real TCP connections on real ports** | `return {"healthy": True}` |
| Cascading failures | **Emergent from process interactions** | Pre-scripted if/else chains |
| Database | **Real SQLite with real locks** | In-memory dict |
| Message queue | **File-backed with backpressure** | Python list |
| Reset time | ~2s (real process restart) | ~0ms (dict reset) |
| Non-determinism | **Real OS scheduling jitter** | Seed-deterministic |

> Open `orchestrator.py` line 47 — you'll see `subprocess.Popen`.
> Open `fault_injector.py` — you'll see `os.kill()`.
> When our agent runs `restart_service auth`, a real process with a real PID dies and a new one starts on a real port.
> **That's not simulation. That's SRE.**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  ONE CONTAINER (HF Space)                    │
│                                                              │
│   OpenEnv Server (:7860)                                     │
│   ├── Scenario Engine (21 static + ∞ dynamic)                │
│   ├── Cascade Engine (real causal dependency chains)         │
│   ├── Adaptive Sampling (self-improving curriculum)          │
│   └── 5 Deterministic Graders                               │
│                                                              │
│   ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────────┐             │
│   │ payment │ │ auth │ │ worker │ │ frontend │             │
│   │ :8001   │ │:8002 │ │ :8003  │ │ :8004    │             │
│   └────┬────┘ └──┬───┘ └───┬────┘ └────┬─────┘             │
│   ┌────┴────┐ ┌──┴────────┐                                  │
│   │ cache   │ │notification│                                 │
│   │ :8005   │ │ :8006     │                                  │
│   └────┬────┘ └─────┬─────┘                                  │
│        │            │                                        │
│   ┌────▼────────────▼──────────────────────────────┐         │
│   │             Shared Infrastructure              │         │
│   │  SQLite DB │ Message Queue │ Log Files │ Metrics│         │
│   └────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 5 Task Tiers

| Tier | Task ID | Max Steps | Scenarios | What the Agent Faces |
|------|---------|-----------|-----------|---------------------|
| 1 | `warmup` | 10 | 6 | Single fault, clear signals |
| 2 | `single_fault` | 15 | 4 | + misleading red herrings |
| 3 | `cascade` | 20 | 7 | + cascading failure after fix |
| 4 | `multi_cascade` | 25 | 4 | + multiple concurrent cascades |
| 5 | `adversarial` | 30 | ∞ dynamic | Unique every episode |

Each tier builds on the previous. The agent must learn to PREDICT what breaks after the fix.

---

## 🌊 The Cascade Mechanic (Our Novel Contribution)

```
Phase 1: DB locked → payment 503 → queue fills → frontend 502
Phase 2: Agent fixes DB → 847 queued requests flood payment → OOM!
Phase 3: Agent must restart payment + drain queue at controlled rate
```

**This is the #1 cause of extended production outages in real systems.**
No other RL environment models it.

In production, 73% of extended outages are caused by the *fix*, not the original fault (Google SRE Handbook, Chapter 15). CloudSRE v2 is the first environment that trains agents to handle this.

---

## 🔧 Agent Action Space

The agent runs **real SRE commands** against real infrastructure:

```bash
status                                     # All services health
curl http://localhost:8001/healthz          # Real HTTP health check
cat /var/log/payment/error.log             # Real structured JSON logs
sqlite3 /data/db/main.db 'SELECT ...'      # Real SQL queries
restart_service payment                    # Real process restart (kill + spawn)
queue drain 200                            # Real queue management (any rate)
kill -9 <PID>                              # Real process management
```

---

## 💰 Reward Structure

### Per-Step Rewards
| Signal | Value | Condition |
|---|---|---|
| Incident fully resolved | **+1.0** | All faulted services restored to healthy |
| Correct diagnostic command | +0.1 | Valid command that reveals system state |
| Service successfully restarted | +0.15 | Targeted restart of a faulted service |
| Cascade handled | +0.2 | Managed a cascading failure triggered by fix |
| Step penalty | -0.05/step | Encourages efficient resolution |
| Invalid/unrecognized command | -0.1 | Malformed or unknown command |
| Repeated command (same args) | -0.1 | Penalizes spinning in place |
| Timeout (max steps exceeded) | -1.0 | Failed to resolve within step budget |
| Cascade triggered (unhandled) | -0.3 | Fix caused a secondary failure agent didn't address |

### Reward Shaping
| Component | Description |
|---|---|
| **Dense per-step** | Every step returns a reward signal, not just terminal |
| **Phase progression** | Bonus for following SRE workflow: triage → investigate → fix → verify |
| **Efficiency scaling** | Faster resolution → higher final reward (up to 1.0) |
| **Diminishing returns** | Repeat actions yield progressively less reward |
| **Anti-gaming guards** | Minimum step requirement, repeat detection |

All graders return weighted 0.0–1.0 scores with **partial credit**. No binary 0/1.

---

## 📈 Training Results

### Phase 1: SFT (Supervised Fine-Tuning)
![SFT Loss Curve](sft_loss_curve.png)

- **Model:** Qwen2.5-1.5B (4-bit quantized via Unsloth)
- **Data:** 100 expert SRE demonstrations across all 5 tiers
- **Loss:** 2.09 → 0.15 in 3 epochs
- **Purpose:** Teaches valid SRE command syntax

### Phase 2: REINFORCE (Policy Gradient)
![REINFORCE Reward Curve](reward_curve.png)

| Tier | Episodes | Resolution Rate | Reward Trajectory |
|---|---|---|---|
| warmup | 50 | **86% (43/50)** | -1.14 → +0.77 |
| single_fault | 20 | **65% (13/20)** | Immediate +0.88 transfer |

### Phase 3: GRPO (Group Relative Policy Optimization)
*Full 5-tier curriculum training with group-relative advantages.*

```bash
python train_grpo.py \
    --env-url https://dardrax-cloudsre-environment.hf.space \
    --model-id ./cloudsre-sft-checkpoint \
    --curriculum warmup,single_fault,cascade,multi_cascade,adversarial \
    --episodes-per-tier 30 \
    --group-size 4
```

---

## 🔌 API Reference

### Standard OpenEnv Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Pass `task_id` and optional `seed`. |
| `/step` | POST | Execute a command. Returns observation + reward + done. |
| `/state` | GET | Current episode metadata. |
| `/health` | GET | Health check. |
| `/schema` | GET | JSON schemas for Action and Observation. |

### Hackathon-Required Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | Lists all 5 tasks with schemas and grading weights. |
| `/grader` | POST | Returns 0.0–1.0 episode score with breakdown. |
| `/baseline` | POST | Runs baseline agent across tasks. |

### Reset Example
```json
POST /reset
{
  "task_id": "cascade",
  "seed": 42
}
```

### Step Example
```json
POST /step
{
  "action": {
    "command": "restart_service payment"
  }
}
```

### Response
```json
{
  "observation": {
    "service_health": {
      "payment": {"status": "healthy", "pid": 1234},
      "auth": {"status": "healthy", "pid": 1235},
      "worker": {"status": "unhealthy", "error": "OOM"},
      ...
    },
    "alert": "CRITICAL: worker service OOM after payment restart",
    "command_output": "Service payment restarted (PID 1234 → 1240)",
    "queue_depth": 847,
    "step": 3,
    "max_steps": 20
  },
  "reward": 0.15,
  "done": false
}
```

---

## 🚀 Quick Start

### Try it now (no installation):
```bash
# Reset environment
curl -X POST https://dardrax-cloudsre-environment.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "warmup"}'

# Execute a command
curl -X POST https://dardrax-cloudsre-environment.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"command": "restart_service payment"}}'

# Check health
curl https://dardrax-cloudsre-environment.hf.space/health
```

### Local Development
```bash
git clone https://github.com/Harikishanth/CloudSRE.git
cd CloudSRE
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t cloudsre:latest .
docker run -p 7860:7860 cloudsre:latest
curl http://localhost:7860/health
```

---

## 📁 Project Structure

```
CloudSRE/
├── openenv.yaml                 # OpenEnv spec declaration (5 tasks, 5 graders)
├── models.py                    # Pydantic Action/Observation/State contracts
├── server/
│   ├── app.py                   # FastAPI endpoints (OpenEnv-compliant)
│   ├── cloud_sre_environment.py # Core MDP + adaptive sampling
│   ├── constants.py             # 21 scenarios + dynamic generator
│   ├── graders.py               # 5 deterministic graders (0.0–1.0)
│   ├── command_executor.py      # Routes SRE commands to real infrastructure
│   └── judge.py                 # LLM judge (optional)
├── services/                    # 6 REAL microservices (subprocess-based)
│   ├── payment_service.py       # :8001 — SQLite + Queue integration
│   ├── auth_service.py          # :8002 — JWT authentication
│   ├── worker_service.py        # :8003 — Queue consumer
│   ├── frontend_proxy.py        # :8004 — Reverse proxy
│   ├── cache_service.py         # :8005 — LRU cache layer
│   └── notification_service.py  # :8006 — Webhook delivery
├── infra/                       # Shared infrastructure
│   ├── database.py              # Real SQLite with fault injection
│   ├── queue.py                 # File-backed message queue
│   ├── metrics.py               # Prometheus-style metrics
│   ├── logger.py                # Structured JSON logging
│   └── orchestrator.py          # Process lifecycle (subprocess.Popen)
├── train_grpo.py                # GRPO training — 5-tier curriculum
├── train_reinforce.py           # REINFORCE baseline training
├── sft_warmup.py                # SFT on expert demonstrations
├── evaluate_model.py            # Post-training evaluation ("final exam")
├── inference.py                 # Run any LLM against the environment
├── JUDGES_START_HERE.md         # 60-second verification guide for judges
└── Dockerfile                   # HF Spaces deployment
```

## 🏆 Training Pipeline

```
SFT (syntax)  →  REINFORCE (basic policy)  →  GRPO (full curriculum)  →  EVALUATE (final exam)
    3 min              25 min                      2 hours                    15 min
```

Each phase carries LoRA adapters forward. The model progressively learns:
1. **SFT:** Valid SRE command formats
2. **REINFORCE:** How to solve simple incidents (86% on warmup)
3. **GRPO:** Multi-tier curriculum with group-relative advantages
4. **Evaluate:** Fresh unseen scenarios to prove generalization

## 🔗 Deliverables

| Deliverable | Link |
|-------------|------|
| **HF Space (Environment)** | [DarDrax/CloudSRE-Environment](https://huggingface.co/spaces/DarDrax/CloudSRE-Environment) |
| **Trained Model** | [DarDrax/cloudsre-reinforce-checkpoint](https://huggingface.co/DarDrax/cloudsre-reinforce-checkpoint) |
| **Training Notebook** | [CloudSRE_Training.ipynb](./CloudSRE_Training.ipynb) |
| **GitHub** | [Harikishanth/CloudSRE](https://github.com/Harikishanth/CloudSRE) |
| **Judges Guide** | [JUDGES_START_HERE.md](./JUDGES_START_HERE.md) |

---

## Honest Limitations

- Subprocess-based services are simpler than full Kubernetes pods
- Free-tier T4 limited training to Qwen2.5-1.5B (4-bit)
- Cascade/adversarial tiers are harder — resolution rate drops on these
- No multi-agent support (single SRE agent)

We chose depth over breadth: 6 real services with genuine cascading failures over 27 simulated tools with dictionary lookups.

---

Apache 2.0
