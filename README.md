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
pinned: false
---

# CloudSRE v2 — Cascading Incident Response Environment

**The first RL environment that models real cascading production failures with real infrastructure.**

> Fix the database lock → payment floods → worker OOMs → frontend 502s.  
> The agent must PREDICT what breaks AFTER the fix.

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

## 🎯 What Makes It Different

| Feature | Kube SRE Gym | CloudSRE v2 |
|---------|-------------|-------------|
| **Cascading failures** | ❌ Independent faults | ✅ Fix triggers new failure |
| **Services** | 3 simulated | **6 real** (subprocess + ports) |
| **Database** | ❌ No DB layer | ✅ Real SQLite with real locks |
| **Auth** | ❌ No auth testing | ✅ Real JWT signing/validation |
| **Message queue** | ❌ No queue | ✅ File-backed with backpressure |
| **Cache** | ❌ | ✅ LRU cache with thundering herd |
| **Reset speed** | 30-60 sec (GKE) | **<1 sec** (subprocess) |
| **Scenarios** | 7 hardcoded | **21 static + unlimited dynamic** |
| **Tasks** | 0 in openenv.yaml | **5 graded tasks** |
| **Curriculum** | ❌ | ✅ Adaptive weighted sampling |

## 🚀 Training Pipeline

### Two-Phase Approach: SFT → GRPO

**Phase 1 — SFT Warmup** (teaches command vocabulary):
```bash
python sft_warmup.py --model-id unsloth/Qwen3-1.7B --epochs 3
```

Uses 60 expert SRE demonstrations across all 5 tiers. Teaches the model valid command formats (`restart_service`, `queue drain`, `cat error.log`).

**Phase 2 — GRPO Training** (teaches strategy):
```bash
python train_colab.py \
  --env-url https://dardrax-cloudsre-environment.hf.space \
  --model-id ./cloudsre-sft-checkpoint \
  --task-id warmup \
  --episodes 50 \
  --no-hints  # organic training for Qwen3+
```

Dense reward signals guide the model from triage → investigation → fix → verification.

## 📊 5 Task Tiers

| Tier | Task | Max Steps | Scenarios | Description |
|------|------|-----------|-----------|-------------|
| 1 | `warmup` | 10 | 6 | Single fault, clear signals |
| 2 | `single_fault` | 15 | 4 | + misleading red herrings |
| 3 | `cascade` | 20 | 7 | + cascading failure after fix |
| 4 | `multi_cascade` | 25 | 4 | + multiple concurrent cascades |
| 5 | `adversarial` | 30 | ∞ dynamic | Unique every episode |

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

## 🌊 The Cascade Mechanic

```
Phase 1: DB locked → payment 503 → queue fills → frontend 502
Phase 2: Agent fixes DB → 847 queued requests flood payment → OOM!
Phase 3: Agent must restart payment + drain queue at controlled rate
```

This is the #1 cause of extended production outages. No other RL environment models it.

## 📈 Reward Design

- **Dense per-step rewards** with diminishing returns
- **Phase progression bonuses** (triage → investigation → fix → verify)
- **Cascade handling bonus** (+0.2 for managing cascading failures)
- **Anti-gaming guards** (repeat detection, minimum step requirement)
- **Efficiency scaling** — faster resolution = higher reward (up to 1.0)

## 📁 Project Structure

```
cloud_sre_v2/
├── openenv.yaml                # OpenEnv spec (5 tasks, 5 graders)
├── models.py                   # Action/Observation/State contracts
├── server/
│   ├── cloud_sre_environment.py  # Core MDP + adaptive sampling
│   ├── app.py                  # FastAPI server + persistent env factory
│   ├── constants.py            # 21 scenarios + dynamic generator
│   ├── graders.py              # 5 deterministic graders
│   ├── command_executor.py     # Routes real SRE commands to infra
│   └── judge.py                # LLM judge (optional)
├── services/                   # 6 real microservices
│   ├── payment_service.py      # :8001 — SQLite + Queue integration
│   ├── auth_service.py         # :8002 — JWT auth
│   ├── worker_service.py       # :8003 — Queue consumer
│   ├── frontend_proxy.py       # :8004 — Reverse proxy
│   ├── cache_service.py        # :8005 — LRU cache layer
│   ├── notification_service.py # :8006 — Webhook delivery
│   └── orchestrator.py         # Process lifecycle management
├── infra/                      # Shared infrastructure
│   ├── database.py             # Real SQLite with fault injection
│   ├── queue.py                # File-backed message queue
│   ├── metrics.py              # Prometheus-style metrics
│   └── logger.py               # Structured JSON logging
├── sft_warmup.py               # Phase 1: SFT on expert demos
├── train_colab.py              # Phase 2: GRPO training loop
├── sft_training_data.jsonl     # 60 expert SRE demonstrations
└── inference.py                # Inference with any LLM
```

## License

Apache 2.0
