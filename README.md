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

**The first RL environment that models real cascading production failures.**

> Fix the database lock → payment floods → worker OOMs → frontend 502s.  
> The agent must PREDICT what breaks AFTER the fix.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│              ONE CONTAINER (HF Space)                │
│                                                      │
│   OpenEnv Server (:7860)                             │
│   ├── Scenario Engine (17 static + dynamic)          │
│   ├── Cascade Engine (real causal chains)             │
│   ├── Adaptive Sampling (self-improving)              │
│   └── 5 Deterministic Graders                        │
│                                                      │
│   ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────────┐     │
│   │ payment  │ │ auth │ │ worker │ │ frontend │     │
│   │ :8001    │ │:8002 │ │ :8003  │ │ :8004    │     │
│   └────┬─────┘ └──┬───┘ └───┬────┘ └────┬─────┘     │
│        │          │         │           │            │
│   ┌────▼──────────▼─────────▼───────────▼────┐       │
│   │           Shared Infrastructure          │       │
│   │  SQLite DB │ Message Queue │ Log Files   │       │
│   └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

## 🎯 What Makes It Different

| Feature | Kube SRE Gym (Winner) | CloudSRE v2 |
|---------|----------------------|-------------|
| **Cascading failures** | ❌ Independent faults | ✅ Fix triggers new failure |
| **Real database** | ❌ No DB layer | ✅ Real SQLite with real locks |
| **Real auth** | ❌ No auth testing | ✅ Real JWT signing/validation |
| **Real message queue** | ❌ No queue | ✅ File-backed queue with backpressure |
| **Reset speed** | 30-60 sec | **<1 sec** |
| **Scenarios** | 7 hardcoded | 17 static + unlimited dynamic |
| **Tasks** | 0 in openenv.yaml | **5 graded tasks** |
| **Self-improvement** | ❌ | ✅ Adaptive weighted sampling |

## 🚀 Quick Start

### Run the environment
```bash
uv run server
# Environment running at http://localhost:7860
```

### Train an agent (GRPO)
```bash
# Terminal 1: Start env server
uv run server

# Terminal 2: Run training
python train.py --model-id Qwen/Qwen3-0.6B --task-id warmup --dataset-size 50
```

### Run inference
```bash
HF_TOKEN=your_token python inference.py
```

## 📊 5 Task Tiers

| Tier | Task | Difficulty | Scenarios |
|------|------|-----------|-----------|
| 1 | `warmup` | 0.15 | Single fault, clear signals |
| 2 | `single_fault` | 0.35 | + misleading red herrings |
| 3 | `cascade` | 0.55 | + cascading failure after fix |
| 4 | `multi_cascade` | 0.75 | + multiple concurrent cascades |
| 5 | `adversarial` | 0.60-0.90 | Dynamic, unique every episode |

## 🔧 Agent Action Space

The agent runs **real SRE commands**, not predefined tool calls:

```bash
curl http://localhost:8001/healthz          # Real HTTP health check
curl http://localhost:8001/metrics           # Real Prometheus metrics
cat /var/log/payment/error.log              # Real structured JSON logs
sqlite3 /data/app.db 'SELECT count(*) ...'  # Real SQL queries
restart_service payment                     # Real process restart
queue drain 10                              # Real queue management
```

## 🌊 The Cascade Mechanic

```
Phase 1: DB locked → payment 503 → queue fills → frontend 502
Phase 2: Agent fixes DB → 847 queued requests flood payment → OOM!
Phase 3: Agent must restart payment + drain queue at controlled rate
```

This is the #1 cause of extended production outages. No other RL environment models it.

## 📈 Training Pipeline

- **Algorithm:** GRPO with LoRA (DAPO loss)
- **5 reward signals:** total, triage, investigation, fix, cascade
- **Adaptive sampling:** Environment targets agent's weak scenarios
- **Multi-panel visualization:** Total reward + phase decomposition

## 🔌 Provider Agnostic

```bash
LLM_BACKEND=gemini     # Free (default, 1500 req/day)
LLM_BACKEND=openai     # GPT-4o
LLM_BACKEND=anthropic  # Claude
```

## 📁 Project Structure

```
cloud_sre_v2/
├── openenv.yaml          # OpenEnv spec (5 tasks, 5 graders)
├── models.py             # Action/Observation/State contracts
├── server/
│   ├── cloud_sre_environment.py  # Core MDP + adaptive sampling
│   ├── constants.py       # 17 scenarios + dynamic generator
│   ├── graders.py         # 5 deterministic graders
│   ├── command_executor.py # Routes real SRE commands
│   └── judge.py           # LLM judge (optional)
├── services/             # 4 real microservices
│   ├── payment_service.py # :8001 — SQLite + Queue
│   ├── auth_service.py    # :8002 — JWT auth
│   ├── worker_service.py  # :8003 — Queue consumer
│   ├── frontend_proxy.py  # :8004 — Reverse proxy
│   └── orchestrator.py    # Process lifecycle
├── infra/                # Shared infrastructure
│   ├── database.py       # Real SQLite with fault injection
│   ├── queue.py          # File-backed message queue
│   ├── metrics.py        # Prometheus-style metrics
│   └── logger.py         # Structured JSON logging
├── train.py              # GRPO training (TRL + vLLM)
└── inference.py          # Inference with any LLM
```

## License

Apache 2.0
