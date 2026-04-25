# CloudSRE v2 — Full Project Context for AI Assistant

## Project Overview
CloudSRE v2 is an OpenEnv-compliant RL environment for training AI agents to resolve real production SRE incidents. It's our submission for the **OpenEnv Hackathon** (PyTorch + Cerebral Valley + HuggingFace). Deadline is April 26, 2026.

### What Makes Us Different
We run **16 REAL microservices as separate OS processes** (subprocess.Popen). When we say "service is down," we mean `os.kill(pid, SIGTERM)`. When we say "restart," we mean a NEW subprocess with a NEW PID. Every other submission uses Python dicts.

---

## Architecture

```
CloudSRE/
├── openenv.yaml                 # OpenEnv spec (5 tasks, 5 graders)
├── models.py                    # Pydantic Action/Observation/State
├── server/
│   ├── app.py                   # FastAPI endpoints (reset/step/state/tasks/grader/baseline)
│   ├── cloud_sre_environment.py # Core MDP + adaptive sampling
│   ├── constants.py             # 21 scenarios + dynamic generator ← NEEDS 10+ NEW SCENARIOS
│   ├── graders.py               # 5 deterministic graders (0.0–1.0)
│   ├── command_executor.py      # Routes SRE commands to real infra
│   └── judge.py                 # LLM judge (optional)
├── services/                    # 16 REAL microservices
│   ├── base_service.py          # BaseService class (all services inherit)
│   ├── _service_worker.py       # Process spawner (handles all 16 services)
│   ├── orchestrator.py          # Process lifecycle (subprocess.Popen, os.kill)
│   ├── payment_service.py       # :8001 — SQLite + Queue
│   ├── auth_service.py          # :8002 — JWT auth
│   ├── worker_service.py        # :8003 — Queue consumer
│   ├── frontend_proxy.py        # :8004 — Reverse proxy
│   ├── cache_service.py         # :8005 — LRU cache
│   ├── notification_service.py  # :8006 — Webhook delivery
│   ├── search_service.py        # :8007 — Full-text search index (NEW)
│   ├── gateway_service.py       # :8008 — API gateway + rate limiter (NEW)
│   ├── scheduler_service.py     # :8009 — Job scheduler (NEW)
│   ├── storage_service.py       # :8010 — Object storage S3-like (NEW)
│   ├── metrics_collector_service.py  # :8011 — Prometheus-like metrics (NEW)
│   ├── email_service.py         # :8012 — SMTP email (NEW)
│   ├── billing_service.py       # :8013 — Usage tracking (NEW)
│   ├── config_service.py        # :8014 — Config management etcd-like (NEW)
│   ├── dns_service.py           # :8015 — Service discovery (NEW)
│   └── loadbalancer_service.py  # :8016 — L7 load balancer (NEW)
├── infra/
│   ├── database.py              # Real SQLite with fault injection
│   ├── queue.py                 # File-backed message queue
│   ├── metrics.py               # Prometheus-style metrics
│   └── logger.py                # Structured JSON logging
├── train_grpo.py                # GRPO training — 5-tier curriculum
├── train_reinforce.py           # REINFORCE baseline
├── sft_warmup.py                # SFT on expert demos
├── evaluate_model.py            # Post-training evaluation
├── Dockerfile                   # HF Spaces deployment
└── JUDGES_START_HERE.md         # Judge verification guide
```

---

## 5 Task Tiers

| Tier | Task ID | Max Steps | Description |
|------|---------|-----------|-------------|
| 1 | warmup | 10 | Single fault, clear signals |
| 2 | single_fault | 15 | + misleading red herrings |
| 3 | cascade | 20 | + cascading failure after fix |
| 4 | multi_cascade | 25 | + multiple concurrent cascades |
| 5 | adversarial | 30 | Unique every episode (dynamic) |

---

## 16 Services — Port Map

```python
SERVICE_PORTS = {
    "payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004,
    "cache": 8005, "notification": 8006, "search": 8007, "gateway": 8008,
    "scheduler": 8009, "storage": 8010, "metrics_collector": 8011,
    "email": 8012, "billing": 8013, "config": 8014, "dns": 8015,
    "loadbalancer": 8016,
}
```

---

## Fault Types Per Service

### Original 6 Services (already have scenarios in constants.py)
- **payment**: db_lock, payment_overload, payment_oom
- **auth**: auth_crash, jwt_expiry, token_blacklist_full
- **worker**: worker_oom, queue_overflow, worker_deadlock
- **frontend**: frontend_502, upstream_timeout, ssl_cert_expired
- **cache**: cache_invalidation, cache_ttl_expired, thundering_herd
- **notification**: webhook_failure, notification_queue_full

### NEW 10 Services (NEED scenarios added to constants.py)
- **search** (:8007): index_corruption, index_lag
- **gateway** (:8008): rate_limit_zero, circuit_breaker_stuck
- **scheduler** (:8009): scheduler_stuck, duplicate_execution
- **storage** (:8010): disk_full, data_corruption
- **metrics_collector** (:8011): scrape_failure, retention_full
- **email** (:8012): smtp_down, email_queue_overflow
- **billing** (:8013): billing_desync, invoice_stuck
- **config** (:8014): config_poisoned, config_locked
- **dns** (:8015): dns_resolution_failure, stale_entries
- **loadbalancer** (:8016): all_backends_removed, session_corruption

---

## Scenario Format in constants.py

Each scenario in constants.py follows this pattern:
```python
{
    "id": "unique_scenario_id",
    "tier": "warmup",  # or single_fault, cascade, multi_cascade, adversarial
    "description": "Brief description of the incident",
    "alert": "CRITICAL: The alert message the agent sees",
    "faults": [
        {"service": "payment", "type": "crash"},  # What to break
    ],
    "cascade": [],  # Optional: faults triggered AFTER fix
    "solution_commands": ["restart_service payment"],  # Expected fix
    "max_steps": 10,
}
```

---

## Cascade Engine

The cascade mechanic is our NOVEL CONTRIBUTION. When the agent fixes the primary fault, a SECONDARY fault triggers:

```
Phase 1: DB locked → payment 503 → queue fills → frontend 502
Phase 2: Agent fixes DB → 847 queued requests flood payment → OOM!
Phase 3: Agent must restart payment + drain queue at controlled rate
```

The new services enable MORE cascade chains:
- DNS down → gateway can't route → ALL services 503
- Config poisoned → rate limit set to 0 → gateway blocks everything
- Storage full → scheduler can't write job state → duplicate execution
- Metrics scrape failure → no alerting → silent failures compound

---

## Training Results (COMPLETED)

### Phase 1: SFT (Supervised Fine-Tuning) ✅
- Model: Qwen2.5-1.5B (4-bit via Unsloth)
- Data: 100 expert SRE demonstrations
- Loss: 1.75 → 0.15 in 3 epochs (2 min 17 sec)

### Phase 2: REINFORCE ✅ (previous run, now superseded by GRPO)
- Warmup: 86% resolution (43/50), avg reward +0.46
- Single_fault: 65% resolution (13/20), avg reward +0.41

### Phase 3: GRPO ✅ (just completed)
- Warmup: **100% resolution (15/15)**, avg reward +0.83
- Single_fault: **93% resolution (14/15)**, avg reward +0.89

### Phase 4: GRPO on cascade/multi_cascade/adversarial ← TODO (use $30 HF credits on L4)
### Phase 5: Evaluation ("final exam") ← TODO

---

## What STILL Needs to Be Done

### HIGH PRIORITY
1. **Add scenarios for 10 new services** in `server/constants.py`
   - At least 2 warmup scenarios per new service
   - At least 1 cascade scenario per new service
   - Need to integrate fault injection in orchestrator.py
2. **Push GRPO model + results to HuggingFace**
3. **Run GRPO on cascade/multi_cascade/adversarial tiers** (HF L4 GPU)
4. **Run evaluate_model.py** on the trained model

### MEDIUM PRIORITY
5. **HF Space Gradio UI** — visual landing page instead of raw API docs
6. **Update scenarios to use new services in cascade chains**
7. **Update JUDGES_START_HERE.md** with GRPO results

### LOW PRIORITY
8. **Colab training notebook cleanup** — make it presentable
9. **Blog post update** with GRPO results

---

## Competitive Landscape

### Our Main Threat: Privilege-Desk (Kirushikesh)
- 6 tasks, 27 tools, 4-component reward
- Excellent documentation
- BUT: All simulated (Python dicts, no real processes)

### Another Threat: Unnamed participant at hackathon
- Claims 40+ services with Kubernetes
- Using 70B model + Claude Pro + local Llama
- GRPO training working

### Our Advantages
1. REAL processes (subprocess.Popen, os.kill)
2. Cascading failures (novel contribution)
3. 16 services (each a real OS process)
4. 3-phase training with evidence (SFT → REINFORCE → GRPO)
5. Dense reward signal

---

## OpenEnv API Spec (must comply)

The environment MUST expose:
- `POST /reset` — Start new episode
- `POST /step` — Execute action, return obs+reward+done
- `GET /state` — Current state
- `GET /health` — Health check
- `GET /tasks` — List all tasks
- `POST /grader` — Grade an episode (0.0–1.0)
- `POST /baseline` — Run baseline agent

---

## Key Files to Modify

If adding scenarios: `server/constants.py`
If adding cascade chains: `server/cloud_sre_environment.py`
If adding fault injection for new services: `services/orchestrator.py`
If updating grading: `server/graders.py`
If updating the API: `server/app.py`
If changing model/observation: `models.py`

---

## GitHub Repo
https://github.com/Harikishanth/CloudSRE

## HF Space (Live Environment)
https://huggingface.co/spaces/DarDrax/CloudSRE-Environment

## HF Model
https://huggingface.co/DarDrax/cloudsre-reinforce-checkpoint
