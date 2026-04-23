"""
CloudSRE v2 — Constants and Scenario Configuration.

Defines all 5 task tiers with deep scenario pools.
Every fault type here works ACROSS PROCESSES via:
  - db_lock: SQLite EXCLUSIVE on shared file
  - queue_overflow: File-backed queue on shared directory
  - process_crash: subprocess.kill() sends real signal
  - db_pool_exhaustion: Exhausts DB connection pool
  - queue_pause: File-based pause flag
  - misleading_signal: Writes to shared log files

Scenario count: 4 warmup + 4 single_fault + 5 cascade + 4 multi_cascade
               + unlimited adversarial = 17 static + dynamic
"""

import random
from dataclasses import field
from cloud_sre_v2.models import ScenarioSpec, CascadeRule


def _adaptive_choice(scenarios: list, **kwargs) -> "ScenarioSpec":
    """Pick a scenario using adaptive weighted sampling (Theme #4).

    If a performance_tracker is provided, scenarios the agent fails at
    are sampled MORE OFTEN. Otherwise falls back to uniform random.
    """
    tracker = kwargs.get("performance_tracker")
    if tracker is None or not scenarios:
        return random.choice(scenarios)

    scenario_ids = [s.scenario_id for s in scenarios]
    weights = tracker.get_weights(scenario_ids)
    return random.choices(scenarios, weights=weights, k=1)[0]


# ── Scenario Pools ────────────────────────────────────────────────────────


def _warmup_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 1: Single clear failure, no red herrings, no cascade."""
    scenarios = [
        ScenarioSpec(
            scenario_id="warmup_db_lock",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 02:47 UTC | Severity: P2\n"
                "Payment service returning HTTP 503 to all requests.\n"
                "User reports: 'Cannot checkout', 'Payment keeps failing'"
            ),
            root_cause="Database is locked — all writes to SQLite are failing",
            correct_fix_description="Restart the payment service or release the DB lock",
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "cat /var/log/payment/error.log",
                "sqlite3 /data/app.db '.tables'",
                "systemctl restart payment",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_process_crash",
            failure_type="process_crash",
            target_service="worker",
            params={"target": "worker", "reason": "SIGSEGV"},
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 09:15 UTC | Severity: P2\n"
                "Worker service not responding. Queue depth increasing rapidly.\n"
                "User reports: 'Orders stuck in processing'"
            ),
            root_cause="Worker process crashed (SIGSEGV) — messages piling up in queue",
            correct_fix_description="Restart the worker service",
            expected_diagnostic_path=[
                "curl http://localhost:8003/healthz",
                "ps aux | grep worker",
                "systemctl restart worker",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_queue_overflow",
            failure_type="queue_overflow",
            target_service="worker",
            params={"fill": 800},
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 06:30 UTC | Severity: P2\n"
                "Worker queue depth critical — 800/1000 messages.\n"
                "Payment service unable to enqueue new orders.\n"
                "User reports: 'Checkout takes forever', 'Orders not going through'"
            ),
            root_cause="Message queue is near capacity — worker cannot keep up",
            correct_fix_description="Drain queue at controlled rate to clear backlog",
            expected_diagnostic_path=[
                "queue status",
                "ls /data/queue/ | wc -l",
                "queue drain 10",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_auth_crash",
            failure_type="process_crash",
            target_service="auth",
            params={"target": "auth", "reason": "OOM"},
            difficulty=0.20,
            alert_message=(
                "🚨 INCIDENT — 00:01 UTC | Severity: P1\n"
                "Auth service returning Connection refused on all requests.\n"
                "User reports: 'Can't log in', 'Auth errors on every page'"
            ),
            root_cause="Auth service process crashed (OOM) — needs restart",
            correct_fix_description="Restart the auth service",
            expected_diagnostic_path=[
                "curl http://localhost:8002/healthz",
                "ps aux | grep auth",
                "systemctl restart auth",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_cache_cold",
            failure_type="cache_invalidation",
            target_service="cache",
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 11:20 UTC | Severity: P2\n"
                "Cache hit ratio dropped from 94% to 0%.\n"
                "All requests falling through to database.\n"
                "User reports: 'Pages loading very slowly'"
            ),
            root_cause="Cache was invalidated — all entries expired simultaneously",
            correct_fix_description="Warm the cache back up or restart cache service",
            expected_diagnostic_path=[
                "curl http://localhost:8005/healthz",
                "curl http://localhost:8005/cache/stats",
                "cat /var/log/cache/error.log",
                "systemctl restart cache",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_notification_timeout",
            failure_type="process_crash",
            target_service="notification",
            params={"target": "notification", "reason": "OOM"},
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 08:45 UTC | Severity: P2\n"
                "Notification service not responding — webhooks not delivering.\n"
                "User reports: 'Not receiving order confirmation emails'"
            ),
            root_cause="Notification service crashed (OOM) — needs restart",
            correct_fix_description="Restart the notification service",
            expected_diagnostic_path=[
                "curl http://localhost:8006/healthz",
                "ps aux | grep notification",
                "systemctl restart notification",
            ],
            task_id="warmup",
        ),
    ]
    return _adaptive_choice(scenarios, **kwargs)


def _single_fault_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 2: Single fault with misleading signals (red herrings)."""
    scenarios = [
        ScenarioSpec(
            scenario_id="single_db_lock_redherring",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.35,
            alert_message=(
                "🚨 INCIDENT — 14:23 UTC | Severity: P1\n"
                "Multiple services showing errors:\n"
                "  - Payment: HTTP 503 on all POST requests\n"
                "  - Auth: Elevated latency (120ms vs 40ms baseline)\n"
                "  - Frontend: 502 Bad Gateway on /api/pay\n"
                "User reports: 'Nothing works', 'All pages broken'"
            ),
            root_cause="Database lock in payment service — auth latency is a red herring",
            correct_fix_description="Fix the database lock in payment; auth is fine",
            misleading_signals={
                "auth": "AuthService: JWT validation slow — 2400ms (normally 5ms)",
                "frontend": "FrontendProxy: upstream connection pool near capacity (48/50)",
            },
            expected_diagnostic_path=[
                "systemctl status",
                "curl http://localhost:8001/healthz",
                "curl http://localhost:8002/healthz",
                "cat /var/log/payment/error.log",
                "sqlite3 /data/app.db 'SELECT count(*) FROM payments WHERE status=\"pending\"'",
                "systemctl restart payment",
            ],
            task_id="single_fault",
        ),
        ScenarioSpec(
            scenario_id="single_worker_crash_redherring",
            failure_type="process_crash",
            target_service="worker",
            params={"target": "worker", "reason": "SIGKILL"},
            difficulty=0.40,
            alert_message=(
                "🚨 INCIDENT — 03:15 UTC | Severity: P1\n"
                "Worker service not responding — port 8003 Connection refused.\n"
                "Queue depth climbing rapidly.\n"
                "  - Payment: response time elevated (180ms vs 20ms baseline)\n"
                "  - Auth: 'token refresh rate 3x normal'\n"
                "User reports: 'Orders stuck in processing', 'Payments slow'"
            ),
            root_cause="Worker process killed (SIGKILL) — payment slowness is queue backpressure",
            correct_fix_description="Restart the worker; payment will recover automatically",
            misleading_signals={
                "payment": "PaymentService: response time elevated (180ms vs 20ms baseline)",
                "auth": "AuthService: token refresh rate 3x normal (450/min)",
            },
            expected_diagnostic_path=[
                "ps aux",
                "curl http://localhost:8003/healthz",
                "queue status",
                "systemctl restart worker",
            ],
            task_id="single_fault",
        ),
        ScenarioSpec(
            scenario_id="single_queue_overflow",
            failure_type="queue_overflow",
            target_service="worker",
            params={"fill": 950},
            difficulty=0.35,
            alert_message=(
                "🚨 INCIDENT — 11:05 UTC | Severity: P1\n"
                "Payment service returning 503 intermittently.\n"
                "Worker queue depth critical.\n"
                "  - Frontend: cache hit ratio dropped (94% → 71%)\n"
                "User reports: 'Payments failing randomly'"
            ),
            root_cause="Message queue overflow — worker can't keep up, payment can't publish",
            correct_fix_description="Drain queue at controlled rate, then investigate worker lag",
            misleading_signals={
                "frontend": "FrontendProxy: cache hit ratio dropped from 94% to 71%",
            },
            expected_diagnostic_path=[
                "queue status",
                "ls /data/queue/ | wc -l",
                "curl http://localhost:8003/healthz",
                "queue drain 10",
            ],
            task_id="single_fault",
        ),
        ScenarioSpec(
            scenario_id="single_payment_crash_db_redherring",
            failure_type="process_crash",
            target_service="payment",
            params={"target": "payment", "reason": "unhandled exception"},
            difficulty=0.45,
            alert_message=(
                "🚨 INCIDENT — 19:42 UTC | Severity: P1\n"
                "Payment service completely unreachable — port 8001 closed.\n"
                "Database showing elevated latency (50ms vs 5ms baseline).\n"
                "  - Worker: GC pause time elevated (400ms)\n"
                "User reports: 'Can't make any payments'"
            ),
            root_cause="Payment crashed — DB latency is from connection retry storms",
            correct_fix_description="Restart payment; DB latency resolves when retries stop",
            misleading_signals={
                "worker": "WorkerService: GC pause time elevated (400ms)",
            },
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "ps aux | grep payment",
                "cat /var/log/payment/error.log",
                "systemctl restart payment",
            ],
            task_id="single_fault",
        ),
    ]
    return _adaptive_choice(scenarios, **kwargs)


def _cascade_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 3: Single fault that triggers a cascade after fix."""
    scenarios = [
        ScenarioSpec(
            scenario_id="cascade_db_thundering_herd",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.55,
            alert_message=(
                "🚨 INCIDENT — 18:44 UTC | Severity: P0\n"
                "Payment service fully down — database connection errors.\n"
                "Queue depth increasing rapidly (847 messages backed up).\n"
                "User reports: 'Cannot make any payments for 30 minutes'"
            ),
            root_cause="Database lock causing payment failures and queue backup",
            correct_fix_description=(
                "1. Release DB lock\n"
                "2. IMPORTANT: Drain queue at CONTROLLED rate (not all at once)\n"
                "3. If you drain all at once, worker will OOM from thundering herd"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DB lock released while queue depth > 100",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="847 queued requests flood worker simultaneously → OOM",
                    agent_must="Use 'queue drain 10' instead of 'queue drain all'",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "cat /var/log/payment/error.log",
                "queue status",
                "systemctl restart payment",
                "queue drain 10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_worker_crash_queue_flood",
            failure_type="process_crash",
            target_service="worker",
            params={"target": "worker", "reason": "SIGSEGV"},
            difficulty=0.55,
            alert_message=(
                "🚨 INCIDENT — 22:05 UTC | Severity: P0\n"
                "Worker crashed — all queued messages accumulating.\n"
                "Queue depth: 750/1000 and climbing 20/sec.\n"
                "Payment service: 'queue publish timeout — backing off'\n"
                "User reports: 'Payments accepted but never processing'"
            ),
            root_cause="Worker segfault → queue fills → payment can't enqueue new work",
            correct_fix_description=(
                "1. Restart worker service (it crashed)\n"
                "2. After restart, drain queue at CONTROLLED rate\n"
                "3. If drain all at once, worker OOMs from burst load"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Worker restarted while queue depth > 200",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="Worker restarts → processes all 750 queued messages → OOM",
                    agent_must="Drain queue at controlled rate before or after restart",
                ),
            ],
            expected_diagnostic_path=[
                "ps aux | grep worker",
                "curl http://localhost:8003/healthz",
                "queue status",
                "systemctl restart worker",
                "queue drain 10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_db_lock_to_payment_crash",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 07:30 UTC | Severity: P0\n"
                "Payment service returning 503 — database errors.\n"
                "Service has been retrying DB connections for 15 minutes.\n"
                "Memory usage climbing due to retry queue buildup.\n"
                "User reports: 'All payments failing since 7:15am'"
            ),
            root_cause="DB lock → payment retries exhaust memory → payment crashes",
            correct_fix_description=(
                "1. Release DB lock first\n"
                "2. Then restart payment (it may have crashed from retries)\n"
                "3. Monitor payment memory after restart"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DB lock released → payment retry storm",
                    cascade_type="process_crash",
                    affected_service="payment",
                    description="Payment retries all failed writes → memory spike → OOM",
                    agent_must="Restart payment after releasing DB lock",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "cat /var/log/payment/error.log",
                "sqlite3 /data/app.db 'PRAGMA journal_mode'",
                "systemctl restart payment",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_queue_overflow_worker_crash",
            failure_type="queue_overflow",
            target_service="worker",
            params={"fill": 990},
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 13:00 UTC | Severity: P0\n"
                "Queue at 990/1000 — nearly full.\n"
                "Worker consuming slowly (10/min vs normal 200/min).\n"
                "New payments being dropped — queue rejecting publishes.\n"
                "User reports: 'Payments disappearing', 'No confirmation emails'"
            ),
            root_cause="Queue near-full → if drained too fast, worker crashes from burst",
            correct_fix_description=(
                "1. Check queue depth: ls /data/queue/ | wc -l\n"
                "2. Drain in batches of 10-20, not all at once\n"
                "3. If worker crashes, restart and drain more slowly"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Queue drained too fast (>50 at once)",
                    cascade_type="process_crash",
                    affected_service="worker",
                    description="Mass drain → 990 messages processed → worker OOM",
                    agent_must="Use 'queue drain 10' repeatedly, not 'queue drain all'",
                ),
            ],
            expected_diagnostic_path=[
                "queue status",
                "ls /data/queue/ | wc -l",
                "curl http://localhost:8003/healthz",
                "queue drain 10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_auth_crash_payment_timeout",
            failure_type="process_crash",
            target_service="auth",
            params={"target": "auth", "reason": "OOM"},
            difficulty=0.55,
            alert_message=(
                "🚨 INCIDENT — 16:20 UTC | Severity: P0\n"
                "Auth service crashed — port 8002 Connection refused.\n"
                "Payment service reporting auth timeout errors.\n"
                "All user sessions invalid — cannot verify any tokens.\n"
                "User reports: 'Logged out everywhere', 'Can't authenticate'"
            ),
            root_cause="Auth OOM → payment can't authenticate → queue fills with retries",
            correct_fix_description=(
                "1. Restart auth service\n"
                "2. After auth is back, payment retries may flood queue\n"
                "3. Monitor queue depth and drain if needed"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Auth restarted → payment retries flood queue",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="All deferred auth-dependent requests retry → queue overflow",
                    agent_must="Monitor queue after auth restart and drain conservatively",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://localhost:8002/healthz",
                "ps aux | grep auth",
                "systemctl restart auth",
                "queue status",
                "queue drain 10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_db_cache_stampede",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 14:55 UTC | Severity: P0\n"
                "Payment service returning 503 — database locked.\n"
                "Cache hit ratio dropping rapidly (94% → 12%).\n"
                "Cache entries expiring without refresh.\n"
                "User reports: 'Checkout keeps timing out'"
            ),
            root_cause="DB lock blocks cache refresh → cache expires → thundering herd on DB recovery",
            correct_fix_description=(
                "1. Release DB lock (root cause)\n"
                "2. IMPORTANT: Warm cache BEFORE traffic resumes\n"
                "3. If cache stays cold, thundering herd will crash DB again"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DB lock released while cache is cold",
                    cascade_type="cache_invalidation",
                    affected_service="cache",
                    description="All cached data expired during outage → 100% miss rate → DB thundering herd",
                    agent_must="Warm cache after fixing DB, before full traffic resumes",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "curl http://localhost:8005/cache/stats",
                "cat /var/log/payment/error.log",
                "systemctl restart payment",
                "curl http://localhost:8005/cache/warmup",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_payment_webhook_storm",
            failure_type="process_crash",
            target_service="payment",
            params={"target": "payment", "reason": "SIGSEGV"},
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 20:30 UTC | Severity: P0\n"
                "Payment service crashed — port 8001 closed.\n"
                "Notification service queuing webhooks (can't deliver).\n"
                "Webhook queue depth climbing: 250 and rising.\n"
                "User reports: 'Payments failing, no confirmation emails'"
            ),
            root_cause="Payment crash → webhooks queue up → when payment recovers, webhook storm overwhelms it",
            correct_fix_description=(
                "1. Restart payment service\n"
                "2. IMPORTANT: Drain webhook queue at controlled rate\n"
                "3. If all webhooks fire at once, payment will crash again"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Payment restarted → queued webhooks fire simultaneously",
                    cascade_type="webhook_storm",
                    affected_service="notification",
                    description="300 queued webhooks retry at once → payment overwhelmed → crashes again",
                    agent_must="Drain webhook queue at controlled rate after payment restart",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://localhost:8001/healthz",
                "curl http://localhost:8006/webhooks/pending",
                "ps aux | grep payment",
                "systemctl restart payment",
                "curl -X POST http://localhost:8006/webhooks/drain?count=10",
            ],
            task_id="cascade",
        ),
    ]
    return _adaptive_choice(scenarios, **kwargs)


def _multi_cascade_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 4: Multiple cascading failures — the hardest deterministic scenarios."""
    scenarios = [
        ScenarioSpec(
            scenario_id="multi_cascade_full_outage",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.75,
            alert_message=(
                "🚨 INCIDENT — 03:00 UTC | Severity: P0 — FULL OUTAGE\n"
                "All services degraded or failing:\n"
                "  - Payment: 503 on all requests (DB locked)\n"
                "  - Worker: Queue depth 940/1000, processing stalled\n"
                "  - Auth: Elevated error rate\n"
                "  - Frontend: 502 Bad Gateway on everything\n"
                "Duration: 45 minutes and escalating\n"
                "User reports: 'Site is completely down'"
            ),
            root_cause="Multi-fault: DB lock (primary) + queue overflow (secondary)",
            correct_fix_description=(
                "PRIORITY ORDER:\n"
                "1. Fix DB lock (primary cause of payment failure)\n"
                "2. Drain queue at controlled rate (prevent thundering herd)\n"
                "3. Verify all services healthy"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DB lock released",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="Queue floods worker after DB release",
                    agent_must="Drain queue at controlled rate",
                ),
            ],
            expected_diagnostic_path=[
                "systemctl status",
                "curl http://localhost:8001/healthz",
                "cat /var/log/payment/error.log",
                "systemctl restart payment",
                "queue drain 10",
            ],
            task_id="multi_cascade",
        ),
        ScenarioSpec(
            scenario_id="multi_cascade_payment_worker_down",
            failure_type="process_crash",
            target_service="payment",
            params={"target": "payment", "reason": "SIGSEGV"},
            difficulty=0.80,
            alert_message=(
                "🚨 INCIDENT — 04:15 UTC | Severity: P0 — FULL OUTAGE\n"
                "Payment AND Worker both unresponsive:\n"
                "  - Payment: Connection refused on port 8001\n"
                "  - Worker: Connection refused on port 8003\n"
                "  - Queue: 870 messages backed up\n"
                "  - Auth: Elevated latency (possible red herring)\n"
                "Duration: 20 minutes\n"
                "User reports: 'Nothing is working'"
            ),
            root_cause="Payment crashed → queue backed up → worker OOM from pressure",
            correct_fix_description=(
                "1. Restart payment first (root cause)\n"
                "2. Drain queue BEFORE restarting worker\n"
                "3. Restart worker\n"
                "4. Ignore auth latency — red herring"
            ),
            misleading_signals={
                "auth": "AuthService: JWT validation slow — 1800ms (normally 5ms)",
            },
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Payment restarted → queue floods worker",
                    cascade_type="process_crash",
                    affected_service="worker",
                    description="Restarted payment publishes burst → worker crashes",
                    agent_must="Drain queue before restarting worker",
                ),
            ],
            expected_diagnostic_path=[
                "ps aux",
                "systemctl restart payment",
                "queue drain 10",
                "systemctl restart worker",
            ],
            task_id="multi_cascade",
        ),
        ScenarioSpec(
            scenario_id="multi_cascade_db_queue_auth",
            failure_type="db_lock",
            target_service="payment",
            difficulty=0.85,
            alert_message=(
                "🚨 INCIDENT — 02:30 UTC | Severity: P0 — TRIPLE FAULT\n"
                "Three simultaneous failures detected:\n"
                "  - Payment: 503 — 'database is locked'\n"
                "  - Worker: Queue at 960/1000, not processing\n"
                "  - Auth: Process crashed — port 8002 Connection refused\n"
                "User reports: 'Complete site outage for 1 hour'"
            ),
            root_cause="DB lock (root) → queue overflow → auth crashed from load",
            correct_fix_description=(
                "CRITICAL ORDER:\n"
                "1. Release DB lock (unblocks payment)\n"
                "2. Drain queue slowly (prevents worker OOM)\n"
                "3. Restart auth (was collateral damage)\n"
                "4. Verify all 4 services via systemctl status"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DB released → queue flood → auth overload",
                    cascade_type="process_crash",
                    affected_service="auth",
                    description="Cascading load reaches auth → auth crashes",
                    agent_must="Fix in order: DB → queue → auth",
                ),
            ],
            expected_diagnostic_path=[
                "systemctl status",
                "cat /var/log/payment/error.log",
                "queue status",
                "ps aux",
                "systemctl restart payment",
                "queue drain 10",
                "systemctl restart auth",
                "systemctl status",
            ],
            task_id="multi_cascade",
        ),
        ScenarioSpec(
            scenario_id="multi_cascade_rolling_crash",
            failure_type="process_crash",
            target_service="worker",
            params={"target": "worker", "reason": "SIGKILL"},
            difficulty=0.90,
            alert_message=(
                "🚨 INCIDENT — 01:00 UTC | Severity: P0 — CASCADING OUTAGE\n"
                "Worker killed by OOM killer. Queue overflowing.\n"
                "  - Worker: DEAD (SIGKILL by kernel OOM)\n"
                "  - Queue: 995/1000 — dropping new messages\n"
                "  - Payment: 'queue publish failed — backing off'\n"
                "  - Frontend: 504 Gateway Timeout\n"
                "This is a ROLLING FAILURE — fixing one may crash another.\n"
                "User reports: 'Has been broken for 2 hours'"
            ),
            root_cause=(
                "Worker OOM → queue fills → if worker restarted without "
                "draining, it OOMs again from burst"
            ),
            correct_fix_description=(
                "CAREFUL — this will cascade if you rush:\n"
                "1. Do NOT restart worker yet — queue will kill it again\n"
                "2. Drain queue to <100 messages (queue drain 10, repeat)\n"
                "3. THEN restart worker\n"
                "4. Monitor: if queue climbs again, worker is overloaded"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Worker restarted while queue > 500",
                    cascade_type="process_crash",
                    affected_service="worker",
                    description="Worker restarts → processes 995 msgs → OOM again",
                    agent_must="Drain queue BEFORE restarting worker",
                ),
            ],
            expected_diagnostic_path=[
                "ps aux | grep worker",
                "queue status",
                "ls /data/queue/ | wc -l",
                "queue drain 10",
                "queue drain 10",
                "queue drain 10",
                "systemctl restart worker",
            ],
            task_id="multi_cascade",
        ),
    ]
    chosen = _adaptive_choice(scenarios, **kwargs)

    # Inject additional faults for multi-cascade complexity
    if "full_outage" in chosen.scenario_id:
        orchestrator.inject_fault("queue_overflow", {"fill": 900})
    if "payment_worker" in chosen.scenario_id:
        orchestrator.inject_fault("process_crash", {"target": "worker", "reason": "OOM"})
    if "db_queue_auth" in chosen.scenario_id:
        orchestrator.inject_fault("queue_overflow", {"fill": 960})
        orchestrator.inject_fault("process_crash", {"target": "auth", "reason": "OOM"})
    if "rolling_crash" in chosen.scenario_id:
        orchestrator.inject_fault("queue_overflow", {"fill": 995})

    return chosen


def _adversarial_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 5: Dynamically generated adversarial scenario.

    Combinatorially mixes fault types, targets, red herrings, and cascade
    chains to produce hundreds of unique scenarios. Fully offline.
    Note: Adaptive sampling doesn't apply here — each scenario is
    dynamically generated and unique.
    """
    return _generate_dynamic_scenario(orchestrator, difficulty_range=(0.6, 0.9))


# ── Dynamic Scenario Generator ───────────────────────────────────────────

# Fault types that work ACROSS PROCESSES (no in-process methods needed)
_FAULT_CATALOG = {
    "db_lock": {
        "services": ["payment"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Database is locked — all writes to SQLite are failing",
        "fix_tpl": "Release the database lock or restart the payment service",
    },
    "db_pool_exhaustion": {
        "services": ["payment"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Database connection pool exhausted — no connections available",
        "fix_tpl": "Restart payment service to reset the connection pool",
    },
    "queue_overflow": {
        "services": ["worker"],
        "gen_params": lambda: {"fill": random.randint(700, 980)},
        "root_cause_tpl": "Message queue overflowed — worker cannot keep up",
        "fix_tpl": "Drain queue at controlled rate (queue drain 10)",
    },
    "queue_pause": {
        "services": ["worker"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Queue consumer paused — messages accumulating",
        "fix_tpl": "Resume queue consumer and check worker health",
    },
    "process_crash": {
        "services": ["payment", "auth", "worker", "cache", "notification"],
        "gen_params": lambda: {
            "target": random.choice(["payment", "auth", "worker", "cache", "notification"]),
            "reason": random.choice([
                "SIGSEGV", "SIGKILL", "OOM", "SIGABRT",
                "unhandled exception", "stack overflow",
            ]),
        },
        "root_cause_tpl": "Service process crashed ({reason}) — needs restart",
        "fix_tpl": "Restart the crashed service via systemctl restart",
    },
    "cache_invalidation": {
        "services": ["cache"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Cache invalidated — all entries expired, 0% hit ratio",
        "fix_tpl": "Warm the cache or restart cache service",
    },
    "webhook_storm": {
        "services": ["notification"],
        "gen_params": lambda: {"count": random.randint(100, 400)},
        "root_cause_tpl": "Webhook retry storm — {count} webhooks firing simultaneously",
        "fix_tpl": "Drain webhook queue at controlled rate, pause delivery if needed",
    },
}

# Red herring generators — 10 templates for diverse misdirection
_RED_HERRING_TEMPLATES = [
    {"service": "auth", "message": "AuthService: JWT validation slow — {latency}ms (normally 5ms)"},
    {"service": "frontend", "message": "FrontendProxy: upstream pool near capacity ({used}/50)"},
    {"service": "worker", "message": "WorkerService: CPU spike to {cpu}% — batch backlog"},
    {"service": "payment", "message": "PaymentService: response time {latency}ms vs 20ms baseline"},
    {"service": "cache", "message": "CacheService: hit ratio dropped to {ratio}% (normally 94%)"},
    {"service": "notification", "message": "NotificationService: webhook delivery delayed {latency}ms"},
    {"service": "frontend", "message": "FrontendProxy: cache hit ratio dropped to {ratio}%"},
    {"service": "auth", "message": "AuthService: token refresh rate 3x normal ({rate}/min)"},
    {"service": "worker", "message": "WorkerService: GC pause elevated ({gc_ms}ms)"},
    {"service": "payment", "message": "PaymentService: retry rate {retries}/min"},
    {"service": "frontend", "message": "FrontendProxy: TLS handshake failures ({tls_errors}/min)"},
    {"service": "worker", "message": "WorkerService: disk I/O latency {io_ms}ms (normally 2ms)"},
]

# 7 cascade chain templates from real-world P0 incident patterns
_CASCADE_TEMPLATES = [
    {
        "trigger_fault": "db_lock",
        "cascade_type": "queue_overflow",
        "affected": "worker",
        "desc": "Queued requests flood worker after DB release → OOM",
        "agent_must": "Use 'queue drain 10' instead of 'queue drain all'",
    },
    {
        "trigger_fault": "process_crash",
        "cascade_type": "queue_overflow",
        "affected": "worker",
        "desc": "Crashed service recovery causes burst of queued requests",
        "agent_must": "Drain queue at controlled rate after restart",
    },
    {
        "trigger_fault": "queue_overflow",
        "cascade_type": "process_crash",
        "affected": "worker",
        "desc": "Mass drain causes worker memory spike → OOM crash",
        "agent_must": "Use controlled drain rate, not drain all",
    },
    {
        "trigger_fault": "db_lock",
        "cascade_type": "process_crash",
        "affected": "payment",
        "desc": "DB lock release → payment retry storm → payment OOM",
        "agent_must": "Monitor payment after DB lock release",
    },
    {
        "trigger_fault": "process_crash",
        "cascade_type": "db_lock",
        "affected": "payment",
        "desc": "Service restart hammers DB with connections → DB locks",
        "agent_must": "Check DB status after restarting crashed service",
    },
    {
        "trigger_fault": "queue_pause",
        "cascade_type": "queue_overflow",
        "affected": "worker",
        "desc": "Unpausing consumer with deep queue → immediate overflow",
        "agent_must": "Drain some messages before unpausing consumer",
    },
    {
        "trigger_fault": "db_pool_exhaustion",
        "cascade_type": "process_crash",
        "affected": "payment",
        "desc": "DB pool reset causes connection storm → payment crashes",
        "agent_must": "Restart payment gracefully after pool reset",
    },
]

_USER_REPORTS = [
    "Cannot checkout", "Payment keeps failing", "Site is completely down",
    "Can't log in", "Orders stuck in processing", "Nothing works",
    "Getting 502 errors", "Auth errors on every page", "Payments failing randomly",
    "All pages broken", "Logged out everywhere", "Extremely slow",
    "Payments disappearing", "No confirmation emails", "Cart won't save",
    "Stuck on loading screen", "Getting timeout errors", "Been broken for hours",
]

_TIMES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
_SEVERITIES = ["P0", "P1", "P2"]
_DURATIONS = [5, 10, 15, 20, 30, 45, 60, 90]


def _generate_dynamic_scenario(
    orchestrator,
    difficulty_range: tuple = (0.3, 0.9),
    force_cascade: bool = False,
) -> ScenarioSpec:
    """Generate a unique scenario by combining random elements.

    Produces 5 fault types x 3 services x N_params x 10 herrings x 7 cascades
    = hundreds of unique scenarios. The model CANNOT memorize solutions.
    """
    fault_type = random.choice(list(_FAULT_CATALOG.keys()))
    catalog = _FAULT_CATALOG[fault_type]
    target = random.choice(catalog["services"])
    params = catalog["gen_params"]()
    params.setdefault("target", target)

    format_args = {
        k: v for k, v in params.items()
        if isinstance(v, (str, int, float)) and k != "target"
    }
    root_cause = catalog["root_cause_tpl"].format(target=target, **format_args)

    difficulty = round(random.uniform(*difficulty_range), 2)
    num_herrings = random.randint(0, 1) if difficulty < 0.5 else random.randint(1, 3)
    available = [h for h in _RED_HERRING_TEMPLATES if h["service"] != target]
    chosen_herrings = random.sample(available, min(num_herrings, len(available)))

    misleading_signals = {}
    for h in chosen_herrings:
        msg = h["message"].format(
            latency=random.randint(80, 2500),
            used=random.randint(40, 49),
            cpu=random.randint(75, 95),
            ratio=random.randint(50, 75),
            rate=random.randint(100, 500),
            gc_ms=random.randint(200, 800),
            retries=random.randint(30, 150),
            tls_errors=random.randint(10, 50),
            io_ms=random.randint(50, 500),
        )
        misleading_signals[h["service"]] = msg

    cascade_rules = []
    applicable = [c for c in _CASCADE_TEMPLATES if c["trigger_fault"] == fault_type]
    if (force_cascade or difficulty > 0.5) and applicable:
        cd = random.choice(applicable)
        cascade_rules.append(CascadeRule(
            trigger_condition=f"{fault_type} fixed → {cd['cascade_type']}",
            cascade_type=cd["cascade_type"],
            affected_service=cd["affected"],
            description=cd["desc"],
            agent_must=cd["agent_must"],
        ))

    primary_symptom = f"  - {target.capitalize()}: "
    symptom_map = {
        "db_lock": "HTTP 503 on all requests (database errors)",
        "queue_overflow": f"Queue depth critical ({params.get('fill', 900)}/1000)",
        "process_crash": f"Process crashed ({params.get('reason', 'SIGSEGV')})",
        "db_pool_exhaustion": "DB connection pool exhausted — all queries failing",
        "queue_pause": "Queue consumer paused — messages accumulating",
    }
    primary_symptom += symptom_map.get(fault_type, "Service degraded")

    secondary = [f"  - {s.capitalize()}: {m}" for s, m in misleading_signals.items()]
    secondary_str = "\n".join(secondary) if secondary else "  (no other issues)"

    sev = "P0" if difficulty > 0.6 else random.choice(["P1", "P2"])
    alert = (
        f"🚨 INCIDENT — {random.choice(_TIMES)} UTC | Severity: {sev}\n"
        f"Multiple services showing errors:\n"
        f"{primary_symptom}\n"
        f"{secondary_str}\n"
        f"Duration: {random.choice(_DURATIONS)} minutes and escalating\n"
        f"User reports: '{random.choice(_USER_REPORTS)}'"
    )

    scenario_id = f"dynamic_{fault_type}_{target}_{random.randint(1000, 9999)}"
    port_map = {"payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004, "cache": 8005, "notification": 8006}

    return ScenarioSpec(
        scenario_id=scenario_id,
        failure_type=fault_type,
        target_service=target,
        params=params,
        difficulty=difficulty,
        alert_message=alert,
        root_cause=root_cause,
        correct_fix_description=catalog["fix_tpl"],
        misleading_signals=misleading_signals,
        cascade_rules=cascade_rules,
        expected_diagnostic_path=[
            "systemctl status",
            f"curl http://localhost:{port_map.get(target, 8001)}/healthz",
            f"cat /var/log/{target}/error.log",
        ],
        task_id="adversarial",
    )


# ── Task Configuration ────────────────────────────────────────────────────

TASK_CONFIGS = {
    "warmup": {
        "tier": 1,
        "max_steps": 10,
        "scenario_picker": _warmup_scenarios,
        "description": "Single clear failure, no red herrings",
    },
    "single_fault": {
        "tier": 2,
        "max_steps": 15,
        "scenario_picker": _single_fault_scenarios,
        "description": "Single fault with misleading signals",
    },
    "cascade": {
        "tier": 3,
        "max_steps": 20,
        "scenario_picker": _cascade_scenarios,
        "description": "Single fault that triggers cascade after fix",
    },
    "multi_cascade": {
        "tier": 4,
        "max_steps": 25,
        "scenario_picker": _multi_cascade_scenarios,
        "description": "Multiple cascading failures",
    },
    "adversarial": {
        "tier": 5,
        "max_steps": 30,
        "scenario_picker": _adversarial_scenarios,
        "description": "Dynamic adversarial — unique every episode",
    },
}
