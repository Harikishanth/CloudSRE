"""
CloudSRE v2 — Constants and Scenario Configuration.

Defines all 5 task tiers with deep scenario pools.
Every fault type here works ACROSS PROCESSES via:
  - db_lock: RDS/Aurora EXCLUSIVE on shared file
  - queue_overflow: File-backed queue on shared directory
  - process_crash: subprocess.kill() sends real signal
  - db_pool_exhaustion: Exhausts DB connection pool
  - queue_pause: File-based pause flag
  - misleading_signal: Writes to shared log files

Scenario count: 26 warmup + 4 single_fault + 17 cascade + 4 multi_cascade
               + unlimited adversarial = 51 static + dynamic
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
            root_cause="RDS connection pool exhausted -- all writes blocked — all writes to RDS/Aurora are failing",
            correct_fix_description="Restart the payment service or release the RDS lock",
            expected_diagnostic_path=[
                "curl http://payment.us-east-1.internal/healthz",
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
                "Worker service not responding. SQS depth increasing rapidly.\n"
                "User reports: 'Orders stuck in processing'"
            ),
            root_cause="ECS task terminated (OOMKilled) (SIGSEGV) — messages piling up in queue",
            correct_fix_description="Restart the worker service",
            expected_diagnostic_path=[
                "curl http://worker.eu-west-1.internal/healthz",
                "ps aux | grep worker",
                "systemctl restart worker",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_queue_overflow",
            failure_type="queue_overflow",
            target_service="worker",
            params={"fill": 550},
            difficulty=0.15,
            alert_message=(
                "🚨 INCIDENT — 06:30 UTC | Severity: P2\n"
                "Worker SQS depth critical — 550/1000 messages.\n"
                "Payment service unable to enqueue new orders.\n"
                "User reports: 'Checkout takes forever', 'Orders not going through'"
            ),
            root_cause="SQS dead letter queue approaching limit — worker cannot keep up",
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
                "Auth service returning ConnectionRefused (ECS task not running) on all requests.\n"
                "User reports: 'Can't log in', 'Auth errors on every page'"
            ),
            root_cause="EC2 instance terminated (OOMKilled) in us-east-1 (OOM) — needs restart",
            correct_fix_description="Restart the auth service",
            expected_diagnostic_path=[
                "curl http://auth.us-east-1.internal/healthz",
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
                "ElastiCache hit ratio dropped from 94% to 0%.\n"
                "All requests falling through to database.\n"
                "User reports: 'Pages loading very slowly'"
            ),
            root_cause="ElastiCache flush -- all entries expired simultaneously — all entries expired simultaneously",
            correct_fix_description="Warm the cache back up or restart cache service",
            expected_diagnostic_path=[
                "curl http://cache.ap-south-1.internal/healthz",
                "curl http://cache.ap-south-1.internal/cache/stats",
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
            root_cause="ECS task evicted (OOMKilled) in ap-south-1 (OOM) — needs restart",
            correct_fix_description="Restart the notification service",
            expected_diagnostic_path=[
                "curl http://notification.ap-south-1.internal/healthz",
                "ps aux | grep notification",
                "systemctl restart notification",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_search_index_corruption",
            failure_type="index_corruption",
            target_service="search",
            difficulty=0.20,
            alert_message=(
                "🚨 INCIDENT — 10:12 UTC | Severity: P2\n"
                "Search service returning empty results for known queries.\n"
                "User reports: 'Product search shows nothing'"
            ),
            root_cause="Search index corruption causing query mismatches",
            correct_fix_description="Restart search service to restore clean index state",
            expected_diagnostic_path=[
                "curl http://search.eu-west-1.internal/healthz",
                "curl http://search.eu-west-1.internal/search/stats",
                "cat /var/log/search/error.log",
                "systemctl restart search",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_search_index_lag",
            failure_type="index_lag",
            target_service="search",
            difficulty=0.22,
            alert_message=(
                "🚨 INCIDENT — 11:40 UTC | Severity: P2\n"
                "Search indexing delay detected — results are stale.\n"
                "User reports: 'New products not searchable yet'"
            ),
            root_cause="Search index lag is delaying document visibility",
            correct_fix_description="Restart search service to clear indexing lag state",
            expected_diagnostic_path=[
                "curl http://search.eu-west-1.internal/healthz",
                "curl http://search.eu-west-1.internal/search/stats",
                "grep \"lag\" /var/log/search/error.log",
                "systemctl restart search",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_gateway_rate_limit_zero",
            failure_type="rate_limit_zero",
            target_service="gateway",
            difficulty=0.20,
            alert_message=(
                "🚨 INCIDENT — 05:25 UTC | Severity: P2\n"
                "Gateway rejecting nearly all API traffic.\n"
                "User reports: 'Every API call gets rejected'"
            ),
            root_cause="Gateway rate limit was set to 0 RPS",
            correct_fix_description="Restart gateway service to restore normal request flow",
            expected_diagnostic_path=[
                "curl http://gateway.us-east-1.internal/healthz",
                "curl http://gateway.us-east-1.internal/gateway/stats",
                "cat /var/log/gateway/error.log",
                "systemctl restart gateway",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_gateway_circuit_breaker_stuck",
            failure_type="circuit_breaker_stuck",
            target_service="gateway",
            difficulty=0.24,
            alert_message=(
                "🚨 INCIDENT — 15:05 UTC | Severity: P2\n"
                "Gateway circuit breaker appears stuck open.\n"
                "User reports: 'Requests fail immediately'"
            ),
            root_cause="Gateway circuit breaker is stuck in open state",
            correct_fix_description="Restart gateway service to reset the circuit breaker",
            expected_diagnostic_path=[
                "curl http://gateway.us-east-1.internal/healthz",
                "curl http://gateway.us-east-1.internal/gateway/stats",
                "grep \"circuit\" /var/log/gateway/error.log",
                "systemctl restart gateway",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_scheduler_stuck",
            failure_type="scheduler_stuck",
            target_service="scheduler",
            difficulty=0.20,
            alert_message=(
                "🚨 INCIDENT — 06:10 UTC | Severity: P2\n"
                "Scheduled jobs have stopped executing.\n"
                "User reports: 'Nightly jobs didn't run'"
            ),
            root_cause="Scheduler execution loop is frozen",
            correct_fix_description="Restart scheduler service to resume job execution",
            expected_diagnostic_path=[
                "curl http://scheduler.eu-west-1.internal/healthz",
                "curl http://scheduler.eu-west-1.internal/scheduler/stats",
                "cat /var/log/scheduler/error.log",
                "systemctl restart scheduler",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_scheduler_duplicate_execution",
            failure_type="duplicate_execution",
            target_service="scheduler",
            difficulty=0.23,
            alert_message=(
                "🚨 INCIDENT — 12:18 UTC | Severity: P2\n"
                "Scheduler reports duplicate job runs.\n"
                "User reports: 'Invoices generated twice'"
            ),
            root_cause="Scheduler duplicate execution mode is enabled",
            correct_fix_description="Restart scheduler service to clear duplicate execution state",
            expected_diagnostic_path=[
                "curl http://scheduler.eu-west-1.internal/healthz",
                "curl http://scheduler.eu-west-1.internal/scheduler/jobs",
                "grep \"duplicate\" /var/log/scheduler/error.log",
                "systemctl restart scheduler",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_storage_disk_full",
            failure_type="disk_full",
            target_service="storage",
            difficulty=0.22,
            alert_message=(
                "🚨 INCIDENT — 02:42 UTC | Severity: P2\n"
                "Storage rejects writes with disk-full errors.\n"
                "User reports: 'File uploads failing'"
            ),
            root_cause="Storage service reached full capacity",
            correct_fix_description="Restart storage service to recover write path",
            expected_diagnostic_path=[
                "curl http://storage.eu-west-1.internal/healthz",
                "curl http://storage.eu-west-1.internal/storage/stats",
                "cat /var/log/storage/error.log",
                "systemctl restart storage",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_storage_data_corruption",
            failure_type="data_corruption",
            target_service="storage",
            difficulty=0.25,
            alert_message=(
                "🚨 INCIDENT — 13:36 UTC | Severity: P2\n"
                "Storage checksum mismatches detected.\n"
                "User reports: 'Downloaded files are corrupted'"
            ),
            root_cause="Storage data corruption causing checksum failures",
            correct_fix_description="Restart storage service to restore clean storage state",
            expected_diagnostic_path=[
                "curl http://storage.eu-west-1.internal/healthz",
                "curl http://storage.eu-west-1.internal/storage/stats",
                "grep \"corrupt\" /var/log/storage/error.log",
                "systemctl restart storage",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_metrics_scrape_failure",
            failure_type="scrape_failure",
            target_service="metrics_collector",
            difficulty=0.20,
            alert_message=(
                "🚨 INCIDENT — 04:20 UTC | Severity: P2\n"
                "Metrics collector failing to scrape all targets.\n"
                "User reports: 'Dashboards look frozen'"
            ),
            root_cause="Metrics scrape pipeline failed across all targets",
            correct_fix_description="Restart metrics_collector service to restore scraping",
            expected_diagnostic_path=[
                "curl http://metrics_collector.eu-west-1.internal/healthz",
                "curl http://metrics_collector.eu-west-1.internal/metrics/stats",
                "cat /var/log/metrics_collector/error.log",
                "systemctl restart metrics_collector",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_metrics_retention_full",
            failure_type="retention_full",
            target_service="metrics_collector",
            difficulty=0.23,
            alert_message=(
                "🚨 INCIDENT — 19:05 UTC | Severity: P2\n"
                "Metrics retention storage is full.\n"
                "User reports: 'New metrics are missing'"
            ),
            root_cause="Metrics retention is full and dropping data",
            correct_fix_description="Restart metrics_collector service to clear retention pressure",
            expected_diagnostic_path=[
                "curl http://metrics_collector.eu-west-1.internal/healthz",
                "curl http://metrics_collector.eu-west-1.internal/metrics/stats",
                "grep \"retention\" /var/log/metrics_collector/error.log",
                "systemctl restart metrics_collector",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_email_smtp_down",
            failure_type="smtp_down",
            target_service="email",
            difficulty=0.21,
            alert_message=(
                "🚨 INCIDENT — 21:14 UTC | Severity: P2\n"
                "Email delivery failing with SMTP errors.\n"
                "User reports: 'No alert emails received'"
            ),
            root_cause="SMTP upstream is unavailable",
            correct_fix_description="Restart email service to re-establish SMTP flow",
            expected_diagnostic_path=[
                "curl http://email.ap-south-1.internal/healthz",
                "curl http://email.ap-south-1.internal/email/stats",
                "cat /var/log/email/error.log",
                "systemctl restart email",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_email_queue_overflow",
            failure_type="email_queue_overflow",
            target_service="email",
            difficulty=0.24,
            alert_message=(
                "🚨 INCIDENT — 17:50 UTC | Severity: P2\n"
                "Email queue size exceeded safe limits.\n"
                "User reports: 'Notifications are delayed or missing'"
            ),
            root_cause="Email queue overflow causing dropped messages",
            correct_fix_description="Restart email service to reset queue pressure",
            expected_diagnostic_path=[
                "curl http://email.ap-south-1.internal/healthz",
                "curl http://email.ap-south-1.internal/email/stats",
                "grep \"overflow\" /var/log/email/error.log",
                "systemctl restart email",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_billing_desync",
            failure_type="billing_desync",
            target_service="billing",
            difficulty=0.21,
            alert_message=(
                "🚨 INCIDENT — 08:05 UTC | Severity: P2\n"
                "Billing totals don't match charge events.\n"
                "User reports: 'Revenue dashboard looks wrong'"
            ),
            root_cause="Billing ledger desynchronization",
            correct_fix_description="Restart billing service to recover accounting consistency",
            expected_diagnostic_path=[
                "curl http://billing.us-east-1.internal/healthz",
                "curl http://billing.us-east-1.internal/billing/stats",
                "cat /var/log/billing/error.log",
                "systemctl restart billing",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_billing_invoice_stuck",
            failure_type="invoice_stuck",
            target_service="billing",
            difficulty=0.24,
            alert_message=(
                "🚨 INCIDENT — 22:18 UTC | Severity: P2\n"
                "Invoice generation pipeline is stalled.\n"
                "User reports: 'Invoices did not generate'"
            ),
            root_cause="Invoice generator stuck in billing service",
            correct_fix_description="Restart billing service to resume invoice generation",
            expected_diagnostic_path=[
                "curl http://billing.us-east-1.internal/healthz",
                "curl http://billing.us-east-1.internal/billing/stats",
                "grep \"invoice\" /var/log/billing/error.log",
                "systemctl restart billing",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_config_poisoned",
            failure_type="config_poisoned",
            target_service="config",
            difficulty=0.22,
            alert_message=(
                "🚨 INCIDENT — 09:27 UTC | Severity: P2\n"
                "Services loading invalid config values.\n"
                "User reports: 'Traffic suddenly blocked'"
            ),
            root_cause="Poisoned configuration values in config service",
            correct_fix_description="Restart config service to reload safe config state",
            expected_diagnostic_path=[
                "curl http://config.us-east-1.internal/healthz",
                "curl http://config.us-east-1.internal/config/list",
                "cat /var/log/config/error.log",
                "systemctl restart config",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_config_locked",
            failure_type="config_locked",
            target_service="config",
            difficulty=0.25,
            alert_message=(
                "🚨 INCIDENT — 18:34 UTC | Severity: P2\n"
                "Config reads and updates are blocked.\n"
                "User reports: 'Config updates never apply'"
            ),
            root_cause="Config store is locked",
            correct_fix_description="Restart config service to clear lock state",
            expected_diagnostic_path=[
                "curl http://config.us-east-1.internal/healthz",
                "curl http://config.us-east-1.internal/config/list",
                "grep \"locked\" /var/log/config/error.log",
                "systemctl restart config",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_dns_resolution_failure",
            failure_type="dns_resolution_failure",
            target_service="dns",
            difficulty=0.22,
            alert_message=(
                "🚨 INCIDENT — 07:44 UTC | Severity: P2\n"
                "Internal service discovery failing.\n"
                "User reports: 'Services cannot reach each other'"
            ),
            root_cause="DNS resolution failure returning NXDOMAIN",
            correct_fix_description="Restart dns service to restore resolution",
            expected_diagnostic_path=[
                "curl http://dns.ap-south-1.internal/healthz",
                "curl http://dns.ap-south-1.internal/dns/registry",
                "cat /var/log/dns/error.log",
                "systemctl restart dns",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_dns_stale_entries",
            failure_type="stale_entries",
            target_service="dns",
            difficulty=0.24,
            alert_message=(
                "🚨 INCIDENT — 01:33 UTC | Severity: P2\n"
                "DNS cache serving stale endpoints.\n"
                "User reports: 'Intermittent routing failures'"
            ),
            root_cause="DNS stale cache entries routing to dead targets",
            correct_fix_description="Restart dns service to refresh service registry",
            expected_diagnostic_path=[
                "curl http://dns.ap-south-1.internal/healthz",
                "curl http://dns.ap-south-1.internal/dns/registry",
                "grep \"stale\" /var/log/dns/error.log",
                "systemctl restart dns",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_loadbalancer_all_backends_removed",
            failure_type="all_backends_removed",
            target_service="loadbalancer",
            difficulty=0.22,
            alert_message=(
                "🚨 INCIDENT — 03:49 UTC | Severity: P2\n"
                "Load balancer has no healthy backends.\n"
                "User reports: 'Every request returns 503'"
            ),
            root_cause="All backends removed from load balancer pool",
            correct_fix_description="Restart loadbalancer service to restore backend pool",
            expected_diagnostic_path=[
                "curl http://loadbalancer.us-east-1.internal/healthz",
                "curl http://loadbalancer.us-east-1.internal/lb/stats",
                "cat /var/log/loadbalancer/error.log",
                "systemctl restart loadbalancer",
            ],
            task_id="warmup",
        ),
        ScenarioSpec(
            scenario_id="warmup_loadbalancer_session_corruption",
            failure_type="session_corruption",
            target_service="loadbalancer",
            difficulty=0.25,
            alert_message=(
                "🚨 INCIDENT — 23:02 UTC | Severity: P2\n"
                "Session affinity behavior is inconsistent.\n"
                "User reports: 'Users keep getting logged out'"
            ),
            root_cause="Sticky session data is corrupted in load balancer",
            correct_fix_description="Restart loadbalancer service to reset session routing state",
            expected_diagnostic_path=[
                "curl http://loadbalancer.us-east-1.internal/healthz",
                "curl http://loadbalancer.us-east-1.internal/lb/stats",
                "grep \"session\" /var/log/loadbalancer/error.log",
                "systemctl restart loadbalancer",
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
            correct_fix_description="Fix the RDS connection pool lock in payment; auth is fine",
            misleading_signals={
                "auth": "AuthService: JWT validation slow — 2400ms (normally 5ms)",
                "frontend": "FrontendProxy: upstream connection pool near capacity (48/50)",
            },
            expected_diagnostic_path=[
                "systemctl status",
                "curl http://payment.us-east-1.internal/healthz",
                "curl http://auth.us-east-1.internal/healthz",
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
                "Worker service not responding — port 8003 ConnectionRefused (ECS task not running).\n"
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
                "curl http://worker.eu-west-1.internal/healthz",
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
                "Worker SQS depth critical.\n"
                "  - Frontend: cache hit ratio dropped (94% → 71%)\n"
                "User reports: 'Payments failing randomly'"
            ),
            root_cause="SQS dead letter queue overflow — worker can't keep up, payment can't publish",
            correct_fix_description="Drain queue at controlled rate, then investigate worker lag",
            misleading_signals={
                "frontend": "FrontendProxy: cache hit ratio dropped from 94% to 71%",
            },
            expected_diagnostic_path=[
                "queue status",
                "ls /data/queue/ | wc -l",
                "curl http://worker.eu-west-1.internal/healthz",
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
                "curl http://payment.us-east-1.internal/healthz",
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
                "SQS depth increasing rapidly (847 messages backed up).\n"
                "User reports: 'Cannot make any payments for 30 minutes'"
            ),
            root_cause="Database lock causing payment failures and queue backup",
            correct_fix_description=(
                "1. Release RDS lock\n"
                "2. IMPORTANT: Drain queue at CONTROLLED rate (not all at once)\n"
                "3. If you drain all at once, worker will OOM from thundering herd"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="RDS lock released while queue depth > 100",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="847 queued requests flood worker simultaneously → OOM",
                    agent_must="Use 'queue drain 10' instead of 'queue drain all'",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://payment.us-east-1.internal/healthz",
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
                "curl http://worker.eu-west-1.internal/healthz",
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
            root_cause="RDS lock → payment retries exhaust memory → payment crashes",
            correct_fix_description=(
                "1. Release RDS lock first\n"
                "2. Then restart payment (it may have crashed from retries)\n"
                "3. Monitor payment memory after restart"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="RDS lock released → payment retry storm",
                    cascade_type="process_crash",
                    affected_service="payment",
                    description="Payment retries all failed writes → memory spike → OOM",
                    agent_must="Restart payment after releasing RDS lock",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://payment.us-east-1.internal/healthz",
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
                "curl http://worker.eu-west-1.internal/healthz",
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
                "Auth service crashed — port 8002 ConnectionRefused (ECS task not running).\n"
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
                "curl http://auth.us-east-1.internal/healthz",
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
                "Payment service returning 503 — RDS connection pool locked.\n"
                "Cache hit ratio dropping rapidly (94% → 12%).\n"
                "Cache entries expiring without refresh.\n"
                "User reports: 'Checkout keeps timing out'"
            ),
            root_cause="RDS lock blocks cache refresh → cache expires → thundering herd on DB recovery",
            correct_fix_description=(
                "1. Release RDS lock (root cause)\n"
                "2. IMPORTANT: Warm cache BEFORE traffic resumes\n"
                "3. If cache stays cold, thundering herd will crash DB again"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="RDS lock released while cache is cold",
                    cascade_type="cache_invalidation",
                    affected_service="cache",
                    description="All cached data expired during outage → 100% miss rate → DB thundering herd",
                    agent_must="Warm cache after fixing DB, before full traffic resumes",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://payment.us-east-1.internal/healthz",
                "curl http://cache.ap-south-1.internal/cache/stats",
                "cat /var/log/payment/error.log",
                "systemctl restart payment",
                "curl http://cache.ap-south-1.internal/cache/warmup",
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
                "curl http://payment.us-east-1.internal/healthz",
                "curl http://notification.ap-south-1.internal/webhooks/pending",
                "ps aux | grep payment",
                "systemctl restart payment",
                "curl -X POST http://notification.ap-south-1.internal/webhooks/drain?count=10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_search_corruption_to_gateway_block",
            failure_type="index_corruption",
            target_service="search",
            difficulty=0.58,
            alert_message=(
                "🚨 INCIDENT — 09:10 UTC | Severity: P0\n"
                "Search index corruption causing empty critical lookups.\n"
                "Gateway fallback traffic surging.\n"
                "User reports: 'Search broken and checkout now timing out'"
            ),
            root_cause="Search corruption first, then gateway throttling under retry surge",
            correct_fix_description=(
                "1. Recover search service first\n"
                "2. Then handle gateway overload cascade safely"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Search restored while retries are still high",
                    cascade_type="rate_limit_zero",
                    affected_service="gateway",
                    description="Recovery surge flips gateway into hard-reject mode",
                    agent_must="Restart gateway after stabilizing search",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://search.eu-west-1.internal/healthz",
                "curl http://gateway.us-east-1.internal/healthz",
                "cat /var/log/search/error.log",
                "systemctl restart search",
                "systemctl restart gateway",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_gateway_circuit_to_scheduler_stuck",
            failure_type="circuit_breaker_stuck",
            target_service="gateway",
            difficulty=0.57,
            alert_message=(
                "🚨 INCIDENT — 10:55 UTC | Severity: P0\n"
                "Gateway is rejecting all upstream calls.\n"
                "Scheduler callbacks now piling up.\n"
                "User reports: 'APIs down and jobs are delayed'"
            ),
            root_cause="Gateway circuit stuck open triggers scheduler backlog",
            correct_fix_description=(
                "1. Restore gateway service\n"
                "2. Monitor and recover scheduler after callback surge"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Gateway restored with pending callback backlog",
                    cascade_type="scheduler_stuck",
                    affected_service="scheduler",
                    description="Job callback flood stalls scheduler loop",
                    agent_must="Restart scheduler after gateway comes back",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://gateway.us-east-1.internal/gateway/stats",
                "curl http://scheduler.eu-west-1.internal/scheduler/stats",
                "systemctl restart gateway",
                "systemctl restart scheduler",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_scheduler_duplicate_to_billing_desync",
            failure_type="duplicate_execution",
            target_service="scheduler",
            difficulty=0.58,
            alert_message=(
                "🚨 INCIDENT — 12:40 UTC | Severity: P0\n"
                "Duplicate scheduler executions detected in billing jobs.\n"
                "User reports: 'Charges and invoices are inconsistent'"
            ),
            root_cause="Scheduler duplicate execution creates inconsistent billing writes",
            correct_fix_description=(
                "1. Recover scheduler first\n"
                "2. Then recover billing consistency"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Scheduler stabilized after duplicate bursts",
                    cascade_type="billing_desync",
                    affected_service="billing",
                    description="Duplicate writes leave billing ledger desynchronized",
                    agent_must="Restart billing after stabilizing scheduler",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://scheduler.eu-west-1.internal/scheduler/stats",
                "curl http://billing.us-east-1.internal/billing/stats",
                "systemctl restart scheduler",
                "systemctl restart billing",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_storage_disk_full_to_email_overflow",
            failure_type="disk_full",
            target_service="storage",
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 02:22 UTC | Severity: P0\n"
                "Storage writes are failing due to full disk.\n"
                "Retry notifications are rapidly accumulating.\n"
                "User reports: 'Uploads failing and alerts delayed'"
            ),
            root_cause="Storage write outage creates retry storm in email pipeline",
            correct_fix_description=(
                "1. Recover storage writes\n"
                "2. Then drain email pressure safely"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Storage restored while retries are queued",
                    cascade_type="email_queue_overflow",
                    affected_service="email",
                    description="Queued retry notices overflow email delivery queue",
                    agent_must="Restart email after restoring storage",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://storage.eu-west-1.internal/storage/stats",
                "curl http://email.ap-south-1.internal/email/stats",
                "systemctl restart storage",
                "systemctl restart email",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_metrics_scrape_failure_to_dns_stale",
            failure_type="scrape_failure",
            target_service="metrics_collector",
            difficulty=0.56,
            alert_message=(
                "🚨 INCIDENT — 14:08 UTC | Severity: P0\n"
                "Metrics collector blind — no fresh scrape data.\n"
                "Service discovery changes are now delayed.\n"
                "User reports: 'Routing feels random and monitoring is dark'"
            ),
            root_cause="Metrics scrape failure hides DNS freshness problems",
            correct_fix_description=(
                "1. Restore metrics collection first\n"
                "2. Then recover DNS freshness"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Metrics restored after prolonged blind period",
                    cascade_type="stale_entries",
                    affected_service="dns",
                    description="Stale DNS cache is discovered and starts causing wrong routing",
                    agent_must="Restart dns after metrics_collector recovery",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://metrics_collector.eu-west-1.internal/metrics/stats",
                "curl http://dns.ap-south-1.internal/dns/registry",
                "systemctl restart metrics_collector",
                "systemctl restart dns",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_email_smtp_down_to_notification_storm",
            failure_type="smtp_down",
            target_service="email",
            difficulty=0.57,
            alert_message=(
                "🚨 INCIDENT — 16:41 UTC | Severity: P0\n"
                "Email SMTP is down and deliveries are backing up.\n"
                "Webhook retries are increasing.\n"
                "User reports: 'No emails and delayed order confirmations'"
            ),
            root_cause="SMTP outage causes notification retry storm after recovery",
            correct_fix_description=(
                "1. Recover email service first\n"
                "2. Then handle notification retry surge"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Email recovered with large pending delivery backlog",
                    cascade_type="webhook_storm",
                    affected_service="notification",
                    description="Notification webhook retries fire in a burst",
                    agent_must="Drain retries conservatively after email recovery",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://email.ap-south-1.internal/email/stats",
                "curl http://notification.ap-south-1.internal/webhooks/pending",
                "systemctl restart email",
                "curl -X POST http://notification.ap-south-1.internal/webhooks/drain?count=10",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_billing_desync_to_config_poisoned",
            failure_type="billing_desync",
            target_service="billing",
            difficulty=0.58,
            alert_message=(
                "🚨 INCIDENT — 18:12 UTC | Severity: P0\n"
                "Billing ledger mismatch detected across services.\n"
                "Automated remediation attempted config changes.\n"
                "User reports: 'Charges wrong and APIs unstable'"
            ),
            root_cause="Billing desync triggers bad auto-config rollout",
            correct_fix_description=(
                "1. Stabilize billing first\n"
                "2. Then recover poisoned config state"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Billing recovered after inconsistent charge stream",
                    cascade_type="config_poisoned",
                    affected_service="config",
                    description="Auto-remediation writes unsafe config values",
                    agent_must="Restart config after billing stabilization",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://billing.us-east-1.internal/billing/stats",
                "curl http://config.us-east-1.internal/config/list",
                "systemctl restart billing",
                "systemctl restart config",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_config_locked_to_gateway_circuit",
            failure_type="config_locked",
            target_service="config",
            difficulty=0.60,
            alert_message=(
                "🚨 INCIDENT — 20:02 UTC | Severity: P0\n"
                "Config service locked and updates blocked.\n"
                "Gateway policy refresh failing repeatedly.\n"
                "User reports: 'Requests rejected after config freeze'"
            ),
            root_cause="Config lock prevents gateway policy updates",
            correct_fix_description=(
                "1. Unlock config service via restart\n"
                "2. Recover gateway from stuck circuit state"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Config restored after prolonged gateway refresh failures",
                    cascade_type="circuit_breaker_stuck",
                    affected_service="gateway",
                    description="Gateway remains stuck open after stale policy cycle",
                    agent_must="Restart gateway after config recovery",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://config.us-east-1.internal/config/list",
                "curl http://gateway.us-east-1.internal/gateway/stats",
                "systemctl restart config",
                "systemctl restart gateway",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_dns_failure_to_loadbalancer_empty",
            failure_type="dns_resolution_failure",
            target_service="dns",
            difficulty=0.61,
            alert_message=(
                "🚨 INCIDENT — 00:48 UTC | Severity: P0\n"
                "DNS resolution failing for internal services.\n"
                "Load balancer health checks cannot resolve backends.\n"
                "User reports: 'Entire platform serving 503s'"
            ),
            root_cause="DNS outage causes load balancer backend removal",
            correct_fix_description=(
                "1. Recover DNS resolution first\n"
                "2. Then recover load balancer backend pool"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="DNS restored after failed LB health checks",
                    cascade_type="all_backends_removed",
                    affected_service="loadbalancer",
                    description="Load balancer has already evicted all backends",
                    agent_must="Restart loadbalancer after DNS recovery",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://dns.ap-south-1.internal/dns/registry",
                "curl http://loadbalancer.us-east-1.internal/lb/stats",
                "systemctl restart dns",
                "systemctl restart loadbalancer",
            ],
            task_id="cascade",
        ),
        ScenarioSpec(
            scenario_id="cascade_loadbalancer_session_to_auth_overflow",
            failure_type="session_corruption",
            target_service="loadbalancer",
            difficulty=0.56,
            alert_message=(
                "🚨 INCIDENT — 05:58 UTC | Severity: P0\n"
                "Load balancer session routing is corrupted.\n"
                "Auth retries climbing sharply.\n"
                "User reports: 'Users constantly logged out'"
            ),
            root_cause="Session corruption at load balancer causes auth retry flood",
            correct_fix_description=(
                "1. Recover load balancer session routing\n"
                "2. Then stabilize auth retry pressure"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="Load balancer restored while retries remain queued",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="Session retry flood overflows worker queue",
                    agent_must="Drain queue conservatively after load balancer recovery",
                ),
            ],
            expected_diagnostic_path=[
                "curl http://loadbalancer.us-east-1.internal/lb/stats",
                "queue status",
                "systemctl restart loadbalancer",
                "queue drain 10",
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
                "  - Payment: 503 on all requests (RDS locked)\n"
                "  - Worker: Queue depth 940/1000, processing stalled\n"
                "  - Auth: Elevated error rate\n"
                "  - Frontend: 502 Bad Gateway on everything\n"
                "Duration: 45 minutes and escalating\n"
                "User reports: 'Site is completely down'"
            ),
            root_cause="Multi-fault: RDS lock (primary) + queue overflow (secondary)",
            correct_fix_description=(
                "PRIORITY ORDER:\n"
                "1. Fix RDS lock (primary cause of payment failure)\n"
                "2. Drain queue at controlled rate (prevent thundering herd)\n"
                "3. Verify all services healthy"
            ),
            cascade_rules=[
                CascadeRule(
                    trigger_condition="RDS lock released",
                    cascade_type="queue_overflow",
                    affected_service="worker",
                    description="Queue floods worker after DB release",
                    agent_must="Drain queue at controlled rate",
                ),
            ],
            expected_diagnostic_path=[
                "systemctl status",
                "curl http://payment.us-east-1.internal/healthz",
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
                "  - Payment: ConnectionRefused (ECS task not running) on port 8001\n"
                "  - Worker: ConnectionRefused (ECS task not running) on port 8003\n"
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
                "  - Payment: 503 — 'RDS connection pool exhausted'\n"
                "  - Worker: Queue at 960/1000, not processing\n"
                "  - Auth: Process crashed — port 8002 ConnectionRefused (ECS task not running)\n"
                "User reports: 'Complete site outage for 1 hour'"
            ),
            root_cause="RDS lock (root) → queue overflow → auth crashed from load",
            correct_fix_description=(
                "CRITICAL ORDER:\n"
                "1. Release RDS lock (unblocks payment)\n"
                "2. Drain queue slowly (prevents worker OOM)\n"
                "3. Restart auth (was collateral damage)\n"
                "4. Verify all services via systemctl status"
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

# Death Spiral scenario definitions — multi-fault cascading failures
# where fix order matters. Zero-shot 70B models CANNOT solve these.
_DEATH_SPIRALS = [
    {
        "id": "death_spiral_dns_cascade",
        "name": "The Global Outage",
        "root_fault": ("dns_resolution_failure", "dns"),
        "description": (
            "DNS corruption cascades through the entire infrastructure. "
            "Gateway can't route, loadbalancer returns 503, payment and billing fail. "
            "Fix order: DNS → gateway auto-recovers → restart payment → restart billing."
        ),
        "alert": (
            "🚨 INCIDENT — {time} UTC | Severity: P0\n"
            "CRITICAL: Multiple services degraded across all regions\n"
            "  - payment.us-east-1.internal: 503 Service Unavailable\n"
            "  - gateway.us-east-1.internal: 502 Bad Gateway\n"
            "  - billing.us-east-1.internal: ledger desync detected\n"
            "  - loadbalancer.us-east-1.internal: all backends removed\n"
            "  - dns.ap-south-1.internal: NXDOMAIN on all lookups\n"
            "Duration: 12 minutes and escalating\n"
            "Customer impact: ALL payment processing halted"
        ),
        "fix_order": ["dns", "gateway", "payment", "billing"],
    },
    {
        "id": "death_spiral_config_poison",
        "name": "The Config Poison",
        "root_fault": ("config_poisoned", "config"),
        "description": (
            "Config store poisoned with bad values. Search and metrics_collector "
            "use config for their operation — both degrade. Metrics blindness means "
            "the scheduler can't auto-scale, causing worker queue to overflow."
        ),
        "alert": (
            "🚨 INCIDENT — {time} UTC | Severity: P0\n"
            "CRITICAL: Configuration corruption detected\n"
            "  - config.us-east-1.internal: poisoned values (rate_limit=0)\n"
            "  - search.eu-west-1.internal: empty results on all queries\n"
            "  - metrics_collector.eu-west-1.internal: scrape failures\n"
            "  - worker.eu-west-1.internal: queue depth 950/1000\n"
            "Duration: 8 minutes and escalating\n"
            "Customer impact: Search broken, monitoring blind"
        ),
        "fix_order": ["config", "search", "metrics_collector"],
        "extra_faults": [("queue_overflow", {"fill": 950})],
    },
    {
        "id": "death_spiral_auth_collapse",
        "name": "The Auth Cascade",
        "root_fault": ("process_crash", "auth"),
        "description": (
            "Auth service OOMKilled. Everything that depends on auth fails: "
            "payment can't verify tokens, billing can't reconcile, frontend "
            "shows 401 errors to all users. Gateway circuit breaker trips."
        ),
        "alert": (
            "🚨 INCIDENT — {time} UTC | Severity: P0\n"
            "CRITICAL: Authentication infrastructure failure\n"
            "  - auth.us-east-1.internal: ECS task OOMKilled\n"
            "  - payment.us-east-1.internal: 401 Unauthorized on all requests\n"
            "  - billing.us-east-1.internal: invoice generation stuck\n"
            "  - frontend.ap-south-1.internal: user sessions expired\n"
            "  - gateway.us-east-1.internal: circuit breaker OPEN\n"
            "Duration: 5 minutes and escalating\n"
            "Customer impact: ALL users logged out, payments failing"
        ),
        "fix_order": ["auth", "gateway", "payment"],
        "extra_faults": [("circuit_breaker_stuck", {})],
        "auth_crash_params": {"target": "auth", "reason": "OOM"},
    },
    {
        "id": "death_spiral_storage_meltdown",
        "name": "The Storage Meltdown",
        "root_fault": ("disk_full", "storage"),
        "description": (
            "Storage disk full. Search can't write indexes, email queue "
            "overflows because logs can't be written, metrics retention "
            "fills up. The cascade looks like 5 independent failures but "
            "they all stem from storage."
        ),
        "alert": (
            "🚨 INCIDENT — {time} UTC | Severity: P1\n"
            "CRITICAL: Storage subsystem failure\n"
            "  - storage.eu-west-1.internal: disk 100% full, writes rejected\n"
            "  - search.eu-west-1.internal: index lag > 5000 documents\n"
            "  - email.ap-south-1.internal: queue overflow, dropping messages\n"
            "  - metrics_collector.eu-west-1.internal: retention full\n"
            "Duration: 15 minutes and escalating\n"
            "Customer impact: Search stale, emails delayed"
        ),
        "fix_order": ["storage", "search", "email", "metrics_collector"],
        "extra_faults": [
            ("index_lag", {}),
            ("email_queue_overflow", {}),
            ("retention_full", {}),
        ],
    },
    {
        "id": "death_spiral_scheduler_chaos",
        "name": "The Scheduler Chaos",
        "root_fault": ("scheduler_stuck", "scheduler"),
        "description": (
            "Scheduler stuck causes worker jobs to pile up. Worker queue "
            "overflows, notification service can't deliver because worker "
            "is backed up. Meanwhile scheduler duplicate execution mode "
            "fires billing jobs twice, causing desync."
        ),
        "alert": (
            "🚨 INCIDENT — {time} UTC | Severity: P1\n"
            "CRITICAL: Job scheduling infrastructure failure\n"
            "  - scheduler.eu-west-1.internal: main loop frozen\n"
            "  - worker.eu-west-1.internal: SQS depth 900/1000\n"
            "  - notification.ap-south-1.internal: delivery backed up\n"
            "  - billing.us-east-1.internal: duplicate charges detected\n"
            "Duration: 20 minutes and escalating\n"
            "Customer impact: Delayed processing, duplicate billing"
        ),
        "fix_order": ["scheduler", "worker"],
        "extra_faults": [
            ("queue_overflow", {"fill": 900}),
            ("billing_desync", {}),
        ],
    },
]


def _adversarial_scenarios(orchestrator, **kwargs) -> ScenarioSpec:
    """Tier 5: Death Spiral scenarios.

    Multi-fault cascading failures with strict fix-order requirements.
    The dependency graph in the orchestrator enforces that fixing
    downstream services without fixing upstream ROOT CAUSE will
    cause immediate re-degradation.

    This makes the environment IMPOSSIBLE for zero-shot 70B models:
    - They see 5 services failing and try to restart the obvious one
    - The service re-degrades because its upstream is still broken
    - Only GRPO-trained agents learn the dependency map
    """
    # 50% chance: Death Spiral, 50% chance: dynamic random
    if random.random() < 0.5 and _DEATH_SPIRALS:
        spiral = random.choice(_DEATH_SPIRALS)
        fault_type, target = spiral["root_fault"]

        # Inject root fault
        if fault_type == "process_crash":
            crash_params = spiral.get("auth_crash_params", {"target": target, "reason": "OOM"})
            orchestrator.inject_fault("process_crash", crash_params)
        else:
            orchestrator.inject_fault(fault_type, {})

        # Inject extra faults (symptoms that look independent but aren't)
        for extra_fault, extra_params in spiral.get("extra_faults", []):
            orchestrator.inject_fault(extra_fault, extra_params)

        time_str = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"
        alert = spiral["alert"].format(time=time_str)

        return ScenarioSpec(
            scenario_id=spiral["id"],
            failure_type=fault_type,
            target_service=target,
            params={},
            difficulty=0.85,
            alert_message=alert,
            root_cause=spiral["description"],
            correct_fix_description=f"Fix order: {' → '.join(spiral['fix_order'])}",
            misleading_signals={},
            cascade_rules={},
            expected_diagnostic_path=[
                "systemctl status",
                f"curl http://{target}.*.internal/healthz",
                f"cat /var/log/{target}/error.log",
            ],
            task_id="adversarial",
        )
    else:
        return _generate_dynamic_scenario(orchestrator, difficulty_range=(0.6, 0.9))


# ── Dynamic Scenario Generator ───────────────────────────────────────────

# Fault types that work ACROSS PROCESSES (no in-process methods needed)
_FAULT_CATALOG = {
    "db_lock": {
        "services": ["payment"],
        "gen_params": lambda: {},
        "root_cause_tpl": "RDS connection pool exhausted -- all writes blocked — all writes to RDS/Aurora are failing",
        "fix_tpl": "Release the RDS connection pool lock or restart the payment service",
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
        "root_cause_tpl": "SQS dead letter queue overflowed — worker cannot keep up",
        "fix_tpl": "Drain queue at controlled rate (queue drain 10)",
    },
    "queue_pause": {
        "services": ["worker"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Queue consumer paused — messages accumulating",
        "fix_tpl": "Resume queue consumer and check worker health",
    },
    "process_crash": {
        "services": [
            "payment", "auth", "worker", "cache", "notification",
            "search", "gateway", "scheduler", "storage", "metrics_collector",
            "email", "billing", "config", "dns", "loadbalancer",
        ],
        "gen_params": lambda: {
            "target": random.choice([
                "payment", "auth", "worker", "cache", "notification",
                "search", "gateway", "scheduler", "storage", "metrics_collector",
                "email", "billing", "config", "dns", "loadbalancer",
            ]),
            "reason": random.choice([
                "SIGSEGV", "SIGKILL", "OOM", "SIGABRT",
                "unhandled exception", "stack overflow",
            ]),
        },
        "root_cause_tpl": "Service instance terminated ({reason}) — needs restart",
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
    "latency_injection": {
        "services": [
            "payment", "auth", "worker", "cache", "notification",
            "search", "gateway", "scheduler", "storage", "metrics_collector",
            "email", "billing", "config", "dns", "loadbalancer",
        ],
        "gen_params": lambda: {
            "target": random.choice([
                "payment", "auth", "worker", "cache", "notification",
                "search", "gateway", "scheduler", "storage", "metrics_collector",
                "email", "billing", "config", "dns", "loadbalancer",
            ]),
            "latency_ms": random.choice([2000, 3000, 5000, 8000, 15000]),
        },
        "root_cause_tpl": "Network latency spike — p95 latency {latency_ms}ms (baseline: 40ms)",
        "fix_tpl": "Restart service to clear network degradation, check upstream dependencies",
    },
    # ── New service-specific fault types (10 new services) ──
    "index_corruption": {
        "services": ["search"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Search index corrupted — queries returning empty or incorrect results",
        "fix_tpl": "Restart search service to rebuild clean index",
    },
    "index_lag": {
        "services": ["search"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Search index lagging by >1000 documents — stale query results",
        "fix_tpl": "Restart search service to clear indexing backlog",
    },
    "rate_limit_zero": {
        "services": ["gateway"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Gateway rate limit misconfigured to 0 RPS — all traffic rejected",
        "fix_tpl": "Restart gateway service to restore default rate limits",
    },
    "circuit_breaker_stuck": {
        "services": ["gateway"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Gateway circuit breaker stuck open — all upstream calls rejected",
        "fix_tpl": "Restart gateway to reset circuit breaker state",
    },
    "scheduler_stuck": {
        "services": ["scheduler"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Scheduler loop frozen — no jobs executing",
        "fix_tpl": "Restart scheduler service to resume job execution",
    },
    "duplicate_execution": {
        "services": ["scheduler"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Scheduler executing jobs multiple times — duplicate processing",
        "fix_tpl": "Restart scheduler service to clear duplicate execution state",
    },
    "disk_full": {
        "services": ["storage"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Storage disk full — all write operations rejected",
        "fix_tpl": "Restart storage service to reclaim space and clear write queue",
    },
    "data_corruption": {
        "services": ["storage"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Storage data corruption — checksum validation failing",
        "fix_tpl": "Restart storage service to trigger integrity check and repair",
    },
    "scrape_failure": {
        "services": ["metrics_collector"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Metrics scrape failures — telemetry pipeline stale",
        "fix_tpl": "Restart metrics collector to reconnect scrape targets",
    },
    "retention_full": {
        "services": ["metrics_collector"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Metrics retention store full — new datapoints being dropped",
        "fix_tpl": "Restart metrics collector to flush and reclaim retention storage",
    },
    "smtp_down": {
        "services": ["email"],
        "gen_params": lambda: {},
        "root_cause_tpl": "SMTP upstream unavailable — email delivery queue backing up",
        "fix_tpl": "Restart email service to reconnect SMTP upstream",
    },
    "email_queue_overflow": {
        "services": ["email"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Email delivery queue overflow — messages being dropped",
        "fix_tpl": "Restart email service to drain and reset queue",
    },
    "billing_desync": {
        "services": ["billing"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Billing ledger desync — charges recorded but not reconciled",
        "fix_tpl": "Restart billing service to force reconciliation",
    },
    "invoice_stuck": {
        "services": ["billing"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Invoice generation stuck — invoices not being produced",
        "fix_tpl": "Restart billing service to resume invoice generation",
    },
    "config_poisoned": {
        "services": ["config"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Config store poisoned — critical keys set to unsafe values",
        "fix_tpl": "Restart config service to restore safe defaults",
    },
    "config_locked": {
        "services": ["config"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Config store locked — read/write operations blocked",
        "fix_tpl": "Restart config service to release lock",
    },
    "dns_resolution_failure": {
        "services": ["dns"],
        "gen_params": lambda: {},
        "root_cause_tpl": "DNS resolution failure — all lookups returning NXDOMAIN",
        "fix_tpl": "Restart DNS service to restore resolution",
    },
    "stale_entries": {
        "services": ["dns"],
        "gen_params": lambda: {},
        "root_cause_tpl": "DNS stale entries — services routed to dead endpoints",
        "fix_tpl": "Restart DNS service to flush stale cache",
    },
    "all_backends_removed": {
        "services": ["loadbalancer"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Load balancer has no healthy backends — all traffic returning 503",
        "fix_tpl": "Restart loadbalancer service to rediscover backends",
    },
    "session_corruption": {
        "services": ["loadbalancer"],
        "gen_params": lambda: {},
        "root_cause_tpl": "Sticky-session table corrupted — users routed inconsistently",
        "fix_tpl": "Restart loadbalancer service to rebuild session table",
    },
}

# Red herring generators — 22 templates for diverse misdirection (original 12 + 10 new services)
_RED_HERRING_TEMPLATES = [
    # ── Original 6 services ──
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
    # ── 10 new services ──
    {"service": "search", "message": "SearchService: index lag {latency}ms behind primary — stale results possible"},
    {"service": "gateway", "message": "GatewayService: circuit breaker half-open — {ratio}% of requests rerouted"},
    {"service": "scheduler", "message": "SchedulerService: job execution {latency}ms slower than SLO"},
    {"service": "storage", "message": "StorageService: disk utilization at {ratio}% — approaching threshold"},
    {"service": "metrics_collector", "message": "MetricsCollector: scrape latency elevated to {latency}ms (normally 50ms)"},
    {"service": "email", "message": "EmailService: delivery latency {latency}ms — SMTP upstream slow"},
    {"service": "billing", "message": "BillingService: reconciliation queue depth {used} — processing slow"},
    {"service": "config", "message": "ConfigService: config reload latency {latency}ms (normally 10ms)"},
    {"service": "dns", "message": "DNSService: resolution latency {latency}ms — cache miss rate {ratio}%"},
    {"service": "loadbalancer", "message": "LoadBalancer: backend health check latency {latency}ms — {ratio}% backends slow"},
]

# 17 cascade chain templates from real-world P0 incident patterns (original 7 + 10 new services)
_CASCADE_TEMPLATES = [
    # ── Original 7 chains ──
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
        "desc": "RDS lock release → payment retry storm → payment OOM",
        "agent_must": "Monitor payment after RDS lock release",
    },
    {
        "trigger_fault": "process_crash",
        "cascade_type": "db_lock",
        "affected": "payment",
        "desc": "Service restart hammers DB with connections → RDS locks",
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
    # ── 10 new service cascade chains ──
    {
        "trigger_fault": "index_corruption",
        "cascade_type": "rate_limit_zero",
        "affected": "gateway",
        "desc": "Search recovery surge triggers gateway rate limiter → blocks all traffic",
        "agent_must": "Restart gateway after stabilizing search",
    },
    {
        "trigger_fault": "circuit_breaker_stuck",
        "cascade_type": "scheduler_stuck",
        "affected": "scheduler",
        "desc": "Gateway blocks callbacks → scheduler job queue stalls",
        "agent_must": "Restart scheduler after gateway comes back",
    },
    {
        "trigger_fault": "duplicate_execution",
        "cascade_type": "billing_desync",
        "affected": "billing",
        "desc": "Duplicate job execution → double billing charges → ledger desync",
        "agent_must": "Restart billing after fixing scheduler duplicates",
    },
    {
        "trigger_fault": "disk_full",
        "cascade_type": "email_queue_overflow",
        "affected": "email",
        "desc": "Storage full → email attachments fail → email queue backs up",
        "agent_must": "Restart email after clearing storage",
    },
    {
        "trigger_fault": "scrape_failure",
        "cascade_type": "stale_entries",
        "affected": "dns",
        "desc": "Metrics blind → missed DNS TTL refresh → stale routing entries",
        "agent_must": "Restart DNS after restoring metrics pipeline",
    },
    {
        "trigger_fault": "smtp_down",
        "cascade_type": "webhook_storm",
        "affected": "notification",
        "desc": "Email SMTP down → notifications fall back to webhooks → webhook storm",
        "agent_must": "Rate-limit webhooks after restoring email",
    },
    {
        "trigger_fault": "billing_desync",
        "cascade_type": "config_poisoned",
        "affected": "config",
        "desc": "Billing desync triggers auto-config update with bad values",
        "agent_must": "Restart config service after fixing billing ledger",
    },
    {
        "trigger_fault": "config_locked",
        "cascade_type": "circuit_breaker_stuck",
        "affected": "gateway",
        "desc": "Config locked → gateway can't read routing config → circuit opens",
        "agent_must": "Restart gateway after releasing config lock",
    },
    {
        "trigger_fault": "dns_resolution_failure",
        "cascade_type": "all_backends_removed",
        "affected": "loadbalancer",
        "desc": "DNS failure → LB can't resolve backends → removes all from pool",
        "agent_must": "Restart loadbalancer after fixing DNS",
    },
    {
        "trigger_fault": "session_corruption",
        "cascade_type": "process_crash",
        "affected": "auth",
        "desc": "LB session corruption → auth hammered with re-auth requests → OOM",
        "agent_must": "Restart auth after fixing loadbalancer sessions",
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

    Produces 28 fault types x 16 services x N_params x 22 herrings x 17 cascades
    = thousands of unique scenarios. The model CANNOT memorize solutions.
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
        # Original faults
        "db_lock": "HTTP 503 on all requests (database errors)",
        "queue_overflow": f"Queue depth critical ({params.get('fill', 900)}/1000)",
        "process_crash": f"Process crashed ({params.get('reason', 'SIGSEGV')})",
        "db_pool_exhaustion": "DB connection pool exhausted — all queries failing",
        "queue_pause": "Queue consumer paused — messages accumulating",
        "cache_invalidation": "Cache fully invalidated — 0% hit ratio",
        "webhook_storm": "Webhook retry storm — hundreds of retries per second",
        "latency_injection": f"P95 latency spike to {params.get('latency_ms', 5000)}ms",
        # New service faults
        "index_corruption": "Search index corrupted — queries returning empty results",
        "index_lag": "Search index lagging >1000 docs — stale results",
        "rate_limit_zero": "Rate limit set to 0 RPS — all traffic rejected",
        "circuit_breaker_stuck": "Circuit breaker stuck open — upstream calls blocked",
        "scheduler_stuck": "Scheduler loop frozen — jobs not executing",
        "duplicate_execution": "Jobs executing multiple times — duplicate processing",
        "disk_full": "Disk usage 100% — write operations rejected",
        "data_corruption": "Data corruption detected — checksum failures",
        "scrape_failure": "Metrics scrape failing — telemetry pipeline stale",
        "retention_full": "Metrics retention full — new datapoints dropped",
        "smtp_down": "SMTP upstream unreachable — email delivery failing",
        "email_queue_overflow": "Email queue overflow — messages being dropped",
        "billing_desync": "Billing ledger desync — charges not reconciled",
        "invoice_stuck": "Invoice generation stuck — no invoices produced",
        "config_poisoned": "Config store poisoned — unsafe values in critical keys",
        "config_locked": "Config store locked — read/write blocked",
        "dns_resolution_failure": "DNS resolution failing — NXDOMAIN on all lookups",
        "stale_entries": "DNS stale entries — routing to dead endpoints",
        "all_backends_removed": "No healthy backends — all traffic returning 503",
        "session_corruption": "Sticky-session table corrupted — inconsistent routing",
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
    port_map = {
        "payment": 8001, "auth": 8002, "worker": 8003, "frontend": 8004,
        "cache": 8005, "notification": 8006, "search": 8007, "gateway": 8008,
        "scheduler": 8009, "storage": 8010, "metrics_collector": 8011,
        "email": 8012, "billing": 8013, "config": 8014, "dns": 8015,
        "loadbalancer": 8016,
    }

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
