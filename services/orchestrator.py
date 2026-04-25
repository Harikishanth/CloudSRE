"""
CloudSRE v2 — Service Orchestrator (Hybrid OS-Level + HTTP Fault Injection).

Manages the lifecycle of all 16 microservices + infrastructure.
Each service runs as a SEPARATE OS PROCESS with its own PID.

This is NOT threading. Each service is a real process:
  - Separate PID (visible in `ps aux`)
  - Separate memory space (real isolation)
  - Real signal handling (kill -9 actually kills it)
  - Real port binding (Connection refused when dead)

FAULT INJECTION — THREE PHYSICAL TIERS:

  Tier 1 — OS-Level (POSIX signals):
    SIGKILL: process_crash — physically kills process, port stops listening
    SIGSTOP: scheduler_stuck, config_locked, dns_failure, smtp_down,
             circuit_breaker_stuck, rate_limit_zero — freezes process,
             TCP connections physically timeout
    SIGCONT: Used by restart to resume frozen processes

  Tier 2 — Physical Infrastructure:
    SQLite EXCLUSIVE lock: db_lock — blocks ALL processes sharing the DB file
    File-backed queue overflow: queue_overflow — real files on disk
    fallocate/file write: disk_full — real disk consumption
    DB corruption: data_corruption — physical garbage bytes in SQLite file

  Tier 3 — HTTP State (for faults that can't be OS-level without root):
    /fault_inject endpoint: Service returns real HTTP 503/429/507 from healthz
    Used for: cache_invalidation, webhook_storm, index_corruption, etc.

Death Spirals emerge NATURALLY through TCP:
  - SIGSTOP dns → gateway can't resolve → ConnectTimeout
  - SIGKILL auth → payment can't authenticate → ConnectionRefused
  No scripted re-injection needed for Tier 1 faults.
"""

import os
import sys
import time
import random
import asyncio
import signal
import subprocess
import threading
import logging
import tempfile
from typing import Dict, Any, Optional, List

from cloud_sre_v2.infra.database import Database
from cloud_sre_v2.infra.queue import MessageQueue
from cloud_sre_v2.server.command_executor import CommandExecutor

logger = logging.getLogger(__name__)

# Port assignments for each service
SERVICE_PORTS = {
    # ── Core Services (original 6) ──
    "payment": 8001,
    "auth": 8002,
    "worker": 8003,
    "frontend": 8004,
    "cache": 8005,
    "notification": 8006,
    # ── Extended Services (10 new) ──
    "search": 8007,
    "gateway": 8008,
    "scheduler": 8009,
    "storage": 8010,
    "metrics_collector": 8011,
    "email": 8012,
    "billing": 8013,
    "config": 8014,
    "dns": 8015,
    "loadbalancer": 8016,
}


class ServiceOrchestrator:
    """Manages all 16 services as SEPARATE OS PROCESSES.

    Each service runs in its own Python process with a unique PID and port:
      payment(:8001) auth(:8002) worker(:8003) frontend(:8004) cache(:8005)
      notification(:8006) search(:8007) gateway(:8008) scheduler(:8009)
      storage(:8010) metrics_collector(:8011) email(:8012) billing(:8013)
      config(:8014) dns(:8015) loadbalancer(:8016)

    Kill a service = process.kill() -> real SIGTERM -> port stops listening
    Restart = spawn new subprocess -> new PID -> port opens again
    """

    def __init__(
        self,
        db_path: str = "/data/app.db",
        log_dir: str = "/var/log",
    ):
        self.db_path = db_path
        self.log_dir = log_dir
        self.queue_dir = os.path.join(os.path.dirname(db_path), "queue")

        # Infrastructure (orchestrator's own connections for fault injection)
        self.database: Optional[Database] = None
        self.queue: Optional[MessageQueue] = None

        # Command executor
        self.executor: Optional[CommandExecutor] = None

        # Process tracking: {name: {"proc": Popen, "pid": int, "port": int}}
        self._processes: Dict[str, dict] = {}
        self._crashed_services: set = set()
        self._degraded_services: Dict[str, str] = {}  # {name: reason} — fallback for HTTP failures
        self._latency_injected: Dict[str, int] = {}  # {name: latency_ms}
        self._running = False

        # Active fault tracking — {service: fault_type}
        # Used by re-injection guard to prevent fixing downstream without upstream
        self._active_faults: Dict[str, str] = {}

        # Cascade state
        self._armed_cascade = None

        # Service dependency graph — if upstream is broken, downstream auto-degrades
        # This creates "Death Spirals" that 70B models can't solve zero-shot
        self._dependency_graph: Dict[str, List[str]] = {
            # DNS is foundational — everything depends on it
            "gateway": ["dns"],
            "loadbalancer": ["dns", "gateway"],
            # Auth is needed by everything that handles users
            "payment": ["auth", "gateway"],
            "billing": ["payment", "auth"],
            "frontend": ["gateway", "auth", "cache"],
            # Worker depends on queue + scheduler
            "worker": ["scheduler"],
            "notification": ["email", "worker"],
            # Config affects everything
            "search": ["config", "storage"],
            "metrics_collector": ["config"],
        }

    def start(self):
        """Start all infrastructure and services as separate processes."""
        logger.info("Starting CloudSRE orchestrator...")

        # 1. Initialize infrastructure (orchestrator's own instances)
        self.database = Database(self.db_path)
        self.database.initialize()
        logger.info("Database initialized")

        os.makedirs(self.queue_dir, exist_ok=True)
        self.queue = MessageQueue(max_size=1000, queue_dir=self.queue_dir)
        logger.info(f"Message queue initialized (files at {self.queue_dir})")

        # 2. Create command executor
        self.executor = CommandExecutor(
            services={},  # No in-process service objects — they're in subprocesses
            infra={
                "database": self.database,
                "queue": self.queue,
            },
            orchestrator=self,
        )
        logger.info("Command executor initialized")

        # 3. Spawn each service as a SEPARATE OS PROCESS
        for name, port in SERVICE_PORTS.items():
            self._spawn_service(name, port)

        # Wait for all services to bind their ports (Windows needs longer)
        time.sleep(2.0)
        self._running = True
        logger.info("CloudSRE orchestrator started — all services running as separate processes")

    def _spawn_service(self, name: str, port: int):
        """Spawn a service as a separate OS process.

        Each service gets:
          - Its own PID
          - Its own memory space
          - Its own Database connection (to the shared file)
          - Its own Queue connection (to the shared directory)
        """
        cmd = [
            sys.executable, "-m", "cloud_sre_v2.services._service_worker",
            "--service", name,
            "--port", str(port),
            "--db-path", self.db_path,
            "--queue-dir", self.queue_dir,
            "--log-dir", self.log_dir,
        ]

        # Compute the package root (parent of cloud_sre_v2/)
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = os.environ.copy()
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=package_root,
            env=env,
        )

        self._processes[name] = {
            "proc": proc,
            "pid": proc.pid,
            "port": port,
        }
        logger.info(f"  {name} → PID={proc.pid} port={port}")

    def _stop_service(self, name: str) -> bool:
        """Kill a service process — sends REAL signal.

        On Linux: SIGTERM then SIGKILL
        The port ACTUALLY stops listening. curl gets 'Connection refused'.
        """
        if name not in self._processes:
            return False

        entry = self._processes[name]
        proc = entry["proc"]

        if proc.poll() is None:  # Still running
            proc.terminate()  # SIGTERM
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()  # SIGKILL — the real deal
                proc.wait(timeout=1.0)

        self._crashed_services.add(name)
        time.sleep(0.3)  # Let OS release the port
        logger.info(f"  SERVICE KILLED: {name} PID={entry['pid']} (port {entry['port']} closed)")
        return True

    def restart_service(self, name: str) -> str:
        """Restart a service — kill old process, spawn new one.

        New process gets a NEW PID. This is a REAL restart.

        RE-INJECTION GUARD: If this service has an upstream dependency
        that is still broken, the fault will re-inject after restart.
        The agent MUST fix the root cause (upstream) first.
        """
        if name not in self._processes:
            return f"Service '{name}' not found"

        port = self._processes[name]["port"]
        old_pid = self._processes[name]["pid"]

        # SIGCONT any SIGSTOP'd process before killing (can't kill a frozen process cleanly)
        if sys.platform != "win32":
            try:
                os.kill(old_pid, signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass

        # Kill if still running
        if name in self._crashed_services:
            self._crashed_services.discard(name)
        else:
            self._stop_service(name)
            self._crashed_services.discard(name)

        # Spawn fresh process
        self._spawn_service(name, port)

        # Wait for port to actually bind (up to 5s)
        import socket
        for attempt in range(25):  # 25 * 0.2s = 5s max
            try:
                with socket.create_connection(("localhost", port), timeout=0.5):
                    break  # Port is open!
            except (ConnectionRefusedError, OSError, socket.timeout):
                time.sleep(0.2)

        new_pid = self._processes[name]["pid"]

        # Clear degraded state and latency — restart fixes both
        self._degraded_services.pop(name, None)
        self._latency_injected.pop(name, None)

        # Clear fault via HTTP in case service was only degraded (not crashed)
        self._clear_fault_via_http(name)

        # Clear from active faults (this service was explicitly restarted)
        self._active_faults.pop(name, None)

        # RE-INJECTION GUARD: Check if upstream dependencies are still broken
        # If upstream is broken, this service will degrade again immediately
        reinjection_msg = self._check_dependency_reinjection(name)

        logger.info(f"  SERVICE RESTARTED: {name} PID={old_pid}→{new_pid} port={port}")
        result = f"Service {name} restarted (PID {old_pid}→{new_pid}, port {port})"
        if reinjection_msg:
            result += f"\n⚠ WARNING: {reinjection_msg}"
        return result

    def _check_dependency_reinjection(self, service_name: str) -> Optional[str]:
        """Check if a restarted service should re-degrade due to broken upstream.

        This is the core of the "Death Spiral" mechanic:
        - Agent restarts payment service (which is broken)
        - But gateway is ALSO broken (upstream dependency)
        - payment comes back up but immediately starts failing
          because gateway can't route traffic to it

        The ONLY way to fix this is to fix gateway FIRST, then payment.
        A 70B model doesn't know this dependency exists.
        """
        deps = self._dependency_graph.get(service_name, [])
        for upstream in deps:
            # Check if upstream has an active fault
            if upstream in self._active_faults:
                upstream_fault = self._active_faults[upstream]
                degraded_msg = (
                    f"Upstream dependency '{upstream}' is still broken "
                    f"({upstream_fault}) — {service_name} re-degrading"
                )
                # Re-inject degradation via real HTTP
                self._inject_fault_via_http(
                    service_name, f"upstream_dependency_failure",
                    f"Cannot operate: upstream '{upstream}' is down ({upstream_fault})"
                )
                self._active_faults[service_name] = f"upstream_{upstream}_{upstream_fault}"
                self._write_service_log(service_name, "error",
                    f"Service re-degraded: upstream dependency '{upstream}' "
                    f"has active fault: {upstream_fault}")
                logger.info(f"  RE-INJECTION: {service_name} re-degraded (upstream {upstream} broken)")
                return degraded_msg
        return None

    def _propagate_fault_to_dependents(self, source_service: str, fault_type: str):
        """When a service gets a fault, propagate degradation to its dependents.

        This creates realistic cascading failures:
        - DNS goes down → gateway, loadbalancer, payment all degrade
        - Config gets poisoned → search, metrics_collector degrade

        The agent sees multiple services failing and must figure out
        which one is the ROOT CAUSE vs which ones are just symptoms.
        """
        for downstream, upstreams in self._dependency_graph.items():
            if source_service in upstreams and downstream not in self._active_faults:
                cascaded_msg = (
                    f"Degraded: upstream '{source_service}' has fault '{fault_type}'"
                )
                self._inject_fault_via_http(
                    downstream, "upstream_dependency_failure", cascaded_msg
                )
                self._active_faults[downstream] = f"upstream_{source_service}_{fault_type}"
                self._write_service_log(downstream, "error",
                    f"SERVICE DEGRADED: upstream dependency '{source_service}' "
                    f"is broken ({fault_type}) — cascading failure")
                logger.info(f"  DEPENDENCY CASCADE: {source_service}→{downstream}")

    def reset(self):
        """Reset everything — kill all processes, wipe state, respawn.

        1. Kill ALL service processes
        2. Wipe database + queue
        3. Re-initialize fresh
        4. Spawn ALL services as new processes (new PIDs)
        """
        start = time.time()

        # 0. SIGCONT any SIGSTOP'd processes before killing them
        if sys.platform != "win32":
            for name, entry in self._processes.items():
                pid = entry.get("pid")
                if pid:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except (ProcessLookupError, OSError):
                        pass

        # 1. Kill all running service processes
        for name in list(self._processes.keys()):
            self._stop_service(name)
        self._crashed_services.clear()
        self._degraded_services.clear()
        self._latency_injected.clear()
        self._active_faults.clear()

        # 2. Clear cascade state
        self._armed_cascade = None

        # 2.5. Clean up OS-level fault artifacts
        # Remove disk_full junk file
        junk_path = os.path.join(
            os.environ.get("DATA_DIR", "/data"), "_disk_full_junk.bin")
        if os.path.exists(junk_path):
            try:
                os.remove(junk_path)
            except OSError:
                pass

        # 3. Reset infrastructure (also repairs corrupted DB)
        self.database.reset()
        self.queue.reset()

        # 4. Seed baseline data
        self._seed_baseline_data()

        # 5. Respawn all services as fresh processes
        for name, port in SERVICE_PORTS.items():
            self._spawn_service(name, port)
        time.sleep(2.0)  # Windows needs longer for port re-binding

        elapsed = (time.time() - start) * 1000
        logger.info(f"Environment reset in {elapsed:.0f}ms (all processes restarted)")

    def _seed_baseline_data(self):
        """Seed realistic baseline data for a new episode."""
        for i in range(20):
            self.database.execute(
                "INSERT INTO payments (amount, user_id, status, created_at) VALUES (?, ?, ?, datetime('now', ?))",
                (
                    round(random.uniform(10, 500), 2),
                    f"user_{random.randint(1, 100)}",
                    "completed",
                    f"-{random.randint(1, 60)} minutes",
                ),
            )
        for i in range(5):
            self.database.execute(
                "INSERT INTO sessions (user_id, token, is_valid) VALUES (?, ?, 1)",
                (f"user_{random.randint(1, 100)}", f"tok_{i}"),
            )

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_fault(self, fault_type: str, params: Dict[str, Any] = None) -> str:
        """Inject a real fault. Works ACROSS processes via shared resources."""
        params = params or {}
        target = params.get("target", params.get("service", "payment"))

        injectors = {
            "db_lock": self._inject_db_lock,
            "db_pool_exhaustion": self._inject_db_pool,
            "queue_overflow": self._inject_queue_overflow,
            "queue_pause": self._inject_queue_pause,
            "process_crash": self._inject_process_crash,
            "misleading_signal": self._inject_misleading_signal,
            "cache_invalidation": self._inject_cache_invalidation,
            "webhook_storm": self._inject_webhook_storm,
            "latency_injection": self._inject_latency,
            # New service-specific faults
            "index_corruption": self._inject_index_corruption,
            "index_lag": self._inject_index_lag,
            "rate_limit_zero": self._inject_rate_limit_zero,
            "circuit_breaker_stuck": self._inject_circuit_breaker_stuck,
            "scheduler_stuck": self._inject_scheduler_stuck,
            "duplicate_execution": self._inject_duplicate_execution,
            "disk_full": self._inject_disk_full,
            "data_corruption": self._inject_data_corruption,
            "scrape_failure": self._inject_scrape_failure,
            "retention_full": self._inject_retention_full,
            "smtp_down": self._inject_smtp_down,
            "email_queue_overflow": self._inject_email_queue_overflow,
            "billing_desync": self._inject_billing_desync,
            "invoice_stuck": self._inject_invoice_stuck,
            "config_poisoned": self._inject_config_poisoned,
            "config_locked": self._inject_config_locked,
            "dns_resolution_failure": self._inject_dns_resolution_failure,
            "stale_entries": self._inject_stale_entries,
            "all_backends_removed": self._inject_all_backends_removed,
            "session_corruption": self._inject_session_corruption,
        }

        fn = injectors.get(fault_type)
        if not fn:
            return f"Unknown fault type: {fault_type}"

        result = fn(target, params)

        # Track this as an active fault
        # Map fault_type → the service it primarily affects
        fault_service_map = {
            "db_lock": "payment", "db_pool": "payment",
            "queue_overflow": "worker", "queue_pause": "worker",
            "cache_invalidation": "cache", "webhook_storm": "notification",
            "index_corruption": "search", "index_lag": "search",
            "rate_limit_zero": "gateway", "circuit_breaker_stuck": "gateway",
            "scheduler_stuck": "scheduler", "duplicate_execution": "scheduler",
            "disk_full": "storage", "data_corruption": "storage",
            "scrape_failure": "metrics_collector", "retention_full": "metrics_collector",
            "smtp_down": "email", "email_queue_overflow": "email",
            "billing_desync": "billing", "invoice_stuck": "billing",
            "config_poisoned": "config", "config_locked": "config",
            "dns_resolution_failure": "dns", "stale_entries": "dns",
            "all_backends_removed": "loadbalancer", "session_corruption": "loadbalancer",
        }
        affected_service = fault_service_map.get(fault_type, target)
        self._active_faults[affected_service] = fault_type

        # Propagate to dependent services (Death Spiral)
        if fault_type != "upstream_dependency_failure":
            self._propagate_fault_to_dependents(affected_service, fault_type)

        return result

    def _inject_db_lock(self, target: str, params: dict) -> str:
        """Lock DB via EXCLUSIVE transaction — blocks ALL service processes."""
        self.database.inject_lock()
        self._write_service_log("payment", "error",
            "DatabaseConnectionPool: Connection timeout after 30s")
        self._write_service_log("payment", "error",
            "DatabaseConnectionPool: Pool exhausted (max_connections=50, active=50)")
        self._write_service_log("worker", "error",
            "Database write failed: database is locked")
        return "Injected: database lock"

    def _inject_db_pool(self, target: str, params: dict) -> str:
        self.database.inject_connection_exhaustion()
        self._write_service_log("payment", "error",
            "Connection pool exhausted — no connections available")
        return "Injected: connection pool exhaustion"

    def _inject_queue_overflow(self, target: str, params: dict) -> str:
        fill = params.get("fill", 900)
        self.queue.inject_overflow(fill_count=fill)
        # NOTE: Do NOT mark worker as degraded here.
        # check_health() dynamically checks queue.depth() > 500 instead.
        # This ensures warmup (max_steps=10) can actually resolve.
        self._write_service_log("worker", "error",
            f"Queue depth critical: {self.queue.depth()}/{self.queue._max_size}")
        self._write_service_log("worker", "error",
            "WorkerService: OOM risk — processing backlog consuming excessive memory")
        return f"Injected: queue overflow ({self.queue.depth()} messages)"

    def _inject_queue_pause(self, target: str, params: dict) -> str:
        self.queue.inject_pause()
        self._write_service_log("worker", "error",
            "Consumer paused — not accepting messages")
        return "Injected: queue pause"

    def _inject_process_crash(self, target: str, params: dict) -> str:
        """Kill a process — REAL signal, port stops listening."""
        if target not in self._processes:
            return f"Service '{target}' not found"
        reason = params.get('reason', 'SIGSEGV')
        self._write_service_log(target, "error", f"Process crashed: {reason}")
        self._stop_service(target)
        pid = self._processes[target]["pid"]
        return f"Injected: {target} process crash (PID {pid} killed, port closed)"


    def _inject_fault_via_http(self, service_name: str, fault_type: str,
                                error_message: str, severity: str = "degraded") -> bool:
        """Send fault injection command to a running service process via HTTP.

        This makes faults REAL — the service process itself starts returning
        HTTP errors (503, 429, 507) from its /healthz endpoint.

        Args:
            service_name: Name of the service (e.g., "search")
            fault_type: Type of fault (e.g., "index_corruption")
            error_message: Human-readable error description
            severity: "degraded" (503 from healthz) or "critical" (503 + unhealthy)

        Returns:
            True if the HTTP injection succeeded, False otherwise.
        """
        import httpx
        port = SERVICE_PORTS.get(service_name)
        if port is None:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(f"http://localhost:{port}/fault_inject", json={
                    "fault_type": fault_type,
                    "params": {
                        "error_message": error_message,
                        "severity": severity,
                    },
                })
            return True
        except Exception:
            # Service might be down — fall back to flag
            self._degraded_services[service_name] = error_message
            return False

    def _clear_fault_via_http(self, service_name: str) -> bool:
        """Clear a fault on a running service via HTTP POST to /fault_clear."""
        import httpx
        port = SERVICE_PORTS.get(service_name)
        if port is None:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                client.post(f"http://localhost:{port}/fault_clear", json={})
            return True
        except Exception:
            return False

    def _inject_misleading_signal(self, target: str, params: dict) -> str:
        message = params.get("message", "Elevated latency detected (120ms vs 40ms baseline)")
        self._write_service_log(target, "error", f"[MISLEADING] {message}")
        return f"Injected: misleading signal in {target}"

    def _inject_latency(self, target: str, params: dict) -> str:
        """Inject network latency into a service — simulates degraded network."""
        latency_ms = params.get("latency_ms", 3000)
        self._latency_injected[target] = latency_ms
        self._inject_fault_via_http(target, "latency",
            f"Network latency spike — p95 latency {latency_ms}ms (baseline: 40ms)")
        self._write_service_log(target, "error",
            f"NetworkMonitor: latency spike detected — "
            f"p95={latency_ms}ms, p99={latency_ms*2}ms, baseline=40ms")
        self._write_service_log(target, "error",
            f"Upstream connections timing out after {latency_ms}ms")
        return f"Injected: {latency_ms}ms latency into {target}"

    def _inject_cache_invalidation(self, target: str, params: dict) -> str:
        """Invalidate cache — physically delete cache data files.

        OS-LEVEL: Deletes all files in the cache data directory, then
        SIGSTOP the cache process for 2s to simulate a cold restart.
        """
        # Physical: delete cache data files
        cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        for fname in os.listdir(cache_dir):
            try:
                os.remove(os.path.join(cache_dir, fname))
            except OSError:
                pass
        # Physical: SIGSTOP cache process briefly to simulate cold restart
        pid = self._processes.get("cache", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                threading.Timer(2.0, lambda: os.kill(pid, signal.SIGCONT)).start()
            except ProcessLookupError:
                pass
        self._inject_fault_via_http("cache", "cache_invalidation",
            "ElastiCache fully invalidated — 100% miss rate, thundering herd risk")
        self._write_service_log("cache", "error",
            "CacheService: FULL INVALIDATION — all entries evicted (files deleted)")
        self._write_service_log("cache", "error",
            "CacheService: Hit ratio dropped to 0.0% — cache is COLD")
        self._write_service_log("payment", "error",
            "PaymentService: cache miss rate 100% — falling through to database")
        return "Injected: cache invalidation (files deleted + SIGSTOP cold restart)"

    def _inject_webhook_storm(self, target: str, params: dict) -> str:
        """Trigger mass webhook retry — sends REAL HTTP requests to overwhelm the service.

        OS-LEVEL: Spawns threads that fire actual HTTP requests at the
        notification service, creating real connection exhaustion.
        """
        count = params.get("count", 100)
        port = self._processes.get("notification", {}).get("port", 8006)
        # Physical: fire real HTTP requests in background threads
        def _storm():
            import httpx as _httpx
            for _ in range(count):
                try:
                    _httpx.post(f"http://localhost:{port}/webhook",
                               json={"event": "payment.retry", "attempt": _},
                               timeout=1.0)
                except Exception:
                    pass
        storm_thread = threading.Thread(target=_storm, daemon=True)
        storm_thread.start()
        self._inject_fault_via_http("notification", "webhook_storm",
            f"Webhook storm ({count} retries) — delivery pipeline overwhelmed")
        self._write_service_log("notification", "error",
            f"NotificationService: WEBHOOK STORM — {count} real HTTP requests fired")
        self._write_service_log("notification", "error",
            f"NotificationService: Queue depth {count}/500 — delivery rate overwhelmed")
        self._write_service_log("payment", "error",
            "PaymentService: Inbound webhook callbacks spiking — 300 req/sec (normal: 10)")
        return f"Injected: webhook storm ({count} real HTTP requests)"

    def _inject_index_corruption(self, target: str, params: dict) -> str:
        """Search index corruption — write corrupt data to index file on disk."""
        index_file = os.path.join(self.data_dir, "search_index.dat")
        try:
            with open(index_file, "wb") as f:
                f.write(os.urandom(1024))  # Physical corruption
        except OSError:
            pass
        self._inject_fault_via_http("search", "index_corruption",
            "Search index corrupted — queries returning empty or incorrect results")
        self._write_service_log("search", "error",
            f"SearchService: index corruption detected — {index_file} has invalid checksum")
        return "Injected: search index corruption (physical file corruption)"

    def _inject_index_lag(self, target: str, params: dict) -> str:
        """Search index lag — write 1000+ pending docs to a backlog file."""
        backlog_file = os.path.join(self.data_dir, "search_backlog.json")
        try:
            import json as _json
            backlog = [{"doc_id": i, "status": "pending"} for i in range(1200)]
            with open(backlog_file, "w") as f:
                _json.dump(backlog, f)
        except OSError:
            pass
        self._inject_fault_via_http("search", "index_lag",
            "Search index lagging by >1000 docs — stale query results")
        self._write_service_log("search", "error",
            f"SearchService: index lag critical — {backlog_file} has 1200 pending docs")
        return "Injected: search index lag (1200 docs in physical backlog)"

    def _inject_rate_limit_zero(self, target: str, params: dict) -> str:
        """Rate limit zero — SIGSTOP the gateway process briefly to cause real connection drops.

        OS-LEVEL: The gateway process is physically frozen for 3 seconds.
        All TCP connections to it will timeout. Then it resumes but with
        the HTTP degraded flag so healthz returns 429.
        """
        pid = self._processes.get("gateway", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)  # Physically freeze the process
                threading.Timer(3.0, lambda: os.kill(pid, signal.SIGCONT)).start()
                self._write_service_log("gateway", "error",
                    "GatewayService: process frozen by kernel (SIGSTOP) — all connections dropped")
            except ProcessLookupError:
                pass
        # Also set HTTP flag so healthz continues to return 429 after SIGCONT
        self._inject_fault_via_http("gateway", "rate_limit_zero",
            "Gateway rate limit misconfigured to 0 RPS — all requests rejected")
        self._write_service_log("gateway", "error", "GatewayService: rate_limit=0, blocking all traffic")
        return "Injected: gateway rate limit zero (SIGSTOP + HTTP 429)"

    def _inject_circuit_breaker_stuck(self, target: str, params: dict) -> str:
        """Circuit breaker stuck — SIGSTOP the gateway to simulate a frozen event loop.

        OS-LEVEL: Gateway process physically frozen for 5 seconds, causing
        real TCP connection refusals from the OS kernel.
        """
        pid = self._processes.get("gateway", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                threading.Timer(5.0, lambda: os.kill(pid, signal.SIGCONT)).start()
                self._write_service_log("gateway", "error",
                    "GatewayService: event loop frozen by kernel (SIGSTOP) — circuit breaker stuck")
            except ProcessLookupError:
                pass
        self._inject_fault_via_http("gateway", "circuit_breaker_stuck",
            "Gateway circuit breaker stuck open — all upstream calls rejected")
        self._write_service_log("gateway", "error", "GatewayService: circuit breaker OPEN and not auto-resetting")
        return "Injected: gateway circuit breaker stuck (SIGSTOP + HTTP 503)"

    def _inject_scheduler_stuck(self, target: str, params: dict) -> str:
        """Scheduler stuck — SIGSTOP the scheduler process entirely.

        OS-LEVEL: The scheduler process is physically frozen by the kernel.
        Its port stops responding. Worker jobs pile up because the scheduler
        literally cannot dispatch them. This is REAL — not a flag.
        """
        pid = self._processes.get("scheduler", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)  # Physically freeze scheduler
                self._write_service_log("scheduler", "error",
                    "SchedulerService: process frozen by kernel (SIGSTOP) — main loop halted")
            except ProcessLookupError:
                pass
        else:
            # Fallback for Windows: HTTP mock
            self._inject_fault_via_http("scheduler", "scheduler_stuck",
                "Scheduler loop stuck — jobs not executing")
        self._write_service_log("scheduler", "error", "SchedulerService: scheduler loop frozen, no jobs executing")
        return "Injected: scheduler stuck (SIGSTOP)"

    def _inject_duplicate_execution(self, target: str, params: dict) -> str:
        self._inject_fault_via_http("scheduler", "duplicate_execution",
            "Scheduler duplicate execution mode — jobs running multiple times")
        self._write_service_log("scheduler", "error", "SchedulerService: duplicate execution detected across cron jobs")
        return "Injected: scheduler duplicate execution"

    def _inject_disk_full(self, target: str, params: dict) -> str:
        """Disk full — create a real large file to consume disk space.

        OS-LEVEL: Creates a physical file in /data/ that consumes real disk.
        On HF Spaces (Linux), uses fallocate for instant allocation.
        Size is controlled (50MB, not 10GB) to avoid killing the Space.
        Combined with HTTP flag so healthz returns 507.
        """
        junk_path = os.path.join(
            os.environ.get("DATA_DIR", "/data"), "_disk_full_junk.bin")
        try:
            if sys.platform != "win32":
                # Linux: fallocate creates a real file allocation instantly
                subprocess.run(
                    ["fallocate", "-l", "50M", junk_path],
                    timeout=5, capture_output=True)
            else:
                # Windows: write 50MB of zeros
                with open(junk_path, "wb") as f:
                    f.write(b"\x00" * (50 * 1024 * 1024))
            self._write_service_log("storage", "error",
                f"StorageService: disk allocation created at {junk_path} — physical disk consumed")
        except Exception as e:
            self._write_service_log("storage", "error", f"StorageService: disk fill failed: {e}")
        # Also set HTTP flag so healthz returns 507
        self._inject_fault_via_http("storage", "disk_full",
            "Storage disk full — writes rejected", severity="critical")
        self._write_service_log("storage", "error", "StorageService: disk usage reached 100%, rejecting write operations")
        return "Injected: storage disk full (physical file + HTTP 507)"

    def _inject_data_corruption(self, target: str, params: dict) -> str:
        """Data corruption — write garbage bytes to the SQLite database.

        OS-LEVEL: Physically corrupts the database file by appending random
        bytes. The next query from any service will get a real
        sqlite3.DatabaseError: database disk image is malformed.
        """
        db_path = self.database._db_path
        try:
            with open(db_path, "ab") as f:
                f.write(os.urandom(512))  # Append 512 random bytes
            self._write_service_log("storage", "error",
                f"StorageService: PHYSICAL data corruption — {db_path} modified with garbage bytes")
        except Exception as e:
            self._write_service_log("storage", "error", f"StorageService: corruption inject failed: {e}")
        self._inject_fault_via_http("storage", "data_corruption",
            "Storage data corruption — checksum mismatches detected")
        self._write_service_log("storage", "error", "StorageService: object corruption detected, checksum validation failed")
        return "Injected: storage data corruption (physical DB corruption + HTTP 503)"

    def _inject_scrape_failure(self, target: str, params: dict) -> str:
        """Metrics scrape failure — SIGSTOP the metrics_collector process."""
        pid = self._processes.get("metrics_collector", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("metrics_collector", "error",
                    "MetricsCollector: process frozen (SIGSTOP) — scrapes halted")
            except ProcessLookupError:
                pass
        self._inject_fault_via_http("metrics_collector", "scrape_failure",
            "Metrics collector scrape failures — telemetry stale, alerting blind spots")
        self._write_service_log("metrics_collector", "error", "MetricsCollector: scrape failures on all targets")
        return "Injected: metrics scrape failure (SIGSTOP)"

    def _inject_retention_full(self, target: str, params: dict) -> str:
        """Metrics retention full — fill the metrics data directory with junk."""
        metrics_dir = os.path.join(self.data_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        try:
            junk_path = os.path.join(metrics_dir, "_retention_full.bin")
            with open(junk_path, "wb") as f:
                f.write(b"\x00" * (5 * 1024 * 1024))  # 5MB junk file
        except OSError:
            pass
        self._inject_fault_via_http("metrics_collector", "retention_full",
            "Metrics retention full — dropping new datapoints", severity="critical")
        self._write_service_log("metrics_collector", "error",
            "MetricsCollector: retention store full (5MB junk file), dropping ingestion")
        return "Injected: metrics retention full (physical disk fill)"

    def _inject_smtp_down(self, target: str, params: dict) -> str:
        """SMTP down — SIGSTOP the email process to physically stop delivery.

        OS-LEVEL: Email process frozen. Notification service (which depends
        on email) will get real TCP timeout when trying to send.
        """
        pid = self._processes.get("email", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("email", "error",
                    "EmailService: process frozen by kernel (SIGSTOP) — SMTP delivery halted")
            except ProcessLookupError:
                pass
        else:
            self._inject_fault_via_http("email", "smtp_down",
                "SMTP upstream unavailable — email delivery queue backing up", severity="critical")
        self._write_service_log("email", "error", "EmailService: SMTP connection refused, queueing all outbound mail")
        return "Injected: email SMTP down (SIGSTOP)"

    def _inject_email_queue_overflow(self, target: str, params: dict) -> str:
        """Email queue overflow — write real backlog file to disk."""
        queue_file = os.path.join(self.data_dir, "email_queue.json")
        try:
            import json as _json
            backlog = [{"to": f"user{i}@example.com", "status": "queued"} for i in range(500)]
            with open(queue_file, "w") as f:
                _json.dump(backlog, f)
        except OSError:
            pass
        self._inject_fault_via_http("email", "email_queue_overflow",
            "Email queue overflow — messages dropped")
        self._write_service_log("email", "error",
            f"EmailService: queue overflow — 500 messages in {queue_file}")
        return "Injected: email queue overflow (500 msgs in physical backlog)"

    def _inject_billing_desync(self, target: str, params: dict) -> str:
        """Billing desync — insert unreconciled charges into the real database."""
        try:
            for i in range(10):
                self.database.execute(
                    "INSERT INTO payments (amount, user_id, status, created_at) "
                    "VALUES (?, ?, 'desync_unreconciled', datetime('now'))",
                    (round(99.99 + i, 2), f"billing_ghost_{i}"),
                )
        except Exception:
            pass
        self._inject_fault_via_http("billing", "billing_desync",
            "Billing desync — charges recorded but not reconciled")
        self._write_service_log("billing", "error",
            "BillingService: ledger desync — 10 unreconciled charges in DB")
        return "Injected: billing desync (10 ghost charges in DB)"

    def _inject_invoice_stuck(self, target: str, params: dict) -> str:
        """Invoice stuck — SIGSTOP the billing process."""
        pid = self._processes.get("billing", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("billing", "error",
                    "BillingService: process frozen (SIGSTOP) — invoices stuck")
            except ProcessLookupError:
                pass
        self._inject_fault_via_http("billing", "invoice_stuck",
            "Invoice generation stuck — invoices not being produced")
        self._write_service_log("billing", "error", "BillingService: invoice generation scheduler stuck")
        return "Injected: invoice generation stuck (SIGSTOP)"

    def _inject_config_poisoned(self, target: str, params: dict) -> str:
        """Config poisoned — write dangerous values to real config file."""
        config_file = os.path.join(self.data_dir, "config.json")
        try:
            import json as _json
            poisoned = {
                "rate_limit": 0, "queue_max": 5, "timeout_ms": 1,
                "debug_mode": True, "auth_bypass": True,
                "_poisoned": True, "_timestamp": time.time(),
            }
            with open(config_file, "w") as f:
                _json.dump(poisoned, f)
        except OSError:
            pass
        self._inject_fault_via_http("config", "config_poisoned",
            "Config store poisoned — critical keys set to unsafe values")
        self._write_service_log("config", "error",
            f"ConfigService: poisoned config written to {config_file} (rate_limit=0, auth_bypass=True)")
        return "Injected: config poisoned (physical config file corrupted)"

    def _inject_config_locked(self, target: str, params: dict) -> str:
        """Config locked — SIGSTOP the config process to physically block reads.

        OS-LEVEL: Config service is frozen. Any service trying to read config
        will get a real TCP timeout (not a mock 423).
        """
        pid = self._processes.get("config", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("config", "error",
                    "ConfigService: process frozen by kernel (SIGSTOP) — config reads blocked")
            except ProcessLookupError:
                pass
        else:
            self._inject_fault_via_http("config", "config_locked",
                "Config store locked — read/write operations blocked", severity="critical")
        self._write_service_log("config", "error", "ConfigService: config store lock engaged, updates blocked")
        return "Injected: config locked (SIGSTOP)"

    def _inject_dns_resolution_failure(self, target: str, params: dict) -> str:
        """DNS failure — SIGSTOP the DNS process so all DNS lookups physically timeout.

        OS-LEVEL: DNS process is frozen. Gateway, which depends on DNS,
        will get real ConnectTimeout when trying to resolve names.
        The death spiral emerges NATURALLY through TCP — no re-injection needed.
        """
        pid = self._processes.get("dns", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("dns", "error",
                    "DNSService: process frozen by kernel (SIGSTOP) — all lookups will timeout")
            except ProcessLookupError:
                pass
        else:
            self._inject_fault_via_http("dns", "dns_resolution_failure",
                "DNS resolution failure — lookups returning NXDOMAIN", severity="critical")
        self._write_service_log("dns", "error", "DNSService: resolution disabled, all lookups failing")
        return "Injected: DNS resolution failure (SIGSTOP)"

    def _inject_stale_entries(self, target: str, params: dict) -> str:
        """DNS stale entries — write stale host mappings to a real DNS cache file."""
        dns_cache = os.path.join(self.data_dir, "dns_cache.json")
        try:
            import json as _json
            stale = {
                "payment": "10.0.0.99",  # dead IP
                "auth": "10.0.0.98",
                "gateway": "192.168.0.1",  # wrong subnet
                "_ttl": 0, "_stale": True,
            }
            with open(dns_cache, "w") as f:
                _json.dump(stale, f)
        except OSError:
            pass
        self._inject_fault_via_http("dns", "stale_entries",
            "DNS stale entries — services routed to dead endpoints")
        self._write_service_log("dns", "error",
            f"DNSService: stale cache at {dns_cache} — serving dead IPs")
        return "Injected: DNS stale entries (physical cache file poisoned)"

    def _inject_all_backends_removed(self, target: str, params: dict) -> str:
        """All backends removed — SIGSTOP the loadbalancer to physically drop traffic."""
        pid = self._processes.get("loadbalancer", {}).get("pid")
        if pid and sys.platform != "win32":
            try:
                os.kill(pid, signal.SIGSTOP)
                self._write_service_log("loadbalancer", "error",
                    "LoadBalancer: process frozen (SIGSTOP) — all backends offline")
            except ProcessLookupError:
                pass
        self._inject_fault_via_http("loadbalancer", "all_backends_removed",
            "Load balancer has no healthy backends — traffic returning 503", severity="critical")
        self._write_service_log("loadbalancer", "error", "LoadBalancer: all backends removed from active pool")
        return "Injected: all backends removed (SIGSTOP)"

    def _inject_session_corruption(self, target: str, params: dict) -> str:
        """Session corruption — physically corrupt session rows in SQLite."""
        try:
            self.database.execute(
                "UPDATE sessions SET token = 'CORRUPTED_' || token, is_valid = 0"
            )
        except Exception:
            pass
        self._inject_fault_via_http("loadbalancer", "session_corruption",
            "Sticky-session corruption — users routed inconsistently")
        self._write_service_log("loadbalancer", "error",
            "LoadBalancer: sticky session table corrupted (DB rows modified)")
        return "Injected: session corruption (physical DB row corruption)"

    # ── Log Writing ──────────────────────────────────────────────────────

    def _write_service_log(self, service: str, level: str, message: str):
        """Write a log entry to a service's log file on disk."""
        import json as _json
        log_dir = os.path.join(self.log_dir, service)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{level}.log")
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": level.upper(),
            "service": service,
            "message": message,
        }
        try:
            with open(log_file, "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── Cascade Injection ────────────────────────────────────────────────

    def inject_cascade(self, primary_fault: str, cascade_fault: str, params: dict = None):
        """Arm a cascading failure — secondary fault triggers when primary is fixed.
        
        NOTE: The primary fault is already injected by _do_reset(). 
        This method only ARMS the cascade trigger, it does NOT re-inject the primary.
        """
        params = params or {}
        # Do NOT re-inject primary — it was already injected in _do_reset()
        self._armed_cascade = {
            "fault_type": cascade_fault,
            "params": params.get("cascade_params", {}),
            "trigger": f"primary_{primary_fault}_fixed",
            "primary_target": params.get("target", ""),  # track which service to watch
        }
        logger.info(f"Cascade armed: {primary_fault} → {cascade_fault}")

    def check_and_trigger_cascade(self) -> Optional[str]:
        """Check if cascade should trigger (primary fault resolved).
        
        Universal logic: if the primary target service is now healthy,
        the agent fixed the primary fault → trigger the cascade.
        """
        if not hasattr(self, "_armed_cascade") or not self._armed_cascade:
            return None

        cascade = self._armed_cascade

        # Universal check: is the primary fault's target service healthy now?
        primary_target = cascade.get("primary_target", "")
        primary_resolved = False
        
        if primary_target:
            health = self.check_health()
            svc_health = health.get(primary_target, {})
            if svc_health.get("status") == "healthy":
                primary_resolved = True
        else:
            # Fallback to legacy checks for backwards compatibility
            trigger = cascade.get("trigger", "")
            if "db_lock" in trigger and not self.database._is_fault_locked:
                primary_resolved = True
            elif "process_crash" in trigger:
                target = cascade["params"].get("target", "payment")
                if target not in self._crashed_services:
                    primary_resolved = True

        if primary_resolved:
            result = self.inject_fault(cascade["fault_type"], cascade["params"])
            self._armed_cascade = None
            logger.info(f"CASCADE TRIGGERED: {result}")
            return result
        return None

    # ── Health Check ─────────────────────────────────────────────────────

    def check_health(self) -> Dict[str, dict]:
        """Check health via REAL HTTP calls to each service process.

        Uses a single httpx.Client for all 16 checks to minimize
        connection overhead during training (called 3x per step).
        """
        import httpx

        health = {}
        with httpx.Client(timeout=2.0) as client:
            for name, port in SERVICE_PORTS.items():
                if name in self._crashed_services:
                    health[name] = {
                        "status": "crashed",
                        "degraded": False,
                        "error": f"Process killed \u2014 PID {self._processes.get(name, {}).get('pid', '?')}",
                        "error_rate": 1.0,
                        "latency_p95_ms": 0,
                        "requests_total": 0,
                        "cpu_percent": 0,
                        "memory_mb": 0,
                    }
                elif name in self._degraded_services:
                    # Service has a fault injected \u2014 check via REAL HTTP
                    try:
                        r = client.get(f"http://localhost:{port}/healthz")
                        if r.status_code == 200:
                            body = r.json()
                            health[name] = {
                                "status": body.get("status", "healthy"),
                                "degraded": body.get("status") != "healthy",
                                "error": body.get("error"),
                                "error_rate": body.get("error_rate", 0.0),
                                "latency_p95_ms": 0,
                                "requests_total": 0,
                                "cpu_percent": 0,
                                "memory_mb": 0,
                            }
                        else:
                            body = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
                            health[name] = {
                                "status": "degraded",
                                "degraded": True,
                                "error": body.get("error", self._degraded_services[name]),
                                "error_rate": body.get("error_rate", 0.8),
                                "latency_p95_ms": 5000,
                                "requests_total": 0,
                                "cpu_percent": 95,
                                "memory_mb": 450,
                            }
                    except (httpx.ConnectError, httpx.TimeoutException):
                        health[name] = {
                            "status": "crashed",
                            "degraded": False,
                            "error": "ConnectionRefused \u2014 process not running",
                            "error_rate": 1.0,
                            "latency_p95_ms": 0,
                            "requests_total": 0,
                            "cpu_percent": 0,
                            "memory_mb": 0,
                        }
                else:
                    try:
                        r = client.get(f"http://localhost:{port}/healthz")
                        status = "healthy" if r.status_code == 200 else "unhealthy"
                        health[name] = {
                            "status": status,
                            "degraded": status != "healthy",
                            "error": None if status == "healthy" else r.text[:100],
                            "error_rate": 0.0 if status == "healthy" else 0.5,
                            "latency_p95_ms": 0,
                            "requests_total": 0,
                            "cpu_percent": 0,
                            "memory_mb": 0,
                        }
                    except (httpx.ConnectError, httpx.TimeoutException):
                        # HTTP failed \u2014 is the process actually running?
                        proc_entry = self._processes.get(name, {})
                        proc = proc_entry.get("proc")
                        process_alive = proc is not None and proc.poll() is None

                        if process_alive:
                            health[name] = {
                                "status": "healthy",
                                "degraded": False,
                                "error": None,
                                "error_rate": 0.0,
                                "latency_p95_ms": 0,
                                "requests_total": 0,
                                "cpu_percent": 0,
                                "memory_mb": 0,
                            }
                        else:
                            health[name] = {
                                "status": "crashed",
                                "degraded": False,
                                "error": "Connection refused \u2014 process dead",
                                "error_rate": 1.0,
                                "latency_p95_ms": 0,
                                "requests_total": 0,
                                "cpu_percent": 0,
                                "memory_mb": 0,
                            }

        # ── Dynamic infrastructure health overlay ────────────────────
        # These catch faults that don't crash processes but degrade service:
        #   - Queue overflow → worker overwhelmed (even if process is alive)
        #   - DB lock → payment can't write (even if process is alive)
        # This replaces the need for explicit _degraded_services on these.

        if self.queue and self.queue.depth() > 500:
            if "worker" in health and health["worker"]["status"] == "healthy":
                depth = self.queue.depth()
                health["worker"] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": f"Queue depth {depth}/{self.queue._max_size} — worker overwhelmed",
                    "error_rate": 0.7,
                    "latency_p95_ms": 3000,
                    "requests_total": 0,
                    "cpu_percent": 90,
                    "memory_mb": 400,
                }

        if self.database and self.database._is_fault_locked:
            if "payment" in health and health["payment"]["status"] == "healthy":
                health["payment"] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": "Database locked — all writes failing",
                    "error_rate": 0.9,
                    "latency_p95_ms": 30000,
                    "requests_total": 0,
                    "cpu_percent": 10,
                    "memory_mb": 200,
                }

        # DB connection pool exhaustion → payment degraded
        if self.database and getattr(self.database, '_max_connections', 50) == 0:
            if "payment" in health and health["payment"]["status"] == "healthy":
                health["payment"] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": "DB connection pool exhausted — all queries failing",
                    "error_rate": 0.95,
                    "latency_p95_ms": 60000,
                    "requests_total": 0,
                    "cpu_percent": 5,
                    "memory_mb": 150,
                }

        # Queue paused → worker degraded
        if self.queue and getattr(self.queue, '_is_paused', False):
            if "worker" in health and health["worker"]["status"] == "healthy":
                health["worker"] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": "Queue consumer paused — messages accumulating, not being processed",
                    "error_rate": 0.5,
                    "latency_p95_ms": 5000,
                    "requests_total": 0,
                    "cpu_percent": 5,
                    "memory_mb": 100,
                }
        # Network latency injection → service degraded with measurable latency
        for svc_name, latency_ms in self._latency_injected.items():
            if svc_name in health and health[svc_name]["status"] == "healthy":
                health[svc_name] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": f"Network latency spike — p95={latency_ms}ms (baseline: 40ms)",
                    "error_rate": 0.3,
                    "latency_p95_ms": latency_ms,
                    "requests_total": 0,
                    "cpu_percent": 30,
                    "memory_mb": 200,
                }

        return health

    # ── Utilities ────────────────────────────────────────────────────────

    def get_process_info(self) -> Dict[str, dict]:
        """Return real PID and port info for all services."""
        return {
            name: {
                "pid": entry["pid"],
                "port": entry["port"],
                "alive": entry["proc"].poll() is None,
            }
            for name, entry in self._processes.items()
        }

    def shutdown(self):
        """Clean shutdown — kill ALL service processes."""
        self._running = False
        for name in list(self._processes.keys()):
            try:
                proc = self._processes[name]["proc"]
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2.0)
            except Exception:
                pass
        logger.info("CloudSRE orchestrator shutdown — all processes terminated")
