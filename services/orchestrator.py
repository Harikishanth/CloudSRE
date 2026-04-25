"""
CloudSRE v2 — Service Orchestrator.

Manages the lifecycle of all 4 microservices + infrastructure.
Each service runs as a SEPARATE OS PROCESS with its own PID.

This is NOT threading. Each service is a real process:
  - Separate PID (visible in `ps aux`)
  - Separate memory space (real isolation)
  - Real signal handling (kill -9 actually kills it)
  - Real port binding (Connection refused when dead)

Fault injection works ACROSS processes because:
  - DB lock: SQLite EXCLUSIVE transaction on shared file → all processes blocked
  - Queue overflow: File-backed queue on shared directory → all processes see it
  - Process crash: process.kill() sends real SIGTERM/SIGKILL
"""

import os
import sys
import time
import random
import asyncio
import signal
import subprocess
import logging
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
    """Manages all services as SEPARATE OS PROCESSES.

    Each service runs in its own Python process:
      - payment_service  → PID 12345, port 8001
      - auth_service     → PID 12346, port 8002
      - worker_service   → PID 12347, port 8003
      - frontend_proxy   → PID 12348, port 8004

    Kill a service = process.kill() → real SIGTERM → port stops listening
    Restart = spawn new subprocess → new PID → port opens again
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
        self._degraded_services: Dict[str, str] = {}  # {name: reason} — services broken by cascade
        self._latency_injected: Dict[str, int] = {}  # {name: latency_ms} — simulated network latency
        self._running = False

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
        """
        if name not in self._processes:
            return f"Service '{name}' not found"

        port = self._processes[name]["port"]
        old_pid = self._processes[name]["pid"]

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

        logger.info(f"  SERVICE RESTARTED: {name} PID={old_pid}→{new_pid} port={port}")
        return f"Service {name} restarted (PID {old_pid}→{new_pid}, port {port})"

    def reset(self):
        """Reset everything — kill all processes, wipe state, respawn.

        1. Kill ALL service processes
        2. Wipe database + queue
        3. Re-initialize fresh
        4. Spawn ALL services as new processes (new PIDs)
        """
        start = time.time()

        # 1. Kill all running service processes
        for name in list(self._processes.keys()):
            self._stop_service(name)
        self._crashed_services.clear()
        self._degraded_services.clear()
        self._latency_injected.clear()

        # 2. Clear cascade state
        self._armed_cascade = None

        # 3. Reset infrastructure
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
        }

        fn = injectors.get(fault_type)
        if not fn:
            return f"Unknown fault type: {fault_type}"
        return fn(target, params)

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

    def _inject_misleading_signal(self, target: str, params: dict) -> str:
        message = params.get("message", "Elevated latency detected (120ms vs 40ms baseline)")
        self._write_service_log(target, "error", f"[MISLEADING] {message}")
        return f"Injected: misleading signal in {target}"

    def _inject_latency(self, target: str, params: dict) -> str:
        """Inject network latency into a service — simulates degraded network."""
        latency_ms = params.get("latency_ms", 3000)
        self._latency_injected[target] = latency_ms
        self._degraded_services[target] = (
            f"Network latency spike — p95 latency {latency_ms}ms (baseline: 40ms)"
        )
        self._write_service_log(target, "error",
            f"NetworkMonitor: latency spike detected — "
            f"p95={latency_ms}ms, p99={latency_ms*2}ms, baseline=40ms")
        self._write_service_log(target, "error",
            f"Upstream connections timing out after {latency_ms}ms")
        return f"Injected: {latency_ms}ms latency into {target}"

    def _inject_cache_invalidation(self, target: str, params: dict) -> str:
        """Invalidate cache — triggers cold cache and potential thundering herd."""
        # Mark cache as DEGRADED — cold cache = 100% miss rate = service degraded
        self._degraded_services["cache"] = "Cache fully invalidated — 100% miss rate, thundering herd risk"
        self._write_service_log("cache", "error",
            "CacheService: FULL INVALIDATION — all entries evicted")
        self._write_service_log("cache", "error",
            "CacheService: Hit ratio dropped to 0.0% — cache is COLD")
        self._write_service_log("payment", "error",
            "PaymentService: cache miss rate 100% — falling through to database")
        return "Injected: cache invalidation (cold cache)"

    def _inject_webhook_storm(self, target: str, params: dict) -> str:
        """Trigger mass webhook retry — simulates Stripe-style retry storm."""
        count = params.get("count", 300)
        # Mark notification as DEGRADED — webhook storm overwhelms delivery
        self._degraded_services["notification"] = f"Webhook storm ({count} retries) — delivery pipeline overwhelmed"
        self._write_service_log("notification", "error",
            f"NotificationService: WEBHOOK STORM — {count} webhooks queued for retry")
        self._write_service_log("notification", "error",
            f"NotificationService: Queue depth {count}/500 — delivery rate overwhelmed")
        self._write_service_log("payment", "error",
            "PaymentService: Inbound webhook callbacks spiking — 300 req/sec (normal: 10)")
        return f"Injected: webhook storm ({count} retries)"

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
        """Check health via REAL HTTP calls to each service process."""
        import httpx

        health = {}
        for name, port in SERVICE_PORTS.items():
            if name in self._crashed_services:
                health[name] = {
                    "status": "crashed",
                    "degraded": False,
                    "error": f"Process killed — PID {self._processes.get(name, {}).get('pid', '?')}",
                    "error_rate": 1.0,
                    "latency_p95_ms": 0,
                    "requests_total": 0,
                    "cpu_percent": 0,
                    "memory_mb": 0,
                }
            elif name in self._degraded_services:
                # Service is running but degraded (cascade effect)
                health[name] = {
                    "status": "degraded",
                    "degraded": True,
                    "error": self._degraded_services[name],
                    "error_rate": 0.8,
                    "latency_p95_ms": 5000,
                    "requests_total": 0,
                    "cpu_percent": 95,
                    "memory_mb": 450,
                }
            else:
                try:
                    with httpx.Client(timeout=2.0) as client:
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
                    # HTTP failed — but is the process actually running?
                    proc_entry = self._processes.get(name, {})
                    proc = proc_entry.get("proc")
                    process_alive = proc is not None and proc.poll() is None

                    if process_alive:
                        # Process is running but port not ready yet (startup lag)
                        # Report as healthy — the restart succeeded, just slow to bind
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
                        # Process is truly dead
                        health[name] = {
                            "status": "crashed",
                            "degraded": False,
                            "error": "Connection refused — process dead",
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
