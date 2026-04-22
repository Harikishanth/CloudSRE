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
    "payment": 8001,
    "auth": 8002,
    "worker": 8003,
    "frontend": 8004,
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

        # Wait for all services to bind their ports
        time.sleep(1.0)
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

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Don't inherit parent's stdin (avoids blocking)
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
        time.sleep(0.5)  # Wait for port to bind

        new_pid = self._processes[name]["pid"]
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
        time.sleep(0.8)

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
        self._write_service_log("worker", "error",
            f"Queue depth critical: {self.queue.depth()}/{self.queue._max_size}")
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
        """Inject a cascading failure — primary + armed secondary."""
        params = params or {}
        self.inject_fault(primary_fault, params)
        self._armed_cascade = {
            "fault_type": cascade_fault,
            "params": params.get("cascade_params", {}),
            "trigger": f"primary_{primary_fault}_fixed",
        }
        logger.info(f"Cascade armed: {primary_fault} → {cascade_fault}")

    def check_and_trigger_cascade(self) -> Optional[str]:
        """Check if cascade should trigger (primary fault resolved)."""
        if not hasattr(self, "_armed_cascade") or not self._armed_cascade:
            return None

        cascade = self._armed_cascade
        primary_resolved = False
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
                    health[name] = {
                        "status": "crashed",
                        "degraded": False,
                        "error": "Connection refused",
                        "error_rate": 1.0,
                        "latency_p95_ms": 0,
                        "requests_total": 0,
                        "cpu_percent": 0,
                        "memory_mb": 0,
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
