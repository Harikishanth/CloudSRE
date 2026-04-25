"""
CloudSRE v2 — SRE Command Executor.

This is the equivalent of Kube SRE Gym's k8s_commands.py (719 lines).
Theirs handles kubectl commands against the K8s API.
Ours handles real SRE commands across MULTIPLE layers:

  Layer 1 — HTTP:      curl health checks, metrics, API calls
  Layer 2 — Database:  sqlite3 queries, table inspection
  Layer 3 — Logs:      cat, tail, head, grep on real log files
  Layer 4 — Process:   ps, kill, restart services
  Layer 5 — Queue:     queue inspection, drain, pause
  Layer 6 — Network:   service connectivity checks
  Layer 7 — Diagnosis: diagnose:/fix: statements (like Kube SRE Gym)

Their agent can only run kubectl. Our agent operates at the APPLICATION layer —
where real SREs spend 90% of their time.
"""

import os
import re
import time
import signal
import subprocess
import shlex
import json
import sqlite3
import threading
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum output length to prevent observation explosion
MAX_OUTPUT_CHARS = 2000

# ── Cloud DNS Mapping ────────────────────────────────────────────────────
# Maps cloud-style internal domains to localhost ports.
# The agent types:  curl http://payment.us-east-1.internal/healthz
# We route it to:   curl http://localhost:8001/healthz
# This makes the environment look like a real multi-region cloud deployment
# while keeping the 30x speed advantage of local OS processes.

CLOUD_DNS = {
    # US-EAST-1 (Primary region)
    "payment.us-east-1.internal": "localhost:8001",
    "auth.us-east-1.internal": "localhost:8002",
    "billing.us-east-1.internal": "localhost:8013",
    "gateway.us-east-1.internal": "localhost:8008",
    "loadbalancer.us-east-1.internal": "localhost:8016",
    "config.us-east-1.internal": "localhost:8014",
    # EU-WEST-1 (Compute region)
    "worker.eu-west-1.internal": "localhost:8003",
    "scheduler.eu-west-1.internal": "localhost:8009",
    "search.eu-west-1.internal": "localhost:8007",
    "storage.eu-west-1.internal": "localhost:8010",
    "metrics_collector.eu-west-1.internal": "localhost:8011",
    # AP-SOUTH-1 (Edge region)
    "frontend.ap-south-1.internal": "localhost:8004",
    "cache.ap-south-1.internal": "localhost:8005",
    "notification.ap-south-1.internal": "localhost:8006",
    "email.ap-south-1.internal": "localhost:8012",
    "dns.ap-south-1.internal": "localhost:8015",
}

# Reverse map: localhost:port -> cloud domain (for output beautification)
CLOUD_DNS_REVERSE = {v: k for k, v in CLOUD_DNS.items()}

# Region grouping for status dashboard
CLOUD_REGIONS = {
    "us-east-1": [
        ("payment", 8001), ("auth", 8002), ("billing", 8013),
        ("gateway", 8008), ("loadbalancer", 8016), ("config", 8014),
    ],
    "eu-west-1": [
        ("worker", 8003), ("scheduler", 8009), ("search", 8007),
        ("storage", 8010), ("metrics_collector", 8011),
    ],
    "ap-south-1": [
        ("frontend", 8004), ("cache", 8005), ("notification", 8006),
        ("email", 8012), ("dns", 8015),
    ],
}


class CommandExecutor:
    """Executes real SRE commands against a simulated multi-region cloud infrastructure.

    When the agent sends a command like:
        curl http://payment.us-east-1.internal/healthz
    This class resolves the cloud domain to localhost:8001 and executes
    the REAL HTTP request against the running service process.

    Supports both cloud-style domains and raw localhost URLs.
    """

    # All known service names
    KNOWN_SERVICES = {
        "payment", "auth", "worker", "frontend", "cache", "notification",
        "search", "gateway", "scheduler", "storage", "metrics_collector",
        "email", "billing", "config", "dns", "loadbalancer",
    }

    def __init__(self, services: Dict[str, Any], infra: Dict[str, Any], orchestrator=None):
        """
        Args:
            services: {"payment": service_instance, "auth": ..., "worker": ..., "frontend": ...}
            infra: {"database": db_instance, "queue": queue_instance}
            orchestrator: ServiceOrchestrator for real restart/stop operations
        """
        self.services = services
        self.infra = infra
        self.orchestrator = orchestrator  # For real service restart
        self._command_count = 0
        self._error_count = 0
        self._command_type_counts = {}  # Track command types for diminishing returns

    def execute(self, command: str) -> Tuple[str, str]:
        """Execute a command and return (output, command_type).

        The command_type is used by the phase detector to determine
        what SRE workflow phase the agent is in.

        Returns:
            (output_string, command_type) where command_type is one of:
            "health_check", "metrics", "logs", "database", "process",
            "queue", "fix", "diagnosis", "unknown"
        """
        cmd = command.strip()
        if not cmd:
            return "Error: empty command", "unknown"

        self._command_count += 1

        try:
            # Route to the appropriate handler
            # Real Linux commands that SREs actually use
            if cmd.startswith("curl "):
                return self._handle_curl(cmd), self._classify_curl(cmd)
            elif cmd.startswith("cat ") or cmd.startswith("tail ") or cmd.startswith("head "):
                return self._handle_log_read(cmd), "logs"
            elif cmd.startswith("grep "):
                return self._handle_grep(cmd), "logs"
            elif cmd.startswith("sqlite3 "):
                return self._handle_sqlite(cmd), "database"
            elif cmd.startswith("kill "):
                return self._handle_kill(cmd), "fix"
            elif cmd.startswith("systemctl restart ") or cmd.startswith("restart_service "):
                return self._handle_restart(cmd), "fix"
            elif cmd.startswith("systemctl stop "):
                return self._handle_kill(cmd), "fix"
            elif cmd.startswith("systemctl status"):
                return self._handle_status(), "health_check"
            elif cmd.startswith("ps ") or cmd == "ps":
                return self._handle_ps(cmd), "diagnosis"
            elif cmd.startswith("ls "):
                return self._handle_ls(cmd), "diagnosis"
            elif cmd.startswith("diagnose:") or cmd.startswith("diagnosis:"):
                return self._handle_diagnosis(cmd), "diagnosis"
            elif cmd.startswith("fix:"):
                return self._handle_fix_statement(cmd), "fix"
            elif cmd.startswith("queue ") or cmd.startswith("drain "):
                return self._handle_queue(cmd), "fix"
            elif cmd.startswith("config "):
                return self._handle_config(cmd), "fix"
            elif cmd == "services" or cmd == "status":
                return self._handle_status(), "health_check"
            else:
                self._error_count += 1
                return self._unknown_command(cmd), "unknown"
        except Exception as e:
            self._error_count += 1
            logger.error(f"Command execution error: {e}", exc_info=True)
            return f"ERROR: {str(e)}", "unknown"

    # ── Layer 1: HTTP (curl) ─────────────────────────────────────────────

    def _handle_curl(self, cmd: str) -> str:
        """Execute a real HTTP request against running services.

        Supports:
            curl http://localhost:8001/healthz
            curl http://localhost:8001/metrics
            curl -X POST http://localhost:8003/queue/drain?rate=10
            curl http://localhost:8002/auth/verify -H "Authorization: Bearer <token>"
        """
        import httpx

        # Parse URL and method from curl command
        url = self._extract_url(cmd)
        method = "POST" if "-X POST" in cmd or "--data" in cmd else "GET"
        headers = self._extract_headers(cmd)
        data = self._extract_data(cmd)

        if not url:
            return "Error: could not parse URL from curl command"

        try:
            with httpx.Client(timeout=10.0) as client:
                if method == "POST":
                    response = client.post(url, headers=headers, content=data)
                else:
                    response = client.get(url, headers=headers)

                # Format output like real curl
                output = f"HTTP/{response.http_version} {response.status_code}\n"
                try:
                    body = response.json()
                    output += json.dumps(body, indent=2)
                except Exception:
                    output += response.text

                return self._truncate(output)

        except httpx.ConnectError:
            # REAL connection refused — service is down
            return f"curl: (7) Failed to connect to {url}: Connection refused"
        except httpx.TimeoutException:
            return f"curl: (28) Connection timed out after 10 seconds"
        except Exception as e:
            return f"curl: error — {str(e)}"

    def _classify_curl(self, cmd: str) -> str:
        """Classify a curl command for phase detection."""
        if "/healthz" in cmd or "/health" in cmd:
            return "health_check"
        elif "/metrics" in cmd:
            return "metrics"
        elif "/logs" in cmd:
            return "logs"
        elif "POST" in cmd or "drain" in cmd or "restart" in cmd:
            return "fix"
        return "health_check"

    def _extract_url(self, cmd: str) -> Optional[str]:
        """Extract URL from a curl command.

        Handles cloud domain interception:
            http://payment.us-east-1.internal/healthz -> http://localhost:8001/healthz
        Also accepts raw localhost URLs for backward compatibility.
        """
        match = re.search(r'(https?://[^\s"\']+)', cmd)
        if not match:
            return None
        url = match.group(1)

        # Cloud DNS interception: resolve fake domains to localhost
        for cloud_domain, local_addr in CLOUD_DNS.items():
            if cloud_domain in url:
                url = url.replace(cloud_domain, local_addr)
                break

        return url

    def _extract_headers(self, cmd: str) -> Dict[str, str]:
        """Extract -H headers from curl command."""
        headers = {}
        for match in re.finditer(r'-H\s+"([^"]+)"', cmd):
            header = match.group(1)
            if ":" in header:
                key, value = header.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers

    def _extract_data(self, cmd: str) -> Optional[str]:
        """Extract -d/--data body from curl command."""
        match = re.search(r'(?:-d|--data)\s+"([^"]*)"', cmd)
        if match:
            return match.group(1)
        match = re.search(r"(?:-d|--data)\s+'([^']*)'", cmd)
        return match.group(1) if match else None

    # ── Layer 2: Database (sqlite3) ──────────────────────────────────────

    def _handle_sqlite(self, cmd: str) -> str:
        """Execute a real SQL query on the real database.

        Supports:
            sqlite3 /data/app.db 'SELECT count(*) FROM payments'
            sqlite3 /data/app.db 'SELECT * FROM payments WHERE status="pending" LIMIT 10'
            sqlite3 /data/app.db '.tables'
            sqlite3 /data/app.db '.schema payments'
        """
        db = self.infra.get("database")
        if not db:
            return "Error: database not initialized"

        # Extract SQL from command
        sql = self._extract_sql(cmd)
        if not sql:
            return "Error: could not parse SQL from sqlite3 command"

        # Handle dot commands
        if sql.startswith("."):
            return self._handle_sqlite_dot(sql, db)

        # Safety: block destructive operations
        sql_upper = sql.upper().strip()
        if any(sql_upper.startswith(kw) for kw in ["DROP", "DELETE", "TRUNCATE", "ALTER"]):
            return "Error: destructive SQL operations are not permitted"

        try:
            if sql_upper.startswith("SELECT") or sql_upper.startswith("PRAGMA"):
                rows = db.query(sql)
                if not rows:
                    return "(no results)"
                # Format as table
                headers = list(rows[0].keys())
                lines = [" | ".join(headers)]
                lines.append("-" * len(lines[0]))
                for row in rows[:50]:  # Limit to 50 rows
                    lines.append(" | ".join(str(row.get(h, "")) for h in headers))
                return self._truncate("\n".join(lines))
            else:
                # INSERT/UPDATE — execute and return affected rows
                rowid = db.execute(sql)
                return f"OK (last_row_id={rowid})"

        except sqlite3.OperationalError as e:
            # REAL database error — agent sees the actual error
            return f"Error: {str(e)}"

    def _extract_sql(self, cmd: str) -> Optional[str]:
        """Extract SQL from sqlite3 command."""
        # Match single-quoted or double-quoted SQL
        match = re.search(r"""sqlite3\s+\S+\s+['"](.*?)['"]""", cmd, re.DOTALL)
        if match:
            return match.group(1)
        # Match unquoted SQL (everything after db path)
        match = re.search(r'sqlite3\s+\S+\s+(.*)', cmd)
        return match.group(1).strip() if match else None

    def _handle_sqlite_dot(self, dot_cmd: str, db) -> str:
        """Handle SQLite dot commands (.tables, .schema, etc.)."""
        if dot_cmd == ".tables":
            rows = db.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return " ".join(r["name"] for r in rows) if rows else "(no tables)"
        elif dot_cmd.startswith(".schema"):
            parts = dot_cmd.split()
            table = parts[1] if len(parts) > 1 else None
            if table:
                rows = db.query("SELECT sql FROM sqlite_master WHERE name=?", (table,))
            else:
                rows = db.query("SELECT sql FROM sqlite_master WHERE type='table'")
            return "\n".join(r["sql"] for r in rows if r["sql"]) if rows else "(no schema)"
        return f"Error: unknown dot command '{dot_cmd}'"

    # ── Layer 3: Logs (cat/tail/head/grep) ───────────────────────────────

    def _handle_log_read(self, cmd: str) -> str:
        """Read real log files from real disk.

        Supports:
            cat /var/log/payment/error.log
            tail -20 /var/log/auth/error.log
            head -5 /var/log/worker/access.log
            cat /var/log/payment/error.log | tail -10
        """
        # Parse the file path and line count
        parts = cmd.split()
        verb = parts[0]  # cat, tail, head

        # Handle pipe: cat ... | tail -N
        if "|" in cmd:
            pipe_parts = cmd.split("|")
            file_path = self._extract_path(pipe_parts[0])
            tail_match = re.search(r'tail\s+-(\d+)', pipe_parts[1])
            line_count = int(tail_match.group(1)) if tail_match else 20
            verb = "tail"
        else:
            file_path = self._extract_path(cmd)
            line_match = re.search(r'-(\d+)', cmd)
            line_count = int(line_match.group(1)) if line_match else None

        if not file_path:
            return "Error: could not parse file path"

        # Security: only allow reading from /var/log/
        if not file_path.startswith("/var/log/"):
            return f"Error: access denied — can only read files under /var/log/"

        try:
            with open(file_path, "r") as f:
                all_lines = f.readlines()

            if not all_lines:
                return "(empty log file)"

            if verb == "tail":
                lines = all_lines[-(line_count or 20):]
            elif verb == "head":
                lines = all_lines[:(line_count or 10)]
            else:  # cat
                lines = all_lines[-(line_count or 50):]  # Default: last 50 for cat

            return self._truncate("".join(lines))
        except FileNotFoundError:
            return f"cat: {file_path}: No such file or directory"
        except PermissionError:
            return f"cat: {file_path}: Permission denied"

    def _handle_grep(self, cmd: str) -> str:
        """Grep through log files.

        Supports:
            grep "ERROR" /var/log/payment/error.log
            grep -i "timeout" /var/log/auth/error.log
            grep -c "database" /var/log/payment/error.log  (count)
        """
        # Parse pattern and file
        case_insensitive = "-i" in cmd
        count_only = "-c" in cmd

        # Extract pattern (quoted)
        pattern_match = re.search(r'"([^"]+)"', cmd)
        if not pattern_match:
            pattern_match = re.search(r"'([^']+)'", cmd)
        if not pattern_match:
            return "Error: pattern required (grep \"PATTERN\" FILE)"

        pattern = pattern_match.group(1)
        file_path = self._extract_path(cmd.replace(f'"{pattern}"', "").replace(f"'{pattern}'", ""))

        if not file_path or not file_path.startswith("/var/log/"):
            return "Error: valid log file path required"

        try:
            with open(file_path, "r") as f:
                lines = f.readlines()

            if case_insensitive:
                matches = [l for l in lines if pattern.lower() in l.lower()]
            else:
                matches = [l for l in lines if pattern in l]

            if count_only:
                return str(len(matches))

            if not matches:
                return "(no matches)"
            return self._truncate("".join(matches[-30:]))  # Last 30 matches
        except FileNotFoundError:
            return f"grep: {file_path}: No such file or directory"

    # ── Layer 4: Process Management (ps/kill/restart) ────────────────────

    def _handle_ps(self, cmd: str) -> str:
        """List running service processes using REAL process info.

        Each service runs as a SEPARATE OS PROCESS. Uses psutil to get
        real PID, CPU, and memory for each subprocess.

        Supports:
            ps aux
            ps aux | grep payment
        """
        import psutil

        service_info = []
        processes = getattr(self.orchestrator, '_processes', {}) if self.orchestrator else {}
        crashed_set = getattr(self.orchestrator, '_crashed_services', set()) if self.orchestrator else set()

        for name, entry in processes.items():
            pid = entry["pid"]
            port = entry["port"]

            if name in crashed_set:
                # Process was killed — REALLY dead
                service_info.append({
                    "user": "root",
                    "pid": str(pid),
                    "cpu": "0.0",
                    "mem": "0.0",
                    "vsz": "0",
                    "rss": "0",
                    "stat": "DEAD",
                    "command": f"[{name}_service:{port}] CRASHED (was PID {pid})",
                })
            else:
                # Get REAL metrics from the actual subprocess
                try:
                    proc = psutil.Process(pid)
                    mem_info = proc.memory_info()
                    cpu = proc.cpu_percent(interval=0.05)
                    rss_mb = mem_info.rss / (1024 * 1024)
                    vsz_mb = mem_info.vms / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss_mb = 0
                    vsz_mb = 0
                    cpu = 0

                service_info.append({
                    "user": "root",
                    "pid": str(pid),
                    "cpu": f"{cpu:.1f}",
                    "mem": f"{rss_mb:.1f}",
                    "vsz": str(int(vsz_mb * 1024)),
                    "rss": str(int(rss_mb * 1024)),
                    "stat": "Sl",
                    "command": f"python uvicorn {name}_service --port {port}",
                })

        # If grep filter is present
        if "|" in cmd and "grep" in cmd:
            grep_match = re.search(r'grep\s+(\S+)', cmd)
            if grep_match:
                filter_term = grep_match.group(1).lower()
                service_info = [s for s in service_info if filter_term in s["command"].lower()]

        lines = ["USER       PID   %CPU  %MEM    VSZ   RSS  STAT COMMAND"]
        for s in service_info:
            lines.append(
                f"{s['user']:8s}  {s['pid']:>5s}  {s['cpu']:>5s} {s['mem']:>5s}  "
                f"{s['vsz']:>5s} {s['rss']:>5s}  {s['stat']:4s} {s['command']}"
            )
        return "\n".join(lines)

    def _handle_ls(self, cmd: str) -> str:
        """List real files/directories — delegates to actual filesystem.

        Supports:
            ls /data/queue/
            ls /var/log/payment/
            ls /data/queue/ | wc -l
        """
        path_match = re.search(r'ls\s+(-\S+\s+)?(/\S+)', cmd)
        if not path_match:
            return "Error: could not parse path from ls command"

        path = path_match.group(2)
        if not os.path.exists(path):
            return f"ls: cannot access '{path}': No such file or directory"

        try:
            entries = os.listdir(path)
            if "|" in cmd and "wc" in cmd:
                return str(len(entries))
            return "\n".join(sorted(entries)) if entries else "(empty directory)"
        except PermissionError:
            return f"ls: cannot open directory '{path}': Permission denied"

    def _match_service_name(self, cmd: str) -> Optional[str]:
        """Extract service name from a command string.
        
        Works in both in-process mode (self.services populated) and
        subprocess mode (self.services empty, use KNOWN_SERVICES).
        """
        cmd_lower = cmd.lower()
        # First try self.services (in-process mode)
        for name in self.services:
            if name in cmd_lower:
                return name
        # Fallback to known service names (subprocess mode)
        for name in self.KNOWN_SERVICES:
            if name in cmd_lower:
                return name
        return None

    def _handle_kill(self, cmd: str) -> str:
        """Kill a service — ACTUALLY stops the uvicorn server.

        Supports:
            kill -9 <pid>
            kill payment
            systemctl stop payment

        This REALLY stops the server. The port stops listening.
        curl will get 'Connection refused' after this.
        """
        service_name = self._match_service_name(cmd)

        if not service_name:
            pid_match = re.search(r'(?:kill|stop)\s+(?:-\d+\s+)?(\d+)', cmd)
            if pid_match:
                return "Error: ambiguous PID — use service name (e.g., kill payment)"
            return "Error: could not identify target service/process"

        # ACTUALLY stop the uvicorn server via orchestrator (subprocess mode)
        if self.orchestrator and hasattr(self.orchestrator, '_stop_service'):
            self.orchestrator._stop_service(service_name)
            return f"Killed {service_name} service — port no longer listening"

        # Fallback: in-process mode
        svc = self.services.get(service_name)
        if svc:
            svc.set_unhealthy(f"Process killed by agent")
            return f"Killed {service_name} service"
        return f"Error: service '{service_name}' not found"

    def _handle_restart(self, cmd: str) -> str:
        """Restart a service — ACTUALLY stops and restarts the uvicorn server.

        Supports:
            restart_service payment
            systemctl restart payment

        This is a REAL restart:
          1. The uvicorn server is stopped (port closes)
          2. Service state is reset
          3. A new uvicorn server starts (port opens again)
          4. If the service used the DB, the DB lock is released
        """
        service_name = self._match_service_name(cmd)

        if not service_name:
            return "Error: could not identify target service"

        # Release DB lock on restart — any service restart should clear
        # the shared DB lock since the lock holder is being killed
        db = self.infra.get("database")
        if db and getattr(db, '_is_fault_locked', False):
            db.release_lock()

        # Release queue pause when restarting worker — worker is the consumer
        queue = self.infra.get("queue")
        if queue and service_name == "worker" and getattr(queue, '_is_paused', False):
            queue.release_pause()

        # Use orchestrator for REAL restart (uvicorn stop + start)
        if self.orchestrator and hasattr(self.orchestrator, 'restart_service'):
            result = self.orchestrator.restart_service(service_name)
            return result

        # Fallback: in-memory reset only (for tests without orchestrator)
        svc = self.services.get(service_name)
        if svc and hasattr(svc, "reset"):
            svc.reset()
            svc.set_healthy()
            svc.logger.info(f"Service restarted by agent")
            return f"Restarted {service_name} service — now healthy"
        return f"Error: service '{service_name}' not found"

    # ── Layer 5: Queue Management ────────────────────────────────────────

    def _handle_queue(self, cmd: str) -> str:
        """Queue inspection and management.

        Supports:
            queue status
            queue depth
            queue drain <N>         (drain N messages at controlled rate — SRE chooses N)
            queue drain all         (DANGEROUS — thundering herd!)
            queue pause
            queue resume
            drain rate=<N>          (controlled drain alias)
        """
        queue = self.infra.get("queue")
        if not queue:
            return "Error: queue not initialized"

        parts = cmd.lower().split()

        if "status" in parts or cmd.strip() == "queue":
            metrics = queue.get_metrics()
            return json.dumps(metrics, indent=2)
        elif "depth" in parts:
            return f"Queue depth: {queue.depth()}"
        elif "drain" in parts:
            if "all" in parts:
                messages = queue.drain_all()
                # drain-all is DANGEROUS — overwhelms worker (thundering herd)
                # Mark worker degraded explicitly since queue.depth() is now 0
                # but the worker is OOM from processing everything at once
                if self.orchestrator and len(messages) > 50:
                    self.orchestrator._degraded_services["worker"] = \
                        f"Thundering herd — {len(messages)} messages processed at once, worker OOM"
                return f"WARNING: Drained {len(messages)} messages at once (thundering herd risk!)"
            else:
                rate_match = re.search(r'(\d+)', cmd)
                rate = int(rate_match.group(1)) if rate_match else 200
                messages = queue.drain_controlled(rate=rate)
                # No need to explicitly clear degradation — check_health()
                # dynamically checks queue.depth() and clears when < 500
                return f"Drained {len(messages)} messages at controlled rate={rate} (remaining: {queue.depth()})"
        elif "pause" in parts:
            queue.inject_pause()
            return "Queue paused — consumer will not accept new messages"
        elif "resume" in parts:
            queue.release_pause()
            return "Queue resumed — consumer accepting messages"
        elif "dead" in parts or "deadletter" in parts:
            return f"Dead letter queue: {queue.dead_letter_count()} messages"
        return f"Error: unknown queue command '{cmd}'"

    # ── Layer 6: Configuration ───────────────────────────────────────────

    def _handle_config(self, cmd: str) -> str:
        """Service configuration changes.

        Supports:
            config payment rate_limit=100
            config auth jwt_ttl=3600
            config worker batch_size=10

        Works in both in-process and subprocess modes.
        """
        parts = cmd.split()
        if len(parts) < 3:
            return "Usage: config <service> <key=value>"

        service_name = parts[1]
        config_str = parts[2]

        if "=" not in config_str:
            return f"Error: invalid config format. Use: config {service_name} key=value"

        key, value = config_str.split("=", 1)

        # In-process mode: direct service access
        svc = self.services.get(service_name)
        if svc:
            if not hasattr(svc, "_config"):
                svc._config = {}
            svc._config[key] = value
            svc.logger.info(f"Config updated: {key}={value}")
            return f"Config updated: {service_name}.{key} = {value}"

        # Subprocess mode: validate service name and apply via orchestrator
        if service_name not in self.KNOWN_SERVICES:
            return f"Error: service '{service_name}' not found"

        # Config changes can clear degraded state for config-related faults
        if self.orchestrator:
            degraded = getattr(self.orchestrator, '_degraded_services', {})
            if service_name in degraded:
                reason = degraded[service_name].lower()
                if any(kw in reason for kw in ["config", "rate_limit", "poisoned", "locked"]):
                    degraded.pop(service_name, None)

        return f"Config updated: {service_name}.{key} = {value}"

    # ── Layer 7: Diagnosis/Fix Statements ────────────────────────────────

    def _handle_diagnosis(self, cmd: str) -> str:
        """Handle free-text diagnosis statements (like Kube SRE Gym).

        Supports:
            diagnose: the root cause is a database lock in the payment service
            diagnosis: auth service JWT token expired causing 401 cascade
        """
        diagnosis_text = cmd.split(":", 1)[1].strip() if ":" in cmd else cmd
        return f"Diagnosis recorded: {diagnosis_text}"

    def _handle_fix_statement(self, cmd: str) -> str:
        """Handle free-text fix statements (like Kube SRE Gym).

        Supports:
            fix: restart payment service and drain queue at rate=10
            fix: rotate JWT signing key in auth service

        Attempts to extract and execute actionable commands from the text.
        """
        fix_text = cmd.split(":", 1)[1].strip() if ":" in cmd else cmd
        fix_lower = fix_text.lower()
        results = [f"Fix recorded: {fix_text}"]

        # Attempt to extract and execute embedded actions
        service_name = self._match_service_name(fix_text)

        if "restart" in fix_lower and service_name:
            restart_result = self._handle_restart(f"systemctl restart {service_name}")
            results.append(f"  -> {restart_result}")

        if "drain" in fix_lower:
            rate_match = re.search(r'rate[=\s]*(\d+)', fix_lower)
            rate = rate_match.group(1) if rate_match else "10"
            drain_result = self._handle_queue(f"queue drain {rate}")
            results.append(f"  -> {drain_result}")

        if "resume" in fix_lower and "queue" in fix_lower:
            resume_result = self._handle_queue("queue resume")
            results.append(f"  -> {resume_result}")

        return "\n".join(results)

    # ── Status Overview ──────────────────────────────────────────────────

    def _handle_status(self) -> str:
        """Cloud-style service health dashboard grouped by region.

        Makes REAL HTTP requests to each service's /healthz endpoint.
        Displays results as a multi-region cloud dashboard.
        """
        import httpx

        lines = []
        lines.append("=" * 72)
        lines.append("  CLOUD SERVICE DASHBOARD")
        lines.append("=" * 72)
        lines.append(f"{'REGION':<14} {'SERVICE':<20} {'STATUS':<12} {'HEALTH'}")
        lines.append("-" * 72)

        for region, services in CLOUD_REGIONS.items():
            for name, port in services:
                try:
                    with httpx.Client(timeout=2.0) as client:
                        r = client.get(f"http://localhost:{port}/healthz")
                        if r.status_code == 200:
                            lines.append(f"{region:<14} {name:<20} {'RUNNING':<12} Healthy")
                        else:
                            body = r.text[:40].replace('\n', ' ')
                            lines.append(f"{region:<14} {name:<20} {'DEGRADED':<12} HTTP {r.status_code}: {body}")
                except httpx.ConnectError:
                    lines.append(f"{region:<14} {name:<20} {'DOWN':<12} ConnectionRefused")
                except httpx.TimeoutException:
                    lines.append(f"{region:<14} {name:<20} {'TIMEOUT':<12} NoResponse")
                except Exception as e:
                    lines.append(f"{region:<14} {name:<20} {'ERROR':<12} {str(e)[:30]}")

        # Infrastructure
        lines.append("-" * 72)
        db = self.infra.get("database")
        if db:
            m = db.get_metrics()
            db_status = 'LOCKED' if m['db_is_locked'] else 'ACTIVE'
            lines.append(f"{'infra':<14} {'rds-primary':<20} {db_status:<12} errors={m['db_errors_total']} queries={m['db_queries_total']}")

        queue = self.infra.get("queue")
        if queue:
            m = queue.get_metrics()
            q_status = 'PAUSED' if m['queue_is_paused'] else 'ACTIVE'
            lines.append(f"{'infra':<14} {'sqs-main':<20} {q_status:<12} depth={m['queue_depth']}/{m['queue_max_size']}")

        lines.append("=" * 72)
        return "\n".join(lines)
    # ── Utilities ────────────────────────────────────────────────────────

    def _extract_path(self, cmd: str) -> Optional[str]:
        """Extract a file path from a command."""
        match = re.search(r'(/\S+\.(?:log|txt|json|db|csv))', cmd)
        if match:
            return match.group(1)
        match = re.search(r'(/var/log/\S+)', cmd)
        return match.group(1) if match else None

    def _truncate(self, text: str) -> str:
        """Truncate output to MAX_OUTPUT_CHARS."""
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(text)} total chars)"

    def _unknown_command(self, cmd: str) -> str:
        """Return helpful error for unknown commands."""
        return (
            f"bash: {cmd.split()[0]}: command not found\n"
            "Available commands:\n"
            "  curl http://<svc>.<region>.internal/healthz   -- service health check\n"
            "  curl http://<svc>.<region>.internal/metrics   -- service metrics\n"
            "  cat /var/log/<service>/error.log              -- read error logs\n"
            "  sqlite3 /data/app.db 'SELECT ...'             -- query RDS\n"
            "  restart_service <service>                     -- restart service\n"
            "  status                                        -- cloud dashboard\n"
            "  queue status|depth|drain                      -- SQS management\n"
            "  diagnose: <text>                              -- record diagnosis\n"
            "  fix: <text>                                   -- apply fix\n"
            "\nRegions: us-east-1, eu-west-1, ap-south-1\n"
            "Example: curl http://payment.us-east-1.internal/healthz\n"
        )

    def get_metrics(self) -> Dict[str, int]:
        """Executor metrics."""
        return {
            "commands_executed": self._command_count,
            "command_errors": self._error_count,
        }
