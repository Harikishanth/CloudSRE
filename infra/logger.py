"""
CloudSRE v2 — Structured JSON Logger Infrastructure.

Real structured logging that writes to real log files on disk.
The agent reads these via `cat /var/log/<service>/error.log | tail -20`.

Each service gets its own log directory:
  /var/log/payment/  → error.log, access.log
  /var/log/auth/     → error.log, access.log
  /var/log/worker/   → error.log, access.log
  /var/log/frontend/ → error.log, access.log

Misleading signals are injected by writing FAKE error lines to the wrong
service's log file. The agent must cross-reference logs with metrics to
distinguish real errors from red herrings.

Kube SRE Gym equivalent: They have container stdout. We have structured
JSON logs on disk — closer to real production logging (ELK/Splunk/Datadog).
"""

import json
import time
import os
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone


class StructuredLogger:
    """Real structured JSON logger that writes to real files.

    Each log line is a JSON object:
    {
        "timestamp": "2024-03-15T02:47:12.345Z",
        "level": "ERROR",
        "service": "payment",
        "message": "Database connection timeout after 30s",
        "context": {"connection_id": 47, "query": "INSERT INTO payments..."}
    }

    Usage:
        logger = StructuredLogger("payment", "/var/log/payment")
        logger.error("DB connection timeout", {"conn_id": 47})
        logger.info("Request processed", {"amount": 100.0})

        # Agent reads:
        # cat /var/log/payment/error.log | tail -5
    """

    def __init__(self, service_name: str, log_dir: str = "/var/log"):
        self.service_name = service_name
        self.log_dir = os.path.join(log_dir, service_name)
        self._lock = threading.Lock()
        self._log_count = 0

        # Create log directory and files
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Two log files: error.log (ERROR + WARN) and access.log (INFO)
        self._error_path = os.path.join(self.log_dir, "error.log")
        self._access_path = os.path.join(self.log_dir, "access.log")

        # Initialize empty log files
        for path in [self._error_path, self._access_path]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    pass

    def _write(self, level: str, message: str, context: Optional[Dict] = None):
        """Write a structured log line to the appropriate file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": level,
            "service": self.service_name,
            "message": message,
        }
        if context:
            entry["context"] = context

        line = json.dumps(entry) + "\n"

        with self._lock:
            # ERROR and WARN go to error.log, INFO goes to access.log
            target = self._error_path if level in ("ERROR", "WARN") else self._access_path
            with open(target, "a") as f:
                f.write(line)
            self._log_count += 1

    def error(self, message: str, context: Optional[Dict] = None):
        """Write an ERROR log. Visible to agent via `cat error.log`."""
        self._write("ERROR", message, context)

    def warn(self, message: str, context: Optional[Dict] = None):
        """Write a WARN log. Visible to agent via `cat error.log`."""
        self._write("WARN", message, context)

    def info(self, message: str, context: Optional[Dict] = None):
        """Write an INFO log. Visible to agent via `cat access.log`."""
        self._write("INFO", message, context)

    def inject_misleading_error(self, fake_message: str, fake_context: Optional[Dict] = None):
        """Write a FAKE error to this service's error.log.

        This is how misleading signals work. The fault injector calls:
            auth_logger.inject_misleading_error("JWT validation slow — 2400ms")

        When the agent reads auth's error.log, it sees this fake error and
        might incorrectly conclude auth is the root cause. A GOOD agent
        cross-references with /metrics (which shows auth is fine) and
        dismisses the red herring.
        """
        self._write("ERROR", fake_message, fake_context or {"injected": True})

    def get_recent(self, log_type: str = "error", lines: int = 20) -> str:
        """Return recent log lines. Used by environment when agent does `cat error.log`."""
        path = self._error_path if log_type == "error" else self._access_path
        try:
            with open(path, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except FileNotFoundError:
            return ""

    def get_log_count(self) -> Dict[str, int]:
        """Return log line counts."""
        error_count = 0
        access_count = 0
        try:
            with open(self._error_path, "r") as f:
                error_count = sum(1 for _ in f)
        except FileNotFoundError:
            pass
        try:
            with open(self._access_path, "r") as f:
                access_count = sum(1 for _ in f)
        except FileNotFoundError:
            pass
        return {"error_lines": error_count, "access_lines": access_count}

    def reset(self):
        """Clear all log files. Called at episode start."""
        with self._lock:
            for path in [self._error_path, self._access_path]:
                try:
                    with open(path, "w") as f:
                        pass  # Truncate
                except OSError:
                    pass
            self._log_count = 0

    def clear(self):
        """Alias for reset."""
        self.reset()
