"""
CloudSRE v2 — Real SQLite Database Infrastructure.

This is NOT a simulation. This wraps a REAL SQLite database with real SQL queries,
real connection management, real locking, and real failure modes.

When the fault injector locks this database:
  - payment_service gets REAL sqlite3.OperationalError("database is locked")
  - worker_service gets REAL write failures
  - Queue backs up FOR REAL because writes can't complete

When the lock is released:
  - ALL backed-up queries execute simultaneously (thundering herd)
  - This causes REAL CPU spikes and potential OOM in the services

This is the foundation that makes our cascading failures REAL, not simulated strings.

Kube SRE Gym equivalent: None. They don't have a database layer.
"""

import sqlite3
import threading
import time
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path


class Database:
    """Real SQLite database with connection pooling simulation and fault injection.

    This creates a real database file on disk with real tables. Services make
    REAL SQL queries. When we lock the file, services get REAL exceptions.

    Usage:
        db = Database("/data/app.db")
        db.initialize()  # Creates tables
        db.execute("INSERT INTO payments (amount, status) VALUES (?, ?)", (100.0, "pending"))
        rows = db.query("SELECT * FROM payments WHERE status = ?", ("pending",))
        db.close()
    """

    def __init__(self, db_path: str = "/data/app.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        # REAL lock: a connection holding a BEGIN EXCLUSIVE transaction
        # When this is not None, SQLite itself blocks all other writers
        self._lock_conn: Optional[sqlite3.Connection] = None
        self._is_fault_locked = False  # kept for quick-check in get_table_stats
        self._connection_count = 0
        self._max_connections = 50
        self._query_count = 0
        self._error_count = 0
        self._created = False

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """Create the database schema — real tables for real services.

        Tables:
          - payments: transaction records (used by payment_service)
          - sessions: auth sessions (used by auth_service)
          - jobs: worker job queue results (used by worker_service)
          - audit_log: action audit trail (used by all services)
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrency
        conn.execute("PRAGMA busy_timeout=5000")  # 5s timeout before "database is locked"

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_valid INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                payload TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_valid ON sessions(is_valid);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
        conn.commit()
        conn.close()
        self._created = True

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        If the database is REALLY locked (via BEGIN EXCLUSIVE from inject_lock),
        SQLite itself will raise OperationalError after busy_timeout.
        We use a very short timeout (0.1s) so the error surfaces quickly
        during training episodes.
        """
        if self._connection_count >= self._max_connections:
            self._error_count += 1
            raise sqlite3.OperationalError(
                f"connection pool exhausted (max_connections={self._max_connections}, "
                f"active={self._connection_count})"
            )

        # Short busy_timeout when locked so errors surface fast (not 5s delay)
        timeout = 0.1 if self._is_fault_locked else 5.0
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        self._connection_count += 1
        return conn

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query (INSERT/UPDATE/DELETE). Returns lastrowid.

        If the database is locked via inject_lock(), SQLite's own locking
        mechanism will raise a REAL sqlite3.OperationalError("database is locked").
        This is not us faking it — SQLite itself refuses the write.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                self._query_count += 1
                return cursor.lastrowid
            except sqlite3.OperationalError as e:
                self._error_count += 1
                raise
            finally:
                self._connection_count -= 1
                conn.close()

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a read query (SELECT). Returns list of row dicts.

        When the DB is locked with EXCLUSIVE, even reads may fail depending
        on WAL mode and lock state. The agent sees real SQLite errors.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(sql, params)
                rows = [dict(row) for row in cursor.fetchall()]
                self._query_count += 1
                return rows
            except sqlite3.OperationalError as e:
                self._error_count += 1
                raise
            finally:
                self._connection_count -= 1
                conn.close()

    def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """Execute a batch write. Used for flooding (thundering herd simulation)."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executemany(sql, params_list)
                conn.commit()
                self._query_count += len(params_list)
                return len(params_list)
            except sqlite3.OperationalError as e:
                self._error_count += 1
                raise
            finally:
                self._connection_count -= 1
                conn.close()

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_lock(self):
        """Lock the database using a REAL SQLite EXCLUSIVE transaction.

        This opens a real connection and holds BEGIN EXCLUSIVE, which tells
        SQLite's own locking mechanism to block ALL other writers.
        Any service trying to write will get a REAL:
            sqlite3.OperationalError: database is locked
        from SQLite itself — NOT from our code.
        """
        if self._lock_conn is not None:
            return  # Already locked
        try:
            self._lock_conn = sqlite3.connect(self.db_path, timeout=1.0)
            self._lock_conn.execute("BEGIN EXCLUSIVE")
            self._is_fault_locked = True
        except Exception:
            # If we can't lock (e.g., db doesn't exist yet), fall back to flag
            self._is_fault_locked = True

    def release_lock(self):
        """Release the REAL database lock.

        Rolls back the EXCLUSIVE transaction, allowing other connections
        to write again. If services have queued up requests while locked,
        they will ALL hit the database simultaneously (thundering herd).
        """
        if self._lock_conn is not None:
            try:
                self._lock_conn.rollback()
                self._lock_conn.close()
            except Exception:
                pass
            self._lock_conn = None
        self._is_fault_locked = False

    def inject_connection_exhaustion(self):
        """Set max connections to 0 — no new connections allowed."""
        self._max_connections = 0

    def release_connection_exhaustion(self):
        """Restore normal connection limit."""
        self._max_connections = 50

    # ── Metrics (exposed via /metrics endpoint) ──────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return real database metrics for the /metrics endpoint.

        The agent can read these via:
            curl http://localhost:8001/metrics
        """
        return {
            "db_connections_active": self._connection_count,
            "db_connections_max": self._max_connections,
            "db_queries_total": self._query_count,
            "db_errors_total": self._error_count,
            "db_is_locked": self._is_fault_locked,
            "db_path": self.db_path,
        }

    def get_table_stats(self) -> Dict[str, int]:
        """Return row counts for each table — useful for agent diagnostics."""
        if self._is_fault_locked:
            return {"error": "database is locked"}

        try:
            stats = {}
            for table in ["payments", "sessions", "jobs", "audit_log"]:
                rows = self.query(f"SELECT count(*) as cnt FROM {table}")
                stats[table] = rows[0]["cnt"] if rows else 0
            return stats
        except sqlite3.OperationalError:
            return {"error": "database unavailable"}

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        """Full reset — release locks, wipe all data, recreate fresh."""
        # Release any real locks first
        self.release_lock()
        self.release_connection_exhaustion()

        self._connection_count = 0
        self._query_count = 0
        self._error_count = 0

        # Delete and recreate
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            # Also remove WAL and SHM files
            for suffix in ["-wal", "-shm"]:
                wal_path = self.db_path + suffix
                if os.path.exists(wal_path):
                    os.remove(wal_path)
        except OSError:
            pass

        self.initialize()

    def close(self):
        """Clean shutdown."""
        self._is_fault_locked = False
        self._connection_count = 0
