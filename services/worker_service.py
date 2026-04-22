"""
CloudSRE v2 — Worker Service (port 8003).

A REAL message queue consumer that:
  - Drains messages from the queue
  - Writes results to the real SQLite database
  - Tracks processing metrics (throughput, errors, lag)
  - Has real failure modes (crash, stale connections, backlog)

This is the service that DIES during a thundering herd cascade:
  1. DB locks → queue fills with 847 messages
  2. Agent fixes DB lock
  3. Worker calls drain_all() → 847 writes hit DB simultaneously
  4. Worker process OOMs → needs restart

The smart agent should:
  - Use drain_controlled(rate=10) instead of drain_all()
  - Monitor worker health after fixing upstream DB
"""

import os
import time
import sqlite3
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse

from cloud_sre_v2.services.base_service import BaseService
from cloud_sre_v2.infra.database import Database
from cloud_sre_v2.infra.queue import MessageQueue, QueueFull


class WorkerService(BaseService):
    """Real queue consumer microservice.

    Routes:
        POST /process           — Process next batch of queued messages
        GET  /queue/status      — Queue depth and consumer status
        POST /queue/drain       — Drain queue (rate parameter controls speed)
        GET  /jobs              — List recent processed jobs
        GET  /healthz           — (inherited)
        GET  /metrics           — (inherited)
    """

    def __init__(
        self,
        database: Database,
        queue: MessageQueue,
        port: int = 8003,
        log_dir: str = "/var/log",
    ):
        super().__init__("worker", port=port, log_dir=log_dir)
        self.database = database
        self.queue = queue
        self._pid = os.getpid()

        # Worker-specific config
        self._batch_size = 10
        self._processing = False
        self._memory_usage_mb = 50.0  # Simulated base memory

        # Worker-specific metrics
        self.messages_processed = self.metrics.add_counter("messages_processed")
        self.messages_failed = self.metrics.add_counter("messages_failed")
        self.queue_depth_gauge = self.metrics.add_gauge("queue_depth")
        self.memory_gauge = self.metrics.add_gauge("memory_simulated_mb")
        self.memory_gauge.set(self._memory_usage_mb)

        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/process")
        async def process_batch(batch_size: int = 10):
            """Process a batch of queued messages.

            Each message triggers a REAL database write. If the DB is
            locked, writes fail and messages go back to the queue.
            """
            if not self._is_healthy:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Worker process crashed — needs restart"},
                )

            batch = self.queue.pop_batch(batch_size)
            if not batch:
                return {"processed": 0, "queue_depth": self.queue.depth()}

            processed = 0
            failed = 0

            for msg in batch:
                try:
                    # Real database write for each message
                    self.database.execute(
                        "INSERT INTO jobs (job_type, payload, status, completed_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                        (msg.topic, str(msg.payload), "completed"),
                    )

                    # Update payment status if it's a payment message
                    if msg.topic == "payment.completed" and "payment_id" in msg.payload:
                        self.database.execute(
                            "UPDATE payments SET status = 'settled', processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                            (msg.payload["payment_id"],),
                        )

                    processed += 1
                    self.messages_processed.inc()
                    self._memory_usage_mb += 0.5  # Memory grows per message

                except sqlite3.OperationalError as e:
                    # DB failed — nack the message
                    self.queue.nack(msg)
                    failed += 1
                    self.messages_failed.inc()
                    self.logger.error(f"DB write failed for message {msg.id}: {e}")

            # Check for OOM (memory exceeds threshold)
            self.memory_gauge.set(self._memory_usage_mb)
            if self._memory_usage_mb > 500:
                self.set_unhealthy("OOM: memory exceeded 500MB threshold")
                self.logger.error(
                    f"Worker OOM! Memory: {self._memory_usage_mb:.0f}MB > 500MB threshold",
                    {"messages_in_batch": len(batch), "queue_depth": self.queue.depth()},
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": f"Worker OOM — memory {self._memory_usage_mb:.0f}MB > 500MB",
                        "processed_before_crash": processed,
                    },
                )

            self.queue_depth_gauge.set(self.queue.depth())
            self.logger.info(
                f"Processed {processed} messages, {failed} failed",
                {"queue_depth": self.queue.depth(), "memory_mb": round(self._memory_usage_mb, 1)},
            )

            return {
                "processed": processed,
                "failed": failed,
                "queue_depth": self.queue.depth(),
                "memory_mb": round(self._memory_usage_mb, 1),
            }

        @self.app.get("/queue/status")
        async def queue_status():
            """Queue status — useful for agent diagnostics."""
            return {
                "queue_depth": self.queue.depth(),
                "dead_letters": self.queue.dead_letter_count(),
                "consumer_healthy": self._is_healthy,
                "memory_mb": round(self._memory_usage_mb, 1),
                **self.queue.get_metrics(),
            }

        @self.app.post("/queue/drain")
        async def drain_queue(rate: int = 10, all: bool = False):
            """Drain the queue.

            rate=10: Safe, controlled drain (CORRECT approach)
            all=True: Drain everything at once (THUNDERING HERD — BAD!)
            """
            if all:
                # DANGEROUS — drain everything at once
                messages = self.queue.drain_all()
                batch_size = len(messages)
                self.logger.warn(f"DRAIN ALL triggered — {batch_size} messages")

                # Process all at once — this can OOM!
                processed = 0
                for msg in messages:
                    try:
                        self.database.execute(
                            "INSERT INTO jobs (job_type, payload, status) VALUES (?, ?, ?)",
                            (msg.topic, str(msg.payload), "completed"),
                        )
                        processed += 1
                        self._memory_usage_mb += 0.5
                    except sqlite3.OperationalError:
                        self.messages_failed.inc()

                # Check OOM after bulk processing
                self.memory_gauge.set(self._memory_usage_mb)
                if self._memory_usage_mb > 500:
                    self.set_unhealthy(f"OOM after drain_all: {self._memory_usage_mb:.0f}MB")
                    self.logger.error(
                        f"THUNDERING HERD → OOM! Drained {batch_size}, memory {self._memory_usage_mb:.0f}MB"
                    )

                return {
                    "drained": batch_size,
                    "processed": processed,
                    "memory_mb": round(self._memory_usage_mb, 1),
                    "warning": "THUNDERING HERD — all messages processed at once",
                }
            else:
                # Safe controlled drain
                batch = self.queue.drain_controlled(rate=rate)
                return {"drained": len(batch), "rate": rate, "queue_remaining": self.queue.depth()}

        @self.app.get("/jobs")
        async def list_jobs(limit: int = 20):
            """List recent jobs from real database."""
            try:
                rows = self.database.query(
                    "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
                )
                return {"jobs": rows, "count": len(rows)}
            except sqlite3.OperationalError as e:
                return JSONResponse(status_code=503, content={"error": str(e)})

    def reset(self):
        super().reset()
        self._memory_usage_mb = 50.0
        self._processing = False
        self.memory_gauge.set(50.0)
        self._pid = os.getpid()
