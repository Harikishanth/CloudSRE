"""
CloudSRE v2 — Payment Service (port 8001).

A REAL FastAPI microservice that:
  - Processes payments via real SQLite INSERT/UPDATE
  - Publishes events to real message queue
  - Tracks real metrics (error rate, latency, queue depth)
  - Writes real structured JSON logs to disk
  - Has real failure modes (DB lock → 503, queue full → 503)

This is NOT a mock. When the database is locked:
  - POST /pay returns REAL HTTP 503
  - Error logs show REAL sqlite3.OperationalError
  - Error rate metric ACTUALLY increases
  - Queue backs up FOR REAL

Kube SRE Gym equivalent: Their "payment-api" is a pre-built Docker image.
They can't see inside it, modify it, or inject application-level faults.
"""

import os
import time
import sqlite3
import json
import random
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse

from cloud_sre_v2.services.base_service import BaseService
from cloud_sre_v2.infra.database import Database
from cloud_sre_v2.infra.queue import MessageQueue, QueueFull, QueuePaused


class PaymentService(BaseService):
    """Real payment processing microservice.

    Routes:
        POST /pay              — Process a payment (writes to DB, publishes to queue)
        GET  /payments         — List recent payments
        GET  /payments/pending — Count pending payments
        POST /payments/retry   — Retry failed payments
        GET  /healthz          — (inherited) Real health check
        GET  /metrics          — (inherited) Real Prometheus metrics
        GET  /logs             — (inherited) Real log access
    """

    def __init__(
        self,
        database: Database,
        queue: MessageQueue,
        port: int = 8001,
        log_dir: str = "/var/log",
    ):
        super().__init__("payment", port=port, log_dir=log_dir)
        self.database = database
        self.queue = queue
        self._pid = os.getpid()

        # Payment-specific metrics
        self.payments_processed = self.metrics.add_counter("payments_processed")
        self.payments_failed = self.metrics.add_counter("payments_failed")
        self.queue_depth_gauge = self.metrics.add_gauge("queue_depth")

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Register payment-specific endpoints."""

        @self.app.post("/pay")
        async def pay(request: Request):
            """Process a payment — REAL database write + queue publish.

            This is a real transaction:
            1. Insert into payments table (real SQL)
            2. Publish payment.completed to queue (real message)
            3. Return result

            When the DB is locked, step 1 throws REAL sqlite3.OperationalError.
            When the queue is full, step 2 throws REAL QueueFull.
            """
            start = time.time()

            try:
                body = await request.json()
            except Exception:
                body = {"amount": round(random.uniform(10, 500), 2), "user_id": f"user_{random.randint(1, 1000)}"}

            amount = body.get("amount", 100.0)
            user_id = body.get("user_id", "anonymous")

            try:
                # Step 1: Write to real database
                payment_id = self.database.execute(
                    "INSERT INTO payments (amount, user_id, status) VALUES (?, ?, ?)",
                    (amount, user_id, "processing"),
                )

                # Step 2: Publish to real queue
                try:
                    self.queue.push("payment.completed", {
                        "payment_id": payment_id,
                        "amount": amount,
                        "user_id": user_id,
                        "timestamp": time.time(),
                    })
                except (QueueFull, QueuePaused) as e:
                    # Queue is full/paused — payment created but not queued
                    self.database.execute(
                        "UPDATE payments SET status = ?, error = ? WHERE id = ?",
                        ("queued_failed", str(e), payment_id),
                    )
                    self.payments_failed.inc()
                    self.logger.error(
                        f"Queue publish failed for payment {payment_id}",
                        {"error": str(e), "amount": amount, "user_id": user_id},
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "Queue unavailable — payment created but not queued",
                            "payment_id": payment_id,
                            "queue_error": str(e),
                        },
                    )

                # Step 3: Mark success
                self.database.execute(
                    "UPDATE payments SET status = ?, processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    ("completed", payment_id),
                )

                self.payments_processed.inc()
                duration_ms = (time.time() - start) * 1000
                self.logger.info(
                    f"Payment {payment_id} processed",
                    {"amount": amount, "user_id": user_id, "duration_ms": round(duration_ms, 1)},
                )

                # Update queue depth gauge
                self.queue_depth_gauge.set(self.queue.depth())

                return {
                    "status": "completed",
                    "payment_id": payment_id,
                    "amount": amount,
                    "duration_ms": round(duration_ms, 1),
                }

            except sqlite3.OperationalError as e:
                # REAL database error — this is what happens when DB is locked
                self.payments_failed.inc()
                self.logger.error(
                    f"Database error processing payment",
                    {"error": str(e), "amount": amount, "user_id": user_id},
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": f"Database unavailable: {str(e)}",
                        "service": "payment",
                    },
                )

        @self.app.get("/payments")
        async def list_payments(limit: int = 20, status: Optional[str] = None):
            """List recent payments from the real database."""
            try:
                if status:
                    rows = self.database.query(
                        "SELECT * FROM payments WHERE status = ? ORDER BY id DESC LIMIT ?",
                        (status, limit),
                    )
                else:
                    rows = self.database.query(
                        "SELECT * FROM payments ORDER BY id DESC LIMIT ?",
                        (limit,),
                    )
                return {"payments": rows, "count": len(rows)}
            except sqlite3.OperationalError as e:
                return JSONResponse(
                    status_code=503,
                    content={"error": str(e)},
                )

        @self.app.get("/payments/pending")
        async def pending_count():
            """Count pending payments — useful for agent diagnostics."""
            try:
                rows = self.database.query(
                    "SELECT count(*) as cnt FROM payments WHERE status IN ('pending', 'processing')"
                )
                return {"pending_count": rows[0]["cnt"] if rows else 0}
            except sqlite3.OperationalError as e:
                return JSONResponse(
                    status_code=503,
                    content={"error": str(e)},
                )

        @self.app.post("/payments/retry")
        async def retry_failed():
            """Retry failed payments — a fix action the agent can take."""
            try:
                rows = self.database.query(
                    "SELECT id, amount, user_id FROM payments WHERE status = 'queued_failed'"
                )
                retried = 0
                for row in rows:
                    try:
                        self.queue.push("payment.completed", {
                            "payment_id": row["id"],
                            "amount": row["amount"],
                            "user_id": row["user_id"],
                            "timestamp": time.time(),
                            "is_retry": True,
                        })
                        self.database.execute(
                            "UPDATE payments SET status = 'processing' WHERE id = ?",
                            (row["id"],),
                        )
                        retried += 1
                    except (QueueFull, QueuePaused):
                        break

                return {"retried": retried, "remaining": len(rows) - retried}
            except sqlite3.OperationalError as e:
                return JSONResponse(
                    status_code=503,
                    content={"error": str(e)},
                )

    def reset(self):
        """Reset payment service for new episode."""
        super().reset()
        self._pid = os.getpid()
