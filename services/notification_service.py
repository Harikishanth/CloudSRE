"""
CloudSRE v2 — Notification Service (port 8006).

Simulates a webhook/notification delivery service.
Enables "webhook retry storm" cascade pattern — a real Stripe failure mode:
  1. Payment goes down → webhooks queue up (can't deliver)
  2. Payment comes back → 500+ queued webhooks fire simultaneously
  3. Downstream overwhelmed → cascades back to payment

Fault types:
  - webhook_storm:     Mass retry of queued webhooks
  - delivery_failure:  Webhook endpoint unreachable
"""

import time
import random
from fastapi import Request
from fastapi.responses import JSONResponse

from services.base_service import BaseService


class NotificationService(BaseService):
    """Webhook/notification delivery service with retry storm mechanics."""

    def __init__(self, port: int = 8006, log_dir: str = "/var/log"):
        super().__init__("notification", port=port, log_dir=log_dir)

        # Notification state
        self._pending_webhooks: list = []
        self._delivered_count = 0
        self._failed_count = 0
        self._retry_count = 0
        self._max_queue = 500
        self._delivery_rate = 50  # per second
        self._is_storming = False  # True during retry storm

        # Seed initial state
        self._seed_state()
        self._register_routes()

    def _seed_state(self):
        """Seed realistic notification state."""
        self._delivered_count = 1247  # Simulate pre-existing traffic
        self._failed_count = 3
        self._pending_webhooks = []

    def _register_routes(self):
        """Register notification-specific API routes."""

        @self.app.get("/webhooks/pending")
        async def pending_webhooks():
            """Show pending webhook queue depth and status."""
            return {
                "service": "notification",
                "pending_count": len(self._pending_webhooks),
                "max_queue": self._max_queue,
                "delivered_total": self._delivered_count,
                "failed_total": self._failed_count,
                "retry_count": self._retry_count,
                "is_storming": self._is_storming,
                "delivery_rate_per_sec": self._delivery_rate,
                "queue_utilization": round(
                    len(self._pending_webhooks) / self._max_queue, 2
                ),
            }

        @self.app.get("/webhooks/recent")
        async def recent_webhooks():
            """Show last 10 webhook attempts."""
            recent = self._pending_webhooks[-10:] if self._pending_webhooks else []
            return {
                "service": "notification",
                "recent": recent,
                "total_pending": len(self._pending_webhooks),
            }

        @self.app.post("/webhooks/drain")
        async def drain_webhooks(count: int = 10):
            """Process pending webhooks at controlled rate."""
            drained = min(count, len(self._pending_webhooks))
            self._pending_webhooks = self._pending_webhooks[drained:]
            self._delivered_count += drained
            if not self._pending_webhooks:
                self._is_storming = False
            self.logger.info(f"Drained {drained} webhooks. Remaining: {len(self._pending_webhooks)}")
            return {
                "drained": drained,
                "remaining": len(self._pending_webhooks),
                "is_storming": self._is_storming,
            }

        @self.app.post("/webhooks/pause")
        async def pause_delivery():
            """Pause webhook delivery — emergency stop for storms."""
            self._is_storming = False
            self._delivery_rate = 0
            self.logger.info("Webhook delivery PAUSED")
            return {"status": "paused", "delivery_rate": 0}

        @self.app.post("/webhooks/resume")
        async def resume_delivery(rate: int = 50):
            """Resume webhook delivery at specified rate."""
            self._delivery_rate = rate
            self.logger.info(f"Webhook delivery resumed at {rate}/sec")
            return {"status": "resumed", "delivery_rate": rate}

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_webhook_storm(self, count: int = 300):
        """Simulate mass webhook retry — all queued hooks fire at once."""
        self._is_storming = True
        now = time.time()
        for i in range(count):
            self._pending_webhooks.append({
                "id": f"whk_{random.randint(10000, 99999)}",
                "type": random.choice([
                    "payment.completed", "payment.failed",
                    "order.shipped", "user.updated",
                    "subscription.renewed", "refund.processed",
                ]),
                "retry_count": random.randint(1, 5),
                "created_at": now - random.randint(60, 3600),
                "status": "pending_retry",
            })
        self._retry_count += count
        self.set_degraded()
        self.logger.error(
            f"WEBHOOK STORM: {count} webhooks queued for simultaneous retry"
        )
        self.logger.error(
            f"WEBHOOK STORM: Queue depth {len(self._pending_webhooks)}/{self._max_queue}"
        )
        self.logger.error(
            "WEBHOOK STORM: Downstream services may be overwhelmed"
        )

    def inject_delivery_failure(self):
        """Simulate webhook endpoint unreachable."""
        self._delivery_rate = 0
        self.set_unhealthy("Webhook delivery endpoint unreachable")
        self.logger.error("FAULT: Cannot deliver webhooks — endpoint down")
        self.logger.error("FAULT: Webhooks will queue up until endpoint recovers")

    def reset(self):
        """Reset notification service for new episode."""
        super().reset()
        self._pending_webhooks.clear()
        self._delivered_count = 0
        self._failed_count = 0
        self._retry_count = 0
        self._is_storming = False
        self._delivery_rate = 50
        self._seed_state()
