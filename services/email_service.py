"""
CloudSRE v2 — Email/Alerting Service (port 8012).

SMTP-like email delivery. Cascade: notification down → email queue backs up → OOM.

Fault types:
  - smtp_connection_refused: Can't connect to SMTP relay
  - queue_overflow: Email queue exceeds capacity, dropping messages
"""

import time
from cloud_sre_v2.services.base_service import BaseService


class EmailService(BaseService):
    def __init__(self, port: int = 8012, log_dir: str = "/var/log"):
        super().__init__("email", port=port, log_dir=log_dir)
        self._queue = []
        self._sent_count = 0
        self._failed_count = 0
        self._max_queue = 500
        self._smtp_connected = True
        self._queue_overflow = False
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/email/stats")
        async def email_stats():
            return {"service": "email", "queue_size": len(self._queue),
                    "max_queue": self._max_queue, "sent": self._sent_count,
                    "failed": self._failed_count, "smtp_connected": self._smtp_connected,
                    "queue_overflow": self._queue_overflow}

        @self.app.post("/email/send")
        async def send_email(to: str = "user@example.com", subject: str = "Alert"):
            if not self._smtp_connected:
                self._failed_count += 1
                self._queue.append({"to": to, "subject": subject, "ts": time.time()})
                if len(self._queue) > self._max_queue:
                    self._queue_overflow = True
                return {"error": "SMTP_CONNECTION_REFUSED", "queued": True}
            self._sent_count += 1
            return {"status": "sent", "to": to}

        @self.app.post("/email/flush_queue")
        async def flush_queue():
            if not self._smtp_connected:
                return {"error": "SMTP_STILL_DOWN", "queue_size": len(self._queue)}
            flushed = len(self._queue)
            self._sent_count += flushed
            self._queue.clear()
            self._queue_overflow = False
            return {"flushed": flushed, "queue_size": 0}

        @self.app.post("/email/reconnect_smtp")
        async def reconnect_smtp():
            self._smtp_connected = True
            self.logger.info("SMTP connection restored")
            return {"smtp_connected": True}

    def inject_smtp_down(self):
        self._smtp_connected = False
        self.set_degraded()
        self.logger.error("FAULT: SMTP connection refused — emails queueing up")

    def inject_queue_overflow(self):
        self._queue = [{"to": f"user{i}@example.com", "subject": "Alert", "ts": time.time()}
                       for i in range(self._max_queue + 50)]
        self._queue_overflow = True
        self.set_degraded()
        self.logger.error("FAULT: Email queue overflow — dropping messages")

    def reset(self):
        super().reset()
        self._queue.clear()
        self._sent_count = 0
        self._failed_count = 0
        self._smtp_connected = True
        self._queue_overflow = False
