"""
CloudSRE v2 — Billing Service (port 8013).

Usage tracking and billing. Cascade: payment down → billing can't record charges → revenue leak.

Fault types:
  - billing_desync: Charges recorded but not applied (data inconsistency)
  - invoice_generation_stuck: Invoice cron stuck, no invoices generated
"""

import time
from services.base_service import BaseService


class BillingService(BaseService):
    def __init__(self, port: int = 8013, log_dir: str = "/var/log"):
        super().__init__("billing", port=port, log_dir=log_dir)
        self._charges = []
        self._invoices = []
        self._total_revenue = 0.0
        self._pending_amount = 0.0
        self._desync = False
        self._invoice_stuck = False
        self._seed_data()
        self._register_routes()

    def _seed_data(self):
        self._charges = [
            {"id": f"chg_{i}", "amount": 29.99, "user": f"user_{i}", "ts": time.time() - i * 3600}
            for i in range(1, 11)
        ]
        self._total_revenue = sum(c["amount"] for c in self._charges)
        self._invoices = [
            {"id": "inv_001", "amount": 299.90, "status": "paid", "period": "2024-Q4"},
        ]

    def _register_routes(self):
        @self.app.get("/billing/stats")
        async def billing_stats():
            return {"service": "billing", "total_revenue": self._total_revenue,
                    "pending_amount": self._pending_amount,
                    "charge_count": len(self._charges),
                    "invoice_count": len(self._invoices),
                    "desync": self._desync, "invoice_stuck": self._invoice_stuck}

        @self.app.post("/billing/charge")
        async def create_charge(amount: float = 29.99, user: str = "user_1"):
            if self._desync:
                self._pending_amount += amount
                self.logger.warn(f"Charge recorded but NOT applied: ${amount} for {user}")
                return {"status": "recorded_not_applied", "desync": True}
            self._charges.append({"id": f"chg_{len(self._charges)+1}", "amount": amount,
                                  "user": user, "ts": time.time()})
            self._total_revenue += amount
            return {"status": "charged", "amount": amount}

        @self.app.post("/billing/reconcile")
        async def reconcile():
            self._total_revenue += self._pending_amount
            reconciled = self._pending_amount
            self._pending_amount = 0.0
            self._desync = False
            return {"reconciled": reconciled, "desync": False}

        @self.app.post("/billing/unstick_invoices")
        async def unstick_invoices():
            self._invoice_stuck = False
            self.logger.info("Invoice generation resumed")
            return {"invoice_stuck": False}

    def inject_desync(self):
        self._desync = True
        self.set_degraded()
        self.logger.error("FAULT: Billing desync — charges recorded but not applied to accounts")

    def inject_invoice_stuck(self):
        self._invoice_stuck = True
        self.set_degraded()
        self.logger.error("FAULT: Invoice generation stuck — no invoices being created")

    def reset(self):
        super().reset()
        self._desync = False
        self._invoice_stuck = False
        self._pending_amount = 0.0
        self._seed_data()
