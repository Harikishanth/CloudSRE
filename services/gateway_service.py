"""
CloudSRE v2 — API Gateway Service (port 8008).

Rate-limiting API gateway. Cascade: auth down → gateway can't validate tokens → 401 storm.

Fault types:
  - rate_limit_misconfigured: Rate limit set to 0 (blocks everything)
  - circuit_breaker_stuck: Circuit breaker stuck open (rejects all traffic)
"""

import time
from fastapi import Request
from cloud_sre_v2.services.base_service import BaseService


class GatewayService(BaseService):
    def __init__(self, port: int = 8008, log_dir: str = "/var/log"):
        super().__init__("gateway", port=port, log_dir=log_dir)
        self._rate_limit = 1000
        self._current_rps = 0
        self._rejected_count = 0
        self._total_requests = 0
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_threshold = 5
        self._upstream_latency_ms = 45
        self._register_routes()

    def _register_routes(self):
        @self.app.get("/gateway/stats")
        async def gateway_stats():
            return {
                "service": "gateway", "rate_limit_rps": self._rate_limit,
                "current_rps": self._current_rps, "rejected": self._rejected_count,
                "total_requests": self._total_requests, "circuit_open": self._circuit_open,
                "circuit_failures": self._circuit_failures,
                "upstream_latency_ms": self._upstream_latency_ms,
            }

        @self.app.post("/gateway/request")
        async def gateway_request(request: Request):
            self._total_requests += 1
            if self._circuit_open:
                self._rejected_count += 1
                return {"status": "rejected", "reason": "CIRCUIT_BREAKER_OPEN"}
            if self._rate_limit == 0 or self._current_rps >= self._rate_limit:
                self._rejected_count += 1
                return {"status": "rejected", "reason": "RATE_LIMITED"}
            self._current_rps += 1
            return {"status": "forwarded", "latency_ms": self._upstream_latency_ms}

        @self.app.post("/gateway/reset_circuit")
        async def reset_circuit():
            self._circuit_open = False
            self._circuit_failures = 0
            self.logger.info("Circuit breaker reset to CLOSED")
            return {"circuit_open": False}

        @self.app.post("/gateway/set_rate_limit")
        async def set_rate_limit(rps: int = 1000):
            self._rate_limit = rps
            return {"rate_limit_rps": rps}

    def inject_rate_limit_zero(self):
        self._rate_limit = 0
        self.set_degraded()
        self.logger.error("FAULT: Rate limit set to 0 — ALL traffic blocked")

    def inject_circuit_open(self):
        self._circuit_open = True
        self._circuit_failures = 99
        self.set_degraded()
        self.logger.error("FAULT: Circuit breaker stuck OPEN — rejecting all requests")

    def reset(self):
        super().reset()
        self._rate_limit = 1000
        self._current_rps = 0
        self._rejected_count = 0
        self._circuit_open = False
        self._circuit_failures = 0
