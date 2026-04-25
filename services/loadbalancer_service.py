"""
CloudSRE v2 — Load Balancer Service (port 8016).

Layer 7 load balancer. Cascade: backend down → LB health checks fail → removes backend → overloads remaining.

Fault types:
  - backend_removed: All backends marked unhealthy (503 for all traffic)
  - sticky_session_corruption: Session affinity broken, users get wrong backends
"""

import time
from cloud_sre_v2.services.base_service import BaseService


class LoadBalancerService(BaseService):
    def __init__(self, port: int = 8016, log_dir: str = "/var/log"):
        super().__init__("loadbalancer", port=port, log_dir=log_dir)
        self._backends = {}
        self._total_requests = 0
        self._active_connections = 0
        self._error_count = 0
        self._algorithm = "round_robin"
        self._all_removed = False
        self._session_corrupted = False
        self._seed_backends()
        self._register_routes()

    def _seed_backends(self):
        self._backends = {
            "payment-1": {"host": "127.0.0.1:8001", "healthy": True, "weight": 1, "connections": 12},
            "payment-2": {"host": "127.0.0.1:8001", "healthy": True, "weight": 1, "connections": 8},
            "frontend-1": {"host": "127.0.0.1:8004", "healthy": True, "weight": 2, "connections": 24},
            "frontend-2": {"host": "127.0.0.1:8004", "healthy": True, "weight": 2, "connections": 18},
            "api-1": {"host": "127.0.0.1:8008", "healthy": True, "weight": 1, "connections": 6},
        }
        self._total_requests = 89432
        self._active_connections = sum(b["connections"] for b in self._backends.values())

    def _register_routes(self):
        @self.app.get("/lb/stats")
        async def lb_stats():
            healthy = sum(1 for b in self._backends.values() if b["healthy"])
            return {"service": "loadbalancer", "algorithm": self._algorithm,
                    "backends_total": len(self._backends), "backends_healthy": healthy,
                    "total_requests": self._total_requests,
                    "active_connections": self._active_connections,
                    "error_count": self._error_count,
                    "all_removed": self._all_removed,
                    "session_corrupted": self._session_corrupted}

        @self.app.get("/lb/backends")
        async def list_backends():
            return {"backends": self._backends}

        @self.app.post("/lb/route")
        async def route_request(path: str = "/"):
            self._total_requests += 1
            if self._all_removed:
                self._error_count += 1
                return {"error": "NO_HEALTHY_BACKENDS", "status": 503}
            healthy = [n for n, b in self._backends.items() if b["healthy"]]
            if not healthy:
                return {"error": "NO_HEALTHY_BACKENDS", "status": 503}
            target = healthy[self._total_requests % len(healthy)]
            return {"routed_to": target, "host": self._backends[target]["host"]}

        @self.app.post("/lb/restore_backends")
        async def restore_backends():
            for b in self._backends.values():
                b["healthy"] = True
            self._all_removed = False
            self.logger.info("All backends restored to healthy")
            return {"backends_healthy": len(self._backends)}

        @self.app.post("/lb/fix_sessions")
        async def fix_sessions():
            self._session_corrupted = False
            self.logger.info("Session affinity restored")
            return {"session_corrupted": False}

    def inject_all_removed(self):
        for b in self._backends.values():
            b["healthy"] = False
        self._all_removed = True
        self.set_degraded()
        self.logger.error("FAULT: All backends removed — LB returning 503 for all traffic")

    def inject_session_corruption(self):
        self._session_corrupted = True
        self.set_degraded()
        self.logger.error("FAULT: Sticky session corruption — users hitting wrong backends")

    def reset(self):
        super().reset()
        self._all_removed = False
        self._session_corrupted = False
        self._error_count = 0
        self._seed_backends()
