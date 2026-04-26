"""
CloudSRE v2 — DNS/Service Discovery (port 8015).

Internal DNS and service registry. Cascade: DNS down → services can't find each other → total outage.

Fault types:
  - dns_resolution_failure: All lookups return NXDOMAIN
  - stale_entries: DNS cache serves old/dead IPs
"""

import time
from services.base_service import BaseService


class DNSService(BaseService):
    def __init__(self, port: int = 8015, log_dir: str = "/var/log"):
        super().__init__("dns", port=port, log_dir=log_dir)
        self._registry = {}
        self._lookup_count = 0
        self._failures = 0
        self._resolution_disabled = False
        self._stale = False
        self._seed_registry()
        self._register_routes()

    def _seed_registry(self):
        self._registry = {
            "payment.internal": {"ip": "127.0.0.1", "port": 8001, "ttl": 300, "healthy": True},
            "auth.internal": {"ip": "127.0.0.1", "port": 8002, "ttl": 300, "healthy": True},
            "worker.internal": {"ip": "127.0.0.1", "port": 8003, "ttl": 300, "healthy": True},
            "frontend.internal": {"ip": "127.0.0.1", "port": 8004, "ttl": 300, "healthy": True},
            "cache.internal": {"ip": "127.0.0.1", "port": 8005, "ttl": 300, "healthy": True},
            "notification.internal": {"ip": "127.0.0.1", "port": 8006, "ttl": 300, "healthy": True},
            "search.internal": {"ip": "127.0.0.1", "port": 8007, "ttl": 300, "healthy": True},
            "gateway.internal": {"ip": "127.0.0.1", "port": 8008, "ttl": 300, "healthy": True},
            "db.internal": {"ip": "127.0.0.1", "port": 5432, "ttl": 600, "healthy": True},
            "queue.internal": {"ip": "127.0.0.1", "port": 5672, "ttl": 600, "healthy": True},
        }
        self._lookup_count = 14392

    def _register_routes(self):
        @self.app.get("/dns/resolve/{hostname}")
        async def resolve(hostname: str):
            self._lookup_count += 1
            if self._resolution_disabled:
                self._failures += 1
                return {"error": "NXDOMAIN", "hostname": hostname}
            entry = self._registry.get(hostname)
            if entry:
                if self._stale:
                    return {"hostname": hostname, "ip": "10.0.0.DEAD", "port": 0,
                            "warning": "STALE_ENTRY"}
                return {"hostname": hostname, **entry}
            return {"error": "NOT_FOUND", "hostname": hostname}

        @self.app.get("/dns/registry")
        async def list_registry():
            return {"service": "dns", "entries": self._registry,
                    "total": len(self._registry), "lookups": self._lookup_count,
                    "failures": self._failures, "resolution_disabled": self._resolution_disabled,
                    "stale": self._stale}

        @self.app.post("/dns/register")
        async def register(hostname: str, ip: str = "127.0.0.1", port: int = 8000):
            self._registry[hostname] = {"ip": ip, "port": port, "ttl": 300, "healthy": True}
            return {"registered": hostname}

        @self.app.post("/dns/flush_cache")
        async def flush_cache():
            self._stale = False
            self._seed_registry()
            self.logger.info("DNS cache flushed and refreshed")
            return {"status": "flushed", "entries": len(self._registry)}

        @self.app.post("/dns/enable_resolution")
        async def enable_resolution():
            self._resolution_disabled = False
            self.logger.info("DNS resolution re-enabled")
            return {"resolution_disabled": False}

    def inject_resolution_failure(self):
        self._resolution_disabled = True
        self.set_degraded()
        self.logger.error("FAULT: DNS resolution disabled — all lookups return NXDOMAIN")

    def inject_stale_entries(self):
        self._stale = True
        self.set_degraded()
        self.logger.error("FAULT: DNS serving stale entries — services connecting to dead IPs")

    def reset(self):
        super().reset()
        self._resolution_disabled = False
        self._stale = False
        self._failures = 0
        self._seed_registry()
