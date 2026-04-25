"""
CloudSRE v2 — Config Service (port 8014).

Configuration management (etcd-like). Cascade: bad config push → all services read wrong values → cascade.

Fault types:
  - config_poisoned: Critical config values set to wrong values
  - config_locked: Config store locked, services can't read updates
"""

import time
from cloud_sre_v2.services.base_service import BaseService


class ConfigService(BaseService):
    def __init__(self, port: int = 8014, log_dir: str = "/var/log"):
        super().__init__("config", port=port, log_dir=log_dir)
        self._store = {}
        self._version = 1
        self._locked = False
        self._poisoned_keys = set()
        self._watch_count = 0
        self._seed_config()
        self._register_routes()

    def _seed_config(self):
        self._store = {
            "db/max_connections": {"value": "100", "version": 1},
            "db/timeout_ms": {"value": "5000", "version": 1},
            "cache/ttl_seconds": {"value": "300", "version": 1},
            "rate_limit/max_rps": {"value": "1000", "version": 1},
            "feature/dark_mode": {"value": "true", "version": 1},
            "service/payment/replicas": {"value": "3", "version": 1},
            "service/worker/memory_limit_mb": {"value": "512", "version": 1},
            "alerting/pagerduty_enabled": {"value": "true", "version": 1},
            "security/jwt_expiry_hours": {"value": "24", "version": 1},
            "queue/max_size": {"value": "1000", "version": 1},
        }
        self._version = 1

    def _register_routes(self):
        @self.app.get("/config/get/{key:path}")
        async def config_get(key: str):
            if self._locked:
                return {"error": "CONFIG_LOCKED", "key": key}
            entry = self._store.get(key)
            if entry:
                poisoned = key in self._poisoned_keys
                return {"key": key, "value": entry["value"], "version": entry["version"],
                        "poisoned": poisoned}
            return {"error": "NOT_FOUND", "key": key}

        @self.app.get("/config/list")
        async def config_list():
            return {"service": "config", "keys": list(self._store.keys()),
                    "version": self._version, "locked": self._locked,
                    "poisoned_keys": len(self._poisoned_keys),
                    "total": len(self._store)}

        @self.app.post("/config/set/{key:path}")
        async def config_set(key: str, value: str = ""):
            if self._locked:
                return {"error": "CONFIG_LOCKED"}
            self._version += 1
            self._store[key] = {"value": value, "version": self._version}
            if key in self._poisoned_keys:
                self._poisoned_keys.discard(key)
            return {"key": key, "value": value, "version": self._version}

        @self.app.post("/config/unlock")
        async def config_unlock():
            self._locked = False
            self.logger.info("Config store unlocked")
            return {"locked": False}

        @self.app.post("/config/rollback")
        async def config_rollback():
            self._seed_config()
            self._poisoned_keys.clear()
            self.logger.info("Config rolled back to defaults")
            return {"status": "rolled_back", "version": self._version}

    def inject_poisoned_config(self):
        self._poisoned_keys = {"db/max_connections", "rate_limit/max_rps", "queue/max_size"}
        self._store["db/max_connections"]["value"] = "1"  # WAY too low
        self._store["rate_limit/max_rps"]["value"] = "0"  # Blocks everything
        self._store["queue/max_size"]["value"] = "5"  # Queue fills instantly
        self._version += 1
        self.set_degraded()
        self.logger.error("FAULT: Config poisoned — db_max_connections=1, rate_limit=0, queue_max=5")

    def inject_lock(self):
        self._locked = True
        self.set_degraded()
        self.logger.error("FAULT: Config store locked — services cannot read updates")

    def reset(self):
        super().reset()
        self._locked = False
        self._poisoned_keys.clear()
        self._watch_count = 0
        self._seed_config()
