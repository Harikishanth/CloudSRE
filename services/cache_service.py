"""
CloudSRE v2 — Cache Service (port 8005).

Simulates a Redis-like caching layer for the SRE environment.
Enables "cache stampede" cascade pattern — a real AWS/Stripe failure mode:
  1. DB goes down → cache can't refresh → cache entries expire
  2. DB comes back → every request misses cache → hits DB directly
  3. DB overwhelmed by thundering herd → crashes again

Fault types:
  - cache_invalidation: Dumps all cached data (cold cache)
  - cache_ttl_expired:  All TTLs expire simultaneously
"""

import time
from fastapi import Request
from fastapi.responses import JSONResponse

from services.base_service import BaseService


class CacheService(BaseService):
    """In-memory cache service with realistic failure modes."""

    def __init__(self, port: int = 8005, log_dir: str = "/var/log"):
        super().__init__("cache", port=port, log_dir=log_dir)

        # Cache state
        self._cache: dict = {}
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._max_size = 10000
        self._default_ttl = 300  # 5 minutes
        self._is_cold = False  # True after invalidation

        # Seed initial cache entries
        self._seed_cache()
        self._register_routes()

    def _seed_cache(self):
        """Seed realistic cache entries for a new episode."""
        now = time.time()
        seed_entries = {
            "session:user_1": {"data": "active", "ttl": now + 600},
            "session:user_2": {"data": "active", "ttl": now + 600},
            "product:pricing_table": {"data": "cached_pricing", "ttl": now + 1800},
            "config:rate_limits": {"data": '{"max_rps": 1000}', "ttl": now + 3600},
            "db:payments_recent": {"data": "cached_query_result", "ttl": now + 120},
            "db:user_profiles": {"data": "cached_profiles", "ttl": now + 300},
            "auth:public_keys": {"data": "cached_jwks", "ttl": now + 7200},
        }
        for key, val in seed_entries.items():
            self._cache[key] = val
        self._hit_count = 847  # Simulate pre-existing traffic
        self._miss_count = 53

    def _register_routes(self):
        """Register cache-specific API routes."""

        @self.app.get("/cache/stats")
        async def cache_stats():
            """Cache statistics — hit ratio, size, evictions."""
            total = self._hit_count + self._miss_count
            hit_ratio = self._hit_count / max(total, 1)
            return {
                "service": "cache",
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_ratio": round(hit_ratio, 4),
                "eviction_count": self._eviction_count,
                "is_cold": self._is_cold,
                "default_ttl_seconds": self._default_ttl,
            }

        @self.app.get("/cache/get/{key}")
        async def cache_get(key: str):
            """Simulate cache lookup."""
            now = time.time()
            entry = self._cache.get(key)
            if entry and entry["ttl"] > now:
                self._hit_count += 1
                return {"key": key, "hit": True, "data": entry["data"]}
            else:
                self._miss_count += 1
                if self._is_cold:
                    self.logger.warn(f"Cache MISS on cold cache: {key}")
                return {"key": key, "hit": False, "data": None}

        @self.app.post("/cache/invalidate")
        async def cache_invalidate():
            """Invalidate all cache entries — triggers cold cache."""
            old_size = len(self._cache)
            self._cache.clear()
            self._is_cold = True
            self._eviction_count += old_size
            self.logger.error(
                f"CACHE INVALIDATED: {old_size} entries evicted. Cache is now COLD."
            )
            return {"invalidated": old_size, "is_cold": True}

        @self.app.post("/cache/warmup")
        async def cache_warmup():
            """Warm the cache back up — fix for cold cache."""
            self._seed_cache()
            self._is_cold = False
            self.logger.info("Cache warmed up with seed data")
            return {"status": "warmed", "cache_size": len(self._cache)}

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_invalidation(self):
        """Dump all cached data — simulates cache stampede trigger."""
        self._cache.clear()
        self._is_cold = True
        self._eviction_count += 100
        self.set_degraded()
        self.logger.error("FAULT: Cache invalidated — all entries evicted")
        self.logger.error("FAULT: Hit ratio will drop to 0% until warmup")

    def inject_ttl_expiry(self):
        """Expire all TTLs — simulates clock skew or mass expiry."""
        now = time.time()
        for key in self._cache:
            self._cache[key]["ttl"] = now - 1  # Already expired
        self._is_cold = True
        self.set_degraded()
        self.logger.error("FAULT: All cache TTLs expired simultaneously")

    def reset(self):
        """Reset cache for new episode."""
        super().reset()
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._is_cold = False
        self._seed_cache()
