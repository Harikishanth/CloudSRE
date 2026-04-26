"""
CloudSRE v2 — Storage Service (port 8010).

Object/blob storage (S3-like). Cascade: disk full → storage rejects writes → workers crash.

Fault types:
  - disk_full: Storage reports 100% usage, rejects all writes
  - corruption: Files return checksum errors
"""

import time
from services.base_service import BaseService


class StorageService(BaseService):
    def __init__(self, port: int = 8010, log_dir: str = "/var/log"):
        super().__init__("storage", port=port, log_dir=log_dir)
        self._objects = {}
        self._total_bytes = 0
        self._max_bytes = 10737418240  # 10 GB
        self._read_count = 0
        self._write_count = 0
        self._disk_full = False
        self._corrupted_keys = set()
        self._seed_storage()
        self._register_routes()

    def _seed_storage(self):
        self._objects = {
            "logs/2024/app.log.gz": {"size": 52428800, "checksum": "a1b2c3", "created": time.time() - 86400},
            "backups/db_daily.sql.gz": {"size": 104857600, "checksum": "d4e5f6", "created": time.time() - 3600},
            "uploads/user_avatar_1.png": {"size": 1048576, "checksum": "g7h8i9", "created": time.time() - 7200},
            "exports/report_q4.csv": {"size": 5242880, "checksum": "j0k1l2", "created": time.time() - 172800},
            "config/feature_flags.json": {"size": 4096, "checksum": "m3n4o5", "created": time.time() - 600},
        }
        self._total_bytes = sum(o["size"] for o in self._objects.values())

    def _register_routes(self):
        @self.app.get("/storage/stats")
        async def storage_stats():
            pct = (self._total_bytes / self._max_bytes) * 100
            return {"service": "storage", "used_bytes": self._total_bytes,
                    "max_bytes": self._max_bytes, "usage_pct": round(pct, 1),
                    "object_count": len(self._objects), "reads": self._read_count,
                    "writes": self._write_count, "disk_full": self._disk_full,
                    "corrupted_keys": len(self._corrupted_keys)}

        @self.app.get("/storage/get/{key:path}")
        async def get_object(key: str):
            self._read_count += 1
            if key in self._corrupted_keys:
                return {"error": "CHECKSUM_MISMATCH", "key": key}
            obj = self._objects.get(key)
            if obj:
                return {"key": key, "size": obj["size"], "checksum": obj["checksum"]}
            return {"error": "NOT_FOUND", "key": key}

        @self.app.post("/storage/put/{key:path}")
        async def put_object(key: str, size: int = 1024):
            if self._disk_full:
                return {"error": "DISK_FULL", "usage_pct": 100.0}
            self._objects[key] = {"size": size, "checksum": "new", "created": time.time()}
            self._total_bytes += size
            self._write_count += 1
            return {"status": "stored", "key": key}

        @self.app.post("/storage/cleanup")
        async def cleanup():
            old_keys = [k for k, v in self._objects.items()
                       if time.time() - v["created"] > 604800]
            for k in old_keys:
                self._total_bytes -= self._objects[k]["size"]
                del self._objects[k]
            self._disk_full = False
            return {"cleaned": len(old_keys), "disk_full": False}

    def inject_disk_full(self):
        self._disk_full = True
        self._total_bytes = self._max_bytes
        self.set_degraded()
        self.logger.error("FAULT: Disk full — all writes rejected")

    def inject_corruption(self):
        self._corrupted_keys = set(list(self._objects.keys())[:3])
        self.set_degraded()
        self.logger.error(f"FAULT: {len(self._corrupted_keys)} objects corrupted — checksum failures")

    def reset(self):
        super().reset()
        self._disk_full = False
        self._corrupted_keys.clear()
        self._read_count = 0
        self._write_count = 0
        self._seed_storage()
