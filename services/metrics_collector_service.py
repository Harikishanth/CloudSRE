"""
CloudSRE v2 — Metrics Service (port 8011).

Prometheus-like metrics collector. Cascade: metrics down → no alerting → silent failures.

Fault types:
  - scrape_failure: Can't scrape targets, metrics stale
  - retention_full: Metrics storage full, dropping new data
"""

import time
from cloud_sre_v2.services.base_service import BaseService


class MetricsCollectorService(BaseService):
    def __init__(self, port: int = 8011, log_dir: str = "/var/log"):
        super().__init__("metrics_collector", port=port, log_dir=log_dir)
        self._series = {}
        self._scrape_targets = ["payment:8001", "auth:8002", "worker:8003",
                                "frontend:8004", "cache:8005", "notification:8006"]
        self._scrape_failures = 0
        self._total_scrapes = 0
        self._retention_full = False
        self._alerting_active = True
        self._active_alerts = []
        self._seed_metrics()
        self._register_routes()

    def _seed_metrics(self):
        now = time.time()
        self._series = {
            "http_requests_total": {"value": 48392, "labels": {"method": "POST"}, "ts": now},
            "http_errors_total": {"value": 127, "labels": {"code": "500"}, "ts": now},
            "db_connections_active": {"value": 24, "labels": {}, "ts": now},
            "queue_depth": {"value": 0, "labels": {}, "ts": now},
            "cache_hit_ratio": {"value": 0.94, "labels": {}, "ts": now},
            "memory_usage_bytes": {"value": 734003200, "labels": {"service": "worker"}, "ts": now},
            "cpu_usage_percent": {"value": 23.4, "labels": {"service": "payment"}, "ts": now},
            "p99_latency_ms": {"value": 145, "labels": {"endpoint": "/pay"}, "ts": now},
        }
        self._total_scrapes = 2847

    def _register_routes(self):
        @self.app.get("/metrics/query")
        async def query_metric(name: str = ""):
            if name and name in self._series:
                return {"metric": name, "data": self._series[name]}
            return {"metrics": list(self._series.keys()), "total": len(self._series)}

        @self.app.get("/metrics/stats")
        async def metrics_stats():
            return {"service": "metrics_collector", "series_count": len(self._series),
                    "scrape_targets": len(self._scrape_targets),
                    "scrape_failures": self._scrape_failures,
                    "total_scrapes": self._total_scrapes,
                    "retention_full": self._retention_full,
                    "alerting_active": self._alerting_active,
                    "active_alerts": self._active_alerts}

        @self.app.get("/metrics/alerts")
        async def list_alerts():
            return {"alerts": self._active_alerts, "alerting_active": self._alerting_active}

        @self.app.post("/metrics/scrape")
        async def force_scrape():
            if self._retention_full:
                return {"error": "RETENTION_FULL", "message": "Cannot store new metrics"}
            self._total_scrapes += len(self._scrape_targets)
            return {"scraped": len(self._scrape_targets), "failures": self._scrape_failures}

    def inject_scrape_failure(self):
        self._scrape_failures = len(self._scrape_targets)
        self._alerting_active = False
        self.set_degraded()
        self.logger.error("FAULT: All scrape targets failing — metrics stale, alerting DISABLED")

    def inject_retention_full(self):
        self._retention_full = True
        self.set_degraded()
        self.logger.error("FAULT: Metrics retention full — dropping new data points")

    def reset(self):
        super().reset()
        self._scrape_failures = 0
        self._retention_full = False
        self._alerting_active = True
        self._active_alerts.clear()
        self._seed_metrics()
