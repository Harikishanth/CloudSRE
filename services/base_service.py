"""
CloudSRE v2 — Base Microservice.

Common base class for all 16 microservices. Provides:
  - /healthz endpoint (real health check)
  - /metrics endpoint (real Prometheus-style metrics)
  - /logs endpoint (recent log lines)
  - Middleware for request counting, latency tracking, error recording
  - Structured logging integration
  - Graceful shutdown handling

Each service inherits from this and adds its own routes and business logic.

Kube SRE Gym equivalent: Their services are pre-built Docker images.
They can't add custom endpoints or modify service behavior. We can.
"""

import os
import time
import signal
import asyncio
import threading
from typing import Optional, Dict, Any, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from infra.metrics import ServiceMetrics
from infra.logger import StructuredLogger


class BaseService:
    """Base class for all CloudSRE microservices.

    Provides standardized:
      - Health checking (/healthz)
      - Metrics exposure (/metrics)
      - Log access (/logs)
      - Request middleware (latency, errors, counting)
      - Fault state tracking (is_healthy, is_degraded)

    Usage:
        class PaymentService(BaseService):
            def __init__(self):
                super().__init__("payment", port=8001)
                self.setup_routes()

            def setup_routes(self):
                @self.app.post("/pay")
                async def pay(request: Request):
                    ...
    """

    def __init__(
        self,
        service_name: str,
        port: int,
        log_dir: str = "/var/log",
    ):
        self.service_name = service_name
        self.port = port
        self.metrics = ServiceMetrics(service_name)
        self.logger = StructuredLogger(service_name, log_dir)

        # Health state — can be set by fault injector
        self._is_healthy = True
        self._is_degraded = False
        self._health_error = ""
        self._start_time = time.time()

        # Active fault state — set via /fault_inject HTTP endpoint
        self._active_fault: Optional[str] = None
        self._fault_params: Dict[str, Any] = {}

        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title=f"CloudSRE — {service_name}",
            description=f"Real {service_name} microservice for CloudSRE v2 environment",
            version="1.0.0",
        )

        # Register common endpoints
        self._register_common_routes()
        self._register_middleware()

    def _register_common_routes(self):
        """Register /healthz, /metrics, /logs endpoints."""

        @self.app.get("/healthz")
        async def healthz():
            """Real health check endpoint.

            Returns actual service health based on fault state.
            When a fault is injected via /fault_inject, this endpoint
            returns REAL HTTP error codes (503, 429, 507) — not flags.
            """
            uptime = round(time.time() - self._start_time, 1)

            if not self._is_healthy:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "service": self.service_name,
                        "error": self._health_error,
                        "fault_type": self._active_fault,
                        "uptime_seconds": uptime,
                    },
                )

            if self._is_degraded:
                # Degraded faults return specific HTTP codes
                status_code = self._fault_status_code()
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "status": "degraded",
                        "service": self.service_name,
                        "error": self._health_error,
                        "fault_type": self._active_fault,
                        "uptime_seconds": uptime,
                        "error_rate": self.metrics.error_rate,
                    },
                )

            return {
                "status": "healthy",
                "service": self.service_name,
                "uptime_seconds": uptime,
                "error_rate": self.metrics.error_rate,
                "latency_p95_ms": self.metrics.latency.percentile(95),
                "active_connections": int(self.metrics.active_connections.value),
            }

        @self.app.post("/fault_inject")
        async def fault_inject(request: Request):
            """Real fault injection endpoint.

            Called by the orchestrator to make this service ACTUALLY broken.
            The service will start returning real HTTP errors from /healthz.
            """
            body = await request.json()
            fault_type = body.get("fault_type", "generic")
            params = body.get("params", {})
            severity = params.get("severity", "degraded")

            self._active_fault = fault_type
            self._fault_params = params
            self._health_error = params.get("error_message", f"Fault active: {fault_type}")

            if severity == "critical":
                self._is_healthy = False
                self._is_degraded = False
            else:
                self._is_healthy = True
                self._is_degraded = True

            self.logger.error(f"FAULT INJECTED: {fault_type} — {self._health_error}")
            return {"status": "fault_injected", "type": fault_type, "severity": severity}

        @self.app.post("/fault_clear")
        async def fault_clear(request: Request):
            """Clear active fault — service recovers to healthy state."""
            old_fault = self._active_fault
            self._active_fault = None
            self._fault_params = {}
            self._is_healthy = True
            self._is_degraded = False
            self._health_error = ""
            self.logger.info(f"FAULT CLEARED: {old_fault} — service restored")
            return {"status": "cleared", "previous_fault": old_fault}

        @self.app.get("/metrics")
        async def metrics():
            """Real Prometheus-style metrics endpoint.

            The agent calls this via:
                curl http://localhost:<port>/metrics

            Returns real request counts, error rates, latency histograms,
            CPU/memory gauges, and service-specific custom metrics.
            """
            return self.metrics.to_dict()

        @self.app.get("/metrics/prometheus")
        async def metrics_prometheus():
            """Prometheus text exposition format."""
            return Response(
                content=self.metrics.to_prometheus(),
                media_type="text/plain",
            )

        @self.app.get("/logs")
        async def logs(log_type: str = "error", lines: int = 20):
            """Return recent log lines.

            The agent can also use `cat /var/log/<service>/error.log | tail -20`
            but this endpoint provides the same data over HTTP.

            Args:
                log_type: "error" or "access"
                lines: number of recent lines to return
            """
            content = self.logger.get_recent(log_type, lines)
            return {
                "service": self.service_name,
                "log_type": log_type,
                "lines": lines,
                "content": content,
                "line_counts": self.logger.get_log_count(),
            }

    def _register_middleware(self):
        """Request middleware — tracks latency, errors, and connection count."""

        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            # Track active connections
            self.metrics.active_connections.inc()
            start = time.time()

            try:
                response = await call_next(request)
                duration_ms = (time.time() - start) * 1000
                is_error = response.status_code >= 400
                self.metrics.record_request(duration_ms, is_error)

                if is_error:
                    self.logger.error(
                        f"HTTP {response.status_code} on {request.method} {request.url.path}",
                        {"duration_ms": round(duration_ms, 1)},
                    )

                return response
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                self.metrics.record_request(duration_ms, is_error=True)
                self.logger.error(
                    f"Unhandled exception on {request.method} {request.url.path}: {str(e)}",
                    {"duration_ms": round(duration_ms, 1), "error_type": type(e).__name__},
                )
                raise
            finally:
                self.metrics.active_connections.dec()

    # ── Fault Injection Interface ────────────────────────────────────────

    def _fault_status_code(self) -> int:
        """Map active fault type to an HTTP status code.

        This makes degraded faults return appropriate HTTP error codes:
        - Rate limiting: 429 Too Many Requests
        - Disk/storage full: 507 Insufficient Storage
        - Config locked: 423 Locked
        - Everything else: 503 Service Unavailable
        """
        fault = self._active_fault or ""
        if fault in ("rate_limit_zero",):
            return 429
        elif fault in ("disk_full", "retention_full", "email_queue_overflow"):
            return 507
        elif fault in ("config_locked",):
            return 423
        return 503

    def set_unhealthy(self, error: str = "Service unavailable"):
        """Mark service as unhealthy. /healthz will return 503."""
        self._is_healthy = False
        self._health_error = error
        self.logger.error(f"Service marked unhealthy: {error}")

    def set_healthy(self):
        """Restore service health."""
        self._is_healthy = True
        self._is_degraded = False
        self._health_error = ""
        self._active_fault = None
        self._fault_params = {}
        self.logger.info("Service restored to healthy state")

    def set_degraded(self):
        """Mark service as degraded (still running but slow/errors)."""
        self._is_degraded = True
        self.logger.warn("Service marked as degraded")

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        """Reset service state for new episode."""
        self._is_healthy = True
        self._is_degraded = False
        self._health_error = ""
        self._active_fault = None
        self._fault_params = {}
        self._start_time = time.time()
        self.metrics.reset()
        self.logger.reset()

    # ── Process Management ───────────────────────────────────────────────

    def get_process_info(self) -> Dict[str, Any]:
        """Return process info for the agent's diagnostic commands.

        The agent can get this via:
            curl http://localhost:<port>/process
        """
        import psutil
        try:
            proc = psutil.Process(os.getpid())
            return {
                "service": self.service_name,
                "pid": proc.pid,
                "cpu_percent": proc.cpu_percent(),
                "memory_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
                "threads": proc.num_threads(),
                "status": proc.status(),
                "create_time": proc.create_time(),
            }
        except Exception:
            return {
                "service": self.service_name,
                "pid": os.getpid(),
                "error": "psutil not available",
            }
