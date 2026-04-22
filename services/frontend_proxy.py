"""
CloudSRE v2 — Frontend Proxy Service (port 8004).

A REAL reverse proxy that routes requests to payment and auth services.
This is the entry point — when end users get errors, they hit this service.

The frontend proxy:
  - Routes /api/pay → payment:8001/pay
  - Routes /api/auth → auth:8002/auth/token
  - Aggregates health from all upstream services
  - Returns real 502 Bad Gateway when upstreams are down

The agent often sees frontend errors FIRST (502s), but the root cause
is usually in payment, auth, or the database. This tests whether the
agent investigates upstream instead of trying to fix frontend.
"""

import os
import time
import httpx
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse

from cloud_sre_v2.services.base_service import BaseService


class FrontendProxy(BaseService):
    """Real reverse proxy microservice.

    Routes:
        GET  /                 — Frontend status page
        POST /api/pay          — Proxy to payment:8001/pay
        POST /api/auth         — Proxy to auth:8002/auth/token
        GET  /api/auth/verify  — Proxy to auth:8002/auth/verify
        GET  /upstream/health  — Aggregated upstream health
        GET  /healthz          — (inherited)
        GET  /metrics          — (inherited)
    """

    def __init__(
        self,
        port: int = 8004,
        log_dir: str = "/var/log",
        payment_url: str = "http://localhost:8001",
        auth_url: str = "http://localhost:8002",
        worker_url: str = "http://localhost:8003",
    ):
        super().__init__("frontend", port=port, log_dir=log_dir)
        self._pid = os.getpid()
        self._payment_url = payment_url
        self._auth_url = auth_url
        self._worker_url = worker_url

        # Frontend-specific metrics
        self.proxy_errors = self.metrics.add_counter("proxy_errors")
        self.upstream_502s = self.metrics.add_counter("upstream_502_count")

        self._setup_routes()

    def _setup_routes(self):

        @self.app.get("/")
        async def index():
            """Frontend status page."""
            return {
                "service": "frontend-proxy",
                "version": "1.0.0",
                "routes": [
                    "POST /api/pay → payment service",
                    "POST /api/auth → auth service",
                    "GET /api/auth/verify → auth service",
                    "GET /upstream/health → aggregated health",
                ],
            }

        @self.app.post("/api/pay")
        async def proxy_pay(request: Request):
            """Proxy payment requests to payment service."""
            return await self._proxy_request("POST", f"{self._payment_url}/pay", request)

        @self.app.post("/api/auth")
        async def proxy_auth(request: Request):
            """Proxy auth requests to auth service."""
            return await self._proxy_request("POST", f"{self._auth_url}/auth/token", request)

        @self.app.get("/api/auth/verify")
        async def proxy_auth_verify(request: Request):
            """Proxy auth verify requests."""
            return await self._proxy_request("GET", f"{self._auth_url}/auth/verify", request)

        @self.app.get("/upstream/health")
        async def upstream_health():
            """Check health of ALL upstream services.

            This is what the agent should check FIRST — it shows
            which upstream services are failing.
            """
            health = {}
            upstream_services = {
                "payment": f"{self._payment_url}/healthz",
                "auth": f"{self._auth_url}/healthz",
                "worker": f"{self._worker_url}/healthz",
            }

            async with httpx.AsyncClient(timeout=5.0) as client:
                for name, url in upstream_services.items():
                    try:
                        resp = await client.get(url)
                        health[name] = {
                            "status": "healthy" if resp.status_code == 200 else "unhealthy",
                            "status_code": resp.status_code,
                            "response": resp.json() if resp.status_code == 200 else None,
                        }
                    except httpx.ConnectError:
                        health[name] = {"status": "unreachable", "error": "Connection refused"}
                    except httpx.TimeoutException:
                        health[name] = {"status": "timeout", "error": "Health check timed out"}
                    except Exception as e:
                        health[name] = {"status": "error", "error": str(e)}

            all_healthy = all(s["status"] == "healthy" for s in health.values())
            return {
                "overall": "healthy" if all_healthy else "DEGRADED",
                "services": health,
            }

    async def _proxy_request(self, method: str, url: str, request: Request) -> JSONResponse:
        """Forward a request to an upstream service."""
        try:
            headers = dict(request.headers)
            # Remove host header to avoid conflicts
            headers.pop("host", None)

            async with httpx.AsyncClient(timeout=10.0) as client:
                if method == "POST":
                    try:
                        body = await request.body()
                    except Exception:
                        body = b"{}"
                    resp = await client.post(url, content=body, headers=headers)
                else:
                    resp = await client.get(url, headers=headers)

                # Return upstream response as-is
                try:
                    return JSONResponse(
                        status_code=resp.status_code,
                        content=resp.json(),
                    )
                except Exception:
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"raw": resp.text[:500]},
                    )

        except httpx.ConnectError:
            self.proxy_errors.inc()
            self.upstream_502s.inc()
            self.logger.error(f"Upstream connection refused: {url}")
            return JSONResponse(
                status_code=502,
                content={"error": f"Bad Gateway — upstream {url} connection refused"},
            )
        except httpx.TimeoutException:
            self.proxy_errors.inc()
            self.logger.error(f"Upstream timeout: {url}")
            return JSONResponse(
                status_code=504,
                content={"error": f"Gateway Timeout — upstream {url} timed out"},
            )
        except Exception as e:
            self.proxy_errors.inc()
            self.logger.error(f"Proxy error: {e}", {"url": url})
            return JSONResponse(
                status_code=502,
                content={"error": f"Bad Gateway — {str(e)}"},
            )

    def reset(self):
        super().reset()
        self._pid = os.getpid()
