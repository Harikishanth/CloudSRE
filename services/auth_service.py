"""
CloudSRE v2 — Auth Service (port 8002).

A REAL JWT authentication service that:
  - Signs and validates JWT tokens with real cryptographic keys
  - Stores sessions in real SQLite database
  - Tracks auth failure rates via real metrics
  - Has real failure modes (cert expiry, key mismatch, rate limit)

When the JWT signing key is rotated mid-episode:
  - ALL existing tokens become invalid (REAL invalidation)
  - Services calling /auth/verify get REAL 401 Unauthorized
  - This cascades to payment (can't auth) → worker (can't process) → frontend (502)
"""

import os
import time
import jwt
import sqlite3
import secrets
from typing import Optional
from fastapi import Request, Header
from fastapi.responses import JSONResponse

from cloud_sre_v2.services.base_service import BaseService
from cloud_sre_v2.infra.database import Database


class AuthService(BaseService):
    """Real JWT authentication microservice.

    Routes:
        POST /auth/token    — Issue a new JWT token (real signing)
        GET  /auth/verify   — Verify a JWT token (real validation)
        POST /auth/rotate   — Rotate the signing key (causes cascade!)
        GET  /auth/sessions — List active sessions
        GET  /healthz       — (inherited)
        GET  /metrics       — (inherited)
    """

    def __init__(
        self,
        database: Database,
        port: int = 8002,
        log_dir: str = "/var/log",
    ):
        super().__init__("auth", port=port, log_dir=log_dir)
        self.database = database
        self._pid = os.getpid()

        # JWT configuration — real keys, real signing
        self._signing_key = secrets.token_hex(32)
        self._key_version = 1
        self._token_ttl = 3600  # 1 hour
        self._algorithm = "HS256"

        # Rate limiting
        self._rate_limit = 100  # requests per minute
        self._request_window: list = []

        # Auth-specific metrics
        self.tokens_issued = self.metrics.add_counter("tokens_issued")
        self.tokens_verified = self.metrics.add_counter("tokens_verified")
        self.tokens_rejected = self.metrics.add_counter("tokens_rejected")
        self.auth_failures = self.metrics.add_counter("auth_failures")

        # Fault state
        self._cert_expired = False
        self._rate_limit_broken = False

        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/auth/token")
        async def issue_token(request: Request):
            """Issue a real JWT token with real cryptographic signing."""
            if self._cert_expired:
                self.auth_failures.inc()
                self.logger.error("Certificate expired — cannot sign tokens")
                return JSONResponse(
                    status_code=503,
                    content={"error": "TLS certificate expired — cannot sign tokens"},
                )

            try:
                body = await request.json()
            except Exception:
                body = {"user_id": "test_user"}

            user_id = body.get("user_id", "anonymous")

            # Real JWT signing
            now = time.time()
            payload = {
                "user_id": user_id,
                "iat": int(now),
                "exp": int(now + self._token_ttl),
                "key_version": self._key_version,
            }
            token = jwt.encode(payload, self._signing_key, algorithm=self._algorithm)

            # Store session in real database
            try:
                self.database.execute(
                    "INSERT INTO sessions (user_id, token, expires_at, is_valid) VALUES (?, ?, datetime('now', '+1 hour'), 1)",
                    (user_id, token[:50]),  # Store truncated token
                )
            except sqlite3.OperationalError as e:
                self.logger.warn(f"Could not store session: {e}")

            self.tokens_issued.inc()
            self.logger.info(f"Token issued for {user_id}", {"key_version": self._key_version})

            return {"token": token, "expires_in": self._token_ttl, "key_version": self._key_version}

        @self.app.get("/auth/verify")
        async def verify_token(authorization: Optional[str] = Header(None)):
            """Verify a JWT token — real cryptographic validation."""
            if self._cert_expired:
                self.auth_failures.inc()
                return JSONResponse(
                    status_code=503,
                    content={"error": "Certificate expired — cannot verify tokens"},
                )

            if not authorization:
                self.tokens_rejected.inc()
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing Authorization header"},
                )

            token = authorization.replace("Bearer ", "")

            try:
                decoded = jwt.decode(token, self._signing_key, algorithms=[self._algorithm])
                self.tokens_verified.inc()
                return {"valid": True, "user_id": decoded.get("user_id"), "key_version": decoded.get("key_version")}
            except jwt.ExpiredSignatureError:
                self.tokens_rejected.inc()
                self.logger.warn("Token expired")
                return JSONResponse(status_code=401, content={"error": "Token expired"})
            except jwt.InvalidSignatureError:
                # This happens after key rotation — tokens signed with old key are INVALID
                self.tokens_rejected.inc()
                self.auth_failures.inc()
                self.logger.error(
                    "JWT signature mismatch — possible key rotation issue",
                    {"current_key_version": self._key_version},
                )
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid signature — signing key mismatch"},
                )
            except jwt.DecodeError as e:
                self.tokens_rejected.inc()
                return JSONResponse(status_code=401, content={"error": f"Invalid token: {e}"})

        @self.app.post("/auth/rotate")
        async def rotate_key():
            """Rotate the JWT signing key.

            WARNING: This invalidates ALL existing tokens!
            This is a CASCADE TRIGGER — after rotation, every service
            that holds a cached token will get 401 on next request.
            """
            old_version = self._key_version
            self._signing_key = secrets.token_hex(32)
            self._key_version += 1

            # Invalidate all sessions in DB
            try:
                self.database.execute("UPDATE sessions SET is_valid = 0")
            except sqlite3.OperationalError:
                pass

            self.logger.warn(
                f"JWT signing key rotated: v{old_version} → v{self._key_version}",
                {"old_version": old_version, "new_version": self._key_version},
            )
            return {"status": "rotated", "new_key_version": self._key_version}

        @self.app.get("/auth/sessions")
        async def list_sessions(limit: int = 20):
            """List active sessions from real database."""
            try:
                rows = self.database.query(
                    "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
                )
                return {"sessions": rows, "count": len(rows)}
            except sqlite3.OperationalError as e:
                return JSONResponse(status_code=503, content={"error": str(e)})

    # ── Fault Injection ──────────────────────────────────────────────────

    def inject_cert_expiry(self):
        """Expire the TLS certificate — all auth operations fail."""
        self._cert_expired = True
        self.set_unhealthy("TLS certificate expired")
        self.logger.error("TLS certificate expired at 00:00:00 UTC")

    def inject_key_mismatch(self):
        """Rotate key silently — existing tokens become invalid."""
        self._signing_key = secrets.token_hex(32)
        self._key_version += 1
        self.logger.error(
            f"CONFIG_VERSION=v{self._key_version} but tokens signed with v{self._key_version - 1} key"
        )

    def reset(self):
        super().reset()
        self._signing_key = secrets.token_hex(32)
        self._key_version = 1
        self._cert_expired = False
        self._rate_limit_broken = False
        self._pid = os.getpid()
