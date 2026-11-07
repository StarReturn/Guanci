"""JWT authentication middleware (replaces API Key on client surface).

Strategy:
1. Try to extract a Bearer access token from ``Authorization`` header.
2. If present, decode + verify via :mod:`app.core.jwt_utils`.
3. On success: attach ``request.state.user_id`` / ``request.state.role``.
4. On failure (bad/expired/no token): 401.
5. Public paths (/health, /api/v1/auth/sms/send-code, /api/v1/auth/sms/login,
   /api/v1/auth/wechat/login, /api/v1/auth/refresh, /docs, /metrics) skip auth.

This middleware runs AFTER the API-Key check in client_main.py? No — we
REPLACE the API-Key check on client app. Admin app keeps API Key.
"""
from __future__ import annotations

import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.jwt_utils import JWTError, decode

logger = logging.getLogger(__name__)


# Paths that don't require a JWT. Login / refresh / register / public
# health checks live here.
PUBLIC_PATHS: tuple[str, ...] = (
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    # SMS auth
    "/api/v1/auth/sms/send-code",
    "/api/v1/auth/sms/login",
    # Password auth (Q10)
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    # WeChat
    "/api/v1/auth/wechat/url",
    "/api/v1/auth/wechat/login",
    # Token refresh
    "/api/v1/auth/refresh",
    # App 版本检查（未登录也应能检查更新）
    "/api/v1/app/version",
    # Payment providers authenticate callbacks with channel signatures.
    # The handlers perform signature/JWS verification before any state change.
    "/api/v1/payments/webhooks/apple",
    "/api/v1/payments/webhooks/wechat",
    "/api/v1/payments/webhooks/alipay",
    # P3 batch 4: WebSocket "phone call" dialogue. The HTTP middleware
    # can't read the Authorization header on a browser-initiated WS
    # handshake (browsers don't allow setting headers on WS), so we
    # let the request through and the WS handler authenticates via
    # the ``?token=<jwt>`` query parameter. Configure your reverse
    # proxy to scrub ``token=`` from access logs.
    "/api/v1/realtime/stream",
)


def _is_public(path: str) -> bool:
    if path == "/api/v1/auth/dev/login":
        from app.core.config import settings
        return settings.app_env.strip().lower() in {"local", "dev", "test"}
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS)


def make_jwt_middleware() -> "callable":
    """Authenticates via Bearer JWT.

    Falls back to CLIENT_API_KEY env-var set (legacy mode) if the
    request has ``X-API-Key`` instead of ``Authorization``. Useful
    for gradual migration: deploy new client app with JWT, old app
    version still uses API key.
    """

    async def _check(request: Request, call_next):
        # CORS preflight - always pass through to CORSMiddleware.
        if request.method == "OPTIONS":
            return await call_next(request)

        if _is_public(request.url.path):
            return await call_next(request)

        # ---- Try JWT first ----
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                payload = decode(token)
            except JWTError as exc:
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"invalid token: {exc}"},
                )
            if payload.get("type") != "access":
                return JSONResponse(
                    status_code=401, content={"detail": "wrong token type"}
                )
            request.state.user_id = payload["sub"]
            request.state.role = payload.get("role", "user")
            request.state.jit = payload.get("jit")
            return await call_next(request)

        # ---- Fallback: legacy API key (CLIENT_API_KEY env + DB) ----
        api_key = request.headers.get("X-API-Key", "").strip()
        if api_key:
            try:
                from app.services.api_key_repository import api_key_repository
                if api_key_repository.verify(api_key):
                    request.state.user_id = None  # legacy mode: no user binding
                    request.state.role = "client"
                    return await call_next(request)
            except Exception as exc:
                logger.warning("jwt_middleware_legacy_key_check_failed: %s", exc)

        return JSONResponse(
            status_code=401,
            content={"detail": "missing or invalid Authorization header"},
        )

    return _check


def get_request_user_id(request: Request) -> str | None:
    """Helper for route handlers to fetch user_id from middleware state."""
    return getattr(request.state, "user_id", None)


def get_request_user_role(request: Request) -> str:
    return getattr(request.state, "role", "user")


__all__ = ["make_jwt_middleware", "get_request_user_id", "get_request_user_role"]
