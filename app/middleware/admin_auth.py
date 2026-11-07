"""Admin auth middleware — combines JWT(role=admin) + API Key fallback.

Auth priority:
1. Public path (e.g. /api/v1/admin/auth/login) → pass
2. Authorization: Bearer <jwt> with role=admin → pass, set request.state.user_id
3. X-API-Key: <ADMIN_API_KEY> → pass (backward compat)
4. Anything else → 401

A regular user JWT (role=user) is rejected with 403 even if valid —
admin surface must not be reachable from client-side tokens.
"""
from __future__ import annotations

import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.jwt_utils import JWTError, decode
from app.core.config import settings
from app.middleware.api_key import split_keys


# Paths on admin surface that don't require auth (admin login itself).
_ADMIN_PUBLIC_PATHS: tuple[str, ...] = (
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    "/api/v1/admin/auth/login",   # admin login endpoint
)


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _ADMIN_PUBLIC_PATHS)


def make_admin_auth_middleware() -> "callable":
    """Verify JWT(role=admin) OR API Key. Fail-closed in production."""

    async def _check(request: Request, call_next):
        path = request.url.path

        # 0. CORS preflight - always pass through to CORSMiddleware.
        if request.method == "OPTIONS":
            return await call_next(request)

        # 1. Public paths (admin login, health, metrics)
        if _is_public(path):
            return await call_next(request)

        # 2. Try JWT first
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                payload = decode(token)
            except JWTError as exc:
                return JSONResponse(status_code=401, content={"detail": f"invalid token: {exc}"})

            if payload.get("type") != "access":
                return JSONResponse(status_code=401, content={"detail": "wrong token type"})

            role = payload.get("role", "user")
            if role not in ("admin", "super_admin"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"admin access required (your role: {role})"},
                )

            request.state.user_id = payload["sub"]
            request.state.role = role  # "admin" or "super_admin"
            request.state.jit = payload.get("jit")
            request.state.authenticated_by = "jwt"
            return await call_next(request)

        # 3. Try API Key (backward compat / bootstrap)
        api_key = request.headers.get("X-API-Key", "").strip()
        admin_keys = split_keys(settings.admin_api_key)
        if api_key and any(secrets.compare_digest(api_key, key) for key in admin_keys):
            request.state._admin_api_key_passed = True  # type: ignore[attr-defined]
            request.state.user_id = None
            request.state.role = "admin"
            request.state.authenticated_by = "api_key"
            return await call_next(request)

        # 4. No valid credential
        return JSONResponse(
            status_code=401,
            content={"detail": "missing or invalid credential (need admin JWT or API Key)"},
        )

    return _check


__all__ = ["make_admin_auth_middleware"]
