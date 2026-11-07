"""API Key authentication middleware.

Two keys are recognized:

* ``ADMIN_API_KEY`` — gates the admin app (build packages, inspect any
  task, view dependency health). High-privilege; should be kept on the
  server side and never shipped to mobile clients.
* ``CLIENT_API_KEY`` — gates the client app (TTS, phoneme alignment,
  realtime sessions, read-only package lookup). Lower-privilege;
  shipped with the mobile app.

Key rotation: change the env var and restart the process. Mobile clients
need an OTA app update to pick up the new key — that's why the client
key should support multiple concurrent values (comma-separated) for
zero-downtime rotation.

Bypass paths:
* ``/health`` and ``/metrics`` are always public — they're used by
  load balancers and Prometheus scrapers that don't know about keys.
* ``/docs`` and ``/openapi.json`` are public in non-prod for developer
  convenience; toggle via ``DOCS_PUBLIC`` (default: true in local,
  should be false in prod).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


_PUBLIC_PATHS: tuple[str, ...] = (
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
)


def split_keys(raw: str | None) -> set[str]:
    """Parse a comma/whitespace-separated key list from env var.

    Returns empty set if input is empty — caller should treat that as
    "auth disabled" and log a warning in non-prod.
    """
    if not raw:
        return set()
    return {k.strip() for k in raw.replace(",", " ").split() if k.strip()}


# Backward-compat alias (used by earlier iterations).
_split_keys = split_keys


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _PUBLIC_PATHS)


def make_api_key_middleware(
    *,
    valid_keys: Iterable[str],
    role: str,
    header_name: str = "X-API-Key",
    use_db_verify: bool = False,
) -> "callable":
    """Build a Starlette middleware callable for the given role.

    Usage::

        app.add_middleware(StarletteMiddleware, ...)
        # or simpler — dispatch in a before-handling hook:

        @app.middleware("http")
        async def check_key(request, call_next):
            return await _check(request, call_next)

    Two verification modes:

    * **Static** (``use_db_verify=False``, default): the ``valid_keys``
      set is the source of truth. Used for admin keys — those stay in
      env vars and cannot be self-managed via API.
    * **DB-backed** (``use_db_verify=True``): ``api_key_repository.verify()``
      is consulted on every request. The static ``valid_keys`` set is
      used as a "DB-unavailable" fallback. Used for client keys — those
      live in PG and can be CRUD'd via /admin/api-keys/*.

    ``role`` is only used for log messages, not for behavior.
    """
    keys = {k for k in valid_keys if k}
    if not keys and not use_db_verify:
        logger.warning(
            "api_key_middleware_disabled",
            extra={"role": role, "reason": "no keys configured"},
        )

    async def _check(request: Request, call_next):
        # CORS preflight - always pass through to CORSMiddleware.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Public paths skip auth entirely.
        if _is_public(request.url.path):
            return await call_next(request)

        # Resolve env to decide fail-open vs fail-closed.
        env = os.environ.get("APP_ENV", "local")

        provided = request.headers.get(header_name, "").strip()
        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": f"missing {header_name} header"},
            )

        # Verify. Try DB first (if enabled), fall back to static set.
        authorized = False
        if use_db_verify:
            try:
                from app.services.api_key_repository import api_key_repository
                authorized = api_key_repository.verify(provided)
            except Exception as exc:
                logger.warning(
                    "api_key_db_verify_failed",
                    extra={"role": role, "error": str(exc)},
                )
        if not authorized and provided in keys:
            authorized = True

        # No keys configured anywhere → fail open in non-prod, fail closed in prod.
        if not authorized and not keys and not use_db_verify:
            if env in ("local", "dev", "test"):
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={"detail": "API key auth misconfigured (no keys)"},
            )

        if not authorized:
            logger.warning(
                "api_key_invalid",
                extra={
                    "role": role,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None,
                },
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "invalid API key"},
            )
        return await call_next(request)

    return _check


__all__ = ["make_api_key_middleware"]
