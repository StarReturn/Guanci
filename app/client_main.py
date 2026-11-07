"""Client app entrypoint.

Run with:
    uvicorn app.client_main:app --host 0.0.0.0 --port 8091

Exposes:
    * /api/v1/tts/synthesize          (rate-limited)
    * /api/v1/phonemes/align          (rate-limited)
    * /api/v1/realtime/sessions       (rate-limited)
    * /api/v1/keyframes/packages/{avatar_id}  (read-only, rate-limited)
    * /api/v1/keyframes/packages/download     (read-only, rate-limited)
    * /metrics
    * /health (liveness, no auth)

Auth: requires X-API-Key matching CLIENT_API_KEY (comma-separated).
Rate limit: per-IP throttling via slowapi decorators.

Build / task-management endpoints are NOT mounted here — a leaked
client key cannot trigger expensive GPU builds.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import client_router
from app.api.v2.router import client_v2_router
from app.core.config import settings, validate_production_security
from app.core.logging import configure_logging
from app.core.metrics import setup_metrics
from app.middleware.api_key import make_api_key_middleware, split_keys
from app.middleware.rate_limit import limiter
from app.services.redis_client import redis_client

logger = logging.getLogger(__name__)


def _build_limits_table() -> list[tuple[str, str]]:
    """Build path-prefix → rate-limit-string table from settings.

    Lives in a function so it picks up env-var changes at runtime rather
    than baking constants at module import time.
    """
    return [
        ("/api/v1/tts/synthesize", settings.rate_limit_tts),
        ("/api/v1/phonemes/align", settings.rate_limit_phoneme),
        ("/api/v1/realtime/sessions", settings.rate_limit_realtime),
        ("/api/v1/keyframes/packages", settings.rate_limit_package_read),
    ]


# Path patterns we never rate-limit even on the client app.
# Add to this set if you mount new diagnostic routes that should be
# exempt from throttling.
_UNLIMITED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
})


def _attach_rate_limits(app: FastAPI) -> None:
    """Walk every route on ``app`` and attach the matching limiter rule.

    Design points:

    * **Idempotent across create_client_app() calls**: we tag the
      underlying function object (``endpoint.__wrapped_by_dh_limiter``)
      instead of the route, so the same endpoint shared by two app
      instances isn't decorated twice (which would stack two limiters
      and double the effective rate).
    * **Safe if endpoint lacks ``request: Request``**: we log and skip
      rather than crash — admin routes reused as read-only client
      routes may have been declared without the param.
    """
    table = _build_limits_table()

    for route in app.routes:
        path = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        if not path or not endpoint or path in _UNLIMITED_PATHS:
            continue

        # Match by longest-prefix so /keyframes/packages/* wins over
        # /keyframes/* if both rules ever exist.
        matched = next(
            (limit for prefix, limit in table
             if path == prefix or path.startswith(prefix.rstrip("/") + "/")),
            None,
        )
        if matched is None:
            continue

        # Idempotency: once is enough per endpoint object.
        if getattr(endpoint, "__wrapped_by_dh_limiter", False):
            continue

        # slowapi wraps the function; the wrapped version still has the
        # same signature so FastAPI keeps working. We replace route.endpoint
        # in-place so the limiter kicks in when FastAPI dispatches.
        try:
            wrapped = limiter.limit(matched)(endpoint)
            wrapped.__wrapped_by_dh_limiter = True  # type: ignore[attr-defined]
            route.endpoint = wrapped
        except Exception as exc:  # slowapi raises if no request param
            logger.warning(
                "rate_limit_attach_failed",
                extra={"path": path, "error": str(exc)},
            )


def _validate_cors_for_production() -> None:
    """Reject insecure CORS in production environments.

    `allow_origins=["*"]` + `allow_credentials=False` is technically
    valid but means *anyone* can read responses from a browser. For
    a client API that ships the API key in a header (not cookie), this
    is acceptable for POC but unsafe for production — an attacker who
    tricks a logged-in user into visiting their site can read the
    user's responses (including phoneme_timeline and download URLs).

    Production deployments must set CORS_ORIGINS to concrete origins.
    """
    env = (settings.app_env or "local").lower()
    if env in ("production", "prod"):
        origins = [o.strip() for o in settings.cors_origins if o and o.strip()]
        if not origins or "*" in origins:
            raise RuntimeError(
                "CORS_ORIGINS must list concrete origins when APP_ENV=production; "
                f"got {settings.cors_origins!r}. Update .env."
            )


def create_client_app() -> FastAPI:
    configure_logging()
    validate_production_security(role="client")
    _validate_cors_for_production()

    app = FastAPI(
        title=f"{settings.app_name} (Client)",
        version="1.0.0",
        description="客户端 (手机 App)：TTS / 音素对齐 / 实时会话 / 素材包只读访问。需 X-API-Key + 限流。",
    )

    # ---- slowapi state ----
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ---- CORS ----
    # IMPORTANT: never combine allow_credentials=True with origins=["*"].
    # CORS spec rejects this and browsers will fail silently. We default
    # to credentials=False so "*" works; flip to True + concrete origins
    # only if your client relies on cookies (we don't — X-API-Key is in
    # a header).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        max_age=3600,
    )

    # ---- Auth: P3 user system uses JWT (with legacy API-Key fallback) ----
    # Mobile apps that have migrated to user login send
    # ``Authorization: Bearer <jwt>``. Legacy clients can still send
    # ``X-API-Key`` — the JWT middleware falls through to the API-Key
    # repository in that case.
    from app.middleware.jwt import make_jwt_middleware
    app.middleware("http")(make_jwt_middleware())

    # ---- Routes ----
    app.include_router(client_router, prefix=settings.api_prefix)
    app.include_router(client_v2_router, prefix="/api/v2")

    # ---- Post-route: attach per-endpoint rate limits ----
    _attach_rate_limits(app)

    # Metrics — public path (no key), used by Prometheus.
    setup_metrics(app)

    # 开源版：会员/支付维护任务不随包分发（完整版含配额回收与支付事件重试）
        # Memory extraction + lifecycle reconciliation moved to the
        # standalone memory-worker container. See app/workers/memory_worker.py.

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        tasks = [
            getattr(app.state, "membership_maintenance_task", None),
            getattr(app.state, "payment_retry_task", None),
        ]
        for task in tasks:
            if task is None:
                continue
            task.cancel()
        for task in tasks:
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
        await redis_client.close()

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "env": settings.app_env,
            "engine": "mobile_keyframe",
            "role": "client",
        }

    return app


app = create_client_app()
