"""Admin app entrypoint.

Run with:
    uvicorn app.admin_main:app --host 0.0.0.0 --port 8090

Exposes:
    * /api/v1/keyframes/* (build, task lookup, package metadata, download)
    * /api/v1/health/dependencies
    * /metrics (Prometheus)
    * /health (liveness, no auth)

Auth: requires X-API-Key matching ADMIN_API_KEY (comma-separated list
for zero-downtime rotation).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import admin_router
from app.core.config import settings, validate_production_security
from app.core.logging import configure_logging
from app.core.metrics import setup_metrics
from app.middleware.admin_auth import make_admin_auth_middleware
from app.services.redis_client import redis_client


def _validate_admin_cors_for_production() -> None:
    """Reject CORS_ORIGINS=['*'] in production for admin surface."""
    env = (settings.app_env or "local").lower()
    if env in ("production", "prod"):
        origins = [o.strip() for o in settings.cors_origins if o and o.strip()]
        if not origins or "*" in origins:
            raise RuntimeError(
                "CORS_ORIGINS must list concrete origins when APP_ENV=production "
                "(admin surface); update .env."
            )


def create_admin_app() -> FastAPI:
    configure_logging()
    validate_production_security(role="admin")
    _validate_admin_cors_for_production()

    app = FastAPI(
        title=f"{settings.app_name} (Admin)",
        version="1.0.0",
        description="管理端：素材包构建 + 任务管理 + 依赖健康检查。需 X-API-Key 鉴权。",
    )

    # CORS — admin should be locked to known origins in production.
    # Use credentials=False so origins=["*"] works; if you really need
    # cookies (e.g. for a same-origin admin UI), set CORS_ORIGINS env
    # to concrete origins and flip allow_credentials=True.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        max_age=3600,
    )

    # ---- Auth: JWT(role=admin) + API Key fallback ----
    # Admin surface accepts two credential types:
    # 1. JWT from admin login (POST /api/v1/admin/auth/login)
    # 2. X-API-Key header (backward compat / bootstrap)
    app.middleware("http")(make_admin_auth_middleware())

    # Routes
    app.include_router(admin_router, prefix=settings.api_prefix)

    setup_metrics(app)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await redis_client.close()

    @app.on_event("startup")
    async def _bootstrap():
        """Optionally create the explicitly configured initial super administrator."""
        from app.api.v1.routes.admin_auth import bootstrap_super_admin
        bootstrap_super_admin()

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "env": settings.app_env,
            "engine": "mobile_keyframe",
            "role": "admin",
        }

    return app


app = create_admin_app()
