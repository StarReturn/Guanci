"""Default app entrypoint.

Behavior depends on ``settings.app_role``:

* ``all``    – legacy mode, mounts every router under /api/v1, no auth
              middleware (backward compat for POC deployments).
* ``admin``  – delegates to :func:`app.admin_main.create_admin_app`.
* ``client`` – delegates to :func:`app.client_main.create_client_app`.

Production deployments should set ``APP_ROLE=admin`` and
``APP_ROLE=client`` in two separate containers (see docker-compose.yml),
so each surface gets role-specific auth + rate limiting.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.metrics import setup_metrics
from app.services.redis_client import redis_client


def create_app() -> FastAPI:
    configure_logging()

    role = (settings.app_role or "all").lower()
    if role == "admin":
        from app.admin_main import create_admin_app
        return create_admin_app()
    if role == "client":
        from app.client_main import create_client_app
        return create_client_app()

    if settings.app_env.strip().lower() in {"production", "prod"}:
        raise RuntimeError("APP_ROLE=all is forbidden in production")

    # Legacy: all routes, no auth middleware. Use only in local dev.
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    setup_metrics(app)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await redis_client.close()

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "env": settings.app_env,
            "engine": "mobile_keyframe",
            "role": "all (legacy, no auth)",
        }

    return app


app = create_app()
