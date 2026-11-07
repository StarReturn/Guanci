"""Prometheus metrics setup.

We rely on ``prometheus-fastapi-instrumentator`` for the standard
HTTP request duration / count histogram (per route, per status). On
top of that we expose the latest generated package size for capacity
planning.

The instrumentator mounts a ``/metrics`` endpoint automatically when
``settings.metrics_enabled`` is true.
"""
from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)


# Total package size in bytes for the last successful build per avatar
# (best-effort; not strictly required but useful for capacity planning).
KEYFRAME_PACKAGE_SIZE_BYTES = Gauge(
    "digitalhuman_keyframe_package_size_bytes",
    "Size in bytes of the most recently built keyframe package.",
    ["avatar_id", "quality_profile"],
)


def setup_metrics(app: Any) -> None:
    """Attach the Prometheus instrumentator to the FastAPI app.

    No-op when ``settings.metrics_enabled`` is False.
    """
    from app.core.config import settings

    if not settings.metrics_enabled:
        logger.info("metrics_disabled_by_config")
        return

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
    )
    logger.info("metrics_enabled_endpoint", extra={"endpoint": "/metrics"})


__all__ = [
    "KEYFRAME_PACKAGE_SIZE_BYTES",
    "setup_metrics",
]
