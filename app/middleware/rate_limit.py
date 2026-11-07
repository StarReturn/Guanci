"""Rate limiting for client-facing endpoints.

Uses ``slowapi`` (Starlette limiter). Each client endpoint declares its
own limit via the ``@limiter.limit`` decorator; the global default is
applied to anything not annotated.

Per-endpoint defaults (POC values, tune in production):

* ``POST /tts/synthesize``         — 2 req/s per IP   (TTS is GPU-heavy)
* ``POST /phonemes/align``         — 5 req/s per IP
* ``POST /realtime/sessions``      — 1 req/s per IP   (session is sticky)
* ``GET  /keyframes/packages/*``   — 10 req/s per IP  (cheap reads)

The limiter keys on ``X-Forwarded-For`` (set by nginx gateway). If the
header is missing, fall back to direct client IP.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_real_ip(request) -> str:
    """Resolve client IP, preferring X-Forwarded-For from trusted gateway.

    The gateway (nginx) always sets this header. Direct connections
    without the header fall back to ``request.client.host``.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        # First hop is the real client
        return xff.split(",")[0].strip()
    return get_remote_address(request)


# Default limiter: per-IP, in-memory counter (no Redis dependency).
# For multi-instance deployments, swap storage_uri to redis://...
limiter = Limiter(key_func=_get_real_ip, storage_uri="memory://")


# Per-endpoint limit constants — used as decorators on routes.
LIMIT_TTS = "2/second"
LIMIT_PHONEME = "5/second"
LIMIT_REALTIME = "1/second"
LIMIT_PACKAGE_READ = "10/second"


__all__ = [
    "limiter",
    "LIMIT_TTS",
    "LIMIT_PHONEME",
    "LIMIT_REALTIME",
    "LIMIT_PACKAGE_READ",
]
