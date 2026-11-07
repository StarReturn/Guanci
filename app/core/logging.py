"""Structured logging via structlog.

Two formats, switched by ``settings.log_format``:

* ``json`` (default in containers) – one JSON object per log line. Best
  for ELK / Loki / CloudWatch.
* ``plain`` – human-readable console output. Best for local dev.

Log level is bound by ``settings.log_level``.

Usage::

    from app.core.logging import configure_logging
    configure_logging()

    import logging
    log = logging.getLogger(__name__)
    log.info("event_name", extra={"task_id": "..."})

We use stdlib ``logging`` everywhere so existing call sites don't need to
change; structlog only formats what stdlib emits.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import settings


_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent. Safe to call from main + worker entrypoints."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Stdlib root logger: everything goes through our handler.
    root = logging.getLogger()
    root.setLevel(level)
    # Remove handlers that uvicorn / pytest might have attached.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_select_formatter())
    root.addHandler(handler)

    # Quiet down chatty libs.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def _select_formatter() -> logging.Formatter:
    fmt = (settings.log_format or "json").lower()
    if fmt == "plain":
        return logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # JSON via structlog. Fall back to plain if structlog import fails
    # (e.g. in a stripped-down test env).
    try:
        from structlog.processors import JSONRenderer
        from structlog.stdlib import ProcessorFormatter, ProcessorFormatterRenderer
        from structlog.processors import TimeStamper, add_log_level
        from structlog import wrap_logger
    except ImportError:
        return logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    class _JSONRenderer(ProcessorFormatterRenderer):
        def __call__(self, logger: Any, name: Any, event_dict: Any) -> Any:
            return JSONRenderer(sort_keys=True)(logger, name, event_dict)

    return ProcessorFormatter(
        foreign_pre_chain=[
            TimeStamper(fmt="iso", utc=True),
            add_log_level,
        ],
        processor=_JSONRenderer(),
    )


__all__ = ["configure_logging"]
