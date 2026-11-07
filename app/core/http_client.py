"""Resilient HTTP client factory.

All outbound HTTP (EchoMimic, TTS providers, MinIO presigned fetches)
should go through :func:`build_resilient_client` so timeouts and retries
are consistent. Business-level retries use :func:`retry_exp`.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Type, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Network-level hiccups we always want to retry. Do NOT retry HTTPStatusError
# here; the caller knows best whether a 4xx is retryable.
_RETRYABLE_EXC: tuple[Type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def build_resilient_client(
    *,
    read_timeout: float,
    connect_timeout: float = 5.0,
    write_timeout: float = 30.0,
    pool_timeout: float = 5.0,
    max_connections: int = 20,
    max_keepalive_connections: int = 10,
) -> httpx.AsyncClient:
    """Construct an AsyncClient with sane defaults.

    ``read_timeout`` is the one knob that should vary per service:
    EchoMimic generation can take several minutes; TTS calls are shorter;
    MinIO object fetches ~10s.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        ),
        transport=httpx.AsyncHTTPTransport(retries=2),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        ),
    )


async def retry_exp(
    func: Callable[[], "Any"],
    *,
    attempts: int = 3,
    min_wait_s: float = 1.0,
    max_wait_s: float = 10.0,
    retry_exceptions: tuple[Type[BaseException], ...] = _RETRYABLE_EXC,
) -> Any:
    """Run ``func`` with exponential backoff.

    ``func`` may be sync or async. After all attempts are exhausted the
    last retryable exception is re-raised (``reraise=True``).
    """
    runner = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait_s, max=max_wait_s),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
    )

    async def _wrapped() -> Any:
        result = func()
        if inspect.isawaitable(result):
            result = await result
        return result

    return await runner(_wrapped)


__all__ = ["build_resilient_client", "retry_exp"]
