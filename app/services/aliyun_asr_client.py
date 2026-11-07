"""Aliyun DashScope Paraformer Real-time streaming ASR client.

Online ASR via the DashScope SDK (``dashscope.audio.asr.Recognition``).
Reuses the same ``DASHSCOPE_API_KEY`` already configured for CosyVoice TTS.

Stage B refactor: true streaming session. Previously the client drained the
entire audio stream before sending any frame to DashScope, which defeats the
purpose of real-time ASR. The new ``AliyunASRSession`` forwards each PCM
chunk to ``Recognition.send_audio_frame`` as it arrives, emits partial
results via ``events()``, and produces exactly one final on ``finish()``.

Contract: implements ``StreamingASRSession`` / ``ASREngine`` (see
``app/services/asr_session.py``) so the realtime WebSocket and the
non-realtime helper (``transcribe_complete_audio``) can use it uniformly.

References:
  - Aliyun docs: Paraformer real-time Python SDK
  - Local async-bridging template: app/services/aliyun_cosyvoice_client.py
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import settings
from app.services.funasr_client import ASRResult

logger = logging.getLogger(__name__)


class AliyunASRError(RuntimeError):
    """Raised when the Aliyun ASR provider is misconfigured or fails."""


# ---------------------------------------------------------------------#
# Lazy SDK import (mirror cosyvoice client so other providers don't pay)
# ---------------------------------------------------------------------#
def _load_sdk() -> tuple[Any, Any, Any, Any]:
    """Return (dashscope, Recognition, RecognitionCallback, RecognitionResult).

    Imported lazily so that a deployment that never uses aliyun_asr does
    not need dashscope installed at import time.
    """
    try:
        import dashscope
        from dashscope.audio.asr import (
            Recognition,
            RecognitionCallback,
            RecognitionResult,
        )
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise AliyunASRError(
            "dashscope is not installed; install backend dependencies to use aliyun_asr"
        ) from exc
    return dashscope, Recognition, RecognitionCallback, RecognitionResult


# ---------------------------------------------------------------------#
# Streaming session
# ---------------------------------------------------------------------#
class AliyunASRSession:
    """One user turn's real-time ASR session.

    The DashScope ``Recognition`` SDK is synchronous and callback-driven.
    We bridge it to async with an ``asyncio.Queue``:

      - ``start`` creates the Recognition object and calls ``recognition.start()``
        in a worker thread.
      - ``send_pcm`` forwards PCM frames to ``recognition.send_audio_frame()``
        in the worker thread (thread-safe because the SDK is synchronous
        and we serialize via the loop's default executor).
      - callbacks (on_event/on_error/on_complete) push results onto the queue
        from the worker thread via ``call_soon_threadsafe``.
      - ``events`` is an async generator that drains the queue.
      - ``finish`` calls ``recognition.stop()`` and awaits the single final.
      - ``cancel`` aborts the session.

    The session is NOT thread-safe beyond the SDK's own internal threading;
    callers drive it from a single asyncio loop.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        sample_rate: int,
        channels: int,
        encoding: str,
        timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._sample_rate = sample_rate
        self._channels = channels
        self._encoding = encoding
        self._timeout = timeout_seconds

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._final_future: asyncio.Future | None = None
        self._recognition: Any = None
        self._recognition_callback: Any = None
        self._started = False
        self._finished = False
        self._cancelled = False
        self._final_seen = False
        self._stream_end_sentinel = object()

    async def start(self) -> None:
        if self._started:
            raise AliyunASRError("ASR session already started")
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._final_future = self._loop.create_future()

        dashscope, Recognition, RecognitionCallback, RecognitionResult = _load_sdk()
        dashscope.api_key = self._api_key

        session_self = self
        loop = self._loop
        queue = self._queue

        class _Bridge(RecognitionCallback):  # type: ignore[misc]
            def on_open(self) -> None:
                logger.debug("aliyun_asr_session_opened")

            def on_event(self, result: Any) -> None:
                try:
                    sentence = result.get_sentence()
                except Exception:  # pragma: no cover - defensive
                    logger.debug("aliyun_asr_event_no_sentence", exc_info=True)
                    return
                if not isinstance(sentence, dict):
                    return
                text = sentence.get("text", "")
                is_end = bool(RecognitionResult.is_sentence_end(sentence))
                mode = "offline" if is_end else "online"
                asr_result = ASRResult(text=text, is_final=is_end, mode=mode)
                if is_end:
                    # Final goes to the future so finish() can await it directly,
                    # without racing with the partial-draining events() consumer.
                    fut = session_self._final_future
                    if fut is not None and not fut.done():
                        loop.call_soon_threadsafe(fut.set_result, asr_result)
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, asr_result)

            def on_complete(self) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, session_self._stream_end_sentinel)

            def on_error(self, result: Any) -> None:
                msg = getattr(result, "message", None) or str(result)
                err = AliyunASRError(f"dashscope asr error: {msg}")
                loop.call_soon_threadsafe(queue.put_nowait, err)

            def on_close(self) -> None:
                logger.debug("aliyun_asr_session_closed")

        self._recognition_callback = _Bridge()
        self._recognition = Recognition(
            model=self._model,
            format="pcm",
            sample_rate=self._sample_rate,
            callback=self._recognition_callback,
        )

        # start() is synchronous and opens the WS; run it in a thread so it
        # does not block the event loop.
        await asyncio.to_thread(self._recognition.start)
        self._started = True

    async def send_pcm(self, pcm: bytes) -> None:
        if not self._started:
            raise AliyunASRError("ASR session not started")
        if self._finished or self._cancelled:
            raise AliyunASRError("ASR session already terminated")
        # send_audio_frame is a synchronous, non-blocking enqueue on the SDK's
        # internal WS writer. Running it via the default executor keeps the
        # event loop responsive while guaranteeing serialization.
        await asyncio.to_thread(self._recognition.send_audio_frame, pcm)

    async def events(self) -> AsyncIterator[ASRResult]:
        if self._queue is None:
            raise AliyunASRError("ASR session not started")
        while True:
            item = await asyncio.wait_for(self._queue.get(), timeout=self._timeout)
            if item is self._stream_end_sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def finish(self) -> ASRResult:
        """Signal end-of-audio and return the single final ASRResult.

        The final result is delivered via _final_future (resolved by the
        on_event callback when it sees is_sentence_end). This avoids racing
        with the partial-draining events() consumer on the same queue.
        """
        if self._finished:
            raise AliyunASRError("ASR session already finished")
        if self._cancelled:
            raise AliyunASRError("ASR session already cancelled")
        self._finished = True

        # stop() triggers the server to flush its final result.
        try:
            await asyncio.to_thread(self._recognition.stop)
        except Exception:  # pragma: no cover - stop rarely fails
            logger.debug("aliyun_asr_stop_failed", exc_info=True)

        # Wait for the final result via the future, not by re-entering events().
        fut = self._final_future
        if fut is None:
            return ASRResult(text="", is_final=True, mode="offline")
        try:
            final = await asyncio.wait_for(fut, timeout=self._timeout)
            return final
        except asyncio.TimeoutError:
            logger.warning("aliyun_asr_final_timeout — no final received")
            return ASRResult(text="", is_final=True, mode="offline")
        except AliyunASRError:
            return ASRResult(text="", is_final=True, mode="offline")

        if final is None:
            final = ASRResult(text="", is_final=True, mode="offline")
        return final

    async def cancel(self) -> None:
        """Abort the session. No final is expected afterwards."""
        if self._cancelled:
            return
        self._cancelled = True
        if self._recognition is not None:
            try:
                await asyncio.to_thread(self._recognition.stop)
            except Exception:  # pragma: no cover
                logger.debug("aliyun_asr_cancel_stop_failed", exc_info=True)
        # Unblock any pending events() consumer.
        if self._queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, self._stream_end_sentinel
            )


# ---------------------------------------------------------------------#
# Engine (factory) — module-level singleton
# ---------------------------------------------------------------------#
class AliyunASRClient:
    """DashScope Paraformer real-time ASR engine.

    Provider identifier: ``"aliyun_asr"``. Implements ``ASREngine`` so
    ``engine_factory.get_asr_engine`` can return it interchangeably with
    the funasr adapter.
    """

    provider = "aliyun_asr"

    def _api_key(self) -> str:
        key = (settings.dashscope_api_key or "").strip()
        if not key:
            raise AliyunASRError("DASHSCOPE_API_KEY is not configured")
        return key

    @property
    def model_name(self) -> str:
        return settings.aliyun_asr_model

    @property
    def sample_rate(self) -> int:
        return settings.aliyun_asr_sample_rate

    async def create_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        encoding: str,
    ) -> AliyunASRSession:
        """Create a new streaming session for one user turn."""
        return AliyunASRSession(
            api_key=self._api_key(),
            model=self.model_name,
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
            timeout_seconds=float(settings.aliyun_asr_timeout_seconds),
        )

    async def health_check(self) -> dict[str, Any]:
        """Verify the DashScope key is set and the SDK imports cleanly.

        We do NOT open a real recognition stream here (that would bill
        the account on every health probe). Importing the SDK and
        asserting the key is present is a sufficient liveness check.
        """
        try:
            _load_sdk()  # raises if SDK missing
            self._api_key()  # raises if key missing
            return {
                "status": "ok",
                "provider": self.provider,
                "model": self.model_name,
                "endpoint": "dashscope (default)",
            }
        except Exception as exc:
            return {
                "status": "error",
                "provider": self.provider,
                "model": self.model_name,
                "error": str(exc),
            }


aliyun_asr_client = AliyunASRClient()
