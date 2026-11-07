"""Streaming ASR session protocol (Gate 0 Stage B).

Defines the contract for true streaming ASR: PCM frames are forwarded to
the recognizer as they arrive (not buffered until end-of-turn). Two roles:

  - ``ASREngine``: factory that creates a fresh session per user turn.
  - ``StreamingASRSession``: one user turn's lifecycle
        start → (send_pcm × N) → finish → final result
        or cancel at any point.

Both ``aliyun_asr`` and ``funasr`` implement this contract so the realtime
WebSocket layer can treat them uniformly. The legacy whisper batch path
(``asr_service.transcribe``) does NOT implement it and stays a separate
helper for non-realtime HTTP endpoints.

See docs/阶段0-Gate0A阿里云实时ASR与音素对齐完整修复实施方案.md §6.1.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.services.funasr_client import ASRResult


@runtime_checkable
class StreamingASRSession(Protocol):
    """One user turn's ASR session.

    Lifecycle invariants:
      - ``start`` is called exactly once before any ``send_pcm``.
      - ``send_pcm`` may be called zero or more times with 16kHz mono s16le
        PCM frames after ``start`` and before ``finish``/``cancel``.
      - ``events`` yields ``ASRResult`` instances (partial and final) as the
        recognizer produces them. It must be consumed concurrently with
        ``send_pcm`` (typically in a separate task).
      - ``finish`` signals end-of-audio and returns the single final
        ``ASRResult`` (is_final=True). After ``finish``, no more events.
      - ``cancel`` aborts the session; no final is expected.
      - Exactly one of ``finish`` / ``cancel`` terminates the session.
    """

    async def start(self) -> None:
        ...

    async def send_pcm(self, pcm: bytes) -> None:
        ...

    def events(self) -> AsyncIterator[ASRResult]:
        ...

    async def finish(self) -> ASRResult:
        ...

    async def cancel(self) -> None:
        ...


@runtime_checkable
class ASREngine(Protocol):
    """Factory that produces one ``StreamingASRSession`` per user turn.

    Implementations: ``AliyunASRClient`` (production), ``funasr_client``
    (legacy adapter). Whisper batch path is intentionally NOT an ASREngine.
    """

    async def create_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        encoding: str,
    ) -> StreamingASRSession:
        ...


__all__ = ["ASREngine", "StreamingASRSession"]
