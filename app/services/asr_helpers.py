"""Helpers for non-realtime ASR consumption of the streaming session API.

The realtime WebSocket drives ``StreamingASRSession`` directly (one PCM
frame at a time). Non-realtime HTTP endpoints (``POST /api/v1/dialogue``)
have a complete audio buffer and just want the final text, so they use
``transcribe_complete_audio`` which drives a session end-to-end.

See docs/阶段0-Gate0A阿里云实时ASR与音素对齐完整修复实施方案.md §7.1.
"""
from __future__ import annotations

from app.services.asr_session import ASREngine


def split_pcm(pcm: bytes, frame_bytes: int) -> list[bytes]:
    """Split a PCM buffer into ``frame_bytes``-sized chunks (last may be short).

    ``frame_bytes`` should be a whole-sample multiple (e.g. 640 = 320 samples
    of 16-bit mono). Odd-byte remainders are returned as-is; callers feeding
    a realtime session must ensure even byte counts.
    """
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]


async def transcribe_complete_audio(
    engine: ASREngine,
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    encoding: str = "pcm_s16le",
    frame_ms: int = 20,
) -> str:
    """Drive a streaming session end-to-end on a complete PCM buffer.

    Used only by non-realtime paths. The realtime WebSocket must NOT use
    this helper — it would defeat the purpose of true streaming (it buffers
    the whole turn before sending any frame).
    """
    session = await engine.create_session(
        sample_rate=sample_rate, channels=channels, encoding=encoding
    )
    await session.start()

    # 20ms frame = sample_rate * channels * 2 * 0.02 bytes
    frame_bytes = int(sample_rate * channels * 2 * frame_ms / 1000)
    final_text = ""

    # Consume events concurrently while we feed frames. The session emits
    # partials and exactly one final; we drain all of them but only the
    # final's text matters to the caller.
    async def _drain() -> None:
        nonlocal final_text
        async for result in session.events():
            if result.is_final:
                final_text = result.text

    import asyncio

    drainer = asyncio.create_task(_drain())
    try:
        for frame in split_pcm(pcm, frame_bytes):
            await session.send_pcm(frame)
        final = await session.finish()
        if final.is_final and final.text:
            final_text = final.text
        # let the drainer observe the final too, then exit
        await asyncio.wait_for(drainer, timeout=2.0)
    except Exception:
        await session.cancel()
        try:
            drainer.cancel()
        except Exception:
            pass
        raise
    finally:
        if not drainer.done():
            drainer.cancel()
    return final_text.strip()


__all__ = ["split_pcm", "transcribe_complete_audio"]
