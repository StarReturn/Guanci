"""CosyVoice 3.0 self-hosted client — zero-shot streaming TTS.

Server: https://github.com/FunAudioLLM/CosyVoice
暴露 FastAPI 端点（runtime/python/fastapi/server.py）：
    GET  /health
    POST /v1/tts/stream  (SSE 响应，每行一个 PCM 块)
    POST /v1/tts         (批量合成 wav，build 阶段用)

零样本（zero-shot）音色克隆：
  每次调用携带 reference_audio_bytes（3-30s 用户参考音频）+ reference_text。
  服务端不需要预注册音色，重新编码参考音频约 50-100ms（vLLM 加速后）。

为什么不用 LoRA fine-tune：
  调研结论（docs/实时对话方案决策文档.md §2.3）：LoRA 需 2 小时训练音频 +
  2-6 小时训练，用户录不了那么长。zero-shot 3-30s 即可，质量 ⭐⭐⭐⭐。
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.config import settings
from app.core.http_client import build_resilient_client, retry_exp
from app.services.audio_utils import normalize_audio_to_wav, wav_duration_ms

logger = logging.getLogger(__name__)


class CosyVoiceError(RuntimeError):
    pass


@dataclass
class CosyVoiceAudio:
    """批量合成结果（用于 build 阶段生成参考音频）。"""

    wav_bytes: bytes
    content_type: str
    duration_ms: int
    used_reference: bool


@dataclass(frozen=True)
class CosyVoiceStreamChunk:
    """One gapless audio chunk returned by CosyVoice's SSE endpoint."""

    pcm_bytes: bytes
    sample_rate: int
    channels: int
    format: str
    duration_ms: int


class CosyVoiceClient:
    """CosyVoice 3.0 HTTP/SSE 客户端。

    两种模式：
    - ``synthesize_stream``：流式合成，yield PCM 块（runtime 实时对话用）
    - ``synthesize_batch`` ：批量合成，返回完整 wav（build 阶段用）
    """

    def __init__(self) -> None:
        self._endpoint = settings.cosyvoice_endpoint.rstrip("/")
        self._read_timeout = float(settings.cosyvoice_timeout_seconds)

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    async def health_check(self) -> dict[str, Any]:
        try:
            async with build_resilient_client(read_timeout=5.0) as client:
                resp = await client.get(self._url(settings.cosyvoice_health_path))
                resp.raise_for_status()
            return {"status": "ok", "provider": "cosyvoice", "endpoint": self._endpoint}
        except Exception as exc:
            return {
                "status": "error",
                "provider": "cosyvoice",
                "endpoint": self._endpoint,
                "error": str(exc),
            }

    async def synthesize_stream(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
    ) -> AsyncIterator[CosyVoiceStreamChunk]:
        """流式合成，返回带格式元数据的 24kHz mono PCM16LE 块。

        服务端用 SSE 格式响应：每个 ``data: {...}`` 行包含一段 base64 PCM。
        本方法解码后按收到顺序 yield。

        服务端使用原生 CosyVoice stream，不在单个块中重复写 WAV header。
        """
        payload = self._build_payload(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
            stream=True,
        )

        async with build_resilient_client(read_timeout=self._read_timeout) as client:
            async with client.stream(
                "POST",
                self._url(settings.cosyvoice_tts_stream_path),
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise CosyVoiceError(
                        f"CosyVoice stream failed: HTTP {resp.status_code} "
                        f"{body[:500]!r}"
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("cosyvoice_sse_skip_invalid_line line=%r", data[:80])
                        continue
                    audio_b64 = chunk.get("audio") or chunk.get("data") or ""
                    if not audio_b64:
                        # 控制消息（progress / error 等），跳过
                        if chunk.get("error"):
                            raise CosyVoiceError(f"server error: {chunk['error']}")
                        continue
                    try:
                        pcm_bytes = base64.b64decode(audio_b64, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise CosyVoiceError("server returned invalid base64 audio") from exc
                    sample_rate = int(chunk.get("sample_rate") or 24000)
                    channels = int(chunk.get("channels") or 1)
                    format_name = str(chunk.get("format") or "pcm_s16le")
                    if format_name != "pcm_s16le" or channels != 1 or sample_rate <= 0:
                        raise CosyVoiceError(
                            "unsupported stream format: "
                            f"format={format_name}, sample_rate={sample_rate}, channels={channels}"
                        )
                    if len(pcm_bytes) % 2:
                        raise CosyVoiceError("PCM16 stream chunk has an odd byte length")
                    duration_ms = int(
                        chunk.get("duration_ms")
                        or round((len(pcm_bytes) // 2) * 1000 / sample_rate)
                    )
                    yield CosyVoiceStreamChunk(
                        pcm_bytes=pcm_bytes,
                        sample_rate=sample_rate,
                        channels=channels,
                        format=format_name,
                        duration_ms=duration_ms,
                    )

    async def synthesize_batch(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        format_: str = "wav",
        emotion: str | None = None,
        user_id: str | None = None,
    ) -> CosyVoiceAudio:
        """批量合成。返回完整 wav bytes（build 阶段生成 13×6 段参考音频用）。"""

        _ = emotion, user_id
        payload = self._build_payload(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
            stream=False,
            format_=format_,
        )

        async def _do_post() -> Any:
            async with build_resilient_client(read_timeout=self._read_timeout) as client:
                resp = await client.post(
                    self._url(settings.cosyvoice_tts_batch_path),
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code != 200:
                    raise CosyVoiceError(
                        f"CosyVoice batch failed: HTTP {resp.status_code} {resp.text[:500]}"
                    )
                return resp

        resp = await retry_exp(_do_post, attempts=3)

        content_type = resp.headers.get("content-type", "audio/wav")
        if "application/json" in content_type:
            raise CosyVoiceError(
                f"CosyVoice returned JSON instead of audio: {resp.text[:500]}"
            )

        wav_bytes = normalize_audio_to_wav(resp.content, content_type, default_sample_rate=16000)
        used_ref = reference_audio_bytes is not None
        return CosyVoiceAudio(
            wav_bytes=wav_bytes,
            content_type="audio/wav",
            duration_ms=wav_duration_ms(wav_bytes),
            used_reference=used_ref,
        )

    @staticmethod
    def _build_payload(
        *,
        text: str,
        reference_audio_bytes: bytes | None,
        reference_text: str | None,
        stream: bool,
        format_: str = "wav",
    ) -> dict[str, Any]:
        """构造 CosyVoice API 请求体。

        CosyVoice 3 zero-shot 接口字段（基于 runtime/python/fastapi/server.py）：
          - text: 要合成的文本
          - reference_audio: base64 wav（zero-shot 参考）
          - reference_text: 参考音频对应文本（提升对齐度）
          - stream: 是否流式

        注意：CosyVoice 3 server.py 内部会把 reference_audio 写入临时文件再传给
        ``inference_zero_shot(text, ref_text, ref_audio_path, stream=...)``。
        采样率 24000 Hz（CosyVoice 3 默认，非 22050）。
        """
        payload: dict[str, Any] = {
            "text": text,
            "stream": stream,
            "format": format_,
            "sample_rate": 24000,   # CosyVoice 3 默认
        }
        if reference_audio_bytes:
            wav = normalize_audio_to_wav(reference_audio_bytes, "audio/wav")
            payload["reference_audio"] = base64.b64encode(wav).decode("ascii")
            if reference_text:
                payload["reference_text"] = reference_text
        return payload


cosyvoice_client = CosyVoiceClient()
