"""IndexTTS2 self-hosted client (基于 neosun/indextts2 镜像).

镜像：docker pull neosun/indextts2:latest（all-in-one，含权重，21GB）
端口：8002（Flask API），7870（Gradio WebUI）
API：
  GET  /health             → {"status":"healthy",...}
  POST /upload_speaker     multipart: audio + speaker_name → {"speaker_id":"spk_xxx"}
  POST /tts_cached         JSON: {text, speaker_id}        → wav bytes
  GET  /speakers           → {"count":N,"speakers":[...]}

工作流：
  1. 第一次请求：上传 reference_audio 到 /upload_speaker → 得 speaker_id（缓存到内存）
  2. 后续请求：用 speaker_id 调 /tts_cached → 拿 wav bytes

注意：
  - IndexTTS2 没有 SSE 流式接口（批量模式）
  - 4.8s 合成约 6s 音频（"你好，我是数字人助理，今天天气不错。"）
  - realtime_ws.py 用 synthesize_batch + chunk 模拟流式下发

License：Apache 2.0 + 商用限制（个人 POC 免费，商用联系 bilibili）
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.http_client import build_resilient_client, retry_exp

logger = logging.getLogger(__name__)


class IndexTTSError(RuntimeError):
    pass


@dataclass
class IndexTTSAudio:
    wav_bytes: bytes
    content_type: str
    duration_ms: int
    used_reference: bool


class IndexTTSClient:
    """IndexTTS2 HTTP 客户端（批量模式）。"""

    def __init__(self) -> None:
        self._endpoint = settings.indextts_endpoint.rstrip("/")
        self._read_timeout = float(getattr(settings, "indextts_timeout_seconds", 60))
        # speaker_id 缓存：{(user_id, audio_md5): speaker_id}
        # 避免同一用户重复上传同一段参考音频
        self._speaker_cache: dict[tuple[str, str], str] = {}

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    async def health_check(self) -> dict[str, Any]:
        try:
            async with build_resilient_client(read_timeout=5.0) as client:
                resp = await client.get(self._url("/health"))
                resp.raise_for_status()
            return {"status": "ok", "provider": "indextts", "endpoint": self._endpoint}
        except Exception as exc:
            return {
                "status": "error",
                "provider": "indextts",
                "endpoint": self._endpoint,
                "error": str(exc),
            }

    async def _ensure_speaker(
        self,
        *,
        user_id: str,
        reference_audio_bytes: bytes,
        speaker_name: str | None = None,
    ) -> str:
        """上传参考音频（如未缓存），返回 speaker_id。"""
        audio_md5 = hashlib.md5(reference_audio_bytes).hexdigest()
        cache_key = (user_id, audio_md5)
        if cache_key in self._speaker_cache:
            return self._speaker_cache[cache_key]

        # 上传到 /upload_speaker（multipart）
        files = {"audio": ("ref.wav", reference_audio_bytes, "audio/wav")}
        data = {"speaker_name": speaker_name or f"user_{user_id[:8]}"}

        async def _do_upload() -> str:
            async with build_resilient_client(read_timeout=self._read_timeout) as client:
                resp = await client.post(
                    self._url("/upload_speaker"),
                    files=files,
                    data=data,
                )
                if resp.status_code != 200:
                    raise IndexTTSError(
                        f"upload_speaker failed: HTTP {resp.status_code} {resp.text[:300]}"
                    )
                result = resp.json()
                speaker_id = result.get("speaker_id")
                if not speaker_id:
                    raise IndexTTSError(f"no speaker_id in response: {result}")
                return speaker_id

        speaker_id = await retry_exp(_do_upload, attempts=2)
        self._speaker_cache[cache_key] = speaker_id
        logger.info(
            "indextts_speaker_uploaded user_id=%s speaker_id=%s md5=%s",
            user_id, speaker_id, audio_md5[:8],
        )
        return speaker_id

    async def synthesize_batch(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        user_id: str = "default",
        emotion: str | None = None,
        format_: str = "wav",
    ) -> IndexTTSAudio:
        """批量合成。返回完整 wav bytes。

        参数：
          text                    : 要合成的文本
          reference_audio_bytes   : 用户参考音频（首次会上传到 /upload_speaker）
          reference_text          : 仅作 logging（IndexTTS2 用参考音频而非文本）
          user_id                 : 用户 ID（用于 speaker_id 缓存键）
          emotion                 : 情绪标签（neutral/happy/sad/surprised/angry/fear）
                                    来自 LLM [EMOTION:xxx] 标签，映射到 emo_vector

        返回：IndexTTSAudio（wav bytes）

        异常：IndexTTSError
        """
        if not reference_audio_bytes:
            raise IndexTTSError("IndexTTS2 requires reference_audio_bytes (zero-shot)")

        # 1. 确保 speaker 已上传
        speaker_id = await self._ensure_speaker(
            user_id=user_id,
            reference_audio_bytes=reference_audio_bytes,
        )

        # 2. emotion → emo_vector + emo_alpha（让 TTS 有情绪，不是冰冷）
        emo_vector = None
        emo_alpha = 1.0
        if emotion:
            try:
                from app.services.emotion_classifier import build_indextts_params
                emo_params = build_indextts_params(emotion)
                emo_vector = emo_params["emo_vector"]
                emo_alpha = emo_params["emo_alpha"]
                logger.info(
                    "indextts_emotion user=%s emotion=%s alpha=%.2f",
                    user_id, emotion, emo_alpha,
                )
            except Exception:
                logger.debug("emo_vector_mapping_failed", exc_info=True)

        # 3. 调 /tts_cached 合成（带情绪）
        payload: dict = {
            "text": text,
            "speaker_id": speaker_id,
            "emo_alpha": emo_alpha,
        }
        if emo_vector is not None:
            payload["emo_vector"] = emo_vector

        async def _do_synth() -> bytes:
            async with build_resilient_client(read_timeout=self._read_timeout) as client:
                resp = await client.post(
                    self._url("/tts_cached"),
                    json=payload,
                    headers={"Accept": "audio/wav"},
                )
                if resp.status_code != 200:
                    raise IndexTTSError(
                        f"tts_cached failed: HTTP {resp.status_code} {resp.text[:300]}"
                    )
                content_type = resp.headers.get("content-type", "")
                if "audio" not in content_type and "octet-stream" not in content_type:
                    raise IndexTTSError(
                        f"unexpected content-type: {content_type}; body[:200]={resp.text[:200]}"
                    )
                return resp.content

        wav_bytes = await retry_exp(_do_synth, attempts=2)

        # 估算时长（16kHz mono 16bit → 32 bytes/ms）
        duration_ms = (len(wav_bytes) - 44) // 32  # 减去 wav header

        return IndexTTSAudio(
            wav_bytes=wav_bytes,
            content_type="audio/wav",
            duration_ms=max(0, duration_ms),
            used_reference=True,
        )

    async def synthesize_stream(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        user_id: str = "default",
        chunk_size: int = 4096,
    ):
        """模拟流式：先 batch 合成，再按 chunk_size 切块 yield。

        IndexTTS2 不支持原生流式。这里用 batch + chunk 模拟，
        首块延迟 ≈ batch 总耗时（4-8s 视文本长度）。
        """
        result = await self.synthesize_batch(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
            user_id=user_id,
        )
        wav = result.wav_bytes
        for offset in range(0, len(wav), chunk_size):
            yield wav[offset:offset + chunk_size]


indextts_client = IndexTTSClient()
