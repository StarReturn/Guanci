"""Engine provider factory.

统一工厂函数，根据 settings 切换 TTS / Visual / ASR 引擎。
设计目标：
  - 业务代码只调 ``get_tts_engine()`` / ``get_asr_engine()``（视觉引擎已收敛进 dhlive-builder）
  - 切换引擎不用动业务代码，只改 ``.env`` 里的 provider 字段
"""
from __future__ import annotations

from typing import Any

from app.core.config import settings


def get_tts_engine() -> Any:
    """返回当前 TTS 客户端实例。

    Provider 取自 ``settings.tts_provider``：
      - "fish_speech" → fish_speech_client（旧，fallback）
      - "cosyvoice"   → cosyvoice_client（Apache 2.0，原生流式）
      - "aliyun_cosyvoice" → Alibaba Cloud CosyVoice v3.5（原生流式）
      - "qwen_tts"    → qwen_tts_client（百炼 Qwen3-TTS，限流独立桶 180 RPM；batch 路径）
      - "indextts"    → indextts_client（B 站，商用限制，但镜像现成）

    返回的 client 接口：
      - ``synthesize_batch(text, reference_audio_bytes, reference_text) -> wav``
      - ``health_check() -> dict``
    """
    from app.services.tts_provider_service import tts_provider_service

    provider = tts_provider_service.runtime().type
    if provider == "aliyun_cosyvoice":
        from app.services.aliyun_cosyvoice_client import aliyun_cosyvoice_client
        return aliyun_cosyvoice_client
    if provider == "qwen_tts":
        from app.services.qwen_tts_client import qwen_tts_client
        return qwen_tts_client
    if provider == "cosyvoice":
        from app.services.cosyvoice_client import cosyvoice_client
        return cosyvoice_client
    if provider == "indextts":
        from app.services.indextts_client import indextts_client
        return indextts_client
    if provider == "fish_speech":
        from app.services.fish_speech_client import fish_speech_client
        return fish_speech_client
    raise RuntimeError(f"unknown tts_provider: {provider!r}")


def get_asr_engine() -> Any:
    """返回当前 ASR 客户端实例。

    Provider 取自 ``settings.asr_provider``：
      - "aliyun_asr" → aliyun_asr_client（在线 DashScope Paraformer Realtime，推荐）
      - "funasr"     → funasr_client（自托管 Docker，流式；受 0.1.13 服务端 bug 影响）
      - "whisper"    → asr_service（旧，本地批量）

    返回的 client 接口：
      - whisper:    ``transcribe(audio_bytes) -> str``（批量）
      - funasr/aliyun_asr: ``transcribe_stream(audio_stream) -> AsyncIterator[ASRResult]``
    """
    provider = settings.asr_provider.lower()
    if provider == "aliyun_asr":
        from app.services.aliyun_asr_client import aliyun_asr_client
        return aliyun_asr_client
    if provider == "funasr":
        from app.services.funasr_client import funasr_client
        return funasr_client
    if provider == "whisper":
        from app.services.asr_service import asr_service
        return asr_service
    raise RuntimeError(f"unknown asr_provider: {provider!r}")


__all__ = ["get_tts_engine", "get_asr_engine"]
