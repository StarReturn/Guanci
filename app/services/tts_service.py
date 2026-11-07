"""Provider-neutral TTS orchestration shared by HTTP and WebSocket routes."""
from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.core.config import settings
from app.services.auth_service import user_avatar_service
from app.services.engine_factory import get_tts_engine
from app.services.storage import storage_service
from app.services.tts_provider_service import tts_provider_service


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceReference:
    wav_bytes: bytes | None = None
    transcript: str | None = None
    audio_key: str | None = None
    user_avatar_id: str | None = None
    digital_human_id: str | None = None
    duration_ms: int | None = None
    provider: str | None = None
    model_id: str | None = None
    provider_voice_id: str | None = None

    @property
    def available(self) -> bool:
        return self.wav_bytes is not None or self.provider_voice_id is not None


def resolve_voice_reference(
    user_id: str | None, digital_human_id: str | None = None
) -> VoiceReference:
    if not user_id:
        return VoiceReference()
    if digital_human_id:
        from app.services.digital_human_service import digital_human_service

        digital_human = digital_human_service.get_owned(user_id, digital_human_id)
        if (
            digital_human.voice_storage_key
            and not digital_human_service.has_active_consent(
                digital_human_id, "voice_processing"
            )
        ):
            from app.services.digital_human_service import DigitalHumanConsentError

            raise DigitalHumanConsentError("voice_processing consent is not active")
        transcript = (digital_human.voice_ref_text or "").strip() or None
        metadata = dict(
            audio_key=digital_human.voice_storage_key,
            digital_human_id=digital_human_id,
            duration_ms=digital_human.voice_ref_duration_ms,
            provider=digital_human.voice_provider,
            model_id=digital_human.voice_model_id,
            provider_voice_id=digital_human.voice_provider_voice_id,
        )
        if (
            provider_name() == "aliyun_cosyvoice"
            and digital_human.voice_provider == "aliyun_cosyvoice"
            and digital_human.voice_model_id == _aliyun_model_id()
            and digital_human.voice_provider_voice_id
        ):
            return VoiceReference(transcript=transcript, **metadata)
        if not digital_human.voice_storage_key:
            return VoiceReference(transcript=transcript, **metadata)
        try:
            wav_bytes = storage_service.get_bytes(digital_human.voice_storage_key)
        except Exception:
            logger.exception("voice_ref_fetch_failed key=%s", digital_human.voice_storage_key)
            return VoiceReference(transcript=transcript, **metadata)
        return VoiceReference(wav_bytes=wav_bytes, transcript=transcript, **metadata)
    binding = user_avatar_service.get_primary_for_user(user_id)
    if binding is None:
        return VoiceReference()
    transcript = (binding.voice_ref_text or "").strip() or None
    metadata = dict(
        audio_key=binding.voice_ref_audio_key,
        user_avatar_id=binding.user_avatar_id,
        duration_ms=binding.voice_ref_duration_ms,
        provider=binding.voice_provider,
        model_id=binding.voice_model_id,
        provider_voice_id=binding.voice_provider_voice_id,
    )
    if (
        provider_name() == "aliyun_cosyvoice"
        and binding.voice_provider == "aliyun_cosyvoice"
        and binding.voice_model_id == _aliyun_model_id()
        and binding.voice_provider_voice_id
    ):
        return VoiceReference(transcript=transcript, **metadata)
    if not binding.voice_ref_audio_key:
        return VoiceReference(transcript=transcript, **metadata)
    if transcript is None:
        logger.warning(
            "voice_ref_transcript_missing key=%s",
            binding.voice_ref_audio_key,
        )
    try:
        wav_bytes = storage_service.get_bytes(binding.voice_ref_audio_key)
    except Exception:
        logger.exception("voice_ref_fetch_failed key=%s", binding.voice_ref_audio_key)
        return VoiceReference(transcript=transcript, **metadata)
    return VoiceReference(
        wav_bytes=wav_bytes,
        transcript=transcript,
        **metadata,
    )


_voice_enrollment_locks: dict[str, asyncio.Lock] = {}


async def ensure_provider_voice(
    *,
    user_id: str | None,
    digital_human_id: str | None = None,
    reference: VoiceReference | None = None,
) -> str | None:
    if provider_name() != "aliyun_cosyvoice" or not user_id:
        return None
    current = reference or resolve_voice_reference(user_id, digital_human_id)
    model_id = _aliyun_model_id()
    if (
        current.provider == "aliyun_cosyvoice"
        and current.model_id == model_id
        and current.provider_voice_id
    ):
        return current.provider_voice_id
    if not current.audio_key or not (current.user_avatar_id or current.digital_human_id):
        return None

    lock = _voice_enrollment_locks.setdefault(user_id, asyncio.Lock())
    async with lock:
        current = resolve_voice_reference(user_id, digital_human_id)
        if (
            current.provider == "aliyun_cosyvoice"
            and current.model_id == model_id
            and current.provider_voice_id
        ):
            return current.provider_voice_id

        engine = get_tts_engine()
        voice_id = await engine.create_voice(
            audio_url=storage_service.presigned_url(current.audio_key),
            user_id=user_id,
            duration_ms=current.duration_ms,
        )
        try:
            if current.digital_human_id:
                from app.services.digital_human_service import digital_human_service

                updated = digital_human_service.set_provider_voice(
                    owner_user_id=user_id,
                    digital_human_id=current.digital_human_id,
                    provider="aliyun_cosyvoice",
                    model_id=model_id,
                    provider_voice_id=voice_id,
                )
            else:
                updated = user_avatar_service.set_provider_voice(
                    user_avatar_id=current.user_avatar_id,
                    provider="aliyun_cosyvoice",
                    model_id=model_id,
                    provider_voice_id=voice_id,
                )
        except Exception:
            await asyncio.to_thread(engine.delete_voice_sync, voice_id)
            raise
        if not updated:
            await asyncio.to_thread(engine.delete_voice_sync, voice_id)
            raise RuntimeError("voice was created but avatar binding no longer exists")
        if (
            current.digital_human_id
            and current.provider == "aliyun_cosyvoice"
            and current.provider_voice_id
            and current.provider_voice_id != voice_id
            and not digital_human_service.is_preset_voice_id(current.provider_voice_id)
        ):
            try:
                await asyncio.to_thread(
                    engine.delete_voice_sync, current.provider_voice_id
                )
                digital_human_service.forget_retired_voice(
                    owner_user_id=user_id,
                    digital_human_id=current.digital_human_id,
                    provider_voice_id=current.provider_voice_id,
                )
            except Exception:
                logger.exception(
                    "retired_voice_cleanup_failed digital_human_id=%s",
                    current.digital_human_id,
                )
        return voice_id


async def synthesize_batch(
    *,
    text: str,
    user_id: str | None,
    digital_human_id: str | None = None,
    emotion: str | None = None,
    output_sample_rate: int | None = None,
) -> Any:
    reference = resolve_voice_reference(user_id, digital_human_id)
    engine = get_tts_engine()
    kwargs = dict(
        text=text,
        reference_audio_bytes=reference.wav_bytes,
        reference_text=reference.transcript,
        emotion=emotion,
        user_id=user_id,
    )
    if provider_name() == "aliyun_cosyvoice":
        kwargs["voice_id"] = await ensure_provider_voice(
            user_id=user_id, digital_human_id=digital_human_id, reference=reference
        )
        if output_sample_rate is not None:
            kwargs["output_sample_rate"] = output_sample_rate
    return await engine.synthesize_batch(**kwargs)


async def synthesize_stream(
    *,
    text: str,
    user_id: str | None,
    digital_human_id: str | None = None,
    emotion: str | None = None,
    output_sample_rate: int | None = None,
) -> AsyncIterator[Any]:
    reference = resolve_voice_reference(user_id, digital_human_id)
    engine = get_tts_engine()
    kwargs = dict(
        text=text,
        reference_audio_bytes=reference.wav_bytes,
        reference_text=reference.transcript,
    )
    if provider_name() == "aliyun_cosyvoice":
        kwargs.update(
            voice_id=await ensure_provider_voice(
                user_id=user_id,
                digital_human_id=digital_human_id,
                reference=reference,
            ),
            emotion=emotion,
            user_id=user_id,
        )
        if output_sample_rate is not None:
            kwargs["output_sample_rate"] = output_sample_rate
    async for chunk in engine.synthesize_stream(**kwargs):
        yield chunk


async def open_synthesis_session(
    *,
    user_id: str | None,
    digital_human_id: str | None = None,
    output_sample_rate: int | None = None,
    speech_rate: float | None = None,
    emotion: str | None = None,
):
    """整轮单会话流式合成（aliyun_cosyvoice 专属；其他 provider 返回 None，
    调用方回退逐句 synthesize_stream）。"""
    if provider_name() != "aliyun_cosyvoice":
        return None
    engine = get_tts_engine()
    if not hasattr(engine, "open_stream_session"):
        return None
    voice_id = await ensure_provider_voice(
        user_id=user_id,
        digital_human_id=digital_human_id,
        reference=resolve_voice_reference(user_id, digital_human_id),
    )
    return engine.open_stream_session(
        voice_id=voice_id,
        output_sample_rate=output_sample_rate,
        speech_rate=speech_rate,
        emotion=emotion,
    )


def stream_sample_rate() -> int:
    engine = get_tts_engine()
    return int(getattr(engine, "sample_rate", settings.cosyvoice_stream_sample_rate))


def provider_name() -> str:
    return tts_provider_service.runtime().type


def _aliyun_model_id() -> str:
    runtime = tts_provider_service.runtime()
    return runtime.model or settings.aliyun_cosyvoice_model


__all__ = [
    "VoiceReference",
    "provider_name",
    "ensure_provider_voice",
    "resolve_voice_reference",
    "synthesize_batch",
    "synthesize_stream",
    "stream_sample_rate",
]
