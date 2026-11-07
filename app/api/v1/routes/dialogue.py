"""Dialogue route — end-to-end voice conversation.

Flow: user audio → ASR → LLM → TTS → phoneme timeline → return

POST /api/v1/dialogue
  Authorization: Bearer <jwt>
  Body: multipart/form-data
    audio: wav file (user's voice)
    session_id: optional (for multi-turn; if omitted, new session is created)

Response JSON:
    {
      "session_id": "...",
      "user_text": "你好",           ← ASR result
      "reply_text": "你好！很高兴...",  ← LLM result
      "reply_audio_url": "https://...", ← TTS wav
      "phoneme_timeline": { "timeline": [...] },
      "duration_ms": 1820,
      "voice_cloned": true             ← 是否用了用户上传的声纹
    }

Voice cloning uses the configured TTS provider. Scheme C selects CosyVoice 3
and passes the user's primary avatar reference WAV and transcript zero-shot.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.middleware.jwt import get_request_user_id
from app.services.asr_service import asr_service
from app.services.auth_service import audit_service
from app.services.llm_service import dialogue_service, llm_provider_service
from app.services.memory_provider_service import (
    MemoryProviderNotConfigured,
    memory_provider_service,
)
from app.services.entitlement_stub import (
    EntitlementDenied,
    MembershipError,
    entitlement_service,
)
from app.services.phoneme_service import phoneme_service
from app.services.storage import storage_service
from app.services.audio_utils import normalize_audio_to_wav, wav_duration_ms
from app.services.tts_service import synthesize_batch
from app.services.digital_human_service import (
    DigitalHumanNotFound,
    digital_human_service,
)
from uuid import uuid4

logger = logging.getLogger(__name__)
router = APIRouter()


class DialogueResponse(BaseModel):
    session_id: str
    user_text: str
    reply_text: str
    reply_audio_url: str
    user_audio_url: str | None = None
    user_audio_duration_ms: int = 0
    phoneme_timeline: dict | None = None
    duration_ms: int
    # True if TTS used the user's uploaded voice sample (clone).
    voice_cloned: bool


@router.post("", response_model=DialogueResponse)
async def dialogue(
    request: Request,
    audio: UploadFile = File(..., description="用户语音 wav"),
    session_id: str | None = Form(default=None, description="多轮对话上下文（首次可不传）"),
    digital_human_id: str | None = Form(
        default=None, description="要对话的数字人；新会话建议必传"
    ),
) -> DialogueResponse:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")

    audio_bytes = await audio.read(10 * 1024 * 1024 + 1)
    if not audio_bytes or len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="audio must be between 1 byte and 10 MiB")
    try:
        audio_bytes = await asyncio.to_thread(
            normalize_audio_to_wav, audio_bytes, audio.content_type or "application/octet-stream", 16000,
        )
        user_audio_duration_ms = wav_duration_ms(audio_bytes)
    except (ValueError, OSError, EOFError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(status_code=400, detail="invalid audio file") from exc

    # ---- 1. ASR: audio → text ----
    # Stage C: route through the unified ASR provider factory instead of
    # calling asr_service directly. The factory returns either an
    # ASREngine (aliyun_asr/funasr) with a streaming session, or the
    # legacy whisper batch service. Non-realtime HTTP endpoints drive
    # the session end-to-end via transcribe_complete_audio.
    try:
        from app.services.engine_factory import get_asr_engine
        from app.services.asr_helpers import transcribe_complete_audio
        from app.services.audio_utils import wav_to_pcm

        asr_engine = get_asr_engine()
        if hasattr(asr_engine, "create_session"):
            # Streaming-capable provider (aliyun_asr / funasr): decode WAV
            # to PCM s16le, then drive a session end-to-end.
            try:
                pcm_bytes, pcm_sample_rate = wav_to_pcm(audio_bytes)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"audio decode failed: {exc}"
                ) from exc
            user_text = await transcribe_complete_audio(
                asr_engine, bytes(pcm_bytes),
                sample_rate=pcm_sample_rate, channels=1, encoding="pcm_s16le",
            )
        else:
            # Whisper batch fallback (debug only).
            user_text = await asyncio.to_thread(asr_service.transcribe, audio_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("asr_failed")
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc

    if not user_text:
        raise HTTPException(status_code=400, detail="no speech detected in audio")

    try:
        active_llm = memory_provider_service.get_provider_for_role("chat")
    except MemoryProviderNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # ---- 2. Get or create session ----
    if session_id:
        session = dialogue_service.get_session(session_id, uid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if digital_human_id and session.digital_human_id != digital_human_id:
            raise HTTPException(status_code=409, detail="session belongs to another digital human")
        digital_human_id = session.digital_human_id or digital_human_id
        if digital_human_id:
            try:
                digital_human = digital_human_service.get_owned(uid, digital_human_id)
            except DigitalHumanNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if digital_human.creation_status != "ready":
                raise HTTPException(status_code=422, detail="digital human is not ready")
        else:
            raise HTTPException(
                status_code=409,
                detail="session is missing digital_human_id; create a new session",
            )
    else:
        if digital_human_id:
            try:
                digital_human = digital_human_service.get_owned(uid, digital_human_id)
            except DigitalHumanNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if digital_human.creation_status != "ready":
                raise HTTPException(status_code=422, detail="digital human is not ready")
        session = dialogue_service.create_session(
            user_id=uid,
            llm_provider_id=active_llm.provider_id if active_llm else None,
            avatar_id=digital_human_id,
            digital_human_id=digital_human_id,
        )
        session_id = session.session_id

    # ---- 3. LLM: text → reply ----
    usage_request_id = request.headers.get("Idempotency-Key") or f"dialogue-{uuid4()}"
    try:
        entitlement_service.reserve(
            user_id=uid,
            key="dialogue_audio_seconds_per_month",
            quantity=120,
            request_id=usage_request_id,
            resource_type="dialogue_session",
            resource_id=session_id,
        )
    except EntitlementDenied as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "ENTITLEMENT_LIMIT_EXCEEDED",
                "entitlement_key": exc.key,
                "limit": exc.limit,
                "used": exc.used,
                "requested": exc.requested,
            },
        ) from exc
    except MembershipError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    system_prompt = active_llm.system_prompt or ""
    if digital_human_id:
        persona_prompt = digital_human_service.build_persona_prompt(uid, digital_human_id)
        system_prompt = f"{system_prompt}\n\n{persona_prompt}".strip()
    membership = entitlement_service.get(uid)
    llm_messages = await asyncio.to_thread(
        dialogue_service.build_llm_messages,
        session_id, user_text, system_prompt=system_prompt,
        user_id=uid, avatar_id=digital_human_id,
        cross_session_memory=bool(
            membership.entitlements.get("cross_session_memory", False)
        ),
        memory_retention_days=int(
            membership.entitlements.get("memory_retention_days", 0)
        ),
        memory_items_per_avatar=int(
            membership.entitlements.get("memory_items_per_avatar", 0)
        ),
    )
    try:
        reply_text = await llm_provider_service.chat(llm_messages, provider=active_llm)
    except Exception as exc:
        logger.exception("llm_failed")
        entitlement_service.release(
            user_id=uid,
            key="dialogue_audio_seconds_per_month",
            request_id=usage_request_id,
        )
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    # ---- 4. TTS: reply → wav (with cloned voice if available) ----
    try:
        tts_audio = await synthesize_batch(
            text=reply_text,
            user_id=uid,
            digital_human_id=digital_human_id,
        )
    except Exception as exc:
        entitlement_service.release(
            user_id=uid,
            key="dialogue_audio_seconds_per_month",
            request_id=usage_request_id,
        )
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}")
    entitlement_service.confirm(
        user_id=uid,
        key="dialogue_audio_seconds_per_month",
        request_id=usage_request_id,
        actual_quantity=max(1, (tts_audio.duration_ms + 999) // 1000),
    )

    # Save wav to storage
    audio_id = str(uuid4())
    storage_prefix = (
        f"digital_humans/{uid}/{digital_human_id}/dialogue/{session_id}"
        if digital_human_id
        else f"dialogue/{session_id}"
    )
    storage_key, _, _ = storage_service.put_bytes(
        key=f"{storage_prefix}/{audio_id}.wav",
        content=tts_audio.wav_bytes,
        content_type="audio/wav",
    )
    reply_audio_url = storage_service.presigned_url(storage_key)
    user_audio_key, _, _ = storage_service.put_bytes(
        key=f"{storage_prefix}/{audio_id}-user.wav",
        content=audio_bytes,
        content_type="audio/wav",
    )
    user_audio_url = storage_service.presigned_url(user_audio_key)

    # ---- 5. Phoneme alignment: reply wav → timeline ----
    # 口型/字幕时间线是锦上添花：对齐服务缺失或失败时降级为无时间线
    # 返回（响应契约允许 phoneme_timeline=None），不能因此 500 掉整轮对话。
    try:
        timeline = await phoneme_service.align_text(
            text=reply_text,
            duration_ms=tts_audio.duration_ms,
            audio_id=session_id,
            audio_wav=tts_audio.wav_bytes,
        )
    except Exception as exc:  # PhonemeAlignmentError 及底层连接错误
        logger.warning("dialogue_phoneme_align_degraded detail=%s", str(exc)[:200])
        timeline = None

    # ---- 6. Atomically persist the completed turn and enqueue memory ----
    from app.services.llm_service import SessionOwnershipError

    try:
        completed_turn = dialogue_service.complete_turn_and_enqueue_memory(
            session_id=session_id, user_id=uid,
            digital_human_id=digital_human_id,
            user_text=user_text, assistant_text=reply_text,
            audio_url=f"storage://{storage_key}",
            user_audio_url=f"storage://{user_audio_key}",
            memory_enabled=True,
        )
    except SessionOwnershipError as exc:
        logger.warning("dialogue_turn_ownership_failed detail=%s", str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Trigger a session summary if the unsummarised message count crossed
    # the threshold. Safe to fail silently — summarisation is best-effort.
    try:
        dialogue_service.maybe_enqueue_session_summary(
            session_id=session_id, user_id=uid,
            digital_human_id=digital_human_id,
        )
    except Exception:
        logger.exception("session_summary_enqueue_failed")

    # ---- 7. Audit + stats ----
    audit_service.log(
        action="user.dialogue",
        actor_user_id=uid,
        detail={
            "session_id": session_id,
            "user_text_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
            "user_text_chars": len(user_text),
            "voice_cloned": tts_audio.used_reference,
        },
    )

    return DialogueResponse(
        session_id=session_id,
        user_text=user_text,
        reply_text=reply_text,
        reply_audio_url=reply_audio_url,
        user_audio_url=user_audio_url,
        user_audio_duration_ms=user_audio_duration_ms,
        phoneme_timeline=timeline.model_dump() if timeline else None,
        duration_ms=tts_audio.duration_ms,
        voice_cloned=tts_audio.used_reference,
    )
