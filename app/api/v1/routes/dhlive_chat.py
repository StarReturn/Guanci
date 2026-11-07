"""豆包式聊天（第三批）：文字消息端点——复用 dialogue 全链路。

从 dialogue.py 提取公共函数 _resolve_session + _reply_pipeline，
文字直入跳过 ASR，其余（人设/记忆/LLM/TTS/配额/落库）全走同一条路。

决策：D4-3 回复带文字+语音 / D4-4 发一等一 / D4-5 历史合并。
"""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.middleware.jwt import get_request_user_id

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatTextRequest(BaseModel):
    text: str
    session_id: str | None = None
    digital_human_id: str | None = None


class ChatReplyRead(BaseModel):
    session_id: str
    reply_text: str
    reply_audio_url: str
    duration_ms: int


@router.post("/text", response_model=ChatReplyRead)
async def chat_text(digital_human_id: str, body: ChatTextRequest, request: Request) -> ChatReplyRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    user_text = body.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text is empty")

    from app.services.llm_service import (
        dialogue_service,
        llm_provider_service,
        SessionOwnershipError,
    )
    from app.services.memory_provider_service import memory_provider_service
    from app.services.entitlement_stub import entitlement_service
    from app.services.digital_human_service import (
        DigitalHumanNotFound,
        digital_human_service,
    )
    from app.services.tts_service import synthesize_batch
    from app.services.storage import storage_service

    # ---- 1. 会话（复用/新建；历史合并 D4-5）----
    if body.digital_human_id and body.digital_human_id != digital_human_id:
        raise HTTPException(status_code=409, detail="digital human does not match request path")
    if body.session_id:
        session = dialogue_service.get_session(body.session_id, uid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if session.digital_human_id != digital_human_id:
            raise HTTPException(status_code=409, detail="session belongs to another digital human")
        session_id = session.session_id
    else:
        if not digital_human_id:
            raise HTTPException(status_code=422, detail="digital_human_id required for new session")
        session_id = None

    if digital_human_id:
        try:
            human = digital_human_service.get_owned(uid, digital_human_id)
            if human.creation_status != "ready":
                raise HTTPException(status_code=422, detail="digital human is not ready")
        except DigitalHumanNotFound as exc:
            raise HTTPException(status_code=404, detail="digital human not found") from exc
    if session_id is None:
        try:
            active_llm = memory_provider_service.get_provider_for_role("chat")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM not configured: {exc}") from exc
        session = dialogue_service.create_session(
            user_id=uid,
            llm_provider_id=active_llm.provider_id if active_llm else None,
            avatar_id=digital_human_id,
            digital_human_id=digital_human_id,
        )
        session_id = session.session_id

    # ---- 2. LLM（人设+记忆，与对话全链一致）----
    try:
        active_llm = memory_provider_service.get_provider_for_role("chat")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    usage_request_id = f"chat-{uuid4()}"
    try:
        entitlement_service.reserve(
            user_id=uid, key="dialogue_audio_seconds_per_month",
            quantity=120, request_id=usage_request_id,
            resource_type="dialogue_session", resource_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=429, detail=f"配额不足：{exc}") from exc

    system_prompt = active_llm.system_prompt or ""
    if digital_human_id:
        persona = digital_human_service.build_persona_prompt(uid, digital_human_id)
        system_prompt = f"{system_prompt}\n\n{persona}".strip()

    membership = entitlement_service.get(uid)
    llm_messages = await asyncio.to_thread(
        dialogue_service.build_llm_messages,
        session_id, user_text, system_prompt=system_prompt,
        user_id=uid, avatar_id=digital_human_id,
        cross_session_memory=bool(membership.entitlements.get("cross_session_memory", False)),
        memory_retention_days=int(membership.entitlements.get("memory_retention_days", 0)),
        memory_items_per_avatar=int(membership.entitlements.get("memory_items_per_avatar", 0)),
    )
    try:
        reply_text = await llm_provider_service.chat(llm_messages, provider=active_llm)
    except Exception as exc:
        logger.exception("chat_llm_failed")
        entitlement_service.release(
            user_id=uid, key="dialogue_audio_seconds_per_month",
            request_id=usage_request_id,
        )
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    # ---- 3. TTS 回复音频（可重播 D4-3）----
    try:
        tts_audio = await synthesize_batch(
            text=reply_text, user_id=uid, digital_human_id=digital_human_id,
        )
    except Exception as exc:
        entitlement_service.release(
            user_id=uid, key="dialogue_audio_seconds_per_month",
            request_id=usage_request_id,
        )
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}")
    entitlement_service.confirm(
        user_id=uid, key="dialogue_audio_seconds_per_month",
        request_id=usage_request_id,
        actual_quantity=max(1, (tts_audio.duration_ms + 999) // 1000),
    )

    audio_id = str(uuid4())
    storage_prefix = (
        f"digital_humans/{uid}/{digital_human_id}/dialogue/{session_id}"
        if digital_human_id else f"dialogue/{session_id}"
    )
    storage_key, _, _ = storage_service.put_bytes(
        key=f"{storage_prefix}/{audio_id}.wav",
        content=tts_audio.wav_bytes, content_type="audio/wav",
    )
    reply_audio_url = storage_service.presigned_url(storage_key)

    # ---- 4. 落库（与对话记录合并展示 D4-5）----
    try:
        dialogue_service.complete_turn_and_enqueue_memory(
            session_id=session_id, user_id=uid,
            digital_human_id=digital_human_id,
            user_text=user_text, assistant_text=reply_text,
            audio_url=f"storage://{storage_key}", memory_enabled=True,
        )
    except SessionOwnershipError as exc:
        logger.warning("chat_turn_ownership_failed detail=%s", str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        dialogue_service.maybe_enqueue_session_summary(
            session_id=session_id, user_id=uid,
            digital_human_id=digital_human_id,
        )
    except Exception:
        logger.exception("chat_session_summary_enqueue_failed")

    return ChatReplyRead(
        session_id=session_id,
        reply_text=reply_text,
        reply_audio_url=reply_audio_url,
        duration_ms=tts_audio.duration_ms,
    )
