"""Realtime session management — NOT WebRTC.

Used by /dialogue endpoint to track multi-turn conversation context.
Mobile app creates a session, then POSTs audio to /dialogue with
session_id for each utterance.

Flow:
  1. POST /realtime/sessions → {session_id}
  2. POST /dialogue (audio + session_id) → {reply_text, reply_audio_url, timeline}
  3. GET /realtime/sessions/{id}/messages → history
"""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.middleware.jwt import get_request_user_id
from app.services.llm_service import dialogue_service
from app.services.storage import storage_service
from app.services.digital_human_service import DigitalHumanNotFound, digital_human_service
from app.services.memory_provider_service import (
    MemoryProviderNotConfigured,
    memory_provider_service,
)

router = APIRouter()


class CreateSessionRequest(BaseModel):
    title: str | None = None
    avatar_id: str


class SessionRead(BaseModel):
    session_id: str
    title: str | None
    avatar_id: str | None
    created_at: str
    updated_at: str | None = None
    last_message: str | None = None
    last_message_role: str | None = None


class MessageRead(BaseModel):
    message_id: str
    role: str
    content: str
    audio_url: str | None
    created_at: str


@router.post("/sessions", response_model=SessionRead)
async def create_session(body: CreateSessionRequest, request: Request) -> SessionRead:
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    try:
        digital_human = digital_human_service.get_owned(uid, body.avatar_id)
    except DigitalHumanNotFound as exc:
        raise HTTPException(status_code=404, detail="digital human not found") from exc
    if digital_human.creation_status != "ready":
        raise HTTPException(status_code=422, detail="digital human is not ready")
    try:
        active_llm = memory_provider_service.get_provider_for_role("chat")
    except MemoryProviderNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    session = dialogue_service.create_session(
        user_id=uid,
        llm_provider_id=active_llm.provider_id if active_llm else None,
        avatar_id=body.avatar_id,
        digital_human_id=body.avatar_id,
        title=body.title or "新对话",
    )
    return SessionRead(
        session_id=session.session_id,
        title=session.title,
        avatar_id=session.avatar_id,
        created_at=session.created_at.isoformat(),
    )


@router.get("/sessions", response_model=list[SessionRead])
async def list_sessions(request: Request, date: str | None = None) -> list[SessionRead]:
    """date=YYYY-MM-DD（可选）：只返回该日（东八区）有活动的会话。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    _validate_date(date)
    sessions = dialogue_service.list_sessions(uid, date=date)
    latest = dialogue_service.latest_messages_for_sessions(uid, [s.session_id for s in sessions])
    return [
        SessionRead(
            session_id=s.session_id, title=s.title, avatar_id=s.avatar_id,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            last_message=latest[s.session_id].content[:120] if s.session_id in latest else None,
            last_message_role=latest[s.session_id].role if s.session_id in latest else None,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(session_id: str, request: Request, date: str | None = None) -> list[MessageRead]:
    """date=YYYY-MM-DD（可选）：只返回该日（东八区）的消息。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    _validate_date(date)
    session = dialogue_service.get_session(session_id, uid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    messages = dialogue_service.list_messages_for_user(
        session_id=session_id,
        user_id=uid,
        digital_human_id=session.digital_human_id,
        limit=50,
        date=date,
    )
    return [
        MessageRead(
            message_id=m.message_id, role=m.role, content=m.content,
            audio_url=storage_service.playback_url(m.audio_url), created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


def _validate_date(date: str | None) -> None:
    if date is None:
        return
    from datetime import datetime

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    """删除整个会话及其全部消息（仅本人）。不可恢复。"""
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    deleted_msgs = dialogue_service.delete_session(session_id, uid)
    if deleted_msgs is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True, "deleted_messages": deleted_msgs}
