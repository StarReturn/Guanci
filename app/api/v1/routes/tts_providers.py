"""Super-administrator routes for TTS provider configuration."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies.admin import AdminPrincipal, require_super_admin
from app.services.auth_service import audit_service
from app.services.tts_provider_service import tts_provider_service

router = APIRouter()


class TtsProviderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: Literal["aliyun_cosyvoice", "cosyvoice", "fish_speech", "indextts"]
    api_key: str | None = Field(default=None, min_length=1, max_length=2048)
    workspace_id: str | None = Field(default=None, max_length=128)
    endpoint: str | None = Field(default=None, max_length=512)
    model: str | None = Field(default=None, max_length=128)
    default_voice_id: str | None = Field(default=None, max_length=256)
    is_active: bool = False


class TtsProviderUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=64)
    type: Literal["aliyun_cosyvoice", "cosyvoice", "fish_speech", "indextts"] | None = None
    api_key: str | None = Field(default=None, min_length=1, max_length=2048)
    workspace_id: str | None = Field(default=None, max_length=128)
    endpoint: str | None = Field(default=None, max_length=512)
    model: str | None = Field(default=None, max_length=128)
    default_voice_id: str | None = Field(default=None, max_length=256)
    is_active: bool | None = None


class TtsProviderRead(BaseModel):
    provider_id: str
    name: str
    type: str
    api_key_masked: str | None
    workspace_id: str | None
    endpoint: str | None
    model: str | None
    default_voice_id: str | None
    is_active: bool


def _to_read(rec) -> TtsProviderRead:
    return TtsProviderRead(
        provider_id=rec.provider_id,
        name=rec.name,
        type=rec.type,
        api_key_masked=tts_provider_service.masked_api_key(rec),
        workspace_id=rec.workspace_id,
        endpoint=rec.endpoint,
        model=rec.model,
        default_voice_id=rec.default_voice_id,
        is_active=rec.is_active,
    )


@router.get("", response_model=list[TtsProviderRead])
async def list_providers(
    principal: AdminPrincipal = Depends(require_super_admin),
) -> list[TtsProviderRead]:
    del principal
    return [_to_read(rec) for rec in tts_provider_service.list()]


@router.post("", response_model=TtsProviderRead, status_code=201)
async def create_provider(
    body: TtsProviderCreate,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> TtsProviderRead:
    rec = tts_provider_service.create(**body.model_dump())
    audit_service.log(
        action="admin.tts_provider.create",
        actor_user_id=principal.audit_user_id,
        target_type="tts_provider",
        target_id=rec.provider_id,
        detail={"name": body.name, "type": body.type},
    )
    return _to_read(rec)


@router.patch("/{provider_id}", response_model=TtsProviderRead)
async def update_provider(
    provider_id: str,
    body: TtsProviderUpdate,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> TtsProviderRead:
    rec = tts_provider_service.update(provider_id, **body.model_dump(exclude_none=True))
    if rec is None:
        raise HTTPException(status_code=404, detail="provider not found")
    audit_service.log(
        action="admin.tts_provider.update",
        actor_user_id=principal.audit_user_id,
        target_type="tts_provider",
        target_id=provider_id,
        detail={"fields": sorted(body.model_dump(exclude_none=True))},
    )
    return _to_read(rec)


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> dict:
    if not tts_provider_service.delete(provider_id):
        raise HTTPException(status_code=404, detail="provider not found")
    audit_service.log(
        action="admin.tts_provider.delete",
        actor_user_id=principal.audit_user_id,
        target_type="tts_provider",
        target_id=provider_id,
    )
    return {"deleted": True}
