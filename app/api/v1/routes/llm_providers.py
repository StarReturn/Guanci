"""Admin routes for LLM Provider management.

Admin can add multiple providers (GLM/豆包/通义) and switch the active one.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.auth_service import audit_service
from app.services.llm_service import llm_provider_service
from app.api.dependencies.admin import AdminPrincipal, require_super_admin


router = APIRouter()


class LlmProviderCreate(BaseModel):
    name: str = Field(..., max_length=64)
    type: Literal["glm", "doubao", "qwen", "openai"]
    api_key: str = Field(..., max_length=2048)
    base_url: str = Field(..., max_length=512)
    model: str = Field(..., max_length=128)
    is_active: bool = False
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


class LlmProviderUpdate(BaseModel):
    name: str | None = None
    api_key: str | None = Field(default=None, min_length=1, max_length=2048)
    base_url: str | None = None
    model: str | None = None
    is_active: bool | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class LlmProviderRead(BaseModel):
    provider_id: str
    name: str
    type: str
    api_key_masked: str
    base_url: str
    model: str
    is_active: bool
    system_prompt: str | None = None
    temperature: float
    max_tokens: int


def _to_read(rec) -> LlmProviderRead:
    return LlmProviderRead(
        provider_id=rec.provider_id, name=rec.name, type=rec.type,
        api_key_masked=llm_provider_service.masked_api_key(rec),
        base_url=rec.base_url, model=rec.model,
        is_active=rec.is_active, system_prompt=rec.system_prompt,
        temperature=rec.temperature, max_tokens=rec.max_tokens,
    )


@router.get("")
async def list_providers(
    principal: AdminPrincipal = Depends(require_super_admin),
) -> list[LlmProviderRead]:
    del principal
    return [_to_read(r) for r in llm_provider_service.list()]


@router.post("", response_model=LlmProviderRead, status_code=201)
async def create_provider(
    body: LlmProviderCreate,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> LlmProviderRead:
    rec = llm_provider_service.create(**body.model_dump())
    audit_service.log(
        action="admin.llm_provider.create",
        actor_user_id=principal.audit_user_id,
        target_type="llm_provider",
        target_id=rec.provider_id,
        detail={"name": body.name, "type": body.type},
    )
    return _to_read(rec)


@router.patch("/{provider_id}", response_model=LlmProviderRead)
async def update_provider(
    provider_id: str,
    body: LlmProviderUpdate,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> LlmProviderRead:
    rec = llm_provider_service.update(provider_id, **body.model_dump(exclude_none=True))
    if rec is None:
        raise HTTPException(status_code=404, detail="provider not found")
    audit_service.log(
        action="admin.llm_provider.update",
        actor_user_id=principal.audit_user_id,
        target_type="llm_provider",
        target_id=provider_id,
        detail={"fields": sorted(body.model_dump(exclude_none=True))},
    )
    return _to_read(rec)


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    principal: AdminPrincipal = Depends(require_super_admin),
) -> dict:
    ok = llm_provider_service.delete(provider_id)
    if not ok:
        raise HTTPException(status_code=404, detail="provider not found")
    audit_service.log(
        action="admin.llm_provider.delete",
        actor_user_id=principal.audit_user_id,
        target_type="llm_provider",
        target_id=provider_id,
    )
    return {"deleted": True}
