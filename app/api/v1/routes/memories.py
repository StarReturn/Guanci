"""User-facing long-term memory routes for a relationship space.

All routes are scoped to ``/api/v1/me/digital-humans/{digital_human_id}/...``
and implicitly use the current authenticated user. The body of any
request MUST NOT carry a ``user_id`` or ``digital_human_id`` field —
both are taken from the path / auth context.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.middleware.jwt import get_request_user_id
from app.schemas.memory import (
    AvatarMemoryRead,
    MemoryExportResponse,
    MemoryItemCreate,
    MemoryItemRead,
    MemoryItemUpdate,
    MemoryStatus,
    RelationshipProfilePatch,
    RelationshipProfileRead,
    RelationshipPurgeResponse,
)
from app.services.auth_service import audit_service
from app.services.digital_human_service import (
    DigitalHumanNotFound,
    digital_human_service,
)
from app.services.memory_profile_service import memory_profile_service
from app.services.memory_service import (
    MemoryNotFound,
    MemoryValidationError,
    memory_service,
)

router = APIRouter()


def _owner(request: Request, digital_human_id: str) -> str:
    user_id = get_request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="user login required")
    try:
        digital_human_service.get_owned(user_id, digital_human_id)
    except DigitalHumanNotFound as exc:
        raise HTTPException(status_code=404, detail="digital human not found") from exc
    return user_id


def _read(row) -> MemoryItemRead:
    return MemoryItemRead(
        memory_id=row.memory_id,
        digital_human_id=row.digital_human_id,
        memory_type=row.memory_type,
        category=row.category,
        fact_key=row.fact_key,
        summary=row.summary or "",
        tags=list(row.tags or []),
        importance_score=row.importance_score,
        confidence_score=row.confidence_score,
        user_confirmed=bool(row.user_confirmed),
        sensitivity=row.sensitivity,
        occurred_at=row.occurred_at,
        last_recalled_at=row.last_recalled_at,
        recall_count=row.recall_count,
        status=row.status,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        superseded_by_memory_id=row.superseded_by_memory_id,
        expires_at=row.expires_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _legacy_read(row) -> AvatarMemoryRead:
    return AvatarMemoryRead(
        memory_id=row.memory_id,
        digital_human_id=row.digital_human_id,
        category=row.category,
        summary=row.summary or "",
        tags=list(row.tags or []),
        importance_score=row.importance_score,
        sensitivity=row.sensitivity,
        occurred_at=row.occurred_at,
        last_recalled_at=row.last_recalled_at,
        recall_count=row.recall_count,
        status=row.status,
        expires_at=row.expires_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/{digital_human_id}/memories")
async def list_memories(
    digital_human_id: str,
    request: Request,
    status: MemoryStatus = "active",
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    user_id = _owner(request, digital_human_id)
    rows, total = memory_service.list_for_user(
        user_id=user_id, digital_human_id=digital_human_id,
        status=status, limit=limit, offset=offset,
    )
    return {"items": [_read(row) for row in rows], "total": total}


@router.post("/{digital_human_id}/memories", response_model=MemoryItemRead)
async def create_memory(
    digital_human_id: str,
    body: MemoryItemCreate,
    request: Request,
) -> MemoryItemRead:
    user_id = _owner(request, digital_human_id)
    try:
        row = memory_service.create_for_user(
            user_id=user_id, digital_human_id=digital_human_id,
            summary=body.summary, category=body.category,
            memory_type=body.memory_type, fact_key=body.fact_key,
            tags=body.tags, importance_score=body.importance_score,
        )
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_service.log(
        action="memory.user_create", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_item", target_id=row.memory_id,
        detail={"digital_human_id": digital_human_id, "category": body.category},
    )
    return _read(row)


@router.patch(
    "/{digital_human_id}/memories/{memory_id}",
    response_model=MemoryItemRead,
)
async def update_memory(
    digital_human_id: str,
    memory_id: str,
    body: MemoryItemUpdate,
    request: Request,
) -> MemoryItemRead:
    user_id = _owner(request, digital_human_id)
    try:
        row = memory_service.update_for_user(
            user_id=user_id, digital_human_id=digital_human_id,
            memory_id=memory_id, **body.model_dump(exclude_unset=True),
        )
    except MemoryNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_service.log(
        action="memory.user_update", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_item", target_id=memory_id,
        detail={"fields": sorted(body.model_dump(exclude_unset=True))},
    )
    return _read(row)


@router.delete(
    "/{digital_human_id}/memories/{memory_id}",
    status_code=204,
)
async def delete_memory(
    digital_human_id: str,
    memory_id: str,
    request: Request,
) -> Response:
    user_id = _owner(request, digital_human_id)
    try:
        memory_service.delete_for_user(
            user_id=user_id, digital_human_id=digital_human_id,
            memory_id=memory_id,
        )
    except MemoryNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audit_service.log(
        action="memory.user_delete", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_item", target_id=memory_id,
    )
    return Response(status_code=204)


@router.get("/{digital_human_id}/memories/export", response_model=MemoryExportResponse)
async def export_memories(
    digital_human_id: str,
    request: Request,
) -> MemoryExportResponse:
    user_id = _owner(request, digital_human_id)
    rows = memory_service.export_for_user(
        user_id=user_id, digital_human_id=digital_human_id,
    )
    audit_service.log(
        action="memory.user_export", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_relationship", target_id=digital_human_id,
    )
    return MemoryExportResponse(
        digital_human_id=digital_human_id,
        items=[_read(row) for row in rows],
    )


@router.delete(
    "/{digital_human_id}/memories",
    response_model=RelationshipPurgeResponse,
)
async def purge_relationship_memories(
    digital_human_id: str,
    request: Request,
) -> RelationshipPurgeResponse:
    user_id = _owner(request, digital_human_id)
    deleted = memory_service.purge_relationship(
        user_id=user_id, digital_human_id=digital_human_id,
    )
    audit_service.log(
        action="memory.user_purge_relationship", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_relationship", target_id=digital_human_id,
        detail={"deleted_count": deleted},
    )
    return RelationshipPurgeResponse(
        digital_human_id=digital_human_id, deleted_count=deleted,
    )


@router.get(
    "/{digital_human_id}/memory-profile",
    response_model=RelationshipProfileRead,
)
async def get_relationship_profile(
    digital_human_id: str,
    request: Request,
) -> RelationshipProfileRead:
    user_id = _owner(request, digital_human_id)
    row = memory_profile_service.get(
        user_id=user_id, digital_human_id=digital_human_id,
    )
    if row is None:
        return RelationshipProfileRead(
            user_id=user_id, digital_human_id=digital_human_id,
            profile_text="", profile_version=0, last_built_at=None,
        )
    return RelationshipProfileRead(
        user_id=row.user_id, digital_human_id=row.digital_human_id,
        profile_text=memory_profile_service.effective_text(row),
        profile_version=row.profile_version,
        last_built_at=row.last_built_at,
    )


@router.patch(
    "/{digital_human_id}/memory-profile",
    response_model=RelationshipProfileRead,
)
async def replace_relationship_profile(
    digital_human_id: str,
    body: RelationshipProfilePatch,
    request: Request,
) -> RelationshipProfileRead:
    """User-edited override. The text is rebuilt into the prompt with
    bounded envelopes and HTML escaping; we never let raw user text
    reach the system prompt unfiltered.
    """
    user_id = _owner(request, digital_human_id)
    row = memory_profile_service.rebuild(
        user_id=user_id, digital_human_id=digital_human_id,
    )
    # Store an explicit override separately so the next automatic rebuild
    # cannot silently erase the user's edit.
    from app.core.database import MemoryRelationshipProfileRecord, SessionLocal
    from datetime import UTC, datetime
    from sqlalchemy import select
    from app.services.memory_security_service import classify, contains_high_risk_credentials

    cleaned = " ".join(body.profile_text.split()).strip()
    if contains_high_risk_credentials(cleaned):
        raise HTTPException(status_code=422, detail="profile must not contain credentials")
    if cleaned and classify(cleaned, category="relationship").sensitivity == "sensitive":
        raise HTTPException(
            status_code=422,
            detail="sensitive information must be stored as an encrypted memory item",
        )
    with SessionLocal() as db:
        row = db.execute(select(MemoryRelationshipProfileRecord).where(
            MemoryRelationshipProfileRecord.user_id == user_id,
            MemoryRelationshipProfileRecord.digital_human_id == digital_human_id,
        ).with_for_update()).scalar_one()
        row.user_override_text = cleaned or None
        row.profile_version = int(row.profile_version or 0) + 1
        row.last_built_at = datetime.now(UTC)
        db.commit()
        db.refresh(row)
    audit_service.log(
        action="memory.user_profile_override", actor_user_id=user_id,
        actor_ip=request.client.host if request.client else None,
        target_type="memory_relationship_profile", target_id=digital_human_id,
        detail={"profile_version": row.profile_version, "chars": len(body.profile_text)},
    )
    return RelationshipProfileRead(
        user_id=row.user_id, digital_human_id=row.digital_human_id,
        profile_text=memory_profile_service.effective_text(row),
        profile_version=row.profile_version,
        last_built_at=row.last_built_at,
    )


__all__ = ["router"]
