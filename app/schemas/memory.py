from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


MemoryCategory = Literal[
    "identity", "relationship", "family", "work",
    "preference", "plan", "event", "health",
]
MemoryType = Literal["semantic_fact", "episodic_event"]
MemoryStatus = Literal["active", "superseded", "suspended", "deleted"]
Sensitivity = Literal["normal", "sensitive"]


class AvatarMemoryRead(BaseModel):
    """Legacy compat alias for callers that import ``AvatarMemoryRead``.

    Maps to the new ``MemoryItemRead`` shape so existing clients continue
    to parse responses. ``summary`` is empty for sensitive rows; callers
    that have explicit permission can request the decrypted view via the
    admin super-admin endpoint.
    """

    memory_id: str
    digital_human_id: str
    category: str
    summary: str
    tags: list[str]
    importance_score: float
    sensitivity: str
    occurred_at: datetime | None
    last_recalled_at: datetime | None
    recall_count: int
    status: str
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class MemoryItemRead(BaseModel):
    memory_id: str
    digital_human_id: str
    memory_type: MemoryType
    category: MemoryCategory
    fact_key: str | None
    summary: str
    tags: list[str]
    importance_score: float
    confidence_score: float
    user_confirmed: bool
    sensitivity: Sensitivity
    occurred_at: datetime | None
    last_recalled_at: datetime | None
    recall_count: int
    status: MemoryStatus
    valid_from: datetime | None
    valid_to: datetime | None
    superseded_by_memory_id: str | None
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class MemoryItemListResponse(BaseModel):
    items: list[MemoryItemRead]
    total: int


class AvatarMemoryListResponse(BaseModel):
    """Legacy alias preserved for older clients."""

    items: list[AvatarMemoryRead]
    total: int


class MemoryItemCreate(BaseModel):
    summary: str = Field(..., min_length=1, max_length=500)
    category: MemoryCategory
    memory_type: MemoryType = "semantic_fact"
    fact_key: str | None = Field(default=None, max_length=128)
    tags: list[str] | None = Field(default=None, max_length=8)
    importance_score: float = Field(default=0.7, ge=0.0, le=1.0)


class AvatarMemoryUpdate(BaseModel):
    """Legacy update shape preserved for old SDK clients."""

    summary: str | None = Field(default=None, min_length=1, max_length=500)
    category: MemoryCategory | None = None
    tags: list[str] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def require_change(self):
        if self.summary is None and self.category is None and self.tags is None:
            raise ValueError("at least one memory field must be provided")
        return self


class MemoryItemUpdate(AvatarMemoryUpdate):
    """New name; same shape as ``AvatarMemoryUpdate``."""


class RelationshipProfileRead(BaseModel):
    user_id: str
    digital_human_id: str
    profile_text: str
    profile_version: int
    last_built_at: datetime | None


class RelationshipProfilePatch(BaseModel):
    """Allow the user to override the auto-built profile. The text is
    sanitised on output the same way auto-built text is, so explicit
    edits cannot smuggle prompt injection into the LLM context.
    """

    profile_text: str = Field(..., min_length=0, max_length=2000)


class AdminMemoryMetadataRead(BaseModel):
    memory_id: str
    user_id: str
    digital_human_id: str
    source_session_id: str | None
    memory_type: MemoryType
    category: MemoryCategory
    importance_score: float
    sensitivity: Sensitivity
    recall_count: int
    status: MemoryStatus
    expires_at: datetime | None
    suspended_at: datetime | None
    recovery_delete_at: datetime | None
    created_at: datetime
    updated_at: datetime


class AdminMemoryMetadataListResponse(BaseModel):
    items: list[AdminMemoryMetadataRead]
    total: int


class MemoryJobRead(BaseModel):
    job_id: str
    job_type: str
    user_id: str
    digital_human_id: str
    session_id: str | None
    trigger_message_id: str | None
    status: str
    attempts: int
    candidate_count: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_code: str | None
    error_detail_redacted: str | None
    created_at: datetime


class MemoryJobListResponse(BaseModel):
    items: list[MemoryJobRead]
    total: int


class MemoryExtractionJobRead(BaseModel):
    """Legacy compat — new code uses MemoryJobRead."""

    job_id: str
    user_id: str
    digital_human_id: str
    session_id: str
    trigger_message_id: str
    status: str
    attempts: int
    candidate_count: int
    next_attempt_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    created_at: datetime


class MemoryExtractionJobListResponse(BaseModel):
    items: list[MemoryExtractionJobRead]
    total: int


class AdminMemoryDeleteRequest(BaseModel):
    reason: str = Field(..., min_length=2, max_length=500)


class MemoryExportResponse(BaseModel):
    digital_human_id: str
    items: list[MemoryItemRead]


class RelationshipPurgeResponse(BaseModel):
    digital_human_id: str
    deleted_count: int


__all__ = [
    "AdminMemoryDeleteRequest",
    "AdminMemoryMetadataListResponse",
    "AdminMemoryMetadataRead",
    "AvatarMemoryListResponse",
    "AvatarMemoryRead",
    "AvatarMemoryUpdate",
    "MemoryCategory",
    "MemoryExtractionJobListResponse",
    "MemoryExtractionJobRead",
    "MemoryItemCreate",
    "MemoryItemListResponse",
    "MemoryItemRead",
    "MemoryItemUpdate",
    "MemoryJobListResponse",
    "MemoryJobRead",
    "MemoryStatus",
    "MemoryType",
    "RelationshipProfilePatch",
    "RelationshipProfileRead",
    "RelationshipPurgeResponse",
    "Sensitivity",
]
