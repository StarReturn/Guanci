"""API schemas for the digital human aggregate."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RelationshipType = Literal[
    "self", "parent", "grandparent", "spouse", "sibling", "child", "friend", "other"
]
LifeStatus = Literal["living", "deceased"]
CreationMode = Literal["self", "helping_other"]
Gender = Literal["female", "male", "other", "unknown"]
AgeGroup = Literal["child", "young", "middle_aged", "senior", "unknown"]
ConsentType = Literal[
    "face_processing",
    "voice_processing",
    "third_party_authorization",
    "deceased_person_representation",
]


class DigitalHumanCreate(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=64)
    relationship_type: RelationshipType = "self"
    custom_relationship: str | None = Field(default=None, max_length=64)
    life_status: LifeStatus = "living"
    age_group: AgeGroup | None = None
    gender: Gender | None = None
    dialect_code: str | None = Field(default=None, max_length=32)
    preferred_address: str | None = Field(default=None, max_length=64)
    creation_mode: CreationMode = "self"

    @model_validator(mode="after")
    def validate_relationship(self):
        if self.relationship_type == "other" and not (
            self.custom_relationship or ""
        ).strip():
            raise ValueError(
                "custom_relationship is required when relationship_type is other"
            )
        return self


class DigitalHumanUpdate(BaseModel):
    expected_revision: int = Field(..., ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    relationship_type: RelationshipType | None = None
    custom_relationship: str | None = Field(default=None, max_length=64)
    life_status: LifeStatus | None = None
    age_group: AgeGroup | None = None
    gender: Gender | None = None
    dialect_code: str | None = Field(default=None, max_length=32)
    preferred_address: str | None = Field(default=None, max_length=64)


class ConsentWrite(BaseModel):
    consent_type: ConsentType
    policy_version: str = Field(..., min_length=1, max_length=32)
    is_granted: bool
    evidence: dict[str, Any] | None = None


class ConsentRead(BaseModel):
    consent_id: str
    consent_type: str
    policy_version: str
    is_granted: bool
    granted_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class PersonaVersionCreate(BaseModel):
    profile: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = Field(default=None, max_length=4000)
    activate: bool = True


class PersonaVersionRead(BaseModel):
    persona_version_id: str
    version: int
    profile: dict[str, Any]
    summary: str | None
    status: str
    created_at: datetime


class BehaviorAnchorWrite(BaseModel):
    category: str = Field(..., min_length=1, max_length=32)
    trigger_text: str = Field(..., min_length=1, max_length=1000)
    response_guidance: str = Field(..., min_length=1, max_length=2000)
    priority: int = Field(default=0, ge=-1000, le=1000)
    is_enabled: bool = True


class BehaviorAnchorRead(BehaviorAnchorWrite):
    behavior_anchor_id: str
    created_at: datetime
    updated_at: datetime


class PackageVersionRead(BaseModel):
    package_version_id: str
    avatar_id: str
    build_task_id: str | None
    version: int
    status: str
    engine: str
    quality_profile: str | None
    # mobile_keyframe_v3 fields (replaces legacy single-value ``resolution``).
    package_format_version: int = 3
    inference_width: int = 768
    inference_height: int = 768
    canvas_width: int = 768
    canvas_height: int = 1024
    canvas_profile: str = "portrait_3_4"
    renderer_mode: str = "base_plus_mouth_roi"
    package_schema_revision: int = 2
    renderer_runtime_version: str | None = None
    build_fingerprint: str | None = None
    source_asset_sha256: str | None = None
    manifest_metadata: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    reference_render_url: str | None = None
    size_bytes: int
    sha256: str
    download_url: str | None
    created_at: datetime
    activated_at: datetime | None


class DigitalHumanBackgroundRead(BaseModel):
    digital_human_background_id: str
    source: str
    background_id: str | None
    preview_url: str
    is_active: bool
    created_at: datetime


class DigitalHumanRead(BaseModel):
    digital_human_id: str
    display_name: str
    relationship_type: str
    custom_relationship: str | None
    life_status: str
    age_group: str | None
    gender: str | None
    dialect_code: str | None
    preferred_address: str | None
    creation_mode: str
    creation_status: str
    persona_status: str
    build_status: str
    revision: int
    has_photo: bool
    has_voice: bool
    photo_url: str | None = None
    avatar_url: str | None = None
    voice_ref_duration_ms: int | None = None
    # 音色来源标注：preset=音色库预设，cloned=用户克隆，default=未绑定走系统默认
    voice_kind: str = "default"
    # voice_kind=preset 时的音色库名称（如"慈祥奶奶·老年"），其余为 None
    voice_display_name: str | None = None
    active_package: PackageVersionRead | None = None
    active_background: DigitalHumanBackgroundRead | None = None
    created_at: datetime
    updated_at: datetime


class DigitalHumanDetail(DigitalHumanRead):
    persona_versions: list[PersonaVersionRead] = Field(default_factory=list)
    behavior_anchors: list[BehaviorAnchorRead] = Field(default_factory=list)
    consents: list[ConsentRead] = Field(default_factory=list)
    package_versions: list[PackageVersionRead] = Field(default_factory=list)


class DigitalHumanListResponse(BaseModel):
    items: list[DigitalHumanRead]
    total: int


class DigitalHumanBuildResponse(BaseModel):
    task_id: str
    digital_human_id: str
    avatar_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    error: str | None = None


class DigitalHumanDeleteResponse(BaseModel):
    delete_job_id: str
    digital_human_id: str
    status: str
    execute_after: datetime


class DigitalHumanDeleteJobRead(DigitalHumanDeleteResponse):
    stage: str
    progress: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
