from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AdminDigitalHumanRead(BaseModel):
    digital_human_id: str
    owner_user_id: str
    display_name: str
    relationship_type: str
    life_status: str
    creation_status: str
    build_status: str
    active_package_version_id: str | None
    photo_size_bytes: int
    voice_size_bytes: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


class AdminDigitalHumanListResponse(BaseModel):
    items: list[AdminDigitalHumanRead]
    total: int


class AdminBuildTaskRead(BaseModel):
    task_id: str
    digital_human_id: str
    owner_user_id: str | None
    avatar_id: str
    status: str
    stage: str | None
    progress: int
    error: str | None
    retry_count: int
    max_retries: int
    build_report: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class AdminBuildTaskListResponse(BaseModel):
    items: list[AdminBuildTaskRead]
    total: int


class AdminPackageRead(BaseModel):
    package_version_id: str
    digital_human_id: str
    avatar_id: str
    build_task_id: str | None
    version: int
    status: str
    engine: str
    quality_profile: str | None
    quality_status: str
    quality_score: float | None
    review_note: str | None
    reviewed_by: str | None
    reviewed_at: datetime | None
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
    reference_render_storage_key: str | None = None
    size_bytes: int
    storage_key: str
    sha256: str
    created_at: datetime
    activated_at: datetime | None


class AdminPackageListResponse(BaseModel):
    items: list[AdminPackageRead]
    total: int


class PackageReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    quality_score: float | None = Field(default=None, ge=0, le=100)
    note: str = Field(..., min_length=2, max_length=2000)


class RefundCreateRequest(BaseModel):
    amount_cents: int = Field(..., gt=0)
    reason: str = Field(..., min_length=2, max_length=2000)


class RefundTransitionRequest(BaseModel):
    status: Literal["processing", "succeeded", "failed", "canceled"]
    channel_refund_id: str | None = Field(default=None, max_length=256)
    error_message: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_transition_details(self):
        if self.status == "failed" and not self.error_message:
            raise ValueError("error_message is required for a failed refund")
        return self


class RefundRead(BaseModel):
    refund_id: str
    refund_no: str
    order_id: str
    order_no: str
    user_id: str
    amount_cents: int
    currency: str
    reason: str
    status: str
    payment_channel: str
    channel_refund_id: str | None
    requested_by: str
    error_message: str | None
    requested_at: datetime
    processed_at: datetime | None
    updated_at: datetime


class RefundListResponse(BaseModel):
    items: list[RefundRead]
    total: int


class ReconciliationRead(BaseModel):
    order_count: int
    paid_amount_cents: int
    recorded_refund_amount_cents: int
    succeeded_refund_amount_cents: int
    pending_refund_amount_cents: int
    inconsistent_order_count: int


class SubscriptionSummaryRead(BaseModel):
    total_users: int
    active_subscriptions: int
    grace_or_past_due_subscriptions: int
    auto_renew_subscriptions: int
    cancel_at_period_end_subscriptions: int
    active_by_plan: dict[str, int]


class MrrRead(BaseModel):
    currency: str = "CNY"
    mrr_cents: int
    source_subscription_count: int
    calculated_at: datetime


class ConversionRead(BaseModel):
    registered_users: int
    paying_users: int
    conversion_rate: float


class RetentionRead(BaseModel):
    previous_active_users: int
    retained_users: int
    retention_rate: float
    window_days: int


class UsageCostItem(BaseModel):
    entitlement_key: str
    quantity: int
    unit: str
    unit_cost_cents: float
    estimated_cost_cents: int


class UsageCostRead(BaseModel):
    from_at: datetime
    to_at: datetime
    currency: str = "CNY"
    estimated: bool = True
    items: list[UsageCostItem]
    total_estimated_cost_cents: int


class RenewalFailuresRead(BaseModel):
    past_due_subscriptions: int
    failed_renewal_orders: int
    items: list[dict[str, Any]]


class PlanVersionCohortRead(BaseModel):
    plan_code: str
    plan_version_id: str
    version_no: int
    active_periods: int
    distinct_users: int


class SettingWrite(BaseModel):
    category: str = Field(..., min_length=1, max_length=32)
    value: Any
    is_secret: bool = False
    description: str | None = Field(default=None, max_length=512)


class SettingRead(BaseModel):
    setting_key: str
    category: str
    value: Any
    is_secret: bool
    description: str | None
    updated_by: str | None
    created_at: datetime
    updated_at: datetime


class SettingListResponse(BaseModel):
    items: list[SettingRead]
    total: int


class ContentSafetyCaseCreate(BaseModel):
    source_type: Literal["dialogue", "memory", "digital_human", "asset", "user"]
    source_id: str = Field(..., min_length=1, max_length=128)
    user_id: str | None = Field(default=None, max_length=64)
    category: str = Field(..., min_length=1, max_length=32)
    severity: Literal["low", "medium", "high", "critical"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    assigned_to: str | None = Field(default=None, max_length=64)


class ContentSafetyCaseUpdate(BaseModel):
    status: Literal["open", "reviewing", "resolved", "dismissed"] | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    assigned_to: str | None = Field(default=None, max_length=64)
    resolution: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_resolution(self):
        if self.status in {"resolved", "dismissed"} and not self.resolution:
            raise ValueError("resolution is required when closing a safety case")
        return self


class ContentSafetyCaseRead(BaseModel):
    case_id: str
    source_type: str
    source_id: str
    user_id: str | None
    category: str
    severity: str
    status: str
    metadata: dict[str, Any]
    assigned_to: str | None
    resolution: str | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


class ContentSafetyCaseListResponse(BaseModel):
    items: list[ContentSafetyCaseRead]
    total: int
