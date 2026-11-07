from collections.abc import Generator
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings

# Optional pgvector support. The main API does not import torch/sentence-transformers;
# it only needs the lightweight ``pgvector.sqlalchemy.Vector`` adapter for the ORM
# column declaration. We degrade gracefully when the package is absent so unit
# tests that monkeypatch a sqlite engine keep working — the embedding column is
# declared as ``Text`` placeholder in that case and is never queried directly.
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import-not-found]

    _MemoryVector = Vector(512)
    _HAS_PGVECTOR = True
except Exception:  # pragma: no cover - exercised only in test envs without pgvector
    _MemoryVector = Text  # type: ignore[assignment]
    _HAS_PGVECTOR = False


class Base(DeclarativeBase):
    pass


class KeyframeTaskRecord(Base):
    __tablename__ = "keyframe_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(128), index=True)
    owner_user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    avatar_id: Mapped[str] = mapped_column(String(160), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    package: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Async build pipeline (P0 修复, 2026-07):
    #   stage: queued|preparing|tts|inference|extract|packaging|done
    #   progress: 0..100, updated by worker
    #   started_at/finished_at: lifecycle timing for stale recovery
    #   retry_count/max_retries: bounded retries on transient failures
    #   idempotency_key: client-supplied dedup key (unique partial index)
    #   build_report: per-build failure detail (legacy compositing chain removed in P1)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=2)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    build_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class KeyframePackageRecord(Base):
    __tablename__ = "keyframe_packages"

    avatar_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    engine: Mapped[str] = mapped_column(String(64), default="mobile_keyframe_v3")
    # mobile_keyframe_v3 fields (replaces legacy single-value ``resolution``).
    # See docs/拼帧数字人实时对话与3比4画布完整调整方案.md §8.2.
    package_format_version: Mapped[int] = mapped_column(Integer, default=3)
    inference_width: Mapped[int] = mapped_column(Integer, default=768)
    inference_height: Mapped[int] = mapped_column(Integer, default=768)
    canvas_width: Mapped[int] = mapped_column(Integer, default=768)
    canvas_height: Mapped[int] = mapped_column(Integer, default=1024)
    canvas_profile: Mapped[str] = mapped_column(String(32), default="portrait_3_4")
    renderer_mode: Mapped[str] = mapped_column(String(32), default="base_plus_mouth_roi")
    package_schema_revision: Mapped[int] = mapped_column(Integer, default=2)
    renderer_runtime_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    build_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_asset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reference_render_storage_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    quality_status: Mapped[str] = mapped_column(String(16), default="unreviewed", index=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ClientApiKeyRecord(Base):
    """DB-managed client API keys.

    Admin can CRUD these via /api/v1/admin/api-keys/*. Key plaintext is
    returned exactly once on creation; only sha256(salt + key) is stored.

    Hybrid model: keys from CLIENT_API_KEY env var are ALSO accepted
    (seed into this table on startup, marked ``source='env'``), so
    legacy deployments and DB-managed deployments coexist.
    """

    __tablename__ = "client_api_keys"

    key_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # sha256(salt + plaintext_key) hex digest — 64 chars.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Random 16-byte salt per row, hex-encoded. Prevents rainbow tables
    # even if the DB dumps.
    salt: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(String(256))
    # "env" = seeded from CLIENT_API_KEY env var, "manual" = created via API.
    source: Mapped[str] = mapped_column(String(16), default="manual")
    is_active: Mapped[bool] = mapped_column(default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


# ===========================================================================
# User system (P3 production-grade auth)
# ===========================================================================

class UserRecord(Base):
    """End-user account. Any of phone/email/wechat_openid/username may be NULL,
    but at least one must be set so the user can log back in."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    phone: Mapped[str | None] = mapped_column(String(32), unique=True, index=True, nullable=True)
    username: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    wechat_openid: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    wechat_unionid: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    app_account_token: Mapped[str | None] = mapped_column(
        String(36), unique=True, nullable=True
    )
    nickname: Mapped[str] = mapped_column(String(64), default="用户")
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(String(60), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    role: Mapped[str] = mapped_column(String(16), default="user")
    phone_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class UserSessionRecord(Base):
    """Refresh-token sessions. token_hash = sha256(refresh_token_plaintext)."""

    __tablename__ = "user_sessions"

    jit: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    token_hash: Mapped[str] = mapped_column(String(64))
    device_info: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class SmsCodeRecord(Base):
    """SMS verification codes. code_hash = sha256(phone+code+salt).

    ``salt`` is per-row random hex (16 bytes). Storing it next to the
    hash lets us verify with the correct salt on lookup, and prevents
    rainbow-table attacks even if the DB dumps.
    """

    __tablename__ = "sms_codes"

    code_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    phone: Mapped[str] = mapped_column(String(32), index=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    salt: Mapped[str] = mapped_column(String(32), default="")
    purpose: Mapped[str] = mapped_column(String(32), default="login")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class UserAvatarRecord(Base):
    """Per-user bound avatars. is_primary=True on exactly one row per user.

    Voice-cloning fields (P3 batch 4 — inline reference mode):

    Self-hosted Fish-Speech has **no model registration HTTP API**;
    voice cloning works by passing the reference audio inline per TTS
    call (base64 in the ``reference_audio`` field). So we only need to
    keep the reference audio clip + its transcript on hand — they get
    pulled from MinIO and fed into each ``/v1/tts`` request.

    Fields:
      * ``voice_ref_audio_key``   – MinIO key of the user-uploaded
        reference clip (5–15s of the digital human's voice).
      * ``voice_ref_text``        – transcript of the clip (improves
        cloning quality; passed as ``reference_text`` to Fish-Speech).
      * ``voice_ref_duration_ms`` – clip duration, surfaced to client
        for "5–15s recommended" validation.

    All three are NULL for admin-built packages without a voice sample —
    TTS falls back to the default Fish-Speech voice in that case.

    Future optimization: if a fish-speech version is found that accepts
    ``reference_id=<dir_name>`` pointing to a pre-loaded ``references/``
    folder, add a ``voice_ref_id`` column here and a shared-volume
    docker-compose mount to skip the per-call upload.
    """

    __tablename__ = "user_avatars"

    user_avatar_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    digital_human_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    avatar_id: Mapped[str] = mapped_column(String(160), index=True)
    is_primary: Mapped[bool] = mapped_column(default=False)
    custom_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Voice cloning (nullable — pre-batch-4 rows stay valid)
    voice_ref_audio_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    voice_ref_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    voice_ref_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    voice_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    voice_model_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    voice_provider_voice_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DigitalHumanRecord(Base):
    """Owner-scoped digital human aggregate root."""

    __tablename__ = "digital_humans"

    digital_human_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), index=True)
    display_name: Mapped[str] = mapped_column(String(64))
    relationship_type: Mapped[str] = mapped_column(String(32), default="self")
    custom_relationship: Mapped[str | None] = mapped_column(String(64), nullable=True)
    life_status: Mapped[str] = mapped_column(String(16), default="living")
    age_group: Mapped[str | None] = mapped_column(String(16), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    dialect_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    preferred_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    creation_mode: Mapped[str] = mapped_column(String(32), default="self")
    creation_status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    persona_status: Mapped[str] = mapped_column(String(16), default="draft")
    build_status: Mapped[str] = mapped_column(String(16), default="not_started", index=True)
    active_package_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    photo_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    photo_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    avatar_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    avatar_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    voice_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    voice_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    voice_ref_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    voice_ref_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    voice_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    voice_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    voice_provider_voice_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    retired_voice_assets: Mapped[list] = mapped_column(JSON, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DigitalHumanPersonaVersionRecord(Base):
    __tablename__ = "digital_human_persona_versions"

    persona_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    profile: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    created_by_user_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DigitalHumanBehaviorAnchorRecord(Base):
    __tablename__ = "digital_human_behavior_anchors"

    behavior_anchor_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(32))
    trigger_text: Mapped[str] = mapped_column(Text)
    response_guidance: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    is_enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class DigitalHumanConsentRecord(Base):
    __tablename__ = "digital_human_consents"

    consent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), index=True)
    consent_type: Mapped[str] = mapped_column(String(48), index=True)
    policy_version: Mapped[str] = mapped_column(String(32))
    is_granted: Mapped[bool] = mapped_column(default=True)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    actor_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DigitalHumanPackageVersionRecord(Base):
    __tablename__ = "digital_human_package_versions"

    package_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    avatar_id: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    build_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    engine: Mapped[str] = mapped_column(String(64))
    quality_profile: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # mobile_keyframe_v3 fields (replaces legacy single-value ``resolution``).
    package_format_version: Mapped[int] = mapped_column(Integer, default=3)
    inference_width: Mapped[int] = mapped_column(Integer, default=768)
    inference_height: Mapped[int] = mapped_column(Integer, default=768)
    canvas_width: Mapped[int] = mapped_column(Integer, default=768)
    canvas_height: Mapped[int] = mapped_column(Integer, default=1024)
    canvas_profile: Mapped[str] = mapped_column(String(32), default="portrait_3_4")
    renderer_mode: Mapped[str] = mapped_column(String(32), default="base_plus_mouth_roi")
    package_schema_revision: Mapped[int] = mapped_column(Integer, default=2)
    renderer_runtime_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    build_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_asset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reference_render_storage_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    quality_status: Mapped[str] = mapped_column(String(16), default="unreviewed", index=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DigitalHumanBackgroundRecord(Base):
    __tablename__ = "digital_human_backgrounds"

    digital_human_background_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16))
    background_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DigitalHumanDeleteJobRecord(Base):
    __tablename__ = "digital_human_delete_jobs"

    delete_job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    execute_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stage: Mapped[str] = mapped_column(String(32), default="scheduled")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ===========================================================================
# Membership, subscriptions, and entitlement accounting
# ===========================================================================

class MembershipPlanRecord(Base):
    __tablename__ = "membership_plans"

    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    tier_rank: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    current_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class MembershipPlanVersionRecord(Base):
    __tablename__ = "membership_plan_versions"
    __table_args__ = (
        UniqueConstraint("plan_id", "version_no", name="uq_membership_plan_version_no"),
    )

    plan_version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(String(64), index=True)
    version_no: Mapped[int] = mapped_column(Integer)
    display_name: Mapped[str] = mapped_column(String(64))
    subtitle: Mapped[str | None] = mapped_column(String(256), nullable=True)
    badge: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entitlement_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    entitlements: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    change_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class MembershipPlanPriceRecord(Base):
    __tablename__ = "membership_plan_prices"
    __table_args__ = (
        UniqueConstraint(
            "plan_version_id", "billing_cycle", "channel",
            name="uq_membership_plan_price_scope",
        ),
    )

    price_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_version_id: Mapped[str] = mapped_column(String(64), index=True)
    billing_cycle: Mapped[str] = mapped_column(String(16))
    duration_months: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    amount_cents: Mapped[int] = mapped_column(Integer)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    external_product_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    auto_renew: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class UserSubscriptionRecord(Base):
    __tablename__ = "user_subscriptions"

    subscription_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    payment_channel: Mapped[str] = mapped_column(String(16))
    channel_subscription_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    agreement_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    auto_renew: Mapped[bool] = mapped_column(Boolean, default=False)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False)
    current_period_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_plan_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class SubscriptionPeriodRecord(Base):
    __tablename__ = "subscription_periods"

    period_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subscription_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_version_id: Mapped[str] = mapped_column(String(64), index=True)
    billing_cycle: Mapped[str] = mapped_column(String(16))
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    entitlement_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    source_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class MembershipOrderRecord(Base):
    __tablename__ = "membership_orders"
    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name="uq_membership_order_idempotency"),
    )

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_no: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_version_id: Mapped[str] = mapped_column(String(64), index=True)
    price_id: Mapped[str] = mapped_column(String(64))
    billing_cycle: Mapped[str] = mapped_column(String(16))
    payment_channel: Mapped[str] = mapped_column(String(16), index=True)
    external_product_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    refunded_amount_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    status: Mapped[str] = mapped_column(String(24), default="created", index=True)
    order_type: Mapped[str] = mapped_column(String(16), default="new")
    idempotency_key: Mapped[str] = mapped_column(String(128))
    channel_transaction_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class MembershipRefundRecord(Base):
    __tablename__ = "membership_refunds"

    refund_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    refund_no: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="requested", index=True)
    channel_refund_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    requested_by: Mapped[str] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class PaymentWebhookEventRecord(Base):
    __tablename__ = "payment_webhook_events"
    __table_args__ = (
        UniqueConstraint("channel", "channel_event_id", name="uq_payment_webhook_channel_event"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    channel_event_id: Mapped[str] = mapped_column(String(256))
    event_type: Mapped[str] = mapped_column(String(64))
    signature_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    payload_digest: Mapped[str] = mapped_column(String(64))
    config_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    normalized_payload_ciphertext: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(24), default="received", index=True)
    related_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)


class PaymentChannelConfigRecord(Base):
    __tablename__ = "payment_channel_configs"

    config_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(64))
    environment: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="draft", index=True)
    public_config: Mapped[dict] = mapped_column(JSON, default=dict)
    secret_config: Mapped[str] = mapped_column(Text)
    config_version: Mapped[int] = mapped_column(Integer, default=1)
    fallback_environment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    fallback_public_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fallback_secret_config: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_config_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fallback_valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    validation_status: Mapped[str] = mapped_column(String(16), default="pending")
    validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PaymentChannelTransactionRecord(Base):
    __tablename__ = "payment_channel_transactions"
    __table_args__ = (
        UniqueConstraint(
            "channel",
            "channel_transaction_id",
            name="uq_payment_channel_transaction",
        ),
    )

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    channel_transaction_id: Mapped[str] = mapped_column(String(256))
    original_transaction_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True, index=True
    )
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    external_product_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    amount_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    purchased_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    raw_digest: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PaymentReconciliationRunRecord(Base):
    __tablename__ = "payment_reconciliation_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    from_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    to_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    checked_count: Mapped[int] = mapped_column(Integer, default=0)
    mismatch_count: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    requested_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EntitlementUsageLedgerRecord(Base):
    __tablename__ = "entitlement_usage_ledger"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "entitlement_key", "request_id",
            name="uq_entitlement_usage_request",
        ),
    )

    usage_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    period_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    entitlement_key: Mapped[str] = mapped_column(String(64), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    confirmed_quantity: Mapped[int] = mapped_column(Integer, default=0)
    unit: Mapped[str] = mapped_column(String(24))
    request_id: Mapped[str] = mapped_column(String(128))
    resource_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="reserved", index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    details: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


class EntitlementUsageCounterRecord(Base):
    __tablename__ = "entitlement_usage_counters"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "entitlement_key", "window_start",
            name="uq_entitlement_usage_counter_window",
        ),
    )

    counter_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    period_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    entitlement_key: Mapped[str] = mapped_column(String(64), index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reserved_quantity: Mapped[int] = mapped_column(Integer, default=0)
    confirmed_quantity: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class UserQuotaRecord(Base):
    """Per-user daily quota. Lazy reset on each request when reset_at < now."""

    __tablename__ = "user_quotas"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    quota_tts_daily: Mapped[int] = mapped_column(Integer, default=100)
    quota_build_daily: Mapped[int] = mapped_column(Integer, default=5)
    quota_download_daily: Mapped[int] = mapped_column(Integer, default=20)
    used_tts_today: Mapped[int] = mapped_column(Integer, default=0)
    used_build_today: Mapped[int] = mapped_column(Integer, default=0)
    used_download_today: Mapped[int] = mapped_column(Integer, default=0)
    reset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class AuditLogRecord(Base):
    """Audit trail. Written async by audit_service; queried by admin."""

    __tablename__ = "audit_logs"

    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    actor_admin_api_key_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    actor_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


class SystemSettingRecord(Base):
    __tablename__ = "system_settings"

    setting_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class ContentSafetyCaseRecord(Base):
    __tablename__ = "content_safety_cases"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    details: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ===========================================================================
# Dialogue + LLM (P3 first batch)
# ===========================================================================

class LlmProviderRecord(Base):
    __tablename__ = "llm_providers"
    provider_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(32))  # glm|doubao|qwen|openai
    # Fernet ciphertext prefixed with ``enc:v1:``. Legacy plaintext rows are
    # read for migration compatibility but every write is encrypted.
    api_key: Mapped[str] = mapped_column(Text)
    base_url: Mapped[str] = mapped_column(String(512))
    model: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(default=False, index=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    temperature: Mapped[float] = mapped_column(default=0.7)
    max_tokens: Mapped[int] = mapped_column(default=1024)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class TtsProviderRecord(Base):
    __tablename__ = "tts_providers"

    provider_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(32))
    api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    workspace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    default_voice_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class DialogueSessionRecord(Base):
    __tablename__ = "dialogue_sessions"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(128), nullable=True)
    llm_provider_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    avatar_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    # Cross-session memory architecture (migration 0018): the canonical
    # relationship key is (user_id, digital_human_id). The new column is
    # NOT NULL on insert and validated against ``digital_humans`` via a
    # composite FK. ``avatar_id`` is kept for legacy storage/SDK paths but
    # is no longer the source of truth for memory namespace.
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_message_cursor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class DialogueMessageRecord(Base):
    __tablename__ = "dialogue_messages"
    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    audio_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class AvatarMemoryRecord(Base):
    __tablename__ = "avatar_memories"

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    source_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list)
    category: Mapped[str] = mapped_column(String(32), index=True)
    summary: Mapped[str] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    sensitivity: Mapped[str] = mapped_column(String(16), default="normal", index=True)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_recalled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recovery_delete_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class MemoryExtractionJobRecord(Base):
    """Legacy queue table. Retained for migration compatibility; new code
    writes to ``MemoryJobRecord`` via ``memory_jobs``.
    """
    __tablename__ = "memory_extraction_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    trigger_message_id: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


# ===========================================================================
# Relationship-scoped long-term memory (migration 0018).
#
# The classes below are the new source of truth. Legacy ``AvatarMemoryRecord``
# and ``MemoryExtractionJobRecord`` are kept above only so old branches of
# admin tooling and tests keep importing; the running service never writes to
# them once migration 0018 has been applied.
# ===========================================================================


class MemoryItemRecord(Base):
    """Long-term memory scoped to ``(user_id, digital_human_id)``.

    There is exactly one legal memory namespace. The previous ``scope`` field
    from the architectural draft is intentionally absent — cross-digital-human
    sharing is forbidden by design.

    Sensitive rows:
      * ``summary`` and ``search_text`` are NULL.
      * ``embedding`` is NULL.
      * ``summary_ciphertext`` carries Fernet ciphertext (``mem:vN:...``).
      * These rows are NEVER indexed for retrieval and NEVER auto-recalled.
    """

    __tablename__ = "memory_items"

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(24), nullable=False)
    category: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    fact_key: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_ciphertext: Mapped[str | None] = mapped_column(Text, nullable=True)
    encryption_key_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.65, nullable=False)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(16), default="normal", index=True, nullable=False)
    tags: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_recalled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by_memory_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_message_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding = mapped_column(_MemoryVector, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    recovery_delete_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class MemoryJobRecord(Base):
    """Durable memory-worker queue.

    A job is always scoped to a relationship key. ``idempotency_key`` is
    globally unique so re-enqueueing the same work (e.g. after crash) never
    produces a duplicate successful write.
    """

    __tablename__ = "memory_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    trigger_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail_redacted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class MemoryRelationshipProfileRecord(Base):
    """Stable summary of the (user, digital_human) relationship.

    Only one row per relationship key. ``profile_text`` is rebuilt by the
    worker from active ``memory_items`` summaries and never contains
    sensitive ciphertext or credentials.
    """

    __tablename__ = "memory_relationship_profiles"

    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    profile_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_override_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile_version: Mapped[int] = mapped_column(Integer, default=0)
    source_memory_ids: Mapped[list] = mapped_column(JSON, default=list)
    last_built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class LlmProviderBindingRecord(Base):
    """Role-based LLM provider binding.

    chat: conversational calls.
    memory_extract: structured JSON memory extractor (low temp, schema probe).
    memory_rerank: optional reranker (not yet used; reserved).

    Even when chat and memory_extract point at the same physical provider,
    both rows must exist explicitly. ``memory_extract`` never silently falls
    back to ``chat``.
    """

    __tablename__ = "llm_provider_bindings"

    binding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(24), unique=True, nullable=False)
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config_overrides: Mapped[dict] = mapped_column(JSON, default=dict)
    probe_status: Mapped[str] = mapped_column(String(16), default="pending")
    probe_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    probed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class EmbeddingProviderConfigRecord(Base):
    """Embedding service contract. The memory-embedding container must report
    a model_name/revision/dim triple that matches an active row, or
    ``embedding_client`` fails closed."""

    __tablename__ = "embedding_provider_configs"

    config_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(512))
    model_name: Mapped[str] = mapped_column(String(128))
    revision: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class MemoryInjectionLogRecord(Base):
    __tablename__ = "memory_injection_logs"

    injection_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    digital_human_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    query_hash: Mapped[str] = mapped_column(String(64))
    memory_ids: Mapped[list] = mapped_column(JSON, default=list)
    injected_count: Mapped[int] = mapped_column(Integer, default=0)
    policy_plan_code: Mapped[str] = mapped_column(String(32))
    retrieval_mode: Mapped[str | None] = mapped_column(String(24), nullable=True)
    candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_scores: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)


class NotificationRecord(Base):
    """User-facing notifications (站内信). No APNs/FCM — polled by client."""
    __tablename__ = "notifications"
    notification_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(32))  # avatar_build_completed|system
    title: Mapped[str] = mapped_column(String(128))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_read: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BackgroundRecord(Base):
    __tablename__ = "backgrounds"
    background_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16), default="preset")  # preset|user
    storage_key: Mapped[str] = mapped_column(String(512))
    uploaded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
