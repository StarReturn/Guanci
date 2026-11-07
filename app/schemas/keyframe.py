from datetime import datetime

from typing import Any

from pydantic import BaseModel


class KeyframePackageRead(BaseModel):
    avatar_id: str
    digital_human_id: str
    version: int
    engine: str = "mobile_keyframe_v3"
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
    download_url: str
    sha256: str
    quality_status: str = "unreviewed"


class KeyframeTaskRead(BaseModel):
    task_id: str
    digital_human_id: str
    owner_user_id: str | None = None
    avatar_id: str
    status: str
    error: str | None = None
    package: KeyframePackageRead | None = None
    # 异步构建相关（P0 修复, 2026-07）。向后兼容，老任务这些字段为 None/0。
    stage: str | None = None
    progress: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 2
    idempotency_key: str | None = None
    build_report: dict | None = None


class KeyframeBuildResponse(BaseModel):
    task_id: str
    status: str
    avatar_id: str
