"""Owner-scoped digital human creation and lifecycle APIs."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.config import settings
from app.core.database import SessionLocal

from app.middleware.jwt import get_request_user_id
from app.schemas.digital_human import (
    BehaviorAnchorRead,
    BehaviorAnchorWrite,
    ConsentRead,
    ConsentWrite,
    DigitalHumanBackgroundRead,
    DigitalHumanBuildResponse,
    DigitalHumanCreate,
    DigitalHumanDeleteJobRead,
    DigitalHumanDeleteResponse,
    DigitalHumanDetail,
    DigitalHumanListResponse,
    DigitalHumanRead,
    DigitalHumanUpdate,
    PackageVersionRead,
    PersonaVersionCreate,
    PersonaVersionRead,
)
from app.services.aliyun_cosyvoice_client import AliyunCosyVoiceError, aliyun_cosyvoice_client
from app.services.auth_service import audit_service
from app.services.digital_human_service import (
    DigitalHumanConflict,
    DigitalHumanConsentError,
    DigitalHumanNotFound,
    DigitalHumanStateError,
    digital_human_service,
)
from app.services.keyframe_builder import keyframe_builder
from app.services.keyframe_repository import keyframe_repository
from app.services.entitlement_stub import (
    EntitlementDenied,
    MembershipValidationError,
    entitlement_service,
)
from app.services.notification_service import background_service
from app.services.storage import storage_service
from app.services.tts_provider_service import tts_provider_service

router = APIRouter()
logger = logging.getLogger(__name__)


class PresetVoiceSelection(BaseModel):
    voice_id: str | None = Field(default=None, min_length=1, max_length=128)


def _user_id(request: Request) -> str:
    user_id = get_request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="user login required")
    return user_id


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def _raise_domain_error(exc: Exception) -> None:
    if isinstance(exc, DigitalHumanNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, DigitalHumanConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (DigitalHumanStateError, DigitalHumanConsentError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, EntitlementDenied):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "ENTITLEMENT_LIMIT_EXCEEDED",
                "entitlement_key": exc.key,
                "limit": exc.limit,
                "used": exc.used,
                "requested": exc.requested,
            },
        ) from exc
    if isinstance(exc, MembershipValidationError):
        raise HTTPException(status_code=422, detail=exc.errors) from exc
    raise exc


def _consent_read(row) -> ConsentRead:
    return ConsentRead(
        consent_id=row.consent_id,
        consent_type=row.consent_type,
        policy_version=row.policy_version,
        is_granted=row.is_granted,
        granted_at=row.granted_at,
        revoked_at=row.revoked_at,
        created_at=row.created_at,
    )


def _persona_read(row) -> PersonaVersionRead:
    return PersonaVersionRead(
        persona_version_id=row.persona_version_id,
        version=row.version,
        profile=row.profile,
        summary=row.summary,
        status=row.status,
        created_at=row.created_at,
    )


def _anchor_read(row) -> BehaviorAnchorRead:
    return BehaviorAnchorRead(
        behavior_anchor_id=row.behavior_anchor_id,
        category=row.category,
        trigger_text=row.trigger_text,
        response_guidance=row.response_guidance,
        priority=row.priority,
        is_enabled=row.is_enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _package_read(row, *, allow_download: bool = True) -> PackageVersionRead:
    return PackageVersionRead(
        package_version_id=row.package_version_id,
        avatar_id=row.avatar_id,
        build_task_id=row.build_task_id,
        version=row.version,
        status=row.status,
        engine=row.engine,
        quality_profile=row.quality_profile,
        package_format_version=row.package_format_version,
        inference_width=row.inference_width,
        inference_height=row.inference_height,
        canvas_width=row.canvas_width,
        canvas_height=row.canvas_height,
        canvas_profile=row.canvas_profile,
        renderer_mode=row.renderer_mode,
        package_schema_revision=row.package_schema_revision,
        renderer_runtime_version=row.renderer_runtime_version,
        build_fingerprint=row.build_fingerprint,
        source_asset_sha256=row.source_asset_sha256,
        manifest_metadata=row.manifest_metadata,
        validation_report=row.validation_report,
        reference_render_url=(
            storage_service.presigned_url(row.reference_render_storage_key)
            if row.reference_render_storage_key
            else None
        ),
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        download_url=(
            storage_service.presigned_url(row.storage_key)
            if allow_download
            else None
        ),
        created_at=row.created_at,
        activated_at=row.activated_at,
    )


def _background_read(row) -> DigitalHumanBackgroundRead:
    return DigitalHumanBackgroundRead(
        digital_human_background_id=row.digital_human_background_id,
        source=row.source,
        background_id=row.background_id,
        preview_url=storage_service.presigned_url(row.storage_key),
        is_active=row.is_active,
        created_at=row.created_at,
    )


def _voice_kind_and_name(row) -> tuple[str, str | None]:
    """preset/cloned/default + 预设音色名（一次点查，列表页 N 次索引查询可接受）。"""
    vid = row.voice_provider_voice_id
    if not vid:
        return "default", None
    if digital_human_service.is_preset_voice_id(vid):
        with SessionLocal() as db:
            (name,) = db.execute(text(
                "select display_name from preset_voices where voice_id = :v"
            ), {"v": vid}).fetchone() or (None,)
        return "preset", name
    return "cloned", None


def _digital_human_read(row) -> DigitalHumanRead:
    active_package = digital_human_service.get_active_package(row.digital_human_id)
    active_background = digital_human_service.get_active_background(row.digital_human_id)
    voice_kind, voice_display_name = _voice_kind_and_name(row)
    return DigitalHumanRead(
        digital_human_id=row.digital_human_id,
        display_name=row.display_name,
        relationship_type=row.relationship_type,
        custom_relationship=row.custom_relationship,
        life_status=row.life_status,
        age_group=row.age_group,
        gender=row.gender,
        dialect_code=row.dialect_code,
        preferred_address=row.preferred_address,
        creation_mode=row.creation_mode,
        creation_status=row.creation_status,
        persona_status=row.persona_status,
        build_status=row.build_status,
        revision=row.revision,
        has_photo=bool(row.photo_storage_key),
        has_voice=bool(row.voice_storage_key or row.voice_provider_voice_id),
        photo_url=(
            storage_service.presigned_url(row.photo_storage_key)
            if row.photo_storage_key
            else None
        ),
        avatar_url=(
            storage_service.presigned_url(row.avatar_storage_key)
            if row.avatar_storage_key
            else None
        ),
        voice_ref_duration_ms=row.voice_ref_duration_ms,
        voice_kind=voice_kind,
        voice_display_name=voice_display_name,
        active_package=_package_read(active_package) if active_package else None,
        active_background=_background_read(active_background) if active_background else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _digital_human_detail(row) -> DigitalHumanDetail:
    base = _digital_human_read(row)
    return DigitalHumanDetail(
        **base.model_dump(),
        persona_versions=[
            _persona_read(item)
            for item in digital_human_service.list_persona_versions(row.digital_human_id)
        ],
        behavior_anchors=[
            _anchor_read(item)
            for item in digital_human_service.list_behavior_anchors(row.digital_human_id)
        ],
        consents=[
            _consent_read(item)
            for item in digital_human_service.list_consents(row.digital_human_id)
        ],
        package_versions=[
            _package_read(item, allow_download=row.creation_status == "ready")
            for item in digital_human_service.list_package_versions(row.digital_human_id)
        ],
    )


@router.post("", response_model=DigitalHumanRead, status_code=201)
async def create_digital_human(body: DigitalHumanCreate, request: Request) -> DigitalHumanRead:
    user_id = _user_id(request)
    try:
        entitlement_service.check_limit(user_id, "active_avatar_count", 1)
    except Exception as exc:
        _raise_domain_error(exc)
    row = digital_human_service.create(owner_user_id=user_id, **body.model_dump())
    audit_service.log(
        action="digital_human.create",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=row.digital_human_id,
        detail={"creation_mode": row.creation_mode, "life_status": row.life_status},
    )
    return _digital_human_read(row)


@router.get("", response_model=DigitalHumanListResponse)
async def list_digital_humans(request: Request) -> DigitalHumanListResponse:
    rows = digital_human_service.list_owned(_user_id(request))
    return DigitalHumanListResponse(
        items=[_digital_human_read(row) for row in rows], total=len(rows)
    )


@router.get("/delete-jobs/{delete_job_id}", response_model=DigitalHumanDeleteJobRead)
async def get_delete_job(delete_job_id: str, request: Request) -> DigitalHumanDeleteJobRead:
    try:
        row = digital_human_service.get_delete_job(_user_id(request), delete_job_id)
    except Exception as exc:
        _raise_domain_error(exc)
    return DigitalHumanDeleteJobRead(
        delete_job_id=row.delete_job_id,
        digital_human_id=row.digital_human_id,
        status=row.status,
        execute_after=row.execute_after,
        stage=row.stage,
        progress=row.progress,
        error=row.error,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


@router.get("/{digital_human_id}", response_model=DigitalHumanDetail)
async def get_digital_human(digital_human_id: str, request: Request) -> DigitalHumanDetail:
    try:
        row = digital_human_service.get_owned(_user_id(request), digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    return _digital_human_detail(row)


@router.patch("/{digital_human_id}", response_model=DigitalHumanRead)
async def update_digital_human(
    digital_human_id: str, body: DigitalHumanUpdate, request: Request
) -> DigitalHumanRead:
    user_id = _user_id(request)
    values = body.model_dump(exclude={"expected_revision"}, exclude_unset=True)
    try:
        row = digital_human_service.update_metadata(
            owner_user_id=user_id,
            digital_human_id=digital_human_id,
            expected_revision=body.expected_revision,
            values=values,
        )
    except Exception as exc:
        _raise_domain_error(exc)
    audit_service.log(
        action="digital_human.update",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"fields": sorted(values)},
    )
    return _digital_human_read(row)


@router.post("/{digital_human_id}/avatar", response_model=DigitalHumanRead)
async def upload_avatar(
    digital_human_id: str,
    request: Request,
    avatar_file: UploadFile = File(...),
) -> DigitalHumanRead:
    """上传/替换数字人头像：multipart 直传，服务端转存 MinIO 并签发预签名 URL。"""
    user_id = _user_id(request)
    content_type = avatar_file.content_type or "image/jpeg"
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported avatar image type")
    content = await avatar_file.read()
    if not content or len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="avatar must be 1 byte to 5 MB")
    try:
        row = digital_human_service.set_avatar(
            owner_user_id=user_id,
            digital_human_id=digital_human_id,
            content=content,
            content_type=content_type,
        )
    except Exception as exc:
        _raise_domain_error(exc)
    audit_service.log(
        action="digital_human.avatar_set",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"size_bytes": len(content), "content_type": content_type},
    )
    return _digital_human_read(row)


@router.delete("/{digital_human_id}/avatar", response_model=DigitalHumanRead)
async def delete_avatar(digital_human_id: str, request: Request) -> DigitalHumanRead:
    user_id = _user_id(request)
    try:
        row = digital_human_service.clear_avatar(
            owner_user_id=user_id, digital_human_id=digital_human_id
        )
    except Exception as exc:
        _raise_domain_error(exc)
    audit_service.log(
        action="digital_human.avatar_cleared",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
    )
    return _digital_human_read(row)


@router.delete("/{digital_human_id}", response_model=DigitalHumanDeleteResponse)
async def delete_digital_human(
    digital_human_id: str, request: Request
) -> DigitalHumanDeleteResponse:
    user_id = _user_id(request)
    try:
        job = digital_human_service.schedule_delete(
            owner_user_id=user_id, digital_human_id=digital_human_id
        )
    except Exception as exc:
        _raise_domain_error(exc)
    try:
        from app.worker.queue import enqueue_delete

        enqueue_delete(job.delete_job_id, job.execute_after)
    except Exception:
        logger.exception(
            "delete_enqueue_failed delete_job_id=%s; worker recovery will retry",
            job.delete_job_id,
        )
    audit_service.log(
        action="digital_human.delete_scheduled",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"delete_job_id": job.delete_job_id, "execute_after": job.execute_after.isoformat()},
    )
    return DigitalHumanDeleteResponse(
        delete_job_id=job.delete_job_id,
        digital_human_id=job.digital_human_id,
        status=job.status,
        execute_after=job.execute_after,
    )


@router.post("/{digital_human_id}/consents", response_model=ConsentRead, status_code=201)
async def write_consent(
    digital_human_id: str, body: ConsentWrite, request: Request
) -> ConsentRead:
    try:
        row = digital_human_service.add_consent(
            owner_user_id=_user_id(request),
            digital_human_id=digital_human_id,
            actor_ip=_client_ip(request),
            **body.model_dump(),
        )
    except Exception as exc:
        _raise_domain_error(exc)
    audit_service.log(
        action="digital_human.consent",
        actor_user_id=row.owner_user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"consent_type": row.consent_type, "is_granted": row.is_granted},
    )
    return _consent_read(row)


@router.get("/{digital_human_id}/consents", response_model=list[ConsentRead])
async def list_consents(digital_human_id: str, request: Request) -> list[ConsentRead]:
    try:
        digital_human_service.get_owned(_user_id(request), digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    return [_consent_read(row) for row in digital_human_service.list_consents(digital_human_id)]


@router.post("/{digital_human_id}/persona-versions", response_model=PersonaVersionRead, status_code=201)
async def create_persona_version(
    digital_human_id: str, body: PersonaVersionCreate, request: Request
) -> PersonaVersionRead:
    try:
        row = digital_human_service.create_persona_version(
            owner_user_id=_user_id(request), digital_human_id=digital_human_id,
            **body.model_dump(),
        )
    except Exception as exc:
        _raise_domain_error(exc)
    return _persona_read(row)


@router.get("/{digital_human_id}/persona-versions", response_model=list[PersonaVersionRead])
async def list_persona_versions(
    digital_human_id: str, request: Request
) -> list[PersonaVersionRead]:
    try:
        digital_human_service.get_owned(_user_id(request), digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    return [
        _persona_read(row)
        for row in digital_human_service.list_persona_versions(digital_human_id)
    ]


@router.put("/{digital_human_id}/behavior-anchors", response_model=list[BehaviorAnchorRead])
async def replace_behavior_anchors(
    digital_human_id: str, body: list[BehaviorAnchorWrite], request: Request
) -> list[BehaviorAnchorRead]:
    if len(body) > 100:
        raise HTTPException(status_code=400, detail="at most 100 behavior anchors are allowed")
    try:
        rows = digital_human_service.replace_behavior_anchors(
            owner_user_id=_user_id(request),
            digital_human_id=digital_human_id,
            anchors=[item.model_dump() for item in body],
        )
    except Exception as exc:
        _raise_domain_error(exc)
    return [_anchor_read(row) for row in rows]


@router.get("/{digital_human_id}/behavior-anchors", response_model=list[BehaviorAnchorRead])
async def list_behavior_anchors(
    digital_human_id: str, request: Request
) -> list[BehaviorAnchorRead]:
    try:
        digital_human_service.get_owned(_user_id(request), digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    return [
        _anchor_read(row)
        for row in digital_human_service.list_behavior_anchors(digital_human_id)
    ]


@router.post("/{digital_human_id}/preset-voice", response_model=DigitalHumanRead)
async def select_preset_voice(
    digital_human_id: str, body: PresetVoiceSelection, request: Request
) -> DigitalHumanRead:
    user_id = _user_id(request)
    runtime = tts_provider_service.runtime()
    if body.voice_id is not None and runtime.type != "aliyun_cosyvoice":
        raise HTTPException(status_code=409, detail="preset voices require Aliyun CosyVoice")
    if body.voice_id is not None:
        with SessionLocal() as db:
            exists = db.execute(text(
                "select 1 from preset_voices where voice_id = :voice_id and is_active = true"
            ), {"voice_id": body.voice_id}).scalar_one_or_none()
        if exists is None:
            raise HTTPException(status_code=404, detail="preset voice not found")
    try:
        previous = digital_human_service.get_owned(user_id, digital_human_id)
        if body.voice_id is None:
            row = digital_human_service.clear_voice_choice(
                owner_user_id=user_id, digital_human_id=digital_human_id)
        else:
            row = digital_human_service.bind_preset_voice(
                owner_user_id=user_id,
                digital_human_id=digital_human_id,
                provider_voice_id=body.voice_id,
                model_id=runtime.model or settings.aliyun_cosyvoice_model,
            )
    except Exception as exc:
        _raise_domain_error(exc)
    if previous.voice_storage_key:
        try:
            storage_service.delete_key(previous.voice_storage_key)
        except Exception:
            logger.exception("old_voice_storage_cleanup_failed digital_human_id=%s", digital_human_id)
    old_cloud_voice = previous.voice_provider_voice_id
    if (old_cloud_voice and old_cloud_voice != body.voice_id
            and previous.voice_provider == "aliyun_cosyvoice"
            and not digital_human_service.is_preset_voice_id(old_cloud_voice)):
        try:
            await asyncio.to_thread(aliyun_cosyvoice_client.delete_voice_sync, old_cloud_voice)
            digital_human_service.forget_retired_voice(
                owner_user_id=user_id, digital_human_id=digital_human_id,
                provider_voice_id=old_cloud_voice)
        except Exception:
            logger.exception("old_voice_cloud_cleanup_failed digital_human_id=%s", digital_human_id)
    audit_service.log(
        action="digital_human.preset_voice_selected" if body.voice_id else "digital_human.voice_reset_default",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"voice_id": body.voice_id},
    )
    return _digital_human_read(row)


@router.post("/{digital_human_id}/voice", response_model=DigitalHumanRead)
async def upload_voice(
    digital_human_id: str,
    request: Request,
    voice_audio: UploadFile | None = File(default=None),
    voice_id: str | None = Form(default=None),
    voice_text: str | None = Form(default=None),
) -> DigitalHumanRead:
    user_id = _user_id(request)
    # 协议统一入口：带 voice_id = 绑定音色库预设（无需音频、无需克隆授权）；
    # 带音频 = 克隆用户声音（原行为）。两者互斥，都不带 = 参数缺失。
    if voice_id is not None:
        if voice_audio is not None:
            raise HTTPException(
                status_code=400, detail="voice_id and voice_audio are mutually exclusive"
            )
        runtime = tts_provider_service.runtime()
        if runtime.type != "aliyun_cosyvoice":
            raise HTTPException(status_code=409, detail="preset voices require Aliyun CosyVoice")
        with SessionLocal() as db:
            exists = db.execute(text(
                "select 1 from preset_voices where voice_id = :voice_id and is_active = true"
            ), {"voice_id": voice_id}).scalar_one_or_none()
        if exists is None:
            raise HTTPException(status_code=404, detail="preset voice not found")
        try:
            row = digital_human_service.bind_preset_voice(
                owner_user_id=user_id,
                digital_human_id=digital_human_id,
                provider_voice_id=voice_id,
                model_id=runtime.model,
            )
        except Exception as exc:
            _raise_domain_error(exc)
        return _digital_human_read(row)
    if voice_audio is None:
        raise HTTPException(status_code=422, detail="voice_audio or voice_id is required")
    try:
        current = digital_human_service.get_owned(user_id, digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    if not digital_human_service.has_active_consent(digital_human_id, "voice_processing"):
        raise HTTPException(status_code=422, detail="voice_processing consent is required")
    from app.services.audio_utils import (
        normalize_audio_to_wav, wav_duration_ms,
        analyze_voice_quality, enhance_voice_wav,
    )

    source = await voice_audio.read()
    if not source:
        raise HTTPException(status_code=400, detail="voice audio is empty")
    try:
        wav = normalize_audio_to_wav(
            source, voice_audio.content_type or "audio/wav", default_sample_rate=16000
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"voice audio decode failed: {exc}") from exc
    # 质量门禁（2026-11）：克隆忠实复刻参考缺陷，削波/低信噪比/电话级
    # 带宽的录音在入口拒绝并给出可操作的重录提示；达标样本自动增强
    # （高通去隆隆 + 峰值归一 0.8）后再进克隆，从源头杜绝低品音色。
    qa = analyze_voice_quality(wav)
    if qa.get("verdict") != "ok":
        raise HTTPException(status_code=422, detail=qa.get("reason", "录音质量不达标，请重录"))
    try:
        wav = enhance_voice_wav(wav)
    except Exception:
        pass  # 增强失败退回原件（体检已过，质量本身有保障）
    duration_ms = wav_duration_ms(wav)
    runtime = tts_provider_service.runtime()
    minimum_ms = 5000 if runtime.type == "aliyun_cosyvoice" else 3000
    if duration_ms < minimum_ms or duration_ms > 60000:
        raise HTTPException(
            status_code=400,
            detail=f"voice duration must be between {minimum_ms}ms and 60000ms",
        )
    transcript = (voice_text or "").strip() or None
    if runtime.type != "aliyun_cosyvoice" and transcript is None:
        raise HTTPException(status_code=400, detail="voice_text is required for this provider")
    try:
        entitlement_service.check_limit(
            user_id, "storage_bytes", max(0, len(wav) - current.voice_size_bytes)
        )
    except Exception as exc:
        _raise_domain_error(exc)
    key = f"digital_humans/{user_id}/{digital_human_id}/source/voice/{uuid4()}.wav"
    storage_service.put_bytes(key=key, content=wav, content_type="audio/wav")
    provider_voice_id = runtime.default_voice_id
    if runtime.type == "aliyun_cosyvoice":
        try:
            provider_voice_id = await aliyun_cosyvoice_client.create_voice(
                audio_url=storage_service.presigned_url(key),
                user_id=user_id,
                duration_ms=duration_ms,
            )
        except AliyunCosyVoiceError as exc:
            storage_service.delete_key(key)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        row = digital_human_service.attach_voice(
            owner_user_id=user_id,
            digital_human_id=digital_human_id,
            storage_key=key,
            size_bytes=len(wav),
            transcript=transcript,
            duration_ms=duration_ms,
            provider=runtime.type,
            model_id=runtime.model,
            provider_voice_id=provider_voice_id,
        )
    except Exception as exc:
        storage_service.delete_key(key)
        if runtime.type == "aliyun_cosyvoice" and provider_voice_id:
            try:
                await asyncio.to_thread(
                    aliyun_cosyvoice_client.delete_voice_sync, provider_voice_id
                )
            except Exception:
                logger.exception(
                    "new_voice_cleanup_failed digital_human_id=%s", digital_human_id
                )
        _raise_domain_error(exc)
    if current.voice_storage_key and current.voice_storage_key != key:
        storage_service.delete_key(current.voice_storage_key)
    old_cloud_voice = (
        current.voice_provider_voice_id
        if current.voice_provider == "aliyun_cosyvoice"
        else None
    )
    if (old_cloud_voice and old_cloud_voice != provider_voice_id
            and not digital_human_service.is_preset_voice_id(old_cloud_voice)):
        try:
            await asyncio.to_thread(
                aliyun_cosyvoice_client.delete_voice_sync, old_cloud_voice
            )
            digital_human_service.forget_retired_voice(
                owner_user_id=user_id,
                digital_human_id=digital_human_id,
                provider_voice_id=old_cloud_voice,
            )
        except Exception:
            logger.exception(
                "old_voice_cleanup_failed digital_human_id=%s", digital_human_id
            )
    audit_service.log(
        action="digital_human.voice_replaced",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"provider": runtime.type, "duration_ms": duration_ms},
    )
    return _digital_human_read(row)


@router.post("/{digital_human_id}/background", response_model=DigitalHumanBackgroundRead)
async def set_background(
    digital_human_id: str,
    request: Request,
    background_id: str | None = Form(default=None),
    background_file: UploadFile | None = File(default=None),
) -> DigitalHumanBackgroundRead:
    user_id = _user_id(request)
    try:
        digital_human_service.get_owned(user_id, digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    if bool(background_id) == bool(background_file):
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of background_id or background_file",
        )
    previous_background = digital_human_service.get_active_background(digital_human_id)
    if background_id:
        source_row = background_service.get(background_id)
        if source_row is None or (
            source_row.source != "preset" and source_row.uploaded_by != user_id
        ):
            raise HTTPException(status_code=404, detail="background not found")
        source = source_row.source
        storage_key = source_row.storage_key
        size_bytes = 0
    else:
        content_type = background_file.content_type or "image/jpeg"
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=400, detail="unsupported background image type")
        content = await background_file.read()
        if not content or len(content) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="background must be 1 byte to 15 MB")
        current_background = previous_background
        count_delta = (
            0 if current_background is not None and current_background.source == "user" else 1
        )
        old_size = (
            current_background.size_bytes
            if current_background is not None and current_background.source == "user"
            else 0
        )
        try:
            entitlement_service.check_limit(
                user_id, "background_upload_count", count_delta
            )
            entitlement_service.check_limit(
                user_id, "storage_bytes", max(0, len(content) - old_size)
            )
        except Exception as exc:
            _raise_domain_error(exc)
        extension = "png" if "png" in content_type else "webp" if "webp" in content_type else "jpg"
        storage_key = (
            f"digital_humans/{user_id}/{digital_human_id}/backgrounds/{uuid4()}.{extension}"
        )
        storage_service.put_bytes(key=storage_key, content=content, content_type=content_type)
        source = "user"
        size_bytes = len(content)
    try:
        row = digital_human_service.set_background(
            owner_user_id=user_id,
            digital_human_id=digital_human_id,
            source=source,
            storage_key=storage_key,
            size_bytes=size_bytes,
            background_id=background_id,
        )
    except Exception as exc:
        if not background_id:
            storage_service.delete_key(storage_key)
        _raise_domain_error(exc)
    if (
        previous_background is not None
        and previous_background.source == "user"
        and previous_background.storage_key != storage_key
    ):
        storage_service.delete_key(previous_background.storage_key)
    audit_service.log(
        action="digital_human.background_changed",
        actor_user_id=user_id,
        actor_ip=_client_ip(request),
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"source": source, "background_id": background_id},
    )
    return _background_read(row)


@router.get("/{digital_human_id}/builds/{task_id}", response_model=DigitalHumanBuildResponse)
async def get_build(
    digital_human_id: str, task_id: str, request: Request
) -> DigitalHumanBuildResponse:
    user_id = _user_id(request)
    try:
        digital_human_service.get_owned(user_id, digital_human_id)
    except Exception as exc:
        _raise_domain_error(exc)
    task = keyframe_repository.get_task(task_id)
    if (
        task is None
        or task.digital_human_id != digital_human_id
        or task.owner_user_id != user_id
    ):
        raise HTTPException(status_code=404, detail="build task not found")
    return DigitalHumanBuildResponse(
        task_id=task.task_id,
        digital_human_id=digital_human_id,
        avatar_id=task.avatar_id,
        status=task.status,
        stage=task.stage,
        progress=task.progress,
        error=task.error,
    )


@router.get("/{digital_human_id}/package-versions", response_model=list[PackageVersionRead])
async def list_package_versions(
    digital_human_id: str, request: Request
) -> list[PackageVersionRead]:
    try:
        digital_human = digital_human_service.get_owned(
            _user_id(request), digital_human_id
        )
    except Exception as exc:
        _raise_domain_error(exc)
    return [
        _package_read(row, allow_download=digital_human.creation_status == "ready")
        for row in digital_human_service.list_package_versions(digital_human_id)
    ]
