"""dhlive 构建路由（§5 A2）——唯一用户入口：照片。

POST /me/digital-humans/{id}/photo-preflight
  单张照片（质量门禁）→ token
POST /me/digital-humans/{id}/build-dhlive
  token → 异步构建任务

产品决策（2026-08-25）：不再开放用户视频上传——用户随手拍的视频人物
大小/构图不可控，且大概率带说话嘴动；照片经活化容器合成微动视频，
人脸大小与构图标准化、嘴部默认闭合，质量由系统保证而非用户保证。
视频构建能力（source_kind=video）保留给内部/管理员/golden 测试。

worker 内单管线：照片活化 → 质量门 → GPU 构建 → 参考渲染门 → 自动
激活。活化输出无任何豁免，逐帧质检 fail-closed。
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.middleware.jwt import get_request_user_id
from app.services.auth_service import audit_service
from app.services.entitlement_stub import EntitlementDenied, entitlement_service
from app.services.storage import storage_service

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PHOTO_BYTES = 20 * 1024 * 1024
TOKEN_TTL_MINUTES = 60


class PhotoPreflightResponse(BaseModel):
    accepted: bool
    entry: str = "photo"
    width: int
    height: int
    format: str
    size_bytes: int
    source_sha256: str
    preprocess_token: str
    expires_at: str


@router.post("/{digital_human_id}/build-dhlive", status_code=202)
async def build_dhlive(
    digital_human_id: str,
    request: Request,
    preprocess_token: str = Form(...),
    package_version: int = Form(default=1),
):
    user_id = get_request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="user login required")

    from app.services.portrait_preprocess_service import read_metadata
    from app.services.digital_human_service import (
        DigitalHumanStateError,
        digital_human_service,
    )
    from app.services.keyframe_builder import keyframe_builder

    meta = read_metadata(storage_service, preprocess_token, user_id)
    if meta is None or meta.get("kind") != "dhlive_photo":
        raise HTTPException(
            status_code=422, detail="preprocess_token expired or invalid"
        )
    if meta.get("owner_user_id") != user_id:
        raise HTTPException(status_code=422, detail="token owner mismatch")
    try:
        expires_at = datetime.fromisoformat(meta["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            raise HTTPException(status_code=422, detail="token expired")
    except KeyError as exc:
        raise HTTPException(status_code=422, detail="token metadata invalid") from exc

    usage_request_id = f"dhlive-build-{uuid4()}"
    try:
        digital_human_service.ensure_buildable(user_id, digital_human_id)
        entitlement_service.reserve(
            user_id=user_id,
            key="avatar_builds_per_period",
            quantity=1,
            request_id=usage_request_id,
            resource_type="digital_human",
            resource_id=digital_human_id,
        )
        digital_human_service.mark_building(user_id, digital_human_id)
    except HTTPException:
        raise
    except EntitlementDenied as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        try:
            entitlement_service.release(
                user_id=user_id,
                key="avatar_builds_per_period",
                request_id=usage_request_id,
            )
        except Exception:
            logger.exception("dhlive_build_entitlement_release_failed")
        if isinstance(exc, DigitalHumanStateError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    photo_bytes = storage_service.get_bytes(meta["source_key"])
    if not photo_bytes:
        digital_human_service.mark_build_failed(digital_human_id)
        raise HTTPException(status_code=422, detail="source photo expired from storage")
    if hashlib.sha256(photo_bytes).hexdigest() != meta.get("source_sha256"):
        digital_human_service.mark_build_failed(digital_human_id)
        raise HTTPException(status_code=422, detail="source photo hash mismatch")

    try:
        result = await keyframe_builder.submit_dhlive(
            digital_human_id=digital_human_id,
            owner_user_id=user_id,
            video_bytes=photo_bytes,
            source_kind="photo",
            package_version=package_version,
            entitlement_key="avatar_builds_per_period",
            entitlement_request_id=usage_request_id,
        )
    except Exception as exc:
        digital_human_service.mark_build_failed(digital_human_id)
        try:
            entitlement_service.release(
                user_id=user_id,
                key="avatar_builds_per_period",
                request_id=usage_request_id,
            )
        except Exception:
            logger.exception("dhlive_build_submit_release_failed")
        raise HTTPException(status_code=503, detail=f"build submission failed: {exc}") from exc

    audit_service.log(
        action="digital_human.dhlive_build_submitted",
        actor_user_id=user_id,
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"task_id": result.task_id, "avatar_id": result.avatar_id},
    )
    return {
        "task_id": result.task_id,
        "digital_human_id": digital_human_id,
        "avatar_id": result.avatar_id,
        "status": result.status,
        "stage": "queued",
        "progress": 0,
        "error": None,
    }


@router.post(
    "/{digital_human_id}/photo-preflight",
    response_model=PhotoPreflightResponse,
)
async def photo_preflight_quick(
    digital_human_id: str,
    request: Request,
    photo: UploadFile = File(...),
) -> PhotoPreflightResponse:
    """快捷入口预检：单张照片（活化在 worker 内执行）。

    照片要求与旧直建路线一致：单人脸、正脸、清晰、无遮挡
    （validate_photo 门禁复用）；活化输出的逐帧级质检由 dhlive-builder
    fail-closed 复核，此处不做视觉推理。
    """
    user_id = get_request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="user login required")

    from app.services.digital_human_service import (
        DigitalHumanNotFound,
        digital_human_service,
    )
    from app.services.photo_validator import PhotoValidationError, validate_photo

    try:
        digital_human_service.get_owned(user_id, digital_human_id)
    except DigitalHumanNotFound:
        raise HTTPException(status_code=404, detail="digital human not found")

    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="photo is empty")
    if len(photo_bytes) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=413, detail="photo exceeds 20MB limit")
    try:
        facts = validate_photo(photo_bytes, photo.content_type or "image/jpeg")
    except PhotoValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.reason) from exc

    # NSFW 内容门（NudeNet，离线；enforce 模式下检测器不可用即拒绝上传）
    from app.services.nsfw_guard import NsfwGuardUnavailable, get_nsfw_guard

    nsfw_guard = get_nsfw_guard()
    if nsfw_guard is not None:
        try:
            nsfw_result = nsfw_guard.check_photo(photo_bytes)
        except NsfwGuardUnavailable:
            logger.exception("nsfw_guard_unavailable dh=%s", digital_human_id)
            raise HTTPException(
                status_code=503,
                detail="内容审查服务未就绪，请稍后重试",
            )
        if not nsfw_result.passed:
            source_sha_for_case = hashlib.sha256(photo_bytes).hexdigest()
            try:
                from app.services.admin_governance_service import (
                    content_safety_service,
                )

                content_safety_service.create({
                    "source_type": "photo_preflight",
                    "source_id": digital_human_id,
                    "user_id": user_id,
                    "category": "nsfw",
                    "severity": "high",
                    "status": "open",
                    "metadata": {
                        "labels": nsfw_result.labels[:10],
                        "max_explicit_score": nsfw_result.max_explicit_score,
                        "source_sha256": source_sha_for_case,
                    },
                })
            except Exception:
                logger.exception("nsfw_case_record_failed dh=%s", digital_human_id)
            # 通用文案：不回显具体标签，避免对抗性试探
            raise HTTPException(
                status_code=422,
                detail="照片未通过内容审查，请更换合规照片后重试",
            )

    source_sha = hashlib.sha256(photo_bytes).hexdigest()
    token = uuid4().hex
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(minutes=TOKEN_TTL_MINUTES)).isoformat()

    prefix = f"preprocess/{user_id}/{token}"
    source_key = f"{prefix}/source.jpg"
    storage_service.put_bytes(source_key, photo_bytes, "image/jpeg")
    meta = {
        "kind": "dhlive_photo",
        "preprocess_token": token,
        "preprocess_version": 1,
        "owner_user_id": user_id,
        "source_sha256": source_sha,
        "source_key": source_key,
        "size_bytes": len(photo_bytes),
        "expires_at": expires_at,
    }
    storage_service.put_bytes(
        f"{prefix}/preprocess.json",
        json.dumps(meta).encode(),
        "application/json",
    )

    # 原始照片挂 source 槽位（复用 photo 槽使 ensure_buildable 的
    # 资产/授权检查统一生效）；admin 重试按内容魔数自动识别 kind
    attached_key = f"avatars/{digital_human_id}/source/{token}.jpg"
    storage_service.put_bytes(attached_key, photo_bytes, facts.get("content_type", "image/jpeg"))
    digital_human_service.attach_photo(
        owner_user_id=user_id,
        digital_human_id=digital_human_id,
        storage_key=attached_key,
        size_bytes=len(photo_bytes),
    )
    audit_service.log(
        action="digital_human.dhlive_photo_preflight",
        actor_user_id=user_id,
        target_type="digital_human",
        target_id=digital_human_id,
        detail={"size_bytes": len(photo_bytes)},
    )

    return PhotoPreflightResponse(
        accepted=True,
        width=int(facts["width"]),
        height=int(facts["height"]),
        format=str(facts["format"]),
        size_bytes=len(photo_bytes),
        source_sha256=source_sha,
        preprocess_token=token,
        expires_at=expires_at,
    )
