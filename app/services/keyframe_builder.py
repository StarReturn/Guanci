"""数字人素材构建编排（唯一路线：dhlive，2026-08-25 起旧图片路线移除）。

两条入口（方案双入口）：
  - 推荐入口：用户拍摄 8-60s 安静人脸视频 → submit_dhlive(source_kind="video")
  - 快捷入口：用户上传单张照片 → 照片活化容器合成微动视频 →
    submit_dhlive(source_kind="photo")，worker 内活化后走同一条
    dhlive 管线（质量门/构建/参考渲染门/自动激活完全一致）

  - ``submit_dhlive()`` : API 侧，校验 + 存储源素材 + 入队，秒级返回
  - ``run_build()``     : worker 侧，实际跑构建管线
"""
import logging
import time
from uuid import uuid4

from app.schemas.keyframe import KeyframeBuildResponse, KeyframePackageRead, KeyframeTaskRead
from app.services.keyframe_repository import keyframe_repository
from app.services.storage import storage_service

logger = logging.getLogger(__name__)


class KeyframeBuilder:
    """Build dhlive-v1 end-side avatar packages (唯一引擎路线)。"""
    async def submit_dhlive(
        self,
        *,
        digital_human_id: str,
        owner_user_id: str,
        video_bytes: bytes,
        source_kind: str = "video",
        package_version: int = 1,
        idempotency_key: str | None = None,
        queue_priority: int = 0,
        entitlement_key: str | None = None,
        entitlement_request_id: str | None = None,
    ) -> KeyframeBuildResponse:
        """dhlive 构建提交（§5 A2）：源素材入 MinIO → 任务行 → 队列。

        source_kind:
          - "video"：用户实拍 8-60s 安静人脸视频（推荐入口，效果最佳）
          - "photo"：用户单张照片（快捷入口）——worker 内先经照片活化容器
            合成微动视频，再走同一条 dhlive 管线
        不要求音色配置（§9.3：声音克隆失败不得拖累视觉包构建）。
        """
        import hashlib as _hashlib

        if source_kind not in {"video", "photo"}:
            raise ValueError(f"unsupported source_kind {source_kind!r}")
        if idempotency_key:
            existing = keyframe_repository.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                if (
                    existing.owner_user_id != owner_user_id
                    or existing.digital_human_id != digital_human_id
                ):
                    raise ValueError("idempotency key is already used by another build")
                return KeyframeBuildResponse(
                    task_id=existing.task_id,
                    status=existing.status,
                    avatar_id=existing.avatar_id,
                )

        task_id = str(uuid4())
        avatar_id = f"mk_{digital_human_id}_{int(time.time())}"
        if source_kind == "photo":
            source_storage_key = (
                f"build_inputs/{digital_human_id}/{task_id}/source_photo.jpg"
            )
            source_mime = "image/jpeg"
        else:
            source_storage_key = (
                f"build_inputs/{digital_human_id}/{task_id}/source_video.mp4"
            )
            source_mime = "video/mp4"
        storage_service.put_bytes(
            key=source_storage_key,
            content=video_bytes,
            content_type=source_mime,
        )
        task = KeyframeTaskRead(
            task_id=task_id,
            digital_human_id=digital_human_id,
            owner_user_id=owner_user_id,
            avatar_id=avatar_id,
            status="queued",
            stage="queued",
            progress=0,
            idempotency_key=idempotency_key,
        )
        keyframe_repository.save_task(task)
        inputs = {
            "renderer": "dhlive",
            "source_kind": source_kind,
            "source_storage_key": source_storage_key,
            "source_mime": source_mime,
            "source_sha256": _hashlib.sha256(video_bytes).hexdigest(),
            "owner_user_id": owner_user_id,
            "package_version": package_version,
            "entitlement_key": entitlement_key,
            "entitlement_request_id": entitlement_request_id,
        }
        try:
            from app.worker.queue import enqueue_build
            enqueue_build(task_id, inputs, queue_priority=queue_priority)
        except Exception:
            logger.exception("dhlive_enqueue_failed task_id=%s", task_id)
            keyframe_repository.mark_finished(
                task_id,
                status="failed",
                error="failed to enqueue build (Redis unreachable?)",
                stage="failed",
            )
            raise
        return KeyframeBuildResponse(
            task_id=task_id,
            status="queued",
            avatar_id=avatar_id,
        )

    # -----------------------------------------------------------------#
    # Worker path: run_build (实际跑 build pipeline)
    # -----------------------------------------------------------------#
    async def run_build(
        self,
        task_id: str,
        digital_human_id: str,
        avatar_id: str,
        inputs: dict,
    ) -> KeyframePackageRead:
        """Build pipeline entry (worker calls this). 唯一路线：dhlive。

        inputs.source_kind: "video"（用户实拍）或 "photo"（照片活化）。
        """
        if inputs.get("renderer") != "dhlive":
            raise RuntimeError(
                f"unsupported renderer {inputs.get('renderer')!r}; "
                "only dhlive is available (legacy photo routes removed 2026-08-25)"
            )
        return await self._run_dhlive_build(
            task_id=task_id,
            digital_human_id=digital_human_id,
            avatar_id=avatar_id,
            inputs=inputs,
        )

    async def _run_dhlive_build(
        self,
        task_id: str,
        digital_human_id: str,
        avatar_id: str,
        inputs: dict,
    ) -> KeyframePackageRead:
        """DH_live route (2026-08-24): 源素材 → dhlive-v1 端侧素材包（§5 Phase A）.

        Flow:
          1. Read source from storage（video 实拍 / photo 快捷入口）
          1b. photo → 照片活化容器合成微动视频（同管线唯一差异点）
          2. Call dhlive_builder_client (HTTP → dhlive-builder container)
          3. Validate with dhlive_package_validator (fail-closed)
          4. 参考渲染门 → 自动激活
        """
        import time as _time
        t0 = _time.time()

        async def _progress(stage: str, percent: int) -> None:
            try:
                keyframe_repository.update_progress(task_id, stage, percent)
            except Exception:
                logger.debug("progress_persist_failed task_id=%s", task_id)

        await _progress("preparing", 5)
        source_storage_key = inputs.get("source_storage_key", "")
        if not source_storage_key:
            raise RuntimeError("dhlive build: source_storage_key missing")
        source_bytes = storage_service.get_bytes(source_storage_key)
        if not source_bytes:
            raise RuntimeError(f"dhlive build: source not found: {source_storage_key}")

        if inputs.get("source_kind") == "photo":
            # 快捷入口：照片 → 活化容器 → 10s 微动视频（嘴闭合）。
            # 溯源保持用户原始照片：source_sha256 不被合成视频覆盖。
            await _progress("photo_activation", 8)
            from app.services.photo_activation_service import get_photo_activation_client

            video_bytes, activation_report = await get_photo_activation_client().animate(
                photo_bytes=source_bytes,
                avatar_id=avatar_id,
            )
            logger.info(
                "photo_activation_done dh=%s avatar=%s report=%s",
                digital_human_id, avatar_id, activation_report,
            )
        else:
            video_bytes = source_bytes
        await _progress("preparing", 10)

        await _progress("dhlive_build", 15)
        from app.services.dhlive_builder_client import (
            DhLiveBuilderError,
            dhlive_builder_client,
        )

        # 版本号以 DB 为准（激活门要求 manifest.package_version == 下一版本号），
        # 表单值仅作提示——否则同一数字人第二次构建必然激活失败
        from app.services.digital_human_service import digital_human_service as _dhs

        versions = _dhs.list_package_versions(digital_human_id)
        package_version_no = max(
            (int(v.version) for v in versions), default=0
        ) + 1
        zip_bytes, report = await dhlive_builder_client.build_avatar(
            video_bytes=video_bytes,
            avatar_id=avatar_id,
            package_version=package_version_no,
        )
        await _progress("dhlive_build", 70)

        # --- fail-closed 契约校验（失败即任务失败，无降级）---
        await _progress("validating", 75)
        from app.services.dhlive_package_validator import dhlive_package_validator

        validation = dhlive_package_validator.validate(zip_bytes)
        await _progress("validating", 80)

        # --- Upload to MinIO ---
        await _progress("upload", 85)
        zip_key = (
            f"avatars/dhlive/{digital_human_id}/{avatar_id}/"
            f"dhlive_v1_p{package_version_no}.zip"
        )
        _, size_bytes, sha256 = storage_service.put_bytes(
            zip_key, zip_bytes, "application/zip"
        )
        try:
            download_url = storage_service.presigned_url(zip_key)
        except Exception:
            download_url = ""

        # --- 参考渲染门（§9.3）→ 自动激活 ---
        # 不落 pending_reference_render 中间态：构建任务内完成
        # 渲染门验收即激活，失败即任务失败（失败包不得激活）。
        await _progress("reference_render", 88)
        from app.services.dhlive_reference_render import (
            ReferenceRenderError,
            render_reference,
        )

        try:
            render = render_reference(zip_bytes)
        except ReferenceRenderError as exc:
            raise RuntimeError(f"dhlive reference render failed: {exc}") from exc
        if not render.report.get("passed"):
            raise RuntimeError(
                "dhlive reference render gate rejected the package: "
                f"{render.report.get('checks')}"
            )
        artifact_key = (
            f"avatars/dhlive/{digital_human_id}/{avatar_id}/"
            f"dhlive_v1_p{package_version_no}_reference.jpg"
        )
        storage_service.put_bytes(artifact_key, render.artifact_jpeg, "image/jpeg")

        await _progress("activating", 92)
        from app.services import digital_human_service as _dh_service
        from app.schemas.keyframe import KeyframePackageRead as PackageRead
        import hashlib as _hashlib

        _dh_service.digital_human_service.activate_validated_dhlive_package(
            digital_human_id=digital_human_id,
            avatar_id=avatar_id,
            build_task_id=task_id,
            storage_key=zip_key,
            package_bytes=zip_bytes,
            source_asset_sha256=inputs.get("source_sha256") or _hashlib.sha256(
                source_bytes
            ).hexdigest(),
            reference_render_storage_key=artifact_key,
            reference_render_report=render.report,
        )

        package = PackageRead(
            avatar_id=avatar_id,
            digital_human_id=digital_human_id,
            version=package_version_no,
            engine="dhlive",
            package_format_version=1,
            renderer_mode="dhlive_webview_v1",
            package_schema_revision=1,
            renderer_runtime_version=validation.manifest["renderer_runtime"]["version"],
            build_fingerprint=report.get("build_fingerprint"),
            source_asset_sha256=inputs.get("source_sha256"),
            manifest_metadata=validation.manifest,
            validation_report={**validation.report, "reference_render": render.report},
            size_bytes=size_bytes,
            storage_key=zip_key,
            download_url=download_url,
            sha256=sha256,
            quality_status="validated",
        )

        await _progress("done", 100)
        logger.info(
            "dhlive_build_complete dh=%s avatar=%s elapsed=%.1fs size=%d fingerprint=%s",
            digital_human_id, avatar_id, _time.time() - t0, size_bytes,
            report.get("build_fingerprint", ""),
        )

        return package

    def get_task(self, task_id: str) -> KeyframeTaskRead | None:
        return keyframe_repository.get_task(task_id)

    def get_package(self, avatar_id: str) -> KeyframePackageRead | None:
        return keyframe_repository.get_package(avatar_id)

    # -----------------------------------------------------------------#
    # Package validation
    # -----------------------------------------------------------------#


keyframe_builder = KeyframeBuilder()
