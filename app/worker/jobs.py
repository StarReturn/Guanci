"""Build job entrypoints called by the RQ worker.

Sync wrappers around async pipeline; RQ's default Worker is sync. We bridge
to asyncio via ``asyncio.run`` per job (one event loop per build).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _release_entitlement(task: Any | None, inputs: dict[str, Any]) -> None:
    """Refund the reserved entitlement when a build ultimately fails.

    Fire-and-forget: errors are logged but never propagate. ``release_stale_reservations``
    runs every worker startup as a backstop, so even a missed release here
    self-heals within ~30 minutes.
    """
    key = inputs.get("entitlement_key") if inputs else None
    request_id = inputs.get("entitlement_request_id") if inputs else None
    user_id = (
        (getattr(task, "owner_user_id", None) or getattr(task, "digital_human_id", None))
        if task is not None else None
    )
    if not (key and request_id and user_id):
        return
    try:
        from app.services.entitlement_stub import entitlement_service
        entitlement_service.release(user_id=user_id, key=key, request_id=request_id)
    except Exception:
        logger.exception(
            "entitlement_release_failed task_id=%s key=%s request_id=%s",
            getattr(task, "task_id", "?"), key, request_id,
        )


def _confirm_entitlement(task: Any | None, inputs: dict[str, Any]) -> None:
    """Convert the reserved entitlement into actual usage on build success.

    Fire-and-forget: if this fails, the package is already built and bound.
    The ledger stays in "reserved" and is reaped by ``release_stale_reservations``
    after 30 min (i.e. user effectively gets the build for free). Business
    success takes priority over billing precision; the alternative — unbuilding
    the avatar because the ledger write failed — is worse.
    """
    key = inputs.get("entitlement_key") if inputs else None
    request_id = inputs.get("entitlement_request_id") if inputs else None
    user_id = (
        (getattr(task, "owner_user_id", None) or getattr(task, "digital_human_id", None))
        if task is not None else None
    )
    if not (key and request_id and user_id):
        return
    try:
        from app.services.entitlement_stub import entitlement_service
        entitlement_service.confirm(
            user_id=user_id, key=key, request_id=request_id, actual_quantity=1,
        )
    except Exception:
        logger.exception(
            "entitlement_confirm_failed task_id=%s key=%s request_id=%s",
            getattr(task, "task_id", "?"), key, request_id,
        )


def run_delete_job(delete_job_id: str) -> None:
    """RQ entrypoint for storage and database erasure."""
    from app.services.digital_human_service import digital_human_service

    try:
        digital_human_service.execute_delete_job(delete_job_id)
    except Exception:
        logger.exception("digital_human_delete_failed delete_job_id=%s", delete_job_id)
        raise


def run_build_task(task_id: str, inputs: dict[str, Any]) -> None:
    """RQ entrypoint. Sync; bridges to async pipeline.

    Failure semantics:
      - On translatable build error → mark task failed in DB, do NOT raise
        (avoids RQ's retry chain which would re-run the full pipeline)
      - On infrastructure error (DB down, MinIO down) → raise to let RQ retry
    """
    try:
        asyncio.run(_run_build_task_async(task_id, inputs))
    except Exception:
        logger.exception("build_job_crashed task_id=%s", task_id)
        # Try to mark failed in DB; if even that fails, re-raise for RQ.
        try:
            from app.services.keyframe_repository import keyframe_repository
            keyframe_repository.mark_finished(
                task_id,
                status="failed",
                error="build job crashed (see worker logs)",
            )
        except Exception:
            logger.exception("mark_finished_after_crash_failed task_id=%s", task_id)
            raise


async def _run_build_task_async(task_id: str, inputs: dict[str, Any]) -> None:
    """Actual build pipeline.

    Steps:
      1. mark_started (status=running, started_at=now)
      2. Pull source_bytes + voice_wav from MinIO (using keys in inputs)
      3. keyframe_builder.run_build (P1: fail-fast stub until LAM builder lands in P2)
      4. _validate_package (zip + frame encoding + manifest hash)
      5. Upload zip to MinIO + save KeyframePackageRecord
      6. user_avatar_service.bind(...)
      7. notification_service.send(...)
      8. mark_finished (status=completed)
    """
    # Late imports to keep RQ worker startup fast
    from app.services.keyframe_builder import keyframe_builder
    from app.services.keyframe_repository import keyframe_repository

    # ----- 1. Mark started -----
    keyframe_repository.mark_started(task_id)

    task = keyframe_repository.get_task(task_id)
    if task is None:
        raise RuntimeError(f"task {task_id} not found in DB (race with cleanup?)")

    # ----- 2. Run build via keyframe_builder.run_build -----
    # run_build handles: storage pull, progress callback, validation,
    # quality gate, package upload, build_report write.
    # On success returns KeyframePackageRead; on failure raises.
    try:
        package = await keyframe_builder.run_build(
            task_id=task_id,
            digital_human_id=task.digital_human_id,
            avatar_id=task.avatar_id,
            inputs=inputs,
        )
    except Exception as exc:
        logger.exception("build_failed task_id=%s", task_id)
        # Refund the reserved entitlement since the build did not succeed.
        _release_entitlement(task, inputs)
        try:
            from app.services.digital_human_service import digital_human_service
            digital_human_service.mark_build_failed(task.digital_human_id)
        except Exception:
            logger.exception("digital_human_mark_failed_failed task_id=%s", task_id)
        keyframe_repository.mark_finished(
            task_id,
            status="failed",
            error=str(exc),
            stage="failed",
        )
        return  # do not raise — failure is recorded, no RQ retry

    # ----- 3. Post-build side effects: bind avatar + notify user -----
    try:
        from app.services.digital_human_service import digital_human_service

        package_version = digital_human_service.complete_build(
            digital_human_id=task.digital_human_id,
            task_id=task_id,
            package=package,
        )
        managed_digital_human = digital_human_service.get_by_id(task.digital_human_id)
        if package_version is not None or managed_digital_human is None:
            _bind_avatar_and_notify(
                user_id=task.owner_user_id or task.digital_human_id,
                digital_human_id=(task.digital_human_id if package_version is not None else None),
                avatar_id=task.avatar_id,
                inputs=inputs,
            )
    except Exception as exc:
        # Build succeeded but bind / notify failed. Task is still "completed"
        # (package exists, user can re-bind manually via admin). Log and move on.
        logger.exception(
            "post_build_side_effects_failed task_id=%s avatar_id=%s err=%s",
            task_id, task.avatar_id, exc,
        )
        try:
            from app.services.digital_human_service import digital_human_service
            if digital_human_service.get_by_id(task.digital_human_id) is not None:
                digital_human_service.mark_build_failed(task.digital_human_id)
                keyframe_repository.mark_finished(
                    task_id,
                    status="failed",
                    error=f"package activation failed: {exc}",
                    stage="failed",
                )
                return
        except Exception:
            logger.exception("digital_human_finalize_failure_persist_failed task_id=%s", task_id)

    # Convert the reserved entitlement into actual usage now that the build
    # has fully succeeded (package uploaded + avatar bound + user notified).
    _confirm_entitlement(task, inputs)
    keyframe_repository.mark_finished(
        task_id,
        status="completed",
        stage="done",
    )
    logger.info(
        "build_completed task_id=%s avatar_id=%s size=%d",
        task_id, task.avatar_id, package.size_bytes,
    )


def _bind_avatar_and_notify(
    *,
    user_id: str,
    digital_human_id: str | None,
    avatar_id: str,
    inputs: dict[str, Any],
) -> None:
    """Bind the successfully built avatar to the user + send notification.

    Same logic that used to live in auth.py /me/avatars/build, moved here
    so the API can return 202 immediately.
    """
    from app.services.auth_service import user_avatar_service
    from app.services.notification_service import notification_service
    from app.services.tts_provider_service import tts_provider_service

    custom_name = inputs.get("custom_name")
    voice_ref_audio_key = inputs.get("voice_ref_audio_key")
    voice_ref_text = inputs.get("voice_ref_text")
    voice_ref_duration_ms = inputs.get("voice_ref_duration_ms")
    provider_voice_id = inputs.get("provider_voice_id")

    user_avatar_service.bind(
        user_id=user_id,
        digital_human_id=digital_human_id,
        avatar_id=avatar_id,
        custom_name=custom_name,
        voice_ref_audio_key=voice_ref_audio_key,
        voice_ref_text=voice_ref_text,
        voice_ref_duration_ms=voice_ref_duration_ms,
        voice_provider=tts_provider_service.runtime().type,
        voice_model_id=(
            tts_provider_service.runtime().model
            if tts_provider_service.runtime().type == "aliyun_cosyvoice"
            else None
        ),
        voice_provider_voice_id=provider_voice_id,
    )

    notification_service.send(
        user_id=user_id,
        type="avatar_build_completed",
        title="数字人构建完成",
        body=f"你的数字人「{custom_name or '默认'}」已就绪，可以开始对话了！",
    )
