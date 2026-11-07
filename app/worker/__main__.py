"""RQ worker process entrypoint.

Run with:
    python -m app.worker

Concurrency model:
  - Default RQ Worker forks per job. For GPU builds we want serialization
    (single GPU, single build at a time), so we set the queue's worker
    pool size to 1 (default).

Startup sequence:
  1. Configure logging (structlog if LOG_FORMAT=json)
  2. Recover stale tasks (running > worker_recover_stale_after_seconds)
     - Re-enqueue them so they run again
  3. Start RQ Worker.work() (blocking)
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.worker.queue import _build_queue, _build_queues, _redis

logger = logging.getLogger(__name__)


def _recover_stale_tasks() -> int:
    """Find tasks stuck in running/queued state and re-enqueue them.

    Returns the count of recovered tasks. Called once on worker startup,
    before .work() blocks. Idempotent: if task is already queued in RQ,
    re-enqueueing creates a duplicate job, but run_build_task is idempotent
    enough (status transitions are guarded by mark_started/mark_finished).
    """
    from app.services.keyframe_repository import keyframe_repository
    from app.worker.queue import enqueue_build

    try:
        stale = keyframe_repository.find_stale_running(
            settings.worker_recover_stale_after_seconds,
        )
    except Exception:
        logger.exception("find_stale_running_failed; skipping recovery")
        return 0

    if not stale:
        return 0

    recovered = 0
    for task in stale:
        # If the task has no inputs recorded (very old task pre-P0 fix),
        # we cannot rebuild — mark failed instead.
        inputs = task.build_report or {}
        if not inputs:
            keyframe_repository.mark_finished(
                task.task_id,
                status="failed",
                error="task stale and inputs unavailable; pre-P0 build",
                stage="failed",
            )
            logger.warning(
                "stale_task_unrecoverable task_id=%s (no inputs)", task.task_id,
            )
            continue

        # Reset to queued so a subsequent crash doesn't make it look running.
        # NOTE: inputs lives in build_report only as a fallback; in normal
        # flow inputs come through RQ job args, not DB. So we cannot rebuild
        # from DB alone if the RQ job is lost. Mark failed for now; user
        # can resubmit.
        keyframe_repository.mark_finished(
            task.task_id,
            status="failed",
            error=(
                f"task was running >{settings.worker_recover_stale_after_seconds}s "
                "and RQ job was lost; please resubmit"
            ),
            stage="failed",
        )
        logger.warning(
            "stale_task_failed task_id=%s (RQ job lost, user must resubmit)",
            task.task_id,
        )
        recovered += 1

    return recovered


def _recover_delete_jobs() -> int:
    """Re-schedule durable pending delete jobs after worker/Redis restarts."""
    from app.services.digital_human_service import digital_human_service
    from app.worker.queue import enqueue_delete

    try:
        jobs = digital_human_service.list_pending_delete_jobs()
        for job in jobs:
            enqueue_delete(job.delete_job_id, job.execute_after)
        return len(jobs)
    except Exception:
        logger.exception("delete_job_recovery_failed")
        return 0


def main() -> None:
    from rq import Worker

    # Match API logging configuration
    try:
        from app.core.logging import configure_logging
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)

    # Connect to Redis (fail fast if unreachable)
    try:
        _redis.ping()
    except Exception:
        logger.exception("redis_unreachable; worker cannot start")
        raise

    from app.services.entitlement_stub import entitlement_service

    released_reservations = entitlement_service.release_stale_reservations()
    recovered = _recover_stale_tasks()
    recovered_deletes = _recover_delete_jobs()
    logger.info(
        "worker_starting queue=%s recovered=%s delete_jobs=%s released_reservations=%s timeout=%ds",
        settings.worker_queue_name, recovered, recovered_deletes, released_reservations,
        settings.worker_job_timeout_seconds,
    )

    worker = Worker(
        [
            _build_queues["max"],
            _build_queues["pro"],
            _build_queues["plus"],
            _build_queues["free"],
            _build_queue,
        ],
        connection=_redis,
        default_result_ttl=86400,
    )
    # with_scheduler=True moves delayed deletion jobs into the queue when their
    # retention deadline is reached.
    worker.work(logging_level="INFO", with_scheduler=True)


if __name__ == "__main__":
    main()
