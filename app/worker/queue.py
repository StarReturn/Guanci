"""RQ queue wrapper for async keyframe builds.

Why RQ + Redis (not Celery / RabbitMQ):
  - Redis is already a required service in this stack (postgres/redis/minio)
  - RQ is a thin Python wrapper, no native deps, no extra broker container
  - Single-GPU single-worker scenario; RQ's strengths (FIFO + retries) suffice

Design:
  - API enqueues only ``task_id`` + small ``inputs`` dict (storage keys + voice id)
  - Photo / voice bytes stay in MinIO, never travel through Redis
  - Worker reads task_id, fetches bytes from MinIO, runs build, writes result back
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import redis
from rq import Queue, Retry

from app.core.config import settings

logger = logging.getLogger(__name__)


_redis: redis.Redis = redis.from_url(
    settings.redis_url,
    decode_responses=False,  # jobs carry bytes-free dicts, but stay safe
    socket_connect_timeout=5.0,
    socket_timeout=10.0,
)

_build_queue: Queue = Queue(
    settings.worker_queue_name,
    connection=_redis,
    is_async=True,
)
_build_queues: dict[str, Queue] = {
    tier: Queue(
        f"{settings.worker_queue_name}:{tier}",
        connection=_redis,
        is_async=True,
    )
    for tier in ("max", "pro", "plus", "free")
}


def enqueue_build(
    task_id: str, inputs: dict[str, Any], queue_priority: int = 0
) -> str:
    """Enqueue a build task. Returns the RQ job id.

    Args:
        task_id: primary key in ``keyframe_tasks`` table.
        inputs: small JSON-able dict (storage keys, voice id, custom_name).
                Photo / audio bytes are NOT passed here; worker fetches them
                from MinIO using the storage keys in this dict.
    """
    tier = (
        "max" if queue_priority >= 30 else
        "pro" if queue_priority >= 20 else
        "plus" if queue_priority >= 10 else "free"
    )
    queue = _build_queues[tier]
    job = queue.enqueue(
        "app.worker.jobs.run_build_task",
        task_id=task_id,
        inputs=inputs,
        job_timeout=settings.worker_job_timeout_seconds,
        result_ttl=86400,        # keep result 24h for inspection
        failure_ttl=86400,       # keep failure info 24h
        ttl=3600,                # drop job if queued >1h (stale)
    )
    logger.info(
        "build_enqueued task_id=%s job_id=%s queue=%s",
        task_id, job.id, queue.name,
    )
    return job.id


def queue_depth() -> int:
    """Return number of jobs waiting in the build queue (for /health)."""
    return _build_queue.count + sum(queue.count for queue in _build_queues.values())


def enqueue_delete(delete_job_id: str, execute_after: datetime) -> str:
    """Schedule a digital-human erasure job after the retention window."""
    job = _build_queue.enqueue_at(
        execute_after,
        "app.worker.jobs.run_delete_job",
        delete_job_id=delete_job_id,
        job_timeout=900,
        result_ttl=86400,
        failure_ttl=604800,
        retry=Retry(max=3, interval=[60, 300, 900]),
    )
    logger.info(
        "delete_enqueued delete_job_id=%s job_id=%s execute_after=%s",
        delete_job_id,
        job.id,
        execute_after.isoformat(),
    )
    return job.id
