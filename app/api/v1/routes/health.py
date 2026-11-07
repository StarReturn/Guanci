from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.engine_factory import get_tts_engine
from app.services.redis_client import redis_client

router = APIRouter()


def postgres_health_check() -> dict:
    try:
        with SessionLocal() as db:
            db.execute(text("select 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.get("/dependencies")
async def dependency_health() -> dict:
    postgres = postgres_health_check()
    redis = await redis_client.health_check()
    tts = await get_tts_engine().health_check()
    statuses = [postgres["status"], redis["status"], tts["status"]]

    # Build queue depth (RQ). Non-fatal if Redis is down — report -1.
    build_queue_depth = -1
    try:
        from app.worker.queue import queue_depth
        build_queue_depth = queue_depth()
    except Exception:
        pass

    return {
        "status": "ok" if all(status == "ok" for status in statuses) else "degraded",
        "dependencies": {
            "postgres": postgres,
            "redis": redis,
            "tts": {**tts, "provider": settings.tts_provider},
        },
        "build_queue_depth": build_queue_depth,
    }


@router.get("/engines")
async def engines_health() -> dict:
    """v4 新增：所有引擎 + GPU 并发状态。

    用于运维监控和 P3.5 调试。返回当前 provider 配置 + 各引擎健康度 +
    GPU 并发水位（active_runtime / active_builds / 限流配置）。
    视觉构建引擎（dhlive-builder / photo-activation）是 worker 内部
    依赖，不入此表——由构建任务自身的 fail-closed 质量门兜底。
    """
    # 根据当前 provider 决定 ping 哪些引擎
    results: dict[str, dict] = {}
    statuses: list[str] = []

    # TTS
    tts = await get_tts_engine().health_check()
    results["tts"] = {**tts, "provider": settings.tts_provider}
    statuses.append(tts["status"])

    # ASR — route through the factory so every provider (aliyun_asr /
    # funasr / whisper) is covered without a hardcoded branch per name.
    from app.services.engine_factory import get_asr_engine
    asr_engine = get_asr_engine()
    if hasattr(asr_engine, "health_check"):
        asr = await asr_engine.health_check()
        results["asr"] = asr
        statuses.append(asr["status"])
    else:
        # whisper (asr_service) has no health_check; treat as ok.
        results["asr"] = {"status": "ok", "provider": "whisper"}

    # GPU semaphore stats
    try:
        from app.core.gpu_semaphore import gpu_semaphore
        gpu_stats = gpu_semaphore.stats()
    except Exception:
        gpu_stats = {"error": "gpu_semaphore not initialized"}

    return {
        "status": "ok" if all(s == "ok" for s in statuses) else "degraded",
        "engines": results,
        "gpu": gpu_stats,
    }
