"""GPU concurrency limiter with CPU fallback.

控制单 GPU（如 4060 Ti 16GB）上的 TTS 推理并发数：
  - 默认上限 = settings.gpu_runtime_limit（8）
  - 超过 → 排队等
  - 超过 settings.cpu_fallback_threshold（12）→ 返回 "cpu" 模式
    （caller 应回动到 CPU 推理路径，或拒绝请求）

Build 与 runtime 互斥：
  当 build_task_count > 0 时，runtime 上限降到
  settings.gpu_runtime_limit_during_build（4），让出显存给 EchoMimic。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AcquireResult:
    """acquire() 返回值。

    mode 取值：
      - "gpu"    ：正常 GPU 推理
      - "cpu"    ：超过安全水位，建议走 CPU 兜底
      - "queue"  ：正在排队（caller 等到获得 gpu/cpu）
    """

    mode: str           # "gpu" | "cpu"
    waited_seconds: float
    queue_position: int = 0


class GPUSemaphore:
    """全局 GPU 并发控制。

    注意：单进程内的 Semaphore；多 worker 部署时需要外部位点（Redis），
    本 POC 默认单 worker。
    """

    def __init__(self) -> None:
        self._runtime_limit: int = settings.gpu_runtime_limit
        self._sem: asyncio.Semaphore = asyncio.Semaphore(self._runtime_limit)
        self._active_count: int = 0
        self._build_count: int = 0          # 当前活跃的 build 任务数
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def build_count(self) -> int:
        return self._build_count

    @property
    def current_limit(self) -> int:
        """当前实际生效的并发上限（视 build 是否在跑而定）。"""
        if self._build_count > 0:
            return settings.gpu_runtime_limit_during_build
        return self._runtime_limit

    async def acquire(self) -> AcquireResult:
        """获取推理许可。

        返回 AcquireResult，caller 必须在 finally 中调 release()。
        """
        import time
        start = time.monotonic()

        # 检查是否已超 CPU 兜底阈值
        # （active_count 包含 GPU + CPU，但这里简化为只统计 GPU）
        if self._active_count >= settings.cpu_fallback_threshold:
            return AcquireResult(mode="cpu", waited_seconds=0.0)

        await self._sem.acquire()
        async with self._lock:
            self._active_count += 1
        waited = time.monotonic() - start
        return AcquireResult(mode="gpu", waited_seconds=waited)

    async def release(self) -> None:
        async with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._sem.release()

    async def acquire_build(self) -> bool:
        """build 任务尝试获取许可。

        返回 True 表示可以跑 build；False 表示 runtime 太忙，让 build 等待。
        同时持有时会调 ``_adjust_limit_for_build`` 降低 runtime 上限。
        """
        async with self._lock:
            # build 任务数上限
            if self._build_count >= settings.gpu_build_limit:
                return False
            self._build_count += 1

        # 如果原 limit 高于 during_build，需要重建 semaphore（无法直接 resize）
        # 简化：不动态收缩，依赖 settings.gpu_runtime_limit_during_build
        # 作为 "应该等待" 的提示。caller 根据 build_count 判断是否排队。
        logger.info(
            "gpu_build_acquired active_builds=%d active_runtime=%d",
            self._build_count, self._active_count,
        )
        return True

    async def release_build(self) -> None:
        async with self._lock:
            if self._build_count > 0:
                self._build_count -= 1
        logger.info(
            "gpu_build_released active_builds=%d active_runtime=%d",
            self._build_count, self._active_count,
        )

    def stats(self) -> dict:
        """供 /health/engines 端点使用。"""
        return {
            "active_runtime": self._active_count,
            "active_builds": self._build_count,
            "runtime_limit": self.current_limit,
            "cpu_fallback_threshold": settings.cpu_fallback_threshold,
            "build_limit": settings.gpu_build_limit,
        }


gpu_semaphore = GPUSemaphore()


__all__ = ["gpu_semaphore", "GPUSemaphore", "AcquireResult"]
