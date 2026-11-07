"""照片活化客户端（快捷入口：一张照片 → 微动视频）。

后端唯一图片消费方：把用户照片发给照片活化容器（LivePortrait 系单图
驱动，复用第 4 版方案遗留的 GPU 容器），合成 10 秒 25fps 微动 idle 视频
（轻微头动 + 自然眨眼，嘴部保持闭合——dhlive 管线要求嘴留白给运行时
TTS 驱动）。

契约（fail-closed）：
- 输入：validate_photo 通过的单张人脸照片（正脸/清晰/无遮挡）
- 输出：H.264 MP4，8-60s、≥360p、≥20fps——即 dhlive 构建质量门的
  合法输入；后续逐帧级质检（含"明显说话"拒绝门）由 dhlive-builder
  容器照常执行，活化输出无豁免
- 超时/非 200/坏 MP4 → DhLivePhotoActivationError，任务失败，不降级

容器部署（服务器）：docker/liveportrait-activation（GPU），
endpoint 由 settings.dhlive_photo_activation_endpoint 配置。
"""
from __future__ import annotations

import json
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class DhLivePhotoActivationError(RuntimeError):
    """照片活化失败（容器不可用 / 输出不合格）。"""


class PhotoActivationClient:
    def __init__(self, endpoint: str | None = None, timeout_s: int | None = None):
        # settings 仅在参数缺省时加载（单测可注入 endpoint 零依赖）
        self._endpoint: str | None = endpoint.rstrip("/") if endpoint else None
        self._timeout = timeout_s
        if self._endpoint is None or self._timeout is None:
            from app.core.config import settings

            self._endpoint = (
                self._endpoint or settings.dhlive_photo_activation_endpoint
            ).rstrip("/")
            if self._timeout is None:
                self._timeout = settings.dhlive_photo_activation_timeout_seconds

    async def animate(
        self, *, photo_bytes: bytes, avatar_id: str | None = None,
    ) -> tuple[bytes, dict]:
        """照片 → (微动视频 MP4 bytes, activation report dict)。"""
        if not photo_bytes:
            raise DhLivePhotoActivationError("photo is empty")

        import httpx

        url = f"{self._endpoint}/v1/animate"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    files={"photo": ("source.jpg", photo_bytes, "image/jpeg")},
                    data={"avatar_id": avatar_id or f"pa-{uuid4().hex[:12]}"},
                )
        except httpx.HTTPError as exc:
            raise DhLivePhotoActivationError(
                f"photo activation service unreachable: {exc}"
            ) from exc

        if response.status_code != 200:
            raise DhLivePhotoActivationError(
                f"photo activation failed: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )

        video_bytes = response.content
        report = {}
        raw_report = response.headers.get("X-Activation-Report", "")
        if raw_report:
            try:
                report = json.loads(raw_report)
            except json.JSONDecodeError:
                logger.warning("photo_activation_report_unparsable")

        # 输出必须是合法 MP4（容器间契约：ftyp box 头）——逐帧级质量门
        # 在 dhlive-builder 内 fail-closed 复核，这里只拦结构级损坏。
        if len(video_bytes) < 1024 or b"ftyp" not in video_bytes[:64]:
            raise DhLivePhotoActivationError(
                "photo activation returned invalid MP4 payload"
            )
        return video_bytes, report


def get_photo_activation_client() -> PhotoActivationClient:
    """惰性单例：首次调用才读 settings（导入零副作用）。"""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = PhotoActivationClient()
    return _client_singleton


_client_singleton: PhotoActivationClient | None = None
