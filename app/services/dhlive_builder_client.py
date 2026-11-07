"""HTTP client for the dhlive-builder service (方案 §5 Phase A / §6 B2).

Calls the dhlive-builder container which wraps DH_live data preparation
behind a FastAPI endpoint. Returns dhlive-v1 package bytes (byte-deterministic).
"""
from __future__ import annotations

import logging
import os

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class DhLiveBuilderError(RuntimeError):
    pass


class DhLiveBuilderClient:
    """Client for the DH_live avatar builder microservice."""

    @property
    def _endpoint(self) -> str:
        return settings.dhlive_builder_endpoint

    @property
    def _timeout(self) -> float:
        return float(settings.dhlive_builder_timeout_seconds)

    async def health(self) -> dict | None:
        """Return health payload, or None when unreachable (不抛异常，供探活)。"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._endpoint}/health")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None

    async def build_avatar(
        self,
        *,
        video_bytes: bytes,
        avatar_id: str,
        package_version: int = 1,
    ) -> tuple[bytes, dict]:
        """视频 → dhlive-v1 ZIP。返回 (zip_bytes, report)。

        report: {build_fingerprint, package_bytes, headers 原样}
        幂等性：builder 侧同指纹产出字节一致 ZIP；本层不缓存。
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._endpoint}/v1/build-avatar",
                    files={"video": ("input.mp4", video_bytes, "video/mp4")},
                    data={"avatar_id": avatar_id, "package_version": str(package_version)},
                )
        except httpx.HTTPError as exc:
            raise DhLiveBuilderError(f"dhlive-builder unreachable: {exc}") from exc
        if resp.status_code == 422:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise DhLiveBuilderError(f"quality gate rejected: {detail}")
        if resp.status_code != 200:
            raise DhLiveBuilderError(
                f"dhlive-builder build failed: HTTP {resp.status_code} {resp.text[:500]}"
            )
        fingerprint = resp.headers.get("X-Build-Fingerprint", "")
        if len(fingerprint) != 64:
            raise DhLiveBuilderError("dhlive-builder returned no build fingerprint")
        report = {
            "build_fingerprint": fingerprint,
            "package_bytes": len(resp.content),
            "x_package_bytes": resp.headers.get("X-Package-Bytes", ""),
        }
        return resp.content, report

    async def voice_reference(self, media_bytes: bytes, filename: str) -> tuple[bytes, dict]:
        """音视频 → 质量门通过的参考 WAV。返回 (wav_bytes, report)。"""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._endpoint}/v1/voice/reference",
                    files={"audio_or_video": (filename, media_bytes, "application/octet-stream")},
                )
        except httpx.HTTPError as exc:
            raise DhLiveBuilderError(f"dhlive-builder unreachable: {exc}") from exc
        if resp.status_code == 422:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise DhLiveBuilderError(f"voice quality gate rejected: {detail}")
        if resp.status_code != 200:
            raise DhLiveBuilderError(
                f"voice reference failed: HTTP {resp.status_code} {resp.text[:500]}"
            )
        import json as _json

        try:
            report = _json.loads(resp.headers.get("X-Voice-Report", "{}"))
        except ValueError:
            report = {}
        return resp.content, report

    async def voice_enroll(self, *, audio_url: str, prefix: str) -> str:
        """公网参考音频 URL → CosyVoice voice_id（ cosyvoice-v2，已验证）。"""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._endpoint}/v1/voice/enroll",
                    json={"audio_url": audio_url, "prefix": prefix},
                )
        except httpx.HTTPError as exc:
            raise DhLiveBuilderError(f"dhlive-builder unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise DhLiveBuilderError(
                f"voice enroll failed: HTTP {resp.status_code} {resp.text[:500]}"
            )
        voice_id = resp.json().get("voice_id", "")
        if not voice_id:
            raise DhLiveBuilderError("voice enroll returned empty voice_id")
        return voice_id


dhlive_builder_client = DhLiveBuilderClient()
