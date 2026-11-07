"""Fail-closed validator for the dhlive-v1 Android avatar package."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

from app.services.avatar_package_validator import (
    AvatarPackageValidationError,
    AvatarPackageValidationResult,
    avatar_package_validator_registry,
)


_REQUIRED_PATHS = {
    "manifest.json",
    "video/01.mp4",
    "data/combined_data.json.gz",
    "preview/thumbnail.jpg",
}
_CONTENT_PATHS = _REQUIRED_PATHS - {"manifest.json"}
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_GEOMETRY_BYTES = 256 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100


def _fail(detail: str) -> None:
    raise AvatarPackageValidationError(detail)


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        _fail(f"{label} fields are invalid: {actual}")
    return value


def _read_limited(stream, limit: int, label: str) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        _fail(f"{label} exceeds the {limit}-byte limit")
    return data


def _assert_json_finite(value: Any, path: str = "geometry") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{path} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")


def _geometry_frame_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        # DH_live 真实格式：顶层 dict 的帧集合在 "json_data"（data_preparation_web 产出）
        for key in ("json_data", "frames", "data", "combined_data"):
            frames = payload.get(key)
            if isinstance(frames, list):
                return len(frames)
        numeric_keys = [key for key in payload if str(key).isdigit()]
        if numeric_keys and len(numeric_keys) == len(payload):
            return len(numeric_keys)
    _fail("geometry JSON has no supported frame collection")
    raise AssertionError("unreachable")


def _default_ffprobe(video_bytes: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        path = handle.name
        handle.write(video_bytes)
    try:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
                    "-of",
                    "json",
                    path,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            _fail(f"ffprobe failed: {exc}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            _fail(f"ffprobe returned invalid JSON: {exc}")
        streams = payload.get("streams") or []
        if len(streams) != 1:
            _fail("package video must contain exactly one video stream")
        return streams[0]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _parse_rate(value: Any) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        if float(denominator) == 0:
            _fail("ffprobe returned an invalid frame rate")
        return float(numerator) / float(denominator)
    return float(text)


class DhLivePackageValidator:
    def __init__(
        self,
        *,
        probe_video: Callable[[bytes], dict[str, Any]] | None = None,
    ) -> None:
        self._probe_video = probe_video or _default_ffprobe

    def validate(self, package_bytes: bytes) -> AvatarPackageValidationResult:
        if not package_bytes or len(package_bytes) > _MAX_ARCHIVE_BYTES:
            _fail("package archive is empty or exceeds the archive size limit")

        try:
            archive = zipfile.ZipFile(io.BytesIO(package_bytes), "r")
        except (zipfile.BadZipFile, OSError) as exc:
            _fail(f"invalid ZIP archive: {exc}")

        with archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                _fail("ZIP archive contains duplicate paths")
            if set(names) != _REQUIRED_PATHS:
                missing = sorted(_REQUIRED_PATHS - set(names))
                extra = sorted(set(names) - _REQUIRED_PATHS)
                _fail(f"unexpected package layout: missing={missing}, extra={extra}")

            total_uncompressed = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                    _fail(f"unsafe ZIP path: {info.filename!r}")
                unix_mode = info.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    _fail(f"ZIP symlink is forbidden: {info.filename!r}")
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                    _fail("ZIP uncompressed size exceeds the package limit")
                if info.file_size and not info.compress_size:
                    _fail(f"invalid compression metadata for {info.filename!r}")
                if info.compress_size and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
                    _fail(f"compression ratio is unsafe for {info.filename!r}")

            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(f"manifest.json is invalid: {exc}")
            self._validate_manifest_shape(manifest)

            file_specs = {item["path"]: item for item in manifest["files"]}
            if set(file_specs) != _CONTENT_PATHS:
                _fail("manifest files must describe each package content file exactly once")

            file_bytes: dict[str, bytes] = {}
            for path in sorted(_CONTENT_PATHS):
                data = archive.read(path)
                file_bytes[path] = data
                spec = file_specs[path]
                if len(data) != spec["size_bytes"]:
                    _fail(f"size mismatch for {path}")
                if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                    _fail(f"sha256 mismatch for {path}")

        geometry = self._validate_geometry(file_bytes["data/combined_data.json.gz"])
        video = self._validate_video(file_bytes["video/01.mp4"], manifest["video"])
        expected_frames = manifest["geometry"]["frame_count"]
        if geometry["frame_count"] != expected_frames:
            _fail("geometry frame_count does not match manifest")
        # DH_live 契约（reverse_option 默认开）：视频 = 几何 ×1 或 ×2（正向+反向循环）
        if manifest["video"]["frame_count"] not in (expected_frames, expected_frames * 2):
            _fail(
                "video frame_count must be geometry x1 or x2 "
                f"(video={manifest['video']['frame_count']} geometry={expected_frames})"
            )

        report = {
            "passed": True,
            "engine": "dhlive",
            "package_schema_revision": 1,
            "file_count": len(_REQUIRED_PATHS),
            "archive_size_bytes": len(package_bytes),
            "uncompressed_size_bytes": total_uncompressed,
            "geometry": geometry,
            "video": video,
        }
        return AvatarPackageValidationResult(
            engine="dhlive",
            renderer_mode="dhlive_webview_v1",
            package_format_version=1,
            package_schema_revision=1,
            manifest=manifest,
            report=report,
        )

    @staticmethod
    def _validate_manifest_shape(manifest: Any) -> None:
        manifest_fields = {
            "schema_revision",
            "engine",
            "renderer_mode",
            "package_format_version",
            "avatar_id",
            "package_version",
            "build_fingerprint",
            "renderer_runtime",
            "video",
            "geometry",
            "preview",
            "files",
        }
        manifest = _require_keys(manifest, manifest_fields, "manifest")
        required_values = {
            "schema_revision": 1,
            "engine": "dhlive",
            "renderer_mode": "dhlive_webview_v1",
            "package_format_version": 1,
        }
        for key, expected in required_values.items():
            if manifest.get(key) != expected:
                _fail(f"manifest {key} must be {expected!r}")
        if not isinstance(manifest["avatar_id"], str) or not manifest["avatar_id"]:
            _fail("manifest avatar_id is invalid")
        if not isinstance(manifest["package_version"], int) or manifest["package_version"] < 1:
            _fail("manifest package_version is invalid")
        fingerprint = manifest["build_fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            _fail("manifest build_fingerprint must be a sha256 hex digest")
        try:
            int(fingerprint, 16)
        except ValueError:
            _fail("manifest build_fingerprint must be hexadecimal")
        runtime = _require_keys(
            manifest["renderer_runtime"],
            {"name", "version", "min_sdk_version"},
            "renderer_runtime",
        )
        if runtime["name"] != "dhlive-mini2" or not all(
            isinstance(runtime[key], str) and runtime[key]
            for key in ("version", "min_sdk_version")
        ):
            _fail("renderer_runtime is invalid")
        preview = _require_keys(manifest["preview"], {"path"}, "preview")
        if preview["path"] != "preview/thumbnail.jpg":
            _fail("manifest preview path is invalid")
        geometry = _require_keys(
            manifest["geometry"], {"path", "frame_count", "compression"}, "geometry"
        )
        if geometry["path"] != "data/combined_data.json.gz":
            _fail("manifest geometry path is invalid")
        if geometry["compression"] != "gzip":
            _fail("manifest geometry compression must be gzip")
        if not isinstance(geometry["frame_count"], int) or geometry["frame_count"] < 1:
            _fail("manifest geometry frame_count is invalid")
        video = _require_keys(
            manifest["video"],
            {
                "path", "codec", "pixel_format", "width", "height", "fps",
                "frame_count", "duration_ms", "loop",
            },
            "video",
        )
        if video["path"] != "video/01.mp4":
            _fail("manifest video path is invalid")
        if video["codec"] != "h264" or video["pixel_format"] != "yuv420p":
            _fail("manifest video codec must be h264/yuv420p")
        for key in ("width", "height", "frame_count", "duration_ms"):
            if not isinstance(video[key], int) or video[key] < 1:
                _fail(f"manifest video {key} is invalid")
        if (
            not isinstance(video["fps"], (int, float))
            or isinstance(video["fps"], bool)
            or not math.isfinite(float(video["fps"]))
            or float(video["fps"]) <= 0
            or video["loop"] is not True
        ):
            _fail("manifest video timing is invalid")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 3:
            _fail("manifest files must contain exactly three entries")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
                _fail("manifest file entry has invalid fields")
            if not isinstance(item["size_bytes"], int) or item["size_bytes"] <= 0:
                _fail("manifest file size must be a positive integer")
            digest = item["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                _fail("manifest file sha256 is invalid")
            try:
                int(digest, 16)
            except ValueError:
                _fail("manifest file sha256 must be hexadecimal")

    @staticmethod
    def _validate_geometry(data: bytes) -> dict[str, Any]:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
                raw = _read_limited(stream, _MAX_GEOMETRY_BYTES, "geometry JSON")
            payload = json.loads(raw)
        except (gzip.BadGzipFile, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(f"geometry payload is invalid: {exc}")
        _assert_json_finite(payload)
        count = _geometry_frame_count(payload)
        if count <= 0:
            _fail("geometry payload contains no frames")
        return {"frame_count": count, "json_size_bytes": len(raw)}

    def _validate_video(
        self, data: bytes, expected: dict[str, Any]
    ) -> dict[str, Any]:
        probe = self._probe_video(data)
        codec = probe.get("codec_name")
        pixel_format = probe.get("pix_fmt")
        if codec != "h264" or pixel_format != "yuv420p":
            _fail(f"video must be h264/yuv420p, got {codec}/{pixel_format}")
        width = int(probe.get("width") or 0)
        height = int(probe.get("height") or 0)
        fps = _parse_rate(probe.get("avg_frame_rate") or "0")
        duration_seconds = float(probe.get("duration") or 0)
        raw_frame_count = probe.get("nb_read_frames") or probe.get("nb_frames")
        try:
            frame_count = int(raw_frame_count)
        except (TypeError, ValueError):
            frame_count = round(duration_seconds * fps)
        if width != expected.get("width") or height != expected.get("height"):
            _fail("ffprobe video dimensions do not match manifest")
        if frame_count != expected.get("frame_count"):
            _fail("ffprobe video frame count does not match manifest")
        if abs(fps - float(expected.get("fps") or 0)) > 0.01:
            _fail("ffprobe video fps does not match manifest")
        expected_duration_ms = int(expected.get("duration_ms") or 0)
        actual_duration_ms = round(duration_seconds * 1000)
        duration_tolerance_ms = max(80, round(1000 / fps))
        if abs(actual_duration_ms - expected_duration_ms) > duration_tolerance_ms:
            _fail("ffprobe video duration does not match manifest")
        return {
            "codec": codec,
            "pixel_format": pixel_format,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_ms": actual_duration_ms,
        }


dhlive_package_validator = DhLivePackageValidator()
avatar_package_validator_registry.register("dhlive", dhlive_package_validator)
