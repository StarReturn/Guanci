"""Photo validator - lightweight pre-check before EchoMimic build.

Checks:
- File format (jpg/png/webp)
- File size (100KB - 20MB)
- Resolution (≥256×256, ≤4096×4096)
- Decodable by Pillow
- Aspect ratio (0.5 - 2.0, reject extreme panoramas)

Does NOT do face detection. Face detection is a mandatory part of
``photo-preflight``. This pre-check rejects malformed or unsupported image
containers before any vision model is invoked.
"""
from __future__ import annotations

import io


class PhotoValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def validate_photo(image_bytes: bytes, content_type: str = "") -> dict:
    """Return decoded image facts or raise ``PhotoValidationError``.

    The decoded container is authoritative. Client-provided MIME values are
    not used to choose a storage extension because mobile clients commonly
    omit or mislabel them.
    """
    if len(image_bytes) < 50_000:
        raise PhotoValidationError("照片太小（<100KB），请上传更清晰的照片")
    if len(image_bytes) > 20_000_000:
        raise PhotoValidationError("照片太大（>20MB），请压缩后上传")

    try:
        from PIL import Image
    except ImportError as exc:
        raise PhotoValidationError("图片校验组件不可用") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            w, h = img.size
            actual_format = (img.format or "").upper()
    except Exception as exc:
        raise PhotoValidationError("无法解析图片，请上传 JPG/PNG/WebP 格式") from exc

    formats = {
        "JPEG": ("jpg", "image/jpeg"),
        "PNG": ("png", "image/png"),
        "WEBP": ("webp", "image/webp"),
    }
    if actual_format not in formats:
        raise PhotoValidationError(
            f"不支持的图片编码 {actual_format or 'unknown'}，请上传 JPG/PNG/WebP"
        )

    if w < 256 or h < 256:
        raise PhotoValidationError(f"分辨率太低（{w}×{h}），至少 256×256")
    if w > 4096 or h > 4096:
        raise PhotoValidationError(f"分辨率太高（{w}×{h}），最多 4096×4096")

    ratio = w / h
    if ratio < 0.5 or ratio > 2.0:
        raise PhotoValidationError(f"宽高比异常（{ratio:.1f}:1），请用正常比例的照片")

    extension, actual_mime = formats[actual_format]
    return {
        "width": w,
        "height": h,
        "format": actual_format,
        "extension": extension,
        "content_type": actual_mime,
        "declared_content_type": content_type,
    }
