"""v3 Portrait preprocessing pipeline.

Single entry point: ``preprocess_photo()`` runs the full pipeline:
EXIF fix → FaceLandmarker → quality gate → rembg segmentation →
CanvasTransform → 768×1024 canvas → 768×768 inference zone →
storage + preprocess_token.

Uses ``VisionModelRegistry`` for model access — no Haar Cascade,
no geometric ROI guessing, no opaque RGBA fallback.

See docs/v3视觉模型-Idle-ReferenceRenderer与测试遗留完整实施方案.md §6.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from PIL import Image, ImageFilter, ImageOps

from app.services.canvas_transform import CanvasTransform
from app.services.vision_model_registry import (
    FaceGeometry,
    Rect,
    SegmentationResult,
    VisionModelError,
    vision_registry,
)

logger = logging.getLogger(__name__)

CANVAS_W = 768
CANVAS_H = 1024
INFER_W = 768
INFER_H = 768
MIN_FACE_PX = 100
TOKEN_TTL_MINUTES = 30
PREPROCESS_VERSION = 3  # v3 = immutable per-asset hashes + real vision assets


class PreprocessError(Exception):
    def __init__(self, reason: str, code: str = "preprocess_failed"):
        self.reason = reason
        self.code = code
        super().__init__(reason)


@dataclass
class MouthROI:
    x: float
    y: float
    w: float
    h: float
    feather_px: int = 8


@dataclass
class PreprocessResult:
    accepted: bool
    quality_score: int
    warnings: list[str] = field(default_factory=list)
    face_count: int = 0
    canvas_profile: str = "portrait_3_4"
    normalized_width: int = CANVAS_W
    normalized_height: int = CANVAS_H
    preprocess_token: str = ""
    source_sha256: str = ""
    mouth_roi: MouthROI | None = None
    source_key: str = ""
    foreground_key: str = ""
    mask_key: str = ""
    base_canvas_key: str = ""
    inference_zone_key: str = ""
    metadata_key: str = ""
    preview_key: str = ""
    expires_at: str = ""


def _load_rgb(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _detect_face_and_geometry(rgb_pil: Image.Image) -> FaceGeometry:
    """Use VisionModelRegistry FaceLandmarker. Fails closed — no Haar fallback."""
    try:
        geo = vision_registry.detect_face(rgb_pil)
    except VisionModelError as exc:
        if "multiple_faces" in str(exc):
            raise PreprocessError("检测到多张人脸，请上传单人照片", "multiple_faces")
        raise PreprocessError(
            f"人脸检测服务不可用: {exc}",
            "vision_models_unavailable",
        ) from exc

    if geo is None:
        raise PreprocessError("未检测到人脸，请上传清晰的正面照片", "no_face")

    # Quality gate
    w, h = rgb_pil.size
    face_w_px = geo.face_bbox.w * w
    face_h_px = geo.face_bbox.h * h
    if face_w_px < MIN_FACE_PX or face_h_px < MIN_FACE_PX:
        raise PreprocessError(
            f"人脸区域太小（{int(face_w_px)}×{int(face_h_px)}px），"
            f"至少需要 {MIN_FACE_PX}px",
            "face_too_small",
        )
    if abs(geo.yaw_deg) > 35 or abs(geo.pitch_deg) > 35:
        raise PreprocessError(
            f"人脸角度过大 (yaw={geo.yaw_deg:.0f}°, pitch={geo.pitch_deg:.0f}°)，"
            f"请上传正面照片",
            "face_angle_too_large",
        )
    if geo.mouth_roi.area < 0.002:
        raise PreprocessError("嘴部区域过小", "mouth_roi_too_small")

    return geo


def _segment_person(image_bytes: bytes) -> SegmentationResult:
    """Run rembg via VisionModelRegistry. Fails closed — no opaque fallback."""
    try:
        result = vision_registry.segment(image_bytes)
    except VisionModelError as exc:
        raise PreprocessError(
            f"人物分割服务不可用: {exc}",
            "vision_models_unavailable",
        ) from exc
    except Exception as exc:
        raise PreprocessError(
            f"人物分割失败: {exc}",
            "person_segmentation_failed",
        ) from exc

    if result.foreground_ratio < 0.05 or result.foreground_ratio > 0.95:
        raise PreprocessError(
            f"分割结果异常 (foreground_ratio={result.foreground_ratio:.2f})，"
            f"请上传更清晰的人物照片",
            "segmentation_abnormal",
        )
    return result


def _compose_canvas(
    source_rgb: Image.Image,
    seg: SegmentationResult,
    transform: CanvasTransform,
    background_mode: str,
    background_bytes: bytes | None,
) -> Image.Image:
    """Compose the 768×1024 canvas using unified CanvasTransform."""
    # Scale foreground using same transform
    new_w = int(source_rgb.width * transform.scale)
    new_h = int(source_rgb.height * transform.scale)
    person_rgba = Image.open(io.BytesIO(seg.foreground_rgba_bytes)).convert("RGBA")
    person_scaled = person_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Mode determines base layer
    if background_mode == "transparent":
        canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    elif background_mode == "blur":
        bg = source_rgb.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=20))
        canvas = bg.convert("RGBA")
    elif background_mode == "replace" and background_bytes:
        bg = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        ratio = max(CANVAS_W / bg.width, CANVAS_H / bg.height)
        bg = bg.resize((int(bg.width * ratio), int(bg.height * ratio)), Image.LANCZOS)
        left = (bg.width - CANVAS_W) // 2
        top = (bg.height - CANVAS_H) // 2
        bg = bg.crop((left, top, left + CANVAS_W, top + CANVAS_H))
        canvas = bg.convert("RGBA")
    else:  # keep
        bg = source_rgb.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        canvas = bg.convert("RGBA")

    # Paste foreground using its alpha (no double-background)
    if background_mode == "keep":
        # Preserve the uploaded background. A softly blurred cover is used only
        # behind contain-padding; the original scaled image remains untouched.
        cover_ratio = max(CANVAS_W / source_rgb.width, CANVAS_H / source_rgb.height)
        cover = source_rgb.resize(
            (int(source_rgb.width * cover_ratio), int(source_rgb.height * cover_ratio)),
            Image.LANCZOS,
        )
        cover_left = max(0, (cover.width - CANVAS_W) // 2)
        cover_top = max(0, (cover.height - CANVAS_H) // 2)
        full = cover.crop(
            (cover_left, cover_top, cover_left + CANVAS_W, cover_top + CANVAS_H)
        ).filter(ImageFilter.GaussianBlur(radius=8)).convert("RGBA")
        source_scaled = source_rgb.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
        full.paste(source_scaled, (transform.offset_x, transform.offset_y))
        canvas = full
    else:
        mask = person_scaled.split()[-1]
        if mask.mode != "L":
            mask = mask.convert("L")
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        canvas.paste(person_scaled, (transform.offset_x, transform.offset_y), mask)

    return canvas


def _crop_inference_zone(canvas: Image.Image, face_bbox_canvas: Rect) -> tuple[Image.Image, int, int]:
    """Crop 768×768 from canvas centred on face upper-third.

    Returns (cropped_image, left_px, top_px) so inference ROI can be mapped
    back to canvas coordinates.
    """
    cx = int(face_bbox_canvas.cx * CANVAS_W)
    cy = int(face_bbox_canvas.cy * CANVAS_H)
    left = cx - INFER_W // 2
    top = cy - INFER_H // 3
    left = max(0, min(left, CANVAS_W - INFER_W))
    top = max(0, min(top, CANVAS_H - INFER_H))
    return canvas.crop((left, top, left + INFER_W, top + INFER_H)), left, top


def preprocess_photo(
    *,
    image_bytes: bytes,
    background_mode: str = "keep",
    background_bytes: bytes | None = None,
    storage_service,
    user_id: str,
) -> PreprocessResult:
    """Full v3 preprocessing pipeline. Raises PreprocessError on hard reject."""

    # This service is also called by admin retry jobs, so the validation must
    # live below the HTTP layer as well.
    from app.services.photo_validator import PhotoValidationError, validate_photo

    try:
        validate_photo(image_bytes)
    except PhotoValidationError as exc:
        raise PreprocessError(exc.reason, "invalid_photo") from exc

    # 1. Decode + EXIF
    rgb = _load_rgb(image_bytes)

    # 2. FaceLandmarker + quality gate
    geo = _detect_face_and_geometry(rgb)

    # 3. rembg segmentation
    seg = _segment_person(image_bytes)

    # 4. CanvasTransform (unified coordinate system)
    transform = CanvasTransform.fit_contain(
        rgb.width, rgb.height, CANVAS_W, CANVAS_H,
        face_cy_norm=geo.face_bbox.cy,
    )

    # Map ROIs through transform
    canvas_face_bbox = transform.map_rect(geo.face_bbox)
    canvas_mouth_roi = transform.map_rect(geo.mouth_roi)
    canvas_left_eye = transform.map_rect(geo.left_eye_roi)
    canvas_right_eye = transform.map_rect(geo.right_eye_roi)

    mouth_roi = MouthROI(
        x=canvas_mouth_roi.x, y=canvas_mouth_roi.y,
        w=canvas_mouth_roi.w, h=canvas_mouth_roi.h,
    )

    # 5. Compose canvas
    canvas = _compose_canvas(rgb, seg, transform, background_mode, background_bytes)

    # 6. Inference safe zone (now returns crop coordinates)
    inference, infer_left_px, infer_top_px = _crop_inference_zone(canvas, canvas_face_bbox)

    # 7. Quality scoring
    quality_score = 100
    warnings: list[str] = []
    if geo.yaw_deg > 15 or geo.pitch_deg > 15:
        quality_score -= 15
        warnings.append("轻微侧脸")
    if seg.foreground_ratio < 0.15:
        quality_score -= 10
        warnings.append("人物占比较小")

    # 8. Storage
    ts = datetime.now(timezone.utc)
    token = uuid4().hex
    prefix = f"preprocess/{user_id}/{token}"
    source_sha = hashlib.sha256(image_bytes).hexdigest()

    asset_sha256: dict[str, str] = {}

    def _store(key_suffix: str, img: Image.Image, fmt: str = "WEBP", quality: int = 90) -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=quality)
        content = buf.getvalue()
        key = f"{prefix}/{key_suffix}"
        storage_service.put_bytes(key=key, content=content, content_type=f"image/{fmt.lower()}")
        asset_sha256[key_suffix.split(".", 1)[0]] = hashlib.sha256(content).hexdigest()
        return key

    # Source: detect real format for correct extension
    source_img_check = Image.open(io.BytesIO(image_bytes))
    source_format = (source_img_check.format or "JPEG").upper()
    source_ext_map = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "jpg"}
    source_ext = source_ext_map.get(source_format, "jpg")
    source_ct_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    source_ct = source_ct_map.get(source_format, "image/jpeg")
    source_key = f"{prefix}/source.{source_ext}"
    storage_service.put_bytes(key=source_key, content=image_bytes, content_type=source_ct)
    asset_sha256["source"] = source_sha

    # Foreground + mask: resize to canvas size using same transform
    person_img = Image.open(io.BytesIO(seg.foreground_rgba_bytes)).convert("RGBA")
    person_canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    new_w = int(rgb.width * transform.scale)
    new_h = int(rgb.height * transform.scale)
    person_scaled = person_img.resize((new_w, new_h), Image.LANCZOS)
    mask_scaled = person_scaled.split()[-1]
    if mask_scaled.mode != "L":
        mask_scaled = mask_scaled.convert("L")
    person_canvas.paste(person_scaled, (transform.offset_x, transform.offset_y), mask_scaled)
    mask_canvas = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    mask_canvas.paste(mask_scaled, (transform.offset_x, transform.offset_y))
    fg_key = _store("foreground.webp", person_canvas)
    mask_key = _store("person_mask.png", mask_canvas, fmt="PNG")

    # Base canvas
    if background_mode == "transparent":
        base_key = _store("base_portrait.webp", canvas, fmt="WEBP")
    else:
        base_key = _store("base_portrait.webp", canvas.convert("RGB"), fmt="WEBP")

    infer_key = _store("face_safe_768.webp", inference.convert("RGB") if inference.mode == "RGBA" else inference)

    preview = (canvas.convert("RGB") if canvas.mode == "RGBA" else canvas).resize((384, 512), Image.LANCZOS)
    preview_key = _store("preview.webp", preview, quality=80)

    # Normalise storage-key names to the package contract. The dictionary is
    # persisted in metadata and rechecked by the worker before GPU inference.
    asset_sha256 = {
        "source": asset_sha256["source"],
        "foreground": asset_sha256["foreground"],
        "mask": asset_sha256["person_mask"],
        "base": asset_sha256["base_portrait"],
        "inference": asset_sha256["face_safe_768"],
        "preview": asset_sha256["preview"],
    }

    # 9. Metadata JSON
    meta = {
        "preprocess_token": token,
        "preprocess_version": PREPROCESS_VERSION,
        "owner_user_id": user_id,
        "source_sha256": source_sha,
        "canvas_profile": "portrait_3_4",
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "inference_width": INFER_W,
        "inference_height": INFER_H,
        "background_mode": background_mode,
        "quality_score": quality_score,
        "warnings": warnings,
        "face_count": 1,
        "vision_pipeline": {
            "real_landmarks": True,
            "real_segmentation": True,
            "face_landmarker_model": seg.model_name if hasattr(seg, "model_name") else "face_landmarker",
            "segmentation_model": seg.model_name,
        },
        "geometry": {
            "source_face_bbox_norm": [geo.face_bbox.x, geo.face_bbox.y, geo.face_bbox.w, geo.face_bbox.h],
            "canvas_face_bbox_norm": [canvas_face_bbox.x, canvas_face_bbox.y, canvas_face_bbox.w, canvas_face_bbox.h],
            "canvas_mouth_roi_norm": [canvas_mouth_roi.x, canvas_mouth_roi.y, canvas_mouth_roi.w, canvas_mouth_roi.h],
            "canvas_left_eye_roi_norm": [canvas_left_eye.x, canvas_left_eye.y, canvas_left_eye.w, canvas_left_eye.h],
            "canvas_right_eye_roi_norm": [canvas_right_eye.x, canvas_right_eye.y, canvas_right_eye.w, canvas_right_eye.h],
            "yaw_deg": geo.yaw_deg,
            "pitch_deg": geo.pitch_deg,
            "roll_deg": geo.roll_deg,
            "landmark_count": len(geo.landmarks),
            "inference_crop": {
                "left_px": infer_left_px,
                "top_px": infer_top_px,
                "width": INFER_W,
                "height": INFER_H,
            },
        },
        "normalized_asset_keys": {
            "source": source_key,
            "foreground": fg_key,
            "mask": mask_key,
            "base": base_key,
            "inference": infer_key,
            "preview": preview_key,
        },
        "asset_sha256": asset_sha256,
        "expires_at": (ts + timedelta(minutes=TOKEN_TTL_MINUTES)).isoformat(),
    }
    meta_key = f"{prefix}/preprocess.json"
    meta["metadata_key"] = meta_key
    meta["normalized_asset_keys"]["metadata"] = meta_key
    storage_service.put_bytes(key=meta_key, content=json.dumps(meta).encode(), content_type="application/json")

    return PreprocessResult(
        accepted=True,
        quality_score=quality_score,
        warnings=warnings,
        face_count=1,
        canvas_profile="portrait_3_4",
        normalized_width=CANVAS_W,
        normalized_height=CANVAS_H,
        preprocess_token=token,
        source_sha256=source_sha,
        mouth_roi=mouth_roi,
        source_key=source_key,
        foreground_key=fg_key,
        mask_key=mask_key,
        base_canvas_key=base_key,
        inference_zone_key=infer_key,
        metadata_key=meta_key,
        preview_key=preview_key,
        expires_at=meta["expires_at"],
    )


def read_metadata(storage_service, token: str, user_id: str) -> dict | None:
    meta_key = f"preprocess/{user_id}/{token}/preprocess.json"
    try:
        raw = storage_service.get_bytes(meta_key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        logger.debug("preprocess_metadata_read_failed token=%s", token, exc_info=True)
        return None
