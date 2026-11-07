"""Idle asset builder for v3 packages.

Blink: Local RGBA eye patches from FaceLandmarker-guided detection or
       deterministic eyelid deformation.

Breathing: Local RGBA torso patches via cv2.remap weighted deformation.
           No whole-person shift, no RGB output.

Both output ValidatedIdleAsset with real quality checks.
"""
from __future__ import annotations

import hashlib
import io
import logging
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from app.domain.idle_assets import IdleAssetQuality, ValidatedIdleAsset

logger = logging.getLogger(__name__)

BLINK_FRAME_COUNT = 9  # open→close→open: 1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0
BLINK_FPS = 25
BREATHING_FRAME_COUNT = 75
BREATHING_FPS = 25
BREATHING_AMPLITUDE_PX = 2
CANVAS_W = 768
CANVAS_H = 1024
FEATHER_PX = 6

_LEFT_EYE_EAR = [33, 159, 158, 133, 145, 153]
_RIGHT_EYE_EAR = [362, 386, 385, 263, 374, 380]


class IdleAssetError(RuntimeError):
    pass


def _make_feathered_alpha(w: int, h: int, feather: int = FEATHER_PX) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    inset = max(2, feather // 2)
    radius = min(w, h) // 4
    draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=radius, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def _encode_rgba_patch(img: Image.Image, mask: Image.Image, fmt: str = "webp") -> bytes:
    result = img.copy()
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    result.putalpha(mask)
    buf = io.BytesIO()
    save_fmt = "PNG" if fmt.lower() == "png" else "WEBP"
    result.save(buf, format=save_fmt, quality=90 if save_fmt == "WEBP" else None)
    return buf.getvalue()


def _inspect_alpha_frames(frames: list[bytes], fmt: str) -> tuple[bool, float]:
    expected_format = "PNG" if fmt.lower() == "png" else "WEBP"
    minimum_opaque_coverage = 1.0
    for index, data in enumerate(frames):
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if (img.format or "").upper() != expected_format:
                raise IdleAssetError(
                    f"frame {index} format mismatch: expected {expected_format}, got {img.format}"
                )
            if "A" not in img.getbands():
                return False, 0.0
            alpha = np.asarray(img.convert("RGBA").getchannel("A"), dtype=np.uint8)
            transparent_ratio = float(np.mean(alpha < 250))
            visible_ratio = float(np.mean(alpha > 16))
            if transparent_ratio < 0.01 or visible_ratio < 0.05:
                return False, 0.0
            minimum_opaque_coverage = min(
                minimum_opaque_coverage,
                float(np.mean(alpha >= 128)),
            )
    return True, minimum_opaque_coverage


def _eye_aspect_ratio(landmarks, indices: list[int]) -> float:
    pts = [landmarks[index] for index in indices]

    def distance(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    horizontal = distance(pts[0], pts[3])
    if horizontal <= 1e-6:
        raise IdleAssetError("eye horizontal span is zero")
    return (distance(pts[1], pts[5]) + distance(pts[2], pts[4])) / (2 * horizontal)


def build_blink_from_landmarks(
    base_image: bytes,
    face_landmarks,
    fmt: str = "webp",
) -> ValidatedIdleAsset:
    """Generate deterministic blink via landmark-guided eyelid deformation.

    Each frame uses a different openness level (no duplication).
    Output is a local RGBA eye patch, not a full frame.
    """
    if len(face_landmarks) <= max(_LEFT_EYE_EAR + _RIGHT_EYE_EAR):
        raise IdleAssetError("insufficient landmarks for blink generation")

    img = Image.open(io.BytesIO(base_image))
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    all_eye_indices = _LEFT_EYE_EAR + _RIGHT_EYE_EAR
    eye_pts = [face_landmarks[i] for i in all_eye_indices]
    xs = [p.x for p in eye_pts]
    ys = [p.y for p in eye_pts]

    pad_x = (max(xs) - min(xs)) * 0.3
    pad_y = (max(ys) - min(ys)) * 0.8
    roi_x = max(0, min(xs) - pad_x)
    roi_y = max(0, min(ys) - pad_y)
    roi_w = min(1.0, max(xs) + pad_x) - roi_x
    roi_h = min(1.0, max(ys) + pad_y) - roi_y

    px_x = int(roi_x * img.width)
    px_y = int(roi_y * img.height)
    px_w = max(1, int(roi_w * img.width))
    px_h = max(1, int(roi_h * img.height))
    eye_patch_base = img.crop((px_x, px_y, px_x + px_w, px_y + px_h))

    openness_curve = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    mask = _make_feathered_alpha(px_w, px_h)

    frames: list[bytes] = []
    frame_hashes: list[str] = []

    eye_specs: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    for indices in (_LEFT_EYE_EAR, _RIGHT_EYE_EAR):
        points = [face_landmarks[index] for index in indices]
        eye_x1 = max(0, int(min(point.x for point in points) * img.width) - px_x - 3)
        eye_x2 = min(px_w, int(max(point.x for point in points) * img.width) - px_x + 3)
        eye_y1 = max(0, int(min(point.y for point in points) * img.height) - px_y - 3)
        eye_y2 = min(px_h, int(max(point.y for point in points) * img.height) - px_y + 3)
        if eye_x2 <= eye_x1 or eye_y2 <= eye_y1:
            raise IdleAssetError("eye ROI is empty")
        sample_y1 = max(0, eye_y1 - max(2, eye_y2 - eye_y1))
        sample = eye_patch_base.crop((eye_x1, sample_y1, eye_x2, eye_y1))
        if sample.width == 0 or sample.height == 0:
            sample = eye_patch_base.crop((eye_x1, eye_y1, eye_x2, eye_y2))
        mean = ImageStat.Stat(sample.convert("RGB")).mean
        skin = (int(mean[0]), int(mean[1]), int(mean[2]), 255)
        eye_specs.append((eye_x1, eye_y1, eye_x2, eye_y2, skin))

    for openness in openness_curve:
        if openness >= 1.0:
            patch = eye_patch_base.copy()
        else:
            patch = eye_patch_base.copy()
            overlay = Image.new("RGBA", patch.size, (0, 0, 0, 0))
            drawer = ImageDraw.Draw(overlay, "RGBA")
            closure = 1.0 - openness
            for x1, y1, x2, y2, skin in eye_specs:
                center_y = (y1 + y2) // 2
                half_cover = max(1, int((y2 - y1) * closure / 2))
                fill = (skin[0], skin[1], skin[2], int(245 * closure))
                drawer.rounded_rectangle(
                    (x1, center_y - half_cover, x2, center_y + half_cover),
                    radius=max(1, (y2 - y1) // 3),
                    fill=fill,
                )
                if closure >= 0.7:
                    eyelid = (
                        max(0, skin[0] - 45),
                        max(0, skin[1] - 45),
                        max(0, skin[2] - 45),
                        int(180 * closure),
                    )
                    drawer.line((x1 + 2, center_y, x2 - 2, center_y), fill=eyelid, width=1)
            patch = Image.alpha_composite(patch, overlay)

        encoded = _encode_rgba_patch(patch, mask, fmt)
        frames.append(encoded)
        frame_hashes.append(hashlib.sha256(encoded).hexdigest())

    reasons: list[str] = []
    alpha_valid, _ = _inspect_alpha_frames(frames, fmt)
    if not alpha_valid:
        reasons.append("ALPHA_INVALID")

    # Measure the generated result after compositing it back onto the full
    # canvas. This avoids claiming blink quality from the requested curve.
    measured_ears: list[float] = []
    try:
        from app.services.vision_model_registry import vision_registry

        for encoded in frames:
            patch = Image.open(io.BytesIO(encoded)).convert("RGBA")
            composed = img.copy()
            composed.alpha_composite(patch, (px_x, px_y))
            geometry = vision_registry.detect_face(composed.convert("RGB"))
            if geometry is None:
                raise IdleAssetError("face landmark missing on generated blink frame")
            left_ear = _eye_aspect_ratio(geometry.landmarks, _LEFT_EYE_EAR)
            right_ear = _eye_aspect_ratio(geometry.landmarks, _RIGHT_EYE_EAR)
            measured_ears.append((left_ear + right_ear) / 2)
    except Exception as exc:
        raise IdleAssetError(f"blink landmark validation failed: {exc}") from exc

    ear_min = min(measured_ears)
    ear_max = max(measured_ears)
    if ear_max <= 0 or ear_min / ear_max > 0.65:
        reasons.append(f"BLINK_RANGE_INSUFFICIENT: min={ear_min:.4f} max={ear_max:.4f}")

    if len(frames) >= 2:
        f0 = np.array(Image.open(io.BytesIO(frames[0])).convert("L"), dtype=np.float32)
        fn = np.array(Image.open(io.BytesIO(frames[-1])).convert("L"), dtype=np.float32)
        diff = float(np.mean(np.abs(f0 - fn)))
        if diff > 15.0:
            reasons.append(f"FIRST_LAST_DIFF_HIGH: {diff:.1f}")
    else:
        diff = 999.0

    if frame_hashes[0] != frame_hashes[-1]:
        reasons.append("BLINK_LOOP_ENDPOINT_MISMATCH")
    # Symmetric open->close->open sequences intentionally repeat mirrored
    # levels. Require five distinct stages, not nine globally unique frames.
    if len(set(frame_hashes)) < (BLINK_FRAME_COUNT // 2 + 1):
        reasons.append("INSUFFICIENT_BLINK_VARIATION")

    quality = IdleAssetQuality(
        passed=len(reasons) == 0,
        failure_reasons=tuple(reasons),
        frame_count=len(frames),
        has_alpha=alpha_valid,
        alpha_valid=alpha_valid,
        ear_min=ear_min,
    )

    if not quality.passed:
        raise IdleAssetError(f"blink quality failed: {'; '.join(reasons)}")

    return ValidatedIdleAsset(
        kind="blink",
        frames=tuple(frames),
        frame_sha256s=tuple(frame_hashes),
        canvas_roi=(roi_x, roi_y, roi_w, roi_h),
        patch_width=px_w,
        patch_height=px_h,
        fps=BLINK_FPS,
        loop=False,
        has_alpha=alpha_valid,
        generator="landmark_guided_deformation",
        quality=quality,
    )


def build_breathing_torso(
    base_image: bytes,
    person_mask: bytes,
    face_bbox_norm: tuple[float, float, float, float],
    fmt: str = "webp",
) -> ValidatedIdleAsset:
    """Generate breathing via torso-only local deformation.

    No whole-person shift. Output is RGBA patch.
    Uses phase = 2*pi*i/(N-1) for seamless loop.
    """
    base = Image.open(io.BytesIO(base_image))
    if base.mode != "RGBA":
        base = base.convert("RGBA")

    mask_img = Image.open(io.BytesIO(person_mask)).convert("L")
    if mask_img.size != base.size:
        mask_img = mask_img.resize(base.size, Image.LANCZOS)

    fx, fy, fw, fh = face_bbox_norm
    torso_top = min(1.0, fy + fh * 0.8)
    torso_bottom = min(1.0, torso_top + 0.35)
    torso_left = max(0.0, fx - fw * 0.5)
    torso_right = min(1.0, fx + fw + fw * 0.5)
    torso_w = torso_right - torso_left
    torso_h = torso_bottom - torso_top

    px_x = int(torso_left * base.width)
    px_y = int(torso_top * base.height)
    px_w = max(1, int(torso_w * base.width))
    px_h = max(1, int(torso_h * base.height))

    torso_base = base.crop((px_x, px_y, px_x + px_w, px_y + px_h))
    torso_mask = mask_img.crop((px_x, px_y, px_x + px_w, px_y + px_h))
    mask_array = np.asarray(torso_mask, dtype=np.uint8)
    foreground_ratio = float(np.mean(mask_array > 16))
    if foreground_ratio < 0.05:
        raise IdleAssetError(
            f"person mask has insufficient torso coverage: {foreground_ratio:.3f}"
        )

    try:
        import cv2
    except ImportError as exc:
        raise IdleAssetError("OpenCV is required for breathing deformation") from exc

    frames: list[bytes] = []
    frame_hashes: list[str] = []
    feather = _make_feathered_alpha(px_w, px_h)

    for i in range(BREATHING_FRAME_COUNT):
        phase = 2 * math.pi * i / max(1, BREATHING_FRAME_COUNT - 1)
        displacement = BREATHING_AMPLITUDE_PX * math.sin(phase)

        arr = np.array(torso_base, dtype=np.float32)
        mask_arr = np.array(torso_mask, dtype=np.float32)
        h, w = arr.shape[:2]
        map_y, map_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx = w / 2
        weight = np.exp(-((np.arange(w) - cx) ** 2) / (2 * (w / 4) ** 2))
        weight = np.tile(weight, (h, 1))
        map_y = map_y + displacement * weight
        remapped = cv2.remap(
            arr,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        remapped_mask = cv2.remap(
            mask_arr,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        patch = Image.fromarray(remapped.astype(np.uint8), "RGBA")
        person_alpha = Image.fromarray(remapped_mask.astype(np.uint8), "L")
        combined_alpha = ImageChops.multiply(feather, person_alpha)

        encoded = _encode_rgba_patch(patch, combined_alpha, fmt)
        frames.append(encoded)
        frame_hashes.append(hashlib.sha256(encoded).hexdigest())

    reasons: list[str] = []
    alpha_valid, _ = _inspect_alpha_frames(frames, fmt)
    if not alpha_valid:
        reasons.append("ALPHA_INVALID")
    diff = 999.0
    if len(frames) >= 2:
        f0 = np.array(Image.open(io.BytesIO(frames[0])).convert("L"), dtype=np.float32)
        fn = np.array(Image.open(io.BytesIO(frames[-1])).convert("L"), dtype=np.float32)
        diff = float(np.mean(np.abs(f0 - fn)))
        if diff > 3.0:
            reasons.append(f"LOOP_SEAM_DIFF_HIGH: {diff:.2f}")

    if len(set(frame_hashes)) < BREATHING_FRAME_COUNT * 0.3:
        reasons.append("INSUFFICIENT_VARIATION")

    quality = IdleAssetQuality(
        passed=len(reasons) == 0,
        failure_reasons=tuple(reasons),
        frame_count=len(frames),
        has_alpha=alpha_valid,
        alpha_valid=alpha_valid,
        first_last_diff=diff,
    )

    if not quality.passed:
        raise IdleAssetError(f"breathing quality failed: {'; '.join(reasons)}")

    return ValidatedIdleAsset(
        kind="breathing",
        frames=tuple(frames),
        frame_sha256s=tuple(frame_hashes),
        canvas_roi=(torso_left, torso_top, torso_w, torso_h),
        patch_width=px_w,
        patch_height=px_h,
        fps=BREATHING_FPS,
        loop=True,
        has_alpha=alpha_valid,
        generator="torso_weighted_remap",
        quality=quality,
    )


__all__ = [
    "IdleAssetError",
    "build_blink_from_landmarks",
    "build_breathing_torso",
]
