"""Vision model registry for v3 portrait preprocessing.

Manages lifecycle of MediaPipe FaceLandmarker and rembg segmentation
session. Models are loaded ONCE from read-only mounts under
``settings.vision_model_root`` and reused across requests.

Production constraints:
  - No runtime model downloads.
  - SHA256 must match ``vision-models.lock.json``.
  - Missing or mismatched models → startup failure (fail closed).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class FaceGeometry:
    """Output of FaceLandmarker, normalised to [0, 1]."""
    landmarks: tuple[Point, ...]
    face_bbox: Rect
    mouth_roi: Rect
    left_eye_roi: Rect
    right_eye_roi: Rect
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    confidence: float


@dataclass(frozen=True)
class SegmentationResult:
    foreground_rgba_bytes: bytes
    alpha_mask_bytes: bytes
    width: int
    height: int
    model_name: str
    model_sha256: str
    foreground_ratio: float


# ---------------------------------------------------------------------------
# MediaPipe landmark index sets (478-point model)
# ---------------------------------------------------------------------------

# Outer + inner lips (MediaPipe Face Mesh canonical indices)
_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

_LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]

_RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# Face oval (for bounding box)
_FACE_OVAL = [
    10, 151, 9, 8, 12, 55, 65, 52, 53, 46,
    127, 234, 93, 132, 58, 172, 136, 150, 149, 176,
    148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454, 356, 389, 251, 284, 332, 297, 338, 10,
]


def _bbox_from_indices(landmarks: list, indices: list[int], padding: float = 0.0) -> Rect:
    pts = [landmarks[i] for i in indices if i < len(landmarks)]
    if not pts:
        return Rect(0, 0, 0, 0)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    x -= padding * w
    y -= padding * h
    w += 2 * padding * w
    h += 2 * padding * h
    return Rect(
        max(0.0, x), max(0.0, y),
        min(1.0, x + w) - max(0.0, x),
        min(1.0, y + h) - max(0.0, y),
    )


def _estimate_pose(landmarks: list) -> tuple[float, float, float]:
    """Estimate head yaw/pitch/roll from key landmarks (degrees)."""
    if len(landmarks) < 200:
        return 0.0, 0.0, 0.0
    # Use nose tip (1), left ear (234), right ear (454) for yaw
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    mid_x = (left_ear.x + right_ear.x) / 2
    yaw = (nose.x - mid_x) * 180

    # Pitch from the eye line's position within the forehead(10)-chin(152)
    # span. Scale-invariant ratio: eyes sit at ~45% of face height for a
    # neutral head; tilting up/down moves the ratio. (The old absolute
    # y-difference formula returned ~-110° for every upright face.)
    forehead = landmarks[10]
    chin = landmarks[152]
    eye_mid_y = (landmarks[33].y + landmarks[263].y) / 2
    upper = eye_mid_y - forehead.y
    lower = chin.y - eye_mid_y
    ratio = upper / max(1e-6, upper + lower)
    pitch = (ratio - 0.45) * 180

    # Roll from eye line
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    import math
    roll = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
    return yaw, pitch, roll


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class VisionModelError(RuntimeError):
    pass


class VisionModelRegistry:
    """Thread-safe singleton managing FaceLandmarker and rembg session.

    Call ``load()`` once at startup. Subsequent ``face_landmarker()`` and
    ``segmentation_session()`` calls return cached instances.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._face_landmarker: Any = None
        self._seg_session: Any = None
        self._lock_data: dict = {}
        self._model_root: Path = Path("/models/vision")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_root: str | Path | None = None) -> None:
        """Load models and verify SHA256. Raises VisionModelError on failure."""
        with self._lock:
            if self._loaded:
                return
            root = Path(model_root or os.environ.get("VISION_MODEL_ROOT", "/models/vision"))
            self._model_root = root
            lock_path = root / "vision-models.lock.json"

            if not lock_path.is_file():
                raise VisionModelError(
                    f"vision-models.lock.json not found at {lock_path}; "
                    f"cannot start without verified models"
                )

            with open(lock_path) as f:
                self._lock_data = json.load(f)

            self._load_face_landmarker(root)
            self._load_segmentation(root)
            self._loaded = True
            logger.info(
                "vision_models_loaded root=%s face_landmarker=%s segmentation=%s",
                root,
                self._lock_data.get("models", {}).get("face_landmarker", {}).get("name"),
                self._lock_data.get("models", {}).get("person_segmentation", {}).get("name"),
            )

    def _verify_sha256(self, path: Path, expected_sha: str) -> None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise VisionModelError(
                f"SHA256 mismatch for {path.name}: expected {expected_sha[:16]}..., "
                f"got {actual[:16]}..."
            )

    def _load_face_landmarker(self, root: Path) -> None:
        info = self._lock_data.get("models", {}).get("face_landmarker", {})
        filename = info.get("filename", "face_landmarker.task")
        expected_sha = info.get("sha256", "")
        path = root / "mediapipe" / filename

        if not path.is_file():
            raise VisionModelError(f"FaceLandmarker model not found: {path}")

        if expected_sha:
            self._verify_sha256(path, expected_sha)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=str(path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise VisionModelError(f"FaceLandmarker init failed: {exc}") from exc

    def _load_segmentation(self, root: Path) -> None:
        info = self._lock_data.get("models", {}).get("person_segmentation", {})
        filename = info.get("filename", "birefnet-portrait.onnx")
        expected_sha = info.get("sha256", "")
        path = root / "rembg" / filename

        if not path.is_file():
            raise VisionModelError(f"Segmentation model not found: {path}")

        if expected_sha:
            self._verify_sha256(path, expected_sha)

        try:
            from rembg import new_session

            model_name = info.get("name", "birefnet-portrait")
            # rembg expects the model in its cache dir; we symlink instead
            cache_dir = Path.home() / ".u2net"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / filename
            if not cache_path.exists():
                os.symlink(path, cache_path)

            self._seg_session = new_session(model_name)
        except Exception as exc:
            raise VisionModelError(f"rembg init failed: {exc}") from exc

    def detect_face(self, rgb_pil) -> FaceGeometry | None:
        """Run FaceLandmarker on a PIL RGB image. Returns FaceGeometry or None."""
        if not self._loaded:
            self.load()
        if not self._loaded or self._face_landmarker is None:
            raise VisionModelError("vision models not loaded")

        import numpy as np
        import mediapipe as mp

        rgb = np.array(rgb_pil)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None
        if len(result.face_landmarks) > 1:
            raise VisionModelError("multiple_faces_detected")

        raw = result.face_landmarks[0]
        landmarks = tuple(Point(lm.x, lm.y) for lm in raw)

        face_bbox = _bbox_from_indices(list(landmarks), _FACE_OVAL, padding=0.05)
        mouth_roi = _bbox_from_indices(list(landmarks), _LIP_INDICES, padding=0.15)
        left_eye = _bbox_from_indices(list(landmarks), _LEFT_EYE_INDICES, padding=0.2)
        right_eye = _bbox_from_indices(list(landmarks), _RIGHT_EYE_INDICES, padding=0.2)
        yaw, pitch, roll = _estimate_pose(list(landmarks))

        return FaceGeometry(
            landmarks=landmarks,
            face_bbox=face_bbox,
            mouth_roi=mouth_roi,
            left_eye_roi=left_eye,
            right_eye_roi=right_eye,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            confidence=1.0,
        )

    def segment(self, image_bytes: bytes) -> SegmentationResult:
        """Run rembg person segmentation. Returns SegmentationResult."""
        if not self._loaded:
            self.load()
        if not self._loaded or self._seg_session is None:
            raise VisionModelError("vision models not loaded")

        from rembg import remove
        from PIL import Image
        import io

        rgba_bytes = remove(image_bytes, session=self._seg_session)
        img = Image.open(io.BytesIO(rgba_bytes))
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        import numpy as np
        alpha = np.array(img.split()[-1])
        foreground_ratio = float(alpha[alpha > 128].size) / alpha.size

        # Serialize alpha mask as PNG
        mask_buf = io.BytesIO()
        img.split()[-1].save(mask_buf, format="PNG")

        body_buf = io.BytesIO()
        img.save(body_buf, format="WEBP", quality=90)

        info = self._lock_data.get("models", {}).get("person_segmentation", {})
        return SegmentationResult(
            foreground_rgba_bytes=body_buf.getvalue(),
            alpha_mask_bytes=mask_buf.getvalue(),
            width=img.width,
            height=img.height,
            model_name=info.get("name", "unknown"),
            model_sha256=info.get("sha256", ""),
            foreground_ratio=foreground_ratio,
        )

    def close(self) -> None:
        with self._lock:
            if self._face_landmarker is not None:
                self._face_landmarker.close()
                self._face_landmarker = None
            self._seg_session = None
            self._loaded = False


# Global instance
vision_registry = VisionModelRegistry()


__all__ = [
    "FaceGeometry",
    "Point",
    "Rect",
    "SegmentationResult",
    "VisionModelError",
    "VisionModelRegistry",
    "vision_registry",
]
