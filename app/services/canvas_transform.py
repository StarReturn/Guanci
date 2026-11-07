"""CanvasTransform: unified coordinate mapping from source image to 3:4 canvas.

All layers (background, foreground, alpha mask, face bbox, mouth ROI, eye ROIs)
must use the same CanvasTransform instance to avoid drift between layers.

See docs/v3视觉模型 §6.6.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.vision_model_registry import Rect


@dataclass(frozen=True)
class CanvasTransform:
    """Immutable mapping from source image coordinates to canvas coordinates.

    Attributes:
        src_w, src_h: source image dimensions in pixels.
        canvas_w, canvas_h: target canvas dimensions (768×1024).
        scale: uniform scale factor applied to source before pasting.
        offset_x, offset_y: paste offset on canvas (pixels).
    """
    src_w: int
    src_h: int
    canvas_w: int
    canvas_h: int
    scale: float
    offset_x: int
    offset_y: int

    @classmethod
    def fit_contain(
        cls, src_w: int, src_h: int, canvas_w: int = 768, canvas_h: int = 1024,
        *, face_cy_norm: float = 0.3,
    ) -> CanvasTransform:
        """Scale source to fit within canvas, centred horizontally,
        face at ~30% from top (upper third).
        """
        scale = min(canvas_w / src_w, canvas_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        offset_x = (canvas_w - new_w) // 2
        # Bias face toward upper third
        offset_y = max(0, min(canvas_h - new_h, int(canvas_h * face_cy_norm * 2 - new_h / 2)))
        return cls(src_w, src_h, canvas_w, canvas_h, scale, offset_x, offset_y)

    def map_rect(self, r: Rect) -> Rect:
        """Map a normalised [0,1] rect from source space to canvas space."""
        cx = r.x * self.src_w * self.scale + self.offset_x
        cy = r.y * self.src_h * self.scale + self.offset_y
        cw = r.w * self.src_w * self.scale
        ch = r.h * self.src_h * self.scale
        # Clamp to canvas
        cx = max(0, min(cx, self.canvas_w))
        cy = max(0, min(cy, self.canvas_h))
        cw = max(0, min(cw, self.canvas_w - cx))
        ch = max(0, min(ch, self.canvas_h - cy))
        # Normalise to [0,1] relative to canvas
        return Rect(
            cx / self.canvas_w,
            cy / self.canvas_h,
            cw / self.canvas_w,
            ch / self.canvas_h,
        )

    def map_point_to_pixel(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        """Map a normalised source-space point to canvas pixel coordinates."""
        px = int(x_norm * self.src_w * self.scale + self.offset_x)
        py = int(y_norm * self.src_h * self.scale + self.offset_y)
        return px, py


__all__ = ["CanvasTransform"]
