"""Strong-typed idle asset domain types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IdleAssetQuality:
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    frame_count: int = 0
    has_alpha: bool = False
    alpha_valid: bool = False
    first_last_diff: float = 0.0  # for breathing loop check
    ear_min: float = 0.0  # for blink


@dataclass(frozen=True)
class ValidatedIdleAsset:
    kind: Literal["blink", "breathing"]
    frames: tuple[bytes, ...]
    frame_sha256s: tuple[str, ...]
    canvas_roi: tuple[float, float, float, float]  # x, y, w, h normalised
    patch_width: int
    patch_height: int
    fps: int
    loop: bool
    has_alpha: bool
    generator: str
    quality: IdleAssetQuality


__all__ = ["IdleAssetQuality", "ValidatedIdleAsset"]
