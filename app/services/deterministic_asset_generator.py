"""Deterministic asset generation abstraction (plan §7.2).

The old visual-engine semantic was ``photo + audio -> mp4`` (EchoMimic,
probabilistic). This module defines the replacement contract: render an
explicit list of states against a prepared portrait, deterministically.

Implementations: LivePortraitBuilderClient (via the liveportrait-builder
container). EchoMimic must NOT be adapted to this interface — the two
engines' inputs, outputs and quality semantics differ (plan §7.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PreparedPortrait:
    """A preprocessed, build-ready source portrait.

    Mirrors what portrait_preprocess_service produces; the inference
    crop (face_safe_768) is the LivePortrait input.
    """

    preprocess_token: str
    source_key: str            # MinIO key of face_safe_768 crop
    base_portrait_key: str
    canvas_mouth_roi: tuple[float, float, float, float]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateBankRef:
    """Versioned reference to the standard motion template bank."""

    revision: str
    sha256: str
    storage_key: str = ""     # MinIO key or /out path for the bank bundle


@dataclass(frozen=True)
class RenderStateSpec:
    """One state to render.

    Parametric mode (Gate T1 validation, no template bank needed):
        kind="parametric", lip_ratio in [0,1]
    Template mode (Phase 1+ production):
        kind="template", emotion, viseme, level 1..6 — activation level
        interpolates from the emotion base toward the target viseme
        template (plan §4.4).
    """

    kind: str                          # "parametric" | "template"
    lip_ratio: float | None = None
    emotion: str | None = None
    viseme: str | None = None
    level: int | None = None

    def validate(self) -> None:
        if self.kind == "parametric":
            if self.lip_ratio is None or not (0.0 <= self.lip_ratio <= 1.0):
                raise ValueError(f"parametric state needs lip_ratio in [0,1]: {self}")
        elif self.kind == "template":
            if not (self.emotion and self.viseme and self.level and 1 <= self.level <= 6):
                raise ValueError(f"template state needs emotion/viseme/level(1-6): {self}")
        else:
            raise ValueError(f"unknown state kind: {self.kind!r}")


@dataclass(frozen=True)
class RenderedState:
    """One rendered result. Images travel by path reference, not bytes."""

    index: int
    status: str                        # "ok" | "error"
    image_path: str = ""
    sha256: str = ""
    width: int = 0
    height: int = 0
    openness: float | None = None
    params: dict = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class RenderBatchResult:
    """Outcome of one render_batch call (idempotent per job_id)."""

    job_id: str
    states_requested: int
    states_ok: int
    results: tuple[RenderedState, ...] = ()
    elapsed_ms_total: int = 0

    @property
    def all_ok(self) -> bool:
        return self.states_ok == self.states_requested


@runtime_checkable
class DeterministicAssetGenerator(Protocol):
    """Plan §7.2 contract. Fail-closed: no silent fallbacks."""

    async def render_batch(
        self,
        source: PreparedPortrait,
        template_bank: TemplateBankRef,
        states: list[RenderStateSpec],
        *,
        job_id: str,
    ) -> RenderBatchResult: ...

    async def health(self) -> dict: ...


__all__ = [
    "PreparedPortrait",
    "TemplateBankRef",
    "RenderStateSpec",
    "RenderedState",
    "RenderBatchResult",
    "DeterministicAssetGenerator",
]
