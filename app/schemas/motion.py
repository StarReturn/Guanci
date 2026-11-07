"""mobile_keyframe_v3 motion protocol schemas.

Defines the ``motion_chunk`` WebSocket message that the server streams to
the mobile client. The chunk carries viseme + openness timeline plus an
energy envelope, all expressed in PCM sample positions of the audio that
will follow in the immediately subsequent binary frame.

Design invariants (see docs/拼帧数字人实时对话与3比4画布完整调整方案.md §7):

- Time is always expressed as integer ``sample`` offsets against a single
  ``audio_id``. Milliseconds are forbidden as primary coordinate because
  per-chunk rounding accumulates drift on long replies.
- ``motion_chunk`` MUST arrive at the client before the PCM bytes covering
  ``[valid_from_sample, valid_to_sample)`` become playable. Server is
  responsible for the lead buffer (``settings.keyframe_motion_lead_ms``).
- ``segments`` carry both viseme (mouth shape category) and openness
  (5-7 intensity levels). The client selects a mouth patch by
  ``(viseme, level)`` and blends inside the ROI only.
- ``energy`` is a downsampled loudness envelope that the client may use to
  fine-tune openness between explicit segment levels.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class MotionSegment(BaseModel):
    """A single viseme event inside a motion chunk."""

    phoneme: str = Field(..., description="IPA / pinyin phoneme label, e.g. 'a'")
    viseme: str = Field(..., description="Viseme class, e.g. 'mouth_a'")
    start_sample: int = Field(..., ge=0)
    peak_sample: int = Field(..., ge=0)
    end_sample: int = Field(..., ge=0)
    start_level: int = Field(..., ge=0, le=6, description="Openness at segment start (0..6)")
    peak_level: int = Field(..., ge=0, le=6, description="Openness at segment peak (0..6)")
    end_level: int = Field(..., ge=0, le=6, description="Openness at segment end (0..6)")
    blend_in_samples: int = Field(..., ge=0, description="Crossfade duration into this segment")
    blend_out_samples: int = Field(..., ge=0, description="Crossfade duration out of this segment")


class MotionChunk(BaseModel):
    """Server -> client motion timeline for a span of PCM samples.

    The chunk must be sent as a JSON WebSocket text message BEFORE the
    binary PCM frame(s) whose playback overlaps ``[valid_from_sample,
    valid_to_sample)``.
    """

    type: str = Field(default="motion_chunk", description="WebSocket message discriminator")
    audio_id: str = Field(..., description="Same audio_id as the companion reply_chunk frames")
    sentence_seq: int = Field(..., ge=0, description="Sentence index within the current reply")
    chunk_seq: int = Field(..., ge=0, description="Motion chunk index within the sentence")
    sample_rate: int = Field(..., gt=0, description="PCM sample rate; fixed for the whole reply")
    valid_from_sample: int = Field(..., ge=0, description="First PCM sample this chunk covers")
    valid_to_sample: int = Field(..., ge=0, description="One past the last PCM sample this chunk covers")
    emotion: str = Field(default="neutral", description="Stable emotion label for the sentence")
    emotion_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    segments: list[MotionSegment] = Field(default_factory=list)
    energy_hop_samples: int = Field(
        default=480,
        gt=0,
        description="PCM samples covered by each entry in `energy`. MUST be > 0.",
    )
    energy: list[float] = Field(
        default_factory=list,
        description=(
            "Downsampled loudness envelope in [0,1], one value per "
            "`energy_hop_samples`. May be an empty list when no energy data is "
            "available, but the field itself is REQUIRED (Gate 0A contract)."
        ),
    )


class MotionInterrupted(BaseModel):
    """Server -> client signal that the current reply was interrupted.

    The client must immediately stop audio playback and clear its motion
    queue. ``at_sample`` is the sample index (against ``audio_id``) at
    which playback must halt.
    """

    type: str = Field(default="motion_interrupted")
    audio_id: str
    at_sample: int = Field(..., ge=0)


__all__ = ["MotionChunk", "MotionSegment", "MotionInterrupted"]
