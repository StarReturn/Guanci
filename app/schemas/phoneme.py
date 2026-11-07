from pydantic import BaseModel, Field


class PhonemeAlignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    duration_ms: int = Field(default=1500, ge=200, le=60000)
    audio_id: str | None = None


class PhonemeSegment(BaseModel):
    phoneme: str
    viseme: str = Field(
        default="mouth_closed",
        description="Visual viseme class (e.g. mouth_a). The mobile renderer "
        "looks up frames by this identifier, not by the raw phoneme.",
    )
    start_ms: int
    end_ms: int
    peak_ms: int | None = Field(
        default=None,
        description="Optional peak openness sample within [start_ms, end_ms]. "
        "Consumed by realtime_ws to compute start/peak/end motion levels.",
    )
    strength: float | None = Field(
        default=None,
        description="Optional phoneme intensity [0,1] used to derive mouth "
        "openness level (0..6) when peak_ms is present.",
    )


class PhonemeTimeline(BaseModel):
    type: str = "phoneme_timeline"
    audio_id: str
    duration_ms: int
    timeline: list[PhonemeSegment]
