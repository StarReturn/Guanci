from pydantic import BaseModel, Field

from app.schemas.phoneme import PhonemeTimeline


class TTSSynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    digital_human_id: str | None = Field(default=None, max_length=64)
    voice_model_key: str | None = None
    with_phoneme_timeline: bool = True


class TTSSynthesizeResponse(BaseModel):
    audio_id: str
    audio_url: str
    storage_key: str
    content_type: str
    duration_ms: int
    provider: str
    phoneme_timeline: PhonemeTimeline | None = None
