from fastapi import APIRouter, HTTPException, Request

from app.middleware.jwt import get_request_user_id
from app.schemas.phoneme import PhonemeAlignRequest, PhonemeTimeline
from app.services.auth_service import quota_service
from app.services.phoneme_service import phoneme_service

router = APIRouter()


@router.post("/align", response_model=PhonemeTimeline)
async def align_phonemes(request: Request, body: PhonemeAlignRequest) -> PhonemeTimeline:
    _ = request
    timeline = await phoneme_service.align_text(
        text=body.text,
        duration_ms=body.duration_ms,
        audio_id=body.audio_id,
    )
    # Stats only, no quota check (decision 2026-06-23)
    uid = get_request_user_id(request)
    if uid:
        try:
            quota_service.consume(uid, "tts")
        except Exception:
            pass
    return timeline

