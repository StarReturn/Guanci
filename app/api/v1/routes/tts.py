from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.middleware.jwt import get_request_user_id
from app.schemas.tts import TTSSynthesizeRequest, TTSSynthesizeResponse
from app.services.auth_service import quota_service
from app.services.digital_human_service import (
    DigitalHumanConsentError,
    DigitalHumanNotFound,
)
from app.services.entitlement_stub import (
    EntitlementDenied,
    MembershipError,
    entitlement_service,
)
from app.services.phoneme_service import phoneme_service
from app.services.storage import storage_service
from app.services.tts_service import provider_name, synthesize_batch

router = APIRouter()


@router.post("/synthesize", response_model=TTSSynthesizeResponse)
async def synthesize_tts(request: Request, body: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    audio_id = str(uuid4())
    uid = get_request_user_id(request)
    if body.digital_human_id and uid is None:
        raise HTTPException(status_code=401, detail="user login required")
    usage_request_id = request.headers.get("Idempotency-Key") or f"tts-{audio_id}"
    if uid:
        try:
            entitlement_service.reserve(
                user_id=uid,
                key="tts_characters_per_month",
                quantity=len(body.text),
                request_id=usage_request_id,
                resource_type="tts",
                resource_id=audio_id,
            )
        except EntitlementDenied as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "ENTITLEMENT_LIMIT_EXCEEDED",
                    "entitlement_key": exc.key,
                    "limit": exc.limit,
                    "used": exc.used,
                    "requested": exc.requested,
                },
            ) from exc
        except MembershipError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        audio = await synthesize_batch(
            text=body.text,
            user_id=uid,
            digital_human_id=body.digital_human_id,
        )
    except DigitalHumanNotFound as exc:
        if uid:
            entitlement_service.release(
                user_id=uid, key="tts_characters_per_month",
                request_id=usage_request_id,
            )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DigitalHumanConsentError as exc:
        if uid:
            entitlement_service.release(
                user_id=uid, key="tts_characters_per_month",
                request_id=usage_request_id,
            )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        if uid:
            entitlement_service.release(
                user_id=uid,
                key="tts_characters_per_month",
                request_id=usage_request_id,
            )
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc
    if uid:
        entitlement_service.confirm(
            user_id=uid,
            key="tts_characters_per_month",
            request_id=usage_request_id,
            actual_quantity=len(body.text),
        )

    # P3: usage stats only — no quota enforcement (product decision 2026-06-23)
    if uid:
        try:
            quota_service.consume(uid, "tts")
        except Exception:
            pass  # stats failure should never break user flow

    storage_prefix = (
        f"digital_humans/{uid}/{body.digital_human_id}/tts"
        if uid and body.digital_human_id
        else "tts"
    )
    storage_key, _, _ = storage_service.put_bytes(
        key=f"{storage_prefix}/{audio_id}.wav",
        content=audio.wav_bytes,
        content_type="audio/wav",
    )

    timeline = None
    if body.with_phoneme_timeline:
        # Pass the wav bytes so that energy/whisperx providers can do
        # real alignment against the audio. Simple provider ignores it.
        timeline = await phoneme_service.align_text(
            text=body.text,
            duration_ms=audio.duration_ms,
            audio_id=audio_id,
            audio_wav=audio.wav_bytes,
        )

    return TTSSynthesizeResponse(
        audio_id=audio_id,
        audio_url=storage_service.presigned_url(storage_key),
        storage_key=storage_key,
        content_type="audio/wav",
        duration_ms=audio.duration_ms,
        provider=provider_name(),
        phoneme_timeline=timeline,
    )
