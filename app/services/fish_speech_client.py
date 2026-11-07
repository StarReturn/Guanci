"""Fish-Speech self-hosted client — inline ``reference_audio`` mode.

Self-hosted Fish-Speech docker image (``fishaudio/fish-speech``) HTTP API
exposes only ``POST /v1/tts``; it does **not** have a ``/model``
registration endpoint (that's a cloud-only feature at ``api.fish.audio``).

Voice cloning on self-hosted works by passing the reference audio
**inline per call** as base64 in the ``reference_audio`` field of the
TTS request body. The reference clip is the one the user uploaded at
avatar-build time — backed up in MinIO under
``user_avatars.voice_ref_audio_key``.

Per-call cost:
  * ~10ms to fetch the wav from MinIO (~2MB on local network)
  * ~33% size bloat from base64 (2MB → ~2.7MB)
  * ~50-100ms on RTX 4090 for voice embedding re-computation

Future optimization (out of scope for POC): mount a shared volume at
fish-speech container's ``references/`` directory and write
``references/<voice_id>/sample.wav`` + ``sample.lab`` from the backend
container. Some fish-speech versions accept ``reference_id=<dir_name>``
in ``/v1/tts`` which would let us skip the per-call upload. Verify by
curling ``http://fish-speech:8080/docs`` on the running container.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.http_client import build_resilient_client, retry_exp
from app.services.audio_utils import normalize_audio_to_wav, wav_duration_ms


class FishSpeechError(RuntimeError):
    pass


@dataclass
class FishSpeechAudio:
    wav_bytes: bytes
    content_type: str
    duration_ms: int
    # True if a user-uploaded reference clip was used (clone), False if
    # Fish-Speech fell back to its default voice.
    used_reference: bool


class FishSpeechClient:
    def __init__(self) -> None:
        self._endpoint = settings.fish_speech_endpoint.rstrip("/")
        self._read_timeout = float(settings.fish_speech_timeout_seconds)

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _headers(self) -> dict[str, str]:
        # NOTE: do NOT set Content-Type here — caller (synthesize) sets it
        # to application/json because the TTS payload is JSON.
        headers: dict[str, str] = {}
        api_key = (settings.fish_speech_api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def health_check(self) -> dict[str, Any]:
        try:
            async with build_resilient_client(read_timeout=5.0) as client:
                resp = await client.get(self._url(settings.fish_speech_health_path))
                resp.raise_for_status()
            return {"status": "ok", "provider": "fish_speech", "endpoint": self._endpoint}
        except Exception as exc:
            return {
                "status": "error",
                "provider": "fish_speech",
                "endpoint": self._endpoint,
                "error": str(exc),
            }

    async def synthesize(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        format_: str = "wav",
    ) -> FishSpeechAudio:
        """Synthesize ``text`` to WAV bytes.

        Voice cloning (self-hosted inline mode):
          Pass ``reference_audio_bytes`` (raw wav) + ``reference_text``
          (transcript). Both are base64-encoded into the TTS request body
          as ``reference_audio`` + ``reference_text`` per the self-hosted
          fish-speech API spec.

          Omit both to use Fish-Speech's default voice.

        Raises :class:`FishSpeechError` on any non-200 response.
        """
        payload: dict[str, Any] = {"text": text, "format": format_}

        used_reference = False
        if reference_audio_bytes:
            # Normalize to WAV first — Fish-Speech expects raw PCM/wav bytes
            # that we then base64. Passing mp3/m4a through base64 works too
            # in newer fish-speech but WAV is the safe common denominator.
            wav = normalize_audio_to_wav(reference_audio_bytes, "audio/wav")
            payload["reference_audio"] = base64.b64encode(wav).decode("ascii")
            if reference_text:
                payload["reference_text"] = reference_text
            used_reference = True

        headers = self._headers()
        headers["Content-Type"] = "application/json"

        async def _do_post() -> Any:
            async with build_resilient_client(read_timeout=self._read_timeout) as client:
                resp = await client.post(
                    self._url(settings.fish_speech_tts_path),
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    raise FishSpeechError(
                        f"Fish-Speech TTS failed: HTTP {resp.status_code} {resp.text[:500]}"
                    )
                return resp

        resp = await retry_exp(_do_post, attempts=3)

        content_type = resp.headers.get("content-type", "audio/wav")
        if "application/json" in content_type:
            raise FishSpeechError(f"Fish-Speech returned JSON instead of audio: {resp.text[:500]}")

        wav_bytes = normalize_audio_to_wav(resp.content, content_type)
        return FishSpeechAudio(
            wav_bytes=wav_bytes,
            content_type="audio/wav",
            duration_ms=wav_duration_ms(wav_bytes),
            used_reference=used_reference,
        )

    async def synthesize_batch(
        self,
        *,
        text: str,
        reference_audio_bytes: bytes | None = None,
        reference_text: str | None = None,
        format_: str = "wav",
        emotion: str | None = None,
        user_id: str | None = None,
    ) -> FishSpeechAudio:
        """Provider-compatible alias used by shared TTS business routes."""
        _ = emotion, user_id
        return await self.synthesize(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            reference_text=reference_text,
            format_=format_,
        )


fish_speech_client = FishSpeechClient()
