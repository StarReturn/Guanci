"""Phoneme alignment service (v3: WhisperX only).

The ``simple`` and ``energy`` providers have been removed in v3.
Production requires ``whisperx`` for real audio forced alignment.

See docs/修复方案-P0.md (P0.1) for the design.
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
import wave
from abc import ABC, abstractmethod
from uuid import uuid4

from pypinyin import Style, lazy_pinyin

from app.core.config import settings
from app.core.phoneme_map import to_viseme
from app.schemas.phoneme import PhonemeSegment, PhonemeTimeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinyin splitter (shared across providers)
# ---------------------------------------------------------------------------

_FINALS = (
    "iang", "iong", "uang",
    "ang", "eng", "ing", "ong",
    "iao", "ian", "uan", "uai",
    "ai", "ei", "ao", "ou", "an", "en", "in", "un", "er",
    "ia", "ie", "iu", "ua", "uo", "ui", "ve", "vn",
    "a", "o", "e", "i", "u", "v",
)


def split_pinyin_syllable(syllable: str) -> list[str]:
    """Split a pinyin syllable into initial + final.

    Mirrors the original ``_split_syllable`` but module-level so providers
    can share it.
    """
    if not syllable:
        return []
    for final in _FINALS:
        if syllable.endswith(final) and syllable != final:
            initial = syllable[: -len(final)]
            return [initial, final]
    return [syllable]


def text_to_phonemes(text: str) -> list[str]:
    """Convert Chinese text to a flat list of phoneme strings."""
    syllables = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    phonemes: list[str] = []
    for s in syllables:
        phonemes.extend(split_pinyin_syllable(s))
    return phonemes


# ---------------------------------------------------------------------------
# Aligner abstraction
# ---------------------------------------------------------------------------

class PhonemeAligner(ABC):
    """Base class for all phoneme alignment strategies."""

    @abstractmethod
    async def align_text(
        self,
        *,
        text: str,
        duration_ms: int,
        audio_id: str | None = None,
        audio_wav: bytes | None = None,
    ) -> PhonemeTimeline:
        ...

    # -- shared helpers ----------------------------------------------------

    @staticmethod
    def _build_segment(phoneme: str, start_ms: int, end_ms: int) -> PhonemeSegment:
        return PhonemeSegment(
            phoneme=phoneme,
            viseme=to_viseme(phoneme),
            start_ms=start_ms,
            end_ms=end_ms,
        )

    @staticmethod
    def _finalize(
        timeline: list[PhonemeSegment],
        duration_ms: int,
        audio_id: str | None,
    ) -> PhonemeTimeline:
        return PhonemeTimeline(
            audio_id=audio_id or str(uuid4()),
            duration_ms=duration_ms,
            timeline=timeline,
        )


# ---------------------------------------------------------------------------
# WhisperX provider (best quality, requires GPU ideally)
# ---------------------------------------------------------------------------

class WhisperXPhonemeAligner(PhonemeAligner):
    """Forced alignment via WhisperX. Lazy-imports to keep the dependency
    optional when not configured.
    """

    _transcribe_model = None
    _align_model = None
    _align_metadata = None

    async def align_text(
        self,
        *,
        text: str,
        duration_ms: int,
        audio_id: str | None = None,
        audio_wav: bytes | None = None,
    ) -> PhonemeTimeline:
        if audio_wav is None:
            raise RuntimeError(
                "whisperx aligner requires audio_wav; the simple/energy "
                "fallback path was removed per mobile_keyframe_v3 spec. "
                "Caller must provide real PCM."
            )
        return await asyncio.to_thread(
            self._align_sync, text, audio_wav, duration_ms, audio_id
        )

    def _align_sync(
        self, text: str, audio_wav: bytes, duration_ms: int, audio_id: str | None
    ) -> PhonemeTimeline:
        import tempfile
        import wave

        import whisperx

        # WhisperX loads audio from a file path; write the wav to a tmp file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio_wav)
            tmp_path = tf.name

        try:
            audio = whisperx.load_audio(tmp_path)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        self._ensure_models()
        result = self._transcribe_model.transcribe(
            audio, language="zh", initial_prompt=text,
        )
        aligned = whisperx.align(
            result["segments"], self._align_model, self._align_metadata, audio,
            device=settings.whisperx_device, return_char_alignments=False,
        )

        timeline: list[PhonemeSegment] = []
        for seg in aligned.get("segments", []):
            for word in seg.get("words", []):
                word_text = word.get("word", "").strip()
                if not word_text:
                    continue
                start_ms = int(word.get("start", 0) * 1000)
                end_ms = int(word.get("end", start_ms) * 1000)
                if end_ms <= start_ms:
                    end_ms = start_ms + 50
                phonemes = text_to_phonemes(word_text)
                if not phonemes:
                    timeline.append(self._build_segment("sil", start_ms, end_ms))
                    continue
                # Distribute word duration across its phonemes (even split
                # within the word — WhisperX gives us word-level only).
                step = max(1, (end_ms - start_ms) // len(phonemes))
                for i, p in enumerate(phonemes):
                    p_start = start_ms + i * step
                    p_end = end_ms if i == len(phonemes) - 1 else p_start + step
                    timeline.append(self._build_segment(p, p_start, p_end))

        if not timeline:
            timeline.append(self._build_segment("sil", 0, duration_ms))

        return self._finalize(timeline, duration_ms, audio_id)

    @classmethod
    def _ensure_models(cls) -> None:
        if cls._transcribe_model is None:
            import whisperx
            cls._transcribe_model = whisperx.load_model(
                settings.whisperx_model_name,
                device=settings.whisperx_device,
                compute_type=settings.whisperx_compute_type,
            )
        if cls._align_model is None:
            import whisperx
            cls._align_model, cls._align_metadata = whisperx.load_align_model(
                language_code="zh", device=settings.whisperx_device,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class PhonemeAlignmentError(RuntimeError):
    """Raised when forced alignment fails. Callers send PHONEME_ALIGNMENT_FAILED."""


class RemotePhonemeAligner(PhonemeAligner):
    """Forced alignment via the standalone phoneme-aligner service.

    Stage D (Gate 0): the Wav2Vec2ForCTC alignment model runs in an
    isolated CPU container (``phoneme-aligner:9890``) rather than inside
    the API image. This client calls ``POST /v1/align`` with the known
    TTS text + real TTS WAV and receives per-character sample boundaries
    in the original PCM coordinate space. It then maps characters →
    pinyin → phonemes → visemes (via the shared helpers) to build a
    PhonemeTimeline.

    Fail-closed: any service unreachable / model mismatch / invalid
    response → raise (the caller sends PHONEME_ALIGNMENT_FAILED and
    skips the sentence's PCM). No fallback to equal-split.
    """

    def _endpoint(self) -> str:
        return settings.phoneme_aligner_endpoint.rstrip("/")

    def _timeout(self) -> float:
        return float(settings.phoneme_aligner_timeout_seconds)

    async def align_text(
        self,
        *,
        text: str,
        duration_ms: int,
        audio_id: str | None = None,
        audio_wav: bytes | None = None,
    ) -> PhonemeTimeline:
        if audio_wav is None:
            raise PhonemeAlignmentError(
                "RemotePhonemeAligner requires audio_wav (real TTS audio)"
            )
        import httpx

        url = self._endpoint() + "/v1/align"
        headers = {"x-request-id": audio_id or str(uuid4())}
        files = {"audio": ("audio.wav", audio_wav, "audio/wav")}
        data = {"text": text, "request_id": audio_id or ""}

        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                resp = await client.post(url, files=files, data=data, headers=headers)
        except Exception as exc:
            raise PhonemeAlignmentError(
                f"phoneme-aligner unreachable at {self._endpoint()}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise PhonemeAlignmentError(
                f"phoneme-aligner returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            payload = resp.json()
        except Exception as exc:
            raise PhonemeAlignmentError(f"phoneme-aligner returned non-JSON: {exc}") from exc

        # Model identity check (fail-closed on mismatch).
        model = payload.get("model", "")
        revision = payload.get("revision", "")
        if model and settings.phoneme_aligner_required_model and model != settings.phoneme_aligner_required_model:
            raise PhonemeAlignmentError(
                f"phoneme-aligner model mismatch: got {model!r}, "
                f"required {settings.phoneme_aligner_required_model!r}"
            )

        units = payload.get("units") or []
        if not units:
            raise PhonemeAlignmentError("phoneme-aligner returned no alignment units")

        # Map character boundaries → phonemes → visemes.
        # Each unit covers [start_sample, end_sample] in the ORIGINAL TTS
        # sample space (the service already did the 16k→original mapping).
        # Convert sample → ms for the PhonemeSegment schema (which is still
        # ms-based until stage E rewrites it to sample-based).
        sample_rate = int(payload.get("sample_rate", 0))
        if sample_rate <= 0:
            raise PhonemeAlignmentError(f"invalid sample_rate in aligner response: {sample_rate}")

        timeline: list[PhonemeSegment] = []
        for unit in units:
            char_text = (unit.get("text") or "").strip()
            if not char_text:
                continue  # unlabeled alignment units carry no viseme information
            start_sample = int(unit.get("start_sample", 0))
            end_sample = int(unit.get("end_sample", 0))
            confidence = float(unit.get("confidence", 0.5))
            if end_sample <= start_sample:
                continue
            start_ms = round(start_sample * 1000 / sample_rate)
            end_ms = round(end_sample * 1000 / sample_rate)
            peak_ms = (start_ms + end_ms) // 2
            # Split the character into its phonemes (pypinyin), then
            # distribute the character's interval across them. This is
            # per-character distribution (allowed); whole-sentence equal
            # split is forbidden.
            phonemes = text_to_phonemes(char_text) or [char_text]
            if len(phonemes) == 1:
                seg = self._build_segment(phonemes[0], start_ms, end_ms)
                seg.peak_ms = peak_ms
                seg.strength = confidence
                timeline.append(seg)
            else:
                span = end_ms - start_ms
                per = span / len(phonemes)
                for i, ph in enumerate(phonemes):
                    ph_start = start_ms + int(i * per)
                    ph_end = start_ms + int((i + 1) * per)
                    seg = self._build_segment(ph, ph_start, ph_end)
                    seg.peak_ms = (ph_start + ph_end) // 2
                    seg.strength = confidence
                    timeline.append(seg)

        if not timeline:
            raise PhonemeAlignmentError("alignment produced no usable segments")

        return self._finalize(timeline, duration_ms, audio_id)


def get_phoneme_service() -> PhonemeAligner:
    """Build the active aligner.

    mobile_keyframe_v3 mandates real audio forced alignment; the legacy
    ``simple`` (equal-split) and ``energy`` (RMS-only) providers are
    removed.

    Providers:
      - ``whisperx_service`` (production default): standalone CPU service
        via RemotePhonemeAligner.
      - ``whisperx`` (debug only): in-process WhisperX, requires the
        whisperx extra and ideally a GPU.
    """
    provider = (settings.phoneme_provider or "whisperx_service").lower()
    if provider == "whisperx_service":
        return RemotePhonemeAligner()
    if provider == "whisperx":
        return WhisperXPhonemeAligner()
    raise ValueError(
        f"phoneme_provider={provider!r} is not supported in v3; "
        "only 'whisperx_service' (production) or 'whisperx' (debug) are allowed "
        "(simple/energy are removed)."
    )


phoneme_service: PhonemeAligner = get_phoneme_service()
