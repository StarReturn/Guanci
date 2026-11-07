"""ASR service — self-hosted Whisper for speech-to-text.

Uses openai-whisper library. Model loaded lazily on first call.
GPU recommended (cuda) but CPU fallback works (10x slower).
"""
from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class AsrService:
    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            import whisper
            model_name = os.environ.get("ASR_MODEL", "large-v3")
            device = os.environ.get("ASR_DEVICE", "cuda")
            logger.info("loading whisper model=%s device=%s ...", model_name, device)
            cls._model = whisper.load_model(model_name, device=device)
            logger.info("whisper model loaded")
        return cls._model

    def transcribe(self, audio_bytes: bytes, language: str = "zh") -> str:
        """Transcribe wav bytes → text.

        Blocks the caller thread — use from asyncio.to_thread().
        """
        model = self._get_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = tf.name
        try:
            result = model.transcribe(tmp_path, language=language)
            return result.get("text", "").strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


asr_service = AsrService()
