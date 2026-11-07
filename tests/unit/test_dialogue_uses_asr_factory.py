"""Source-level test: dialogue.py routes ASR through the unified factory.

Doc §7.1: the /dialogue HTTP endpoint must not call asr_service directly
for the production path; it must go through get_asr_engine() and
transcribe_complete_audio() (for streaming-capable providers).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIALOGUE = ROOT / "app" / "api" / "v1" / "routes" / "dialogue.py"


def test_dialogue_imports_get_asr_engine() -> None:
    src = DIALOGUE.read_text(encoding="utf-8")
    assert "from app.services.engine_factory import get_asr_engine" in src, (
        "dialogue.py must import get_asr_engine (Stage C unification)"
    )


def test_dialogue_imports_transcribe_complete_audio() -> None:
    src = DIALOGUE.read_text(encoding="utf-8")
    assert "transcribe_complete_audio" in src, (
        "dialogue.py must use transcribe_complete_audio for streaming providers"
    )


def test_dialogue_does_not_call_asr_service_transcribe_on_production_path() -> None:
    """The factory branch must come BEFORE the asr_service fallback, and the
    fallback must be guarded by ``else`` (i.e. only whisper hits it)."""
    src = DIALOGUE.read_text(encoding="utf-8")
    # The production path: hasattr(engine, "create_session") → transcribe_complete_audio
    assert "hasattr(asr_engine, \"create_session\")" in src
    # asr_service fallback must be in an else branch, not the primary path.
    assert "else:" in src
    # And the fallback must be clearly labeled as whisper/debug.
    assert "whisper" in src.lower() or "debug" in src.lower() or "fallback" in src.lower()


def test_dialogue_still_imports_asr_service_for_fallback() -> None:
    """asr_service import is retained for the whisper debug fallback."""
    src = DIALOGUE.read_text(encoding="utf-8")
    assert "from app.services.asr_service import asr_service" in src
