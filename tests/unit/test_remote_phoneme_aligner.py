"""Unit tests for RemotePhonemeAligner.

Mocks httpx.AsyncClient so no real HTTP is made. Validates:
  - happy path: aligner returns units → PhonemeTimeline with viseme/peak_ms
  - model mismatch → PhonemeAlignmentError (fail-closed)
  - service unreachable → PhonemeAlignmentError
  - empty units → PhonemeAlignmentError
"""
from __future__ import annotations

import asyncio

import pytest

from app.services import phoneme_service as svc
from app.services.phoneme_service import PhonemeAlignmentError, RemotePhonemeAligner


def _configure(monkeypatch):
    monkeypatch.setattr(svc.settings, "phoneme_aligner_endpoint", "http://aligner.test")
    monkeypatch.setattr(svc.settings, "phoneme_aligner_timeout_seconds", 5)
    monkeypatch.setattr(svc.settings, "phoneme_aligner_required_model", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
    monkeypatch.setattr(svc.settings, "phoneme_aligner_required_revision", "99ccb27")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Captures POST args; returns a queued response."""
    last_post: dict | None = None
    next_response: _FakeResponse | None = None
    raise_on_post: Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        type(self).last_post = {"url": url, **kwargs}
        if self.raise_on_post is not None:
            raise self.raise_on_post
        return self.next_response


def _patch_httpx(monkeypatch, response: _FakeResponse | None = None, raise_exc: Exception | None = None):
    import httpx as _real_httpx

    _FakeClient.next_response = response
    _FakeClient.raise_on_post = raise_exc

    # Patch AsyncClient at the module level where RemotePhonemeAligner imports it.
    monkeypatch.setattr(_real_httpx, "AsyncClient", _FakeClient)


def test_align_happy_path(monkeypatch):
    _configure(monkeypatch)
    resp = _FakeResponse(200, {
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "revision": "abc123",
        "sample_rate": 22050,
        "sample_count": 44100,
        "units": [
            {"text": "你", "start_sample": 0, "end_sample": 5000, "confidence": 0.9},
            {"text": "好", "start_sample": 5000, "end_sample": 11025, "confidence": 0.8},
        ],
        "elapsed_ms": 100,
    })
    _patch_httpx(monkeypatch, response=resp)

    async def _run():
        a = RemotePhonemeAligner()
        return await a.align_text(text="你好", duration_ms=2000, audio_id="aid", audio_wav=b"RIFF...")

    tl = asyncio.run(_run())
    assert tl.audio_id == "aid"
    assert len(tl.timeline) >= 2
    # Each segment has viseme + peak_ms set.
    for seg in tl.timeline:
        assert seg.viseme
        assert seg.peak_ms is not None
        assert seg.strength is not None
    # Boundaries are monotonic.
    starts = [s.start_ms for s in tl.timeline]
    assert starts == sorted(starts)


def test_align_model_mismatch_fail_closed(monkeypatch):
    _configure(monkeypatch)
    resp = _FakeResponse(200, {
        "model": "wrong/model",
        "revision": "x",
        "sample_rate": 16000,
        "sample_count": 100,
        "units": [{"text": "a", "start_sample": 0, "end_sample": 100, "confidence": 0.5}],
        "elapsed_ms": 1,
    })
    _patch_httpx(monkeypatch, response=resp)

    async def _run():
        a = RemotePhonemeAligner()
        return await a.align_text(text="a", duration_ms=100, audio_id="x", audio_wav=b"wav")

    with pytest.raises(PhonemeAlignmentError, match="model mismatch"):
        asyncio.run(_run())


def test_align_unreachable_fail_closed(monkeypatch):
    _configure(monkeypatch)
    _patch_httpx(monkeypatch, raise_exc=ConnectionError("refused"))

    async def _run():
        a = RemotePhonemeAligner()
        return await a.align_text(text="a", duration_ms=100, audio_id="x", audio_wav=b"wav")

    with pytest.raises(PhonemeAlignmentError, match="unreachable"):
        asyncio.run(_run())


def test_align_empty_units_fail_closed(monkeypatch):
    _configure(monkeypatch)
    resp = _FakeResponse(200, {
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "revision": "abc",
        "sample_rate": 16000,
        "sample_count": 100,
        "units": [],
        "elapsed_ms": 1,
    })
    _patch_httpx(monkeypatch, response=resp)

    async def _run():
        a = RemotePhonemeAligner()
        return await a.align_text(text="a", duration_ms=100, audio_id="x", audio_wav=b"wav")

    with pytest.raises(PhonemeAlignmentError, match="no alignment units"):
        asyncio.run(_run())


def test_align_requires_audio_wav(monkeypatch):
    _configure(monkeypatch)
    a = RemotePhonemeAligner()
    with pytest.raises(PhonemeAlignmentError, match="requires audio_wav"):
        asyncio.run(a.align_text(text="a", duration_ms=100, audio_id="x", audio_wav=None))


def test_get_phoneme_service_returns_remote_for_whisperx_service(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(svc.settings, "phoneme_provider", "whisperx_service")
    svc_inst = svc.get_phoneme_service()
    assert isinstance(svc_inst, RemotePhonemeAligner)


def test_get_phoneme_service_rejects_simple(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(svc.settings, "phoneme_provider", "simple")
    with pytest.raises(ValueError, match="not supported"):
        svc.get_phoneme_service()
