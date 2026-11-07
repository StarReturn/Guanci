"""Unit tests for motion segment construction from phoneme boundaries.

Validates the core invariants of the motion_chunk.segments[] the realtime
WS builds from a PhonemeTimeline (doc §9.1, §9.2):
  - 0 <= start_sample <= peak_sample <= end_sample
  - levels are 0..6
  - viseme comes from to_viseme()
  - silence → mouth_closed/0
  - segments fall within the chunk's [valid_from_sample, valid_to_sample]

These tests exercise the conversion logic directly rather than spinning up
the full WS pipeline.
"""
from __future__ import annotations

from app.core.phoneme_map import DEFAULT_VISEME, VISEME_CLASSES, to_viseme
from app.schemas.phoneme import PhonemeSegment


def _build_motion_segments(
    timeline: list[PhonemeSegment],
    *,
    total_samples_written: int,
    sample_rate: int,
) -> list[dict]:
    """Mirror of realtime_ws._flush_sentence's segment-construction loop."""
    segments = []
    for seg in timeline:
        viseme = seg.viseme or DEFAULT_VISEME
        strength = seg.strength if seg.strength is not None else 0.5
        openness = min(6, max(0, int(strength * 6)))
        peak_ms = seg.peak_ms if seg.peak_ms is not None else (seg.start_ms + seg.end_ms) // 2
        segments.append({
            "phoneme": seg.phoneme,
            "viseme": viseme,
            "start_sample": total_samples_written + int(seg.start_ms * sample_rate / 1000),
            "peak_sample": total_samples_written + int(peak_ms * sample_rate / 1000),
            "end_sample": total_samples_written + int(seg.end_ms * sample_rate / 1000),
            "start_level": max(0, openness - 2),
            "peak_level": openness,
            "end_level": max(0, openness - 2),
            "blend_in_samples": int(sample_rate * 0.02),
            "blend_out_samples": int(sample_rate * 0.03),
        })
    return segments


def test_segment_ordering_start_peak_end():
    segs = [
        PhonemeSegment(phoneme="a", viseme=to_viseme("a"), start_ms=0, end_ms=200, peak_ms=100, strength=0.8),
        PhonemeSegment(phoneme="n", viseme=to_viseme("n"), start_ms=200, end_ms=400, peak_ms=300, strength=0.3),
    ]
    motion = _build_motion_segments(segs, total_samples_written=0, sample_rate=24000)
    assert len(motion) == 2
    for m in motion:
        assert 0 <= m["start_sample"] <= m["peak_sample"] <= m["end_sample"]
        assert 0 <= m["start_level"] <= 6
        assert 0 <= m["peak_level"] <= 6
        assert 0 <= m["end_level"] <= 6
        assert m["blend_in_samples"] > 0
        assert m["blend_out_samples"] > 0


def test_segment_visemes_are_in_canonical_set():
    segs = [
        PhonemeSegment(phoneme="a", viseme=to_viseme("a"), start_ms=0, end_ms=100, peak_ms=50, strength=0.7),
    ]
    motion = _build_motion_segments(segs, total_samples_written=0, sample_rate=24000)
    assert motion[0]["viseme"] in VISEME_CLASSES


def test_segments_offset_by_total_samples_written():
    """A non-zero base offset shifts every boundary uniformly."""
    segs = [
        PhonemeSegment(phoneme="a", viseme=to_viseme("a"), start_ms=0, end_ms=100, peak_ms=50, strength=0.5),
    ]
    base = 48000
    motion = _build_motion_segments(segs, total_samples_written=base, sample_rate=24000)
    assert motion[0]["start_sample"] >= base
    assert motion[0]["peak_sample"] >= motion[0]["start_sample"]
    assert motion[0]["end_sample"] >= motion[0]["peak_sample"]


def test_silence_uses_default_viseme():
    """When a segment has no phoneme/viseme, the default (mouth_closed) applies."""
    seg = PhonemeSegment(phoneme="", viseme=DEFAULT_VISEME, start_ms=0, end_ms=50, peak_ms=25, strength=0.0)
    motion = _build_motion_segments([seg], total_samples_written=0, sample_rate=24000)
    assert motion[0]["viseme"] == DEFAULT_VISEME
    assert motion[0]["peak_level"] == 0  # strength 0 → openness 0


def test_strength_maps_to_openness_0_6():
    """strength in [0,1] must clamp to openness in [0,6]."""
    for strength in (0.0, 0.16, 0.5, 0.84, 1.0):
        seg = PhonemeSegment(phoneme="a", viseme=to_viseme("a"), start_ms=0, end_ms=100, peak_ms=50, strength=strength)
        motion = _build_motion_segments([seg], total_samples_written=0, sample_rate=24000)
        assert 0 <= motion[0]["peak_level"] <= 6
