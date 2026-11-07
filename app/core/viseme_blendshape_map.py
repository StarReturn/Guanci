"""Viseme to ARKit BlendShape mapping table.

Maps our 13-viseme system (VISEME_NATURAL_OPENNESS) to ARKit 52 BlendShape
coefficients. Each viseme maps to a set of BlendShape weights [0.0, 1.0].

Usage:
    from app.core.viseme_blendshape_map import VISEME_TO_BLENDSHAPE, get_blendshape_frame
    weights = get_blendshape_frame("mouth_a", openness=0.8)
    # Returns: {"jawOpen": 0.72, "mouthFunnel": 0.3, ...}

The openness parameter (0.0-1.0, from VISEME_NATURAL_OPENNESS) scales the
base values proportionally.
"""
from __future__ import annotations

# ARKit 52 BlendShape names (standard set, per Apple ARFaceAnchor.BlendShapeLocation)
ARKIT_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",  # the 52nd BlendShape
]

# Our 13 visemes → ARKit BlendShape base weights
# Base weights are for full openness (level 6); lower levels scale proportionally
VISEME_TO_BLENDSHAPE = {
    # 大开口元音 a — jaw wide open, lips relaxed
    "mouth_a": {
        "jawOpen": 0.85,
        "mouthFunnel": 0.05,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
        "mouthStretchLeft": 0.15,
        "mouthStretchRight": 0.15,
    },
    # 圆唇元音 o — jaw half open, lips rounded
    "mouth_o": {
        "jawOpen": 0.45,
        "mouthFunnel": 0.65,
        "mouthPucker": 0.55,
        "mouthClose": 0.0,
    },
    # 中开口元音 e — jaw medium, slight funnel
    "mouth_e": {
        "jawOpen": 0.50,
        "mouthFunnel": 0.20,
        "mouthPucker": 0.05,
        "mouthClose": 0.0,
        "mouthStretchLeft": 0.10,
        "mouthStretchRight": 0.10,
    },
    # 高元音 i — jaw slightly open, lips spread
    "mouth_i": {
        "jawOpen": 0.25,
        "mouthFunnel": 0.0,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
        "mouthStretchLeft": 0.30,
        "mouthStretchRight": 0.30,
    },
    # 圆唇高元音 u/ü — jaw closed, lips pursed
    "mouth_u": {
        "jawOpen": 0.10,
        "mouthFunnel": 0.75,
        "mouthPucker": 0.70,
        "mouthClose": 0.0,
    },
    "mouth_v": {
        "jawOpen": 0.15,
        "mouthFunnel": 0.55,
        "mouthPucker": 0.50,
        "mouthClose": 0.0,
    },
    # 唇音 m/b/p — lips together
    "mouth_mf": {
        "jawOpen": 0.0,
        "mouthFunnel": 0.0,
        "mouthPucker": 0.10,
        "mouthClose": 0.85,
    },
    # 舌尖音 d/t — jaw slightly open, tongue tip up
    "mouth_dt": {
        "jawOpen": 0.20,
        "mouthFunnel": 0.05,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
        "mouthLowerDownLeft": 0.20,
        "mouthLowerDownRight": 0.20,
    },
    # 舌根音 g/k — jaw slightly open
    "mouth_gk": {
        "jawOpen": 0.25,
        "mouthFunnel": 0.10,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
    },
    # 卷舌音 zh/ch/sh/r — lips slightly pursed
    "mouth_zhr": {
        "jawOpen": 0.20,
        "mouthFunnel": 0.35,
        "mouthPucker": 0.25,
        "mouthClose": 0.0,
    },
    # 平舌音 z/c/s — lips spread
    "mouth_zcs": {
        "jawOpen": 0.15,
        "mouthFunnel": 0.15,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
        "mouthStretchLeft": 0.15,
        "mouthStretchRight": 0.15,
    },
    # 舌面音 j/q/x — similar to i
    "mouth_jqx": {
        "jawOpen": 0.20,
        "mouthFunnel": 0.10,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
        "mouthStretchLeft": 0.20,
        "mouthStretchRight": 0.20,
    },
    # 闭嘴（静默）— all zeros
    "mouth_closed": {
        "jawOpen": 0.0,
        "mouthFunnel": 0.0,
        "mouthPucker": 0.0,
        "mouthClose": 0.0,
    },
}

# Emotion overlays (additive, on top of viseme)
EMOTION_BLENDSHAPE = {
    "happy": {
        "mouthSmileLeft": 0.40,
        "mouthSmileRight": 0.40,
        "cheekSquintLeft": 0.20,
        "cheekSquintRight": 0.20,
        "browInnerUp": 0.05,
    },
    "sad": {
        "mouthFrownLeft": 0.30,
        "mouthFrownRight": 0.30,
        "browInnerUp": 0.25,
        "mouthShrugLower": 0.10,
    },
    "surprised": {
        "jawOpen": 0.20,  # additive to viseme
        "eyeWideLeft": 0.30,
        "eyeWideRight": 0.30,
        "browInnerUp": 0.35,
        "browOuterUpLeft": 0.30,
        "browOuterUpRight": 0.30,
    },
    "angry": {
        "browDownLeft": 0.40,
        "browDownRight": 0.40,
        "mouthPressLeft": 0.20,
        "mouthPressRight": 0.20,
        "noseSneerLeft": 0.15,
        "noseSneerRight": 0.15,
    },
    "neutral": {},
}


# Visemes whose BlendShape weights should NOT be scaled by openness.
# Bilabial consonants (m/b/p) require lips fully closed regardless of
# the "openness level" — the openness dimension doesn't apply to them.
# Similarly, closed/pucker shapes maintain their form at any level.
_NON_SCALING_VISEMES = {
    "mouth_mf",      # m/b/p: lips must be fully closed
    "mouth_closed",  # silence: all zeros regardless
    "mouth_u",       # rounded high vowel: pucker is the defining feature
    "mouth_v",       # similar to u
}

# Within a viseme, some BlendShapes should not be scaled even for
# scalable visemes (e.g., mouthPucker for "o" should stay high even
# when jaw is less open).
_NON_SCALING_KEYS = {
    "mouthClose",    # lip closure is binary, not openness-dependent
    "mouthPucker",   # pucker amount is shape-dependent, not openness-dependent
    "mouthPressLeft", "mouthPressRight",
}


def get_blendshape_frame(viseme: str, openness: float = 1.0,
                         emotion: str = "neutral",
                         emotion_intensity: float = 0.0) -> dict[str, float]:
    """Get ARKit BlendShape weights for a single frame.

    Args:
        viseme: One of our 13 visemes (e.g. "mouth_a")
        openness: 0.0-1.0, from VISEME_NATURAL_OPENNESS (normalize level/6)
        emotion: One of "happy", "sad", "surprised", "angry", "neutral"
        emotion_intensity: 0.0-1.0

    Returns:
        Dict of BlendShape name -> weight [0.0, 1.0]
    """
    base = VISEME_TO_BLENDSHAPE.get(viseme, VISEME_TO_BLENDSHAPE["mouth_closed"])

    # Determine scaling strategy for this viseme
    is_non_scaling = viseme in _NON_SCALING_VISEMES
    clamped_openness = max(0.0, min(1.0, openness))

    result = {}
    for key, value in base.items():
        if is_non_scaling:
            # Bilabial/pucker visemes: use full base values regardless of openness
            result[key] = value
        elif key in _NON_SCALING_KEYS:
            # Shape-defining keys within scalable visemes: don't scale
            result[key] = value
        else:
            # Openness-dependent keys: scale proportionally
            result[key] = min(1.0, value * clamped_openness)

    # Add emotion overlay
    if emotion_intensity > 0:
        overlay = EMOTION_BLENDSHAPE.get(emotion, {})
        for key, value in overlay.items():
            result[key] = min(1.0, result.get(key, 0.0) + value * emotion_intensity)

    return result


def interpolate_frames(frame_a: dict[str, float], frame_b: dict[str, float],
                       t: float) -> dict[str, float]:
    """Linearly interpolate between two BlendShape frames.

    Args:
        frame_a: Start frame
        frame_b: End frame
        t: Interpolation factor [0.0, 1.0]

    Returns:
        Interpolated frame
    """
    result = {}
    all_keys = set(frame_a.keys()) | set(frame_b.keys())
    for key in all_keys:
        a = frame_a.get(key, 0.0)
        b = frame_b.get(key, 0.0)
        result[key] = a + (b - a) * t
    return result


def frames_to_expression_chunk(frames: list[dict],
                               fps: int = 30,
                               start_sample: int = 0,
                               sample_rate: int = 24000) -> dict:
    """Convert a list of BlendShape frames to a WebSocket expression_chunk.

    Args:
        frames: List of dicts from get_blendshape_frame()
        fps: Frame rate
        start_sample: Audio sample offset
        sample_rate: Audio sample rate

    Returns:
        expression_chunk WebSocket message payload
    """
    samples_per_frame = sample_rate // fps
    return {
        "type": "expression_chunk",
        "audio_id": "",  # filled by caller
        "sentence_seq": 0,
        "chunk_seq": 0,
        "sample_rate": sample_rate,
        "valid_from_sample": start_sample,
        "valid_to_sample": start_sample + len(frames) * samples_per_frame,
        "fps": fps,
        "coefficient_set": "arkit_52_v1",
        "frames": [
            {
                "at_sample": start_sample + i * samples_per_frame,
                "weights": [round(frames[i].get(bs, 0.0), 4) for bs in ARKIT_BLENDSHAPES],
            }
            for i in range(len(frames))
        ],
    }
