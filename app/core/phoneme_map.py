"""Chinese phoneme to viseme mapping.

Viseme classes follow the Preston Blair 13-shape standard, adapted for
Mandarin pinyin initials and finals. Visual equivalent phonemes are grouped
so that the mobile renderer only needs to switch between 13 mouth shapes
instead of 50+ phonemes.

See docs/修复方案-P0.md (P0.2) for the full rationale.
"""

# Phoneme string -> viseme class identifier.
# Visemes are stable identifiers used inside the keyframe package
# (frames/<expression>/<viseme>.png). Do NOT rename without bumping
# the manifest version.
PHONEME_TO_VISEME: dict[str, str] = {
    # ---- Silence / pause ----
    "sil": "mouth_closed",
    "sp": "mouth_closed",
    "": "mouth_closed",

    # ---- Initials grouped by articulation point ----
    # Labial stops/fricatives (M/B/P/F): closed-lip burst
    "b": "mouth_mf",
    "p": "mouth_mf",
    "m": "mouth_mf",
    "f": "mouth_mf",

    # Alveolar (D/T/N/L): tongue tip on upper teeth ridge
    "d": "mouth_dt",
    "t": "mouth_dt",
    "n": "mouth_dt",
    "l": "mouth_dt",

    # Velar (G/K/H): back of tongue raised
    "g": "mouth_gk",
    "k": "mouth_gk",
    "h": "mouth_gk",

    # Palatal (J/Q/X + Y): tongue blade to hard palate
    "j": "mouth_jqx",
    "q": "mouth_jqx",
    "x": "mouth_jqx",
    "y": "mouth_jqx",

    # Retroflex (ZH/CH/SH/R): curled tongue
    "zh": "mouth_zhr",
    "ch": "mouth_zhr",
    "sh": "mouth_zhr",
    "r": "mouth_zhr",

    # Dental sibilants (Z/C/S): tongue tip behind lower teeth
    "z": "mouth_zcs",
    "c": "mouth_zcs",
    "s": "mouth_zcs",

    # W shares the visual of U (rounded narrow)
    "w": "mouth_u",

    # ---- Simple finals (single vowels) ----
    "a": "mouth_a",
    "o": "mouth_o",
    "e": "mouth_e",
    "i": "mouth_i",
    "u": "mouth_u",
    "v": "mouth_v",

    # ---- Compound finals grouped by main vowel ----
    # A family (open mouth, low vowel)
    "ai": "mouth_a",
    "ao": "mouth_a",
    "an": "mouth_a",
    "ang": "mouth_a",
    "ia": "mouth_a",
    "ian": "mouth_a",
    "iao": "mouth_a",
    "iang": "mouth_a",
    "ua": "mouth_a",
    "uai": "mouth_a",
    "uan": "mouth_a",
    "uang": "mouth_a",

    # O family (rounded, mid)
    "ou": "mouth_o",
    "uo": "mouth_o",
    "ong": "mouth_o",
    "iong": "mouth_o",
    "iu": "mouth_o",  # iu = iou (main vowel o)

    # E family (half-open, spread)
    "ei": "mouth_e",
    "en": "mouth_e",
    "eng": "mouth_e",
    "er": "mouth_e",
    "ie": "mouth_e",
    "ue": "mouth_e",
    "ve": "mouth_e",
    "ui": "mouth_e",  # ui = uei (main vowel e)

    # I family (narrow, spread)
    "in": "mouth_i",
    "ing": "mouth_i",

    # U family (narrow, rounded)
    "un": "mouth_u",

    # V (Ü) family
    "vn": "mouth_v",
}

# Fallback when a phoneme is not in the map (rare, defensive).
DEFAULT_VISEME = "mouth_closed"

# Stable enumeration of viseme classes. Useful for the mobile renderer
# to pre-allocate sprite slots and for tests to verify completeness.
VISEME_CLASSES: tuple[str, ...] = (
    "mouth_closed",
    "mouth_mf",
    "mouth_dt",
    "mouth_gk",
    "mouth_jqx",
    "mouth_zhr",
    "mouth_zcs",
    "mouth_a",
    "mouth_o",
    "mouth_e",
    "mouth_i",
    "mouth_u",
    "mouth_v",
)


def to_viseme(phoneme: str) -> str:
    """Map a single phoneme string to its viseme class.

    Unknown / unmapped phonemes fall back to ``DEFAULT_VISEME`` (silence),
    so the mobile renderer never crashes on unexpected input.
    """
    return PHONEME_TO_VISEME.get(phoneme.lower(), DEFAULT_VISEME)


# Natural mouth aperture per viseme (0..6). This is the ANATOMICAL openness
# of the viseme, NOT alignment confidence — "a" opens wide, "i" barely parts.
# The realtime motion builder uses this table so the mouth's open/close
# rhythm follows the syllables (what viewers read as lip-sync).
VISEME_NATURAL_OPENNESS: dict[str, int] = {
    "mouth_a": 6,       # 大开口元音
    "mouth_o": 5,       # 圆口中开
    "mouth_e": 4,
    "mouth_u": 4,       # 圆唇
    "mouth_v": 3,
    "mouth_i": 3,       # 扁唇小开
    "mouth_jqx": 3,
    "mouth_zhr": 3,
    "mouth_zcs": 2,
    "mouth_gk": 2,
    "mouth_dt": 2,
    "mouth_mf": 1,      # 双唇闭合辅音
    "mouth_closed": 0,
}
