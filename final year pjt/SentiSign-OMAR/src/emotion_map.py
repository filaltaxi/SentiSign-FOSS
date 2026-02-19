# src/emotion_map.py
# ─────────────────────────────────────────────────────────────────────────────
# Maps the 7 universal emotion labels to Chatterbox-TTS generation parameters.
#
# Chatterbox has TWO architecturally trained emotion levers:
#
#   exaggeration (float, 0.25 → 2.0)
#       Emotional INTENSITY.
#       0.25 = flat/subdued   |   0.5 = neutral   |   1.5+ = very dramatic
#       Higher values naturally speed up delivery too.
#
#   cfg_weight (float, 0.0 → 1.0)
#       Pacing DELIBERATENESS (classifier-free guidance weight).
#       0.0 = fast, loose     |   0.5 = default   |   1.0 = slow, careful
#
# Tuning logic per emotion:
#   neutral  — default values, no push in either direction
#   happy    — moderate exaggeration (bright/warm), medium cfg (steady pace)
#   sad      — very low exaggeration (flat/heavy), very low cfg (slow/dragging)
#   angry    — very high exaggeration (intense), low cfg (fast/clipped)
#   fear     — medium exaggeration (heightened), very low cfg (hesitant/slow)
#   disgust  — low exaggeration (cold/flat), high cfg (deliberate/measured)
#   surprise — high exaggeration (dramatic), medium-high cfg (controlled burst)
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PARAMS = {
    #  label      exaggeration  cfg_weight  description
    "neutral":   (0.50,         0.50,       "Balanced, natural, conversational"),
    "happy":     (0.85,         0.60,       "Warm, bright, energetic but steady"),
    "sad":       (0.25,         0.20,       "Subdued, heavy, slow dragging pace"),
    "angry":     (1.40,         0.20,       "Intense, forceful, fast clipped delivery"),
    "fear":      (0.95,         0.05,       "Heightened tension, hesitant slow pace"),
    "disgust":   (0.20,         0.90,       "Cold, flat, deliberate measured delivery"),
    "surprise":  (1.20,         0.65,       "Dramatic sharp onset, expressive"),
}

DEFAULT_EMOTION = "neutral"


def get_params(emotion: str) -> dict:
    """
    Returns Chatterbox generation parameters for the given emotion.

    Args:
        emotion: One of the 7 supported labels (case-insensitive).

    Returns:
        {"exaggeration": float, "cfg_weight": float, "description": str}
    """
    key = emotion.strip().lower()
    if key not in EMOTION_PARAMS:
        print(f"[emotion_map] '{emotion}' not recognised → using '{DEFAULT_EMOTION}'.")
        key = DEFAULT_EMOTION

    exaggeration, cfg_weight, description = EMOTION_PARAMS[key]
    return {
        "exaggeration": exaggeration,
        "cfg_weight":   cfg_weight,
        "description":  description,
    }


def list_emotions() -> list:
    return list(EMOTION_PARAMS.keys())


def print_emotion_table():
    print("\n  Emotion → Chatterbox Parameters")
    print("  " + "─" * 68)
    print(f"  {'Emotion':<12} {'Exaggeration':<16} {'CFG Weight':<14} What you hear")
    print("  " + "─" * 68)
    for label, (exag, cfg, desc) in EMOTION_PARAMS.items():
        print(f"  {label:<12} {exag:<16} {cfg:<14} {desc}")
    print("  " + "─" * 68)
