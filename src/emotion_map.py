# src/emotion_map.py
# ─────────────────────────────────────────────────────────────────────────────
# Maps the 7 universal emotion labels to Cartesia emotion-control hints.
# The documented Cartesia path uses `voice.__experimental_controls.emotion`.
# We also keep a simple string label for a fallback payload shape.
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PARAMS = {
    "neutral": {
        "experimental_emotions": [],
        "generation_emotion": "neutral",
        "description": "Balanced, natural, conversational",
    },
    "happy": {
        "experimental_emotions": ["positivity:high"],
        "generation_emotion": "happy",
        "description": "Warm, bright, energetic",
    },
    "sad": {
        "experimental_emotions": ["sadness:high"],
        "generation_emotion": "sad",
        "description": "Subdued, heavy, slower and softer",
    },
    "angry": {
        "experimental_emotions": ["anger:high"],
        "generation_emotion": "angry",
        "description": "Intense, forceful, clipped",
    },
    "fear": {
        "experimental_emotions": ["fear:high"],
        "generation_emotion": "fear",
        "description": "Tense, uneasy, hesitant",
    },
    "disgust": {
        "experimental_emotions": ["disgust:high"],
        "generation_emotion": "disgust",
        "description": "Cold, flat, deliberate",
    },
    "surprise": {
        "experimental_emotions": ["surprise:high"],
        "generation_emotion": "surprise",
        "description": "Sharp, lifted, expressive",
    },
}

DEFAULT_EMOTION = "neutral"


def get_params(emotion: str) -> dict:
    """Return Cartesia control hints for the given emotion label."""
    key = emotion.strip().lower()
    if key not in EMOTION_PARAMS:
        print(f"[emotion_map] '{emotion}' not recognised → using '{DEFAULT_EMOTION}'.")
        key = DEFAULT_EMOTION
    return dict(EMOTION_PARAMS[key])


def list_emotions() -> list[str]:
    return list(EMOTION_PARAMS.keys())


def print_emotion_table():
    print("\n  Emotion → Cartesia Voice Hints")
    print("  " + "─" * 72)
    print(f"  {'Emotion':<12} {'Control Tags':<28} What you hear")
    print("  " + "─" * 72)
    for label, params in EMOTION_PARAMS.items():
        tags = ", ".join(params["experimental_emotions"]) or "none"
        print(f"  {label:<12} {tags:<28} {params['description']}")
    print("  " + "─" * 72)
