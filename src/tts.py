# src/tts.py
# ─────────────────────────────────────────────────────────────────────────────
# Emotion-aware TTS using Chatterbox-TTS.
#
# Why Chatterbox for emotion:
#   - Only open-source TTS with ARCHITECTURALLY TRAINED emotion control
#   - Two real parameters: exaggeration (intensity) + cfg_weight (pacing)
#   - Not a post-processing hack — emotion is baked into model training
#   - No SOX, no espeak-ng, no system dependencies on Windows
#   - Chatterbox model (~1GB) auto-downloads on first run
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import torch

_SRC = os.path.dirname(__file__)
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass

from emotion_map import get_params
from play_audio import play_audio, save_wav

_model = None  # chatterbox loaded once, reused forever


def get_output_extension() -> str:
    return "wav"


def _get_device() -> str:
    if torch.cuda.is_available():
        print(f"[tts] CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    # Apple Silicon GPU (Metal Performance Shaders).
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("[tts] MPS available — using Apple GPU.")
        return "mps"
    print("[tts] No GPU — using CPU (slower).")
    return "cpu"


def load_model():
    """Load Chatterbox once and cache it for all subsequent calls."""
    return _load()


def _load():
    global _model
    if _model is not None:
        return _model
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise ImportError(
            "[tts] chatterbox-tts not installed.\n"
            "Fix: pip install chatterbox-tts"
        )
    device = _get_device()

    # `chatterbox-tts` expects `perth.PerthImplicitWatermarker` to exist, but the
    # optional backend can be missing (e.g., `pkg_resources`/setuptools not installed).
    # Fall back to a no-op-ish dummy watermarker so TTS still works.
    try:
        import perth
        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
            print("[tts] perth watermark backend unavailable — using DummyWatermarker.")
    except Exception:
        pass

    # Prefer local checkpoints (supports manual download to models/chatterbox/).
    local_ckpt_dir = os.path.join(_ROOT, "models", "chatterbox")
    required = [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ]
    if all(os.path.exists(os.path.join(local_ckpt_dir, f)) for f in required):
        print(f"[tts] Loading Chatterbox from local checkpoints: {local_ckpt_dir}")
        _model = ChatterboxTTS.from_local(local_ckpt_dir, device=device)
    else:
        print("[tts] Loading Chatterbox (~1GB download on first run) ...")
        _model = ChatterboxTTS.from_pretrained(device=device)

    print(f"[tts] Chatterbox ready. Sample rate: {_model.sr}Hz\n")
    return _model


def speak(
    text: str,
    emotion: str = "neutral",
    play: bool = True,
):
    """
    Synthesise text with Chatterbox.

    Args:
        text:    Sentence to speak.
        emotion: neutral / happy / sad / angry / fear / disgust / surprise
        play:    Play audio immediately if True.

    Returns:
        torch.Tensor [1, N] at 24000 Hz
    """
    params = get_params(emotion)

    print(f"[tts] Synthesising ...")
    print(f"  Text         : {text}")
    print(f"  Emotion      : {emotion}")
    print(f"  Exaggeration : {params['exaggeration']}  ← emotional intensity")
    print(f"  CFG Weight   : {params['cfg_weight']}  ← pacing control")
    print(f"  Character    : {params['description']}")

    model = _load()

    with torch.inference_mode():
        wav = model.generate(
            text,
            exaggeration = params["exaggeration"],
            cfg_weight   = params["cfg_weight"],
        )

    duration = wav.shape[-1] / model.sr
    print(f"[tts] Generated {duration:.2f}s of audio.")

    if play:
        play_audio(wav, sample_rate=model.sr)

    return wav


def speak_and_save(
    text: str,
    emotion: str = "neutral",
    path: str = "output.wav",
    also_play: bool = True,
) -> str:
    """Generate speech, save to disk (.wav), and optionally play."""
    model = _load()
    wav = speak(text, emotion=emotion, play=also_play)
    save_wav(wav, path, sample_rate=model.sr)
    return path


# ── Standalone emotion test ───────────────────────────────────────────────────
if __name__ == "__main__":
    from emotion_map import list_emotions
    sentence = "I will go to the hospital tomorrow."
    print(f'\nTesting all emotions with: "{sentence}"\n')
    for em in list_emotions():
        print(f"\n{'─'*48}")
        print(f"  Emotion: {em.upper()}")
        speak_and_save(sentence, emotion=em, path=f"test_{em}.wav", also_play=True)
        input("  [Enter for next emotion] ")
