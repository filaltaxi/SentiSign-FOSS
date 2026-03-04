# src/tts.py
# ─────────────────────────────────────────────────────────────────────────────
# Emotion-aware TTS using Chatterbox-TTS.
#
# Why Chatterbox for emotion:
#   - Only open-source TTS with ARCHITECTURALLY TRAINED emotion control
#   - Two real parameters: exaggeration (intensity) + cfg_weight (pacing)
#   - Not a post-processing hack — emotion is baked into model training
#   - Benchmarked above ElevenLabs in naturalness evaluations
#   - No SOX, no espeak-ng, no system dependencies on Windows
#   - Chatterbox model (~1GB) auto-downloads on first run
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import torch
from typing import Any

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
_eleven_client = None

DEFAULT_TTS_PROVIDER = "chatterbox"
SUPPORTED_TTS_PROVIDERS = {"chatterbox", "elevenlabs"}
DEFAULT_ELEVEN_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_ELEVEN_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_ELEVEN_OUTPUT_FORMAT = "mp3_44100_128"


def resolve_provider(provider: str | None = None) -> str:
    raw = provider or os.environ.get("SENTISIGN_TTS_PROVIDER", DEFAULT_TTS_PROVIDER)
    key = (raw or DEFAULT_TTS_PROVIDER).strip().lower()
    if key not in SUPPORTED_TTS_PROVIDERS:
        raise ValueError(
            f"[tts] Unsupported provider '{raw}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_TTS_PROVIDERS))}."
        )
    return key


def get_output_extension(provider: str | None = None) -> str:
    key = resolve_provider(provider)
    return "mp3" if key == "elevenlabs" else "wav"


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


def load_model(provider: str | None = None):
    """Load TTS provider once and cache it for all subsequent calls."""
    key = resolve_provider(provider)
    if key == "elevenlabs":
        return _load_elevenlabs()
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


def _load_elevenlabs():
    global _eleven_client
    if _eleven_client is not None:
        return _eleven_client

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "[tts] ELEVENLABS_API_KEY is not set. "
            "Set it in your environment before using provider='elevenlabs'."
        )
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise ImportError(
            "[tts] elevenlabs package not installed.\n"
            "Fix: pip install elevenlabs"
        )

    _eleven_client = ElevenLabs(api_key=api_key)
    print("[tts] ElevenLabs client ready.")
    return _eleven_client


def _collect_audio_bytes(audio: Any) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, str):
        return audio.encode("utf-8")
    chunks: list[bytes] = []
    try:
        for chunk in audio:
            if isinstance(chunk, (bytes, bytearray)):
                chunks.append(bytes(chunk))
            elif chunk is not None:
                chunks.append(str(chunk).encode("utf-8"))
    except TypeError:
        return bytes(str(audio), "utf-8")
    return b"".join(chunks)


def _speak_elevenlabs(
    text: str,
    emotion: str,
    play: bool = True,
    voice_id: str | None = None,
    model_id: str | None = None,
):
    client = _load_elevenlabs()
    resolved_voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_ELEVEN_VOICE_ID)
    resolved_model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID", DEFAULT_ELEVEN_MODEL_ID)

    print("[tts] Synthesising with ElevenLabs ...")
    print(f"  Text      : {text}")
    print(f"  Emotion   : {emotion} (metadata)")
    print(f"  Voice ID  : {resolved_voice_id}")
    print(f"  Model ID  : {resolved_model_id}")

    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=resolved_voice_id,
        model_id=resolved_model_id,
        output_format=DEFAULT_ELEVEN_OUTPUT_FORMAT,
    )
    audio_bytes = _collect_audio_bytes(audio_stream)
    print(f"[tts] ElevenLabs generated {len(audio_bytes)} bytes.")

    if play:
        try:
            from elevenlabs.play import play as eleven_play

            eleven_play(audio_bytes)
        except Exception as e:
            print(f"[tts] ElevenLabs playback failed: {e}")
            print("[tts] Audio was generated and can still be saved.")

    return audio_bytes


def speak(
    text: str,
    emotion: str = "neutral",
    play: bool = True,
    provider: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
):
    """
    Synthesise text using the selected provider.

    Args:
        text:    Sentence to speak.
        emotion: neutral / happy / sad / angry / fear / disgust / surprise
        play:    Play audio immediately if True.
        provider: chatterbox | elevenlabs

    Returns:
        Chatterbox: torch.Tensor [1, N] at 24000 Hz
        ElevenLabs: bytes (encoded audio stream)
    """
    selected_provider = resolve_provider(provider)
    if selected_provider == "elevenlabs":
        return _speak_elevenlabs(
            text=text,
            emotion=emotion,
            play=play,
            voice_id=voice_id,
            model_id=model_id,
        )

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
    provider: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> str:
    """Generate speech, save to disk (.wav/.mp3), and optionally play."""
    selected_provider = resolve_provider(provider)
    if selected_provider == "elevenlabs":
        audio_bytes = speak(
            text,
            emotion=emotion,
            play=also_play,
            provider=selected_provider,
            voice_id=voice_id,
            model_id=model_id,
        )
        with open(path, "wb") as f:
            f.write(audio_bytes)
        print(f"[tts] Saved: {path}")
        return path

    model = _load()
    wav = speak(text, emotion=emotion, play=also_play, provider=selected_provider)
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
