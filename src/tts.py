# src/tts.py
# ─────────────────────────────────────────────────────────────────────────────
# Emotion-aware TTS with a runtime-selectable backend.
# Default: Cartesia API
# Optional fallback: Chatterbox-TTS
# Select with: SENTISIGN_TTS_ENGINE=cartesia|chatterbox
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

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

_model = None
_loaded_engine = None

_DEFAULT_TTS_ENGINE = "cartesia"
_DEFAULT_API_URL = "https://api.cartesia.ai/tts/bytes"
_DEFAULT_API_VERSION = "2025-04-16"
_DEFAULT_MODEL_ID = "sonic-3"
_DEFAULT_VOICE_ID = "5ee9feff-1265-424a-9d7f-8e4d431a12c7"
_DEFAULT_SAMPLE_RATE = 44100

_ENGINE_LABELS = {
    "cartesia": "Cartesia",
    "chatterbox": "Chatterbox",
}

_CHATTERBOX_EXAGGERATION = {
    "neutral": 0.50,
    "happy": 0.85,
    "sad": 0.25,
    "angry": 1.40,
    "fear": 0.95,
    "disgust": 0.20,
    "surprise": 1.20,
}

_CHATTERBOX_CFG_WEIGHT = {
    "neutral": 0.50,
    "happy": 0.60,
    "sad": 0.20,
    "angry": 0.20,
    "fear": 0.05,
    "disgust": 0.90,
    "surprise": 0.65,
}


@dataclass(frozen=True)
class CartesiaConfig:
    api_key: str
    api_url: str
    api_version: str
    model_id: str
    voice_id: str
    sample_rate: int
    language: str
    timeout_seconds: float


def get_output_extension() -> str:
    return "wav"


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def get_engine_id() -> str:
    engine = os.getenv("SENTISIGN_TTS_ENGINE", _DEFAULT_TTS_ENGINE).strip().lower()
    if engine not in _ENGINE_LABELS:
        print(
            f"[tts] Unknown SENTISIGN_TTS_ENGINE='{engine}' "
            f"-> using '{_DEFAULT_TTS_ENGINE}'."
        )
        return _DEFAULT_TTS_ENGINE
    return engine


def get_engine_label() -> str:
    return _ENGINE_LABELS[get_engine_id()]


def load_model():
    """Validate the configured backend once and cache it for subsequent calls."""
    return _load_backend()


def _load_backend():
    global _model, _loaded_engine

    engine = get_engine_id()
    if _model is not None and _loaded_engine == engine:
        return _model

    if engine == "chatterbox":
        _model = _load_chatterbox()
    else:
        _model = _load_cartesia()

    _loaded_engine = engine
    return _model


def _load_cartesia() -> CartesiaConfig:
    api_key = _env_first("CARTESIA_API_KEY", "CARTESIA_API")
    if not api_key:
        raise RuntimeError(
            "[tts] Cartesia API key missing.\n"
            "Set CARTESIA_API or CARTESIA_API_KEY in .env."
        )

    voice_id = _env_first("CARTESIA_VOICE_ID", "CARTESIA_VOICE") or _DEFAULT_VOICE_ID
    api_url = _env_first("CARTESIA_TTS_URL") or _DEFAULT_API_URL
    api_version = _env_first("CARTESIA_VERSION") or _DEFAULT_API_VERSION
    model_id = _env_first("CARTESIA_MODEL_ID") or _DEFAULT_MODEL_ID
    language = _env_first("CARTESIA_LANGUAGE") or "en"
    sample_rate_raw = _env_first("CARTESIA_SAMPLE_RATE")
    timeout_raw = _env_first("CARTESIA_TIMEOUT_SECONDS")

    try:
        sample_rate = int(sample_rate_raw) if sample_rate_raw else _DEFAULT_SAMPLE_RATE
    except ValueError as e:
        raise RuntimeError(f"[tts] Invalid CARTESIA_SAMPLE_RATE: {sample_rate_raw}") from e

    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else 45.0
    except ValueError as e:
        raise RuntimeError(
            f"[tts] Invalid CARTESIA_TIMEOUT_SECONDS: {timeout_raw}"
        ) from e

    config = CartesiaConfig(
        api_key=api_key,
        api_url=api_url,
        api_version=api_version,
        model_id=model_id,
        voice_id=voice_id,
        sample_rate=sample_rate,
        language=language,
        timeout_seconds=timeout_seconds,
    )
    print(f"[tts] {get_engine_label()} ready.")
    print(f"[tts] Model: {config.model_id}")
    print(f"[tts] Voice: {config.voice_id}")
    print(f"[tts] Output: wav / pcm_f32le / {config.sample_rate}Hz\n")
    return config


def _get_chatterbox_device() -> str:
    import torch

    if torch.cuda.is_available():
        print(f"[tts] CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("[tts] MPS available -> using Apple GPU.")
        return "mps"
    print("[tts] No GPU -> using CPU (slower).")
    return "cpu"


def _load_chatterbox():
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as e:
        raise RuntimeError(
            "[tts] Chatterbox selected but chatterbox-tts is not installed.\n"
            "Run: uv sync --extra tts"
        ) from e

    device = _get_chatterbox_device()

    try:
        import perth

        if getattr(perth, "PerthImplicitWatermarker", None) is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
            print("[tts] perth watermark backend unavailable -> using DummyWatermarker.")
    except Exception:
        pass

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
        model = ChatterboxTTS.from_local(local_ckpt_dir, device=device)
    else:
        print("[tts] Loading Chatterbox (~1GB download on first run) ...")
        model = ChatterboxTTS.from_pretrained(device=device)

    print(f"[tts] Chatterbox ready. Sample rate: {model.sr}Hz\n")
    return model


def _base_payload(config: CartesiaConfig, text: str) -> dict:
    payload = {
        "model_id": config.model_id,
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": config.voice_id,
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": config.sample_rate,
        },
        "speed": "normal",
    }
    if config.language:
        payload["language"] = config.language
    return payload


def _post_tts_bytes(config: CartesiaConfig, payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.api_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "X-API-Key": config.api_key,
            "Cartesia-Version": config.api_version,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as response:
            audio_bytes = response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Cartesia TTS request failed: {e.code} {e.reason}. {detail[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cartesia TTS request failed: {e.reason}") from e

    if not audio_bytes:
        raise RuntimeError("Cartesia TTS returned an empty audio response.")
    return audio_bytes


def _synthesise_cartesia(text: str, emotion: str) -> bytes:
    config = _load_cartesia()
    params = get_params(emotion)
    base_payload = _base_payload(config, text)
    attempts = []

    if params["experimental_emotions"]:
        experimental_payload = dict(base_payload)
        experimental_payload["voice"] = {
            **base_payload["voice"],
            "__experimental_controls": {
                "speed": "normal",
                "emotion": params["experimental_emotions"],
            },
        }
        attempts.append(("experimental_controls", experimental_payload))

    generation_payload = dict(base_payload)
    generation_payload["generation_config"] = {
        "speed": 1,
        "volume": 1,
        "emotion": params["generation_emotion"],
    }
    attempts.append(("generation_config", generation_payload))
    attempts.append(("plain", base_payload))

    failures = []
    for label, payload in attempts:
        try:
            return _post_tts_bytes(config, payload)
        except Exception as e:
            failures.append(f"{label}: {e}")

    joined = " | ".join(failures)
    raise RuntimeError(f"Cartesia TTS failed after retries. {joined}")


def _play_wav_bytes(audio_bytes: bytes):
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))
    audio = np.asarray(data)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    play_audio(audio, sample_rate=int(sample_rate))


def _speak_cartesia(text: str, emotion: str, play: bool) -> bytes:
    params = get_params(emotion)

    print("[tts] Synthesising ...")
    print(f"  Engine       : {get_engine_label()}")
    print(f"  Text         : {text}")
    print(f"  Emotion      : {emotion}")
    print(f"  Character    : {params['description']}")
    if params["experimental_emotions"]:
        print(f"  Cartesia Tags: {', '.join(params['experimental_emotions'])}")
    else:
        print("  Cartesia Tags: none")

    audio_bytes = _synthesise_cartesia(text, emotion)
    if play:
        _play_wav_bytes(audio_bytes)
    return audio_bytes


def _speak_chatterbox(text: str, emotion: str, play: bool):
    model = _load_chatterbox()
    params = get_params(emotion)
    key = emotion.strip().lower()
    if key not in _CHATTERBOX_EXAGGERATION:
        key = "neutral"

    print("[tts] Synthesising ...")
    print(f"  Engine       : {get_engine_label()}")
    print(f"  Text         : {text}")
    print(f"  Emotion      : {emotion}")
    print(f"  Character    : {params['description']}")
    print(f"  Exaggeration : {_CHATTERBOX_EXAGGERATION[key]}")
    print(f"  CFG Weight   : {_CHATTERBOX_CFG_WEIGHT[key]}")

    import torch

    with torch.inference_mode():
        wav = model.generate(
            text,
            exaggeration=_CHATTERBOX_EXAGGERATION[key],
            cfg_weight=_CHATTERBOX_CFG_WEIGHT[key],
        )

    if play:
        play_audio(wav, sample_rate=model.sr)
    return wav


def speak(
    text: str,
    emotion: str = "neutral",
    play: bool = True,
):
    """
    Synthesise text using the configured backend and optionally play it.

    Returns:
        Backend-specific audio object.
    """
    engine = get_engine_id()
    if engine == "chatterbox":
        return _speak_chatterbox(text, emotion, play)
    return _speak_cartesia(text, emotion, play)


def speak_and_save(
    text: str,
    emotion: str = "neutral",
    path: str = "output.wav",
    also_play: bool = True,
) -> str:
    """Generate speech, save to disk (.wav), and optionally play."""
    engine = get_engine_id()
    audio = speak(text, emotion=emotion, play=False)

    if engine == "chatterbox":
        model = _load_chatterbox()
        save_wav(audio, path, sample_rate=model.sr)
        if also_play:
            play_audio(audio, sample_rate=model.sr)
        return path

    with open(path, "wb") as f:
        f.write(audio)
    print(f"[tts] Saved: {path}")
    if also_play:
        _play_wav_bytes(audio)
    return path


if __name__ == "__main__":
    from emotion_map import list_emotions

    sentence = "I will go to the hospital tomorrow."
    print(f'\nTesting all emotions with: "{sentence}"\n')
    for em in list_emotions():
        print(f"\n{'─'*48}")
        print(f"  Emotion: {em.upper()}")
        speak_and_save(sentence, emotion=em, path=f"test_{em}.wav", also_play=True)
        input("  [Enter for next emotion] ")
