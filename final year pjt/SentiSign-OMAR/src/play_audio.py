# src/play_audio.py
# ─────────────────────────────────────────────────────────────────────────────
# Audio playback and .wav saving.
# Handles both torch.Tensor (Chatterbox output) and numpy arrays.
# Uses sounddevice for playback — no SOX, no system audio tools needed.
# Uses torchaudio for saving (already installed with chatterbox-tts).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np


def _to_numpy(audio) -> np.ndarray:
    """Converts torch.Tensor or any array to flat float32 numpy array."""
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()
    except ImportError:
        pass
    return np.array(audio, dtype=np.float32).flatten()


def play_audio(audio, sample_rate: int = 24000):
    """Play audio through speakers. Accepts torch.Tensor or numpy array."""
    data = _to_numpy(audio)
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak

    try:
        import sounddevice as sd
        print(f"[play_audio] Playing {len(data)/sample_rate:.2f}s ...")
        sd.play(data, samplerate=sample_rate)
        sd.wait()
        print("[play_audio] Done.")
    except ImportError:
        print("[play_audio] sounddevice not found — trying fallback.")
        _fallback(data, sample_rate)
    except Exception as e:
        print(f"[play_audio] Error: {e} — trying fallback.")
        _fallback(data, sample_rate)


def save_wav(audio, path: str, sample_rate: int = 24000):
    """Save audio to a .wav file. Accepts torch.Tensor or numpy array."""
    try:
        import torch, torchaudio as ta
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(_to_numpy(audio)).unsqueeze(0)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        ta.save(path, audio.cpu(), sample_rate)
        print(f"[play_audio] Saved: {path}")
    except Exception:
        # scipy fallback
        import scipy.io.wavfile as wav
        data = _to_numpy(audio)
        peak = np.max(np.abs(data))
        if peak > 1.0:
            data = data / peak
        wav.write(path, sample_rate, (data * 32767).astype(np.int16))
        print(f"[play_audio] Saved (scipy): {path}")


def _fallback(data: np.ndarray, sample_rate: int):
    """Last resort: write temp .wav and play with simpleaudio."""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = tmp.name
    tmp.close()
    import scipy.io.wavfile as wav
    wav.write(path, sample_rate, (data * 32767).astype(np.int16))
    try:
        import simpleaudio as sa
        sa.WaveObject.from_wave_file(path).play().wait_done()
    except Exception as e:
        print(f"[play_audio] All playback failed: {e}")
        print(f"[play_audio] Audio saved at: {path} — open it manually.")
    finally:
        try: os.remove(path)
        except: pass
