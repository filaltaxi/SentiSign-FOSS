# SentiSign — Setup & Run Guide
## Stack: Ollama qwen3.5:0.8b + Chatterbox TTS | Python 3.10 | Windows

---

## Using `uv` (recommended)

From the repo root:

```bash
uv --preview-features extra-build-dependencies sync
uv --preview-features extra-build-dependencies run python run_pipeline.py
```

Web UI:
```bash
uv --preview-features extra-build-dependencies run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you want to run training/retraining tools (plots, etc.):
```bash
uv --preview-features extra-build-dependencies sync --extra train
```

---

## Project Structure
```
SentiSign-OMAR/
│
├── run_pipeline.py          ← MAIN ENTRY POINT (run this)
├── requirements.txt         ← all dependencies
│
├── src/                     ← TTS + emotion modules
│   ├── tts.py               ← Chatterbox TTS wrapper
│   ├── emotion_map.py       ← emotion → exaggeration + cfg_weight
│   └── play_audio.py        ← sounddevice playback, .wav saving
│
└── slm/                     ← sentence generation module
    ├── download_model.py    ← optional flan-t5-large fallback download
    └── src/
        ├── sentence_model.py    ← Ollama / Hugging Face sentence backend
        └── generate_sentence.py ← word buffer → sentence
```

---

## Step 1 — Create & Activate Virtual Environment

Open Command Prompt in your project folder:

```bat
cd "C:\path\to\SentiSign-OMAR"
```

Create venv:
```bat
py -3.10 -m venv .venv
```

Activate:
```bat
.venv\Scripts\activate.bat
```

You should now see `(.venv)` at the start of your prompt.

---

## Step 2 — Install Dependencies

```bat
pip install -U pip
pip install -r requirements.txt
```

> `chatterbox-tts` pulls in torch + torchaudio automatically.
> No SOX. No espeak-ng. No system tools needed on Windows.

---

## Step 3 — Install the Ollama sentence model

Make sure the Ollama server is available, then pull the local model:

```bash
ollama pull qwen3.5:0.8b
```

Create `.env` if you want the explicit local config:

```bash
copy .env.example .env
```

Optional legacy fallback:

```bat
python slm\download_model.py
```

> Chatterbox model (~1GB) auto-downloads on first `run_pipeline.py` run.
> `slm\download_model.py` is only needed when using `SENTISIGN_SENTENCE_PROVIDER=hf`.

---

## Step 4 — Run the Pipeline

```bat
python run_pipeline.py
```

### Example session:
```
  Words > I, TOMORROW, HOSPITAL, GO
  Emotion > sad

  [1/2] Generating sentence...
  ✓  Sentence: "I will go to the hospital tomorrow."

  [2/2] Synthesising (exaggeration=0.25, cfg_weight=0.20)...
  🔊  Slow, heavy, subdued — genuinely sad delivery
```

---

## Emotion → Parameter Reference

| Emotion  | Exaggeration | CFG Weight | What you hear                       |
|----------|-------------|------------|-------------------------------------|
| neutral  | 0.50        | 0.50       | Calm, balanced, natural             |
| happy    | 0.85        | 0.60       | Warm, bright, energetic             |
| sad      | 0.25        | 0.20       | Subdued, heavy, slow                |
| angry    | 1.40        | 0.20       | Intense, forceful, fast             |
| fear     | 0.70        | 0.15       | Tense, hesitant, slow               |
| disgust  | 0.30        | 0.75       | Cold, flat, deliberate              |
| surprise | 1.20        | 0.65       | Dramatic, sharp, expressive         |

To fine-tune: edit values in `src/emotion_map.py`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied` creating venv | Close VS Code, run terminal as Administrator |
| `No module named 'chatterbox'` | `pip install chatterbox-tts` |
| `No module named 'sentencepiece'` | `pip install sentencepiece` |
| `(.venv)` not showing | Run `.venv\Scripts\activate.bat` |
| sounddevice plays nothing | `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| Ollama unreachable | Start `ollama serve` and verify `ollama list` shows `qwen3.5:0.8b` |
| flan-t5-large missing in HF mode | `python slm\download_model.py` |
| CUDA out of memory | Both models fall back to CPU automatically |

---

## Coming Next
- Step 4: Sign Language Recognition → auto-fills word buffer
- Step 5: Real-time webcam demo
