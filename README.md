# SentiSign

**Multimodal Sign Language and Emotion-Aware Speech Synthesis**

SentiSign translates ASL hand signs and facial emotion into natural, emotionally-expressive speech — in real time, on a standard webcam. No specialised hardware required.

---

## Demo

| Step | What happens |
|------|-------------|
| 1 | Signer faces webcam and performs ASL signs |
| 2 | MediaPipe extracts hand landmarks (126 features, both hands) |
| 3 | MLP classifier maps landmarks to vocabulary words |
| 4 | ResNet CNN detects facial emotion simultaneously |
| 5 | Flan-T5-Large generates a grammatical sentence from word buffer |
| 6 | Chatterbox (local) or ElevenLabs (cloud) synthesises speech |

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 |
| Python | 3.10 |
| GPU | NVIDIA RTX (CUDA 12.4+) recommended |
| RAM | 8GB minimum, 16GB recommended |
| Webcam | Any standard USB or built-in webcam |

---

## Installation

**1 — Clone the repository**
```bash
git clone https://github.com/filaltaxi/SentiSign-OMAR.git
cd SentiSign-OMAR
```

**2 — Create virtual environment (pip)**
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**3 — Install dependencies (pip)**
```bash
pip install -r requirements.txt
```

**3.1 — Configure TTS provider (optional)**
```bash
cp .env.example .env
```
Set `ELEVENLABS_API_KEY` if you want ElevenLabs. You can switch provider in the web UI (`Voice Engine`) or desktop pipeline prompt.

**Alternative: install + run with `uv`**
```bash
uv sync
uv run python run_pipeline.py
```

**4 — Models**

This repo includes the non-SLM model files under `models/` (landmark + emotion).

The sentence model (`flan-t5-large/`) is not committed (too large) and is downloaded into `slm/models/` on first run (or via `slm/download_model.py`).

**5 — Collect training data (first time only)**
```bash
python collect_landmarks.py
```
Follow the on-screen instructions. Hold each sign steady — the system auto-saves samples. Takes approximately 2-3 hours for all 36 classes.

**6 — Train the landmark classifier**
```bash
python train_landmark_classifier.py
```
Takes under 2 minutes on GPU.

---

## Running

**Desktop pipeline (terminal)**
```bash
python run_pipeline.py
```

**Web interface (Development)**
```bash
make backend
make frontend
```
Open `http://localhost:5173` in your browser.

**Web interface (Production)**
```bash
cd frontend && npm run build
cd .. && uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser.

**With `uv`**
```bash
uv --preview-features extra-build-dependencies run python run_pipeline.py
uv --preview-features extra-build-dependencies run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project Structure

```
SentiSign-OMAR/
│
├── main.py                         # FastAPI backend
├── collect_landmarks.py            # Training data collection
├── train_landmark_classifier.py    # MLP + RF training
├── requirements.txt
├── pyproject.toml                  # uv support
│
├── src/
│   ├── sign_recognizer.py          # Desktop pipeline (terminal)
│   ├── emotion_detector.py         # ResNet FER
│   ├── tts.py                      # Multi-provider TTS (Chatterbox + ElevenLabs)
│   └── emotion_map.py              # Emotion → prosody parameters
│
├── frontend/
│   ├── src/                        # Vite + React + TS App
│   │   ├── components/             # Reusable UI components
│   │   ├── pages/                  # Route views (Communicate, Signs, etc.)
│   │   └── index.css               # Tailwind v4 configuration
│   └── dist/                       # Production build output
│
├── website/
│   └── audio/                      # Generated .wav files
│
├── models/
│   ├── landmark/                   # MLP, RF, label map (not in repo)
│   └── emotion/                    # ResNet weights (not in repo)
│
├── slm/
│   ├── models/flan-t5-large/       # Language model (auto/down/manual)
│   └── src/                        # Sentence generation
│
└── data/
    └── landmarks/
        ├── raw/                    # Per-class CSV files (not in repo)
        ├── references/             # Reference GIFs per sign
        └── plots/                  # Training evaluation plots
```

---

## Vocabulary (36 classes → 35 words)

| Class | Word | Class | Word | Class | Word |
|-------|------|-------|------|-------|------|
| A | I | M | PAIN | NUM_4 | PLEASE |
| B | YOU | N | EMERGENCY | NUM_5 | THANK YOU |
| C | WE | O | MOTHER | NUM_6 | SORRY |
| D | NEED | P | FATHER | NUM_7 | UNDERSTAND |
| E | WANT | Q | CHILD | NUM_8 | TODAY |
| F | HELP | R | FAMILY | NUM_9 | TOMORROW |
| G | GO | S | FOOD | NOTHING | (no sign) |
| H | COME | T | WATER | | |
| I | DOCTOR | U | TOILET | | |
| J | HOSPITAL | V | SLEEP | | |
| K | MEDICINE | W | HOME | | |
| L | SICK | X | NOW | | |
| | | Y | WHERE | | |
| | | Z | WHAT | | |
| | | NUM_1 | NOT | | |
| | | NUM_2 | YES | | |
| | | NUM_3 | NO | | |

---

## Architecture

### Sign Recognition
- **Input:** 126-dimensional landmark vector (21 landmarks × 3 coords × 2 hands)
- **Normalisation:** Wrist-relative, scale-invariant (background/lighting independent)
- **Model:** MLP — 126 → 512 → 256 → 128 → N classes
- **Parameters:** <200K
- **Training accuracy:** 100% (val), 99.88% (test)
- **Real-world accuracy:** >85%

### Emotion Recognition
- **Model:** ResNet CNN fine-tuned on RAF-DB
- **Classes:** angry, disgust, fear, happy, neutral, sad, surprise
- **Validation accuracy:** 96.04%

### Sentence Generation
- **Model:** Flan-T5-Large (zero-shot, no fine-tuning)
- **Input:** word buffer (e.g. `["I", "NEED", "HELP"]`)
- **Output:** grammatical sentence (`"I need help."`)

### Speech Synthesis
- **Providers:** Chatterbox-TTS (local), ElevenLabs (`eleven_flash_v2_5`)
- **Emotion profiles:** 7 (one per emotion class)
- **Parameters tuned:** exaggeration, CFG weight

---

## Community Sign Contribution

New signs can be added via the web interface at `/contribute`:

**Gate 1 — Word check:** Instant lookup. If word exists, shows reference GIF.

**Gate 2 — Gesture check:** Live webcam detects if your gesture collides with an existing sign (>75% confidence match triggers warning).

**Recording:** Collect 100–200 samples. Reference GIF captured automatically.

**Retraining:** Full MLP retrain triggers automatically in background (~45 seconds on GPU). Model reloads into memory without server restart.

---

## Domain Gap Analysis

Standard CNN+Transformer trained on ASL Alphabet Kaggle dataset (studio images):
- Validation accuracy: **99.99%**
- Real-world accuracy: **~20%**

SentiSign landmark MLP trained on webcam data:
- Validation accuracy: **99.88%**
- Real-world accuracy: **>85%**

The gap in the standard approach is caused by distribution mismatch between studio backgrounds and real webcam conditions. Landmark-based features are invariant to background, lighting, and skin tone — only hand geometry matters.

---

## Key Dependencies

| Package | Version | Reason pinned |
|---------|---------|---------------|
| torch | 2.6.0+cu124 | CUDA 12.4 compatibility |
| numpy | <2.0 | mediapipe ABI compatibility |
| mediapipe | 0.10.9 | Last stable version before breaking API change |
| onnx | 1.14.1 | protobuf 3.x compatibility |
| protobuf | 3.20.3 | mediapipe requirement |
| elevenlabs | latest | Cloud TTS provider (`eleven_flash_v2_5`) |

---

## Team

| Name | Role |
|------|------|
| Omar Sheriff H | Student, ECE |
| P S Arjun | Student, ECE |
| Sree Harinandan | Student, ECE |
| Sudipto Bagchi | Student, ECE |
| Jasmin Sebastin | Guide, Asst. Professor |

Rajagiri School of Engineering and Technology, Kerala, India

---

## Citation

If you use SentiSign in your research:

```
@misc{sentisign2026,
  title  = {SentiSign: A Multimodal Sign Language and Emotion-Aware Speech Synthesis System},
  author = {Omar Sheriff H and P S Arjun and Sree Harinandan and Sudipto Bagchi},
  year   = {2026},
  note   = {Final Year Project, Rajagiri School of Engineering and Technology}
}
```

---

## License

This project is for academic and research purposes.
