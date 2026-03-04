# main.py — SentiSign Website Backend
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI backend wrapping all existing SentiSign Python modules.
# Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import base64
import copy
import pickle
import tempfile
import threading
import uuid
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Allow PyTorch MPS to fall back to CPU for unsupported ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Add src/ to path so existing modules are importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'slm', 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import torch
import torch.nn as nn
import cv2

app = FastAPI(title='SentiSign API', version='4.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(_ROOT, 'models', 'landmark')
MLP_PATH     = os.path.join(MODEL_DIR, 'landmark_mlp.pth')
LABEL_PATH   = os.path.join(MODEL_DIR, 'label_map.json')
EMO_PATH     = os.path.join(_ROOT, 'models', 'emotion', 'resnet_emotion.pth')
RAW_DIR      = os.path.join(_ROOT, 'data', 'landmarks', 'raw')
REF_DIR      = os.path.join(_ROOT, 'data', 'landmarks', 'references')

AUDIO_DIR    = os.path.join(_ROOT, 'website', 'audio')

os.makedirs(AUDIO_DIR, exist_ok=True)

HAAR_PATH    = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── Vocabulary ────────────────────────────────────────────────────────────────
CLASS_TO_WORD = {
    'A':'I','B':'YOU','C':'WE','D':'NEED','E':'WANT','F':'HELP',
    'G':'GO','H':'COME','I':'DOCTOR','J':'HOSPITAL','K':'MEDICINE',
    'L':'SICK','M':'PAIN','N':'EMERGENCY','O':'MOTHER','P':'FATHER',
    'Q':'CHILD','R':'FAMILY','S':'FOOD','T':'WATER','U':'TOILET',
    'V':'SLEEP','W':'HOME','X':'NOW','Y':'WHERE','Z':'WHAT',
    'NUM_1':'NOT','NUM_2':'YES','NUM_3':'NO','NUM_4':'PLEASE',
    'NUM_5':'THANK YOU','NUM_6':'SORRY','NUM_7':'UNDERSTAND',
    'NUM_8':'TODAY','NUM_9':'TOMORROW','NOTHING': None,
}

EMOTION_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

# ── Cached models ─────────────────────────────────────────────────────────────
_mlp_model    = None
_idx_to_label = None
_emo_model    = None
_face_cascade = None
_device       = None
_label_map    = None

# Retrain state
_retrain_status  = {'state': 'idle', 'message': ''}
_retrain_lock    = threading.Lock()
_models_ready    = False

# Async TTS jobs (in-memory, best-effort for local dev)
_tts_jobs      = {}          # job_id -> job dict (includes internal filepath)
_tts_jobs_lock = threading.Lock()

# Startup progress (so the frontend can show accurate "backend loading" UI).
_startup_lock = threading.Lock()
_startup_thread_started = False
_startup_status = {
    'state': 'starting',  # starting | loading | ready | error
    'message': 'Backend starting...',
    'updated_at': datetime.now().isoformat(),
    'core_ready': False,   # sign+emotion ready
    'ready': False,        # all steps done
    'steps': [
        {'id': 'mlp', 'label': 'Sign model (landmark MLP)', 'state': 'pending', 'detail': None},
        {'id': 'emotion', 'label': 'Emotion model', 'state': 'pending', 'detail': None},
        {'id': 'sentence', 'label': 'Sentence model (flan-t5-large)', 'state': 'pending', 'detail': None},
        {'id': 'tts', 'label': 'TTS provider (Chatterbox/ElevenLabs)', 'state': 'pending', 'detail': None},
        {'id': 'custom_signs', 'label': 'Custom sign mappings', 'state': 'pending', 'detail': None},
    ],
}

# Retrain config
RETRAIN_HIDDEN   = [512, 256, 128]
RETRAIN_DROPOUT  = 0.3
RETRAIN_LR       = 1e-3
RETRAIN_EPOCHS   = 80
RETRAIN_BATCH    = 64
RETRAIN_PATIENCE = 15


# ── MLP Architecture ──────────────────────────────────────────────────────────
class LandmarkMLP(nn.Module):
    def __init__(self, input_dim, hidden, num_classes, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# ── Model loading ─────────────────────────────────────────────────────────────
def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _device = torch.device('mps')
        else:
            _device = torch.device('cpu')
    return _device


def _configure_torch_threads():
    """Tune PyTorch thread counts for CPU-bound ops (override via env vars)."""
    try:
        default_threads = os.cpu_count() or 1
        num_threads = int(os.environ.get('SENTISIGN_TORCH_THREADS', default_threads))
        num_threads = max(1, num_threads)
    except Exception:
        num_threads = 1

    # Set OMP threads for underlying CPU kernels unless user already configured it.
    os.environ.setdefault('OMP_NUM_THREADS', str(num_threads))

    # Interop threads: keep small to reduce overhead.
    try:
        default_interop = min(4, num_threads)
        interop_threads = int(os.environ.get('SENTISIGN_TORCH_INTEROP_THREADS', default_interop))
        interop_threads = max(1, interop_threads)
    except Exception:
        interop_threads = 1

    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


def load_mlp():
    global _mlp_model, _idx_to_label, _label_map
    if _mlp_model is not None:
        return _mlp_model, _idx_to_label
    device = get_device()
    with open(LABEL_PATH) as f:
        _label_map = json.load(f)
    _idx_to_label = {int(k): v for k, v in _label_map['idx_to_label'].items()}
    ckpt   = torch.load(MLP_PATH, map_location=device)
    model  = LandmarkMLP(
        input_dim=ckpt.get('input_dim', 126),
        hidden=ckpt.get('hidden', [512,256,128]),
        num_classes=len(_idx_to_label),
        dropout=ckpt.get('dropout', 0.3)
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)
    _mlp_model = model
    print(f'[API] MLP loaded. Classes: {len(_idx_to_label)}  Device: {device}')
    return _mlp_model, _idx_to_label


def load_emotion():
    global _emo_model, _face_cascade
    if _emo_model is not None:
        return _emo_model, _face_cascade
    try:
        from emotion_detector import _load_emotion_model
        _emo_model    = _load_emotion_model()
        _face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        print('[API] Emotion model loaded.')
    except Exception as e:
        print(f'[API] Emotion model failed: {e}')
    return _emo_model, _face_cascade


# ── Startup progress helpers ──────────────────────────────────────────────────
def _startup_set_overall(state: str, message: str):
    with _startup_lock:
        _startup_status['state'] = state
        _startup_status['message'] = message
        _startup_status['updated_at'] = datetime.now().isoformat()


def _startup_set_step(step_id: str, state: str, detail: str | None = None):
    with _startup_lock:
        for step in _startup_status.get('steps', []):
            if step.get('id') == step_id:
                step['state'] = state
                step['detail'] = detail
                break
        _startup_status['updated_at'] = datetime.now().isoformat()


def _startup_mark_core_ready():
    global _models_ready
    with _startup_lock:
        _startup_status['core_ready'] = True
        _startup_status['updated_at'] = datetime.now().isoformat()
    _models_ready = True


def _startup_mark_ready():
    with _startup_lock:
        _startup_status['ready'] = True
        _startup_status['state'] = 'ready'
        _startup_status['message'] = 'Backend ready.'
        _startup_status['updated_at'] = datetime.now().isoformat()


def _startup_snapshot() -> dict:
    with _startup_lock:
        snap = copy.deepcopy(_startup_status)
    # Include retrain status in the same call (handy for UI).
    with _retrain_lock:
        snap['retrain'] = dict(_retrain_status)
    return snap


def _background_startup_load_all():
    """
    Load heavy models in a background thread so the API can respond to /api/status
    while warming up.
    """
    global _startup_thread_started

    try:
        _startup_set_overall('loading', 'Loading sign model...')
        _startup_set_step('mlp', 'loading')
        try:
            load_mlp()
            _startup_set_step('mlp', 'done')
            _startup_mark_core_ready()
        except Exception as e:
            _startup_set_step('mlp', 'error', str(e))
            raise

        _startup_set_overall('loading', 'Loading emotion model...')
        _startup_set_step('emotion', 'loading')
        try:
            emo_model, _ = load_emotion()
            if emo_model is None:
                _startup_set_step('emotion', 'error', 'Emotion model failed to load.')
            else:
                _startup_set_step('emotion', 'done')
        except Exception as e:
            _startup_set_step('emotion', 'error', str(e))

        _startup_set_overall('loading', 'Preloading sentence model...')
        _startup_set_step('sentence', 'loading')
        try:
            from sentence_model import load_model as load_sentence_model
            load_sentence_model()
            _startup_set_step('sentence', 'done')
        except Exception as e:
            _startup_set_step('sentence', 'error', str(e))

        _startup_set_overall('loading', 'Preloading TTS model...')
        _startup_set_step('tts', 'loading')
        try:
            from tts import load_model as load_tts_model
            load_tts_model()
            _startup_set_step('tts', 'done')
        except Exception as e:
            _startup_set_step('tts', 'error', str(e))

        _startup_set_overall('loading', 'Restoring custom signs...')
        _startup_set_step('custom_signs', 'loading')
        _restore_custom_signs()
        _startup_set_step('custom_signs', 'done')

        _startup_mark_ready()
        print('[API] Startup warmup done.')
    except Exception as e:
        _startup_set_overall('error', f'Startup failed: {e}')
    finally:
        with _startup_lock:
            _startup_thread_started = True


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event('startup')
async def startup():
    _configure_torch_threads()
    global _startup_thread_started
    with _startup_lock:
        # Avoid starting warmup twice in the same process (can happen in some reload modes).
        if _startup_thread_started:
            return
        _startup_thread_started = True
        _startup_status['state'] = 'starting'
        _startup_status['message'] = 'Backend starting...'
        _startup_status['updated_at'] = datetime.now().isoformat()
    threading.Thread(target=_background_startup_load_all, daemon=True).start()

def _restore_custom_signs():
    """Reload any custom signs from label_map.json into CLASS_TO_WORD on startup."""
    try:
        if not os.path.exists(LABEL_PATH):
            return
        with open(LABEL_PATH) as f:
            lm = json.load(f)
        restored = 0
        for cls in lm.get('classes', []):
            if cls not in CLASS_TO_WORD:
                # Custom sign — derive word from class key
                word = cls.replace('CUSTOM_', '').replace('_', ' ')
                CLASS_TO_WORD[cls] = word
                restored += 1
        if restored:
            print(f'[API] Restored {restored} custom sign(s) from label_map.json')
    except Exception as e:
        print(f'[API] Failed to restore custom signs: {e}')


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize_hand(pts_list):
    pts = np.array(pts_list, dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts.flatten().tolist()


# ── Request models ────────────────────────────────────────────────────────────
class LandmarkRequest(BaseModel):
    landmarks: List[float]          # 126 pre-normalised values from browser

class EmotionRequest(BaseModel):
    image: str                      # base64 JPEG face crop

class GenerateRequest(BaseModel):
    words: List[str]
    emotion: str = 'neutral'
    tts_provider: Optional[str] = None
    tts_voice_id: Optional[str] = None
    tts_model_id: Optional[str] = None

class SignCheckRequest(BaseModel):
    word: str

class SignAddRequest(BaseModel):
    word: str
    sign_class: Optional[str] = None   # custom class key, auto-generated if None
    samples: List[List[float]]         # list of 126-feature vectors
    gif_frames: Optional[List[str]] = None   # base64 JPEG frames


FRONTEND_DIST = os.path.join(_ROOT, 'frontend', 'dist')

# Optional mount of built static assets if present
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, 'assets')), name="assets")

# Audio and references mounting (keep existing)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/references", StaticFiles(directory=REF_DIR), name="references")

@app.get('/api/status')
async def get_status():
    """Backend startup + retrain status (used by frontend to show accurate loading UI)."""
    return _startup_snapshot()


@app.post('/api/recognise')
async def recognise(req: LandmarkRequest):
    """Normalise raw landmarks server-side then run MLP — matches training exactly."""
    model, idx_to_label = load_mlp()
    device = get_device()

    raw = np.array(req.landmarks, dtype=np.float32)
    if len(raw) != 126:
        raise HTTPException(400, f'Expected 126 features, got {len(raw)}')

    # Normalise each hand using exact same logic as collect_landmarks.py
    def norm_hand(pts_flat):
        pts = pts_flat.reshape(21, 3)
        pts = pts - pts[0]                        # wrist = origin
        scale = np.max(np.abs(pts))
        if scale > 0:
            pts = pts / scale
        return pts.flatten()

    right_norm = norm_hand(raw[:63])
    left_norm  = norm_hand(raw[63:])

    # If left hand was all zeros (absent), keep as zeros after norm
    if np.allclose(raw[63:], 0):
        left_norm = np.zeros(63, dtype=np.float32)

    features = np.concatenate([right_norm, left_norm]).astype(np.float32)
    tensor   = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    idx        = int(np.argmax(probs))
    cls_name   = idx_to_label.get(idx, '')
    confidence = float(probs[idx])
    word       = CLASS_TO_WORD.get(cls_name)
    return {'class': cls_name, 'word': word, 'confidence': confidence}


@app.post('/api/emotion')
async def detect_emotion(req: EmotionRequest):
    """Classify emotion from base64 face crop."""
    emo_model, _ = load_emotion()
    if emo_model is None:
        return {'emotion': 'neutral', 'confidence': 0.0}
    try:
        import torchvision.transforms as T
        from PIL import Image
        img_bytes = base64.b64decode(req.image)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        transform = T.Compose([
            T.Resize((44,44)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        # Detect face and crop before inference
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return {'emotion': 'neutral', 'confidence': 0.0}
        x, y, w, h = faces[0]
        pad = 20
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        tensor  = transform(pil_img).unsqueeze(0).to(get_device())
        with torch.no_grad():
            out  = emo_model(tensor)
            if isinstance(out, dict):
                out = out.get('logits', list(out.values())[0])
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx   = int(np.argmax(probs))
        return {'emotion': EMOTION_LABELS[idx], 'confidence': float(probs[idx])}
    except Exception as e:
        print(f'[API] Emotion error: {e}')
        return {'emotion': 'neutral', 'confidence': 0.0}


@app.post('/api/generate_and_speak')
async def generate_and_speak(req: GenerateRequest):
    """Generate sentence from words + synthesise speech. Returns audio (wav/mp3)."""
    if not req.words:
        raise HTTPException(400, 'No words provided')
    try:
        from generate_sentence import generate_sentence
        sentence = generate_sentence(req.words)
    except Exception as e:
        raise HTTPException(500, f'Sentence generation failed: {e}')

    try:
        from tts import get_output_extension, resolve_provider, speak_and_save
        provider = resolve_provider(req.tts_provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f'TTS setup failed: {e}')

    try:
        ext = get_output_extension(provider)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f'sentisign_{req.emotion}_{provider}_{timestamp}.{ext}'
        filepath  = os.path.join(AUDIO_DIR, filename)
        speak_and_save(
            sentence,
            req.emotion,
            filepath,
            also_play=False,
            provider=provider,
            voice_id=req.tts_voice_id,
            model_id=req.tts_model_id,
        )
    except Exception as e:
        raise HTTPException(500, f'TTS failed: {e}')

    def iter_file():
        with open(filepath, 'rb') as f:
            yield from f

    media_type = 'audio/mpeg' if filename.lower().endswith('.mp3') else 'audio/wav'

    return StreamingResponse(
        iter_file(),
        media_type=media_type,
        headers={
            'X-Sentence': sentence,
            'X-Filename': filename,
            'X-TTS-Provider': provider,
            'X-TTS-Model': req.tts_model_id or 'default',
            'Content-Disposition': f'inline; filename="{filename}"',
        }
    )


def _tts_job_public(job: dict) -> dict:
    """Return a safe/public view of a job dict."""
    return {
        'job_id': job.get('job_id'),
        'state': job.get('state'),
        'created_at': job.get('created_at'),
        'updated_at': job.get('updated_at'),
        'sentence': job.get('sentence'),
        'emotion': job.get('emotion'),
        'tts_provider': job.get('tts_provider'),
        'tts_voice_id': job.get('tts_voice_id'),
        'tts_model_id': job.get('tts_model_id'),
        'filename': job.get('filename'),
        'audio_url': job.get('audio_url'),
        'error': job.get('error'),
    }


def _run_tts_job(
    job_id: str,
    sentence: str,
    emotion: str,
    filepath: str,
    tts_provider: str | None = None,
    tts_voice_id: str | None = None,
    tts_model_id: str | None = None,
):
    """Background job that renders TTS to disk and updates job state."""
    now = datetime.now().isoformat()
    with _tts_jobs_lock:
        job = _tts_jobs.get(job_id)
        if not job:
            return
        job['state'] = 'running'
        job['updated_at'] = now

    try:
        from tts import speak_and_save
        speak_and_save(
            sentence,
            emotion,
            filepath,
            also_play=False,
            provider=tts_provider,
            voice_id=tts_voice_id,
            model_id=tts_model_id,
        )
        now = datetime.now().isoformat()
        with _tts_jobs_lock:
            job = _tts_jobs.get(job_id)
            if job:
                job['state'] = 'done'
                job['updated_at'] = now
    except Exception as e:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
        now = datetime.now().isoformat()
        with _tts_jobs_lock:
            job = _tts_jobs.get(job_id)
            if job:
                job['state'] = 'error'
                job['error'] = str(e)
                job['updated_at'] = now


@app.post('/api/generate_and_speak_async')
async def generate_and_speak_async(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate sentence now, synthesise speech in the background, and return a job id."""
    if not req.words:
        raise HTTPException(400, 'No words provided')
    try:
        from generate_sentence import generate_sentence
        sentence = generate_sentence(req.words)
    except Exception as e:
        raise HTTPException(500, f'Sentence generation failed: {e}')

    try:
        from tts import get_output_extension, resolve_provider
        provider = resolve_provider(req.tts_provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f'TTS setup failed: {e}')

    ext = get_output_extension(provider)

    job_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'sentisign_{req.emotion}_{provider}_{timestamp}_{job_id[:8]}.{ext}'
    filepath  = os.path.join(AUDIO_DIR, filename)
    audio_url = f'/audio/{filename}'
    status_url = f'/api/tts_jobs/{job_id}'

    job = {
        'job_id': job_id,
        'state': 'queued',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'sentence': sentence,
        'emotion': req.emotion,
        'tts_provider': provider,
        'tts_voice_id': req.tts_voice_id,
        'tts_model_id': req.tts_model_id,
        'filename': filename,
        'audio_url': audio_url,
        'error': None,
        # internal-only
        'filepath': filepath,
    }
    with _tts_jobs_lock:
        _tts_jobs[job_id] = job

    background_tasks.add_task(
        _run_tts_job,
        job_id,
        sentence,
        req.emotion,
        filepath,
        provider,
        req.tts_voice_id,
        req.tts_model_id,
    )
    return _tts_job_public(job) | {'status_url': status_url}


@app.get('/api/tts_jobs/{job_id}')
async def get_tts_job(job_id: str):
    with _tts_jobs_lock:
        job = _tts_jobs.get(job_id)
        if not job:
            raise HTTPException(404, 'Unknown job id')
        return JSONResponse(
            _tts_job_public(job),
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0',
            },
        )


@app.get('/api/signs')
async def get_signs():
    """Return all known sign class → word mappings."""
    with open(LABEL_PATH) as f:
        lm = json.load(f)
    signs = []
    for cls in lm['classes']:
        word    = CLASS_TO_WORD.get(cls)
        gif_key = word.replace(' ', '_') if word else None
        gif_url = f'/api/references/{gif_key}' if gif_key else None
        signs.append({'class': cls, 'word': word, 'gif_url': gif_url})
    return {'signs': [s for s in signs if s['word'] is not None]}


@app.get('/api/references/{word}')
async def get_reference(word: str):
    """Serve reference GIF for a word."""
    gif_path = os.path.join(REF_DIR, f'{word}.gif')
    if not os.path.exists(gif_path):
        raise HTTPException(404, f'No reference GIF for {word}')
    return FileResponse(gif_path, media_type='image/gif')


@app.post('/api/signs/check')
async def check_sign(req: SignCheckRequest):
    """Gate 1: Check if a word already exists in the system."""
    word = req.word.strip().upper()
    existing_words = {v for v in CLASS_TO_WORD.values() if v}
    # Also check label_map for community-added signs
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH) as f:
            lm = json.load(f)
        for cls in lm.get('classes', []):
            w = CLASS_TO_WORD.get(cls) or lm.get('label_to_idx', {}).get(cls)
            if w:
                existing_words.add(str(w).upper())
    exists  = word in existing_words
    gif_key = word.replace(' ', '_')
    gif_url = f'/api/references/{gif_key}' if exists else None
    return {'exists': exists, 'word': word, 'gif_url': gif_url}


@app.post('/api/signs/gate2')
async def gate2_check(req: LandmarkRequest):
    """Gate 2: Run single landmark sample through classifier to detect gesture collision."""
    model, idx_to_label = load_mlp()
    device = get_device()
    features = np.array(req.landmarks, dtype=np.float32)
    tensor   = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx        = int(np.argmax(probs))
    cls_name   = idx_to_label.get(idx, '')
    confidence = float(probs[idx])
    word       = CLASS_TO_WORD.get(cls_name)
    collision  = confidence > 0.75 and word is not None
    return {
        'collision': collision,
        'matched_class': cls_name,
        'matched_word': word,
        'confidence': confidence,
    }


@app.post('/api/signs/add')
async def add_sign(req: SignAddRequest, background_tasks: BackgroundTasks):
    """Add new sign samples and trigger full MLP retrain in background."""
    global _retrain_status

    word    = req.word.strip().upper()
    samples = req.samples
    if len(samples) < 100:
        raise HTTPException(400, f'Need at least 100 samples. Got {len(samples)}.')

    # Check not already retraining
    with _retrain_lock:
        if _retrain_status['state'] == 'retraining':
            raise HTTPException(409, 'Retraining already in progress. Please wait.')

    # Generate unique class key
    cls_key = req.sign_class or f'CUSTOM_{word.replace(" ","_")}'

    # Save samples to CSV
    import csv
    csv_path  = os.path.join(RAW_DIR, f'{cls_key}.csv')
    feat_cols = (
        [f'r_lm{i}_{ax}' for i in range(21) for ax in ['x','y','z']] +
        [f'l_lm{i}_{ax}' for i in range(21) for ax in ['x','y','z']]
    )
    def norm_hand(pts_flat):
        pts = np.array(pts_flat, dtype=np.float32).reshape(21, 3)
        pts = pts - pts[0]
        scale = np.max(np.abs(pts))
        if scale > 0:
            pts = pts / scale
        return pts.flatten().tolist()

    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(feat_cols + ['label'])
        for sample in samples:
            raw = np.array(sample, dtype=np.float32)
            # Normalise each hand to match collect_landmarks.py
            right_norm = norm_hand(raw[:63])
            left_norm  = norm_hand(raw[63:]) if not np.allclose(raw[63:], 0) else [0.0]*63
            writer.writerow(right_norm + left_norm + [cls_key])

    # Update vocab in memory
    CLASS_TO_WORD[cls_key] = word

    # Generate GIF from captured frames or placeholder
    if req.gif_frames and len(req.gif_frames) >= 6:
        _generate_gif_from_frames(word, req.gif_frames)
    else:
        _generate_placeholder_gif(word)

    # Trigger background MLP retrain
    background_tasks.add_task(_background_retrain, cls_key, word)

    return {
        'success'  : True,
        'message'  : f'"{word}" saved. Model retraining started — takes 2-3 minutes.',
        'class_key': cls_key,
        'retraining': True,
    }


def _background_retrain(cls_key: str, word: str):
    """
    Full MLP retrain in background thread.
    Reloads model into memory when done.
    """
    global _retrain_status, _mlp_model, _idx_to_label, _label_map

    with _retrain_lock:
        _retrain_status = {'state': 'retraining', 'message': f'Retraining for "{word}"...'}

    print(f'[Retrain] Starting retrain for {cls_key} -> {word}')

    try:
        import csv
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader

        # ── Load all CSVs ──────────────────────────────────────────
        dfs = []
        for csv_file in sorted(Path(RAW_DIR).glob('*.csv')):
            df = pd.read_csv(csv_file)
            dfs.append(df)
        if not dfs:
            raise ValueError('No CSV files found in raw dir')

        data = pd.concat(dfs, ignore_index=True)
        feat_cols = [c for c in data.columns
                     if c.startswith('r_lm') or c.startswith('l_lm')]

        # ── Encode labels ──────────────────────────────────────────
        le          = LabelEncoder()
        data['idx'] = le.fit_transform(data['label'])
        classes     = list(le.classes_)
        num_classes = len(classes)

        print(f'[Retrain] {len(data)} samples, {num_classes} classes')

        # ── Split ──────────────────────────────────────────────────
        X = data[feat_cols].values.astype(np.float32)
        y = data['idx'].values
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)

        # ── Build model ────────────────────────────────────────────
        device = get_device()
        model  = LandmarkMLP(
            input_dim=len(feat_cols),
            hidden=RETRAIN_HIDDEN,
            num_classes=num_classes,
            dropout=RETRAIN_DROPOUT,
        ).to(device)

        Xt = torch.from_numpy(X_train).to(device)
        yt = torch.from_numpy(y_train).long().to(device)
        Xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val).long().to(device)

        loader    = DataLoader(TensorDataset(Xt, yt),
                               batch_size=RETRAIN_BATCH, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=RETRAIN_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=RETRAIN_EPOCHS)

        best_acc   = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, RETRAIN_EPOCHS + 1):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()

            if acc > best_acc:
                best_acc   = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 20 == 0:
                print(f'[Retrain] Epoch {epoch}/{RETRAIN_EPOCHS}  val={acc*100:.1f}%')

            if no_improve >= RETRAIN_PATIENCE:
                print(f'[Retrain] Early stop at epoch {epoch}')
                break

            # Update progress message
            pct = int(epoch / RETRAIN_EPOCHS * 100)
            with _retrain_lock:
                _retrain_status['message'] = (
                    f'Retraining... {pct}% (val={best_acc*100:.1f}%)')

        model.load_state_dict(best_state)

        # ── Save model ─────────────────────────────────────────────
        idx_to_label = {str(i): c for i, c in enumerate(classes)}
        label_to_idx = {c: i for i, c in enumerate(classes)}
        torch.save({
            'model_state' : best_state,
            'input_dim'   : len(feat_cols),
            'hidden'      : RETRAIN_HIDDEN,
            'num_classes' : num_classes,
            'dropout'     : RETRAIN_DROPOUT,
            'classes'     : classes,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
        }, MLP_PATH)

        new_label_map = {
            'classes'       : classes,
            'label_to_idx'  : label_to_idx,
            'idx_to_label'  : idx_to_label,
            'input_dim'     : len(feat_cols),
        }
        with open(LABEL_PATH, 'w') as f:
            json.dump(new_label_map, f, indent=2)

        # ── Reload into memory ─────────────────────────────────────
        model.eval()
        _mlp_model    = model
        _idx_to_label = {i: c for i, c in enumerate(classes)}
        _label_map    = new_label_map
        print(f'[Retrain] idx_to_label updated: {_idx_to_label}')

        # Update CLASS_TO_WORD for any new custom classes
        for cls in classes:
            if cls.startswith('CUSTOM_') and cls not in CLASS_TO_WORD:
                CLASS_TO_WORD[cls] = cls.replace('CUSTOM_', '').replace('_', ' ')

        print(f'[Retrain] Done. {num_classes} classes. Best val={best_acc*100:.2f}%')
        with _retrain_lock:
            _retrain_status = {
                'state'  : 'idle',
                'message': f'"{word}" added. Model updated ({num_classes} classes, val={best_acc*100:.1f}%)',
            }

    except Exception as e:
        print(f'[Retrain] ERROR: {e}')
        with _retrain_lock:
            _retrain_status = {
                'state'  : 'error',
                'message': f'Retrain failed: {str(e)}',
            }


def _generate_gif_from_frames(word: str, frames_b64: list):
    """Build reference GIF from base64 JPEG frames captured in browser."""
    try:
        from PIL import Image
        from io import BytesIO
        gif_word = word.replace(' ', '_')
        gif_path = os.path.join(REF_DIR, f'{gif_word}.gif')
        pil_frames = []
        for b64 in frames_b64[:60]:   # max 60 frames
            try:
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                img = img.resize((200, 200), Image.LANCZOS)
                pil_frames.append(img)
            except Exception:
                continue
        if len(pil_frames) >= 3:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=50,
                loop=0
            )
            print(f'[API] Real GIF saved: {gif_path} ({len(pil_frames)} frames)')
        else:
            _generate_placeholder_gif(word)
    except Exception as e:
        print(f'[API] GIF from frames failed: {e}')
        _generate_placeholder_gif(word)


def _generate_placeholder_gif(word: str):
    """Create a simple animated placeholder GIF for contributed signs."""
    try:
        from PIL import Image, ImageDraw
        gif_word = word.replace(' ', '_')
        gif_path = os.path.join(REF_DIR, f'{gif_word}.gif')
        if os.path.exists(gif_path):
            # Always regenerate to fix any broken placeholders
            pass
        frames = []
        colors = [(0, 212, 170), (0, 180, 145), (0, 150, 120)]
        for i in range(6):
            img  = Image.new('RGB', (200, 200), color=(17, 23, 32))
            draw = ImageDraw.Draw(img)
            # Animated border — pulses between frames
            c = colors[i % len(colors)]
            draw.rectangle([8, 8, 192, 192], outline=c, width=3)
            draw.rectangle([20, 20, 180, 180], outline=(30, 40, 55), width=1)
            # Hand shape drawn with lines (no emoji needed)
            cx, cy = 100, 90
            # Palm circle
            draw.ellipse([cx-22, cy-18, cx+22, cy+18], outline=c, width=2)
            # Five fingers as lines
            fingers = [(-18,-36),(-9,-40),(0,-42),(9,-40),(18,-36)]
            for fx, fy in fingers:
                draw.line([cx+fx//2, cy-18, cx+fx, cy+fy], fill=c, width=3)
            # Word text — split long words
            display = word[:10] + ('.' if len(word) > 10 else '')
            # Draw word with manual pixel font sizing
            text_y = 140
            draw.rectangle([15, text_y-8, 185, text_y+14], fill=(25, 35, 48))
            draw.text((100, text_y), display, fill=(232, 240, 254), anchor='mm')
            # Community label
            draw.text((100, 168), 'Community Sign', fill=(60, 90, 105), anchor='mm')
            frames.append(img)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,
            loop=0
        )
        print(f'[API] Placeholder GIF created: {gif_path}')
    except Exception as e:
        print(f'[API] GIF generation failed: {e}')


# SPA Fallback for any client routes not caught by API.
# Must be LAST so it doesn't shadow real API endpoints.
@app.get('/{full_path:path}')
async def spa_fallback(full_path: str):
    if (
        full_path.startswith('api/')
        or full_path.startswith('audio/')
        or full_path.startswith('references/')
        or full_path.startswith('assets/')
    ):
        raise HTTPException(404, "Not Found")

    index_file = os.path.join(FRONTEND_DIST, 'index.html')
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(404, "Frontend not built. Run 'npm run build' in the frontend directory.")
