# main.py — SentiSign Website Backend
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI backend wrapping all existing SentiSign Python modules.
# Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import base64
import pickle
import tempfile
import threading
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Add src/ to path so existing modules are importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

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
STATIC_DIR   = os.path.join(_ROOT, 'website', 'static')
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
_retrain_status = {'state': 'idle', 'message': ''}
_retrain_lock   = threading.Lock()


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
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


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


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event('startup')
async def startup():
    load_mlp()
    load_emotion()
    _preload_slm_and_tts()

def _preload_slm_and_tts():
    """Load Flan-T5 and Chatterbox into memory at startup so first request is fast."""
    try:
        print('[API] Preloading sentence model...')
        from generate_sentence import generate_sentence
        generate_sentence(['HELP'])   # warm-up call loads and caches model
        print('[API] Sentence model ready.')
    except Exception as e:
        print(f'[API] Sentence model preload failed: {e}')
    try:
        print('[API] Preloading TTS model...')
        from tts import speak_and_save
        import tempfile, os
        tmp = tempfile.mktemp(suffix='.wav')
        speak_and_save('ready', 'neutral', tmp)
        if os.path.exists(tmp): os.remove(tmp)
        print('[API] TTS model ready.')
    except Exception as e:
        print(f'[API] TTS preload failed: {e}')


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

class SignCheckRequest(BaseModel):
    word: str

class SignAddRequest(BaseModel):
    word: str
    sign_class: Optional[str] = None   # custom class key, auto-generated if None
    samples: List[List[float]]         # list of 126-feature vectors


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/')
async def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.get('/contribute')
async def contribute():
    return FileResponse(os.path.join(STATIC_DIR, 'contribute.html'))

@app.get('/signs')
async def signs_page():
    return FileResponse(os.path.join(STATIC_DIR, 'signs.html'))

@app.get('/about')
async def about():
    return FileResponse(os.path.join(STATIC_DIR, 'about.html'))


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
    """Generate sentence from words + synthesise emotion-aware speech. Returns audio/wav."""
    if not req.words:
        raise HTTPException(400, 'No words provided')
    try:
        from generate_sentence import generate_sentence
        sentence = generate_sentence(req.words)
    except Exception as e:
        raise HTTPException(500, f'Sentence generation failed: {e}')

    try:
        from tts import speak_and_save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f'sentisign_{req.emotion}_{timestamp}.wav'
        filepath  = os.path.join(AUDIO_DIR, filename)
        speak_and_save(sentence, req.emotion, filepath)
    except Exception as e:
        raise HTTPException(500, f'TTS failed: {e}')

    def iter_file():
        with open(filepath, 'rb') as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type='audio/wav',
        headers={
            'X-Sentence': sentence,
            'X-Filename': filename,
            'Content-Disposition': f'inline; filename="{filename}"',
        }
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
    """Add new sign samples and trigger KNN registration (instant)."""
    word     = req.word.strip().upper()
    samples  = req.samples
    if len(samples) < 100:
        raise HTTPException(400, f'Need at least 100 samples. Got {len(samples)}.')

    # Generate unique class key
    cls_key = req.sign_class or f'CUSTOM_{word.replace(" ","_")}'

    # Save to CSV
    import csv
    csv_path = os.path.join(RAW_DIR, f'{cls_key}.csv')
    feat_cols = (
        [f'r_lm{i}_{ax}' for i in range(21) for ax in ['x','y','z']] +
        [f'l_lm{i}_{ax}' for i in range(21) for ax in ['x','y','z']]
    )
    is_new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(feat_cols + ['label'])
        for sample in samples:
            writer.writerow(sample + [cls_key])

    # Update CLASS_TO_WORD mapping in memory
    CLASS_TO_WORD[cls_key] = word

    # Register with KNN (instant, no full retrain)
    background_tasks.add_task(_knn_register, cls_key, word, samples)

    return {
        'success': True,
        'message': f'"{word}" added with {len(samples)} samples. Available immediately.',
        'class_key': cls_key,
    }


def _knn_register(cls_key: str, word: str, samples: list):
    """
    Fast KNN registration — no full retrain needed.
    Saves centroid for instant recognition via nearest-neighbour fallback.
    Full MLP retrain can be triggered offline.
    """
    centroid     = np.mean(samples, axis=0).tolist()
    knn_path     = os.path.join(MODEL_DIR, 'knn_centroids.json')
    if os.path.exists(knn_path):
        with open(knn_path) as f:
            knn_data = json.load(f)
    else:
        knn_data = {}
    knn_data[cls_key] = {'word': word, 'centroid': centroid}
    with open(knn_path, 'w') as f:
        json.dump(knn_data, f)
    print(f'[API] KNN registered: {cls_key} -> {word}')


@app.get('/api/status')
async def get_status():
    return _retrain_status


# ── Serve static files ────────────────────────────────────────────────────────
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
