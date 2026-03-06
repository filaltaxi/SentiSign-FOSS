from __future__ import annotations

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
import re
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
from dataclasses import asdict, dataclass

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
from typing import Any, List, Optional

import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader, Dataset

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
TEMPORAL_MODEL_DIR = os.path.join(_ROOT, 'models', 'temporal')
TEMPORAL_MODEL_PATH = os.path.join(TEMPORAL_MODEL_DIR, 'temporal_lstm.pth')
TEMPORAL_LABEL_PATH = os.path.join(TEMPORAL_MODEL_DIR, 'temporal_label_map.json')
TEMPORAL_PLANNED_PATH = os.path.join(TEMPORAL_MODEL_DIR, 'temporal_planned_words.json')
TEMPORAL_DATASET_DIR = os.path.join(_ROOT, 'data', 'temporal', 'asl_dataset')
TEMPORAL_N_FRAMES = 60
TEMPORAL_INPUT_DIM = 126
TEMPORAL_REPS_TARGET = 15
TEMPORAL_TRAIN_DEFAULT_EPOCHS = 100
TEMPORAL_TRAIN_DEFAULT_PATIENCE = 15
TEMPORAL_TRAIN_DEFAULT_VAL_SPLIT = 0.15
TEMPORAL_TRAIN_DEFAULT_LR = 3e-4
TEMPORAL_TRAIN_DEFAULT_BATCH = 32
TEMPORAL_TRAIN_DEFAULT_STRIDE = 5

AUDIO_DIR    = os.path.join(_ROOT, 'website', 'audio')

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMPORAL_DATASET_DIR, exist_ok=True)

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
_temporal_model = None
_temporal_meta = None
_temporal_lock = threading.Lock()

# Retrain state
_retrain_status  = {'state': 'idle', 'message': ''}
_retrain_lock    = threading.Lock()
_models_ready    = False
_temporal_train_status = {
    'state': 'idle',
    'message': 'Temporal training idle.',
    'progress': 0.0,
    'updated_at': datetime.now().isoformat(),
    'started_at': None,
    'finished_at': None,
    'config': None,
    'diagnostics': None,
}
_temporal_train_lock = threading.Lock()

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
        {'id': 'temporal', 'label': 'Temporal sign model (optional)', 'state': 'pending', 'detail': None},
        {'id': 'emotion', 'label': 'Emotion model', 'state': 'pending', 'detail': None},
        {'id': 'sentence', 'label': 'Sentence model', 'state': 'pending', 'detail': None},
        {'id': 'tts', 'label': 'TTS engine (Chatterbox)', 'state': 'pending', 'detail': None},
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


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)[..., :x.size(-1)]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[..., :x.size(-1)]))
        out = self.drop(out)
        if self.down is not None:
            residual = self.down(residual)
        return self.relu(out + residual)


class TCNBiLSTMAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 126,
        tcn_channels: int = 128,
        tcn_layers: int = 3,
        lstm_hidden: int = 256,
        num_classes: int = 100,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        blocks, in_ch = [], input_dim
        for i in range(tcn_layers):
            blocks.append(TemporalBlock(in_ch, tcn_channels, 3, 2**i, dropout))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        self.bilstm = nn.LSTM(
            tcn_channels,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)
        return self.classifier(context)


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


def get_temporal_device() -> torch.device:
    forced = os.environ.get('SENTISIGN_TEMPORAL_DEVICE', '').strip().lower()
    if forced == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        raise RuntimeError('SENTISIGN_TEMPORAL_DEVICE=cuda but CUDA is unavailable.')
    if forced == 'mps':
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
        raise RuntimeError('SENTISIGN_TEMPORAL_DEVICE=mps but MPS is unavailable.')
    if forced == 'cpu':
        return torch.device('cpu')
    return torch.device('cpu')


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


def normalize_temporal_word(raw: str) -> str:
    word = re.sub(r'\s+', '_', raw.strip().upper())
    word = re.sub(r'_+', '_', word)
    word = re.sub(r'[^A-Z0-9_]', '', word)
    word = re.sub(r'_+', '_', word).strip('_')
    if not word:
        raise ValueError('Word cannot be empty after normalization.')
    return word


def temporal_word_to_display(word: str) -> str:
    return word.replace('_', ' ')


def _temporal_artifacts_available() -> bool:
    return os.path.exists(TEMPORAL_MODEL_PATH) and os.path.exists(TEMPORAL_LABEL_PATH)


def _temporal_load_label_map() -> dict:
    if not os.path.exists(TEMPORAL_LABEL_PATH):
        return {'classes': [], 'label_to_idx': {}, 'idx_to_label': {}}
    with open(TEMPORAL_LABEL_PATH) as f:
        data = json.load(f)
    classes = [str(c) for c in data.get('classes', [])]
    return {
        'classes': classes,
        'label_to_idx': {str(k): int(v) for k, v in data.get('label_to_idx', {}).items()},
        'idx_to_label': {int(k): str(v) for k, v in data.get('idx_to_label', {}).items()},
        'input_dim': int(data.get('input_dim', TEMPORAL_INPUT_DIM)),
        'n_frames': int(data.get('n_frames', TEMPORAL_N_FRAMES)),
    }


def _temporal_load_planned_categories() -> list[dict]:
    if not os.path.exists(TEMPORAL_PLANNED_PATH):
        return []
    with open(TEMPORAL_PLANNED_PATH) as f:
        data = json.load(f)
    categories = []
    for raw in data.get('categories', []):
        name = str(raw.get('name', '')).strip()
        words = [normalize_temporal_word(str(w)) for w in raw.get('words', []) if str(w).strip()]
        if name and words:
            categories.append({'name': name, 'words': words})
    return categories


def _temporal_planned_lookup(categories: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in categories:
        category = str(entry.get('name', '')).strip()
        for word in entry.get('words', []):
            out[normalize_temporal_word(str(word))] = category
    return out


def _temporal_dataset_word_dir(word: str) -> str:
    return os.path.join(TEMPORAL_DATASET_DIR, word)


def _temporal_count_reps(word: str) -> int:
    sign_dir = _temporal_dataset_word_dir(word)
    if not os.path.isdir(sign_dir):
        return 0
    return len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])


def _temporal_list_dataset_words() -> list[str]:
    if not os.path.isdir(TEMPORAL_DATASET_DIR):
        return []
    words = []
    for entry in os.listdir(TEMPORAL_DATASET_DIR):
        path = os.path.join(TEMPORAL_DATASET_DIR, entry)
        if os.path.isdir(path):
            words.append(entry)
    return sorted(words)


def _temporal_save_rep(word: str, frames: np.ndarray) -> int:
    sign_dir = _temporal_dataset_word_dir(word)
    os.makedirs(sign_dir, exist_ok=True)
    next_idx = _temporal_count_reps(word) + 1
    path = os.path.join(sign_dir, f'sample_{next_idx:03d}.npy')
    np.save(path, frames.astype(np.float32))
    return _temporal_count_reps(word)


def load_temporal():
    global _temporal_model, _temporal_meta
    if _temporal_model is not None and _temporal_meta is not None:
        return _temporal_model, _temporal_meta

    if not os.path.exists(TEMPORAL_MODEL_PATH):
        raise FileNotFoundError('Temporal checkpoint not found.')

    device = get_temporal_device()
    with _temporal_lock:
        if _temporal_model is not None and _temporal_meta is not None:
            return _temporal_model, _temporal_meta

        ckpt = torch.load(TEMPORAL_MODEL_PATH, map_location=device)
        classes = [str(c) for c in ckpt.get('classes', [])]
        num_classes = int(ckpt.get('num_classes', len(classes)))
        model = TCNBiLSTMAttention(
            input_dim=int(ckpt.get('input_dim', TEMPORAL_INPUT_DIM)),
            tcn_channels=int(ckpt.get('tcn_channels', 128)),
            tcn_layers=int(ckpt.get('tcn_layers', 3)),
            lstm_hidden=int(ckpt.get('lstm_hidden', 256)),
            num_classes=num_classes,
            dropout=float(ckpt.get('dropout', 0.35)),
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval().to(device)

        _temporal_model = model
        _temporal_meta = {
            'input_dim': int(ckpt.get('input_dim', TEMPORAL_INPUT_DIM)),
            'n_frames': int(ckpt.get('n_frames', TEMPORAL_N_FRAMES)),
            'num_classes': num_classes,
            'classes': classes,
            'label_to_idx': {str(k): int(v) for k, v in ckpt.get('label_to_idx', {}).items()},
            'idx_to_label': {int(k): str(v) for k, v in ckpt.get('idx_to_label', {}).items()},
        }
        print(f'[API] Temporal LSTM loaded. Classes: {num_classes}  Device: {device}')
    return _temporal_model, _temporal_meta


def _temporal_train_status_snapshot() -> dict:
    with _temporal_train_lock:
        return copy.deepcopy(_temporal_train_status)


def _set_temporal_train_status(
    *,
    state: Optional[str] = None,
    message: Optional[str] = None,
    progress: Optional[float] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    config: Optional[dict] = None,
    diagnostics: Optional[dict[str, Any]] = None,
) -> None:
    with _temporal_train_lock:
        if state is not None:
            _temporal_train_status['state'] = state
        if message is not None:
            _temporal_train_status['message'] = message
        if progress is not None:
            _temporal_train_status['progress'] = max(0.0, min(1.0, float(progress)))
        if started_at is not None:
            _temporal_train_status['started_at'] = started_at
        if finished_at is not None:
            _temporal_train_status['finished_at'] = finished_at
        if config is not None:
            _temporal_train_status['config'] = config
        if diagnostics is not None:
            _temporal_train_status['diagnostics'] = diagnostics
        _temporal_train_status['updated_at'] = datetime.now().isoformat()


def _parse_temporal_train_request(req: TemporalTrainRequest) -> TemporalTrainConfig:
    epochs = int(req.epochs if req.epochs is not None else TEMPORAL_TRAIN_DEFAULT_EPOCHS)
    patience = int(req.patience if req.patience is not None else TEMPORAL_TRAIN_DEFAULT_PATIENCE)
    val_split = float(req.val_split if req.val_split is not None else TEMPORAL_TRAIN_DEFAULT_VAL_SPLIT)
    lr = float(req.lr if req.lr is not None else TEMPORAL_TRAIN_DEFAULT_LR)
    batch = int(req.batch if req.batch is not None else TEMPORAL_TRAIN_DEFAULT_BATCH)
    stride = int(req.stride if req.stride is not None else TEMPORAL_TRAIN_DEFAULT_STRIDE)

    if epochs < 1 or epochs > 500:
        raise ValueError('epochs must be between 1 and 500.')
    if patience < 1 or patience > epochs:
        raise ValueError('patience must be between 1 and epochs.')
    if not np.isfinite(val_split) or val_split <= 0.0 or val_split >= 0.5:
        raise ValueError('val_split must be > 0 and < 0.5.')
    if not np.isfinite(lr) or lr <= 0.0 or lr > 0.1:
        raise ValueError('lr must be > 0 and <= 0.1.')
    if batch < 1 or batch > 512:
        raise ValueError('batch must be between 1 and 512.')
    if stride < 1 or stride > TEMPORAL_N_FRAMES:
        raise ValueError(f'stride must be between 1 and {TEMPORAL_N_FRAMES}.')

    return TemporalTrainConfig(
        epochs=epochs,
        patience=patience,
        val_split=val_split,
        lr=lr,
        batch=batch,
        stride=stride,
    )


class _TemporalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.augment:
            n = x.shape[0]

            speed = np.random.uniform(0.75, 1.25)
            center = n / 2.0
            half = (n / 2.0) * speed
            start = max(0, int(center - half))
            end = min(n - 1, int(center + half))
            indices = np.clip(np.linspace(start, end, n).astype(np.int64), 0, n - 1)
            x = x[indices]

            shift = int(np.random.randint(0, 3))
            if shift > 0:
                x = torch.roll(x, shifts=shift, dims=0)

            x = x + torch.randn_like(x) * 0.008

            if np.random.random() < 0.3:
                drop_start = int(np.random.randint(0, max(1, n - 4)))
                drop_len = int(np.random.randint(1, 4))
                x[drop_start:drop_start + drop_len] = 0.0

            if np.random.random() < 0.5:
                x = torch.cat([x[:, 63:], x[:, :63]], dim=1)

        return x, self.y[idx]


def _load_temporal_training_windows(stride: int) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    if not os.path.isdir(TEMPORAL_DATASET_DIR):
        raise ValueError(f'Temporal dataset directory not found: {TEMPORAL_DATASET_DIR}')

    sign_dirs = sorted(
        [
            d for d in os.listdir(TEMPORAL_DATASET_DIR)
            if os.path.isdir(os.path.join(TEMPORAL_DATASET_DIR, d))
        ]
    )
    if not sign_dirs:
        raise ValueError(f'No sign folders found in {TEMPORAL_DATASET_DIR}')

    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    classes: list[str] = []
    class_to_idx: dict[str, int] = {}
    sign_counts: dict[str, int] = {}

    for sign in sign_dirs:
        sign_dir = os.path.join(TEMPORAL_DATASET_DIR, sign)
        class_name = normalize_temporal_word(sign)
        samples = sorted([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
        windows_for_sign = 0

        for fname in samples:
            file_path = os.path.join(sign_dir, fname)
            try:
                seq = np.load(file_path).astype(np.float32)
            except Exception:
                continue

            if seq.ndim != 2 or seq.shape[1] != TEMPORAL_INPUT_DIM:
                continue

            raw_len = seq.shape[0]
            if raw_len < TEMPORAL_N_FRAMES:
                pad = np.zeros((TEMPORAL_N_FRAMES - raw_len, TEMPORAL_INPUT_DIM), dtype=np.float32)
                seq = np.vstack([seq, pad])
                raw_len = TEMPORAL_N_FRAMES

            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(classes)
                classes.append(class_name)
            class_idx = class_to_idx[class_name]

            for start in range(0, raw_len - TEMPORAL_N_FRAMES + 1, stride):
                window = seq[start:start + TEMPORAL_N_FRAMES]
                X_all.append(window)
                y_all.append(class_idx)
                windows_for_sign += 1

        if windows_for_sign > 0:
            sign_counts[class_name] = sign_counts.get(class_name, 0) + windows_for_sign

    if not X_all:
        raise ValueError('No valid temporal training samples found.')

    X_np = np.asarray(X_all, dtype=np.float32)
    y_np = np.asarray(y_all, dtype=np.int64)
    return X_np, y_np, classes, sign_counts


def _write_temporal_artifacts(
    *,
    model_state: dict,
    classes: list[str],
    best_val: float,
    diagnostics: Optional[dict[str, Any]] = None,
):
    os.makedirs(TEMPORAL_MODEL_DIR, exist_ok=True)

    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {str(i): c for i, c in enumerate(classes)}

    model_payload = {
        'model_state': model_state,
        'input_dim': TEMPORAL_INPUT_DIM,
        'n_frames': TEMPORAL_N_FRAMES,
        'num_classes': len(classes),
        'tcn_channels': 128,
        'tcn_layers': 3,
        'lstm_hidden': 256,
        'dropout': 0.35,
        'classes': classes,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'model_type': 'tcn_bilstm_attention',
        'best_val_acc': float(best_val),
        'diagnostics': diagnostics or {},
    }
    label_payload = {
        'classes': classes,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'input_dim': TEMPORAL_INPUT_DIM,
        'n_frames': TEMPORAL_N_FRAMES,
        'model_type': 'tcn_bilstm_attention',
    }

    model_tmp = f'{TEMPORAL_MODEL_PATH}.tmp.{uuid.uuid4().hex}'
    label_tmp = f'{TEMPORAL_LABEL_PATH}.tmp.{uuid.uuid4().hex}'
    try:
        torch.save(model_payload, model_tmp)
        with open(label_tmp, 'w') as f:
            json.dump(label_payload, f, indent=2)

        os.replace(model_tmp, TEMPORAL_MODEL_PATH)
        os.replace(label_tmp, TEMPORAL_LABEL_PATH)
    finally:
        if os.path.exists(model_tmp):
            os.remove(model_tmp)
        if os.path.exists(label_tmp):
            os.remove(label_tmp)


def _evaluate_temporal_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    if len(X) == 0:
        return {'count': 0, 'loss': None, 'accuracy': None, 'avg_confidence': None}

    loader = DataLoader(
        _TemporalDataset(X, y, augment=False),
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    total_confidence = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, yb)

            total_loss += float(loss.item()) * len(yb)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_confidence += float(probs.max(dim=1).values.sum().item())
            total_items += len(yb)

    return {
        'count': total_items,
        'loss': (total_loss / total_items) if total_items else None,
        'accuracy': (total_correct / total_items) if total_items else None,
        'avg_confidence': (total_confidence / total_items) if total_items else None,
    }


def _sample_temporal_rep_predictions(
    model: nn.Module,
    classes: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for sign in classes:
            sign_dir = os.path.join(TEMPORAL_DATASET_DIR, sign)
            if not os.path.isdir(sign_dir):
                continue

            files = sorted([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
            if not files:
                continue

            seq = np.load(os.path.join(sign_dir, files[0])).astype(np.float32)
            if seq.shape != (TEMPORAL_N_FRAMES, TEMPORAL_INPUT_DIM):
                continue

            logits = model(torch.from_numpy(seq).unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            top_idx = int(np.argmax(probs))
            samples.append(
                {
                    'sign': sign,
                    'sample': files[0],
                    'predicted': classes[top_idx],
                    'confidence': float(probs[top_idx]),
                }
            )

    return samples


def _load_temporal_checkpoint_model(model_path: str, device: torch.device) -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(model_path, map_location=device)
    classes = [str(c) for c in ckpt.get('classes', [])]
    model = TCNBiLSTMAttention(
        input_dim=int(ckpt.get('input_dim', TEMPORAL_INPUT_DIM)),
        tcn_channels=int(ckpt.get('tcn_channels', 128)),
        tcn_layers=int(ckpt.get('tcn_layers', 3)),
        lstm_hidden=int(ckpt.get('lstm_hidden', 256)),
        num_classes=int(ckpt.get('num_classes', len(classes))),
        dropout=float(ckpt.get('dropout', 0.35)),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, classes


def _run_temporal_training(config: TemporalTrainConfig):
    from sklearn.model_selection import train_test_split

    _set_temporal_train_status(
        state='training',
        message='Preparing temporal dataset...',
        progress=0.03,
    )

    X_all, y_all, classes, _ = _load_temporal_training_windows(config.stride)
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError('Need at least 2 classes with valid reps for temporal training.')

    class_counts = np.bincount(y_all, minlength=num_classes)
    if int(class_counts.min()) < 2:
        sparse = [classes[i] for i, count in enumerate(class_counts.tolist()) if count < 2]
        raise ValueError(f'Need at least 2 windows per class for stratified split: {", ".join(sparse)}')

    _set_temporal_train_status(
        state='training',
        message=f'Temporal windows ready: {len(X_all)} samples across {num_classes} classes.',
        progress=0.15,
    )

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=config.val_split,
            stratify=y_all,
            random_state=42,
        )
    except ValueError as exc:
        raise ValueError(f'Train/val split failed: {exc}') from exc

    device = get_temporal_device()
    model = TCNBiLSTMAttention(
        input_dim=TEMPORAL_INPUT_DIM,
        tcn_channels=128,
        tcn_layers=3,
        lstm_hidden=256,
        num_classes=num_classes,
        dropout=0.35,
    ).to(device)

    train_loader = DataLoader(
        _TemporalDataset(X_train, y_train, augment=True),
        batch_size=config.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        _TemporalDataset(X_val, y_val, augment=False),
        batch_size=max(16, min(64, config.batch * 2)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    train_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (train_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.from_numpy(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=min(20, max(5, config.epochs)),
        T_mult=2,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    _set_temporal_train_status(
        state='training',
        message=f'Temporal training started ({config.epochs} epochs)...',
        progress=0.2,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(yb)
            train_correct += int((logits.argmax(dim=1) == yb).sum().item())
            train_total += len(yb)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * len(yb)
                correct += int((preds == yb).sum().item())
                total += len(yb)

        train_acc = (train_correct / train_total) if train_total else 0.0
        val_acc = (correct / total) if total else 0.0
        epoch_row = {
            'epoch': epoch,
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss': (train_loss / train_total) if train_total else None,
            'train_acc': train_acc,
            'val_loss': (val_loss / total) if total else None,
            'val_acc': val_acc,
        }
        history.append(epoch_row)

        is_better_val = val_acc > best_val
        is_equal_val = np.isclose(val_acc, best_val)
        is_better_loss = epoch_row['val_loss'] is not None and epoch_row['val_loss'] < best_val_loss

        if is_better_val or (is_equal_val and is_better_loss):
            best_val = val_acc
            best_val_loss = float(epoch_row['val_loss']) if epoch_row['val_loss'] is not None else float('inf')
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        progress = 0.2 + 0.7 * (epoch / config.epochs)
        _set_temporal_train_status(
            state='training',
            message=(
                f'Epoch {epoch}/{config.epochs} '
                f'(train={train_acc * 100:.1f}%, val={val_acc * 100:.1f}%, best={best_val * 100:.1f}%).'
            ),
            progress=progress,
            diagnostics={
                'dataset': {
                    'num_classes': num_classes,
                    'total_windows': int(len(X_all)),
                    'train_windows': int(len(X_train)),
                    'val_windows': int(len(X_val)),
                    'classes': classes,
                    'windows_per_class': {classes[i]: int(class_counts[i]) for i in range(num_classes)},
                },
                'history_tail': history[-10:],
                'best_epoch': best_epoch,
                'best_val_acc': best_val,
            },
        )

        if no_improve >= config.patience:
            break

    if best_state is None:
        raise RuntimeError('Temporal training failed to produce a checkpoint.')

    _set_temporal_train_status(
        state='training',
        message='Saving temporal artifacts...',
        progress=0.93,
    )
    model.load_state_dict(best_state)
    in_memory_diagnostics = {
        'train': _evaluate_temporal_model(model, X_train, y_train, device, batch_size=config.batch),
        'val': _evaluate_temporal_model(model, X_val, y_val, device, batch_size=config.batch),
        'sample_predictions': _sample_temporal_rep_predictions(model, classes, device),
    }
    training_diagnostics = {
        'dataset': {
            'num_classes': num_classes,
            'total_windows': int(len(X_all)),
            'train_windows': int(len(X_train)),
            'val_windows': int(len(X_val)),
            'classes': classes,
            'windows_per_class': {classes[i]: int(class_counts[i]) for i in range(num_classes)},
        },
        'config': asdict(config),
        'best_epoch': best_epoch,
        'best_val_acc': best_val,
        'history_tail': history[-10:],
        'in_memory': in_memory_diagnostics,
    }
    _write_temporal_artifacts(
        model_state=best_state,
        classes=classes,
        best_val=best_val,
        diagnostics=training_diagnostics,
    )

    _set_temporal_train_status(
        state='training',
        message='Verifying saved checkpoint...',
        progress=0.97,
    )
    checkpoint_model, checkpoint_classes = _load_temporal_checkpoint_model(TEMPORAL_MODEL_PATH, device)
    checkpoint_diagnostics = {
        'train': _evaluate_temporal_model(checkpoint_model, X_train, y_train, device, batch_size=config.batch),
        'val': _evaluate_temporal_model(checkpoint_model, X_val, y_val, device, batch_size=config.batch),
        'sample_predictions': _sample_temporal_rep_predictions(checkpoint_model, checkpoint_classes, device),
    }

    _set_temporal_train_status(
        state='training',
        message='Reloading temporal model...',
        progress=0.985,
    )
    with _temporal_lock:
        global _temporal_model, _temporal_meta
        _temporal_model = None
        _temporal_meta = None
    load_temporal()

    final_diagnostics = {
        **training_diagnostics,
        'checkpoint': checkpoint_diagnostics,
    }
    _set_temporal_train_status(
        state='idle',
        message=(
            f'Temporal model updated ({num_classes} classes, best val={best_val * 100:.2f}%, '
            f'checkpoint val={checkpoint_diagnostics["val"]["accuracy"] * 100:.2f}%).'
        ),
        progress=1.0,
        finished_at=datetime.now().isoformat(),
        diagnostics=final_diagnostics,
    )


def _background_temporal_train(config: TemporalTrainConfig):
    try:
        _run_temporal_training(config)
    except Exception as e:
        print(f'[TemporalTrain] ERROR: {e}')
        _set_temporal_train_status(
            state='error',
            message=f'Temporal training failed: {e}',
            progress=0.0,
            finished_at=datetime.now().isoformat(),
        )


def temporal_models_overview() -> dict:
    lstm_available = _temporal_artifacts_available()
    n_frames = TEMPORAL_N_FRAMES
    trained_count = 0

    if lstm_available:
        try:
            label_map = _temporal_load_label_map()
            n_frames = int(label_map.get('n_frames', TEMPORAL_N_FRAMES))
            trained_count = len(label_map.get('classes', []))
        except Exception:
            lstm_available = False

    return {
        'models': [
            {'id': 'mlp', 'label': 'Landmark MLP', 'available': True},
            {
                'id': 'lstm',
                'label': 'Temporal LSTM',
                'available': lstm_available,
                'n_frames': n_frames,
                'trained_count': trained_count,
            },
        ],
        'default': 'mlp',
    }


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


def _refresh_sentence_startup_label():
    try:
        from sentence_model import get_backend_label

        label = f"Sentence model ({get_backend_label()})"
    except Exception:
        label = "Sentence model"

    with _startup_lock:
        for step in _startup_status['steps']:
            if step['id'] == 'sentence':
                step['label'] = label
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
    with _temporal_train_lock:
        snap['temporal_train'] = copy.deepcopy(_temporal_train_status)
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

        _startup_set_overall('loading', 'Checking temporal model...')
        _startup_set_step('temporal', 'loading')
        try:
            if _temporal_artifacts_available():
                load_temporal()
                _startup_set_step('temporal', 'done')
            else:
                _startup_set_step('temporal', 'done', 'Optional model not configured.')
        except Exception as e:
            # Temporal is optional; do not block backend readiness.
            _startup_set_step('temporal', 'done', f'Optional model unavailable: {e}')

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
    _refresh_sentence_startup_label()
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

class SignCheckRequest(BaseModel):
    word: str

class SignAddRequest(BaseModel):
    word: str
    sign_class: Optional[str] = None   # custom class key, auto-generated if None
    samples: List[List[float]]         # list of 126-feature vectors
    gif_frames: Optional[List[str]] = None   # base64 JPEG frames


class TemporalSignCheckRequest(BaseModel):
    word: str


class TemporalRepAddRequest(BaseModel):
    word: str
    frames: List[List[float]]


class TemporalRecogniseRequest(BaseModel):
    sequence: List[List[float]]


class TemporalTrainRequest(BaseModel):
    epochs: Optional[int] = None
    patience: Optional[int] = None
    val_split: Optional[float] = None
    lr: Optional[float] = None
    batch: Optional[int] = None
    stride: Optional[int] = None


@dataclass(frozen=True)
class TemporalTrainConfig:
    epochs: int
    patience: int
    val_split: float
    lr: float
    batch: int
    stride: int


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


def _validate_temporal_frames(frames: list[list[float]], expected_frames: int) -> np.ndarray:
    if len(frames) != expected_frames:
        raise HTTPException(400, f'Expected {expected_frames} frames, got {len(frames)}')
    matrix = np.asarray(frames, dtype=np.float32)
    if matrix.shape != (expected_frames, TEMPORAL_INPUT_DIM):
        raise HTTPException(
            400,
            f'Expected shape [{expected_frames}, {TEMPORAL_INPUT_DIM}], got {list(matrix.shape)}',
        )
    if not np.isfinite(matrix).all():
        raise HTTPException(400, 'Frames contain non-finite values.')
    return matrix


@app.get('/api/models')
async def get_models():
    return temporal_models_overview()


@app.get('/api/temporal/planned')
async def get_temporal_planned():
    return {'categories': _temporal_load_planned_categories()}


@app.get('/api/temporal/status')
async def get_temporal_status():
    label_map = _temporal_load_label_map()
    planned = _temporal_load_planned_categories()
    planned_lookup = _temporal_planned_lookup(planned)
    trained = sorted({normalize_temporal_word(w) for w in label_map.get('classes', [])})
    dataset_words = _temporal_list_dataset_words()
    all_words = sorted(set(trained) | set(dataset_words) | set(planned_lookup.keys()))

    collection = []
    for word in all_words:
        reps_collected = _temporal_count_reps(word)
        collection.append(
            {
                'word': word,
                'reps_collected': reps_collected,
                'reps_target': TEMPORAL_REPS_TARGET,
                'is_planned': word in planned_lookup,
                'is_trained': word in trained,
                'category': planned_lookup.get(word),
            }
        )

    return {'trained': trained, 'collection': collection}


@app.get('/api/temporal/signs')
async def get_temporal_signs():
    label_map = _temporal_load_label_map()
    signs = []
    for cls in sorted({normalize_temporal_word(w) for w in label_map.get('classes', [])}):
        signs.append({'class': cls, 'word': temporal_word_to_display(cls)})
    return {'signs': signs}


@app.post('/api/temporal/signs/check')
async def check_temporal_sign(req: TemporalSignCheckRequest):
    try:
        word = normalize_temporal_word(req.word)
    except ValueError as e:
        raise HTTPException(400, str(e))

    label_map = _temporal_load_label_map()
    planned = _temporal_load_planned_categories()
    planned_lookup = _temporal_planned_lookup(planned)
    trained = {normalize_temporal_word(w) for w in label_map.get('classes', [])}
    reps_collected = _temporal_count_reps(word)

    return {
        'word': word,
        'exists_trained': word in trained,
        'exists_in_dataset': reps_collected > 0,
        'reps_collected': reps_collected,
        'reps_target': TEMPORAL_REPS_TARGET,
        'is_planned': word in planned_lookup,
        'category': planned_lookup.get(word),
    }


@app.post('/api/temporal/reps/add')
async def add_temporal_rep(req: TemporalRepAddRequest):
    try:
        word = normalize_temporal_word(req.word)
    except ValueError as e:
        raise HTTPException(400, str(e))

    frames = _validate_temporal_frames(req.frames, TEMPORAL_N_FRAMES)
    reps_collected = _temporal_save_rep(word, frames)
    return {
        'ok': True,
        'word': word,
        'reps_collected': reps_collected,
        'reps_target': TEMPORAL_REPS_TARGET,
    }


@app.get('/api/temporal/train/status')
async def get_temporal_train_status():
    return _temporal_train_status_snapshot()


@app.post('/api/temporal/train')
async def start_temporal_train(req: Optional[TemporalTrainRequest] = None):
    try:
        config = _parse_temporal_train_request(req or TemporalTrainRequest())
    except ValueError as e:
        raise HTTPException(400, str(e))

    now = datetime.now().isoformat()
    with _temporal_train_lock:
        if _temporal_train_status.get('state') == 'training':
            return {'started': False, 'message': 'Temporal training is already in progress.'}
        _temporal_train_status['state'] = 'training'
        _temporal_train_status['message'] = 'Temporal training queued...'
        _temporal_train_status['progress'] = 0.0
        _temporal_train_status['updated_at'] = now
        _temporal_train_status['started_at'] = now
        _temporal_train_status['finished_at'] = None
        _temporal_train_status['config'] = asdict(config)
        _temporal_train_status['diagnostics'] = None

    threading.Thread(target=_background_temporal_train, args=(config,), daemon=True).start()
    return {'started': True}


@app.post('/api/temporal/recognise')
async def recognise_temporal(req: TemporalRecogniseRequest):
    if not _temporal_artifacts_available():
        raise HTTPException(503, 'Temporal model artifacts are not available.')

    model, meta = load_temporal()
    device = next(model.parameters()).device
    n_frames = int(meta.get('n_frames', TEMPORAL_N_FRAMES))
    input_dim = int(meta.get('input_dim', TEMPORAL_INPUT_DIM))
    if input_dim != TEMPORAL_INPUT_DIM:
        raise HTTPException(500, f'Unexpected temporal input_dim={input_dim}; expected {TEMPORAL_INPUT_DIM}.')

    sequence = _validate_temporal_frames(req.sequence, n_frames)
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    classes = list(meta.get('classes', []))
    if not classes or len(classes) != probs.shape[0]:
        raise HTTPException(500, 'Temporal model metadata/classes are invalid.')

    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [[classes[int(i)], float(probs[int(i)])] for i in top5_idx]
    return {
        'class': top_class,
        'word': temporal_word_to_display(top_class),
        'confidence': float(probs[top_idx]),
        'top5': top5,
    }


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
    """Generate sentence from words + synthesise speech. Returns audio (wav)."""
    if not req.words:
        raise HTTPException(400, 'No words provided')
    try:
        from generate_sentence import generate_sentence
        sentence = generate_sentence(req.words)
    except Exception as e:
        raise HTTPException(500, f'Sentence generation failed: {e}')

    try:
        from tts import get_output_extension, speak_and_save
    except Exception as e:
        raise HTTPException(500, f'TTS setup failed: {e}')

    try:
        ext = get_output_extension()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f'sentisign_{req.emotion}_{timestamp}.{ext}'
        filepath  = os.path.join(AUDIO_DIR, filename)
        speak_and_save(
            sentence,
            req.emotion,
            filepath,
            also_play=False,
        )
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
            'X-TTS-Engine': 'chatterbox',
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
        'filename': job.get('filename'),
        'audio_url': job.get('audio_url'),
        'error': job.get('error'),
    }


def _run_tts_job(
    job_id: str,
    sentence: str,
    emotion: str,
    filepath: str,
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
        from tts import get_output_extension
    except Exception as e:
        raise HTTPException(500, f'TTS setup failed: {e}')

    ext = get_output_extension()

    job_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'sentisign_{req.emotion}_{timestamp}_{job_id[:8]}.{ext}'
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
