# src/sign_recognizer.py
# ─────────────────────────────────────────────────────────────────────────────
# SentiSign Phase 3 — Combined Sign + Emotion Recognition (Landmark MLP)
#
# Architecture change from CNN+Transformer to Landmark MLP:
#   OLD: Full frame → MediaPipe (bounding box) → crop ROI → CNN+Transformer
#   NEW: Full frame → MediaPipe (21 landmarks) → 63 normalised coords → MLP
#
# Why landmarks:
#   CNN+Transformer trained on studio images failed in real-world (~80% failure)
#   due to domain gap (plain white background vs real webcam background).
#   Landmark coordinates describe hand SHAPE only — background, lighting,
#   skin tone, distance from camera are all irrelevant.
#
# Single webcam session captures BOTH:
#   - ASL hand shapes → vocabulary words (MLP on landmarks)
#   - Facial emotion  → emotion string   (ResNet emotion model)
#
# Returns: (words: list, emotion: str)
#   e.g. (['I', 'NEED', 'HELP'], 'fear')
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import json
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Vocabulary ────────────────────────────────────────────────────────────────
CLASS_TO_WORD = {
    'A': 'I',           'B': 'YOU',         'C': 'WE',
    'D': 'NEED',        'E': 'WANT',        'F': 'HELP',
    'G': 'GO',          'H': 'COME',        'I': 'DOCTOR',
    'J': 'HOSPITAL',    'K': 'MEDICINE',    'L': 'SICK',
    'M': 'PAIN',        'N': 'EMERGENCY',   'O': 'MOTHER',
    'P': 'FATHER',      'Q': 'CHILD',       'R': 'FAMILY',
    'S': 'FOOD',        'T': 'WATER',       'U': 'TOILET',
    'V': 'SLEEP',       'W': 'HOME',        'X': 'NOW',
    'Y': 'WHERE',       'Z': 'WHAT',
    'NUM_1': 'NOT',         'NUM_2': 'YES',         'NUM_3': 'NO',
    'NUM_4': 'PLEASE',      'NUM_5': 'THANK YOU',   'NUM_6': 'SORRY',
    'NUM_7': 'UNDERSTAND',  'NUM_8': 'TODAY',       'NUM_9': 'TOMORROW',
    'NOTHING': None,
}

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Constants ─────────────────────────────────────────────────────────────────
HOLD_FRAMES     = 10      # consecutive frames same sign must be held
FLUSH_SECONDS   = 30.0    # seconds of no new word → auto end
EMOTION_EVERY_N = 5       # run emotion on every Nth frame to save CPU
DEFAULT_EMOTION = 'neutral'

# MediaPipe Hands tuning
# For two-hand models, `static_image_mode=True` tends to detect a newly introduced
# second hand more reliably than tracking mode.
HANDS_DETECT_CONF_SINGLE = 0.70
HANDS_TRACK_CONF_SINGLE = 0.60
HANDS_DETECT_CONF_MULTI = 0.50
HANDS_TRACK_CONF_MULTI = 0.50
HANDS_MODEL_COMPLEXITY = 1

# Sign commit thresholds (fast path + slow path for low-confidence but stable signs)
SIGN_HIGH_CONFIDENCE = 0.60
SIGN_LOW_CONFIDENCE = 0.30
SIGN_LOW_CONF_MARGIN = 0.12
SIGN_LOW_HOLD_FRAMES = 26

# Emotion/face tracking (reduce flicker + adapt faster)
EMO_CONFIDENCE_THRESHOLD = 0.40
EMO_PROBS_EMA_ALPHA = 0.60
EMO_RECENT_WINDOW_SIZE = 28
FACE_DETECT_EVERY_N = 1
FACE_BOX_EMA_ALPHA = 0.45
FACE_BOX_HOLD_FRAMES = 10
FACE_BOX_IOU_HINT_THRESHOLD = 0.15
FACE_ROI_PADDING_RATIO = 0.10

# ── Model paths ───────────────────────────────────────────────────────────────
_MLP_PATH      = os.path.join(_ROOT, 'models', 'landmark', 'landmark_mlp.pth')
_LABEL_PATH    = os.path.join(_ROOT, 'models', 'landmark', 'label_map.json')
_EMO_PATH      = os.path.join(_ROOT, 'models', 'emotion',  'resnet_emotion.pth')
_HAAR_PATH     = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_SIGN_OVERRIDES_PATH = os.path.join(_ROOT, 'models', 'landmark', 'sign_overrides.json')

# ── Cached globals ────────────────────────────────────────────────────────────
_mlp_model    = None
_idx_to_label = None
_mlp_input_dim = None
_emo_model    = None
_face_cascade = None
_device       = None

_clahe = None

# ── MLP architecture (must match train_landmark_classifier.py) ────────────────

class LandmarkMLP(nn.Module):
    def __init__(self, input_dim=63, hidden=[256, 128, 64],
                 num_classes=36, dropout=0.3):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Landmark normalisation (must match collect_landmarks.py) ──────────────────

def _normalize_landmarks(landmarks):
    """
    Normalise 21 MediaPipe landmarks relative to wrist.
    Returns flat numpy array of 63 values.
    Invariant to position, scale, and distance from camera.
    """
    pts = np.array([[lm.x, lm.y, lm.z]
                    for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                          # wrist = origin
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale                       # normalise to [-1, 1]
    return pts.flatten()                   # (63,)


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[sign_recognizer] Device: {_device}')
    return _device


def _load_mlp():
    global _mlp_model, _idx_to_label, _mlp_input_dim
    if _mlp_model is not None:
        return _mlp_model, _idx_to_label

    if not os.path.isfile(_MLP_PATH):
        raise FileNotFoundError(
            f'[sign_recognizer] Landmark MLP not found: {_MLP_PATH}\n'
            f'Run collect_landmarks.py then train_landmark_classifier.py first.'
        )
    if not os.path.isfile(_LABEL_PATH):
        raise FileNotFoundError(
            f'[sign_recognizer] label_map.json not found: {_LABEL_PATH}'
        )

    print('[sign_recognizer] Loading landmark MLP ...')
    device = _get_device()

    with open(_LABEL_PATH) as f:
        lm = json.load(f)
    _idx_to_label = {int(k): v for k, v in lm['idx_to_label'].items()}
    num_classes   = len(_idx_to_label)

    ckpt   = torch.load(_MLP_PATH, map_location=device)
    input_dim = int(ckpt.get('input_dim', 63))
    hidden = ckpt.get('hidden',  [256, 128, 64])
    drop   = ckpt.get('dropout', 0.3)

    model  = LandmarkMLP(input_dim=input_dim, hidden=hidden,
                          num_classes=num_classes, dropout=drop)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)

    _mlp_model = model
    _mlp_input_dim = input_dim
    print(f'[sign_recognizer] Landmark MLP loaded. Classes: {num_classes}')
    _apply_sign_mappings(_idx_to_label)
    return _mlp_model, _idx_to_label


def _load_emotion_model():
    global _emo_model, _face_cascade
    if _emo_model is not None:
        return _emo_model, _face_cascade
    try:
        from emotion_detector import _load_emotion_model as _emo_load_fn
        _emo_model    = _emo_load_fn()
        _face_cascade = cv2.CascadeClassifier(_HAAR_PATH)
        print('[sign_recognizer] Emotion model loaded.')
        return _emo_model, _face_cascade
    except Exception as e:
        print(f'[sign_recognizer] Emotion model failed: {e}')
        return None, None


# ── Inference helpers ─────────────────────────────────────────────────────────

def _apply_sign_mappings(idx_to_label):
    """Extend CLASS_TO_WORD from model classes + optional overrides file."""
    # Auto-map CUSTOM_* classes to a reasonable default word.
    custom_added = 0
    try:
        classes = set(idx_to_label.values()) if isinstance(idx_to_label, dict) else set()
    except Exception:
        classes = set()
    for cls in classes:
        if not isinstance(cls, str):
            continue
        if not cls.startswith('CUSTOM_'):
            continue
        if cls in CLASS_TO_WORD:
            continue
        word = cls.replace('CUSTOM_', '', 1).replace('_', ' ').strip()
        if word:
            CLASS_TO_WORD[cls] = word.upper()
            custom_added += 1

    # Apply explicit overrides if present.
    overrides_applied = 0
    if os.path.isfile(_SIGN_OVERRIDES_PATH):
        try:
            with open(_SIGN_OVERRIDES_PATH) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('sign_overrides.json must be a JSON object: {"CLASS": "WORD"}')
            for cls, word in data.items():
                if not isinstance(cls, str) or not cls.strip():
                    continue
                key = cls.strip()
                if word is None:
                    CLASS_TO_WORD[key] = None
                    overrides_applied += 1
                    continue
                if not isinstance(word, str):
                    continue
                w = word.strip()
                CLASS_TO_WORD[key] = (w.upper() if w else None)
                overrides_applied += 1
        except Exception as e:
            print(f'[sign_recognizer] Warning: failed to load sign overrides: {e}')

    if custom_added or overrides_applied:
        parts = []
        if custom_added:
            parts.append(f'auto-mapped {custom_added} CUSTOM_*')
        if overrides_applied:
            parts.append(f'applied {overrides_applied} overrides')
        print(f'[sign_recognizer] Sign mapping updated ({", ".join(parts)})')


def _load_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _box_area(box):
    _, _, w, h = box
    return max(0, int(w)) * max(0, int(h))


def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = _box_area(a) + _box_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _smooth_box(prev_box, new_box, alpha):
    px, py, pw, ph = prev_box
    nx, ny, nw, nh = new_box
    inv = 1.0 - alpha
    return (
        int(round((px * inv) + (nx * alpha))),
        int(round((py * inv) + (ny * alpha))),
        int(round((pw * inv) + (nw * alpha))),
        int(round((ph * inv) + (nh * alpha))),
    )


def _filter_face_boxes(frame_shape, boxes):
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return boxes

    keep = []
    for x, y, bw, bh in boxes:
        if bw <= 0 or bh <= 0:
            continue
        area = float(bw * bh) / frame_area
        aspect = float(bw) / float(bh)
        if area < 0.003:
            continue
        if area > 0.75:
            continue
        if aspect < 0.55 or aspect > 1.80:
            continue
        keep.append((int(x), int(y), int(bw), int(bh)))
    return keep


def _select_face_box(boxes, hint_box):
    if not boxes:
        return None

    if hint_box is not None:
        best_iou = -1.0
        best = None
        for b in boxes:
            iou = _box_iou(hint_box, b)
            if iou > best_iou:
                best_iou = iou
                best = b
        if best is not None and best_iou >= FACE_BOX_IOU_HINT_THRESHOLD:
            return best

    return max(boxes, key=_box_area)


def _crop_face_roi(frame, box, padding_ratio):
    x, y, w, h = box
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _classify_landmarks(model, idx_to_label, landmarks, device):
    """Run MLP on extracted feature vector. Returns (class_name, top1_conf, top1_minus_top2)."""
    try:
        tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if probs.size >= 2:
            top2 = float(np.partition(probs, -2)[-2])
        else:
            top2 = 0.0
        margin = float(confidence - top2)
        return idx_to_label.get(idx, ''), confidence, margin
    except Exception:
        return '', 0.0, 0.0


def _extract_landmark_features(results, expected_dim: int):
    """
    Extract a flat feature vector from MediaPipe results.

    Supports:
      - 63 dims:  one hand (21 landmarks × xyz)
      - 126 dims: two hands concatenated (Left + Right), zero-padded if missing
    """
    if not results or not getattr(results, 'multi_hand_landmarks', None):
        return None

    hands = results.multi_hand_landmarks
    if expected_dim <= 63:
        return _normalize_landmarks(hands[0].landmark)

    # Build Left/Right feature vectors (63 each). Default to zeros if missing.
    zeros = np.zeros(63, dtype=np.float32)
    left = None
    right = None

    handedness = getattr(results, 'multi_handedness', None)
    if handedness and len(handedness) == len(hands):
        for hand_lm, hand_info in zip(hands, handedness):
            try:
                label = hand_info.classification[0].label  # "Left" / "Right"
            except Exception:
                label = None
            feats = _normalize_landmarks(hand_lm.landmark)
            if label == 'Left' and left is None:
                left = feats
            elif label == 'Right' and right is None:
                right = feats
            else:
                if left is None:
                    left = feats
                elif right is None:
                    right = feats
    else:
        # No handedness info; use detection order.
        if len(hands) >= 1:
            left = _normalize_landmarks(hands[0].landmark)
        if len(hands) >= 2:
            right = _normalize_landmarks(hands[1].landmark)

    base = np.concatenate([left if left is not None else zeros,
                           right if right is not None else zeros])
    if expected_dim == 126:
        return base
    if expected_dim < 126:
        return base[:expected_dim]
    pad = np.zeros(expected_dim - 126, dtype=np.float32)
    return np.concatenate([base, pad])


def _pad_or_truncate(vec: np.ndarray, expected_dim: int) -> np.ndarray:
    if vec.shape[0] == expected_dim:
        return vec
    if vec.shape[0] > expected_dim:
        return vec[:expected_dim]
    pad = np.zeros(expected_dim - vec.shape[0], dtype=np.float32)
    return np.concatenate([vec, pad])


def _extract_landmark_feature_candidates(results, expected_dim: int):
    """
    Build one or more candidate feature vectors for sign classification.

    For two-hand models, MediaPipe handedness can be unreliable; we instead:
    - order hands by wrist x position (leftmost/rightmost in image)
    - try both slot assignments (A,B) and (B,A)
    - let the model confidence choose
    """
    if not results or not getattr(results, 'multi_hand_landmarks', None):
        return None

    hands = results.multi_hand_landmarks
    if expected_dim <= 63:
        feats = _normalize_landmarks(hands[0].landmark)
        return [_pad_or_truncate(feats, expected_dim)]

    zeros = np.zeros(63, dtype=np.float32)

    # One hand: try placing it in either slot.
    if len(hands) == 1:
        f0 = _normalize_landmarks(hands[0].landmark)
        a = _pad_or_truncate(np.concatenate([f0, zeros]), expected_dim)
        b = _pad_or_truncate(np.concatenate([zeros, f0]), expected_dim)
        return [a, b]

    # Two+ hands: choose the two extreme wrists in x, then try both assignments.
    wrists_x = []
    for h in hands:
        try:
            wrists_x.append(float(h.landmark[0].x))
        except Exception:
            wrists_x.append(0.0)

    idx_left = int(np.argmin(wrists_x))
    idx_right = int(np.argmax(wrists_x))
    if idx_left == idx_right:
        idx_left = 0
        idx_right = 1 if len(hands) > 1 else 0

    f_left = _normalize_landmarks(hands[idx_left].landmark)
    f_right = _normalize_landmarks(hands[idx_right].landmark)

    a = _pad_or_truncate(np.concatenate([f_left, f_right]), expected_dim)
    b = _pad_or_truncate(np.concatenate([f_right, f_left]), expected_dim)
    return [a, b]


def _classify_emotion(emo_model, frame, device, face_box, clahe):
    """Run ResNet emotion classifier on a provided face box. Returns probs or None."""
    if emo_model is None or face_box is None:
        return None
    try:
        from inference import preprocess_roi
        face_roi = _crop_face_roi(frame, face_box, FACE_ROI_PADDING_RATIO)
        if face_roi is None or face_roi.size == 0:
            return None
        tensor = preprocess_roi(
            roi_bgr=face_roi,
            img_size=44,
            in_channels=3,
            imagenet_norm=True,
            clahe=clahe,
        ).to(device)
        with torch.inference_mode():
            out = emo_model(tensor)
            if isinstance(out, dict):
                out = out.get('logits', list(out.values())[0])
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        return probs
    except Exception:
        return None


# ── Main function ─────────────────────────────────────────────────────────────

def capture_words_and_emotion() -> tuple:
    """
    Single webcam session — captures BOTH sign words and facial emotion.

    Returns:
        (words: list, emotion: str)
        e.g. (['I', 'NEED', 'HELP'], 'fear')
    """
    print('\n' + '-' * 64)
    print('  [SentiSign]  Webcam opening — sign words + emotion together.')
    print('  Hold each ASL hand shape steady to register a word.')
    print('  Your facial expression is read automatically.')
    print(f'  Session ends after {FLUSH_SECONDS}s pause or press ENTER.\n')

    # Load models (landmark MLP is optional; emotion model can run without it)
    mlp_model = None
    idx_to_label = None
    try:
        mlp_model, idx_to_label = _load_mlp()
    except Exception as e:
        print(f'  [sign_recognizer] Landmark model disabled: {e}')
        print('  Continuing with emotion-only webcam session; words will be manual.')

    emo_model, face_cascade = _load_emotion_model()
    device = _get_device()

    hands = None
    _hands_module = None
    _drawing = None
    if mlp_model is not None:
        try:
            expected_dim = int(mlp_model.net[0].in_features)
        except Exception:
            expected_dim = 63
        try:
            import mediapipe as mp
            _hands_module = mp.solutions.hands
            _drawing      = mp.solutions.drawing_utils
            use_two_hands = expected_dim > 63
            hands = _hands_module.Hands(
                static_image_mode=True if use_two_hands else False,
                max_num_hands=2 if use_two_hands else 1,
                model_complexity=HANDS_MODEL_COMPLEXITY,
                min_detection_confidence=HANDS_DETECT_CONF_MULTI if use_two_hands else HANDS_DETECT_CONF_SINGLE,
                min_tracking_confidence=HANDS_TRACK_CONF_MULTI if use_two_hands else HANDS_TRACK_CONF_SINGLE,
            )
            print(
                f'[sign_recognizer] Hands: expected_dim={expected_dim} '
                f'max_num_hands={2 if use_two_hands else 1} '
                f'static_image_mode={"on" if use_two_hands else "off"}'
            )
        except ImportError:
            print('  Warning: mediapipe not installed; sign recognition disabled.')
            hands = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('  Error: webcam unavailable.')
        return _manual_word_fallback(), DEFAULT_EMOTION

    word_buffer    = []
    emotion_votes  = Counter()
    emotion_scores = Counter()
    recent_items   = []
    recent_scores  = Counter()
    smoothed_probs = None
    now_emo_label  = None
    now_emo_conf   = 0.0
    tracked_face_box = None
    face_hold = 0
    hold_counter_high = 0
    hold_counter_low = 0
    current_cls    = ''
    last_word_time = time.time()
    frame_count    = 0
    clahe = _load_clahe()

    enter_pressed = threading.Event()
    def _wait_enter():
        try:
            input()
        except Exception:
            pass
        enter_pressed.set()
    threading.Thread(target=_wait_enter, daemon=True).start()

    try:
        while not enter_pressed.is_set():

            if word_buffer and time.time() - last_word_time > FLUSH_SECONDS:
                print(f'\n  [Auto-flush after {FLUSH_SECONDS}s pause]')
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            cls_name, confidence, margin = '', 0.0, 0.0

            # ── SIGN RECOGNITION (every frame via landmarks, if enabled) ───
            if hands is not None and _hands_module is not None and _drawing is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = hands.process(frame_rgb)
                hands_n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        _drawing.draw_landmarks(
                            frame, hand_lm, _hands_module.HAND_CONNECTIONS)

                    feats = _extract_landmark_features(results, expected_dim)
                    candidates = _extract_landmark_feature_candidates(results, expected_dim)
                    if candidates:
                        best = None
                        for cand in candidates:
                            c, conf, m = _classify_landmarks(
                                mlp_model, idx_to_label, cand, device)
                            if best is None or conf > best[1]:
                                best = (c, conf, m)
                        if best is not None:
                            cls_name, confidence, margin = best

                    # Hold logic
                    if cls_name != current_cls:
                        current_cls = cls_name
                        hold_counter_high = 0
                        hold_counter_low = 0
                    else:
                        if confidence >= SIGN_HIGH_CONFIDENCE:
                            hold_counter_high += 1
                        else:
                            hold_counter_high = 0

                        if confidence >= SIGN_LOW_CONFIDENCE and margin >= SIGN_LOW_CONF_MARGIN:
                            hold_counter_low += 1
                        else:
                            hold_counter_low = 0

                    # Register word when held long enough
                    should_commit = (hold_counter_high >= HOLD_FRAMES) or (hold_counter_low >= SIGN_LOW_HOLD_FRAMES)
                    if (should_commit
                            and cls_name in CLASS_TO_WORD
                            and CLASS_TO_WORD[cls_name] is not None):
                        word = CLASS_TO_WORD[cls_name]
                        if not word_buffer or word_buffer[-1] != word:
                            word_buffer.append(word)
                            last_word_time = time.time()
                            hold_counter_high = 0
                            hold_counter_low = 0
                            mode = 'HIGH' if confidence >= SIGN_HIGH_CONFIDENCE else 'LOW'
                            print(f'  + ({mode}) [{cls_name}] -> {word:<14}  Words: {word_buffer}')

            # ── FACE TRACKING (fast) ──────────────────────────────────────
            # Update the face box frequently so it follows head movement quickly.
            if (
                emo_model is not None
                and face_cascade is not None
                and frame_count % FACE_DETECT_EVERY_N == 0
            ):
                detected_box = None
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    boxes = _filter_face_boxes(
                        frame.shape,
                        [tuple(map(int, f)) for f in faces],
                    )
                    detected_box = _select_face_box(boxes, tracked_face_box)
                except Exception:
                    detected_box = None

                if detected_box is not None:
                    if tracked_face_box is None:
                        tracked_face_box = detected_box
                    else:
                        # Catch up faster on larger movements (low IoU).
                        iou = _box_iou(tracked_face_box, detected_box)
                        alpha = FACE_BOX_EMA_ALPHA if iou >= 0.25 else min(0.75, FACE_BOX_EMA_ALPHA * 2.0)
                        tracked_face_box = _smooth_box(tracked_face_box, detected_box, alpha)
                    face_hold = FACE_BOX_HOLD_FRAMES
                elif face_hold > 0:
                    face_hold -= 1
                else:
                    tracked_face_box = None

            # ── EMOTION RECOGNITION (slower cadence) ──────────────────────
            if frame_count % EMOTION_EVERY_N == 0:
                probs = _classify_emotion(emo_model, frame, device, tracked_face_box, clahe)
                if probs is not None:
                    if smoothed_probs is None:
                        smoothed_probs = probs
                    else:
                        smoothed_probs = (EMO_PROBS_EMA_ALPHA * probs) + ((1.0 - EMO_PROBS_EMA_ALPHA) * smoothed_probs)

                    idx = int(np.argmax(smoothed_probs))
                    now_emo_label = EMOTION_LABELS[idx]
                    now_emo_conf = float(smoothed_probs[idx])

                    if tracked_face_box is not None and now_emo_conf >= EMO_CONFIDENCE_THRESHOLD:
                        emotion_votes[now_emo_label] += 1
                        emotion_scores[now_emo_label] += now_emo_conf

                        recent_items.append((now_emo_label, now_emo_conf))
                        recent_scores[now_emo_label] += now_emo_conf
                        if len(recent_items) > EMO_RECENT_WINDOW_SIZE:
                            old_em, old_conf = recent_items.pop(0)
                            recent_scores[old_em] -= old_conf
                            if recent_scores[old_em] <= 1e-6:
                                del recent_scores[old_em]

            # ── OVERLAY ───────────────────────────────────────────────────
            # Draw face box every frame to avoid flicker (box updates run every Nth frame).
            if tracked_face_box is not None:
                x, y, w, h = tracked_face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
                if now_emo_label is not None:
                    cv2.putText(
                        frame,
                        f'{now_emo_label} ({now_emo_conf:.2f})',
                        (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 100, 0),
                        2,
                    )

            word_disp = CLASS_TO_WORD.get(cls_name) or ''
            buf_disp  = ' | '.join(word_buffer) if word_buffer else '...'
            if recent_scores:
                top_emo = recent_scores.most_common(1)[0][0]
            elif emotion_votes:
                top_emo = emotion_votes.most_common(1)[0][0]
            else:
                top_emo = '?'

            if hands is not None:
                cv2.putText(frame,
                    f'Sign: {cls_name} ({confidence:.0%}) -> {word_disp}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(frame,
                    f'Hands: {hands_n}',
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

                # Hold progress bar
                high_p = min(hold_counter_high / max(1, HOLD_FRAMES), 1.0)
                low_p = min(hold_counter_low / max(1, SIGN_LOW_HOLD_FRAMES), 1.0)
                bar_w = int(max(high_p, low_p) * 180)
                cv2.rectangle(frame, (10, 38), (190, 52), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 38), (10 + bar_w, 52), (0, 255, 0), -1)
            else:
                cv2.putText(frame,
                    'Sign: (disabled)  Words: manual',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.putText(frame, f'Words: {buf_disp}',
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)
            now_txt = f'{now_emo_label} ({now_emo_conf:.2f})' if now_emo_label else '...'
            cv2.putText(
                frame,
                f'Emotion: Now {now_txt}  |  Recent {top_emo}  ({sum(emotion_votes.values())} reads)',
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (100, 200, 255),
                1,
            )
            cv2.putText(frame,
                'GREEN=hand landmarks  BLUE=face  ENTER/q to finish',
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

            cv2.imshow('SentiSign  |  Sign + Emotion', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10, ord('q')):
                enter_pressed.set()
                break
    finally:
        if hands is not None:
            hands.close()

    cap.release()
    cv2.destroyAllWindows()

    # Final results
    if not word_buffer:
        print('  No words captured -> manual input.')
        word_buffer = _manual_word_fallback()

    if emotion_votes:
        score_source = recent_scores if recent_scores else emotion_scores
        final_emotion = score_source.most_common(1)[0][0]
        total_readings = sum(emotion_votes.values())
        print(f'\n  Emotion votes : {dict(emotion_votes.most_common())}')
        if recent_scores:
            print(f'  Emotion recent: {dict(recent_scores.most_common())}')
        print(f'  Final emotion : {final_emotion}  ({total_readings} readings)')
    else:
        final_emotion = DEFAULT_EMOTION
        print(f'  No face detected. Defaulting to: {final_emotion}')

    print(f'  Final words   : {word_buffer}')
    return word_buffer, final_emotion


# ── Legacy alias ──────────────────────────────────────────────────────────────

def capture_words() -> list:
    words, _ = capture_words_and_emotion()
    return words


# ── Manual fallbacks ──────────────────────────────────────────────────────────

def _manual_word_fallback() -> list:
    VALID = sorted(set(CLASS_TO_WORD.values()) - {None})
    print(f'\n  Manual input. Available: {", ".join(VALID)}')
    while True:
        raw = input('  Words > ').strip()
        if not raw:
            continue
        sep   = ',' if ',' in raw else ' '
        words = [w.strip().upper() for w in raw.split(sep) if w.strip()]
        if words:
            return words


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '=' * 64)
    print('  SentiSign — Sign + Emotion Recognizer (Landmark MLP)')
    print('  GREEN = hand landmarks   BLUE = face emotion')
    print('=' * 64 + '\n')
    words, emotion = capture_words_and_emotion()
    print(f'\n  Words  : {words}')
    print(f'  Emotion: {emotion}\n')
