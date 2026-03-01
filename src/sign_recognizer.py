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
from PIL import Image
import torchvision.transforms as T

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
MIN_CONFIDENCE  = 0.60    # softmax probability threshold
HOLD_FRAMES     = 10      # consecutive frames same sign must be held
FLUSH_SECONDS   = 30.0    # seconds of no new word → auto end
EMOTION_EVERY_N = 5       # run emotion on every Nth frame to save CPU
FACE_PADDING    = 20
DEFAULT_EMOTION = 'neutral'

# ── Model paths ───────────────────────────────────────────────────────────────
_MLP_PATH      = os.path.join(_ROOT, 'models', 'landmark', 'landmark_mlp.pth')
_LABEL_PATH    = os.path.join(_ROOT, 'models', 'landmark', 'label_map.json')
_EMO_PATH      = os.path.join(_ROOT, 'models', 'emotion',  'resnet_emotion.pth')
_HAAR_PATH     = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── Cached globals ────────────────────────────────────────────────────────────
_mlp_model    = None
_idx_to_label = None
_emo_model    = None
_face_cascade = None
_device       = None

_emo_preprocess = T.Compose([
    T.Resize((44, 44)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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
    global _mlp_model, _idx_to_label
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
    hidden = ckpt.get('hidden',  [256, 128, 64])
    drop   = ckpt.get('dropout', 0.3)

    model  = LandmarkMLP(input_dim=63, hidden=hidden,
                          num_classes=num_classes, dropout=drop)
    model.load_state_dict(ckpt['model_state'])
    model.eval().to(device)

    _mlp_model = model
    print(f'[sign_recognizer] Landmark MLP loaded. Classes: {num_classes}')
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

def _classify_landmarks(model, idx_to_label, landmarks, device):
    """Run MLP on normalised landmarks. Returns (class_name, confidence)."""
    try:
        features = _normalize_landmarks(landmarks)
        tensor   = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        return idx_to_label.get(idx, ''), confidence
    except Exception:
        return '', 0.0


def _classify_emotion(emo_model, face_cascade, frame, device):
    """Detect face and run ResNet emotion classifier."""
    if emo_model is None or face_cascade is None:
        return None, None
    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        x, y, w, h = faces[0]
        x1 = max(0, x - FACE_PADDING)
        y1 = max(0, y - FACE_PADDING)
        x2 = min(frame.shape[1], x + w + FACE_PADDING)
        y2 = min(frame.shape[0], y + h + FACE_PADDING)
        face_roi = frame[y1:y2, x1:x2]
        tensor   = _emo_preprocess(
            Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            out = emo_model(tensor)
            if isinstance(out, dict):
                out = out.get('logits', list(out.values())[0])
            pred = torch.argmax(out, dim=1).item()
        return EMOTION_LABELS[pred], (x1, y1, x2, y2)
    except Exception:
        return None, None


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
    except FileNotFoundError as e:
        print(f'  {e}')
        print('  Continuing with emotion-only webcam session; words will be manual.')

    emo_model, face_cascade = _load_emotion_model()
    device = _get_device()

    hands = None
    _hands_module = None
    _drawing = None
    if mlp_model is not None:
        try:
            import mediapipe as mp
            _hands_module = mp.solutions.hands
            _drawing      = mp.solutions.drawing_utils
            hands = _hands_module.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.6,
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
    hold_counter   = 0
    current_cls    = ''
    last_word_time = time.time()
    frame_count    = 0

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

            cls_name, confidence = '', 0.0

            # ── SIGN RECOGNITION (every frame via landmarks, if enabled) ───
            if hands is not None and _hands_module is not None and _drawing is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_lm = results.multi_hand_landmarks[0]
                    _drawing.draw_landmarks(
                        frame, hand_lm, _hands_module.HAND_CONNECTIONS)

                    cls_name, confidence = _classify_landmarks(
                        mlp_model, idx_to_label, hand_lm.landmark, device)

                    # Hold logic
                    if cls_name == current_cls and confidence >= MIN_CONFIDENCE:
                        hold_counter += 1
                    else:
                        current_cls  = cls_name
                        hold_counter = 0

                    # Register word when held long enough
                    if (hold_counter >= HOLD_FRAMES
                            and cls_name in CLASS_TO_WORD
                            and CLASS_TO_WORD[cls_name] is not None):
                        word = CLASS_TO_WORD[cls_name]
                        if not word_buffer or word_buffer[-1] != word:
                            word_buffer.append(word)
                            last_word_time = time.time()
                            hold_counter   = 0
                            print(f'  + [{cls_name}] -> {word:<14}  Words: {word_buffer}')

            # ── EMOTION RECOGNITION (every Nth frame) ─────────────────────
            if frame_count % EMOTION_EVERY_N == 0:
                emo_label, fbox = _classify_emotion(
                    emo_model, face_cascade, frame, device)
                if emo_label:
                    emotion_votes[emo_label] += 1
                    if fbox:
                        cv2.rectangle(frame,
                                      (fbox[0], fbox[1]),
                                      (fbox[2], fbox[3]),
                                      (255, 100, 0), 2)
                        cv2.putText(frame, emo_label,
                                    (fbox[0], fbox[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 100, 0), 2)

            # ── OVERLAY ───────────────────────────────────────────────────
            word_disp = CLASS_TO_WORD.get(cls_name) or ''
            buf_disp  = ' | '.join(word_buffer) if word_buffer else '...'
            top_emo   = emotion_votes.most_common(1)[0][0] \
                        if emotion_votes else '?'

            if hands is not None:
                cv2.putText(frame,
                    f'Sign: {cls_name} ({confidence:.0%}) -> {word_disp}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # Hold progress bar
                bar_w = int((min(hold_counter, HOLD_FRAMES) / HOLD_FRAMES) * 180)
                cv2.rectangle(frame, (10, 38), (190, 52), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 38), (10 + bar_w, 52), (0, 255, 0), -1)
            else:
                cv2.putText(frame,
                    'Sign: (disabled)  Words: manual',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.putText(frame, f'Words: {buf_disp}',
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)
            cv2.putText(frame,
                f'Emotion: {top_emo}  ({sum(emotion_votes.values())} readings)',
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1)
            cv2.putText(frame,
                'GREEN=hand landmarks  BLUE=face  ENTER to finish',
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

            cv2.imshow('SentiSign  |  Sign + Emotion', frame)
            cv2.waitKey(1)
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
        final_emotion  = emotion_votes.most_common(1)[0][0]
        total_readings = sum(emotion_votes.values())
        print(f'\n  Emotion votes : {dict(emotion_votes.most_common())}')
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
