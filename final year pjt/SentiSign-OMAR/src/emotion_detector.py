# src/emotion_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Facial Emotion Detection using ResNet CNN (pure PyTorch)
#
# Changes from v1:
#   - Runs until user presses ENTER (not 5 second timeout)
#   - Bounding box drawn around detected face
#   - Emotion counts accumulate in background across all frames
#   - On tie: user is prompted to pick from tied emotions
#   - capture_emotion() is the single function called by run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import torch
import cv2
from collections import Counter

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.40   # 0.0–1.0

VALID_EMOTIONS = [
    "neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"
]

_PROJECT_ROOT       = os.path.dirname(_SRC)
_DEFAULT_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "emotion", "resnet_emotion.pth"
)

_NUM_CLASSES        = 7
_IN_CHANNELS        = 3
_LINEAR_IN_FEATURES = 2048
_IMG_SIZE           = 44

_emotion_model = None
_face_cascade  = None
_device        = None


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[emotion_detector] Device: {_device}")
    return _device


def _load_emotion_model(model_path: str = _DEFAULT_MODEL_PATH):
    global _emotion_model
    if _emotion_model is not None:
        return _emotion_model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[emotion_detector] Model not found:\n  {model_path}"
        )
    try:
        from inference import load_model
    except ImportError:
        raise ImportError(
            "[emotion_detector] inference.py not found in src/."
        )
    print("[emotion_detector] Loading ResNet emotion model ...")
    _emotion_model = load_model(
        model_path         = model_path,
        num_classes        = _NUM_CLASSES,
        in_channels        = _IN_CHANNELS,
        linear_in_features = _LINEAR_IN_FEATURES,
        device             = _get_device(),
    )
    print("[emotion_detector] ResNet emotion model ready.")
    return _emotion_model


def _load_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascade_path)
    if _face_cascade.empty():
        raise RuntimeError("[emotion_detector] Haar cascade failed to load.")
    return _face_cascade


# ── Single frame detection ────────────────────────────────────────────────────

def detect_emotion(frame: np.ndarray) -> tuple:
    """
    Detect emotion from one BGR frame.
    Returns (emotion_str, confidence_float, face_box_or_None).
    face_box is (x, y, w, h) of the largest detected face, or None.
    Never raises.
    """
    if frame is None or frame.size == 0:
        return "neutral", 0.0, None

    try:
        from inference import predict_emotion

        model   = _load_emotion_model()
        cascade = _load_face_cascade()
        device  = _get_device()

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return "neutral", 0.0, None

        # Largest face
        box = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = box
        face_roi = frame[y:y + h, x:x + w]

        if face_roi.size == 0:
            return "neutral", 0.0, None

        emotion, confidence = predict_emotion(
            model         = model,
            face_roi      = face_roi,
            device        = device,
            img_size      = _IMG_SIZE,
            in_channels   = _IN_CHANNELS,
            imagenet_norm = True,
        )

        emotion = emotion.lower().strip()
        if emotion not in VALID_EMOTIONS:
            return "neutral", 0.0, None
        if confidence < CONFIDENCE_THRESHOLD:
            return "neutral", 0.0, tuple(box)

        return emotion, confidence, tuple(box)

    except Exception:
        return "neutral", 0.0, None


# ── Tie resolution ────────────────────────────────────────────────────────────

def _resolve_tie(tied_emotions: list, counts: dict) -> str:
    """
    When two or more emotions have equal top counts, ask the user to pick.
    Shows the tied emotions and their counts clearly.
    """
    print("\n" + "─" * 48)
    print("  Tie detected between emotions:")
    for em in tied_emotions:
        print(f"    {em:<12} — detected {counts[em]} times")
    print("\n  Which emotion should set the voice tone?")
    for i, em in enumerate(tied_emotions, 1):
        print(f"    [{i}] {em}")

    while True:
        raw = input("  Your choice (number) > ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(tied_emotions):
                chosen = tied_emotions[idx]
                print(f"  ✓  You chose: {chosen}")
                return chosen
        print(f"  ⚠  Enter a number between 1 and {len(tied_emotions)}")


# ── Main capture function (called by run_pipeline.py) ─────────────────────────

def capture_emotion() -> str:
    """
    Opens the webcam and runs continuous emotion detection until the user
    presses ENTER. Emotion counts accumulate silently in the background.
    A bounding box is drawn around the face on the live feed.

    On exit:
    - If one emotion has the highest count → returns it automatically.
    - If there is a tie → prompts user to choose from tied options.
    - If no confident detections → falls back to manual text input.

    Returns:
        emotion: str — one of the 7 valid SentiSign labels.
    """
    print("\n" + "─" * 64)
    print("  [Emotion Detection]  Webcam opening ...")
    print("  Express your emotion naturally. Counts accumulate in background.")
    print("  Press ENTER in this terminal when ready to continue.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠  Webcam unavailable → manual input.")
        return _manual_fallback()

    # Emotion counter — accumulates across entire session
    emotion_counts: Counter = Counter()
    frame_idx = 0

    # Run detection in a loop; stop when user presses ENTER
    # We use cv2.waitKey(1) for the window and check a flag for ENTER.
    import threading
    enter_pressed = threading.Event()

    def _wait_for_enter():
        input()   # blocks until ENTER
        enter_pressed.set()

    t = threading.Thread(target=_wait_for_enter, daemon=True)
    t.start()

    while not enter_pressed.is_set():
        ret, frame = cap.read()
        if not ret:
            print("  ⚠  Webcam read failed.")
            break

        # Run detection every 3rd frame to reduce load
        box = None
        if frame_idx % 3 == 0:
            emotion, confidence, box = detect_emotion(frame)
            if confidence >= CONFIDENCE_THRESHOLD and box is not None:
                emotion_counts[emotion] += 1

        # ── Draw bounding box ─────────────────────────────────────────────────
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── Overlay: current top emotion + total detections ───────────────────
        top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "..."
        total       = sum(emotion_counts.values())

        cv2.putText(frame,
            f"Leading: {top_emotion}  |  Total readings: {total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame,
            "Press ENTER in terminal when done",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("SentiSign  |  Emotion Detection", frame)
        cv2.waitKey(1)   # 1ms poll — keeps window responsive

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── No confident detections at all ───────────────────────────────────────
    if not emotion_counts:
        print("  ⚠  No confident detections → manual input.")
        return _manual_fallback()

    # ── Print final counts ────────────────────────────────────────────────────
    print("\n  Emotion detection complete. Results:")
    print("  " + "─" * 36)
    for em, count in emotion_counts.most_common():
        bar = "█" * min(count, 40)
        print(f"  {em:<12} {count:>4}  {bar}")
    print("  " + "─" * 36)

    # ── Determine winner ──────────────────────────────────────────────────────
    top_count   = emotion_counts.most_common(1)[0][1]
    top_emotions = [em for em, cnt in emotion_counts.items() if cnt == top_count]

    if len(top_emotions) == 1:
        winner = top_emotions[0]
        print(f"\n  ✓  Detected emotion: {winner}  ({top_count} readings)")
        return winner
    else:
        # Tie — let user decide
        return _resolve_tie(sorted(top_emotions), dict(emotion_counts))


def _manual_fallback() -> str:
    """Ask user to type emotion when webcam detection fails or produces no data."""
    print(f"  Supported emotions: {', '.join(VALID_EMOTIONS)}")
    while True:
        em = input("  Emotion > ").strip().lower()
        if em in VALID_EMOTIONS:
            print(f"  ✓  Emotion: {em}")
            return em
        print(f"  ⚠  '{em}' not recognised. Options: {', '.join(VALID_EMOTIONS)}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  ResNet Emotion Detector — Standalone Test")
    print("  Express different emotions at the camera.")
    print("  Press ENTER in this terminal when done.")
    print("=" * 64 + "\n")

    result = capture_emotion()
    print(f"\n  Final emotion selected: {result}\n")
