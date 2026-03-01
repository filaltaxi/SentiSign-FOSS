# collect_landmarks.py
# ─────────────────────────────────────────────────────────────────────────────
# SentiSign — Landmark Dataset Collection Script
#
# Collects 200 MediaPipe hand landmark samples per sign class using webcam.
# Saves to data/landmarks/raw/<CLASS>.csv
# Resumable — skips classes that already have 200 samples.
#
# Usage:
#   python collect_landmarks.py
#
# Controls during collection:
#   SPACE  → save current frame as a sample (when hand detected)
#   ENTER  → finish current class early (if enough samples)
#   Q      → quit and save progress
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import csv
import time
import cv2
import numpy as np
from pathlib import Path

# ── Vocabulary — 36 classes in collection order ───────────────────────────────
# Letters first (A-Z), then digits (1-9), then NOTHING
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'NUM_1', 'NUM_2', 'NUM_3', 'NUM_4', 'NUM_5',
    'NUM_6', 'NUM_7', 'NUM_8', 'NUM_9',
    'NOTHING'
]

CLASS_TO_WORD = {
    'A': 'I',        'B': 'YOU',       'C': 'WE',        'D': 'NEED',
    'E': 'WANT',     'F': 'HELP',      'G': 'GO',        'H': 'COME',
    'I': 'DOCTOR',   'J': 'HOSPITAL',  'K': 'MEDICINE',  'L': 'SICK',
    'M': 'PAIN',     'N': 'EMERGENCY', 'O': 'MOTHER',    'P': 'FATHER',
    'Q': 'CHILD',    'R': 'FAMILY',    'S': 'FOOD',      'T': 'WATER',
    'U': 'TOILET',   'V': 'SLEEP',     'W': 'HOME',      'X': 'NOW',
    'Y': 'WHERE',    'Z': 'WHAT',
    'NUM_1': 'NOT',  'NUM_2': 'YES',   'NUM_3': 'NO',    'NUM_4': 'PLEASE',
    'NUM_5': 'THANK YOU', 'NUM_6': 'SORRY', 'NUM_7': 'UNDERSTAND',
    'NUM_8': 'TODAY', 'NUM_9': 'TOMORROW', 'NOTHING': '(no sign)'
}

SAMPLES_PER_CLASS = 200
SAVE_DELAY_MS     = 300   # minimum ms between auto-saves to avoid duplicates

_ROOT     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(_ROOT, 'data', 'landmarks', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# 63 feature column names: lm0_x, lm0_y, lm0_z ... lm20_x, lm20_y, lm20_z
FEATURE_COLS = [f'lm{i}_{ax}' for i in range(21) for ax in ['x', 'y', 'z']]
CSV_COLS     = FEATURE_COLS + ['label']


def normalize_landmarks(landmarks):
    """
    Normalize 21 landmarks relative to wrist (landmark 0).
    Makes features invariant to hand position in frame and distance from camera.

    Returns flat numpy array of 63 values.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Translate: wrist becomes origin
    pts -= pts[0]

    # Scale: divide by max absolute value so all values in [-1, 1]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale

    return pts.flatten()  # shape (63,)


def get_existing_count(cls):
    """Count samples already saved for this class."""
    path = os.path.join(RAW_DIR, f'{cls}.csv')
    if not os.path.exists(path):
        return 0
    with open(path, 'r') as f:
        return sum(1 for row in csv.reader(f)) - 1  # subtract header


def append_sample(cls, features):
    """Append one sample to class CSV file."""
    path    = os.path.join(RAW_DIR, f'{cls}.csv')
    is_new  = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_COLS)
        writer.writerow(list(features) + [cls])


def draw_progress_bar(frame, current, total, y=40):
    w      = frame.shape[1]
    bar_w  = int((current / total) * (w - 40))
    cv2.rectangle(frame, (20, y), (w - 20, y + 18), (40, 40, 40), -1)
    color = (0, 255, 0) if current < total else (0, 200, 255)
    cv2.rectangle(frame, (20, y), (20 + bar_w, y + 18), color, -1)
    cv2.putText(frame, f'{current}/{total}',
                (w // 2 - 30, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def collect_class(cap, hands, drawing, hands_module, cls, start_count):
    """Collect samples for one class. Returns number of samples collected."""
    word       = CLASS_TO_WORD.get(cls, cls)
    count      = start_count
    last_save  = 0
    auto_save  = True   # auto-save when hand is detected and stable

    print(f'\n  Class: {cls} → "{word}"')
    print(f'  Already have: {start_count} samples. Need {SAMPLES_PER_CLASS - start_count} more.')
    print(f'  SPACE=save  ENTER=next class  Q=quit')

    stable_frames = 0  # frames with consistent detection before auto-save

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(frame_rgb)

        features     = None
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            drawing.draw_landmarks(frame, hand_lm, hands_module.HAND_CONNECTIONS)
            features      = normalize_landmarks(hand_lm.landmark)
            hand_detected = True
            stable_frames += 1
        else:
            stable_frames = 0

        # Auto-save when hand stable for 5+ frames and delay passed
        now = time.time() * 1000
        if (hand_detected and stable_frames >= 5
                and now - last_save >= SAVE_DELAY_MS
                and count < SAMPLES_PER_CLASS):
            append_sample(cls, features)
            count    += 1
            last_save = now
            stable_frames = 0  # reset after save

        # ── Overlay ───────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
        status = 'HAND DETECTED - saving' if hand_detected else 'No hand detected'
        color  = (0, 255, 0) if hand_detected else (0, 100, 255)
        cv2.putText(frame, f'{cls} = "{word}"  |  {status}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Progress bar
        draw_progress_bar(frame, count, SAMPLES_PER_CLASS, y=35)

        # Instructions
        cv2.putText(frame, 'SPACE=save manually  ENTER=next  Q=quit',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (160, 160, 160), 1)

        # Sample counter large
        cv2.putText(frame, f'{count}',
                    (w - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 255), 3)

        cv2.imshow('SentiSign — Landmark Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and hand_detected:
            # Manual save
            now2 = time.time() * 1000
            if now2 - last_save >= SAVE_DELAY_MS:
                append_sample(cls, features)
                count    += 1
                last_save = now2
                print(f'  Manual save: {count}/{SAMPLES_PER_CLASS}')

        elif key == 13:  # ENTER
            if count >= 20:
                print(f'  Skipping to next class with {count} samples.')
                break
            else:
                print(f'  Need at least 20 samples. Currently have {count}.')

        elif key == ord('q') or key == ord('Q'):
            print(f'  Quit requested. Saved {count} samples for {cls}.')
            return count, True  # True = quit signal

    if count >= SAMPLES_PER_CLASS:
        print(f'  ✓  {cls} complete — {count} samples saved.')
    return count, False


def main():
    print('=' * 64)
    print('  SentiSign — Landmark Dataset Collection')
    print(f'  Target: {SAMPLES_PER_CLASS} samples × {len(CLASSES)} classes = {SAMPLES_PER_CLASS * len(CLASSES):,} total')
    print(f'  Saving to: {RAW_DIR}')
    print('=' * 64)

    # Check existing progress
    print('\n  Current progress:')
    total_done = 0
    classes_to_do = []
    for cls in CLASSES:
        n = get_existing_count(cls)
        total_done += min(n, SAMPLES_PER_CLASS)
        status = '✓ DONE' if n >= SAMPLES_PER_CLASS else f'{n}/{SAMPLES_PER_CLASS}'
        word   = CLASS_TO_WORD.get(cls, cls)
        print(f'    {cls:<10} ({word:<12}): {status}')
        if n < SAMPLES_PER_CLASS:
            classes_to_do.append(cls)

    print(f'\n  Total collected: {total_done}/{SAMPLES_PER_CLASS * len(CLASSES)}')
    print(f'  Classes remaining: {len(classes_to_do)}')

    if not classes_to_do:
        print('\n  All classes complete! Run train_landmark_classifier.py next.')
        return

    input('\n  Press ENTER to start collection > ')

    try:
        import mediapipe as mp
        hands_module = mp.solutions.hands
        drawing      = mp.solutions.drawing_utils
    except ImportError:
        print('  Error: mediapipe not installed. Run: pip install mediapipe==0.10.9')
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('  Error: webcam unavailable.')
        return

    with hands_module.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.6
    ) as hands:

        for cls in classes_to_do:
            existing = get_existing_count(cls)
            if existing >= SAMPLES_PER_CLASS:
                continue

            word = CLASS_TO_WORD.get(cls, cls)
            print(f'\n  ─── Next: {cls} = "{word}" ───')
            print(f'  Get ready to sign "{word}" repeatedly.')
            print(f'  Position your hand clearly in frame.')

            # Show ready screen
            ready = False
            while not ready:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)
                cv2.putText(frame, f'Next sign: {cls} = "{word}"',
                            (w//2 - 180, h//2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, 'Press SPACE when ready',
                            (w//2 - 140, h//2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, 'Press Q to quit',
                            (w//2 - 90, h//2 + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                cv2.imshow('SentiSign — Landmark Collection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    ready = True
                elif key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print('\n  Collection stopped. Run again to resume.')
                    return

            count, quit_signal = collect_class(
                cap, hands, drawing, hands_module, cls, existing)

            if quit_signal:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    print('\n' + '=' * 64)
    print('  COLLECTION SUMMARY')
    print('=' * 64)
    total = 0
    for cls in CLASSES:
        n    = get_existing_count(cls)
        total += min(n, SAMPLES_PER_CLASS)
        done  = '✓' if n >= SAMPLES_PER_CLASS else f'{n}'
        print(f'  {cls:<10}: {done}')
    print(f'\n  Total: {total}/{SAMPLES_PER_CLASS * len(CLASSES)}')
    remaining = len([c for c in CLASSES if get_existing_count(c) < SAMPLES_PER_CLASS])
    if remaining == 0:
        print('\n  All done! Run: python train_landmark_classifier.py')
    else:
        print(f'\n  {remaining} classes remaining. Run this script again to continue.')
    print('=' * 64)


if __name__ == '__main__':
    main()
