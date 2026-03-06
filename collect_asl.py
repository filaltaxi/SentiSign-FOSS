"""
SentiSign — ASL Temporal Data Collector
========================================
Keyboard controls:
  SPACE  — start recording current rep
  B      — take a break (pause between signs)
  R      — resume from break
  S      — skip current sign
  Q      — quit and save progress summary

Data saved instantly to disk after each rep.
Resume automatically if you quit and restart.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
N_FRAMES       = 60       # frames per rep (2 seconds at 30fps)
N_REPS         = 15       # reps per sign
RECORD_SECONDS = 2.0      # how long each rep records
FIRST_COUNTDOWN= 3        # seconds before first rep of a sign
REP_GAP        = 1.0      # seconds between reps (same sign)
DATASET_DIR    = 'data/temporal/asl_dataset'
FPS            = 30

# ── Signs grouped by category ─────────────────────────────────────────────────
CATEGORIES = [
    ('Basic_Responses',  ['FINE','FINISH','FORGET','GO','HELLO','HELP',
                          'DONT_WANT','LIKE','MORE','NEED','WRONG','NO',
                          'NOT','PLEASE','RIGHT','THANK_YOU','WANT','YES']),
    ('Places',           ['BATHROOM','CAR','HOME','KITCHEN','LIVING_ROOM','PHONE']),
    ('Food_Drink',       ['CHEESE','COOK','COOKIES','DRINK','EAT',
                          'HAMBURGER','MILK','SANDWICH']),
    ('Pronouns',         ['ME','MINE','YOU','YOUR']),
    ('People',           ['BOY','GIRL','MAN','WOMAN','DEAF','HEARING']),
    ('Questions',        ['HOW','WHAT','WHEN','WHERE','WHO','WHY']),
    ('Education',        ['ASL','BOOK','CLASS','KNOW','LEARN','SENTENCE',
                          'SIGN','STUDENT','TEACHER']),
    ('Time',             ['AFTERNOON','DAY','HOUR','MINUTE','MONTH',
                          'MORNING','NIGHT','NOON','TIME','WEEK','YEAR']),
    ('Family',           ['BABY','BROTHER','CHILDREN','FAMILY','FATHER',
                          'GRANDFATHER','GRANDMOTHER','HUSBAND','MOTHER',
                          'SISTER','WIFE']),
    ('Animals',          ['DOG','CAT']),
    ('Colors',           ['BLUE','GREEN','ORANGE','PINK','PURPLE',
                          'RED','YELLOW']),
    ('Emotions',         ['SO_SO','GOOD','HAPPY','SAD','BAD']),
    ('Actions',          ['CHAT','SIGNING','LOOK','PAY_ATTENTION']),
    ('Negation',         ['DONT','NOT_YET']),
    ('Greetings',        ['BYE','LOVE']),
]

SIGNS = []
SIGN_CATEGORY = {}
for cat, signs in CATEGORIES:
    for s in signs:
        SIGNS.append(s)
        SIGN_CATEGORY[s] = cat

# ── Colors ────────────────────────────────────────────────────────────────────
C_TEAL   = (170, 212,   0)
C_AMBER  = (  0, 165, 245)
C_RED    = ( 77,  77, 255)
C_WHITE  = (240, 240, 240)
C_DARK   = ( 18,  18,  26)
C_MUTED  = ( 90,  90, 120)
C_GREEN  = (  0, 200, 100)

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Normalisation (matches train_temporal.py exactly) ─────────────────────────
def normalize_hand(landmarks):
    pts   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()

def extract_features(results):
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)
    if results.multi_hand_landmarks:
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            if i >= len(results.multi_handedness):
                continue
            label = results.multi_handedness[i].classification[0].label
            norm  = normalize_hand(hand_lm.landmark)
            if label == 'Right': right = norm
            else:                left  = norm
    return np.concatenate([right, left])

# ── Dataset helpers ────────────────────────────────────────────────────────────
def get_reps_done(sign):
    sign_dir = os.path.join(DATASET_DIR, sign)
    if not os.path.exists(sign_dir):
        return 0
    return len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])

def save_rep(sign, frames):
    sign_dir = os.path.join(DATASET_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)
    existing = get_reps_done(sign)
    path     = os.path.join(sign_dir, f'sample_{existing+1:03d}.npy')
    np.save(path, np.array(frames, dtype=np.float32))
    return path

def total_collected():
    total = 0
    for sign in SIGNS:
        total += get_reps_done(sign)
    return total

def first_incomplete_sign():
    for i, sign in enumerate(SIGNS):
        if get_reps_done(sign) < N_REPS:
            return i
    return len(SIGNS)  # all done

# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_bg(frame, x, y, w, h, color, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def put_text(frame, text, x, y, color=C_WHITE, scale=0.65, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def draw_progress_bar(frame, x, y, w, h, pct, color=C_TEAL):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 60), -1)
    filled = int(w * pct)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 100), 1)

def draw_hud(frame, sign, sign_idx, reps_done, state, countdown_val,
             frame_count, hands_visible, on_break):
    h, w = frame.shape[:2]

    # ── Top bar ──
    draw_bg(frame, 0, 0, w, 70, C_DARK, 0.85)

    # Sign name
    put_text(frame, sign.upper(), 16, 28,
             C_TEAL if not on_break else C_AMBER, scale=0.9, thickness=2)

    # Category
    cat = SIGN_CATEGORY.get(sign, '')
    put_text(frame, cat.replace('_', ' '), 16, 52, C_MUTED, scale=0.5)

    # Sign index
    put_text(frame, f'{sign_idx+1} / {len(SIGNS)}', w-120, 28, C_MUTED, scale=0.6)

    # Total collected
    total = total_collected()
    put_text(frame, f'total: {total}', w-120, 52, C_MUTED, scale=0.5)

    # ── Rep counter ──
    draw_bg(frame, 0, h-80, w, 80, C_DARK, 0.85)

    # Rep dots
    dot_x = 16
    for i in range(N_REPS):
        color = C_TEAL if i < reps_done else (C_AMBER if i == reps_done else (50, 50, 65))
        cv2.circle(frame, (dot_x + i*22, h-52), 7, color, -1)

    put_text(frame, f'Rep {reps_done+1} of {N_REPS}', 16, h-18, C_WHITE, scale=0.6)

    # ── State overlay ──
    if on_break:
        draw_bg(frame, w//2-160, h//2-50, 320, 100, C_DARK, 0.9)
        put_text(frame, 'ON BREAK', w//2-80, h//2, C_AMBER, scale=1.0, thickness=2)
        put_text(frame, 'Press R to resume', w//2-90, h//2+30, C_MUTED, scale=0.55)
        return

    if state == 'COUNTDOWN':
        draw_bg(frame, w//2-80, h//2-60, 160, 120, C_DARK, 0.85)
        put_text(frame, str(countdown_val), w//2-25, h//2+20,
                 C_AMBER, scale=2.5, thickness=3)
        put_text(frame, 'get ready', w//2-45, h//2+50, C_MUTED, scale=0.5)

    elif state == 'RECORDING':
        # Red recording indicator
        cv2.circle(frame, (w-30, 90), 10, C_RED, -1)
        put_text(frame, 'REC', w-65, 95, C_RED, scale=0.55, thickness=2)
        # Frame progress bar
        pct = frame_count / N_FRAMES
        draw_progress_bar(frame, 16, h-100, w-32, 10, pct, C_RED)
        put_text(frame, f'{frame_count}/{N_FRAMES} frames', w-130, h-88, C_MUTED, scale=0.45)
        # Hands indicator
        color = C_TEAL if hands_visible else C_RED
        label = f'hands: {hands_visible}' if hands_visible else 'NO HANDS'
        put_text(frame, label, 16, 90, color, scale=0.55)

    elif state == 'WAIT':
        put_text(frame, 'SPACE to record', w//2-90, h//2, C_WHITE, scale=0.7)
        put_text(frame, 'B = break   S = skip   Q = quit',
                 w//2-140, h//2+28, C_MUTED, scale=0.48)

    elif state == 'SAVED':
        draw_bg(frame, w//2-100, h//2-30, 200, 60, C_DARK, 0.85)
        put_text(frame, f'Rep {reps_done} saved!', w//2-80, h//2+10,
                 C_GREEN, scale=0.75, thickness=2)


# ── Main collection loop ───────────────────────────────────────────────────────
def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Find where to resume
    start_idx = first_incomplete_sign()
    if start_idx >= len(SIGNS):
        print("All signs already collected! Run train_temporal.py to train.")
        return

    print(f"\nSentiSign ASL Collector")
    print(f"Dataset dir : {os.path.abspath(DATASET_DIR)}")
    print(f"Resuming    : sign {start_idx+1}/{len(SIGNS)} — {SIGNS[start_idx]}")
    print(f"Collected   : {total_collected()} / {len(SIGNS)*N_REPS} total reps\n")
    print("Controls: SPACE=record  B=break  R=resume  S=skip  Q=quit\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    sign_idx    = start_idx
    state       = 'WAIT'      # WAIT | COUNTDOWN | RECORDING | SAVED | BREAK
    on_break    = False
    frame_buf   = []
    countdown   = 0
    countdown_start = 0
    saved_time  = 0
    is_first_rep = True       # first rep of this sign needs full countdown

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while sign_idx < len(SIGNS):
            sign      = SIGNS[sign_idx]
            reps_done = get_reps_done(sign)

            # Sign already complete — auto advance
            if reps_done >= N_REPS:
                sign_idx += 1
                is_first_rep = True
                state = 'WAIT'
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            hands_visible = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=C_TEAL, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=C_TEAL, thickness=2)
                    )

            # ── State machine ──────────────────────────────────────────────
            now = time.time()

            if state == 'COUNTDOWN':
                remaining = int(countdown_start + countdown - now) + 1
                remaining = max(1, remaining)
                if now >= countdown_start + countdown:
                    # Countdown done — start recording
                    state     = 'RECORDING'
                    frame_buf = []
                else:
                    draw_hud(frame, sign, sign_idx, reps_done, state,
                             remaining, 0, hands_visible, on_break)
                    cv2.imshow('SentiSign — ASL Collector', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    continue

            if state == 'RECORDING':
                feat = extract_features(results)
                frame_buf.append(feat)

                draw_hud(frame, sign, sign_idx, reps_done, state,
                         0, len(frame_buf), hands_visible, on_break)
                cv2.imshow('SentiSign — ASL Collector', frame)
                cv2.waitKey(1)

                if len(frame_buf) >= N_FRAMES:
                    # Save rep immediately to disk
                    path = save_rep(sign, frame_buf)
                    reps_done = get_reps_done(sign)
                    print(f"  [{sign}] rep {reps_done}/{N_REPS} saved → {os.path.basename(path)}")

                    if reps_done >= N_REPS:
                        print(f"  ✓ {sign} complete — moving to next sign\n")
                        sign_idx    += 1
                        is_first_rep = True
                        state        = 'WAIT'
                    else:
                        state        = 'SAVED'
                        saved_time   = now
                        is_first_rep = False
                continue

            if state == 'SAVED':
                # Brief "saved" display then auto-trigger next rep countdown
                draw_hud(frame, sign, sign_idx, reps_done, state,
                         0, N_FRAMES, hands_visible, on_break)
                cv2.imshow('SentiSign — ASL Collector', frame)
                cv2.waitKey(1)
                if now - saved_time >= REP_GAP:
                    # Auto start short 1s countdown for next rep
                    state           = 'COUNTDOWN'
                    countdown       = 1
                    countdown_start = now
                continue

            # ── WAIT / BREAK state ─────────────────────────────────────────
            draw_hud(frame, sign, sign_idx, reps_done, state,
                     0, 0, hands_visible, on_break)
            cv2.imshow('SentiSign — ASL Collector', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' ') and not on_break:
                # Start recording — 3s countdown for first rep, 1s for rest
                cd              = FIRST_COUNTDOWN if is_first_rep else 1
                state           = 'COUNTDOWN'
                countdown       = cd
                countdown_start = time.time()

            elif key == ord('b'):
                on_break = True
                state    = 'WAIT'
                print(f"  [break] — press R to resume")

            elif key == ord('r'):
                on_break = False
                print(f"  [resumed] — continuing with {sign}")

            elif key == ord('s') and not on_break:
                print(f"  [skipped] {sign}")
                sign_idx    += 1
                is_first_rep = True
                state        = 'WAIT'

    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ────────────────────────────────────────────────────
    print("\n" + "="*44)
    print("  SESSION SUMMARY")
    print("="*44)
    total  = total_collected()
    target = len(SIGNS) * N_REPS
    complete_signs = [s for s in SIGNS if get_reps_done(s) >= N_REPS]
    incomplete     = [s for s in SIGNS if 0 < get_reps_done(s) < N_REPS]
    not_started    = [s for s in SIGNS if get_reps_done(s) == 0]

    print(f"  Total reps collected : {total} / {target}")
    print(f"  Signs complete       : {len(complete_signs)} / {len(SIGNS)}")
    if incomplete:
        print(f"  Signs partial        : {[(s, get_reps_done(s)) for s in incomplete]}")
    if not_started:
        print(f"  Signs not started    : {len(not_started)}")
    print(f"  Dataset location     : {os.path.abspath(DATASET_DIR)}")
    print("="*44)
    print("\nWhen all signs are collected, run:")
    print("  python train_temporal.py --data asl_dataset/ --out models/")


if __name__ == '__main__':
    main()
