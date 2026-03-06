"""
SentiSign — Temporal ASL Webcam Verification (Fixed)
=====================================================
Key fix: sign-segmented inference instead of blind rolling window.

The model was trained on clean 60-frame windows of active signing.
This script now:
  1. Waits for hands to appear  → starts collecting
  2. Collects frames while hands are visible
  3. When hands disappear (sign ended) OR buffer hits 60 frames → predicts
  4. Resets buffer ready for next sign

Controls:
  Q  — quit
  R  — manually reset buffer (if stuck)
  +  — raise confidence threshold
  -  — lower confidence threshold
"""

import cv2, json, argparse, collections, time
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

parser = argparse.ArgumentParser()
parser.add_argument('--model',      default='models/temporal_lstm.pth')
parser.add_argument('--confidence', type=float, default=0.45,
                    help='Min confidence to show prediction (lowered from 0.6)')
parser.add_argument('--no_hand_frames', type=int, default=10,
                    help='Frames of no-hands before sign is considered ended')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load model ────────────────────────────────────────────────────────────────
print(f'Loading model from {args.model} ...')
ckpt        = torch.load(args.model, map_location=device, weights_only=False)
N_FRAMES    = ckpt['n_frames']
N_FEATURES  = ckpt['input_dim']
num_classes = ckpt['num_classes']
classes     = ckpt['classes']

# ── Model architecture (must match train_temporal.py exactly) ─────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad        = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)[..., :x.size(-1)]))
        out = self.drop(out)
        out = self.relu(self.bn2(self.conv2(out)[..., :x.size(-1)]))
        out = self.drop(out)
        if self.down: res = self.down(res)
        return self.relu(out + res)

class TCNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim=126, tcn_channels=128, tcn_layers=3,
                 lstm_hidden=256, num_classes=100, dropout=0.35):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        blocks, in_ch = [], input_dim
        for i in range(tcn_layers):
            blocks.append(TemporalBlock(in_ch, tcn_channels, 3, 2**i, dropout))
            in_ch = tcn_channels
        self.tcn    = nn.Sequential(*blocks)
        self.bilstm = nn.LSTM(tcn_channels, lstm_hidden, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.attn   = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_w  = torch.softmax(self.attn(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)
        return self.classifier(context)

model = TCNBiLSTMAttention(
    input_dim=N_FEATURES,
    tcn_channels=ckpt.get('tcn_channels', 128),
    tcn_layers=ckpt.get('tcn_layers', 3),
    lstm_hidden=ckpt.get('lstm_hidden', 256),
    num_classes=num_classes,
    dropout=ckpt.get('dropout', 0.35)
).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Model loaded — {num_classes} classes, {N_FRAMES}-frame window')
print(f'Classes: {classes[:10]}{"..." if len(classes) > 10 else ""}')

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_hand(landmarks):
    """Identical to collect_asl.py — wrist-centred, max-abs scaled."""
    pts   = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts))
    return (pts / scale if scale > 0 else pts).flatten()

def extract_features(results):
    right = np.zeros(63, dtype=np.float32)
    left  = np.zeros(63, dtype=np.float32)
    if results.multi_hand_landmarks:
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            if i >= len(results.multi_handedness): continue
            label = results.multi_handedness[i].classification[0].label
            norm  = normalize_hand(hand_lm.landmark)
            if label == 'Right': right = norm
            else:                left  = norm
    return np.concatenate([right, left])

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(buffer):
    """
    Pad or trim buffer to exactly N_FRAMES, then run inference.
    Padding mirrors training behaviour (zero-pad short sequences).
    """
    seq = np.array(buffer, dtype=np.float32)          # [len, 126]

    if len(seq) < N_FRAMES:
        # Zero-pad at the end (same as training)
        pad = np.zeros((N_FRAMES - len(seq), N_FEATURES), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        # Take the most recent N_FRAMES
        seq = seq[-N_FRAMES:]

    x = torch.from_numpy(seq).unsqueeze(0).to(device)  # [1, 60, 126]
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top5_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    return classes[top_idx], top_conf, [(classes[i], float(probs[i])) for i in top5_idx]

# ── Colors ────────────────────────────────────────────────────────────────────
C_TEAL  = (170, 212,   0)
C_AMBER = (  0, 165, 245)
C_RED   = ( 77,  77, 255)
C_GREEN = (  0, 200, 100)
C_DARK  = ( 18,  18,  26)
C_WHITE = (240, 240, 240)
C_MUTED = ( 90,  90, 120)

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_overlay(frame, state, prediction, confidence, top5,
                 buf_size, hands_n, history, conf_threshold):
    h, w = frame.shape[:2]

    def bg(x, y, bw, bh, alpha=0.82):
        ov = frame.copy()
        cv2.rectangle(ov, (x,y), (x+bw, y+bh), C_DARK, -1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

    def txt(text, x, y, color=C_WHITE, scale=0.6, thick=1):
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    # ── Top: current prediction ──
    bg(0, 0, w, 82)
    if prediction and confidence >= conf_threshold:
        bar_w = int((w - 32) * confidence)
        cv2.rectangle(frame, (16, 62), (16+bar_w, 72), C_TEAL, -1)
        cv2.rectangle(frame, (16, 62), (w-16,     72), (60,60,80), 1)
        txt(prediction.upper(), 16, 46, C_TEAL, scale=1.1, thick=2)
        txt(f'{confidence*100:.0f}%', w-80, 46, C_AMBER, scale=0.8)
    else:
        txt('Perform a sign...', 16, 46, C_MUTED, scale=0.8, thick=1)

    # ── State badge ──
    state_colors = {
        'IDLE':       C_MUTED,
        'COLLECTING': C_AMBER,
        'PREDICTED':  C_GREEN,
    }
    s_col = state_colors.get(state, C_MUTED)
    txt(state, w-130, 20, s_col, scale=0.5, thick=1)

    # ── Buffer fill bar ──
    bg(w-170, 28, 170, 48)
    buf_pct = min(buf_size / N_FRAMES, 1.0)
    cv2.rectangle(frame, (w-160, 42), (w-14, 54),          (40,40,55), -1)
    cv2.rectangle(frame, (w-160, 42), (w-160+int(146*buf_pct), 54), C_AMBER, -1)
    txt(f'buf {buf_size}/{N_FRAMES}', w-160, 38, C_MUTED, scale=0.42)
    h_col = C_TEAL if hands_n > 0 else C_RED
    txt(f'hands: {hands_n}', w-160, 68, h_col, scale=0.42)

    # ── Top-5 panel ──
    if top5:
        bg(0, h-168, 230, 168)
        txt('TOP 5', 12, h-152, C_MUTED, scale=0.45)
        for i, (cls, prob) in enumerate(top5):
            y   = h - 130 + i * 26
            bar = int(200 * prob)
            col = C_TEAL if i == 0 else (40, 70, 60)
            cv2.rectangle(frame, (12, y+2), (12+bar, y+16), col, -1)
            txt(f'{cls}', 16, y+14, C_WHITE if i==0 else C_MUTED, scale=0.48)
            txt(f'{prob*100:.0f}%', 180, y+14, C_MUTED, scale=0.42)

    # ── History strip ──
    if history:
        bg(0, h-10-18*min(len(history),4), 200, 18*min(len(history),4)+10)
        for i, (hc, hconf) in enumerate(list(history)[-4:]):
            txt(f'{hc}  {hconf*100:.0f}%', 8, h - 8 - i*18, C_MUTED, scale=0.40)

    # ── Controls ──
    txt('Q=quit  R=reset  +/- confidence', 12, h-6, C_MUTED, scale=0.38)
    txt(f'thresh: {conf_threshold*100:.0f}%', w-130, h-6, C_MUTED, scale=0.38)


# ── Main loop ─────────────────────────────────────────────────────────────────
print('\nWebcam opening...')
print('HOW TO USE:')
print('  1. Hold your hand up and perform a sign clearly')
print('  2. Lower your hand / pause — prediction fires when sign ends')
print('  3. Buffer also fires automatically when it hits 60 frames')
print('Press Q to quit, R to reset buffer\n')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── State ─────────────────────────────────────────────────────────────────────
frame_buffer    = []          # active collection buffer (only hand frames)
no_hand_counter = 0           # consecutive frames without hands
state           = 'IDLE'      # IDLE | COLLECTING | PREDICTED
prediction      = ''
confidence      = 0.0
top5_preds      = []
history         = collections.deque(maxlen=8)
conf_threshold  = args.confidence
MIN_FRAMES      = 10          # need at least this many frames to bother predicting

with mp_hands.Hands(
    model_complexity=1, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        frame   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        hands_n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=C_TEAL, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=C_TEAL, thickness=2)
                )

        feat = extract_features(results)

        # ── Sign segmentation state machine ───────────────────────────────
        if hands_n > 0:
            no_hand_counter = 0

            # Start or continue collecting
            state = 'COLLECTING'
            frame_buffer.append(feat)

            # Buffer full — predict immediately (long/held sign)
            if len(frame_buffer) >= N_FRAMES:
                prediction, confidence, top5_preds = predict(frame_buffer)
                if confidence >= conf_threshold:
                    history.appendleft((prediction, confidence))
                    print(f'  [{state}] {prediction}  {confidence*100:.1f}%')
                state        = 'PREDICTED'
                frame_buffer = []   # reset for next sign

        else:
            # No hands this frame
            no_hand_counter += 1

            if state == 'COLLECTING' and no_hand_counter >= args.no_hand_frames:
                # Hands just disappeared — sign ended, predict what we collected
                if len(frame_buffer) >= MIN_FRAMES:
                    prediction, confidence, top5_preds = predict(frame_buffer)
                    if confidence >= conf_threshold:
                        history.appendleft((prediction, confidence))
                        print(f'  [sign end] {prediction}  {confidence*100:.1f}%')
                    state = 'PREDICTED'
                else:
                    state = 'IDLE'
                frame_buffer = []

            elif no_hand_counter > args.no_hand_frames * 3:
                # Been idle for a while — reset state cleanly
                state = 'IDLE'

        # ── Draw & show ───────────────────────────────────────────────────
        draw_overlay(frame, state, prediction, confidence, top5_preds,
                     len(frame_buffer), hands_n, history, conf_threshold)

        cv2.imshow('SentiSign — ASL Verify (Q=quit)', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_buffer    = []
            no_hand_counter = 0
            state           = 'IDLE'
            prediction      = ''
            print('  [reset]')
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(conf_threshold + 0.05, 0.95)
            print(f'  Confidence threshold: {conf_threshold*100:.0f}%')
        elif key == ord('-'):
            conf_threshold = max(conf_threshold - 0.05, 0.10)
            print(f'  Confidence threshold: {conf_threshold*100:.0f}%')

cap.release()
cv2.destroyAllWindows()
print('\nSession ended.')
if history:
    print('Last predictions:')
    for hc, hconf in history:
        print(f'  {hc}  {hconf*100:.1f}%')