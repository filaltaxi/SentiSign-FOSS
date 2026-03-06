"""
SentiSign — Temporal ASL Training Script
=========================================
Input  : data/temporal/asl_dataset/  (folder created by collect_asl.py)
Output : models/temporal/temporal_lstm.pth + temporal_label_map.json + evaluation report

Architecture: TCN + BiLSTM + Attention
  - Sees ALL 60 frames at once before predicting
  - BiLSTM reads forward AND backward through the full sign
  - Attention layer decides which frames matter most
  - One prediction per complete sign

Run:
    pip install torch scikit-learn numpy
    python train_temporal.py --data data/temporal/asl_dataset/ --out models/temporal/
"""

import os, json, argparse, numpy as np
import torch, torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',      default='data/temporal/asl_dataset/', help='Dataset folder from collect_asl.py')
parser.add_argument('--out',       default='models/temporal/',           help='Output folder for model files')
parser.add_argument('--epochs',    type=int,   default=100)
parser.add_argument('--patience',  type=int,   default=15)
parser.add_argument('--val_split', type=float, default=0.15)
parser.add_argument('--lr',        type=float, default=3e-4)
parser.add_argument('--batch',     type=int,   default=32)
parser.add_argument('--window',    type=int,   default=60,  help='Frames per training window')
parser.add_argument('--stride',    type=int,   default=5,   help='Sliding window stride for augmentation')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')

N_FRAMES   = args.window
N_FEATURES = 126

# ── Load dataset from folder ──────────────────────────────────────────────────
print(f'\nLoading dataset from {args.data} ...')

signs_found = sorted([
    d for d in os.listdir(args.data)
    if os.path.isdir(os.path.join(args.data, d))
])

if not signs_found:
    print(f'ERROR: No sign folders found in {args.data}')
    exit(1)

label_to_idx = {s: i for i, s in enumerate(signs_found)}
idx_to_label = {i: s for i, s in enumerate(signs_found)}
num_classes  = len(signs_found)

print(f'Signs found : {num_classes}')

# ── Build training windows using sliding window augmentation ──────────────────
# Each raw recording is ~90 frames (3 seconds)
# We slide a 60-frame window over it with stride 5
# This turns 15 reps × 1 sign into ~105 training samples
# The model learns to recognise the sign regardless of where in the window it sits

X_all, y_all = [], []
sign_counts  = {}

for sign in signs_found:
    sign_dir = os.path.join(args.data, sign)
    samples  = sorted([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
    windows_for_sign = 0

    for fname in samples:
        seq = np.load(os.path.join(sign_dir, fname))  # [raw_frames, 126]

        if seq.ndim != 2 or seq.shape[1] != N_FEATURES:
            print(f'  WARNING: {sign}/{fname} wrong shape {seq.shape} — skipped')
            continue

        raw_len = seq.shape[0]

        if raw_len < N_FRAMES:
            # Pad with zeros if recording was too short
            pad = np.zeros((N_FRAMES - raw_len, N_FEATURES), dtype=np.float32)
            seq = np.vstack([seq, pad])
            raw_len = N_FRAMES

        # Slide window over raw recording
        for start in range(0, raw_len - N_FRAMES + 1, args.stride):
            window = seq[start:start + N_FRAMES]
            X_all.append(window)
            y_all.append(label_to_idx[sign])
            windows_for_sign += 1

    sign_counts[sign] = windows_for_sign

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)

print(f'\nAfter sliding window augmentation:')
print(f'  Total training windows : {len(X_all)}')
print(f'  Windows per sign (avg) : {len(X_all)//num_classes}')
print(f'  Shape                  : {X_all.shape}')

# Check for very sparse signs
sparse = [(s, sign_counts[s]) for s in signs_found if sign_counts[s] < 30]
if sparse:
    print(f'\n  WARNING — signs with < 30 windows (collect more reps):')
    for s, c in sparse:
        print(f'    {s}: {c} windows')

# ── Dataset ───────────────────────────────────────────────────────────────────
class TemporalDataset(Dataset):
    """
    All augmentation happens at the sequence level — never shuffles frames.
    Frame order is sacred (it carries the motion pattern).
    
    Augmentations:
    1. Speed augmentation  — resample at 0.75x–1.25x (faster/slower signing)
    2. Temporal jitter     — shift whole sequence 0–2 frames
    3. Gaussian noise      — tiny noise on landmark values
    4. Frame drop          — zero 1–3 frames (occlusion)
    5. Left-right swap     — swap right/left hand features (dominant hand variation)
    """
    def __init__(self, X, y, augment=False):
        self.X       = torch.from_numpy(X)
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # [60, 126]
        if self.augment:
            n = x.shape[0]

            # 1. Speed augmentation
            speed  = np.random.uniform(0.75, 1.25)
            centre = n / 2.0
            half   = (n / 2.0) * speed
            s      = max(0,   int(centre - half))
            e      = min(n-1, int(centre + half))
            x      = x[np.clip(np.linspace(s, e, n).astype(int), 0, n-1)]

            # 2. Temporal jitter
            shift = np.random.randint(0, 3)
            if shift > 0: x = torch.roll(x, shift, dims=0)

            # 3. Landmark noise
            x = x + torch.randn_like(x) * 0.008

            # 4. Random frame drop
            if np.random.random() < 0.3:
                ds = np.random.randint(0, n - 4)
                x[ds:ds + np.random.randint(1, 4)] = 0.0

            # 5. Left-right swap (swap first 63 and last 63 features)
            if np.random.random() < 0.5:
                x = torch.cat([x[:, 63:], x[:, :63]], dim=1)

        return x, self.y[idx]


X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=args.val_split, stratify=y_all, random_state=42
)
print(f'\nTrain: {len(X_train)}  Val: {len(X_val)}')

train_loader = DataLoader(TemporalDataset(X_train, y_train, augment=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=(device.type=='cuda'))
val_loader   = DataLoader(TemporalDataset(X_val, y_val, augment=False),
                          batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=(device.type=='cuda'))

# ── Model ─────────────────────────────────────────────────────────────────────
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
    """
    Input  : [batch, 60, 126]   — full sign sequence, all frames at once
    Output : [batch, num_classes] — one prediction per sign

    TCN    → catches local finger motion patterns (dilations 1,2,4)
    BiLSTM → reads all frames forward AND backward
    Attention → weights which frames matter most for this sign
    Dense  → final prediction
    """
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
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_w  = torch.softmax(self.attn(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)
        return self.classifier(context)


model = TCNBiLSTMAttention(
    input_dim=N_FEATURES, tcn_channels=128, tcn_layers=3,
    lstm_hidden=256, num_classes=num_classes, dropout=0.35
).to(device)

print(f'\nModel: TCN + BiLSTM + Attention')
print(f'Parameters  : {sum(p.numel() for p in model.parameters()):,}')
print(f'Input shape : [batch, {N_FRAMES}, {N_FEATURES}]')
print(f'Output shape: [batch, {num_classes}]')

# ── Training ──────────────────────────────────────────────────────────────────
class_counts_arr = np.bincount(y_train, minlength=num_classes).astype(np.float32)
class_weights    = 1.0 / (class_counts_arr + 1e-6)
class_weights    = class_weights / class_weights.sum() * num_classes
class_weights_t  = torch.from_numpy(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
scaler    = GradScaler(enabled=(device.type == 'cuda'))

PATIENCE   = args.patience
best_val   = 0.0
best_state = None
no_improve = 0
history    = []

print(f'\nTraining {args.epochs} epochs | patience={PATIENCE} | batch={args.batch}')
print(f'{"Epoch":>6}  {"Loss":>8}  {"TrAcc":>7}  {"VlAcc":>7}  {"Top5":>6}')
print('─' * 48)

for epoch in range(1, args.epochs + 1):
    # Train
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == 'cuda')):
            out  = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tl += loss.item() * len(yb)
        tc += (out.argmax(1) == yb).sum().item()
        tt += len(yb)
    scheduler.step()

    # Validate
    model.eval()
    vc, vt = 0, 0
    all_logits, all_gt = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(xb)
            vc += (logits.argmax(1) == yb.to(device)).sum().item()
            vt += len(yb)
            all_logits.append(logits.cpu().float().numpy())
            all_gt.extend(yb.numpy())
    val_acc = vc / vt
    k       = min(5, num_classes)
    top5    = top_k_accuracy_score(np.array(all_gt),
                                   np.concatenate(all_logits), k=k)

    history.append({'epoch': epoch, 'tr_loss': tl/tt,
                    'tr_acc': tc/tt, 'val_acc': val_acc, 'top5': top5})

    marker = ''
    if val_acc > best_val:
        best_val   = val_acc
        best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
        no_improve = 0
        marker     = ' ◀ best'
    else:
        no_improve += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f'{epoch:6d}  {tl/tt:8.4f}  {tc/tt*100:6.2f}%  {val_acc*100:6.2f}%  {top5*100:5.1f}%{marker}')

    if no_improve >= PATIENCE:
        print(f'\nEarly stop at epoch {epoch}')
        break

print(f'\nBest val accuracy: {best_val*100:.2f}%')

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(args.out, 'temporal_lstm.pth')
torch.save({
    'model_state' : best_state,
    'input_dim'   : N_FEATURES,
    'n_frames'    : N_FRAMES,
    'num_classes' : num_classes,
    'tcn_channels': 128,
    'tcn_layers'  : 3,
    'lstm_hidden' : 256,
    'dropout'     : 0.35,
    'classes'     : signs_found,
    'label_to_idx': label_to_idx,
    'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
    'model_type'  : 'tcn_bilstm_attention',
}, model_path)
print(f'Model saved → {model_path}')

label_map_path = os.path.join(args.out, 'temporal_label_map.json')
with open(label_map_path, 'w') as f:
    json.dump({
        'classes'     : signs_found,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(v): k for k, v in label_to_idx.items()},
        'input_dim'   : N_FEATURES,
        'n_frames'    : N_FRAMES,
        'model_type'  : 'tcn_bilstm_attention',
    }, f, indent=2)
print(f'Label map  → {label_map_path}')

# ── Evaluation ────────────────────────────────────────────────────────────────
print('\n' + '='*50)
print('EVALUATION')
print('='*50)

model.load_state_dict(best_state)
model.eval()

all_logits, all_preds, all_gt = [], [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        with autocast(enabled=(device.type == 'cuda')):
            logits = model(xb).cpu().float().numpy()
        all_logits.append(logits)
        all_preds.extend(logits.argmax(1).tolist())
        all_gt.extend(yb.numpy())

all_logits = np.concatenate(all_logits)
preds_arr  = np.array(all_preds)
gt_arr     = np.array(all_gt)

top1   = (preds_arr == gt_arr).mean()
top5   = top_k_accuracy_score(gt_arr, all_logits, k=min(5, num_classes))
report = classification_report(gt_arr, preds_arr,
                                target_names=signs_found,
                                digits=3, zero_division=0)

print(f'Top-1 accuracy : {top1*100:.2f}%')
print(f'Top-5 accuracy : {top5*100:.2f}%')
print()
print(report)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(gt_arr, preds_arr)
fig_size = max(12, num_classes // 4)
fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
sns.heatmap(cm, annot=(num_classes <= 30), fmt='d',
            xticklabels=signs_found, yticklabels=signs_found,
            cmap='YlOrRd', ax=ax, linewidths=0.3)
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('Actual',    fontsize=10)
ax.set_title(f'ASL Confusion Matrix — Top-1: {top1*100:.1f}%', fontsize=12)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
cm_path = os.path.join(args.out, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=120)
print(f'Confusion matrix → {cm_path}')

# ── Training curve ────────────────────────────────────────────────────────────
epochs_done = [h['epoch']   for h in history]
tr_accs     = [h['tr_acc']  for h in history]
val_accs    = [h['val_acc'] for h in history]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_done, [a*100 for a in tr_accs],  label='Train', color='#00d4aa')
ax.plot(epochs_done, [a*100 for a in val_accs], label='Val',   color='#f5a623')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Training Curve — TCN+BiLSTM+Attention')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(args.out, 'training_curve.png')
plt.savefig(curve_path, dpi=120)
print(f'Training curve  → {curve_path}')

# ── Report file ───────────────────────────────────────────────────────────────
report_path = os.path.join(args.out, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write('SentiSign Custom ASL Temporal Model — Evaluation Report\n')
    f.write('='*55 + '\n')
    f.write(f'Architecture   : TCN(3 blocks) + BiLSTM(256x2) + Attention\n')
    f.write(f'Input          : {N_FRAMES} frames x {N_FEATURES} features\n')
    f.write(f'Classes        : {num_classes}\n')
    f.write(f'Train windows  : {len(X_train)}\n')
    f.write(f'Val windows    : {len(X_val)}\n')
    f.write(f'Top-1 Accuracy : {top1*100:.2f}%\n')
    f.write(f'Top-5 Accuracy : {top5*100:.2f}%\n')
    f.write(f'Best Val Acc   : {best_val*100:.2f}%\n\n')
    f.write(report)
print(f'Evaluation report → {report_path}')

print('\n' + '='*50)
print('  DONE')
print('='*50)
print(f'  Top-1 : {top1*100:.2f}%')
print(f'  Top-5 : {top5*100:.2f}%')
print(f'\nFiles in {args.out}:')
print(f'  temporal_lstm.pth')
print(f'  temporal_label_map.json')
print(f'  confusion_matrix.png')
print(f'  training_curve.png')
print(f'  evaluation_report.txt')
print('='*50)
print('\nTo verify with webcam:')
print('  python verify_temporal.py --model models/temporal_lstm.pth')
