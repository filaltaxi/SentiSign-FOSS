# train_landmark_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# SentiSign — Landmark Classifier Training
#
# Trains both MLP (deployed) and Random Forest (comparison) on collected
# hand landmark data. Generates full evaluation graphs for guide/viva.
#
# Usage:
#   python train_landmark_classifier.py
#
# Outputs (all saved to data/landmarks/):
#   landmark_mlp.pth              - deployed MLP model
#   landmark_rf.pkl               - Random Forest (comparison only)
#   label_map.json                - class index mapping
#   plots/confusion_matrix.png
#   plots/per_class_accuracy.png
#   plots/mlp_training_curves.png
#   plots/rf_oob_error.png
#   plots/feature_importance.png
#   plots/cv_scores.png
#   plots/model_comparison.png
#   classification_report.txt
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(_ROOT, 'data', 'landmarks', 'raw')
OUT_DIR    = os.path.join(_ROOT, 'data', 'landmarks')
PLOTS_DIR  = os.path.join(OUT_DIR, 'plots')
MODEL_DIR  = os.path.join(_ROOT, 'models', 'landmark')

for d in [OUT_DIR, PLOTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

MLP_SAVE    = os.path.join(MODEL_DIR, 'landmark_mlp.pth')
RF_SAVE     = os.path.join(MODEL_DIR, 'landmark_rf.pkl')
LABEL_SAVE  = os.path.join(MODEL_DIR, 'label_map.json')
REPORT_SAVE = os.path.join(OUT_DIR, 'classification_report.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Hyperparameters ───────────────────────────────────────────────────────────
MLP_HIDDEN   = [256, 128, 64]
MLP_DROPOUT  = 0.3
MLP_LR       = 1e-3
MLP_EPOCHS   = 150
MLP_BATCH    = 64
WEIGHT_DECAY = 1e-4

RF_TREES     = 300
RF_MAX_DEPTH = None
CV_FOLDS     = 5

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print('\n=== Loading landmark data ===')
dfs = []
raw_path = Path(RAW_DIR)
for csv_file in sorted(raw_path.glob('*.csv')):
    df = pd.read_csv(csv_file)
    dfs.append(df)
    print(f'  {csv_file.stem:<12}: {len(df)} samples')

if not dfs:
    print(f'  ERROR: No CSV files found in {RAW_DIR}')
    print(f'  Run collect_landmarks.py first.')
    exit(1)

data = pd.concat(dfs, ignore_index=True)
print(f'\nTotal samples: {len(data):,}')
print(f'Classes      : {data["label"].nunique()}')

# ── 2. ENCODE LABELS ──────────────────────────────────────────────────────────

le          = LabelEncoder()
data['idx'] = le.fit_transform(data['label'])
classes     = list(le.classes_)
NUM_CLASSES = len(classes)

label_map = {
    'classes'      : classes,
    'label_to_idx' : {c: int(i) for i, c in enumerate(classes)},
    'idx_to_label' : {str(i): c for i, c in enumerate(classes)},
}
with open(LABEL_SAVE, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f'Label map saved → {LABEL_SAVE}')

# ── 3. SPLIT DATA ─────────────────────────────────────────────────────────────

feature_cols = [c for c in data.columns if c.startswith('lm')]
X = data[feature_cols].values.astype(np.float32)
y = data['idx'].values

# 80% train, 10% val, 10% test — stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f'\nData split:')
print(f'  Train : {len(X_train):,}  ({len(X_train)/len(X)*100:.0f}%)')
print(f'  Val   : {len(X_val):,}   ({len(X_val)/len(X)*100:.0f}%)')
print(f'  Test  : {len(X_test):,}   ({len(X_test)/len(X)*100:.0f}%)')

# ── 4. MLP MODEL ──────────────────────────────────────────────────────────────

class LandmarkMLP(nn.Module):
    """
    3-layer MLP for hand landmark classification.
    Input: 63 normalised landmark coordinates
    Output: num_classes logits
    """
    def __init__(self, input_dim=63, hidden=[256, 128, 64],
                 num_classes=36, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
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


# ── 5. TRAIN MLP ──────────────────────────────────────────────────────────────

print('\n=== Training MLP ===')

# Tensors
Xt  = torch.from_numpy(X_train).to(device)
yt  = torch.from_numpy(y_train).long().to(device)
Xv  = torch.from_numpy(X_val).to(device)
yv  = torch.from_numpy(y_val).long().to(device)

train_ds     = TensorDataset(Xt, yt)
train_loader = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True)

mlp       = LandmarkMLP(input_dim=63, hidden=MLP_HIDDEN,
                         num_classes=NUM_CLASSES, dropout=MLP_DROPOUT).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS)

history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
best_val_acc = 0.0
best_state   = None
patience, no_improve = 20, 0

print(f'{"Epoch":>6}  {"TrLoss":>8}  {"TrAcc":>8}  {"VlLoss":>8}  {"VlAcc":>8}')
print('─' * 52)

for epoch in range(1, MLP_EPOCHS + 1):
    # Train
    mlp.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = mlp(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_correct += (logits.argmax(1) == yb).sum().item()
        tr_total   += len(yb)
        tr_loss    += loss.item() * len(yb)
    scheduler.step()

    # Validate
    mlp.eval()
    with torch.no_grad():
        vl_logits  = mlp(Xv)
        vl_loss    = criterion(vl_logits, yv).item()
        vl_correct = (vl_logits.argmax(1) == yv).sum().item()

    tr_l = tr_loss / tr_total
    tr_a = tr_correct / tr_total
    vl_a = vl_correct / len(y_val)

    history['train_loss'].append(tr_l)
    history['train_acc' ].append(tr_a)
    history['val_loss'  ].append(vl_loss)
    history['val_acc'   ].append(vl_a)

    if vl_a > best_val_acc:
        best_val_acc = vl_a
        best_state   = {k: v.clone() for k, v in mlp.state_dict().items()}
        no_improve   = 0
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f'{epoch:>6}  {tr_l:>8.4f}  {tr_a*100:>7.2f}%  '
              f'{vl_loss:>8.4f}  {vl_a*100:>7.2f}%')

    if no_improve >= patience:
        print(f'  Early stopping at epoch {epoch}')
        break

# Save best MLP
mlp.load_state_dict(best_state)
torch.save({
    'model_state' : best_state,
    'input_dim'   : 63,
    'hidden'      : MLP_HIDDEN,
    'num_classes' : NUM_CLASSES,
    'dropout'     : MLP_DROPOUT,
    'classes'     : classes,
    'label_to_idx': label_map['label_to_idx'],
    'idx_to_label': label_map['idx_to_label'],
}, MLP_SAVE)
print(f'Best MLP saved → {MLP_SAVE}  (val_acc={best_val_acc*100:.2f}%)')

# ── 6. TRAIN RANDOM FOREST ────────────────────────────────────────────────────

print('\n=== Training Random Forest ===')
oob_errors = []
tree_counts = list(range(10, RF_TREES + 1, 10))

for n in tree_counts:
    rf_temp = RandomForestClassifier(
        n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    oob_errors.append(1 - rf_temp.oob_score_)

rf = RandomForestClassifier(
    n_estimators=RF_TREES, oob_score=True,
    random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
with open(RF_SAVE, 'wb') as f:
    pickle.dump(rf, f)
print(f'Random Forest saved → {RF_SAVE}  (OOB acc={rf.oob_score_*100:.2f}%)')

# ── 7. CROSS-VALIDATION (MLP) ─────────────────────────────────────────────────

print('\n=== 5-Fold Cross-Validation (MLP) ===')
cv_accs = []
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
    Xtr, Xvl = X[tr_idx], X[vl_idx]
    ytr, yvl = y[tr_idx], y[vl_idx]

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xvl_t = torch.from_numpy(Xvl).to(device)
    yvl_t = torch.from_numpy(yvl).long().to(device)

    fold_ds     = TensorDataset(Xtr_t, ytr_t)
    fold_loader = DataLoader(fold_ds, batch_size=MLP_BATCH, shuffle=True)

    fold_mlp = LandmarkMLP(63, MLP_HIDDEN, NUM_CLASSES, MLP_DROPOUT).to(device)
    fold_opt = optim.AdamW(fold_mlp.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)

    best_fold_acc = 0.0
    for ep in range(80):  # shorter for CV
        fold_mlp.train()
        for xb, yb in fold_loader:
            fold_opt.zero_grad(set_to_none=True)
            criterion(fold_mlp(xb), yb).backward()
            fold_opt.step()
        fold_mlp.eval()
        with torch.no_grad():
            acc = (fold_mlp(Xvl_t).argmax(1) == yvl_t).float().mean().item()
        best_fold_acc = max(best_fold_acc, acc)

    cv_accs.append(best_fold_acc)
    print(f'  Fold {fold}: {best_fold_acc*100:.2f}%')

cv_mean = np.mean(cv_accs) * 100
cv_std  = np.std(cv_accs) * 100
print(f'  Mean: {cv_mean:.2f}%  ±{cv_std:.2f}%')

# ── 8. TEST SET EVALUATION ────────────────────────────────────────────────────

print('\n=== Test Set Evaluation ===')
mlp.eval()
Xtest_t = torch.from_numpy(X_test).to(device)
with torch.no_grad():
    mlp_test_preds = mlp(Xtest_t).argmax(1).cpu().numpy()

rf_test_preds = rf.predict(X_test)

mlp_acc = accuracy_score(y_test, mlp_test_preds)
rf_acc  = accuracy_score(y_test, rf_test_preds)
mlp_f1  = f1_score(y_test, mlp_test_preds, average='macro', zero_division=0)
rf_f1   = f1_score(y_test, rf_test_preds,  average='macro', zero_division=0)

print(f'  MLP Test Accuracy : {mlp_acc*100:.2f}%  F1: {mlp_f1:.4f}')
print(f'  RF  Test Accuracy : {rf_acc*100:.2f}%   F1: {rf_f1:.4f}')

# Save classification report
report = classification_report(
    y_test, mlp_test_preds,
    target_names=classes, zero_division=0
)
with open(REPORT_SAVE, 'w') as f:
    f.write('SentiSign Landmark MLP — Classification Report\n')
    f.write('=' * 60 + '\n')
    f.write(f'Test Accuracy : {mlp_acc*100:.2f}%\n')
    f.write(f'Macro F1      : {mlp_f1:.4f}\n')
    f.write(f'CV Mean Acc   : {cv_mean:.2f}% +/- {cv_std:.2f}%\n\n')
    f.write(report)
print(f'Report saved → {REPORT_SAVE}')

# ── 9. PLOTS ──────────────────────────────────────────────────────────────────

print('\n=== Generating plots ===')
plt.style.use('seaborn-v0_8-darkgrid')

# ── Plot 1: MLP Training Curves ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MLP Training Curves — Hand Landmark Classifier', fontsize=14, fontweight='bold')
ep = range(1, len(history['train_loss']) + 1)

axes[0].plot(ep, history['train_loss'], label='Train', color='#2196F3', lw=2)
axes[0].plot(ep, history['val_loss'],   label='Val',   color='#F44336', lw=2)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, [a*100 for a in history['train_acc']], label='Train', color='#2196F3', lw=2)
axes[1].plot(ep, [a*100 for a in history['val_acc']],   label='Val',   color='#F44336', lw=2)
axes[1].axhline(best_val_acc*100, color='orange', ls=':', alpha=0.8,
                label=f'Best val: {best_val_acc*100:.2f}%')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'mlp_training_curves.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: mlp_training_curves.png')

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, mlp_test_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.3, ax=ax, annot_kws={'size': 7})
ax.set_title(f'Confusion Matrix — MLP (Test Acc={mlp_acc*100:.1f}%)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: confusion_matrix.png')

# ── Plot 3: Per-class accuracy ────────────────────────────────────────────────
pc = {}
for t, p_pred in zip(y_test, mlp_test_preds):
    c = classes[t]
    pc.setdefault(c, {'correct': 0, 'total': 0})
    pc[c]['total']   += 1
    pc[c]['correct'] += int(t == p_pred)

cls_s  = sorted(pc)
accs   = [pc[c]['correct'] / max(pc[c]['total'], 1) * 100 for c in cls_s]
colors = [
    '#F44336' if c.startswith('NUM_') else
    '#4CAF50' if a >= 90 else
    '#FFC107' if a >= 70 else '#EF5350'
    for c, a in zip(cls_s, accs)
]

fig, ax = plt.subplots(figsize=(18, 6))
bars = ax.bar(cls_s, accs, color=colors, edgecolor='white')
ax.axhline(np.mean(accs), color='blue', ls='--', alpha=0.7,
           label=f'Mean = {np.mean(accs):.1f}%')
ax.set_title('Per-Class Test Accuracy — MLP Landmark Classifier',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 115); ax.legend()
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f'{acc:.0f}', ha='center', va='bottom', fontsize=7)
plt.xticks(rotation=45, ha='right')
legend_patches = [
    mpatches.Patch(color='#4CAF50', label='>= 90% (excellent)'),
    mpatches.Patch(color='#FFC107', label='>= 70% (acceptable)'),
    mpatches.Patch(color='#EF5350', label='< 70% (needs work)'),
    mpatches.Patch(color='#F44336', label='Digit classes'),
]
ax.legend(handles=legend_patches, loc='lower right')
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'per_class_accuracy.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: per_class_accuracy.png')

# ── Plot 4: RF OOB Error curve ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tree_counts, [e*100 for e in oob_errors], color='#9C27B0', lw=2)
ax.set_title('Random Forest — OOB Error vs Number of Trees',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('OOB Error (%)')
ax.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'rf_oob_error.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: rf_oob_error.png')

# ── Plot 5: Feature Importance (RF) ──────────────────────────────────────────
importances = rf.feature_importances_
feature_cols = [f'lm{i}_{ax}' for i in range(21) for ax in ['x', 'y', 'z']]

# Group by landmark (sum x+y+z importance per landmark)
lm_importance = np.array([
    importances[i*3] + importances[i*3+1] + importances[i*3+2]
    for i in range(21)
])
lm_names = [
    'Wrist', 'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
]
sorted_idx = np.argsort(lm_importance)[::-1]

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(21), lm_importance[sorted_idx], color='#2196F3', edgecolor='white')
ax.set_xticks(range(21))
ax.set_xticklabels([lm_names[i] for i in sorted_idx], rotation=45, ha='right')
ax.set_title('Landmark Feature Importance (Random Forest)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Importance (sum of x+y+z)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'feature_importance.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: feature_importance.png')

# ── Plot 6: Cross-validation scores ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fold_labels = [f'Fold {i+1}' for i in range(CV_FOLDS)]
bar_colors  = ['#4CAF50' if a >= 0.90 else '#FFC107' if a >= 0.80
               else '#F44336' for a in cv_accs]
bars = ax.bar(fold_labels, [a*100 for a in cv_accs], color=bar_colors, edgecolor='white')
ax.axhline(cv_mean, color='blue', ls='--', lw=2,
           label=f'Mean = {cv_mean:.2f}% ± {cv_std:.2f}%')
ax.set_title(f'{CV_FOLDS}-Fold Cross-Validation — MLP',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 115); ax.legend()
for bar, acc in zip(bars, cv_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'cv_scores.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: cv_scores.png')

# ── Plot 7: Model comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

models     = ['CNN+Transformer\n(ASL Dataset)', 'Random Forest\n(Landmarks)', 'MLP\n(Landmarks)']
val_accs   = [99.99, rf_acc*100, best_val_acc*100]
test_accs  = [20.0,  rf_acc*100, mlp_acc*100]   # 20% = real-world estimate for CNN

x     = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, val_accs,  width, label='Val/Train Accuracy',
               color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, test_accs, width, label='Real-world Test Accuracy',
               color='#F44336', edgecolor='white')

ax.set_title('Model Comparison — Validation vs Real-world Accuracy',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 115)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.legend(); ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)

ax.annotate('Domain gap:\n99.99% → ~20%',
            xy=(0, 20), xytext=(0.4, 45),
            fontsize=9, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'model_comparison.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f'  Saved: model_comparison.png')

# ── 10. FINAL SUMMARY ─────────────────────────────────────────────────────────

print('\n' + '=' * 62)
print('  TRAINING COMPLETE')
print('=' * 62)
print(f'  MLP Test Accuracy  : {mlp_acc*100:.2f}%')
print(f'  MLP Macro F1       : {mlp_f1:.4f}')
print(f'  MLP Best Val Acc   : {best_val_acc*100:.2f}%')
print(f'  CV Mean Accuracy   : {cv_mean:.2f}% +/- {cv_std:.2f}%')
print(f'  RF  Test Accuracy  : {rf_acc*100:.2f}%  (comparison only)')
print()
print(f'  Deployed model     : {MLP_SAVE}')
print(f'  Label map          : {LABEL_SAVE}')
print(f'  Plots              : {PLOTS_DIR}/')
print(f'  Report             : {REPORT_SAVE}')
print()
print('  Next: update sign_recognizer.py to use landmark MLP')
print('=' * 62)
