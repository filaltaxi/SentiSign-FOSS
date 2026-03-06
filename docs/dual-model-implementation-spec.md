# SentiSign — Dual Sign Model Support (MLP + Temporal LSTM)
**Implementation Spec**  
Date: 2026-03-05  
Scope: FastAPI backend (`main.py`) + Vite/React frontend (`frontend/`)  
Goal: Let users choose between the current landmark **MLP** model and the new temporal **LSTM** model on first visit and anytime later; make Communicate / Signs / Contribute adapt to the selected model.

---

## 0) Requirements (Confirmed)
- No auth/accounts.
- Ask user **once** on first arrival which model to use; persist selection locally; allow switching later anytime.
- LSTM collector defaults (enforced):
  - `N_FRAMES = 60` frames per rep
  - `N_REPS = 15` reps per word (**always enforce 15**; no “quick demo” shorter option)
- Planned words list = the categories list from `collect_asl.py` (derived from `collect_asl (1).py`), but also allow **custom words**.
- For custom words (LSTM contribute):
  - always check if word already exists; if it exists, treat contribution as “add more reps”, not a brand-new word.
- Signs Gallery for LSTM: **no GIFs** (text-only list/cards are fine).
- Switching models must **clear the current word buffer** and reset session state.

---

## 1) Model Definitions
### 1.1 MLP (Current)
- Input per inference: `126` floats (2 hands × 21 landmarks × xyz; already used).
- Backend endpoint today: `POST /api/recognise` with `{ landmarks: number[126] }`.

### 1.2 Temporal LSTM (New)
- Files from untracked set:
  - `temporal_lstm.pth` (PyTorch checkpoint)
  - `temporal_label_map.json` (classes + metadata)
  - `collect_asl (1).py` (planned list + collector UX reference)
  - `train_temporal (1).py` (training logic reference; contains plotting deps not in requirements)
  - `verify_temporal.py` (architecture + inference reference)
- Input per inference: sequence of `60` frames, each frame is `126` floats → shape `[60, 126]`.
- Output: class string (e.g. `DONT_WANT`) + confidence.

---

## 2) Data + Naming Conventions
### 2.1 Canonical Word Normalization (LSTM)
Define one normalization rule shared by backend + frontend:
- `trim`
- uppercase
- spaces → `_`
- collapse multiple `_`
- allow only `[A-Z0-9_]` (optional strictness)

Examples:
- `"dont want"` → `DONT_WANT`
- `" Thank you "` → `THANK_YOU`

### 2.2 Dataset Storage (LSTM)
Store contributed reps on the backend filesystem (local dev friendly):
- `data/temporal/asl_dataset/<WORD>/sample_<NNN>.npy`
- each `.npy` is float32 array shaped `[60, 126]`

Also track collection progress by counting `.npy` files per word.

### 2.3 Model Artifacts (LSTM)
Store trained outputs in a consistent place:
- `models/temporal/temporal_lstm.pth`
- `models/temporal/temporal_label_map.json`
- (optional) `models/temporal/evaluation_report.txt` (text only; avoid matplotlib/seaborn)

### 2.4 Planned Words List
Persist planned words as a JSON file to avoid duplicating `collect_asl.py` constants in multiple places:
- `models/temporal/temporal_planned_words.json`

Format:
```json
{
  "categories": [
    { "name": "Basic_Responses", "words": ["FINE", "FINISH"] },
    { "name": "Places", "words": ["BATHROOM", "CAR"] }
  ]
}
```

---

## 3) Backend API Design (FastAPI)
### 3.1 Principles
- Do not break existing MLP endpoints.
- Add LSTM endpoints under `/api/temporal/*`.
- Keep backend “ready” even if LSTM files are missing (LSTM is optional).

### 3.2 New Endpoints (LSTM)
#### (A) Meta + availability
1) `GET /api/models`  
Returns what the frontend can offer in the model picker.
```json
{
  "models": [
    { "id": "mlp", "label": "Landmark MLP", "available": true },
    { "id": "lstm", "label": "Temporal LSTM", "available": true, "n_frames": 60, "trained_count": 25 }
  ],
  "default": "mlp"
}
```

2) `GET /api/temporal/planned`  
Returns planned categories.
```json
{ "categories": [ { "name": "Places", "words": ["BATHROOM", "CAR"] } ] }
```

3) `GET /api/temporal/status`  
Returns trained + collection progress (for contribute page indicators).
```json
{
  "trained": ["HELLO", "HELP", "DONT_WANT"],
  "collection": [
    { "word": "BATHROOM", "reps_collected": 7, "reps_target": 15, "is_planned": true, "is_trained": false, "category": "Places" }
  ]
}
```

4) `GET /api/temporal/signs`  
For Signs Gallery (text-only):
```json
{
  "signs": [
    { "class": "HELLO", "word": "HELLO" },
    { "class": "DONT_WANT", "word": "DONT WANT" }
  ]
}
```

#### (B) Word existence check (custom words)
5) `POST /api/temporal/signs/check`  
Request:
```json
{ "word": "dont want" }
```
Response:
```json
{
  "word": "DONT_WANT",
  "exists_trained": true,
  "exists_in_dataset": true,
  "reps_collected": 15,
  "reps_target": 15,
  "is_planned": true,
  "category": "Basic_Responses"
}
```

Rules:
- `exists_trained` is based on `temporal_label_map.json`.
- `exists_in_dataset` / `reps_collected` is based on filesystem dataset folder.
- If a custom word normalizes to an existing trained word → frontend should offer “add more reps” (not “create new”).

#### (C) Upload reps (incremental)
6) `POST /api/temporal/reps/add`  
Uploads *one* rep.
Request:
```json
{
  "word": "BATHROOM",
  "frames": [[...126 floats...], "... 60 total frames ..."]
}
```
Response:
```json
{ "ok": true, "word": "BATHROOM", "reps_collected": 8, "reps_target": 15 }
```

Validation:
- `frames.length === 60`
- each frame `length === 126`
- floats finite (reject NaN/inf)

#### (D) Train / retrain (background)
7) `POST /api/temporal/train`  
Request:
```json
{ "epochs": 100, "patience": 15 }
```
Response:
```json
{ "started": true }
```

8) `GET /api/temporal/train/status`  
Response:
```json
{ "state": "idle|training|error", "message": "…", "progress": 0.42 }
```

Training requirements:
- Re-implement core training from `train_temporal.py` but **remove plotting** (matplotlib/seaborn are not in current requirements).
- Use the same checkpoint keys as `verify_temporal.py` expects:
  - `model_state`, `input_dim`, `n_frames`, `num_classes`, `classes`, `label_to_idx`, `idx_to_label`, etc.
- After training completes:
  - write checkpoint + label map to `models/temporal/…`
  - hot-reload LSTM model in memory for inference

#### (E) Inference (webcam) — chosen approach
9) `POST /api/temporal/recognise` (**sequence upload**)  
Request:
```json
{ "sequence": [[...126 floats...], "... 60 frames ..."] }
```
Response:
```json
{ "class": "HELLO", "word": "HELLO", "confidence": 0.83, "top5": [["HELLO", 0.83], ["HELP", 0.05]] }
```

Notes:
- This is the recommended v1 approach: stateless backend, simplest integration.
- Frontend should only send when it has exactly 60 frames (sliding window), and only every `stride` frames (e.g. `stride=5`) to reduce load.

### 3.3 `/api/status` (Startup Gate)
- Keep current behavior: backend must become `ready` even if LSTM is missing.
- If you include an LSTM step in the status `steps`, mark it `done` when LSTM files are missing but LSTM is optional; otherwise `BackendGate` will block the whole app.

---

## 4) Frontend Architecture
### 4.1 Model Selection Persistence + UX
- Persist choice in:
  - `localStorage['sentisign:model']` = `'mlp' | 'lstm'`

**First-visit behavior**
- If the key is missing:
  - show a blocking modal on the first route load (typically on `/` Communicate) with:
    - “Landmark MLP (fast, per-frame)” vs “Temporal LSTM (motion-aware, 60-frame)”
    - a short description and “Continue” button
  - store selection, then render app content normally

**Switch anytime**
- Add a model switch control in the top navbar (right side):
  - segmented control or dropdown: `MLP` / `LSTM`
  - changing it updates context + localStorage

**Switch side-effects (required)**
- Switching models must:
  - stop webcam session if active
  - clear the current word buffer
  - reset detection state (label/confidence) and stability counters

### 4.2 Shared State
Create a `ModelProvider` with:
- `model: 'mlp' | 'lstm' | null`
- `setModel(next)`
- `isOnboardingOpen` when `model === null`

Expose `useModel()` hook for pages.

---

## 5) Page-by-Page Behavior
### 5.1 Communicate (`/`)
**MLP mode**
- Use existing `WebcamPane` logic:
  - per-frame feature extraction
  - call `POST /api/recognise`
  - apply hold logic in `Communicate.tsx` (`HOLD_FRAMES`, `MIN_CONFIDENCE`)

**LSTM mode**
- Update webcam pipeline to build a sliding window of 60 frames:
  - Each MediaPipe result → extract `126` floats (same as MLP)
  - Push into `frameBuffer` (max length 60)
  - Once buffer is full, every `stride` frames (e.g. 5):
    - call `POST /api/temporal/recognise` with `{ sequence: frameBuffer }`
- Output mapping:
  - `class` is already the word key (`DONT_WANT`)
  - `word` for UI display: replace `_` with space

Stability logic suggestion (to avoid spam):
- Keep the existing “hold the same class N times” logic, but treat each LSTM prediction as one “tick” in that counter.

### 5.2 Signs Gallery (`/signs`)
**MLP mode**
- Current behavior: `GET /api/signs` and show GIF cards.

**LSTM mode**
- Fetch: `GET /api/temporal/signs`
- Render text-first cards/list:
  - prominent `word` (with spaces)
  - smaller `class` (underscore form)
- Disable/hide GIF-related UI:
  - hide “Only signs with GIFs” toggle
  - do not render image placeholders for LSTM

### 5.3 Contribute (`/contribute`)
Split into two flows selected by model:
- `ContributeMLP`: keep current wizard (Gate 1–4).
- `ContributeLSTM`: new temporal contribution flow.

#### 5.3.1 ContributeLSTM UX
**Top section**
- “Planned words” panel with:
  - category tabs or collapsible sections
  - each word chip shows status:
    - `Trained` (in label map)
    - `In progress` (dataset has 1..14 reps)
    - `Not started`
  - “Next planned word” CTA selects first planned word that is not trained (and maybe lowest reps)

**Custom word**
- Input field + “Check word” button:
  - calls `POST /api/temporal/signs/check`
  - if exists trained → show “Add more reps” path
  - if not planned → allow proceed as custom (still recorded into dataset folder with normalized name)

**Recording step (strict defaults)**
- Must always collect exactly:
  - 60 frames per rep
  - 15 reps per word
- Provide clear state machine UI:
  - `Ready` → `Countdown (3s first rep, 1s after)` → `Recording` → `Saved` → back to countdown
- After each rep saved:
  - call `POST /api/temporal/reps/add` with the rep frames
  - update progress indicator from response (`reps_collected`)

**Training step**
- Only enabled when `reps_collected >= 15`:
  - show “Train / Retrain LSTM Model” button
  - click triggers `POST /api/temporal/train`
  - poll `GET /api/temporal/train/status` until idle or error
- On success:
  - show “Done” state
  - trained indicator should update (refetch `GET /api/temporal/status`)
  - Communicate LSTM should now recognize this word after backend reload

---

## 6) Training Implementation Notes (Backend)
- Start from `train_temporal.py` logic but remove:
  - seaborn / matplotlib graphs
- Keep:
  - sliding window augmentation
  - dataset split stratified
  - early stopping by validation accuracy
  - checkpoint format compatibility with `verify_temporal.py`
- Ensure training does not block main thread:
  - run in `BackgroundTasks` or a dedicated thread with a lock
- Maintain separate retrain status from the MLP retrain to avoid mixing UI states.

---

## 7) Migration / File Cleanup (Repo Hygiene)
When implementing, normalize the untracked filenames:
- `collect_asl (1).py` → `collect_asl.py`
- `train_temporal (1).py` → `train_temporal.py`
- Move artifacts:
  - `temporal_lstm.pth` → `models/temporal/temporal_lstm.pth`
  - `temporal_label_map.json` → `models/temporal/temporal_label_map.json`
- Add planned list JSON:
  - `models/temporal/temporal_planned_words.json`

---

## 8) Acceptance Criteria
1) First visit with no localStorage key shows a model picker modal; selection persists; no repeated prompt on refresh.
2) Navbar allows switching MLP/LSTM at any time; switching clears the word buffer and resets session state.
3) Communicate:
   - MLP works as before.
   - LSTM shows predictions after buffer fills; word commit logic prevents rapid duplicates.
4) Signs:
   - MLP shows current GIF-based gallery.
   - LSTM shows a text-only list of trained temporal signs; no GIF UI.
5) Contribute:
   - MLP flow unchanged.
   - LSTM flow shows planned words + trained indicators + custom word check; records 15 reps × 60 frames; triggers retraining and updates trained set.

