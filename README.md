# PadelLabs-ML — Stroke Classifier

Random Forest stroke classifier for padel, trained on 100 Hz wrist IMU data from an Apple Watch. Classifies 9 stroke types in real-time and ships to the PadelLabs iOS app via Firebase Remote Config + Firebase Storage.

**Current model:** v4 (v5 in `models/`) · 10 players · 6 437 windows · Player-GroupKFold 57.2 ± 7.3%

---

## Stroke classes

`forehand` · `backhand` · `forehand_volley` · `backhand_volley` · `forehand_lob` · `backhand_lob` · `smash` · `bandeja` · `vibora`

---

## Repo layout

```
padelLabs-ML/
├── scripts/                    pipeline scripts (do not duplicate — extend in place)
├── labeled-strokes/            training corpus (gitignored — large binary)
│   ├── train/ validation/ test/    one sub-folder per class
│   └── metadata.csv
├── metadata/
│   ├── players.csv             playerId, handedness, watchWrist, skillLevel
│   └── sessions.csv            per-session stats + split assignment
├── models/
│   ├── v1/ … v5/               shipped versions (.mlmodel + .json metadata)
│   └── evaluation_report.md    per-version accuracy log
├── CLAUDE.md                   agent operating guide (full pipeline spec)
├── DYNAMIC_MODEL_UPDATE.md     Firebase wiring for OTA model updates
└── LABEL_STUDIO_SETUP.md       legacy Label Studio QC path
```

Raw sessions and the web labeling tool live **outside** this repo:

```
/DataCollection/<Player>/<Date>/    raw 100 Hz CSV + JSON + _classified.csv
/labeling-tool/index.html           video-based stroke validation tool
```

---

## Pipeline stages

| Stage | Trigger | Key script |
|---|---|---|
| 1 — Classify session | "prepare for labeling" | `classify_session.py` |
| 2 — Label via web tool | user-driven, no script | `labeling-tool/index.html` |
| 3 — Integrate validated session | "_validation.csv ready" | `apply_validation.py` → `extract_windows.py` |
| 4 — Retrain readiness check | "can we retrain?" | bump criteria in `CLAUDE.md §5` |
| 5 — Retrain + evaluate + ship | "train v<N+1>" | `retrain.py` → `evaluate_cv.py` → `upload_model_firebase.py` |

Full step-by-step procedure in `CLAUDE.md`.

---

## Setup

Two Python virtual environments (both gitignored):

| Env | Python | Purpose |
|---|---|---|
| `padel-ml/` | 3.11 (sklearn ≤ 1.5.1) | Training + CoreML export via `retrain.py` |
| `padel-ml-eval/` | 3.14 | Evaluation + everything else |

Activate before running scripts:

```bash
source padel-ml/bin/activate        # for retrain.py
source padel-ml-eval/bin/activate   # for everything else
```

Key deps: `scikit-learn`, `coremltools`, `numpy`, `pandas`, `firebase-admin`.

---

## Quick commands

```bash
# Classify a raw session
python3 scripts/classify_session.py --session "/path/to/<sessionId>.csv"

# Apply video-validated corrections + extract windows into training corpus
python3 scripts/apply_validation.py --session <sessionId> --sessions-dir ../DataCollection
python3 scripts/extract_windows.py --sessions ../DataCollection --output ./labeled-strokes

# Three-way cross-validation (WARNING: overwrites evaluation_report.md — copy first)
python3 scripts/evaluate_cv.py --n-estimators 300

# Retrain (use padel-ml venv — coremltools export requires sklearn ≤ 1.5.1)
python3 scripts/retrain.py --output-version v<N+1>

# Upload to Firebase Storage + print Remote Config values
python3 scripts/upload_model_firebase.py --version v<N+1>
```

---

## Model accuracy (v4 — 10 players, player-independent split)

| Metric | Accuracy |
|---|---|
| Stratified 5-fold (optimistic) | 76.9% ± 0.6% |
| Session-GroupKFold (realistic) | 63.9% ± 8.7% |
| Player-GroupKFold (most honest) | 57.2% ± 7.3% |
| Held-out val (Victor + Carlos) | 67.5% |

Weakest classes: `backhand_lob` 25%, `forehand_lob` 34%, `vibora` 38%. Main confusion: lobs ↔ drives (similar wrist kinematics). See `models/evaluation_report.md` for full per-class breakdown.

---

## Retrain policy

Bump to v(N+1) only when **all** hold:
1. ≥ 500 new windows since last shipped version
2. New player with ≥ 100 windows across ≥ 3 classes **or** thin class grew by ≥ 100 windows
3. val + test each still contain ≥ 1 player not in train
4. Player-GroupKFold accuracy ≥ prior version baseline

See `CLAUDE.md §5` for the full gate checklist.
