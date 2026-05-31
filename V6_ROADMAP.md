# V6 Roadmap & Handoff — Stroke Classifier

Single source of truth for "what's next" on the stroke classifier. Read this +
`CLAUDE.md` (the contract) + `models/evaluation_report.md` (the evidence log) before acting.
Written 2026-05-31 after the v4/v5 + temporal-model + orientation + segmentation investigation.

---

## 1. Current shipped state (DO NOT silently change)

- **Live model: v5** — RandomForest, trained on ALL 10 players (train+val+test, 6437 windows).
  - File: `models/v5/PadelLabs-StrokeClassifier-v5.mlmodel` (23.8 MB)
  - Hosted: `gs://padellabs-f40f7.firebasestorage.app/ml-models/PadelLabs-StrokeClassifier-v5.mlmodel`
  - Firebase Remote Config (project `padellabs-f40f7`): `ml_stroke_model_version=v5`,
    `ml_stroke_model_url=<firebase download-token URL>`. RC at version 10.
  - App: `ModelUpdateService.swift` downloads any HTTPS URL, compares version by STRING equality,
    compiles on iPhone, syncs to Watch. No App Store update needed to ship a new version.
- **v4** = the honest held-out-eval reference (trained on train/ only, 6 players). Keep for provenance.
- Corpus backup before the v4 rebuild: `labeled-strokes.bak-20260531-*`.

## 2. The core conclusion of the investigation (don't re-litigate)

Held-out (player-independent) accuracy on brand-new players is ~37% (RF). Everything tried:

| approach | result | why |
|---|---|---|
| Temporal 1D-CNN (raw) | val 69% / test 8.6% | overfits — 6 training players |
| CNN + gravity-relative | val 67% / test 26.7% | helped lefty (5→29) but still overfits |
| CNN + contact-aligned | val 71% / test 9.7% | alignment fixed (93%+) but still overfits |
| RF + gravity/world-frame (3 variants) | 26-30% (worse than 37.5) | RF is order-invariant → normalization only loses signal |
| RF + re-centering | worse | same — RF ignores temporal position |

**THE blocker is PLAYER DIVERSITY.** With ~6 training players any expressive model (CNN) overfits:
it already reaches ~70% on players similar to training but collapses on new ones. The RF only
"wins" held-out because it is too high-bias to overfit, but it is at its STRUCTURAL ceiling —
it uses order-invariant stats (mean/std/min/max) so it is blind to time-shape and physically
cannot separate lobs from drives (backhand_lob 2%, forehand_lob 19%, vibora confusions).

Corollary: no segmentation/orientation/normalization trick beats v5 right now. The next real
gain needs the TEMPORAL stack, and all parts together: more players + a 1D-CNN + clean
contact-aligned windows. The CNN-gravity run lifting backhand_lob 2%→49% proves the temporal
model sees the shape the RF can't — it just needs enough players to generalize.

## 3. Last changes shipped this session (recording enrichment — already in the app)

On-device DataCollection now records gravity + orientation quaternion, so a CORRECT
gravity-aligned world frame is possible offline (the earlier Euler-only reconstruction was
convention-fragile and lossy). v5 inference is UNTOUCHED.

- `Watch/.../WatchServices/MotionServiceProtocol.swift` — `MotionSample.gravity: CMAcceleration?`
- `Watch/.../WatchServices/MotionService.swift` — captures `motion.gravity`
- `Watch/.../DataCollection/.../DataCollectionViewModel.swift` — CSV header + row append:
  `gravityX,gravityY,gravityZ,quatW,quatX,quatY,quatZ` (7 trailing columns)

Backward-compatible: Python reads CSV by column NAME (ignores extras); the in-app CSV parser
reads only cols[0..3]. Old recordings without the columns still work.

## 4. THE PLAYBOOK — execute when new players are collected

Goal: train a temporal model that GENERALIZES (closes the val→test gap), then ship it as v6.

### Pre-req: collect ~20-30 players
- Each: a few sessions, ~5-10 of every stroke type, varied technique / wrist / watch model.
- Include MORE LEFT-HANDERS (only 1 today — Guillermo). Needed for honest LH eval in val AND test.
- New recordings carry gravity+quat columns automatically.
- Validate each session with the labeling tool (Stage 2 → `_validation.csv`).

### Step 1 — integrate validated sessions (use the `ml-ship-validated` skill)
- `apply_validation.py --json <each NEW session>` (only sessions NOT already in metadata.csv).
- playerId collision check (CLAUDE.md §8 rule 0); register new players in `metadata/players.csv`.
- Choose a player-independent split: hold out several players (incl. ≥1 left-hander) for val + test.

### Step 2 — re-extract with CONTACT-ALIGNED segmentation
- Today `extract_windows.py` anchors on the early spike marker (windows land peak ~45, not 30).
- `contact_segment.py` has the working contact detector (jerk / accel-peak forward search →
  93-98% aligned). PORT that anchor into `extract_windows.py` (replace `find_peak_index` with the
  contact search) so the saved corpus is contact-aligned. Keep the burst grouping + split logic.
- Re-audit with `segmentation_audit.py` → confirm contact@30±5 ≥ 90%.

### Step 3 — add a CORRECT gravity-aligned world-frame transform (now possible)
- Use the recorded gravity vector (not Euler) to rotate accel+gyro into a gravity frame.
- Re-test the variants in `directional_frame.py` / `prototype_gravity.py` BUT sourced from the
  real `gravity*`/`quat*` columns. Gate: it must HELP held-out (it hurt the RF on Euler data).
- For the CNN this is expected to help (gravity-relative already lifted the lefty 5→29).

### Step 4 — train the temporal model + honest eval
- `train_cnn.py` (raw) and the gravity/contact variants. Eval on the held-out test.
- DECISION GATE: ship the CNN ONLY if it BEATS v5 (RF) on the player-independent held-out test,
  especially on the held-out left-hander. If val≫test persists, you still need more players — hold.
- Also keep an RF retrain (`retrain.py --all-data`) as the safe fallback.

### Step 5 — ship v6
- Export the winning model to CoreML (CNN: torch→coremltools in the `padel-ml-eval` venv;
  RF: `retrain.py` in the `padel-ml` venv).
- `upload_model_firebase.py --version v6 --set-remote-config` (needs the service-account key at
  `GOOGLE_APPLICATION_CREDENTIALS`, in `PadelLabs Secrets/`).
- Verify: `firebase remoteconfig:get --project padellabs-f40f7`; curl the URL → 200.
- If the CNN input differs from the RF's 9-feature window, update the on-device
  `MotionDataProcessor`/`StrokeClassifier` to match the new model's input — and keep training and
  inference feature pipelines BYTE-IDENTICAL.

## 5. Tooling reference (all in scripts/, run via the venvs)

| script | purpose | venv |
|---|---|---|
| `apply_validation.py` | validation verdicts → JSON markers | either |
| `extract_windows.py` | raw → window corpus (TODO: port contact anchor) | either |
| `retrain.py` | train RF + CoreML export (`--all-data` for production) | **padel-ml** (sklearn 1.5.1) |
| `evaluate_cv.py` | player-GroupKFold honest CV (OVERWRITES evaluation_report.md — copy first) | padel-ml |
| `compare_versions.py` | byte-exact v_a vs v_b on held-out via CoreML predict | **padel-ml-eval** (py3.12) |
| `segmentation_audit.py` | window alignment / double-peak / prominence audit | padel-ml-eval |
| `contact_segment.py` | contact-aligned re-extraction + CNN eval (the anchor to port) | padel-ml-eval |
| `train_cnn.py` | 1D-CNN baseline on raw windows | padel-ml-eval |
| `prototype_gravity.py` / `directional_frame.py` | world-frame normalization experiments | padel-ml-eval |
| `upload_model_firebase.py` | upload .mlmodel → Firebase Storage + Remote Config | padel-ml |

### venv gotchas (do not lose)
- **`padel-ml`** = Python 3.14, sklearn 1.5.1 — REQUIRED for CoreML export (sklearn ≤1.5.1).
  CANNOT run `.predict()` on a .mlmodel (libcoremlpython missing on 3.14).
- **`padel-ml-eval`** = Python 3.12, sklearn 1.5.1 + torch (MPS) + coremltools — CAN run
  `.predict()`. Use for any held-out eval of a .mlmodel and all torch work.
- Global `python3` (3.14 / sklearn 1.8) CANNOT export CoreML — never use it for retrain.
- The CoreML model outputs an int64 class index; decode via embedded `classes` metadata.
- `PadelLabs-ML` is NOT under git — copy `evaluation_report.md` before running evaluate_cv.py,
  and never let the service-account key / `.env` get committed.

## 6. Open items / nice-to-haves
- Put `PadelLabs-ML` under git (lost v3's baseline + report history because it isn't).
- Port the contact anchor into `extract_windows.py` (Step 2) as the canonical change.
- Label-noise pass on lobs/bandeja/vibora (chronic confusions) during the next validation round.
- Consider hierarchical classes (overhead / groundstroke / volley → subtype) to sidestep the
  hardest confusions if the flat CNN still struggles.
- When the CNN ships, the on-device segmentation must ALSO be contact-aligned (match training).
