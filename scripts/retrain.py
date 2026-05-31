#!/usr/bin/env python3
"""
retrain.py — PadelLabs ML Toolchain
=====================================
Retrains the stroke classifier from the labeled-strokes/ directory and exports
a new Core ML model.

Steps:
  1. Reads all per-stroke window CSVs from labeled-strokes/train/
  2. Extracts 89 features per window (matches StrokeClassifier.swift feature list)
  3. Trains a Random Forest classifier
  4. Evaluates on labeled-strokes/validation/ (if populated) or cross-val
  5. Exports the model to models/PadelLabs-StrokeClassifier-v<N>.mlmodel (90 features)

Usage:
    python3 retrain.py
    python3 retrain.py --output-version v2
    python3 retrain.py --n-estimators 300 --dry-run

Output:
    models/PadelLabs-StrokeClassifier-v<N>.mlmodel  — Core ML model
    models/PadelLabs-StrokeClassifier-v<N>.json      — version metadata
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
ML_DIR       = SCRIPT_DIR.parent
STROKES_DIR  = ML_DIR / "labeled-strokes"
MODELS_DIR   = ML_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Feature extraction (must match StrokeClassifier.swift + extract_windows.py) ──

WINDOW_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 90 features from a 100-sample window DataFrame.

    Feature layout (90 total):
      9 channels × 9 per-channel stats  = 81  (mean,std,min,max,range,totalVar,rms,p25,p75)
      accelMag × 5 stats                =  5  (max,mean,std,totalVar,peakPos)
      accelX_gyroZ_corr                 =  1
      3 directional / temporal features =  3  (gyroX_slope, accelZ_slope, gyroY_asym)
    """
    feats = []
    for col in WINDOW_COLS:
        vals = df[col].values.astype(float)
        feats += [
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.max(vals) - np.min(vals)),
            float(np.sum(np.abs(np.diff(vals)))),      # total variation
            float(np.sqrt(np.mean(vals**2))),           # RMS
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)),
        ]
    # Resultant acceleration magnitude features
    if all(c in df.columns for c in ["accelX", "accelY", "accelZ"]):
        mag = np.sqrt(df["accelX"]**2 + df["accelY"]**2 + df["accelZ"]**2)
        feats += [float(np.max(mag)), float(np.mean(mag)), float(np.std(mag)),
                  float(np.sum(np.abs(np.diff(mag.values)))), float(np.argmax(mag.values) / len(mag))]
    else:
        feats += [0.0] * 5
    # Cross-correlation between accelX and gyroZ (proxy for wrist rotation timing)
    if "accelX" in df.columns and "gyroZ" in df.columns:
        a, g = df["accelX"].values, df["gyroZ"].values
        if np.std(a) > 0 and np.std(g) > 0:
            feats.append(float(np.corrcoef(a, g)[0, 1]))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # ── Directional / temporal features (lob vs volley separators) ────────────
    # gyroX_slope: last-30 minus first-30 mean — positive = arm lifting (lob signature)
    if "gyroX" in df.columns:
        gx = df["gyroX"].values.astype(float)
        feats.append(float(gx[-30:].mean() - gx[:30].mean()))
    else:
        feats.append(0.0)

    # accelZ_slope: vertical acceleration trend through the stroke
    if "accelZ" in df.columns:
        az = df["accelZ"].values.astype(float)
        feats.append(float(az[-30:].mean() - az[:30].mean()))
    else:
        feats.append(0.0)

    # gyroY_asym: wrist-rotation timing — ratio of first-half to second-half mean magnitude
    # lobs have a more sustained follow-through; volleys are front-loaded
    if "gyroY" in df.columns:
        gy = np.abs(df["gyroY"].values.astype(float))
        mid = len(gy) // 2
        denom = gy[mid:].mean() if gy[mid:].mean() > 1e-6 else 1e-6
        feats.append(float(gy[:mid].mean() / denom))
    else:
        feats.append(1.0)

    return np.array(feats, dtype=np.float64)

def load_split(split_dir: Path, label_filter=None):
    """Load all windows from a split directory. Returns (X, y, paths)."""
    X, y, paths = [], [], []
    if not split_dir.exists():
        return np.array([]), [], []
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label_filter and label not in label_filter:
            continue
        for csv_file in sorted(label_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                if len(df) < 60:
                    continue
                feats = extract_features(df)
                if not np.isfinite(feats).all():
                    continue
                X.append(feats)
                y.append(label)
                paths.append(str(csv_file))
            except Exception as e:
                print(f"  [WARN] Skipping {csv_file.name}: {e}")
    return (np.array(X) if X else np.array([])), y, paths

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain PadelLabs stroke classifier")
    parser.add_argument("--output-version", default=None, help="e.g. v2 (auto-increments if omitted)")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees (default 300)")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth (default None = unlimited)")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Train but do not save the model")
    parser.add_argument("--all-data", action="store_true",
                        help="PRODUCTION mode: train on train+validation+test (every labeled window). "
                             "Use only after held-out eval is done — there is no held-out set left.")
    parser.add_argument("--exclude-labels", nargs="*", default=["unknown"], help="Labels to exclude")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Include 'unknown' as a 10th class (10-label model). "
                             "Swift client must be updated to handle the extra label. "
                             "Auto-subsamples unknown to --unknown-cap to avoid class imbalance.")
    parser.add_argument("--unknown-cap", type=int, default=500,
                        help="Max 'unknown' samples to keep (default 500, ~avg of stroke classes)")
    parser.add_argument("--calibrate", choices=["none", "isotonic", "sigmoid"], default="none",
                        help="Wrap classifier in CalibratedClassifierCV so probabilities are honest. "
                             "WARNING: Core ML export of a calibrated RF is NOT supported — only enable "
                             "if evaluating; pair with --dry-run unless Core ML issue is resolved.")
    parser.add_argument("--dump-importance", action="store_true",
                        help="Write feature importance table beside the model")
    args = parser.parse_args()

    # ── Auto-increment version ─────────────────────────────────────────────────
    if args.output_version is None:
        existing = sorted(MODELS_DIR.glob("v*/PadelLabs-StrokeClassifier-v*.mlmodel"))
        if existing:
            last = existing[-1].stem  # e.g. PadelLabs-StrokeClassifier-v1
            try:
                num = int(last.split("-v")[-1])
                args.output_version = f"v{num + 1}"
            except ValueError:
                args.output_version = "v2"
        else:
            args.output_version = "v1"

    model_name = f"PadelLabs-StrokeClassifier-{args.output_version}"
    version_dir = MODELS_DIR / args.output_version
    version_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  PadelLabs Stroke Classifier — Retrain → {model_name}")
    print(f"{'='*60}\n")

    # ── Load training data ─────────────────────────────────────────────────────
    print("Loading training windows...")
    exclude = set(args.exclude_labels)
    if args.include_unknown and "unknown" in exclude:
        exclude.discard("unknown")
        print("  [flag] --include-unknown active: 'unknown' added as a 10th class")
    if args.all_data:
        print("  [flag] --all-data active: training on train+validation+test (NO held-out set)")
        parts_X, parts_y = [], []
        for sp in ("train", "validation", "test"):
            Xp, yp, _ = load_split(STROKES_DIR / sp, label_filter=None)
            if len(Xp):
                parts_X.append(Xp); parts_y.extend(yp)
        X_train = np.vstack(parts_X) if parts_X else np.array([])
        y_train = parts_y
    else:
        X_train, y_train, _ = load_split(STROKES_DIR / "train", label_filter=None)

    if len(X_train) == 0:
        print("[ERROR] No training windows found in labeled-strokes/")
        sys.exit(1)

    # Filter excluded labels
    mask = np.array([label not in exclude for label in y_train])
    X_train, y_train = X_train[mask], [y for y, m in zip(y_train, mask) if m]

    # Cap unknown samples if included
    if args.include_unknown and args.unknown_cap > 0:
        unknown_idx = [i for i, lbl in enumerate(y_train) if lbl == "unknown"]
        if len(unknown_idx) > args.unknown_cap:
            rng = np.random.default_rng(42)
            keep_unknown = set(rng.choice(unknown_idx, size=args.unknown_cap, replace=False))
            keep_mask = np.array([lbl != "unknown" or i in keep_unknown
                                  for i, lbl in enumerate(y_train)])
            X_train = X_train[keep_mask]
            y_train = [y for y, m in zip(y_train, keep_mask) if m]
            print(f"  [subsample] unknown capped at {args.unknown_cap} (dropped {len(unknown_idx)-args.unknown_cap})")

    class_counts = Counter(y_train)
    print(f"  Training samples: {len(y_train)}")
    print(f"  Classes ({len(class_counts)}): {', '.join(sorted(class_counts.keys()))}")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:25s}: {count}")

    # ── Load validation data ───────────────────────────────────────────────────
    X_val, y_val, _ = load_split(STROKES_DIR / "validation")
    mask_val = np.array([label not in exclude for label in y_val]) if len(y_val) > 0 else np.array([], dtype=bool)
    if len(X_val) > 0 and mask_val.any():
        X_val = X_val[mask_val]
        y_val = [y for y, m in zip(y_val, mask_val) if m]
        print(f"\n  Validation samples: {len(y_val)}")
    else:
        X_val, y_val = None, None
        print("\n  No separate validation split — will use cross-validation on training set.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining Random Forest (n_estimators={args.n_estimators})...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    class_names = list(le.classes_)

    base_clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    if args.calibrate != "none":
        from sklearn.calibration import CalibratedClassifierCV
        print(f"  [flag] wrapping in CalibratedClassifierCV(method={args.calibrate}) — probabilities will be calibrated")
        clf = CalibratedClassifierCV(base_clf, method=args.calibrate, cv=3)
    else:
        clf = base_clf
    clf.fit(X_train, y_encoded)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n── Evaluation ──────────────────────────────────────────────────")
    if X_val is not None:
        y_val_enc = le.transform(y_val)
        val_acc = clf.score(X_val, y_val_enc)
        print(f"  Validation accuracy: {val_acc*100:.1f}%")
    else:
        cv_scores = cross_val_score(clf, X_train, y_encoded, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"  5-fold cross-val accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Per-class train accuracy
    train_preds = clf.predict(X_train)
    from sklearn.metrics import classification_report
    print("\n── Per-class accuracy (train) ──────────────────────────────────")
    report = classification_report(y_encoded, train_preds, target_names=class_names, digits=2)
    print(report)

    # ── Feature importance dump ───────────────────────────────────────────────
    if args.dump_importance:
        feat_names = []
        for col in WINDOW_COLS:
            for stat in ["mean","std","min","max","range","totalVar","rms","p25","p75"]:
                feat_names.append(f"{col}_{stat}")
        feat_names += ["accelMag_max","accelMag_mean","accelMag_std","accelMag_totalVar","accelMag_peakPos",
                       "accelX_gyroZ_corr","gyroX_slope","accelZ_slope","gyroY_asym"]
        imp_source = base_clf if args.calibrate != "none" else clf
        try:
            imp = imp_source.feature_importances_
            order = np.argsort(imp)[::-1]
            imp_rows = ["feature,importance,rank"]
            for rk, i in enumerate(order, 1):
                imp_rows.append(f"{feat_names[i]},{imp[i]:.6f},{rk}")
            imp_path = version_dir / f"{model_name}-feature_importance.csv"
            imp_path.write_text("\n".join(imp_rows) + "\n")
            print(f"  Feature importance: {imp_path}")
        except AttributeError:
            print("  [WARN] feature_importances_ unavailable for calibrated model — skipping dump")

    if args.dry_run:
        print("\n[DRY RUN] Skipping model export.")
        return

    if args.calibrate != "none":
        print("\n[ERROR] Core ML export of CalibratedClassifierCV is not supported.")
        print("        Use --dry-run alongside --calibrate to evaluate calibrated metrics only.")
        sys.exit(2)

    # ── Export to Core ML ──────────────────────────────────────────────────────
    print("Exporting to Core ML...")
    import coremltools as ct

    # Feature names must match StrokeClassifier.swift
    feature_names = []
    for col in WINDOW_COLS:
        for stat in ["mean", "std", "min", "max", "range", "totalVar", "rms", "p25", "p75"]:
            feature_names.append(f"{col}_{stat}")
    feature_names += ["accelMag_max", "accelMag_mean", "accelMag_std", "accelMag_totalVar", "accelMag_peakPos"]
    feature_names += ["accelX_gyroZ_corr"]
    feature_names += ["gyroX_slope", "accelZ_slope", "gyroY_asym"]

    assert len(feature_names) == X_train.shape[1], \
        f"Feature count mismatch: {len(feature_names)} names vs {X_train.shape[1]} features"

    # coremltools sklearn API: positional args (feature_names, target_name)
    cml_model = ct.converters.sklearn.convert(clf, feature_names, "strokeType")

    # Set metadata
    cml_model.short_description = f"PadelLabs Stroke Classifier {args.output_version}"
    cml_model.version = args.output_version
    cml_model.author = "PadelLabs"
    cml_model.license = "Proprietary"
    cml_model.user_defined_metadata["classes"] = json.dumps(class_names)
    cml_model.user_defined_metadata["trained_at"] = datetime.utcnow().isoformat()
    cml_model.user_defined_metadata["n_train_samples"] = str(len(y_train))
    cml_model.user_defined_metadata["n_estimators"] = str(args.n_estimators)

    out_mlmodel = version_dir / f"{model_name}.mlmodel"
    cml_model.save(str(out_mlmodel))
    print(f"  Saved: {out_mlmodel}")

    # ── Write version metadata ──────────────────────────────────────────────────
    meta = {
        "version": args.output_version,
        "model_file": f"{model_name}.mlmodel",
        "trained_at": datetime.utcnow().isoformat(),
        "n_estimators": args.n_estimators,
        "n_train_samples": len(y_train),
        "classes": class_names,
        "class_distribution": dict(class_counts),
    }
    out_json = version_dir / f"{model_name}.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata: {out_json}")

    print(f"\n✅ Done — {model_name}")
    print(f"   Next: compile + zip the model, upload to Supabase Storage, update Remote Config version.\n")

if __name__ == "__main__":
    main()
