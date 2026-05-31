#!/usr/bin/env python3
"""
evaluate.py — PadelLabs ML Toolchain
======================================
Evaluates a trained stroke classifier against the test split.
Prints a confusion matrix, per-class precision/recall/F1, and overall accuracy.

Expects the test folder structure produced by extract_windows.py:
    labeled-strokes/test/
        smash/      smash_00001.csv …
        forehand/   …
        unknown/    …

The classifier is loaded from a CoreML .mlmodel or from a scikit-learn .pkl file
(output of train_sklearn.py). If neither is provided, runs a random baseline.

Usage:
    # Evaluate against a sklearn model:
    python3 evaluate.py \
        --test   ../labeled-strokes/test \
        --model  ../models/v1/model.pkl

    # Dry-run (random baseline, shows expected output format):
    python3 evaluate.py --test ../labeled-strokes/test --baseline
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

FEATURE_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]

STROKE_TYPES = [
    "smash", "vibora", "bandeja", "rulo",
    "forehand", "backhand",
    "forehandLob", "backhandLob",
    "forehandVolley", "backhandVolley",
    "serve", "unknown",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load all windows from a split folder. Returns (X [N, 100*9], y [N])."""
    X_list, y_list = [], []

    for stroke_dir in sorted(split_dir.iterdir()):
        if not stroke_dir.is_dir():
            continue
        label = stroke_dir.name
        if label not in STROKE_TYPES:
            print(f"  SKIP unknown class dir: {label}")
            continue

        csv_files = list(stroke_dir.glob("*.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if not all(c in df.columns for c in FEATURE_COLS):
                    continue
                features = df[FEATURE_COLS].values  # (100, 9)
                if features.shape[0] != 100 or not np.isfinite(features).all():
                    continue
                X_list.append(features.flatten())   # (900,)
                y_list.append(label)
            except Exception:
                continue

    if not X_list:
        print(f"ERROR: No valid windows found in {split_dir}", file=sys.stderr)
        sys.exit(1)

    return np.array(X_list), y_list


# ── Classifier wrappers ───────────────────────────────────────────────────────

def predict_sklearn(X: np.ndarray, model_path: Path) -> list[str]:
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return list(model.predict(X))


def predict_baseline(y_true: list[str]) -> list[str]:
    """Random classifier that respects class distribution (lower bound)."""
    classes, counts = np.unique(y_true, return_counts=True)
    probs = counts / counts.sum()
    return list(np.random.choice(classes, size=len(y_true), p=probs))


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(y_true: list[str], y_pred: list[str], present_labels: list[str]):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  Overall accuracy: {acc*100:.1f}%  ({sum(yt==yp for yt,yp in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"{'='*60}")

    # Acceptance criteria check
    if acc >= 0.88:
        print("  ✅ PASS — meets Sprint 9 target (≥ 88%)")
    elif acc >= 0.80:
        print("  ⚠️  PARTIAL — meets Sprint 8 baseline (≥ 80%) but not production target")
    else:
        print("  ❌ FAIL — below Sprint 8 minimum (80%). Collect more data.")

    print("\n── Per-class report ──────────────────────────────────────")
    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_labels,
        zero_division=0,
        output_dict=True
    )
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for label in present_labels:
        r = report.get(label, {})
        p  = r.get("precision", 0)
        rc = r.get("recall", 0)
        f1 = r.get("f1-score", 0)
        sup = int(r.get("support", 0))
        flag = " ⚠️" if f1 < 0.78 else ""
        print(f"  {label:<22} {p:>10.3f} {rc:>8.3f} {f1:>8.3f} {sup:>9}{flag}")

    print("\n── Confusion matrix ──────────────────────────────────────")
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    header = "  " + "".join(f"{l[:6]:>8}" for l in present_labels)
    print(header)
    for i, row_label in enumerate(present_labels):
        row_str = "  " + f"{row_label[:10]:<12}" + "".join(
            f"{'['+str(v)+']':>8}" if j == i else f"{v:>8}"
            for j, v in enumerate(cm[i])
        )
        print(row_str)

    # Highlight worst confusions
    print("\n── Top confusions (off-diagonal) ─────────────────────────")
    off_diag = []
    for i in range(len(present_labels)):
        for j in range(len(present_labels)):
            if i != j and cm[i][j] > 0:
                off_diag.append((cm[i][j], present_labels[i], present_labels[j]))
    off_diag.sort(reverse=True)
    for count, true_label, pred_label in off_diag[:8]:
        pct = count / max(1, sum(cm[present_labels.index(true_label)]))
        print(f"  {true_label:<22} → {pred_label:<22}  {count:>4}×  ({pct*100:.0f}%)")

    if not off_diag:
        print("  (none)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",     required=True, help="Path to labeled-strokes/test/")
    parser.add_argument("--model",    help="Path to .pkl sklearn model")
    parser.add_argument("--baseline", action="store_true", help="Run random baseline (no model needed)")
    args = parser.parse_args()

    test_dir = Path(args.test)
    if not test_dir.exists():
        print(f"ERROR: {test_dir} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading test split from: {test_dir}")
    X, y_true = load_split(test_dir)
    present_labels = [l for l in STROKE_TYPES if l in set(y_true)]
    print(f"  {len(X)} windows  |  {len(present_labels)} classes")

    if args.baseline:
        print("\nRunning random baseline classifier…")
        y_pred = predict_baseline(y_true)
    elif args.model:
        model_path = Path(args.model)
        print(f"Loading model: {model_path.name}")
        y_pred = predict_sklearn(X, model_path)
    else:
        print("ERROR: provide --model or --baseline", file=sys.stderr)
        sys.exit(1)

    print_report(y_true, y_pred, present_labels)

    # Save confusion matrix image if matplotlib available
    try:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_labels)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
        ax.set_title(f"PadelLabs Stroke Classifier — Accuracy {accuracy_score(y_true, y_pred)*100:.1f}%")
        fig.tight_layout()
        out_path = Path(args.test).parent.parent / "models" / "confusion_matrix.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"\nConfusion matrix saved to: {out_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
