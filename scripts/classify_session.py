#!/usr/bin/env python3
"""
classify_session.py — PadelLabs ML Toolchain
==============================================
Classifies all spike markers in a raw session (CSV + JSON), compares against
ground-truth labels (drillLabeled mode), and prints a full report.

For an honest evaluation the model is retrained on all labeled-strokes/train windows
EXCEPT those from the target session (leave-one-session-out). This avoids inflated
accuracy from training on the session being tested.

Usage:
    python3 classify_session.py --session <path/to/session.csv>
    python3 classify_session.py --session <path/to/session.csv> --all-train --version v3

    --all-train   skip the hold-out exclusion (biased but shows max accuracy)
    --version     model version label used in the output filename (default: v3)
    --threshold   confidence threshold for unknown (default: 0.55)
    --no-output   don't save results CSV
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
ML_DIR      = SCRIPT_DIR.parent
STROKES_DIR = ML_DIR / "labeled-strokes"

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_COLS  = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]
STAT_NAMES   = ["mean", "std", "min", "max", "range", "totalVar", "rms", "p25", "p75"]
PRE_PEAK     = 30
POST_PEAK    = 70
WINDOW_SIZE  = PRE_PEAK + POST_PEAK   # 100

BURST_GAP    = 1.2   # seconds — same as extract_windows.py

# ── Feature extraction (identical to retrain.py) ──────────────────────────────

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """90 features from a 100-sample window. Must match retrain.py exactly."""
    feats = []
    for col in WINDOW_COLS:
        vals = df[col].values.astype(float)
        feats += [
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.max(vals) - np.min(vals)),
            float(np.sum(np.abs(np.diff(vals)))),       # totalVar
            float(np.sqrt(np.mean(vals**2))),            # rms
            float(np.percentile(vals, 25)),
            float(np.percentile(vals, 75)),
        ]
    # accelMag: max, mean, std, totalVar, peakPos
    if all(c in df.columns for c in ["accelX", "accelY", "accelZ"]):
        mag = np.sqrt(df["accelX"]**2 + df["accelY"]**2 + df["accelZ"]**2)
        feats += [
            float(np.max(mag)),
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.sum(np.abs(np.diff(mag.values)))),
            float(np.argmax(mag.values) / len(mag)),
        ]
    else:
        feats += [0.0] * 5
    # accelX vs gyroZ correlation
    if "accelX" in df.columns and "gyroZ" in df.columns:
        a, g = df["accelX"].values, df["gyroZ"].values
        feats.append(float(np.corrcoef(a, g)[0, 1]) if np.std(a) > 0 and np.std(g) > 0 else 0.0)
    else:
        feats.append(0.0)
    # Directional / temporal features
    gx = df["gyroX"].values.astype(float) if "gyroX" in df.columns else np.zeros(len(df))
    feats.append(float(gx[-30:].mean() - gx[:30].mean()))          # gyroX_slope
    az = df["accelZ"].values.astype(float) if "accelZ" in df.columns else np.zeros(len(df))
    feats.append(float(az[-30:].mean() - az[:30].mean()))          # accelZ_slope
    if "gyroY" in df.columns:
        gy = np.abs(df["gyroY"].values.astype(float))
        mid = len(gy) // 2
        denom = gy[mid:].mean() if gy[mid:].mean() > 1e-6 else 1e-6
        feats.append(float(gy[:mid].mean() / denom))               # gyroY_asym
    else:
        feats.append(1.0)
    return np.array(feats, dtype=np.float64)

# ── Training ──────────────────────────────────────────────────────────────────

def load_split(split_dir: Path, exclude_session_id: str | None = None):
    """Load all windows from split_dir, optionally excluding one session."""
    meta_path = STROKES_DIR / "metadata.csv"
    excluded_files: set[str] = set()
    if exclude_session_id and meta_path.exists():
        meta = pd.read_csv(meta_path)
        excluded_files = set(
            meta[meta["session_id"] == exclude_session_id]["window_file"].tolist()
        )

    X, y = [], []
    if not split_dir.exists():
        return np.array([]), []

    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for csv_file in sorted(label_dir.glob("*.csv")):
            rel_path = str(csv_file.relative_to(STROKES_DIR))
            if rel_path in excluded_files:
                continue
            try:
                df = pd.read_csv(csv_file)
                if len(df) < 60:
                    continue
                feats = extract_features(df)
                if not np.isfinite(feats).all():
                    continue
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"  [WARN] Skipping {csv_file.name}: {e}")

    return (np.array(X) if X else np.array([])), y


def train_classifier(exclude_session_id: str | None = None):
    """Train RF on labeled-strokes/train, excluding one session for honest eval."""
    exclude_label = "unknown"
    X_train, y_train = load_split(STROKES_DIR / "train", exclude_session_id)

    if len(X_train) == 0:
        print("[ERROR] No training windows found.")
        sys.exit(1)

    mask = [label != exclude_label for label in y_train]
    X_train = X_train[mask]
    y_train  = [y for y, m in zip(y_train, mask) if m]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_enc)
    return clf, le

# ── Session loading & window extraction ───────────────────────────────────────

def group_bursts(markers: list, burst_gap: float = BURST_GAP) -> list:
    """Collapse markers within burst_gap seconds — same logic as extract_windows.py."""
    valid = sorted(
        [m for m in markers if float(m.get("timestamp", -1)) >= 0],
        key=lambda m: float(m["timestamp"])
    )
    if not valid:
        return []

    bursts, current = [], []
    for m in valid:
        ts = float(m["timestamp"])
        if not current or ts - float(current[0]["timestamp"]) <= burst_gap:
            current.append(m)
        else:
            bursts.append(current)
            current = [m]
    if current:
        bursts.append(current)

    reps = []
    for burst in bursts:
        label = next(
            (m.get("strokeType", "unknown") for m in burst
             if m.get("strokeType", "unknown") not in ("unknown", "")),
            "unknown"
        )
        reps.append({
            "timestamp": float(burst[0]["timestamp"]),
            "strokeType": label,
            "burst_size": len(burst),
        })
    return reps


def dedup_preparation_bursts(
    bursts: list,
    imu_df: pd.DataFrame,
    slow_window_s: float = 3.5,
    weak_ratio: float = 0.40,
) -> list:
    """
    Drop preparation-movement spikes: if burst[i] is within slow_window_s of
    burst[i+1] AND burst[i]'s accelMag peak < weak_ratio × burst[i+1]'s peak,
    burst[i] was a backswing — drop it.
    """
    if len(bursts) <= 1:
        return bursts

    def peak_mag(ts: float) -> float:
        idx = int((imu_df["timestamp"] - ts).abs().idxmin())
        window = imu_df.iloc[max(0, idx - 30): idx + 50]
        mag = np.sqrt(window.accelX**2 + window.accelY**2 + window.accelZ**2)
        return float(mag.max())

    kept, skip_next = [], False
    for i in range(len(bursts)):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(bursts):
            gap = float(bursts[i + 1]["timestamp"]) - float(bursts[i]["timestamp"])
            if gap <= slow_window_s:
                mag_i = peak_mag(float(bursts[i]["timestamp"]))
                mag_j = peak_mag(float(bursts[i + 1]["timestamp"]))
                if mag_i < weak_ratio * mag_j:
                    if bursts[i]["strokeType"] != "unknown" and bursts[i + 1]["strokeType"] == "unknown":
                        bursts[i + 1]["strokeType"] = bursts[i]["strokeType"]
                    skip_next = True
                    continue
        kept.append(bursts[i])
    if not skip_next and bursts:
        kept.append(bursts[-1])
    elif bursts:
        kept.append(bursts[-1])
    return kept


def extract_window(df: pd.DataFrame, peak_idx: int) -> pd.DataFrame | None:
    start, end = peak_idx - PRE_PEAK, peak_idx + POST_PEAK
    if start < 0 or end > len(df):
        return None
    w = df.iloc[start:end][WINDOW_COLS].copy().reset_index(drop=True)
    if len(w) != WINDOW_SIZE or not np.isfinite(w.values).all():
        return None
    return w

# ── Classification ────────────────────────────────────────────────────────────

def classify_session(csv_path: Path, json_path: Path,
                     clf: RandomForestClassifier, le: LabelEncoder,
                     threshold: float = 0.55):
    """Classify all markers in a session. Returns a list of result dicts."""
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        meta = json.load(f)

    markers = group_bursts(meta.get("markers", []))
    before = len(markers)
    markers = dedup_preparation_bursts(markers, df)
    dropped = before - len(markers)
    if dropped:
        print(f"  Dedup: removed {dropped} preparation-spike bursts (weak backswing before strong contact)")
    if not markers:
        print("[WARN] No markers found in session JSON.")
        return []

    results = []
    skipped = 0

    for m in markers:
        ts = m["timestamp"]
        true_label = m["strokeType"]

        # Find nearest sample then re-anchor on actual accelMag peak (±20 samples)
        if "timestamp" not in df.columns:
            break
        coarse = int((df["timestamp"] - ts).abs().idxmin())
        s0 = max(0, coarse - 20)
        s1 = min(len(df), coarse + 21)
        mag_search = np.sqrt(df["accelX"].iloc[s0:s1]**2 + df["accelY"].iloc[s0:s1]**2 + df["accelZ"].iloc[s0:s1]**2)
        peak_idx = s0 + int(mag_search.values.argmax())

        window = extract_window(df, peak_idx)
        if window is None:
            skipped += 1
            continue

        feats = extract_features(window)
        if not np.isfinite(feats).all():
            skipped += 1
            continue

        proba = clf.predict_proba(feats.reshape(1, -1))[0]

        # Sort by probability descending → top-3
        top_indices = np.argsort(proba)[::-1]
        top1_label = le.classes_[top_indices[0]]
        top1_conf  = float(proba[top_indices[0]])
        top2_label = le.classes_[top_indices[1]] if len(top_indices) > 1 else ""
        top2_conf  = float(proba[top_indices[1]]) if len(top_indices) > 1 else 0.0
        top3_label = le.classes_[top_indices[2]] if len(top_indices) > 2 else ""
        top3_conf  = float(proba[top_indices[2]]) if len(top_indices) > 2 else 0.0

        above = top1_conf >= threshold
        final_label = top1_label if above else "unknown"

        results.append({
            "timestamp_s":      round(ts, 4),
            "original_label":   true_label,
            "predicted_stroke": final_label,
            "confidence":       round(top1_conf, 4),
            "above_threshold":  "yes" if above else "no",
            "samples_in_window": len(window),
            "burst_size":       m["burst_size"],
            "correct":          final_label == true_label,
            "top1":             top1_label,
            "conf1":            round(top1_conf, 4),
            "top2":             top2_label,
            "conf2":            round(top2_conf, 4),
            "top3":             top3_label,
            "conf3":            round(top3_conf, 4),
        })

    if skipped:
        print(f"  [INFO] Skipped {skipped} markers (window out of bounds)")

    return results

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list, threshold: float, version: str = "v3"):
    df = pd.DataFrame(results)
    if df.empty:
        print("No results to report.")
        return

    # Exclude 'unknown' ground-truth from accuracy (spikes with no label)
    labeled = df[df["original_label"] != "unknown"]
    unlabeled = df[df["original_label"] == "unknown"]

    print(f"\n{'='*65}")
    print(f"  CLASSIFICATION REPORT — {version} model (threshold={threshold})")
    print(f"{'='*65}")
    print(f"  Total markers:        {len(df)}")
    print(f"  Labeled strokes:      {len(labeled)}  (ground-truth known)")
    print(f"  Unlabeled spikes:     {len(unlabeled)}  (true label = unknown)")

    if len(labeled) == 0:
        print("  No labeled strokes to evaluate.")
        return

    acc = labeled["correct"].mean()
    print(f"\n  Accuracy on labeled:  {acc*100:.1f}%")

    # Confidence distribution
    print(f"\n── Confidence Distribution ─────────────────────────────────────")
    bins = [0, 0.3, 0.4, 0.5, 0.55, 0.65, 0.75, 0.9, 1.01]
    labels_b = ["0–30%","30–40%","40–50%","50–55%","55–65%","65–75%","75–90%","90–100%"]
    counts, _ = np.histogram(df["confidence"], bins=bins)
    for lbl, cnt in zip(labels_b, counts):
        bar = "█" * (cnt * 30 // max(counts, default=1))
        print(f"  {lbl:12s} {cnt:5d}  {bar}")

    above = (df["above_threshold"] == "yes").sum()
    print(f"\n  Above threshold:  {above}/{len(df)} ({above/len(df)*100:.0f}%) classified")
    print(f"  Below threshold:  {len(df)-above}/{len(df)} → unknown")

    # Per-class results (labeled only)
    print(f"\n── Per-class Accuracy ──────────────────────────────────────────")
    classes = sorted(labeled["original_label"].unique())
    print(f"  {'Stroke':<22} {'Total':>6} {'Correct':>8} {'Acc':>6} {'→unknown':>9}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*6} {'-'*9}")
    for cls in classes:
        sub = labeled[labeled["original_label"] == cls]
        correct = sub["correct"].sum()
        to_unknown = (sub["predicted_stroke"] == "unknown").sum()
        cls_acc = correct / len(sub) if len(sub) > 0 else 0
        print(f"  {cls:<22} {len(sub):>6} {correct:>8} {cls_acc*100:>5.0f}% {to_unknown:>9}")

    # Where errors go
    print(f"\n── Where Errors Go ─────────────────────────────────────────────")
    wrong = labeled[~labeled["correct"]]
    if len(wrong) == 0:
        print("  No errors! Perfect accuracy.")
    else:
        confusion = wrong.groupby(["original_label", "predicted_stroke"]).size().reset_index(name="count")
        confusion = confusion.sort_values("count", ascending=False).head(15)
        print(f"  {'True':<22} → {'Predicted':<22} {'Count':>6}")
        print(f"  {'-'*22}   {'-'*22} {'-'*6}")
        for _, row in confusion.iterrows():
            print(f"  {row['original_label']:<22} → {row['predicted_stroke']:<22} {row['count']:>6}")

    # Unlabeled spike predictions
    if len(unlabeled) > 0:
        print(f"\n── Unlabeled Spikes (model guesses, no ground truth) ───────────")
        pred_counts = unlabeled["predicted_stroke"].value_counts()
        for pred, cnt in pred_counts.items():
            avg_conf = unlabeled[unlabeled["predicted_stroke"]==pred]["confidence"].mean()
            print(f"  {pred:<22} {cnt:>5}  avg conf: {avg_conf:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify a PadelLabs session.")
    parser.add_argument("--session", required=True,
                        help="Path to session .csv file (JSON sidecar must exist alongside it)")
    parser.add_argument("--all-train", action="store_true",
                        help="Do NOT exclude the target session from training (biased eval)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Confidence threshold for 'unknown' (default 0.55)")
    parser.add_argument("--version", default="v3",
                        help="Model version label used in output filename (default: v3)")
    parser.add_argument("--no-output", action="store_true",
                        help="Don't save results CSV")
    args = parser.parse_args()

    csv_path  = Path(args.session)
    json_path = csv_path.with_suffix(".json")

    if not csv_path.exists():
        print(f"[ERROR] Session CSV not found: {csv_path}")
        sys.exit(1)
    if not json_path.exists():
        print(f"[ERROR] Session JSON not found: {json_path}")
        sys.exit(1)

    # Read session metadata
    with open(json_path) as f:
        meta = json.load(f)

    session_id    = meta.get("sessionId", csv_path.stem)
    session_mode  = meta.get("mode", "unknown")
    session_date  = meta.get("date", "?")[:10]

    print(f"\nSession: {csv_path.name}")
    print(f"  Date: {session_date}  |  Mode: {session_mode}  |  Markers: {len(meta.get('markers',[]))}")

    # Train (excluding this session for honest eval)
    exclude_id = None if args.all_train else session_id
    hold_out_note = "(all-train — biased)" if args.all_train else f"(hold-out: {session_id[:8]}…)"
    print(f"\nTraining RF on labeled-strokes/train {hold_out_note} ...")
    clf, le = train_classifier(exclude_session_id=exclude_id)
    print(f"  Classes: {list(le.classes_)}")
    print(f"  RF ready.")

    # Classify
    print(f"\nClassifying {csv_path.name} ...")
    results = classify_session(csv_path, json_path, clf, le, threshold=args.threshold)

    if not results:
        print("No results produced.")
        sys.exit(1)

    # Report
    print_report(results, threshold=args.threshold, version=args.version)

    # Save
    if not args.no_output:
        out_stem = csv_path.stem[:8]
        out_path = csv_path.parent / f"{out_stem}_{args.version}_classified.csv"
        col_order = [
            "timestamp_s", "original_label", "predicted_stroke", "confidence",
            "above_threshold", "samples_in_window", "burst_size", "correct",
            "top1", "conf1", "top2", "conf2", "top3", "conf3",
        ]
        pd.DataFrame(results)[col_order].to_csv(out_path, index=False)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
