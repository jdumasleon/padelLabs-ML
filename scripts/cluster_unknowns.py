#!/usr/bin/env python3
"""
cluster_unknowns.py — PadelLabs ML Toolchain
==============================================
Groups low-confidence spikes by IMU feature similarity (k-means),
then lets you label one representative per cluster instead of
reviewing every spike individually.

After running:
  1. Open the output cluster_review/ folder
  2. Watch each cluster_NNN_representative.mp4 (one per cluster)
  3. Fill in the label column in cluster_summary.csv
  4. Run with --apply to push cluster labels back into the session JSON

Usage:
    # Step 1: Cluster and extract representative clips
    python3 scripts/cluster_unknowns.py \
        --classified "/path/to/8C051DBC_v2_classified.csv" \
        --session-csv "/path/to/8C051DBC-....csv" \
        --video "/path/to/21 April.MOV" \
        --review-csv "/path/to/review/21-april/review.csv" \
        --output-dir "/path/to/cluster_review" \
        --n-clusters 25 \
        --max-conf 0.45      # only cluster spikes below this confidence

    # Step 2: After filling cluster_summary.csv with labels, apply them
    python3 scripts/cluster_unknowns.py \
        --apply \
        --cluster-summary "/path/to/cluster_review/cluster_summary.csv" \
        --session-json "/path/to/8C051DBC-....json"

Requirements (all in padel-ml):
    numpy, pandas, scipy, sklearn
    ffmpeg for clip extraction
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Feature extraction (must match retrain.py / StrokeClassifier.swift) ───────

WINDOW_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]

def extract_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for col in WINDOW_COLS:
        vals = df[col].values.astype(float)
        feats += [
            float(np.mean(vals)), float(np.std(vals)),
            float(np.min(vals)),  float(np.max(vals)),
            float(np.max(vals) - np.min(vals)),
            float(np.sum(np.abs(np.diff(vals)))),
            float(np.sqrt(np.mean(vals**2))),
            float(np.percentile(vals, 25)), float(np.percentile(vals, 75)),
        ]
    mag = np.sqrt(df["accelX"]**2 + df["accelY"]**2 + df["accelZ"]**2)
    feats += [float(np.max(mag)), float(np.mean(mag)), float(np.std(mag)),
              float(np.sum(np.abs(np.diff(mag.values)))),
              float(np.argmax(mag.values) / len(mag))]
    a, g = df["accelX"].values, df["gyroZ"].values
    feats.append(float(np.corrcoef(a, g)[0, 1]) if np.std(a) > 0 and np.std(g) > 0 else 0.0)
    gx = df["gyroX"].values.astype(float)
    feats.append(float(gx[-30:].mean() - gx[:30].mean()))
    az = df["accelZ"].values.astype(float)
    feats.append(float(az[-30:].mean() - az[:30].mean()))
    gy = np.abs(df["gyroY"].values.astype(float))
    mid = len(gy) // 2
    denom = gy[mid:].mean() if gy[mid:].mean() > 1e-6 else 1e-6
    feats.append(float(gy[:mid].mean() / denom))
    return np.array(feats, dtype=np.float64)


def find_peak_index(df: pd.DataFrame, marker_ts: float) -> int:
    coarse = int((df["timestamp"] - marker_ts).abs().idxmin())
    search_start = max(0, coarse - 20)
    search_end   = min(len(df), coarse + 21)
    mag = np.sqrt(
        df["accelX"].iloc[search_start:search_end] ** 2 +
        df["accelY"].iloc[search_start:search_end] ** 2 +
        df["accelZ"].iloc[search_start:search_end] ** 2
    )
    return search_start + int(mag.values.argmax())


def extract_window(df: pd.DataFrame, peak_idx: int, pre=30, post=70) -> pd.DataFrame | None:
    start = peak_idx - pre
    end   = peak_idx + post
    if start < 0 or end > len(df):
        return None
    return df.iloc[start:end].reset_index(drop=True)


# ── Clip extraction ────────────────────────────────────────────────────────────

def extract_clip(video_path: Path, video_time_s: float, out_path: Path,
                 pre_s: float = 0.6, duration: float = 2.5) -> bool:
    start = max(0.0, video_time_s - pre_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.1f}",
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", "-b:a", "64k",
        "-loglevel", "error",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


# ── Main: cluster mode ─────────────────────────────────────────────────────────

def run_cluster(args):
    classified_path = Path(args.classified)
    session_csv_path = Path(args.session_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classified CSV
    clf_df = pd.read_csv(classified_path)

    # Auto-mark very low confidence spikes as unknown (almost certainly not real strokes)
    noise_mask = clf_df["conf1"] < args.noise_conf
    noise_df = clf_df[noise_mask].copy()
    if len(noise_df) > 0:
        noise_path = output_dir / "noise_spikes.csv"
        noise_df.to_csv(noise_path, index=False)
        print(f"Auto-marking {len(noise_df)} spikes as unknown (conf1 < {args.noise_conf} — likely movement noise)")

    # Cluster only spikes in the uncertain band
    to_cluster = clf_df[
        (clf_df["conf1"] >= args.noise_conf) & (clf_df["conf1"] < args.max_conf)
    ].copy().reset_index(drop=True)
    print(f"Spikes to cluster: {len(to_cluster)} (conf1 {args.noise_conf}–{args.max_conf})")

    if len(to_cluster) == 0:
        print("Nothing to cluster — all spikes are above the confidence threshold.")
        return

    # Load raw IMU CSV
    imu_df = pd.read_csv(session_csv_path)

    # Extract features for each low-confidence spike
    print("Extracting IMU features...")
    features, valid_indices = [], []
    for i, row in to_cluster.iterrows():
        peak_idx = find_peak_index(imu_df, row["timestamp_s"])
        window = extract_window(imu_df, peak_idx)
        if window is None:
            continue
        feats = extract_features(window)
        if np.isfinite(feats).all():
            features.append(feats)
            valid_indices.append(i)

    if not features:
        print("No valid windows extracted.")
        return

    X = np.array(features)
    to_cluster = to_cluster.iloc[[to_cluster.index.get_loc(i) for i in valid_indices]].reset_index(drop=True)
    print(f"  Valid windows: {len(X)}")

    # K-means clustering
    n_clusters = min(args.n_clusters, len(X))
    print(f"Clustering into {n_clusters} groups...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    to_cluster["cluster"] = labels

    # Save all clustered spikes so --apply can look up cluster members
    clustered_path = output_dir / "clustered_spikes.csv"
    to_cluster.to_csv(clustered_path, index=False)

    # Find representative (closest to centroid) per cluster
    reps = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_X = X_scaled[mask]
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        local_rep = np.argmin(dists)
        global_rep = np.where(mask)[0][local_rep]
        reps.append(global_rep)

    # Load video timestamps from review.csv if provided
    video_time_map = {}
    if args.review_csv:
        rev = pd.read_csv(args.review_csv)
        for _, r in rev.iterrows():
            video_time_map[round(r["timestamp_s"], 2)] = r["video_time_s"]

    # Build cluster summary CSV
    rows = []
    for c, rep_idx in enumerate(reps):
        rep_row = to_cluster.iloc[rep_idx]
        cluster_members = to_cluster[to_cluster["cluster"] == c]
        ts = rep_row["timestamp_s"]
        video_ts = video_time_map.get(round(ts, 2), ts + (args.sync_offset or 0))
        rows.append({
            "cluster": c,
            "n_members": len(cluster_members),
            "rep_timestamp_s": round(ts, 3),
            "rep_video_time_s": round(video_ts, 3),
            "rep_model_guess": rep_row["predicted_stroke"],
            "rep_conf": round(rep_row["conf1"], 3),
            "label": "",           # ← user fills this in
            "notes": "",
        })

    summary_df = pd.DataFrame(rows).sort_values("n_members", ascending=False).reset_index(drop=True)
    summary_path = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCluster summary: {summary_path}")
    print(f"  Fill in the 'label' column after watching each representative clip.\n")

    # Print preview table
    print(f"{'Cluster':>7}  {'Members':>7}  {'Model guess':>15}  {'Conf':>5}  {'Video time':>10}")
    print("-" * 55)
    for _, r in summary_df.iterrows():
        print(f"  {int(r['cluster']):>5}  {int(r['n_members']):>7}  "
              f"{r['rep_model_guess']:>15}  {r['rep_conf']:>5.2f}  {r['rep_video_time_s']:>10.1f}s")

    # Extract representative clips
    if args.video:
        video_path = Path(args.video)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        print(f"\nExtracting {n_clusters} representative clips...")
        for _, r in summary_df.iterrows():
            c = int(r["cluster"])
            fname = clips_dir / f"cluster_{c:03d}_{r['rep_model_guess']}_conf{r['rep_conf']:.2f}.mp4"
            ok = extract_clip(video_path, r["rep_video_time_s"], fname)
            status = "✓" if ok else "✗"
            print(f"  {status} cluster {c:03d} ({int(r['n_members'])} members) → {fname.name}")
        print(f"\nOpen {clips_dir} in Finder to watch clips.")
    else:
        print("\nTip: re-run with --video to extract representative clips automatically.")

    print(f"\nNext: fill 'label' in {summary_path}, then run:")
    print(f"  python3 scripts/cluster_unknowns.py --apply \\")
    print(f"    --cluster-summary {summary_path} \\")
    print(f"    --session-json <path/to/session.json>")


# ── Main: apply mode ──────────────────────────────────────────────────────────

def run_apply(args):
    summary_df = pd.read_csv(args.cluster_summary)
    json_path   = Path(args.session_json)

    with open(json_path) as f:
        session = json.load(f)

    # Accept any non-empty label, including explicit "unknown" (= confirmed non-stroke)
    # Empty label = not reviewed yet = leave marker unchanged
    labeled = summary_df[summary_df["label"].notna() & (summary_df["label"].str.strip() != "")]
    unknown_clusters = labeled[labeled["label"].str.strip() == "unknown"]
    stroke_clusters  = labeled[labeled["label"].str.strip() != "unknown"]
    print(f"Applying {len(labeled)}/{len(summary_df)} labeled clusters:")
    print(f"  {len(stroke_clusters)} stroke clusters → update stroke type")
    print(f"  {len(unknown_clusters)} non-stroke clusters → keep as unknown (excluded from training)")

    # Build timestamp → label map from clustered_spikes.csv
    classified_path = Path(args.cluster_summary).parent / "clustered_spikes.csv"
    if not classified_path.exists():
        print(f"ERROR: need {classified_path} — re-run without --apply first.")
        return

    spike_df = pd.read_csv(classified_path)
    label_map = {}
    for _, row in labeled.iterrows():
        members = spike_df[spike_df["cluster"] == row["cluster"]]
        for _, m in members.iterrows():
            label_map[round(m["timestamp_s"], 3)] = row["label"].strip()

    # Also apply auto-noise markers (conf < noise_conf, saved alongside cluster output)
    noise_path = Path(args.cluster_summary).parent / "noise_spikes.csv"
    if noise_path.exists():
        noise_df = pd.read_csv(noise_path)
        for _, m in noise_df.iterrows():
            label_map[round(m["timestamp_s"], 3)] = "unknown"
        print(f"  {len(noise_df)} auto-noise spikes → unknown")

    # Update JSON markers
    updated_strokes, kept_unknown = 0, 0
    for marker in session["markers"]:
        ts = round(float(marker["timestamp"]), 3)
        if ts in label_map:
            lbl = label_map[ts]
            if lbl == "unknown":
                kept_unknown += 1
            else:
                marker["strokeType"] = lbl
                updated_strokes += 1

    with open(json_path, "w") as f:
        json.dump(session, f, indent=2)

    print(f"\nResult: {updated_strokes} stroke labels applied, {kept_unknown} confirmed as non-strokes")
    print(f"Re-run extract_windows.py and retrain.py to incorporate these labels.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Apply labels from cluster_summary.csv back to session JSON")

    # Cluster mode args
    parser.add_argument("--classified",   help="Path to _classified.csv")
    parser.add_argument("--session-csv",  help="Path to raw session IMU CSV")
    parser.add_argument("--video",        help="Path to session MOV (for clip extraction)")
    parser.add_argument("--review-csv",   help="Path to review.csv from video_label_assist.py")
    parser.add_argument("--output-dir",   default="cluster_review")
    parser.add_argument("--n-clusters",   type=int, default=25)
    parser.add_argument("--max-conf",     type=float, default=0.45,
                        help="Only cluster spikes below this confidence (default 0.45)")
    parser.add_argument("--noise-conf",   type=float, default=0.15,
                        help="Spikes below this are auto-marked unknown/non-stroke (default 0.15)")
    parser.add_argument("--sync-offset",  type=float, default=None,
                        help="IMU→video offset in seconds (used if review.csv not available)")

    # Apply mode args
    parser.add_argument("--cluster-summary", help="Path to cluster_summary.csv")
    parser.add_argument("--session-json",    help="Path to session JSON to update")

    args = parser.parse_args()

    if args.apply:
        run_apply(args)
    else:
        if not args.classified or not args.session_csv:
            parser.error("--classified and --session-csv are required")
        run_cluster(args)


if __name__ == "__main__":
    main()
