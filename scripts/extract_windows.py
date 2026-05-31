#!/usr/bin/env python3
"""
extract_windows.py — PadelLabs ML Toolchain
============================================
Reads raw session CSV + JSON sidecar files exported from the PadelLabs iPhone app
and extracts per-stroke 100-sample windows (300ms pre-peak + 700ms post-peak at 100 Hz).

Burst grouping: multiple IMU markers within 1.2s of each other are treated as a
single physical stroke. The first non-unknown label in the burst wins. The window
is anchored on the first peak in the burst. `burst_size` is saved as metadata.

Output folder structure (Create ML Activity Classifier format):
    labeled-strokes/
      train/
        smash/       smash_001.csv, smash_002.csv …
        forehand/    …
        unknown/     …
      validation/    (--val-players split, player-independent)
      test/          (--test-players split, player-independent)

Each output CSV has exactly 100 rows and these columns:
    timestamp, accelX, accelY, accelZ, gyroX, gyroY, gyroZ, roll, pitch, yaw

A companion metadata CSV (labeled-strokes/metadata.csv) tracks every saved window:
    session_id, player_id, split, stroke_type, peak_ts, burst_size, window_file

Usage:
    python3 extract_windows.py \\
        --sessions  ../../DataCollection \\
        --output    ../../labeled-strokes \\
        [--val-players  player_uuid_1] \\
        [--test-players player_uuid_2] \\
        [--min-samples 80] \\
        [--burst-gap 1.2] \\
        [--dry-run]

Rules:
    - Split is player-independent (never by stroke).
    - Burst grouping: markers within --burst-gap seconds (default 1.2s) → one window.
    - A window is REJECTED if:
        * it has fewer than --min-samples samples (default 80)
        * any channel contains NaN / Inf
        * resolved stroke type is "unknown" or empty (unknown windows extracted separately)
    - Windows for label "unknown" are extracted from the full session data
      at positions WITHOUT a marker (background windows every 2 seconds).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 100          # Hz
PRE_PEAK_SAMPLES = 30      # 300 ms before spike
POST_PEAK_SAMPLES = 70     # 700 ms after spike
WINDOW_SIZE = PRE_PEAK_SAMPLES + POST_PEAK_SAMPLES  # 100

FEATURE_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]
OUTPUT_COLS  = ["timestamp"] + FEATURE_COLS

UNKNOWN_WINDOW_STRIDE = 2.0   # seconds between background (unknown) windows
UNKNOWN_GUARD_SECONDS = 1.5   # skip windows within this distance of any marker

# All stroke types as they appear in the JSON (snake_case)
VALID_STROKE_TYPES = [
    "smash",
    "vibora",
    "bandeja",
    "forehand",
    "backhand",
    "forehand_lob",
    "backhand_lob",
    "forehand_volley",
    "backhand_volley",
    "unknown",
]

DEFAULT_BURST_GAP = 1.2   # seconds — markers closer than this = same physical stroke


# ── Burst grouping ────────────────────────────────────────────────────────────

def group_markers_into_bursts(markers: list[dict], burst_gap: float) -> list[dict]:
    """
    Collapse markers within `burst_gap` seconds of each other into one representative.

    Strategy:
      - Sort by timestamp.
      - Any marker within burst_gap of the previous one is part of the same burst.
      - Winner = first non-unknown label in the burst (falls back to 'unknown').
      - Anchor timestamp = first peak in the burst.
      - burst_size = number of raw markers in the burst.

    Returns a list of representative marker dicts with keys:
        timestamp, strokeType, burst_size
    """
    if not markers:
        return []

    # Only work with markers that have a valid timestamp
    valid = [m for m in markers if float(m.get("timestamp", -1)) >= 0]
    valid.sort(key=lambda m: float(m["timestamp"]))

    bursts = []
    current_burst = []

    for marker in valid:
        ts = float(marker["timestamp"])
        if not current_burst:
            current_burst = [marker]
        elif ts - float(current_burst[0]["timestamp"]) <= burst_gap:
            current_burst.append(marker)
        else:
            bursts.append(current_burst)
            current_burst = [marker]

    if current_burst:
        bursts.append(current_burst)

    representatives = []
    for burst in bursts:
        # Anchor = first peak in burst
        anchor_ts = float(burst[0]["timestamp"])
        burst_size = len(burst)

        # Pick first non-unknown label
        label = "unknown"
        for m in burst:
            st = str(m.get("strokeType", "")).strip()
            if st and st != "unknown":
                label = st
                break

        representatives.append({
            "timestamp": anchor_ts,
            "strokeType": label,
            "burst_size": burst_size,
        })

    return representatives


def dedup_preparation_bursts(
    bursts: list[dict],
    imu_df: pd.DataFrame,
    slow_window_s: float = 3.5,
    weak_ratio: float = 0.40,
) -> list[dict]:
    """
    Second-pass deduplication for slow strokes where the backswing triggers a
    separate spike 1–3 seconds before the actual ball contact.

    Rule: if burst[i] and burst[i+1] are within slow_window_s seconds AND
    burst[i]'s accelMag peak < weak_ratio × burst[i+1]'s accelMag peak,
    then burst[i] is the preparation movement → drop it.

    Uses the raw IMU CSV to look up actual peak magnitudes.
    """
    if len(bursts) <= 1:
        return bursts

    def peak_mag(ts: float) -> float:
        idx = int((imu_df["timestamp"] - ts).abs().idxmin())
        search = imu_df.iloc[max(0, idx - 30): idx + 50]
        mag = np.sqrt(search.accelX**2 + search.accelY**2 + search.accelZ**2)
        return float(mag.max())

    kept = []
    skip_next = False
    for i in range(len(bursts)):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(bursts):
            gap = float(bursts[i + 1]["timestamp"]) - float(bursts[i]["timestamp"])
            if gap <= slow_window_s:
                mag_i  = peak_mag(float(bursts[i]["timestamp"]))
                mag_j  = peak_mag(float(bursts[i + 1]["timestamp"]))
                if mag_i < weak_ratio * mag_j:
                    # bursts[i] is a preparation — prefer bursts[i+1]'s label if known
                    if bursts[i]["strokeType"] != "unknown" and bursts[i + 1]["strokeType"] == "unknown":
                        bursts[i + 1]["strokeType"] = bursts[i]["strokeType"]
                    skip_next = True  # drop bursts[i], keep bursts[i+1]
                    continue
        kept.append(bursts[i])
    if not skip_next:
        kept.append(bursts[-1])
    elif bursts:
        kept.append(bursts[-1])
    return kept


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_session(csv_path: Path, json_path: Path):
    """Return (df_raw, metadata_dict, markers_list) or raise."""
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        meta = json.load(f)
    markers = meta.get("markers", [])
    return df, meta, markers


def player_id_from_meta(meta: dict) -> str:
    return str(meta.get("playerId", "unknown"))


def session_id_from_meta(meta: dict) -> str:
    return str(meta.get("sessionId", "unknown"))


def extract_window(df: pd.DataFrame, peak_idx: int) -> pd.DataFrame | None:
    """Extract 100-sample window centred on peak_idx (30 pre + 70 post)."""
    start = peak_idx - PRE_PEAK_SAMPLES
    end   = peak_idx + POST_PEAK_SAMPLES   # exclusive

    if start < 0 or end > len(df):
        return None

    window = df.iloc[start:end][OUTPUT_COLS].copy().reset_index(drop=True)

    if len(window) != WINDOW_SIZE:
        return None

    # Reject windows with NaN / Inf
    if not np.isfinite(window[FEATURE_COLS].values).all():
        return None

    return window


def find_peak_index(df: pd.DataFrame, marker_ts: float) -> int | None:
    """Find the DataFrame row index closest to marker_ts, then re-anchor on the
    actual accelMag maximum within ±20 samples.  The spike detector fires slightly
    late, so the raw marker timestamp rarely lands on the true IMU peak — without
    this correction the peak sits at ~sample 50 instead of the intended sample 30."""
    if "timestamp" not in df.columns:
        return None
    coarse = int((df["timestamp"] - marker_ts).abs().idxmin())

    # Search ±20 samples for the true acceleration-magnitude peak
    search_start = max(0, coarse - 20)
    search_end   = min(len(df), coarse + 21)
    mag = np.sqrt(
        df["accelX"].iloc[search_start:search_end] ** 2 +
        df["accelY"].iloc[search_start:search_end] ** 2 +
        df["accelZ"].iloc[search_start:search_end] ** 2
    )
    return search_start + int(mag.values.argmax())


def extract_unknown_windows(df: pd.DataFrame, marker_timestamps: list[float]) -> list[pd.DataFrame]:
    """
    Extract background (unknown) windows from positions without nearby markers.
    Walks the session every UNKNOWN_WINDOW_STRIDE seconds.
    """
    if df.empty or "timestamp" not in df.columns:
        return []

    ts_start = df["timestamp"].iloc[0]
    ts_end   = df["timestamp"].iloc[-1]
    marker_ts_arr = np.array(marker_timestamps) if marker_timestamps else np.array([])

    windows = []
    t = ts_start + 1.0  # skip first second

    while t + (POST_PEAK_SAMPLES / SAMPLE_RATE) <= ts_end:
        # Skip if too close to any labeled marker
        if len(marker_ts_arr) > 0:
            dist_to_nearest = np.min(np.abs(marker_ts_arr - t))
            if dist_to_nearest < UNKNOWN_GUARD_SECONDS:
                t += UNKNOWN_WINDOW_STRIDE
                continue

        peak_idx = find_peak_index(df, t)
        if peak_idx is not None:
            w = extract_window(df, peak_idx)
            if w is not None:
                windows.append(w)

        t += UNKNOWN_WINDOW_STRIDE

    return windows


def assign_split(player_id: str, val_players: set, test_players: set) -> str:
    if player_id in test_players:
        return "test"
    if player_id in val_players:
        return "validation"
    return "train"


def save_window(window: pd.DataFrame, output_root: Path, split: str, stroke_type: str, idx: int) -> Path:
    stroke_dir = output_root / split / stroke_type
    stroke_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stroke_type}_{idx:05d}.csv"
    out_path = stroke_dir / filename
    window.to_csv(out_path, index=False)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract per-stroke windows from raw PadelLabs sessions.")
    parser.add_argument("--sessions",     required=True,  help="Root folder containing session CSV+JSON files")
    parser.add_argument("--output",       required=True,  help="Output root (labeled-strokes/)")
    parser.add_argument("--val-players",  nargs="*", default=[], help="Player IDs for validation split")
    parser.add_argument("--test-players", nargs="*", default=[], help="Player IDs for test split (hold-out)")
    parser.add_argument("--min-samples",  type=int, default=80, help="Minimum non-NaN samples in window (default 80)")
    parser.add_argument("--burst-gap",    type=float, default=DEFAULT_BURST_GAP,
                        help=f"Seconds within which markers are grouped as one burst (default {DEFAULT_BURST_GAP})")
    parser.add_argument("--no-unknown",   action="store_true", help="Skip automatic unknown window extraction")
    parser.add_argument("--dry-run",      action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    sessions_root = Path(args.sessions)
    output_root   = Path(args.output)
    val_players   = set(args.val_players)
    test_players  = set(args.test_players)

    overlap = val_players & test_players
    if overlap:
        print(f"ERROR: Players in both val and test splits: {overlap}", file=sys.stderr)
        sys.exit(1)

    # ── Scan sessions ─────────────────────────────────────────────────────────

    csv_files = sorted(sessions_root.rglob("*.csv"))

    counters   = {st: {"train": 0, "validation": 0, "test": 0} for st in VALID_STROKE_TYPES}
    rejected   = 0
    total_sessions = 0
    burst_stats = {"total_raw_markers": 0, "total_bursts": 0, "multi_marker_bursts": 0}

    # Metadata log for every saved window
    metadata_rows = []

    for csv_path in csv_files:
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  SKIP (no sidecar JSON): {csv_path.name}")
            continue

        try:
            df, meta, raw_markers = load_session(csv_path, json_path)
        except Exception as e:
            print(f"  ERROR loading {csv_path.name}: {e}")
            continue

        total_sessions += 1
        player_id  = player_id_from_meta(meta)
        session_id = session_id_from_meta(meta)
        split      = assign_split(player_id, val_players, test_players)

        # ── Burst grouping + preparation dedup ───────────────────────────────
        bursts = group_markers_into_bursts(raw_markers, args.burst_gap)
        before_dedup = len(bursts)
        bursts = dedup_preparation_bursts(bursts, df)
        dropped = before_dedup - len(bursts)
        if dropped:
            print(f"  Dedup: removed {dropped} preparation-spike bursts")
        multi  = sum(1 for b in bursts if b["burst_size"] > 1)

        burst_stats["total_raw_markers"]   += len(raw_markers)
        burst_stats["total_bursts"]        += len(bursts)
        burst_stats["multi_marker_bursts"] += multi

        print(
            f"\n[{split.upper()}] {csv_path.name}  player={player_id[:8]}…  "
            f"raw_markers={len(raw_markers)}  bursts={len(bursts)}  multi={multi}"
        )

        # Ensure required columns exist
        missing = [c for c in OUTPUT_COLS if c not in df.columns]
        if missing:
            print(f"  SKIP: missing columns {missing}")
            continue

        # ── Labeled stroke windows ─────────────────────────────────────────────
        labeled_timestamps = []   # used to keep unknown windows away from labeled ones

        for burst in bursts:
            stroke_type = str(burst.get("strokeType", "")).strip()
            ts          = float(burst["timestamp"])
            burst_size  = int(burst["burst_size"])

            # Skip unknown — extracted separately below
            if stroke_type == "unknown" or stroke_type not in VALID_STROKE_TYPES:
                labeled_timestamps.append(ts)  # still guard unknown windows from this position
                continue
            if ts < 0:
                continue

            labeled_timestamps.append(ts)
            peak_idx = find_peak_index(df, ts)

            if peak_idx is None:
                rejected += 1
                continue

            window = extract_window(df, peak_idx)
            if window is None or len(window) < args.min_samples:
                rejected += 1
                print(f"  REJECT {stroke_type} @{ts:.2f}s (burst_size={burst_size})")
                continue

            idx = sum(counters[stroke_type].values())
            if not args.dry_run:
                out_path = save_window(window, output_root, split, stroke_type, idx)
                metadata_rows.append({
                    "session_id": session_id,
                    "player_id": player_id,
                    "split": split,
                    "stroke_type": stroke_type,
                    "peak_ts": round(ts, 4),
                    "burst_size": burst_size,
                    "window_file": str(out_path.relative_to(output_root)),
                })

            counters[stroke_type][split] += 1
            print(f"  ✓ {stroke_type:<22} @{ts:.2f}s  burst={burst_size}  → {split}")

        # ── Unknown (background) windows ──────────────────────────────────────
        if not args.no_unknown:
            unknown_windows = extract_unknown_windows(df, labeled_timestamps)
            for w in unknown_windows:
                idx = sum(counters["unknown"].values())
                if not args.dry_run:
                    out_path = save_window(w, output_root, split, "unknown", idx)
                    # Use midpoint timestamp as peak_ts for background windows
                    mid_ts = round(float(w["timestamp"].iloc[WINDOW_SIZE // 2]), 4)
                    metadata_rows.append({
                        "session_id": session_id,
                        "player_id": player_id,
                        "split": split,
                        "stroke_type": "unknown",
                        "peak_ts": mid_ts,
                        "burst_size": 0,
                        "window_file": str(out_path.relative_to(output_root)),
                    })
                counters["unknown"][split] += 1

            if unknown_windows:
                print(f"  ✓ unknown (background) ×{len(unknown_windows)}  → {split}")

    # ── Write metadata CSV ────────────────────────────────────────────────────
    if not args.dry_run and metadata_rows:
        meta_path = output_root / "metadata.csv"
        pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)
        print(f"\nMetadata written to: {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"Sessions processed:       {total_sessions}")
    print(f"Windows rejected:         {rejected}")
    print(f"Raw markers total:        {burst_stats['total_raw_markers']}")
    print(f"Grouped bursts total:     {burst_stats['total_bursts']}")
    print(f"Multi-marker bursts:      {burst_stats['multi_marker_bursts']}")
    if burst_stats["total_raw_markers"] > 0:
        reduction = 1 - burst_stats["total_bursts"] / burst_stats["total_raw_markers"]
        print(f"Burst reduction:          {reduction:.0%} fewer windows than raw markers")

    print(f"\n{'Stroke Type':<24} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>8}")
    print(f"{'-'*24} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    grand_total = 0
    for stroke in VALID_STROKE_TYPES:
        tr  = counters[stroke]["train"]
        val = counters[stroke]["validation"]
        te  = counters[stroke]["test"]
        tot = tr + val + te
        grand_total += tot
        if stroke == "unknown":
            status = "  "
        elif tot < 50:
            status = "⚠️ "
        elif tot >= 200:
            status = "✓ "
        else:
            status = "  "
        print(f"  {status}{stroke:<22} {tr:>7} {val:>7} {te:>7} {tot:>8}")

    print(f"{'-'*60}")
    print(f"  {'TOTAL':<24} {sum(c['train'] for c in counters.values()):>7} "
          f"{sum(c['validation'] for c in counters.values()):>7} "
          f"{sum(c['test'] for c in counters.values()):>7} "
          f"{grand_total:>8}")

    # Warn on classes with too few samples
    labeled_types = [s for s in VALID_STROKE_TYPES if s != "unknown"]
    low_classes = [s for s in labeled_types if sum(counters[s].values()) < 50]
    if low_classes:
        print(f"\n⚠️  Low-sample classes (< 50): {', '.join(low_classes)}")
        print("   Run targeted top-up sessions for these classes before training.")

    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print(f"\nOutput written to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
