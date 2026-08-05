#!/usr/bin/env python3
"""
extract_windows.py — PadelLabs ML Toolchain
============================================
Reads raw session CSV + JSON sidecar files exported from the PadelLabs iPhone app
and extracts per-stroke 100-sample windows (300ms pre-peak + 700ms post-peak at 100 Hz).

Burst grouping: multiple IMU markers within 1.2s of each other are treated as a
single physical stroke. The first non-unknown label in the burst wins. The window
is anchored on the strongest accelMag peak inside the burst span (the ball contact),
NOT on the marker timestamps — markers are stamped when classification finishes, so
they lag the contact by ~0.7-1.0s. `burst_size` is saved as metadata.

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

# Peak search span around a burst, in seconds.
#
# Markers LAG the ball contact. A marker is stamped when classification finishes —
# after the detector's 700ms post-peak buffer plus CoreML inference — so it lands
# ~0.7-1.0s AFTER the contact it describes, and the lag varies with inference time.
# (Measured: median accelMag at marker timestamps is only ~1.1g, i.e. the swing is
# already over by then.) The search therefore has to reach well BACK from the first
# marker, not forward from it.
#
# Sweeping PEAK_SEARCH_PRE on a real session, mean accelMag at the resolved anchor:
#   0.3s → 3.80g   0.5s → 5.18g   0.7s → 8.19g   0.9s → 8.34g   1.5s → 8.37g
# It plateaus at 0.9s, matching the expected lag. Going wider only risks reaching
# into the previous stroke.
PEAK_SEARCH_PRE  = 0.9        # seconds before the first marker in the burst
PEAK_SEARCH_POST = 0.7        # seconds after the last marker in the burst
FALLBACK_PEAK_SEARCH = 0.2    # ± seconds when no burst span is available

# Two bursts whose resolved contact peaks are closer than this are the same physical
# stroke — burst grouping is anchored on the FIRST marker, so a long preparation ramp
# can push the tail of one swing past the burst_gap boundary and open a second burst
# that then resolves onto the very same contact peak. Must match the watch-side
# minimumSpikeSeparation in MotionService.swift.
MIN_PEAK_SEPARATION = 0.6     # seconds

# ── Marker format ─────────────────────────────────────────────────────────────
# Two generations of recordings exist and they need different handling:
#
#   "crossing"  (watch builds before f06e54c) — the detector fired on every
#               threshold crossing, so ONE swing produced several markers spread
#               over ~1.2s, and each was stamped when classification finished,
#               ~0.7-1.0s after the contact. These need burst-grouping, the
#               preparation-spike dedup, and a wide backward peak search.
#
#   "peak"      (f06e54c onward) — the peak-picker emits ONE marker per swing,
#               stamped at the contact itself. Burst-grouping this format is
#               actively destructive: two markers 1.2s apart are two different
#               strokes, and grouping them deletes one. Measured on a real
#               session, the 1.2s grouping merged 56 of 627 markers and the
#               preparation dedup dropped 17 more — a 12% loss of real strokes.
#
# The formats are trivially separable: the peak-picker enforces a hard
# MIN_PEAK_SEPARATION refractory, so a "peak" session has essentially no marker
# gaps below it, while a "crossing" session is full of them (~40%).
PEAK_FORMAT_MAX_SUBGAP_FRACTION = 0.02
PEAK_FORMAT_MIN_MARKERS = 20   # too few markers to judge → assume legacy
PEAK_STAMPED_SEARCH_PRE = 0.2  # the marker IS the contact; only nudge for jitter


def markers_are_peak_stamped(markers: list[dict]) -> bool:
    """True when markers came from the peak-picking detector (one per swing)."""
    stamps = sorted(
        float(m["timestamp"]) for m in markers if float(m.get("timestamp", -1)) >= 0
    )
    if len(stamps) < PEAK_FORMAT_MIN_MARKERS:
        return False
    gaps = np.diff(stamps)
    return float(np.mean(gaps < MIN_PEAK_SEPARATION)) <= PEAK_FORMAT_MAX_SUBGAP_FRACTION

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
      - Anchor timestamp = first marker (kept only as a coarse locator; the true
        window anchor is resolved later by find_peak_index over [span_start, span_end]).
      - burst_size = number of raw markers in the burst.

    Returns a list of representative marker dicts with keys:
        timestamp, strokeType, burst_size, span_start, span_end
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
        span_start = float(burst[0]["timestamp"])
        span_end   = float(burst[-1]["timestamp"])
        burst_size = len(burst)

        # Pick first non-unknown label
        label = "unknown"
        for m in burst:
            st = str(m.get("strokeType", "")).strip()
            if st and st != "unknown":
                label = st
                break

        representatives.append({
            "timestamp": span_start,
            "strokeType": label,
            "burst_size": burst_size,
            "span_start": span_start,
            "span_end": span_end,
            # Timestamps of the markers carrying this burst's label — used to break
            # label ties when two bursts merge onto the same contact peak.
            "label_ts": [
                float(m["timestamp"]) for m in burst
                if str(m.get("strokeType", "")).strip() == label
            ],
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

    A burst carrying a human label that disagrees with its neighbour's is never dropped:
    a reviewer labelled two strokes there, so neither is a backswing.

    Uses the raw IMU CSV to look up actual peak magnitudes over each burst's span.
    """
    if len(bursts) <= 1:
        return bursts

    def peak_mag(burst: dict) -> float:
        idx = burst.get("peak_idx")
        if idx is None:
            idx = find_peak_index(
                imu_df,
                float(burst["timestamp"]),
                burst.get("span_start"),
                burst.get("span_end"),
            )
        if idx is None:
            return 0.0
        return float(_accel_mag_slice(imu_df, idx, idx + 1)[0])

    def anchor(burst: dict) -> float:
        return float(burst.get("peak_ts", burst["timestamp"]))

    def labels_conflict(a: dict, b: dict) -> bool:
        la, lb = a["strokeType"], b["strokeType"]
        return la != "unknown" and lb != "unknown" and la != lb

    kept = []
    for i, burst in enumerate(bursts):
        if i + 1 < len(bursts):
            gap = anchor(bursts[i + 1]) - anchor(burst)
            if (gap <= slow_window_s
                    and not labels_conflict(burst, bursts[i + 1])
                    and peak_mag(burst) < weak_ratio * peak_mag(bursts[i + 1])):
                # `burst` is a preparation movement — prefer the next burst's window,
                # but carry this burst's label over if the next one is unlabeled.
                if burst["strokeType"] != "unknown" and bursts[i + 1]["strokeType"] == "unknown":
                    bursts[i + 1]["strokeType"] = burst["strokeType"]
                continue          # drop `burst`, keep bursts[i + 1]
        kept.append(burst)
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


def _accel_mag_slice(df: pd.DataFrame, start: int, end: int) -> np.ndarray:
    """Acceleration magnitude over df row range [start, end)."""
    return np.sqrt(
        df["accelX"].iloc[start:end].values ** 2 +
        df["accelY"].iloc[start:end].values ** 2 +
        df["accelZ"].iloc[start:end].values ** 2
    )


def peak_index_in_span(df: pd.DataFrame, t_lo: float, t_hi: float) -> int | None:
    """Row index of the accelMag maximum inside the time range [t_lo, t_hi]."""
    if "timestamp" not in df.columns or df.empty:
        return None
    start = int((df["timestamp"] - t_lo).abs().idxmin())
    end   = int((df["timestamp"] - t_hi).abs().idxmin()) + 1
    start = max(0, start)
    end   = min(len(df), max(end, start + 1))
    mag = _accel_mag_slice(df, start, end)
    if mag.size == 0:
        return None
    return start + int(mag.argmax())


def find_peak_index(
    df: pd.DataFrame,
    marker_ts: float,
    span_start: float | None = None,
    span_end: float | None = None,
    search_pre: float | None = None,
) -> int | None:
    """Resolve the row index of the ball contact for a burst.

    The watch spike detector arms on the FIRST sample over threshold, which in padel
    is the backswing — the real contact peak follows 300-900ms later. So the search
    runs over the whole burst span plus the follow-through tail, not a fixed window
    around a single marker.

    When no span is given (background/unknown windows) it falls back to a symmetric
    ±FALLBACK_PEAK_SEARCH search around marker_ts.
    """
    if "timestamp" not in df.columns:
        return None

    if span_start is None or span_end is None:
        t_lo = marker_ts - FALLBACK_PEAK_SEARCH
        t_hi = marker_ts + FALLBACK_PEAK_SEARCH
    else:
        pre = PEAK_SEARCH_PRE if search_pre is None else search_pre
        t_lo = float(span_start) - pre
        t_hi = float(span_end) + PEAK_SEARCH_POST

    return peak_index_in_span(df, t_lo, t_hi)


def resolve_burst_peaks(
    bursts: list[dict],
    df: pd.DataFrame,
    min_separation: float = MIN_PEAK_SEPARATION,
    search_pre: float | None = None,
) -> list[dict]:
    """Resolve every burst onto its ball-contact peak, then merge bursts that land
    on the same contact.

    Burst grouping cuts on the FIRST marker of a burst, so a swing with a long
    preparation ramp can spill past burst_gap and open a second burst that resolves
    onto the identical peak — one physical stroke, two training windows. This pass
    collapses those. Each surviving burst gains `peak_idx` and `peak_ts`.

    Two bursts carrying DIFFERENT human labels are never merged. A reviewer who labeled
    them separately saw two strokes, and their verdict outranks this heuristic — merging
    would silently rewrite validated data.
    """
    resolved = []
    for b in bursts:
        idx = find_peak_index(df, float(b["timestamp"]), b.get("span_start"), b.get("span_end"), search_pre)
        if idx is None:
            continue
        b = dict(b)
        b["peak_idx"] = idx
        b["peak_ts"]  = float(df["timestamp"].iloc[idx])
        resolved.append(b)

    resolved.sort(key=lambda b: b["peak_ts"])

    def label_distance(burst: dict, peak_ts: float) -> float:
        """How far the burst's labeled markers sit from a contact peak."""
        stamps = burst.get("label_ts") or []
        if not stamps:
            return float("inf")
        return min(abs(t - peak_ts) for t in stamps)

    def labelsConflict(a: dict, b: dict) -> bool:
        """Both sides carry a human label and the labels disagree."""
        la, lb = a["strokeType"], b["strokeType"]
        return la != "unknown" and lb != "unknown" and la != lb

    merged: list[dict] = []
    for b in resolved:
        if (merged and b["peak_ts"] - merged[-1]["peak_ts"] < min_separation
                and not labelsConflict(merged[-1], b)):
            prev = merged[-1]
            # Same physical stroke: keep the stronger peak and union the metadata.
            prev_mag = _accel_mag_slice(df, prev["peak_idx"], prev["peak_idx"] + 1)[0]
            this_mag = _accel_mag_slice(df, b["peak_idx"], b["peak_idx"] + 1)[0]
            if this_mag > prev_mag:
                prev["peak_idx"], prev["peak_ts"] = b["peak_idx"], b["peak_ts"]

            # Label goes to whichever burst put a labeled marker nearest the contact.
            # Preferring `prev` unconditionally mislabels a hard smash as the softer
            # stroke that happened to open the earlier burst.
            if prev["strokeType"] == "unknown":
                prev["strokeType"] = b["strokeType"]
                prev["label_ts"]   = b.get("label_ts") or []
            elif b["strokeType"] != "unknown":
                peak = prev["peak_ts"]
                if label_distance(b, peak) < label_distance(prev, peak):
                    prev["strokeType"] = b["strokeType"]
                    prev["label_ts"]   = b.get("label_ts") or []

            prev["burst_size"] += b["burst_size"]
            prev["span_start"] = min(prev["span_start"], b["span_start"])
            prev["span_end"]   = max(prev["span_end"], b["span_end"])
            continue
        merged.append(b)

    return merged


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

        # Ensure required columns exist — peak resolution below reads accel channels
        missing = [c for c in OUTPUT_COLS if c not in df.columns]
        if missing:
            print(f"  SKIP: missing columns {missing}")
            continue

        # ── Burst grouping → contact-peak resolution → preparation dedup ─────
        # Recordings from the peak-picking watch build already carry exactly one
        # marker per swing, stamped at the contact. Grouping or dedup-ing those
        # deletes real strokes, so both passes are skipped for that format.
        peak_stamped = markers_are_peak_stamped(raw_markers)
        burst_gap  = 0.0 if peak_stamped else args.burst_gap
        search_pre = PEAK_STAMPED_SEARCH_PRE if peak_stamped else None

        bursts = group_markers_into_bursts(raw_markers, burst_gap)
        before_merge = len(bursts)
        bursts = resolve_burst_peaks(bursts, df, search_pre=search_pre)
        collapsed = before_merge - len(bursts)
        if collapsed:
            print(f"  Merge: collapsed {collapsed} bursts sharing one contact peak")

        if not peak_stamped:
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
            f"markers={'peak' if peak_stamped else 'crossing'}-stamped  "
            f"raw_markers={len(raw_markers)}  bursts={len(bursts)}  multi={multi}"
        )

        # ── Labeled stroke windows ─────────────────────────────────────────────
        labeled_timestamps = []   # used to keep unknown windows away from labeled ones

        for burst in bursts:
            stroke_type = str(burst.get("strokeType", "")).strip()
            ts          = float(burst.get("peak_ts", burst["timestamp"]))
            burst_size  = int(burst["burst_size"])

            # Skip unknown — extracted separately below
            if stroke_type == "unknown" or stroke_type not in VALID_STROKE_TYPES:
                labeled_timestamps.append(ts)  # still guard unknown windows from this position
                continue
            if ts < 0:
                continue

            peak_idx = burst.get("peak_idx")
            if peak_idx is None:
                rejected += 1
                continue

            window = extract_window(df, peak_idx)
            if window is None or len(window) < args.min_samples:
                rejected += 1
                print(f"  REJECT {stroke_type} @{ts:.2f}s (burst_size={burst_size})")
                continue

            # Record the RESOLVED contact peak, not the marker timestamp — the two can
            # be up to ~1s apart and downstream tools re-locate windows by peak_ts.
            peak_ts = float(df["timestamp"].iloc[peak_idx])
            labeled_timestamps.append(peak_ts)

            idx = sum(counters[stroke_type].values())
            if not args.dry_run:
                out_path = save_window(window, output_root, split, stroke_type, idx)
                metadata_rows.append({
                    "session_id": session_id,
                    "player_id": player_id,
                    "split": split,
                    "stroke_type": stroke_type,
                    "peak_ts": round(peak_ts, 4),
                    "burst_size": burst_size,
                    "window_file": str(out_path.relative_to(output_root)),
                })

            counters[stroke_type][split] += 1
            print(f"  ✓ {stroke_type:<22} @{peak_ts:.2f}s  burst={burst_size}  → {split}")

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
