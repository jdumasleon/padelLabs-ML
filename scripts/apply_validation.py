#!/usr/bin/env python3
"""
apply_validation.py — PadelLabs ML Toolchain
=============================================
Applies the human verdicts from a `<sessionId>_validation.csv` (produced by the
web labeling tool in Stage 2) back onto the session JSON's `markers[].strokeType`,
so that `extract_windows.py` produces VALIDATION-QUALITY training windows instead
of the raw in-app classifier guesses.

Why this exists
---------------
The labeling tool operates on burst representatives (one row per physical stroke,
anchored on the first IMU marker of the burst). The session JSON stores the raw
markers (multiple per burst) with whatever label the on-watch classifier assigned
at record time. This script reconciles the two:

  For every burst (same 1.2s grouping extract_windows uses):
    - match the burst anchor timestamp to a validation row (exact match; the tool
      writes timestamp_s == burst anchor, verified at 100% / 0.0000s drift)
    - resolve the human truth label:
        verdict == correct    -> predicted_stroke (top1 the reviewer confirmed)
        verdict == wrong      -> corrected_label
        verdict == skip        -> unknown   (reviewer couldn't tell -> exclude)
        verdict == duplicate   -> unknown   (drop -> exclude)
        empty / no match       -> unknown   (never reviewed -> exclude)
    - write the resolved label onto EVERY raw marker in the burst, so the
      "first non-unknown label in the burst" winner logic in extract_windows is
      deterministic and equals the human truth.

Only human-confirmed strokes (correct + wrong) keep a real label; everything else
becomes `unknown` and is therefore excluded from the labeled training set.

The original JSON is backed up to `<sessionId>.json.orig` (once; never clobbered).

Usage
-----
    # single session
    python3 apply_validation.py --json "/path/to/<sessionId>.json"

    # whole tree (every *.json that has a sibling *_validation.csv)
    python3 apply_validation.py --sessions ../DataCollection

    # preview only
    python3 apply_validation.py --sessions ../DataCollection --dry-run
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

BURST_GAP = 1.2  # MUST match extract_windows.py
MATCH_TOL = 0.03  # seconds; anchors match validation rows exactly in practice

VALID_STROKE_TYPES = {
    "smash", "vibora", "bandeja", "forehand", "backhand",
    "forehand_lob", "backhand_lob", "forehand_volley", "backhand_volley",
}


def group_bursts(markers):
    """Member-preserving burst grouping (same algorithm as extract_windows)."""
    valid = [m for m in markers if float(m.get("timestamp", -1)) >= 0]
    valid.sort(key=lambda m: float(m["timestamp"]))
    bursts, cur = [], []
    for m in valid:
        ts = float(m["timestamp"])
        if not cur:
            cur = [m]
        elif ts - float(cur[0]["timestamp"]) <= BURST_GAP:
            cur.append(m)
        else:
            bursts.append(cur)
            cur = [m]
    if cur:
        bursts.append(cur)
    return bursts


def _clean(label):
    if label is None:
        return ""
    if isinstance(label, float) and math.isnan(label):
        return ""
    return str(label).strip()


def resolve_truth(row):
    """Human-truth label for one validation row, or 'unknown' to exclude."""
    verdict = _clean(row.get("verdict")).lower()
    if verdict == "correct":
        label = _clean(row.get("predicted_stroke")) or _clean(row.get("top1"))
    elif verdict == "wrong":
        label = _clean(row.get("corrected_label"))
    else:  # skip / duplicate / empty
        return "unknown"
    return label if label in VALID_STROKE_TYPES else "unknown"


def find_validation_csv(json_path: Path) -> Path | None:
    direct = json_path.with_name(json_path.stem + "_validation.csv")
    if direct.exists():
        return direct
    cands = list(json_path.parent.glob("*_validation.csv"))
    return cands[0] if len(cands) == 1 else None


def apply_one(json_path: Path, dry_run: bool) -> dict | None:
    vcsv = find_validation_csv(json_path)
    if vcsv is None:
        return None

    with open(json_path) as f:
        meta = json.load(f)
    markers = meta.get("markers", [])
    if not markers:
        print(f"  SKIP (no markers): {json_path.name}")
        return None

    vdf = pd.read_csv(vcsv)
    if "timestamp_s" not in vdf.columns or "verdict" not in vdf.columns:
        print(f"  SKIP (validation csv missing columns): {vcsv.name}")
        return None

    v_ts = vdf["timestamp_s"].astype(float).values
    bursts = group_bursts(markers)

    stats = {"correct": 0, "wrong": 0, "excluded": 0, "labeled_markers": 0, "bursts": len(bursts)}

    for burst in bursts:
        anchor = float(burst[0]["timestamp"])
        # nearest validation row
        diffs = [abs(t - anchor) for t in v_ts]
        j = int(min(range(len(diffs)), key=lambda i: diffs[i])) if diffs else -1
        if j < 0 or diffs[j] > MATCH_TOL:
            truth = "unknown"  # burst never presented to reviewer
        else:
            row = vdf.iloc[j]
            truth = resolve_truth(row)
            verdict = _clean(row.get("verdict")).lower()
            if truth != "unknown":
                stats["correct" if verdict == "correct" else "wrong"] += 1
            else:
                stats["excluded"] += 1
        for m in burst:
            m["strokeType"] = truth
        if truth != "unknown":
            stats["labeled_markers"] += len(burst)

    if not dry_run:
        orig = json_path.with_suffix(json_path.suffix + ".orig")
        if not orig.exists():
            orig.write_text(json.dumps(json.load(open(json_path))))  # snapshot raw
        with open(json_path, "w") as f:
            json.dump(meta, f)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Apply validation verdicts onto session JSON markers.")
    ap.add_argument("--json", help="Single session JSON to patch")
    ap.add_argument("--sessions", help="Root tree; patch every JSON with a sibling _validation.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.json and not args.sessions:
        ap.error("pass --json or --sessions")

    if args.json:
        paths = [Path(args.json)]
    else:
        paths = [
            Path(p) for p in glob.glob(os.path.join(args.sessions, "**", "*.json"), recursive=True)
            if "_validation" not in p and "_classified" not in p and not p.endswith(".orig")
        ]

    grand = {"correct": 0, "wrong": 0, "excluded": 0, "labeled_markers": 0, "sessions": 0}
    for jp in sorted(paths):
        st = apply_one(jp, args.dry_run)
        if st is None:
            continue
        grand["sessions"] += 1
        for k in ("correct", "wrong", "excluded", "labeled_markers"):
            grand[k] += st[k]
        folder = jp.parent.name if jp.parent.name else jp.stem
        print(f"  [{folder[:26]:26}] bursts={st['bursts']:4} "
              f"correct={st['correct']:4} wrong={st['wrong']:4} excluded={st['excluded']:4} "
              f"labeled_markers={st['labeled_markers']:5}")

    print("=" * 70)
    mode = "(dry-run, no writes)" if args.dry_run else "(JSONs patched; originals → *.json.orig)"
    print(f"Sessions patched: {grand['sessions']}  {mode}")
    print(f"Validated strokes kept: correct={grand['correct']} wrong={grand['wrong']} "
          f"(total {grand['correct'] + grand['wrong']})")
    print(f"Excluded (skip/dup/unreviewed) bursts: {grand['excluded']}")
    print(f"Labeled raw markers written: {grand['labeled_markers']}")


if __name__ == "__main__":
    main()
