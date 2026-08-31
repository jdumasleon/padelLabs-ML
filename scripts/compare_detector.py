#!/usr/bin/env python3
"""
compare_detector.py — PadelLabs ML Toolchain
=============================================
Compares recorded sessions to check whether a watch-side spike-detector change
behaved as intended. Run it on a session recorded with the new build alongside one
or more recorded with the old build.

    python3 scripts/compare_detector.py \\
        --baseline "../dataCollection/Jose/20 July (train for v6)/734605F7-….json" \\
        --new      "../dataCollection/Jose/<next match>/<sessionId>.json"

Paths may point at the .json, the .csv, or the common prefix of both.

WHAT TO READ, AND IN WHAT ORDER
-------------------------------
1. `marker lag`  — median (marker timestamp − resolved contact peak).
   Old ActiveWorkout recordings stamp the marker when CLASSIFICATION FINISHES, so
   this sits around +0.7…+1.0s. After the `detectedAt: spike.timestamp` fix it
   should collapse to roughly 0. This is the check that the fix shipped.

2. `of those UNMARKED` — THE decisive missed-stroke test, and the one to trust.
   It counts hard contacts (>6g) present in the raw IMU that carry no marker at
   all. Unlike every other row it needs no baseline and is immune to differences
   in player or match intensity: a hard contact the watch never marked is simply
   a lost stroke. Anything above ~0.05/min deserves investigation.

3. `strokes/min` — a cross-session recall sanity check, measured AFTER burst
   grouping and peak resolution so duplicates are collapsed on both sides. Useful
   but WEAK on its own: it moves with how hard and how often the players actually
   hit, so a quieter match or a different player shifts it. Read it together with
   `contacts>6g / min`, which measures the same thing about play intensity. If
   both drop together, that is the match being calmer, not the detector failing.

4. `markers/min` — expected to fall a lot, and that is GOOD, not a regression.
   The old detector emitted several markers per swing; the peak-picker emits one.
   Do not read a drop here as missed strokes; read `strokes/min` for that.

5. `gaps < 0.6s` and `0.3–0.4s band` — the duplicate signature. The old detector's
   0.3s refractory produced a dense band of 0.3–0.4s gaps. Both should go to ~0.

6. `median g at marker` — acceleration magnitude where the markers actually sit.
   Lagged markers land on the follow-through (~1g). Correct ones land on or near
   the contact (>2g).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent


def _load_pipeline():
    """Import extract_windows so grouping/peak-resolution stay in one place."""
    path = SCRIPT_DIR / "extract_windows.py"
    spec = importlib.util.spec_from_file_location("extract_windows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EW = _load_pipeline()


def resolve_paths(raw: str) -> tuple[Path, Path]:
    """Accept the .json, the .csv, or the shared prefix; return (csv, json)."""
    p = Path(raw)
    base = p.with_suffix("") if p.suffix in {".json", ".csv"} else p
    csv_path, json_path = base.with_suffix(".csv"), base.with_suffix(".json")
    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"need both {csv_path.name} and {json_path.name}")
    return csv_path, json_path


def accel_magnitude(df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(df.accelX**2 + df.accelY**2 + df.accelZ**2).values


UNMARKED_THRESHOLD_G = 6.0   # a contact this hard is unambiguously a stroke


def unmarked_strong_peaks(ts, mag, markers, thr=UNMARKED_THRESHOLD_G, min_sep=0.6):
    """Independent peaks above `thr` with no marker nearby.

    This is the decisive missed-stroke test and it needs no baseline: a hard
    contact in the raw signal that the watch never marked is a stroke that was
    lost, full stop. The acceptance window spans both marker formats — peak-stamped
    markers land on the contact, older ones lag it by up to ~1s.
    """
    idx = np.where(mag >= thr)[0]
    if idx.size == 0:
        return 0, 0
    kept_t = []
    for i in idx[np.argsort(-mag[idx])]:          # strongest first, greedily spaced
        t = ts[i]
        if all(abs(t - k) >= min_sep for k in kept_t):
            kept_t.append(t)
    mk = np.array(sorted(markers))
    unmarked = 0
    for t in kept_t:
        lo, hi = np.searchsorted(mk, t - 0.4), np.searchsorted(mk, t + 1.5)
        if hi <= lo:
            unmarked += 1
    return len(kept_t), unmarked


def analyse(csv_path: Path, json_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    meta = json.load(open(json_path))
    markers = sorted(
        (float(m["timestamp"]) for m in meta.get("markers", []) if float(m.get("timestamp", -1)) >= 0)
    )

    ts = df["timestamp"].values
    mag = accel_magnitude(df)
    duration = float(meta.get("recordingActiveSecs") or (ts[-1] - ts[0]))
    minutes = max(duration / 60.0, 1e-9)

    # Acceleration where the markers actually landed.
    at_marker = []
    for t in markers:
        i0, i1 = np.searchsorted(ts, t - 0.05), np.searchsorted(ts, t + 0.05)
        if i1 > i0:
            at_marker.append(mag[i0:i1].max())
    at_marker = np.array(at_marker) if at_marker else np.array([0.0])

    gaps = np.diff(markers) if len(markers) > 1 else np.array([np.nan])

    hard_peaks, unmarked = unmarked_strong_peaks(ts, mag, markers)

    # Run the real offline pipeline so duplicates are collapsed the same way on
    # every session being compared.
    has_cols = all(c in df.columns for c in EW.OUTPUT_COLS)
    if has_cols:
        raw = list(meta.get("markers", []))
        # Same format-aware path extract_windows.py uses, so both generations of
        # recording are collapsed the way they will be for training.
        peak_stamped = EW.markers_are_peak_stamped(raw)
        bursts = EW.group_markers_into_bursts(raw, 0.0 if peak_stamped else EW.DEFAULT_BURST_GAP)
        bursts = EW.resolve_burst_peaks(
            bursts, df, search_pre=EW.PEAK_STAMPED_SEARCH_PRE if peak_stamped else None
        )
        if not peak_stamped:
            bursts = EW.dedup_preparation_bursts(bursts, df)
        strokes = len(bursts)
        labelled = sum(1 for b in bursts if b["strokeType"] not in ("unknown", ""))
        # Marker lag: how far each burst's first marker sits AFTER its contact peak.
        lags = [b["span_start"] - b["peak_ts"] for b in bursts]
        lag = float(np.median(lags)) if lags else float("nan")
        peak_g = float(np.median([mag[b["peak_idx"]] for b in bursts])) if bursts else float("nan")
    else:
        strokes = labelled = 0
        lag = peak_g = float("nan")
        peak_stamped = False

    return {
        "name": json_path.parent.name or json_path.stem,
        "format": "peak" if peak_stamped else "crossing",
        "mode": meta.get("mode", "?"),
        "classifier": meta.get("classifierVersion", "?"),
        "minutes": minutes,
        "markers": len(markers),
        "markers_min": len(markers) / minutes,
        "median_g_at_marker": float(np.median(at_marker)),
        "lag": lag,
        "strokes": strokes,
        "strokes_min": strokes / minutes,
        "labelled": labelled,
        "median_g_at_peak": peak_g,
        "hard_peaks_min": hard_peaks / minutes,
        "unmarked": unmarked,
        "unmarked_min": unmarked / minutes,
        "gap_min": float(np.nanmin(gaps)) if gaps.size else float("nan"),
        "pct_under_060": 100.0 * float(np.mean(gaps < 0.6)) if gaps.size and not np.isnan(gaps).all() else float("nan"),
        "pct_band_03_04": 100.0 * float(np.mean((gaps >= 0.3) & (gaps < 0.4))) if gaps.size and not np.isnan(gaps).all() else float("nan"),
    }


ROWS = [
    ("marker format",       "format",              "{:>13}"),
    ("mode",                "mode",                "{:>13}"),
    ("classifier",          "classifier",          "{:>13}"),
    ("minutes recorded",    "minutes",             "{:13.1f}"),
    ("",                    None,                  None),
    ("marker lag (s)",      "lag",                 "{:+13.2f}"),
    ("median g at marker",  "median_g_at_marker",  "{:13.2f}"),
    ("median g at peak",    "median_g_at_peak",    "{:13.2f}"),
    ("",                    None,                  None),
    (f"contacts>{int(UNMARKED_THRESHOLD_G)}g / min", "hard_peaks_min", "{:13.2f}"),
    ("  of those UNMARKED",  "unmarked",            "{:13d}"),
    ("",                    None,                  None),
    ("strokes (pipeline)",  "strokes",             "{:13d}"),
    ("strokes / min",       "strokes_min",         "{:13.1f}"),
    ("labelled strokes",    "labelled",            "{:13d}"),
    ("",                    None,                  None),
    ("raw markers",         "markers",             "{:13d}"),
    ("markers / min",       "markers_min",         "{:13.1f}"),
    ("min gap (s)",         "gap_min",             "{:13.3f}"),
    ("gaps < 0.6s  (%)",    "pct_under_060",       "{:13.1f}"),
    ("0.3-0.4s band (%)",   "pct_band_03_04",      "{:13.1f}"),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare spike-detector behaviour across recorded sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--new", help="session recorded with the new build")
    ap.add_argument("--baseline", nargs="*", default=[], help="reference session(s) from the old build")
    ap.add_argument("sessions", nargs="*", help="additional sessions to tabulate")
    args = ap.parse_args()

    wanted = [(p, "baseline") for p in args.baseline]
    wanted += [(p, "session") for p in args.sessions]
    if args.new:
        wanted.append((args.new, "NEW"))
    if not wanted:
        ap.error("pass --new and/or --baseline / session paths")

    results, tags = [], []
    for raw, tag in wanted:
        try:
            csv_path, json_path = resolve_paths(raw)
        except FileNotFoundError as e:
            print(f"[SKIP] {raw}: {e}", file=sys.stderr)
            continue
        results.append(analyse(csv_path, json_path))
        tags.append(tag)

    if not results:
        print("nothing to compare", file=sys.stderr)
        return 1

    width = 13
    print()
    print(f"{'':22}" + "".join(f"{t:>{width}}" for t in tags))
    print(f"{'':22}" + "".join(f"{r['name'][:12]:>{width}}" for r in results))
    print("-" * (22 + width * len(results)))
    for label, key, fmt in ROWS:
        if key is None:
            print()
            continue
        cells = ""
        for r in results:
            v = r[key]
            if isinstance(v, str):
                cells += f"{v[:width]:>{width}}"
            elif isinstance(v, float) and np.isnan(v):
                cells += f"{'n/a':>{width}}"
            else:
                cells += fmt.format(v)
        print(f"{label:22}" + cells)
    print()

    # Verdict — only meaningful with a NEW session and at least one baseline.
    if args.new and len(results) >= 2:
        new = results[-1]
        base = [r for r, t in zip(results[:-1], tags[:-1]) if t in ("baseline", "session")]
        if base:
            b_strokes = float(np.median([r["strokes_min"] for r in base]))
            b_lag = float(np.median([r["lag"] for r in base if not np.isnan(r["lag"])] or [np.nan]))
            delta = 100.0 * (new["strokes_min"] - b_strokes) / b_strokes if b_strokes else float("nan")

            print("VERDICT")
            print("-------")
            if not np.isnan(b_lag):
                print(f"marker lag   {b_lag:+.2f}s -> {new['lag']:+.2f}s", end="  ")
                print("OK - markers now sit on the contact peak"
                      if abs(new["lag"]) < 0.25 <= abs(b_lag)
                      else "check - expected the new session to be near 0.00s")

            print(f"unmarked >{int(UNMARKED_THRESHOLD_G)}g contacts  {new['unmarked']} "
                  f"({new['unmarked_min']:.2f}/min)", end="  ")
            if new["unmarked_min"] <= 0.05:
                print("OK - the detector marked every hard contact")
            elif new["unmarked_min"] <= 0.3:
                print("a few hard contacts unmarked - worth a video spot-check")
            else:
                print("WARNING - hard contacts going unmarked; loosen the refractory")

            print(f"strokes/min  {b_strokes:.1f} -> {new['strokes_min']:.1f}  ({delta:+.0f}%)", end="  ")
            if new["unmarked_min"] <= 0.05:
                print("- explained by play intensity, not misses (see above)")
            elif delta < -20:
                print("WARNING - likely MISSING strokes; loosen the refractory")
            elif delta < -10:
                print("borderline - check a few rallies against video")
            else:
                print("OK - no sign of missed strokes")
            print(f"             baseline hard contacts/min "
                  f"{float(np.median([r['hard_peaks_min'] for r in base])):.2f} -> {new['hard_peaks_min']:.2f}"
                  "   (compare play intensity, not detector quality)")

            print(f"markers/min  {float(np.median([r['markers_min'] for r in base])):.1f} -> {new['markers_min']:.1f}"
                  "   (a large DROP here is expected: duplicates removed, not strokes)")
            print(f"0.3-0.4s dup band  {float(np.median([r['pct_band_03_04'] for r in base])):.1f}% -> "
                  f"{new['pct_band_03_04']:.1f}%")
            print()
            print("Caveat: strokes/min only compares fairly across sessions of similar")
            print("intensity. A quiet match legitimately has fewer strokes per minute.")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
