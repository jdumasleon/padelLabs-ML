#!/usr/bin/env python3
"""
segmentation_audit.py — is each window actually centred on ball-contact?
=========================================================================
Read-only. For every extracted stroke window, looks at the accel-magnitude
profile and reports:
  - peak_pos   : index of max |accel| (extract_windows anchors it at 30 → 30 pre)
  - double_pct : windows with a SECOND peak > 0.5×main, ≥10 samples away
                 (prep-swing contamination OR two merged strokes)
  - prominence : main_peak / median(|accel|)  (low → no clear impact)
Broken down by split and player, so held-out players (Diego/Guillermo) can be
compared to the training distribution.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ML = Path(__file__).parent.parent
STROKES = ML / "labeled-strokes"
META = pd.read_csv(STROKES / "metadata.csv")
WF2P = {Path(r.window_file).name: str(r.player_id)[:8] for _, r in META.iterrows()}
ID2N = {"E168B6E8": "Jose", "3FF5CBC3": "Ela", "20A80EF6": "Enrique", "863A5EC0": "Irene",
        "4812651B": "Javier", "62B73338": "Jorge", "14A0FF96": "Victor", "1DC25578": "Carlos",
        "5E354AD0": "Diego", "1880202A": "Guillermo*LH"}


def peak_stats(mag):
    n = len(mag)
    main = int(np.argmax(mag))
    mainval = mag[main]
    med = np.median(mag) + 1e-9
    # secondary peak away from main
    msk = np.ones(n, bool)
    lo, hi = max(0, main - 10), min(n, main + 11)
    msk[lo:hi] = False
    second = mag[msk].max() if msk.any() else 0.0
    return main, mainval / med, (second > 0.5 * mainval)


def run():
    rows = []
    for split in ("train", "validation", "test"):
        for cls_dir in sorted((STROKES / split).glob("*")):
            if not cls_dir.is_dir() or cls_dir.name == "unknown":
                continue
            for csv in cls_dir.glob("*.csv"):
                df = pd.read_csv(csv)
                if len(df) != 100 or "accelX" not in df.columns:
                    continue
                mag = np.sqrt(df.accelX**2 + df.accelY**2 + df.accelZ**2).to_numpy()
                pos, prom, dbl = peak_stats(mag)
                rows.append((split, WF2P.get(csv.name, "?"), cls_dir.name, pos, prom, dbl))
    d = pd.DataFrame(rows, columns=["split", "pid", "cls", "pos", "prom", "dbl"])

    print(f"TOTAL windows audited: {len(d)}  (anchor target = sample 30)\n")
    print("=== by split ===")
    print(f"{'split':<12}{'n':>6}{'pos_mean':>10}{'pos_std':>9}{'%peak@30±5':>12}{'double%':>9}{'prom_med':>10}")
    for sp, g in d.groupby("split"):
        near = (g.pos.between(25, 35)).mean() * 100
        print(f"{sp:<12}{len(g):>6}{g.pos.mean():>10.1f}{g.pos.std():>9.1f}{near:>11.0f}%{g.dbl.mean()*100:>8.0f}%{g.prom.median():>10.1f}")

    print("\n=== by player (sorted; held-out = Diego, Guillermo) ===")
    print(f"{'player':<14}{'n':>6}{'pos_mean':>10}{'pos_std':>9}{'%@30±5':>9}{'double%':>9}{'prom_med':>10}")
    for pid, g in sorted(d.groupby("pid"), key=lambda kv: -len(kv[1])):
        near = (g.pos.between(25, 35)).mean() * 100
        print(f"{ID2N.get(pid,pid):<14}{len(g):>6}{g.pos.mean():>10.1f}{g.pos.std():>9.1f}{near:>8.0f}%{g.dbl.mean()*100:>8.0f}%{g.prom.median():>10.1f}")

    print("\n=== double-peak rate by class (prep-swing / merge contamination) ===")
    for cls, g in sorted(d.groupby("cls"), key=lambda kv: -kv[1].dbl.mean()):
        print(f"  {cls:<18}{g.dbl.mean()*100:>4.0f}%   prom_med={g.prom.median():.1f}")


if __name__ == "__main__":
    run()
