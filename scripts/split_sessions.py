#!/usr/bin/env python3
"""
split_sessions.py — PadelLabs ML Toolchain
============================================
Moves whole sessions out of labeled-strokes/train/ into validation/ and test/
so retrain.py and evaluate.py have honest held-out splits.

Strategy:
  * Reads labeled-strokes/metadata.csv to know which file belongs to which session.
  * Sorts sessions by size (descending). Walks largest → smallest, greedily
    filling val / test pools until their target sample counts are met.
  * With ≥ 2 players present, prefers sending different-player sessions to val
    and test so both holdouts contain at least some player-independent signal.
  * Dry-run by default. Pass --apply to actually move files and update metadata.

Usage:
  python3 scripts/split_sessions.py                 # dry-run, default 15%/15%
  python3 scripts/split_sessions.py --apply
  python3 scripts/split_sessions.py --val-frac 0.2 --test-frac 0.2 --apply

Safety:
  * Refuses to move if either split is already non-empty (to avoid double-moves).
  * Refuses if < 3 distinct sessions in train/.
  * Augmented files (*_aug_*) follow their base session.
"""

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
STROKES_DIR = ML_DIR / "labeled-strokes"
META_CSV = STROKES_DIR / "metadata.csv"


def split_is_empty(split: str) -> bool:
    split_dir = STROKES_DIR / split
    if not split_dir.exists():
        return True
    for p in split_dir.rglob("*.csv"):
        return False
    return True


def pick_sessions(sessions_ordered, target_stroke_count, used, prefer_players=None):
    """Greedy-by-diversity: favour smaller + more-diverse sessions so the holdout
    covers as many classes as possible without ballooning past the stroke target.

    `target_stroke_count` is in NON-UNKNOWN stroke samples (unknowns are much larger
    but less important for holdout balance)."""
    chosen, stroke_total = [], 0
    seen_classes: set = set()

    # Rank candidates by (class diversity desc, non-unknown count desc, session size asc)
    def score(info):
        non_unk = sum(v for k, v in info["labels"].items() if k != "unknown")
        diversity = len([k for k in info["labels"] if k != "unknown"])
        return (diversity, non_unk, -info["n"])

    ordered_by_diversity = sorted(sessions_ordered, key=lambda kv: score(kv[1]), reverse=True)

    # Player preference pass
    if prefer_players:
        for sid, info in ordered_by_diversity:
            if sid in used or stroke_total >= target_stroke_count: continue
            if info["player"] not in prefer_players: continue
            chosen.append(sid); used.add(sid)
            stroke_total += sum(v for k, v in info["labels"].items() if k != "unknown")
            seen_classes.update(k for k in info["labels"] if k != "unknown")

    # Main pass — prefer sessions that add NEW classes
    for sid, info in ordered_by_diversity:
        if sid in used or stroke_total >= target_stroke_count: continue
        new_classes = {k for k in info["labels"] if k != "unknown"} - seen_classes
        if not new_classes and stroke_total > 0.5 * target_stroke_count:
            continue  # skip sessions that don't expand coverage once we're half-full
        chosen.append(sid); used.add(sid)
        stroke_total += sum(v for k, v in info["labels"].items() if k != "unknown")
        seen_classes.update(k for k in info["labels"] if k != "unknown")

    # Compute total (incl. unknown) for reporting
    total_with_unk = sum(sessions_ordered[next(i for i,(s,_) in enumerate(sessions_ordered) if s==sid)][1]["n"]
                        for sid in chosen)
    return chosen, stroke_total, total_with_unk


def main():
    parser = argparse.ArgumentParser(description="Move sessions from train/ to validation/ and test/")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--apply", action="store_true", help="Actually move files (default: dry-run)")
    parser.add_argument("--prefer-player-holdout", action="store_true",
                        help="If true, prefer val/test sessions from minority players (more honest)")
    args = parser.parse_args()

    if not META_CSV.exists():
        print(f"[ERROR] {META_CSV} not found", file=sys.stderr); sys.exit(1)

    # Refuse if val/test already populated
    for split in ("validation", "test"):
        if not split_is_empty(split):
            print(f"[ERROR] labeled-strokes/{split}/ is already non-empty. "
                  "Move its contents back to train/ before re-splitting.", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(META_CSV)
    df_train = df[df["split"] == "train"].copy()
    if df_train.empty:
        print("[ERROR] metadata has no rows with split=train"); sys.exit(1)

    # Group by session
    sessions = defaultdict(lambda: {"player": None, "n": 0, "labels": Counter(), "files": []})
    for _, r in df_train.iterrows():
        s = sessions[r["session_id"]]
        s["player"] = r["player_id"]
        s["n"] += 1
        s["labels"][r["stroke_type"]] += 1
        s["files"].append(r["window_file"])

    if len(sessions) < 3:
        print(f"[ERROR] only {len(sessions)} sessions in train/ — need ≥3 to split"); sys.exit(1)

    total_strokes = sum(sum(v for k, v in s["labels"].items() if k != "unknown")
                        for s in sessions.values())
    total_all = sum(s["n"] for s in sessions.values())
    target_val = int(total_strokes * args.val_frac)
    target_test = int(total_strokes * args.test_frac)

    ordered = list(sessions.items())
    player_session_count = Counter(info["player"] for _, info in ordered)
    minority_players = {p for p, c in player_session_count.items() if c <= 2}

    used = set()
    val_ids, val_strokes, val_total = pick_sessions(
        ordered, target_val, used,
        prefer_players=minority_players if args.prefer_player_holdout else None,
    )
    test_ids, test_strokes, test_total = pick_sessions(
        ordered, target_test, used,
        prefer_players=minority_players if args.prefer_player_holdout else None,
    )

    print(f"\nTotal rows: {total_all}  (labeled strokes: {total_strokes})  in {len(sessions)} sessions")
    print(f"Target val  = {target_val} strokes ({args.val_frac:.0%})  → picked {val_strokes} strokes + unknowns across {len(val_ids)} session(s)")
    print(f"Target test = {target_test} strokes ({args.test_frac:.0%})  → picked {test_strokes} strokes + unknowns across {len(test_ids)} session(s)")

    def summarise(label, ids):
        labels = Counter()
        players = Counter()
        for sid in ids:
            labels.update(sessions[sid]["labels"])
            players[sessions[sid]["player"]] += 1
        print(f"\n── {label} ──")
        for sid in ids:
            s = sessions[sid]
            print(f"  {sid[:8]}… player={s['player'][:8]}  n={s['n']}  labels={dict(s['labels'])}")
        print(f"  players: {dict(players)}")
        print(f"  labels:  {dict(labels)}")

    summarise("VAL",  val_ids)
    summarise("TEST", test_ids)

    # Safety: warn if any class is entirely missing from val or test
    remaining_train = {sid: info for sid, info in sessions.items() if sid not in used}
    train_labels = Counter()
    for info in remaining_train.values():
        train_labels.update(info["labels"])
    val_labels = Counter(); test_labels = Counter()
    for sid in val_ids: val_labels.update(sessions[sid]["labels"])
    for sid in test_ids: test_labels.update(sessions[sid]["labels"])

    print("\n── Class coverage ──")
    all_classes = set(train_labels) | set(val_labels) | set(test_labels)
    for c in sorted(all_classes):
        t = train_labels.get(c, 0); v = val_labels.get(c, 0); te = test_labels.get(c, 0)
        warn = " ⚠️" if (v == 0 or te == 0) else ""
        print(f"  {c:18s}  train={t:>4d}  val={v:>4d}  test={te:>4d}{warn}")

    if not args.apply:
        print("\n[DRY-RUN] Re-run with --apply to actually move files + update metadata.csv")
        return

    # Apply: move files and rewrite metadata
    def move_session(sid, target_split):
        for rel_file in sessions[sid]["files"]:
            src = STROKES_DIR / rel_file
            if not src.exists():  # some files may already be gone
                continue
            label = Path(rel_file).parts[1]
            dst_dir = STROKES_DIR / target_split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            shutil.move(str(src), str(dst))

    moved = 0
    for sid in val_ids:  move_session(sid, "validation"); moved += sessions[sid]["n"]
    for sid in test_ids: move_session(sid, "test");       moved += sessions[sid]["n"]

    # Rewrite metadata.csv with updated split column + window_file paths
    def new_row(row):
        sid = row["session_id"]
        if sid in set(val_ids):
            new_split = "validation"
        elif sid in set(test_ids):
            new_split = "test"
        else:
            return row  # unchanged
        old_path = Path(row["window_file"])
        row["split"] = new_split
        row["window_file"] = str(Path(new_split) / old_path.parts[1] / old_path.name)
        return row

    df_updated = df.apply(new_row, axis=1)
    df_updated.to_csv(META_CSV, index=False)

    print(f"\n✅ Moved {moved} files and updated metadata.csv")
    print("   Next: run `python3 scripts/evaluate_cv.py` to see the honest numbers.")


if __name__ == "__main__":
    main()
