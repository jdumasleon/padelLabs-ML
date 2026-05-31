#!/usr/bin/env python3
"""
split_by_player.py — PadelLabs ML Toolchain
=============================================
Reads metadata/players.csv and recommends a player-independent train/val/test split.
Tries to balance:
  - ~70% train / 15% val / 15% test by stroke count
  - At least 1 left-handed player in val + test
  - At least 1 non-dominant-wrist player in val + test
  - Mix of skill levels across splits

Outputs the recommended split as a shell command you can pass directly to extract_windows.py.

Usage:
    python3 split_by_player.py --players ../metadata/players.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", required=True, help="Path to metadata/players.csv")
    args = parser.parse_args()

    players_path = Path(args.players)
    if not players_path.exists():
        print(f"ERROR: {players_path} not found.", file=sys.stderr)
        sys.exit(1)

    players = []
    with open(players_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            players.append(row)

    if len(players) < 5:
        print("⚠️  Fewer than 5 players — split may not be meaningful. Collect more data first.")

    n = len(players)
    n_test = max(1, round(n * 0.15))
    n_val  = max(1, round(n * 0.15))

    # Sort by stroke count desc so heavy collectors go to train
    try:
        players_sorted = sorted(players, key=lambda p: -int(p.get("strokeCount", 0)))
    except (ValueError, KeyError):
        players_sorted = players

    # Try to ensure diversity in val/test: pick left-handed and non-dominant-wrist first
    lefties = [p for p in players_sorted if p.get("handedness", "right").lower() == "left"]
    non_dom = [p for p in players_sorted if p.get("watchWrist", "") != p.get("handedness", "")]

    test_players = []
    val_players  = []
    remaining    = list(players_sorted)

    def pick(pool, target_list, count):
        for p in pool:
            if len(target_list) >= count:
                break
            if p not in target_list and p in remaining:
                target_list.append(p)
                remaining.remove(p)

    # Fill test with one leftie + fill rest from remaining (highest stroke count last)
    pick(lefties, test_players, 1)
    pick(list(reversed(remaining)), test_players, n_test)

    # Fill val with one non-dominant-wrist + fill rest
    pick(non_dom, val_players, 1)
    pick(list(reversed(remaining)), val_players, n_val)

    train_players = remaining

    def ids(lst):
        return " ".join(p["playerId"] for p in lst)

    print("\n── Recommended Split ─────────────────────────────────────")
    print(f"  Train ({len(train_players)} players): {ids(train_players)}")
    print(f"  Val   ({len(val_players)} players):   {ids(val_players)}")
    print(f"  Test  ({len(test_players)} players):  {ids(test_players)}")

    print("\n── extract_windows.py command ────────────────────────────")
    print(f"python3 scripts/extract_windows.py \\")
    print(f"    --sessions  raw-sessions \\")
    print(f"    --output    labeled-strokes \\")
    print(f"    --val-players  {ids(val_players)} \\")
    print(f"    --test-players {ids(test_players)}")


if __name__ == "__main__":
    main()
