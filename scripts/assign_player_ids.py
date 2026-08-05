#!/usr/bin/env python3
"""
assign_player_ids.py — PadelLabs ML Toolchain
==============================================
Gives every recorded session the playerId of the person who ACTUALLY played it.

Why this exists
---------------
Sessions are recorded on one phone/watch, so the app stamps the logged-in
account's id no matter who is wearing the watch. Every borrower therefore lands
in the data as the owner. `metadata/players.csv` carries five separate hand-written
remediation notes for exactly this ("was sharing playerId … with Jose"), so it is
worth doing with a repeatable, verifiable script.

It matters more than a cosmetic id fix:

  * `extract_windows.py` splits train/validation/test BY player_id. If several
    people share one id, the split is no longer player-independent and every
    cross-player accuracy number is meaningless — which is the one number the
    v6 corpus exists to improve.
  * The reverse case leaks: one person recorded under TWO ids can appear in both
    train and test at once, inflating held-out accuracy. Jose's account id changed
    from E168B6E8 to 39F3D932 partway through, so his newer sessions must be
    mapped back onto his corpus id rather than given a fresh one.

Usage
-----
    python3 scripts/assign_player_ids.py --sessions ../dataCollection --dry-run
    python3 scripts/assign_player_ids.py --sessions ../dataCollection

Safe to re-run: assignments are declarative, ids for new players are derived
deterministically from the display name, and a one-time backup of each patched
JSON is written to <session>.json.bak-playerid (deliberately NOT the `.orig`
suffix, which apply_validation.py uses to mark a session as label-applied).
"""

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

# Stable namespace so a given display name always derives the same id, on any
# machine, on any re-run.
PLAYER_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://padellabs.app/players")


def derive_player_id(display_name: str) -> str:
    return str(uuid.uuid5(PLAYER_NAMESPACE, display_name.strip().lower())).upper()


# Ids of people already present in the training corpus. Reusing these is what
# keeps a returning player from being counted as somebody new.
KNOWN = {
    "Jose":     "E168B6E8-2D2B-4998-9428-9B4D26F79A8D",
    "Enrique":  "20A80EF6-375D-4FCE-AB9B-BAFCBF1A03A6",
    "Jorge":    "62B73338-18A2-4C13-AE72-061AE70AB95C",
    "Javier":   "4812651B-BFD4-4D4C-A03B-5BE88DE33AF7",
    "Carlos":   "1DC25578-923C-4EFE-9717-237E45B71EDE",
    "Irene":    "863A5EC0-8729-496E-9645-2764E5BDACAA",
    "Diego":    "5E354AD0-50F2-4EC7-B653-1FB9989DC793",
    "Victor":   "14A0FF96-6B40-423D-ADB2-D94E8AD96EF6",
    "Guillermo":"1880202A-848B-444A-886C-B771B2D18998",
    "Ela":      "3FF5CBC3-8124-4B75-85C4-AD1A52B3B57C",
}

# Session folder (relative to --sessions) → who actually played it.
# "Javier S" is a DIFFERENT person from the existing "Javier"; keep them apart.
ASSIGNMENTS = {
    "Adrian/2 July (train for v6)":      dict(name="Adrian",   handedness="right", wrist="right"),
    "Carlitos (train for v6)":           dict(name="Carlitos", handedness="right", wrist="right"),
    "Daniel/4 August (train for v6)":    dict(name="Daniel",   handedness="right", wrist="right"),
    "Edu/29 June (train for v6)":        dict(name="Edu",      handedness="right", wrist="right"),
    "Javier S/8 July (train for v6)":    dict(name="Javier S", handedness="right", wrist="right"),
    "Enrique /26 June (train for v6)":   dict(name="Enrique",  handedness="right", wrist="right"),
    "Jorge/12 June (train for v6)":      dict(name="Jorge",    handedness="right", wrist="right"),
    "Jorge/6 July (train for v6)":       dict(name="Jorge",    handedness="right", wrist="right"),
    "Jose/11 June (train for v6)":       dict(name="Jose",     handedness="right", wrist="right"),
    "Jose/2 August (train for v6)":      dict(name="Jose",     handedness="right", wrist="right"),
    "Jose/20 July (train for v6)":       dict(name="Jose",     handedness="right", wrist="right"),
}

PLAYERS_CSV = Path(__file__).parent.parent / "metadata" / "players.csv"


def player_id_for(name: str) -> str:
    return KNOWN.get(name) or derive_player_id(name)


def session_jsons(root: Path):
    for p in sorted(root.rglob("*.json")):
        n = p.name
        if "_validation" in n or "_classified" in n or n.endswith(".orig"):
            continue
        yield p


def patch_sessions(root: Path, dry_run: bool) -> dict:
    touched, unchanged, unmapped = [], 0, []

    for json_path in session_jsons(root):
        rel = os.path.relpath(json_path.parent, root)
        spec = ASSIGNMENTS.get(rel)
        if spec is None:
            continue

        target = player_id_for(spec["name"])
        meta = json.loads(json_path.read_text())
        current = meta.get("playerId", "")

        if current == target and meta.get("handedness") == spec["handedness"] \
                and meta.get("watchWrist") == spec["wrist"]:
            unchanged += 1
            continue

        meta["playerId"]   = target
        meta["handedness"] = spec["handedness"]
        meta["watchWrist"] = spec["wrist"]

        if not dry_run:
            backup = json_path.with_suffix(json_path.suffix + ".bak-playerid")
            if not backup.exists():
                backup.write_text(json.dumps(json.loads(json_path.read_text())))
            json_path.write_text(json.dumps(meta))

        touched.append((rel, spec["name"], current[:8], target[:8]))

    # Anything still on a shared id that we have no assignment for
    for json_path in session_jsons(root):
        rel = os.path.relpath(json_path.parent, root)
        if rel in ASSIGNMENTS:
            continue
        meta = json.loads(json_path.read_text())
        if meta.get("playerId", "") not in KNOWN.values():
            unmapped.append((rel, meta.get("playerId", "?")[:8]))

    return {"touched": touched, "unchanged": unchanged, "unmapped": unmapped}


def update_players_csv(dry_run: bool) -> list:
    """Add a row for every assigned player that is missing from players.csv."""
    if not PLAYERS_CSV.exists():
        print(f"[WARN] {PLAYERS_CSV} not found — skipping registry update")
        return []

    with open(PLAYERS_CSV, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    existing = {r[0] for r in body if r}

    added = []
    for spec in ASSIGNMENTS.values():
        pid = player_id_for(spec["name"])
        if pid in existing:
            continue
        existing.add(pid)
        row = [""] * len(header)
        row[0] = pid
        row[1] = spec["name"]
        row[2] = spec["handedness"]
        row[3] = spec["wrist"]
        if len(row) > 4:
            row[4] = "intermediate"
        row[-1] = ("Registered by assign_player_ids.py: recorded on Jose's watch under a "
                   "shared account id, re-assigned to a distinct player.")
        body.append(row)
        added.append((pid[:8], spec["name"]))

    if added and not dry_run:
        with open(PLAYERS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="Assign the real playerId to each recorded session.")
    ap.add_argument("--sessions", required=True, help="root of the DataCollection tree")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    root = Path(args.sessions)
    if not root.exists():
        print(f"[ERROR] no such folder: {root}", file=sys.stderr)
        return 1

    result = patch_sessions(root, args.dry_run)
    added = update_players_csv(args.dry_run)

    print(f"\n{'session':38} {'player':10} {'from':>9} {'to':>9}")
    print("-" * 70)
    for rel, name, was, now in result["touched"]:
        print(f"{rel[:38]:38} {name:10} {was:>9} → {now:>9}")

    print(f"\npatched {len(result['touched'])} session(s), {result['unchanged']} already correct")
    if added:
        print(f"players.csv: added {len(added)} → " + ", ".join(f"{n} ({p})" for p, n in added))
    if result["unmapped"]:
        print(f"\n[WARN] {len(result['unmapped'])} session(s) carry an id not in the known set "
              "and have no assignment — check whether they are a new player:")
        for rel, pid in result["unmapped"][:10]:
            print(f"  {pid}  {rel}")
    if args.dry_run:
        print("\n(dry-run — nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
