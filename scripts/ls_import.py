#!/usr/bin/env python3
"""
ls_import.py — PadelLabs ML Toolchain
======================================
Converts extracted stroke-window CSV files into a Label Studio JSON import
file for the PadelLabs Stroke QC project (12-class Time Series labeling).

How it works:
- Reads every .csv file from a stroke-window folder (e.g. labeled-strokes/train/smash/)
- Embeds the CSV content inline so Label Studio can render the time series without
  a file server (works out of the box with Docker + bind-mounted data)
- Pre-fills the "strokeType" prediction from the folder name so the labeler only
  needs to confirm or correct — not select from scratch
- Optionally processes multiple class folders at once with --all-classes

Usage:
    # Import one class at a time (recommended for QC):
    python3 ls_import.py \\
        --windows  ../labeled-strokes/train/smash \\
        --output   ../label-studio-import/smash_tasks.json

    # Import all classes from a split folder:
    python3 ls_import.py \\
        --split    ../labeled-strokes/train \\
        --output   ../label-studio-import/train_all_tasks.json \\
        [--max-per-class 100]

    # Import with 'good' quality pre-filled (fastest review path):
    python3 ls_import.py \\
        --windows  ../labeled-strokes/train/forehand \\
        --output   ../label-studio-import/forehand_tasks.json \\
        --prefill-quality good

Then in Label Studio:
    Project → Import → Upload the generated .json file.

Notes:
- Label Studio TimeSeries uses value="$csv" to load inline data.
- The task metadata field "source_file" helps trace back to the original CSV.
- See LABEL_STUDIO_SETUP.md for the full labeling interface XML config.
"""

import argparse
import json
import sys
from pathlib import Path

VALID_STROKE_TYPES = [
    "smash", "vibora", "bandeja", "rulo",
    "forehand", "backhand",
    "forehandLob", "backhandLob",
    "forehandVolley", "backhandVolley",
    "serve", "unknown",
]

VALID_QUALITY_VALUES = ["good", "mishit", "double_spike", "discard"]


# ── Task builder ──────────────────────────────────────────────────────────────

def csv_to_task(csv_path: Path, stroke_type: str, prefill_quality: str | None) -> dict:
    """Build a single Label Studio task dict from a window CSV file."""
    csv_content = csv_path.read_text(encoding="utf-8")

    task: dict = {
        "data": {
            "csv": csv_content,
            "strokeType": stroke_type,           # visible in the task list
            "source_file": csv_path.name,        # traceability
        },
        "meta": {
            "stroke_type": stroke_type,
            "source_file": str(csv_path),
        },
    }

    # Pre-populate predictions so the labeler just confirms, not re-selects
    prediction_result = [
        {
            "from_name": "strokeType",
            "to_name": "ts",
            "type": "choices",
            "value": {"choices": [stroke_type]},
        }
    ]

    if prefill_quality:
        prediction_result.append({
            "from_name": "quality",
            "to_name": "ts",
            "type": "choices",
            "value": {"choices": [prefill_quality]},
        })

    task["predictions"] = [{"result": prediction_result, "score": 1.0}]
    return task


# ── Folder processing ─────────────────────────────────────────────────────────

def process_class_folder(
    folder: Path,
    stroke_type: str,
    prefill_quality: str | None,
    max_count: int | None,
) -> list[dict]:
    """Return a list of Label Studio task dicts for all CSVs in folder."""
    csv_files = sorted(folder.glob("*.csv"))
    if max_count is not None:
        csv_files = csv_files[:max_count]

    tasks = []
    for csv_path in csv_files:
        try:
            task = csv_to_task(csv_path, stroke_type, prefill_quality)
            tasks.append(task)
        except Exception as e:
            print(f"  WARN: could not read {csv_path.name}: {e}", file=sys.stderr)

    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert stroke-window CSVs to a Label Studio JSON import file."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--windows", metavar="DIR",
        help="Single stroke-class folder (e.g. labeled-strokes/train/smash)"
    )
    source.add_argument(
        "--split", metavar="DIR",
        help="Split folder containing one sub-folder per class (e.g. labeled-strokes/train)"
    )

    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output .json file path (e.g. label-studio-import/smash_tasks.json)"
    )
    parser.add_argument(
        "--prefill-quality", metavar="QUALITY",
        choices=VALID_QUALITY_VALUES,
        help="Pre-fill quality choice for all tasks (default: none)"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None, metavar="N",
        help="Maximum tasks per class (useful for large datasets)"
    )

    args = parser.parse_args()

    all_tasks: list[dict] = []

    if args.windows:
        windows_dir = Path(args.windows)
        if not windows_dir.is_dir():
            print(f"ERROR: {windows_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)

        stroke_type = windows_dir.name
        if stroke_type not in VALID_STROKE_TYPES:
            print(
                f"WARNING: folder name '{stroke_type}' is not a known stroke type. "
                f"Using it as-is.",
                file=sys.stderr
            )

        print(f"Processing: {windows_dir.name} ({stroke_type})")
        tasks = process_class_folder(windows_dir, stroke_type, args.prefill_quality, args.max_per_class)
        all_tasks.extend(tasks)
        print(f"  {len(tasks)} tasks generated")

    else:  # --split
        split_dir = Path(args.split)
        if not split_dir.is_dir():
            print(f"ERROR: {split_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            stroke_type = class_dir.name
            if stroke_type not in VALID_STROKE_TYPES:
                print(f"  SKIP (unknown class): {stroke_type}")
                continue

            print(f"Processing: {stroke_type}")
            tasks = process_class_folder(class_dir, stroke_type, args.prefill_quality, args.max_per_class)
            all_tasks.extend(tasks)
            print(f"  {len(tasks)} tasks generated")

    if not all_tasks:
        print("ERROR: No tasks generated. Are there .csv files in the folder?", file=sys.stderr)
        sys.exit(1)

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Output:      {output_path.resolve()}")
    print(f"\nNext steps:")
    print(f"  1. Open Label Studio → your project")
    print(f"  2. Click Import → Upload → select {output_path.name}")
    print(f"  3. Review each window: confirm/correct strokeType and set quality")


if __name__ == "__main__":
    main()
