#!/usr/bin/env python3
"""
compare_versions.py — byte-exact A/B of two shipped .mlmodel files
===================================================================
Runs the ACTUAL CoreML models (not the sklearn stand-in) through coremltools'
predict on the held-out test split, so v3 and v4 are compared on the exact same
player-independent test set (§6.2.1 of CLAUDE.md).

Requires a coremltools that can run inference — use the padel-ml-eval venv
(Python 3.12), NOT padel-ml (3.14, where libcoremlpython is unavailable).

Usage:
  ../padel-ml-eval/bin/python compare_versions.py --a v3 --b v4
  ../padel-ml-eval/bin/python compare_versions.py --a v3 --b v4 --split test
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from retrain import load_split  # canonical 90-feature extractor + window loader

from coremltools.models import MLModel


def ordered_input_names(model: MLModel) -> list[str]:
    return [f.name for f in model.get_spec().description.input]


def class_output_name(model: MLModel) -> str:
    return model.get_spec().description.predictedFeatureName or "strokeType"


def predict_all(model_path: Path, X: np.ndarray) -> list[str]:
    import json
    model = MLModel(str(model_path))
    names = ordered_input_names(model)
    out = class_output_name(model)
    # The CoreML model emits an int64 class index; decode via embedded `classes` list.
    classes = json.loads(model.get_spec().description.metadata.userDefined["classes"])
    if len(names) != X.shape[1]:
        sys.exit(f"[ERROR] {model_path.name}: model expects {len(names)} features, data has {X.shape[1]}")
    preds = []
    for row in X:
        d = {n: float(v) for n, v in zip(names, row)}
        idx = int(model.predict(d)[out])
        preds.append(classes[idx])
    return preds


def recall_table(y_true, y_pred):
    hit = defaultdict(int); tot = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        tot[t] += 1
        if t == p:
            hit[t] += 1
    return {c: (hit[c], tot[c]) for c in sorted(tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="v3", help="baseline version")
    ap.add_argument("--b", default="v4", help="new version")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    args = ap.parse_args()

    split_dir = ML_DIR / "labeled-strokes" / args.split
    X, y, _ = load_split(split_dir)
    y = np.array(y)
    mask = y != "unknown"
    X, y = X[mask], y[mask]
    print(f"Eval split: {args.split}  |  {len(y)} labeled windows  |  classes={sorted(set(y))}\n")

    models = {}
    for v in (args.a, args.b):
        mp = ML_DIR / "models" / v / f"PadelLabs-StrokeClassifier-{v}.mlmodel"
        if not mp.exists():
            sys.exit(f"[ERROR] missing {mp}")
        models[v] = predict_all(mp, X)

    ra = recall_table(y, models[args.a])
    rb = recall_table(y, models[args.b])

    acc_a = np.mean([t == p for t, p in zip(y, models[args.a])])
    acc_b = np.mean([t == p for t, p in zip(y, models[args.b])])

    classes = sorted(set(y))
    print(f"{'class':<18}{args.a:>12}{args.b:>12}{'Δ':>8}")
    print("-" * 50)
    for c in classes:
        ha, ta = ra.get(c, (0, 0))
        hb, tb = rb.get(c, (0, 0))
        pa = 100 * ha / ta if ta else 0
        pb = 100 * hb / tb if tb else 0
        flag = "  ⬇REGRESS" if pb + 1e-9 < pa - 5 else ("  ⬆" if pb > pa + 5 else "")
        print(f"{c:<18}{ha:>4}/{ta:<3}={pa:>3.0f}%{hb:>4}/{tb:<3}={pb:>3.0f}%{pb-pa:>+7.0f}{flag}")
    print("-" * 50)
    print(f"{'OVERALL acc':<18}{acc_a*100:>11.1f}%{acc_b*100:>11.1f}%{(acc_b-acc_a)*100:>+7.1f}")

    regressed = [c for c in classes
                 if (rb.get(c, (0, 1))[0] / max(1, rb.get(c, (0, 1))[1]))
                 < (ra.get(c, (0, 1))[0] / max(1, ra.get(c, (0, 1))[1])) - 0.05]
    print()
    if regressed:
        print(f"⚠️  Classes regressed >5pp on the held-out {args.split} set: {regressed}")
    else:
        print(f"✅ No class regressed >5pp on the held-out {args.split} set.")


if __name__ == "__main__":
    main()
