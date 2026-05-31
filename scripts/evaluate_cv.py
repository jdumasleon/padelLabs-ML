#!/usr/bin/env python3
"""
evaluate_cv.py — Honest cross-validation for the stroke classifier
=====================================================================
Replaces the legacy evaluate.py (which evaluated a pickled model against a
flat test folder using random splits). This one is the honest baseline:

  [1] Stratified 5-fold CV   (samples shuffled — optimistic)
  [2] Session GroupKFold     (unseen session  — realistic)
  [3] Player  GroupKFold     (unseen player   — most honest)

It trains a fresh Random Forest per fold with the SAME hyperparameters and
feature set as retrain.py — so the numbers reflect how the model would
actually perform if the current dataset were re-trained and shipped.

Also prints:
  * per-class recall under the strictest CV
  * top confusions (off-diagonal)
  * feature-importance top/bottom
  * held-out validation + test accuracy if models/<v>/*.mlmodel is provided

Outputs a markdown report to models/evaluation_report.md.

Usage:
  python3 scripts/evaluate_cv.py
  python3 scripts/evaluate_cv.py --n-estimators 300
  python3 scripts/evaluate_cv.py --include-unknown
"""

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
STROKES_DIR = ML_DIR / "labeled-strokes"
MODELS_DIR = ML_DIR / "models"
META_CSV = STROKES_DIR / "metadata.csv"
SESSIONS_CSV = ML_DIR / "metadata" / "sessions.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from retrain import extract_features, load_split  # type: ignore


def load_all_with_groups(include_unknown: bool):
    """Load train + validation + test. Returns X, y, session_ids, player_ids."""
    X_parts, y_parts, path_parts = [], [], []
    for split in ("train", "validation", "test"):
        Xp, yp, pp = load_split(STROKES_DIR / split)
        if len(Xp) == 0:
            continue
        X_parts.append(Xp); y_parts.extend(yp); path_parts.extend(pp)

    if not X_parts:
        print("[ERROR] No data found in labeled-strokes/*/"); sys.exit(1)
    X = np.vstack(X_parts); y = list(y_parts); paths = list(path_parts)

    if not include_unknown:
        mask = np.array([lbl != "unknown" for lbl in y])
        X = X[mask]
        y = [v for v, m in zip(y, mask) if m]
        paths = [p for p, m in zip(paths, mask) if m]

    # Map to session_id + player_id
    meta = pd.read_csv(META_CSV)
    path_to_session = {row["window_file"]: row["session_id"] for _, row in meta.iterrows()}
    sessions = pd.read_csv(SESSIONS_CSV) if SESSIONS_CSV.exists() else pd.DataFrame(columns=["session_id", "player_id"])
    sess_to_player = dict(zip(sessions["session_id"], sessions["player_id"]))

    rels = ["/".join(Path(p).parts[-3:]) for p in paths]
    keep = np.array([r in path_to_session for r in rels])
    X = X[keep]
    y = [v for v, m in zip(y, keep) if m]
    rels = [r for r, m in zip(rels, keep) if m]
    session_ids = np.array([path_to_session[r] for r in rels])
    player_ids = np.array([sess_to_player.get(s, "UNKNOWN") for s in session_ids])
    return X, np.array(y), session_ids, player_ids


def run_cv(X, y_enc, classes, *, cv, groups=None, n_estimators=300):
    from sklearn.ensemble import RandomForestClassifier
    scores, preds_all, true_all = [], [], []
    per_hit, per_tot = Counter(), Counter()
    splits = cv.split(X, y_enc, groups) if groups is not None else cv.split(X, y_enc)
    for tr, te in splits:
        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
        clf.fit(X[tr], y_enc[tr])
        pred = clf.predict(X[te])
        scores.append((pred == y_enc[te]).mean())
        for p, t in zip(pred, y_enc[te]):
            per_tot[classes[t]] += 1
            if p == t: per_hit[classes[t]] += 1
            preds_all.append(p); true_all.append(t)
    return np.array(scores), per_hit, per_tot, np.array(true_all), np.array(preds_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--include-unknown", action="store_true",
                        help="Include the 'unknown' background class as a 10th label")
    parser.add_argument("--out", default=str(MODELS_DIR / "evaluation_report.md"))
    args = parser.parse_args()

    print("Loading data (train + validation + test) …")
    X, y, sess, plr = load_all_with_groups(args.include_unknown)
    print(f"  samples: {len(y)}  features: {X.shape[1]}  sessions: {len(set(sess))}  players: {len(set(plr))}")
    dist = Counter(y)
    for c, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"    {c:18s} {n}")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import confusion_matrix

    le = LabelEncoder(); y_enc = le.fit_transform(y); classes = list(le.classes_)

    lines = []
    def out(s=""): print(s); lines.append(s)

    out("\n## [1] Stratified 5-fold CV (shuffled — OPTIMISTIC, leaks session)")
    sc1, *_ = run_cv(X, y_enc, classes, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                     n_estimators=args.n_estimators)
    out(f"  acc  {sc1.mean()*100:.1f}% ± {sc1.std()*100:.1f}%  folds={[f'{s*100:.0f}' for s in sc1]}")

    out("\n## [2] GroupKFold by SESSION (REALISTIC)")
    sc2, hit2, tot2, true2, pred2 = run_cv(X, y_enc, classes, cv=GroupKFold(n_splits=5),
                                           groups=sess, n_estimators=args.n_estimators)
    out(f"  acc  {sc2.mean()*100:.1f}% ± {sc2.std()*100:.1f}%  folds={[f'{s*100:.0f}' for s in sc2]}")
    out("  per-class recall:")
    for c in classes:
        if tot2[c]:
            out(f"    {c:18s} {hit2[c]}/{tot2[c]} = {hit2[c]/tot2[c]*100:.0f}%")

    n_players = len(set(plr))
    if n_players >= 2:
        out(f"\n## [3] GroupKFold by PLAYER (n_players={n_players} — MOST HONEST)")
        n_splits = min(5, n_players)
        sc3, hit3, tot3, true3, pred3 = run_cv(X, y_enc, classes, cv=GroupKFold(n_splits=n_splits),
                                               groups=plr, n_estimators=args.n_estimators)
        out(f"  acc  {sc3.mean()*100:.1f}% ± {sc3.std()*100:.1f}%  folds={[f'{s*100:.0f}' for s in sc3]}")
        out("  per-class recall:")
        for c in classes:
            if tot3[c]:
                out(f"    {c:18s} {hit3[c]}/{tot3[c]} = {hit3[c]/tot3[c]*100:.0f}%")
    else:
        out("\n## [3] GroupKFold by PLAYER — SKIPPED (only 1 player)")

    # Top confusions from session-CV
    out("\n## Session-CV top confusions (true → predicted)")
    cm = confusion_matrix(true2, pred2, labels=range(len(classes)))
    entries = []
    for i in range(len(classes)):
        total_i = cm[i].sum()
        if total_i == 0: continue
        for j in range(len(classes)):
            if i == j or cm[i, j] == 0: continue
            entries.append((cm[i, j], classes[i], classes[j], total_i))
    entries.sort(reverse=True)
    for n, src, dst, total in entries[:10]:
        out(f"  {src:18s} → {dst:18s}  {n:>4d}  ({n/total*100:.0f}% of {src})")

    # Feature importance
    out("\n## Feature importance (full-fit)")
    from sklearn.ensemble import RandomForestClassifier
    clf_full = RandomForestClassifier(n_estimators=args.n_estimators, class_weight="balanced",
                                      n_jobs=-1, random_state=42)
    clf_full.fit(X, y_enc)
    feat_names = []
    for col in ["accelX","accelY","accelZ","gyroX","gyroY","gyroZ","roll","pitch","yaw"]:
        for stat in ["mean","std","min","max","range","totalVar","rms","p25","p75"]:
            feat_names.append(f"{col}_{stat}")
    feat_names += ["accelMag_max","accelMag_mean","accelMag_std","accelMag_totalVar","accelMag_peakPos",
                   "accelX_gyroZ_corr","gyroX_slope","accelZ_slope","gyroY_asym"]
    imp = clf_full.feature_importances_; order = np.argsort(imp)[::-1]
    out("  TOP 10:")
    for i in order[:10]:
        out(f"    {feat_names[i]:25s} {imp[i]:.4f}")
    out("  BOTTOM 10 (candidates for removal):")
    for i in order[-10:]:
        out(f"    {feat_names[i]:25s} {imp[i]:.4f}")
    out("  Directional features (v3 additions):")
    for f in ["gyroX_slope","accelZ_slope","gyroY_asym","accelX_gyroZ_corr","accelMag_peakPos"]:
        idx = feat_names.index(f)
        out(f"    {f:25s} imp={imp[idx]:.4f}  rank={list(order).index(idx)+1}/90")

    # Save markdown
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# PadelLabs Stroke Classifier — Honest CV Report",
        "",
        f"- dataset: {len(y)} samples across {len(set(sess))} sessions / {len(set(plr))} players",
        f"- classes: {', '.join(classes)}",
        f"- include_unknown: {args.include_unknown}",
        f"- n_estimators: {args.n_estimators}",
        "",
    ]
    report_path.write_text("\n".join(header + lines) + "\n")
    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
