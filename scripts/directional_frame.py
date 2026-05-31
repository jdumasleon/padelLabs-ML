#!/usr/bin/env python3
"""
directional_frame.py — direction-preserving orientation normalization test
===========================================================================
Tests two world-frame variants against the raw RF baseline (37.5% held-out),
keeping directional signal (unlike the magnitude-only gravity-relative attempt
that hurt the RF).

  V1 = gravity-align only: rotate accel+gyro device->world via attitude.
       World Z = gravity. Horizontal heading left as-is (arbitrary per session).
       Removes wrist-strap TILT variance only. 6 world channels + pitch.
  V2 = V1 + per-window heading alignment: rotate the world horizontal plane so the
       window's principal horizontal-accel axis -> +X (keeps the full 2D horizontal
       VECTOR, not a magnitude). FH/BH stay separable via signed rotation-about-
       gravity (wZ). Left-handers: reflect perpendicular axis + flip wZ.

Run: ../padel-ml-eval/bin/python directional_frame.py
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ML = SCRIPT_DIR.parent
STROKES = ML / "labeled-strokes"
META = pd.read_csv(STROKES / "metadata.csv")
WF2PLAYER = {Path(r.window_file).name: str(r.player_id)[:8] for _, r in META.iterrows()}
LEFTIES = {"1880202A"}
ID2NAME = {"5E354AD0": "Diego", "1880202A": "Guillermo", "14A0FF96": "Victor", "1DC25578": "Carlos"}
CLASSES = ["backhand", "backhand_lob", "backhand_volley", "bandeja", "forehand",
           "forehand_lob", "forehand_volley", "smash", "vibora"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
RAW = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]


def euler_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll); cp, sp = np.cos(pitch), np.sin(pitch); cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty(roll.shape + (3, 3))
    R[..., 0, 0] = cy*cp; R[..., 0, 1] = cy*sp*sr - sy*cr; R[..., 0, 2] = cy*sp*cr + sy*sr
    R[..., 1, 0] = sy*cp; R[..., 1, 1] = sy*sp*sr + cy*cr; R[..., 1, 2] = sy*sp*cr - cy*sr
    R[..., 2, 0] = -sp;   R[..., 2, 1] = cp*sr;            R[..., 2, 2] = cp*cr
    return R  # device->world (treat consistently)


def to_world(window):
    accel = window[:, 0:3]; gyro = window[:, 3:6]
    roll, pitch, yaw = window[:, 6], window[:, 7], window[:, 8]
    R = euler_to_R(roll, pitch, yaw)
    aw = np.einsum("nij,nj->ni", R, accel)
    gw = np.einsum("nij,nj->ni", R, gyro)
    return aw, gw, pitch


def variant(window, mode, is_lefty):
    aw, gw, pitch = to_world(window)            # world frame, Z=vertical
    if mode == "V1":
        if is_lefty:                            # reflect one horizontal axis + spin about gravity
            aw = aw.copy(); gw = gw.copy()
            aw[:, 1] *= -1; gw[:, 2] *= -1
        out = np.concatenate([aw, gw, pitch[:, None]], axis=1)  # 7ch
        return out.astype(np.float32)
    # V2: align horizontal to principal axis of horizontal accel
    h = aw[:, 0:2]                              # [100,2] horizontal accel
    # principal axis via top eigenvector of covariance
    C = h.T @ h
    w, V = np.linalg.eigh(C)
    axis = V[:, -1]                              # dominant direction (sign-ambiguous)
    # sign: orient so peak horizontal-accel sample projects +
    peak = np.argmax(np.linalg.norm(h, axis=1))
    if h[peak] @ axis < 0:
        axis = -axis
    perp = np.array([-axis[1], axis[0]])
    Rh = np.stack([axis, perp], axis=0)         # world-horiz -> canonical [2,2]
    ah = (Rh @ aw[:, 0:2].T).T                   # [100,2]
    gh = (Rh @ gw[:, 0:2].T).T
    aZ = aw[:, 2:3]; gZ = gw[:, 2:3]
    if is_lefty:
        ah = ah.copy(); ah[:, 1] *= -1           # reflect perpendicular
        gh = gh.copy(); gh[:, 1] *= -1
        gZ = -gZ                                  # rotation about gravity reverses
    out = np.concatenate([ah, aZ, gh, gZ, pitch[:, None]], axis=1)  # 7ch
    return out.astype(np.float32)


def load(split, mode):
    Xs, ys, fs = [], [], []
    for cls in CLASSES:
        d = STROKES / split / cls
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            df = pd.read_csv(csv)
            if len(df) != 100 or any(c not in df.columns for c in RAW):
                continue
            w = df[RAW].to_numpy(float)
            if not np.isfinite(w).all():
                continue
            lefty = WF2PLAYER.get(csv.name, "") in LEFTIES
            Xs.append(variant(w, mode, lefty).T)      # [C,100]
            ys.append(CLS2IDX[cls]); fs.append(csv.name)
    return np.stack(Xs), np.array(ys), fs


def rf_eval(mode):
    from sklearn.ensemble import RandomForestClassifier
    def stats(X):
        return np.concatenate([X.mean(2), X.std(2), X.min(2), X.max(2),
                               X.max(2)-X.min(2), np.sqrt((X**2).mean(2)),
                               np.percentile(X, 25, 2), np.percentile(X, 75, 2)], axis=1)
    Xtr, ytr, _ = load("train", mode); Xva, yva, _ = load("validation", mode); Xte, yte, fte = load("test", mode)
    Xtr = np.concatenate([stats(Xtr), stats(Xva)]); ytr = np.concatenate([ytr, yva])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                 random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(stats(Xte)); acc = (pred == yte).mean()
    players = np.array([WF2PLAYER.get(x, "?") for x in fte])
    g = players == "1880202A"; d = players == "5E354AD0"
    print(f"  {mode:3} RF held-out test = {acc*100:.1f}%   Guillermo={ (pred[g]==yte[g]).mean()*100:.0f}%  Diego={(pred[d]==yte[d]).mean()*100:.0f}%")


if __name__ == "__main__":
    print("baseline raw RF = 37.5%  (Guillermo 40, Diego 27)")
    rf_eval("V1")
    rf_eval("V2")
