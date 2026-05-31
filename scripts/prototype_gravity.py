#!/usr/bin/env python3
"""
prototype_gravity.py — test gravity-relative, heading-invariant normalization
==============================================================================
Recordings use CoreMotion .xArbitraryZVertical: Z = gravity (vertical), but the
horizontal heading is arbitrary per session. So absolute world X/Y are not
comparable across players. This builds HEADING-INVARIANT, gravity-relative
channels from each sample and re-trains the CNN to see if Guillermo (LH) + Diego
recover vs the raw-axis baseline (CNN test 8.6%, Guillermo 5%, Diego 20%).

Per sample, from accel(user, gravity-removed), gyro, attitude(roll,pitch,yaw):
  g_hat   = unit gravity direction in device frame (from attitude)
  a_vert  = accel · g_hat                       (signed; up/down swing)
  a_hmag  = |accel - a_vert·g_hat|              (horizontal effort, heading-free)
  a_mag   = |accel|
  w_vert  = gyro · g_hat                         (signed; rotation about gravity → FH/BH)
  w_hmag  = |gyro - w_vert·g_hat|
  w_mag   = |gyro|
  pitch   = elevation angle (gravity-relative, heading-free)
Handedness: w_vert and a_vert keep sign; the FH/BH-distinguishing axis is w_vert,
which reverses for a mirrored (left-handed) swing → flip w_vert (and a_vert stays,
vertical is handedness-symmetric) for left-handed players to canonicalise.
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
# left-handed player ids (from players.csv notes): Guillermo
LEFTIES = {"1880202A"}
ID2NAME = {"5E354AD0": "Diego", "1880202A": "Guillermo", "14A0FF96": "Victor", "1DC25578": "Carlos"}
CLASSES = ["backhand", "backhand_lob", "backhand_volley", "bandeja", "forehand",
           "forehand_lob", "forehand_volley", "smash", "vibora"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
RAW = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]


def euler_to_R(roll, pitch, yaw):
    """Body->world rotation, aircraft ZYX (yaw·pitch·roll). Per-sample arrays in."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) Ry(pitch) Rx(roll), shape [...,3,3]
    R = np.empty(roll.shape + (3, 3), dtype=np.float64)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def gravity_relative(window, is_lefty):
    """window: [100,9] raw -> [100,8] gravity-relative heading-invariant channels."""
    accel = window[:, 0:3]
    gyro = window[:, 3:6]
    roll, pitch, yaw = window[:, 6], window[:, 7], window[:, 8]
    R = euler_to_R(roll, pitch, yaw)              # body->world per sample
    # gravity world-down [0,0,-1] expressed in device frame = R^T · [0,0,-1]
    g_world = np.array([0.0, 0.0, -1.0])
    g_dev = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), g_world)  # [100,3]
    g_hat = g_dev / (np.linalg.norm(g_dev, axis=1, keepdims=True) + 1e-9)

    def decomp(v):
        vert = np.sum(v * g_hat, axis=1)                 # signed along gravity
        hvec = v - vert[:, None] * g_hat
        hmag = np.linalg.norm(hvec, axis=1)
        mag = np.linalg.norm(v, axis=1)
        return vert, hmag, mag

    a_vert, a_hmag, a_mag = decomp(accel)
    w_vert, w_hmag, w_mag = decomp(gyro)
    if is_lefty:
        w_vert = -w_vert      # rotation about gravity reverses for mirrored swing
    out = np.stack([a_vert, a_hmag, a_mag, w_vert, w_hmag, w_mag, pitch, np.cos(pitch)], axis=1)
    return out.astype(np.float32)                        # [100,8]


def load_split(split):
    Xs, ys, fs = [], [], []
    for cls in CLASSES:
        d = STROKES / split / cls
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            df = pd.read_csv(csv)
            if len(df) != 100 or any(c not in df.columns for c in RAW):
                continue
            w = df[RAW].to_numpy(np.float64)
            if not np.isfinite(w).all():
                continue
            lefty = WF2PLAYER.get(csv.name, "") in LEFTIES
            Xs.append(gravity_relative(w, lefty).T)   # [8,100]
            ys.append(CLS2IDX[cls]); fs.append(csv.name)
    return np.stack(Xs), np.array(ys), fs


def main():
    import torch, torch.nn as nn
    torch.manual_seed(42); np.random.seed(42)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr, ytr, _ = load_split("train"); Xva, yva, _ = load_split("validation"); Xte, yte, fte = load_split("test")
    print(f"gravity-relative channels={Xtr.shape[1]}  train {Xtr.shape} test {Xte.shape} dev={dev}")
    mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    f = lambda X: ((X - mu) / sd).astype("float32")
    Xtr, Xva, Xte = f(Xtr), f(Xva), f(Xte)
    t = lambda X: torch.tensor(X).to(dev)
    Xtr_t, ytr_t, Xva_t, Xte_t = t(Xtr), torch.tensor(ytr).to(dev), t(Xva), t(Xte)
    counts = Counter(ytr.tolist()); w = torch.tensor([1/counts[i] for i in range(9)], dtype=torch.float32)
    w = (w/w.sum()*9).to(dev)
    C = Xtr.shape[1]

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(C,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,9))
        def forward(s,x): return s.net(x)
    m = CNN().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w); n = len(Xtr_t); best=-1; best_state=None
    for ep in range(80):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            idx = perm[i:i+64]; opt.zero_grad(); lossf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): va = (m(Xva_t).argmax(1).cpu().numpy()==yva).mean()
        if va>best: best=va; best_state={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): pred = m(Xte_t).argmax(1).cpu().numpy()
    acc = (pred==yte).mean()
    print(f"\n{'='*56}\nGRAVITY-RELATIVE CNN  val={best*100:.1f}%  held-out test={acc*100:.1f}%")
    print(f"(raw-axis CNN was: val 69.1%, test 8.6%  |  RF v4 test 37.5%)\n{'='*56}")
    players = np.array([WF2PLAYER.get(x,"?") for x in fte])
    for pid in sorted(set(players)):
        mm = players==pid
        print(f"  {ID2NAME.get(pid,pid):10} n={mm.sum():4} acc={(pred[mm]==yte[mm]).mean()*100:.0f}%")
    print("per-class:")
    for ci,c in enumerate(CLASSES):
        mm=yte==ci
        print(f"  {c:<18}{int((pred[mm]==yte[mm]).sum()):>4}/{int(mm.sum()):<4}={(pred[mm]==yte[mm]).mean()*100 if mm.sum() else 0:>3.0f}%")


if __name__ == "__main__":
    main()
