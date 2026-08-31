#!/usr/bin/env python3
"""
train_cnn.py — temporal 1D-CNN baseline for the stroke classifier
==================================================================
Trains a small 1D convolutional net on the RAW 100x9 stroke windows (not the 90
hand-crafted RF features), so the model can learn the motion *shape* that
separates lobs/drives/overheads. Apples-to-apples vs the RF: same train/val/test
player-independent split.

Run in the padel-ml-eval venv (Python 3.12 + torch).

  ../padel-ml-eval/bin/python train_cnn.py --epochs 80            # baseline
  ../padel-ml-eval/bin/python train_cnn.py --epochs 160 --aug     # with IMU augmentation

Augmentation (--aug): per-epoch rotation perturbation (accel+gyro vectors,
attitude offset approximation), magnitude scaling, smooth time-warp,
time-shift, and noise jitter — synthesizes player/wrist/technique variation
to fight the few-players overfit. Val/test are never augmented.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
STROKES = ML_DIR / "labeled-strokes"
META = pd.read_csv(STROKES / "metadata.csv")
WF2PLAYER = {Path(r.window_file).name: str(r.player_id)[:8] for _, r in META.iterrows()}
ID2NAME = {"5E354AD0": "Diego", "1880202A": "Guillermo"}

FEATURE_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]
CLASSES = ["backhand", "backhand_lob", "backhand_volley", "bandeja", "forehand",
           "forehand_lob", "forehand_volley", "smash", "vibora"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_split(split: str):
    """Return X [N,9,100], y [N], files [N]."""
    Xs, ys, fs = [], [], []
    for cls in CLASSES:
        d = STROKES / split / cls
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            df = pd.read_csv(csv)
            if len(df) != 100 or any(c not in df.columns for c in FEATURE_COLS):
                continue
            arr = df[FEATURE_COLS].to_numpy(dtype=np.float32)  # [100,9]
            if not np.isfinite(arr).all():
                continue
            Xs.append(arr.T)            # [9,100]
            ys.append(CLS2IDX[cls])
            fs.append(csv.name)
    return np.stack(Xs), np.array(ys), fs


# ── IMU augmentation ─────────────────────────────────────────────────────────
# Applied per EPOCH on the raw (un-normalized) train windows, so rotations act
# on physical units. Val/test are never augmented. Each transform hits a random
# ~50% of samples per epoch. Channel layout: accel 0:3, gyro 3:6, attitude 6:9.
#
# Rationale (V6_ROADMAP): the CNN sees the temporal shape the RF can't, but
# overfits at ~10 players. Augmentation synthesizes wrist/watch/technique
# variation to close the val→test gap before more players are collected.

AX_ACC, AX_GYR, AX_ATT = slice(0, 3), slice(3, 6), slice(6, 9)


def _rand_rotations(n, max_deg, rng):
    """[n,3,3] rotation matrices: random axis, angle ~ U(-max_deg, max_deg)."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    K = np.zeros((n, 3, 3), dtype=np.float32)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = np.eye(3, dtype=np.float32)[None]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return (I + s * K + (1 - c) * (K @ K)).astype(np.float32)


def _time_warp(X, strength, rng):
    """Smooth monotonic time-warp per sample (piecewise-linear, 4 inner knots)."""
    N, C, T = X.shape
    k = 4
    knots_src = np.linspace(0, 1, k + 2)
    knots_dst = np.tile(knots_src, (N, 1))
    knots_dst[:, 1:-1] += rng.uniform(-strength, strength, size=(N, k)) / (k + 1)
    knots_dst = np.sort(np.clip(knots_dst, 0, 1), axis=1)
    t = np.linspace(0, 1, T)
    out = np.empty_like(X)
    for i in range(N):  # np.interp per sample for the warp curve, gather is vectorized
        src_pos = np.interp(t, knots_src, knots_dst[i]) * (T - 1)
        lo = np.floor(src_pos).astype(np.int64)
        hi = np.minimum(lo + 1, T - 1)
        w = (src_pos - lo).astype(np.float32)
        out[i] = X[i, :, lo].T * (1 - w) + X[i, :, hi].T * w
    return out


def augment(X, args, rng):
    """Return an augmented copy of raw windows X [N,9,100]."""
    X = X.copy()
    N = len(X)

    def mask(p=0.5):
        return rng.random(N) < p

    # 1. Rotation perturbation — same R on accel+gyro vectors; attitude channels
    #    get small constant offsets (approximation: a slightly rotated sensor
    #    frame shifts absolute orientation ~constantly over a 1 s window).
    if args.aug_rot > 0:
        m = mask()
        if m.any():
            R = _rand_rotations(m.sum(), args.aug_rot, rng)
            X[m, AX_ACC] = np.einsum("nij,njt->nit", R, X[m, AX_ACC])
            X[m, AX_GYR] = np.einsum("nij,njt->nit", R, X[m, AX_GYR])
            X[m, AX_ATT] += np.deg2rad(
                rng.uniform(-args.aug_rot, args.aug_rot, size=(m.sum(), 3))
            )[:, :, None].astype(np.float32)

    # 2. Magnitude scale — swing energy varies player-to-player. Angles don't scale.
    if args.aug_scale > 0:
        m = mask()
        s = rng.uniform(1 - args.aug_scale, 1 + args.aug_scale, size=(m.sum(), 1, 1)).astype(np.float32)
        X[m, AX_ACC] *= s
        X[m, AX_GYR] *= s

    # 3. Time-warp — faster/slower swings.
    if args.aug_warp > 0:
        m = mask()
        if m.any():
            X[m] = _time_warp(X[m], args.aug_warp, rng)

    # 4. Time-shift — imperfect window anchoring (edge-replicated).
    if args.aug_shift > 0:
        m = np.where(mask())[0]
        shifts = rng.integers(-args.aug_shift, args.aug_shift + 1, size=len(m))
        for k_ in np.unique(shifts):
            if k_ == 0:
                continue
            idx = m[shifts == k_]
            rolled = np.roll(X[idx], k_, axis=2)
            if k_ > 0:
                rolled[:, :, :k_] = rolled[:, :, k_:k_ + 1]
            else:
                rolled[:, :, k_:] = rolled[:, :, k_ - 1:k_]
            X[idx] = rolled

    # 5. Jitter — sensor noise, σ relative to per-channel train std.
    if args.aug_jitter > 0:
        ch_sd = X.std(axis=(0, 2), keepdims=True)
        X += (rng.normal(size=X.shape) * ch_sd * args.aug_jitter).astype(np.float32)

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # Augmentation (off by default → byte-identical baseline). --aug enables the
    # recommended set; individual strengths tunable, 0 disables a transform.
    ap.add_argument("--aug", action="store_true", help="enable the default augmentation set")
    ap.add_argument("--aug-rot", type=float, default=None, help="max rotation deg (default 15 with --aug)")
    ap.add_argument("--aug-scale", type=float, default=None, help="magnitude scale ± (default 0.2)")
    ap.add_argument("--aug-warp", type=float, default=None, help="time-warp strength (default 0.1)")
    ap.add_argument("--aug-shift", type=int, default=None, help="max time-shift samples (default 5)")
    ap.add_argument("--aug-jitter", type=float, default=None, help="noise σ × channel std (default 0.03)")
    args = ap.parse_args()

    # Resolve augmentation strengths: explicit flag wins; --aug supplies defaults.
    defaults = {"aug_rot": 15.0, "aug_scale": 0.2, "aug_warp": 0.1, "aug_shift": 5, "aug_jitter": 0.03}
    for k, dv in defaults.items():
        v = getattr(args, k)
        setattr(args, k, (dv if args.aug else 0) if v is None else v)
    aug_on = any(getattr(args, k) > 0 for k in defaults)

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    Xtr_raw, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("validation")
    Xte, yte, fte = load_split("test")
    print(f"train {Xtr_raw.shape}  val {Xva.shape}  test {Xte.shape}  dev={dev}")
    if aug_on:
        print(f"augmentation ON  rot±{args.aug_rot}°  scale±{args.aug_scale}  "
              f"warp {args.aug_warp}  shift±{args.aug_shift}  jitter {args.aug_jitter}")

    # Per-channel standardization from TRAIN stats (over samples+time).
    mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
    sd = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: (X - mu) / sd

    def T(X): return torch.tensor(X, dtype=torch.float32)
    ytr_t = torch.tensor(ytr).to(dev)
    Xva_t = T(norm(Xva)).to(dev); Xte_t = T(norm(Xte)).to(dev)
    # Baseline path: normalize once. Aug path: refreshed every epoch below.
    Xtr_t = T(norm(Xtr_raw)).to(dev)
    aug_rng = np.random.default_rng(args.seed)

    # Class-weighted loss (smash/vibora are rare).
    counts = Counter(ytr.tolist())
    w = torch.tensor([1.0 / counts[i] for i in range(len(CLASSES))], dtype=torch.float32)
    w = (w / w.sum() * len(CLASSES)).to(dev)

    class CNN(nn.Module):
        def __init__(s, ch=9, n=len(CLASSES)):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n),
            )
        def forward(s, x): return s.net(x)

    model = CNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    n = len(Xtr_t)
    best_va, best_state = -1, None
    for ep in range(args.epochs):
        if aug_on:
            # Fresh augmented view of the raw train set each epoch.
            Xtr_t = T(norm(augment(Xtr_raw, args, aug_rng))).to(dev)
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(Xva_t).argmax(1).cpu().numpy() == yva).mean()
        if va > best_va:
            best_va = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val_acc {va*100:.1f}%  (best {best_va*100:.1f}%)")

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()

    acc = (pred == yte).mean()
    print(f"\n{'='*54}\nHELD-OUT TEST (same split as RF):  CNN {acc*100:.1f}%   (RF v4 = 37.5%)\n{'='*54}")
    print(f"{'class':<18}{'recall':>10}")
    for ci, c in enumerate(CLASSES):
        m = yte == ci
        r = (pred[m] == yte[m]).mean() * 100 if m.sum() else 0
        print(f"{c:<18}{int((pred[m]==yte[m]).sum()):>4}/{int(m.sum()):<4}={r:>3.0f}%")
    # per player
    players = np.array([WF2PLAYER.get(f, "?") for f in fte])
    print("\nper-player:")
    for pid in sorted(set(players)):
        m = players == pid
        print(f"  {ID2NAME.get(pid, pid):10} n={m.sum():4}  acc={(pred[m]==yte[m]).mean()*100:.0f}%")

    out = ML_DIR / "models" / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    fname = "cnn_aug.pt" if aug_on else "cnn.pt"
    aug_cfg = {k: getattr(args, k) for k in ("aug_rot", "aug_scale", "aug_warp", "aug_shift", "aug_jitter")}
    torch.save({"state": best_state, "mu": mu, "sd": sd, "classes": CLASSES, "aug": aug_cfg if aug_on else None},
               out / fname)
    print(f"\nsaved {out/fname}  (val {best_va*100:.1f}%)")
    if aug_on:
        print("GATE: compare held-out test above vs the no-aug baseline (cnn.pt) and RF v4 37.5% —"
              " ship-worthy only if player-independent test improves, esp. Guillermo (LH).")


if __name__ == "__main__":
    main()
