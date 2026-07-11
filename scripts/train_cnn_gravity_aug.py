#!/usr/bin/env python3
"""
train_cnn_gravity_aug.py — gravity-relative CNN + IMU augmentation combo
=========================================================================
Stacks the two levers that individually beat the raw-axis CNN baseline
(test 8.6%): the heading-invariant gravity-relative channels from
prototype_gravity.py (test 26.7%) and the augmentation set from
train_cnn.py --aug (test 16.3%).

Augmentation nuance: a consistent device-frame rotation is a NO-OP in the
gravity-relative representation (all channels are dot products / norms
against g_hat, which are rotation-invariant — that invariance is the point
of the transform). So instead of rotating vectors, we perturb the estimated
gravity direction itself (small random tilt on g_hat before decomposition),
which models real CoreMotion attitude drift. Time-warp and time-shift run in
raw space; scale and jitter run on the transformed channels.

Run in the padel-ml-eval venv:

  ../padel-ml-eval/bin/python train_cnn_gravity_aug.py --epochs 160          # gravity only
  ../padel-ml-eval/bin/python train_cnn_gravity_aug.py --epochs 160 --aug    # the combo
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse corpus map + augmentation primitives from the sibling scripts.
from train_cnn import _rand_rotations, _time_warp
from prototype_gravity import euler_to_R, LEFTIES, ID2NAME

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
STROKES = ML_DIR / "labeled-strokes"
META = pd.read_csv(STROKES / "metadata.csv")
WF2PLAYER = {Path(r.window_file).name: str(r.player_id)[:8] for _, r in META.iterrows()}

RAW = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]
CLASSES = ["backhand", "backhand_lob", "backhand_volley", "bandeja", "forehand",
           "forehand_lob", "forehand_volley", "smash", "vibora"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Gravity-relative channel layout (see transform below):
#   0 a_vert  1 a_hmag  2 a_mag  3 w_vert  4 w_hmag  5 w_mag  6 pitch  7 cos(pitch)
G_CHANNELS = 8
SCALABLE = slice(0, 6)   # motion-energy channels; pitch/cos(pitch) are angles


def load_split_raw(split):
    """Raw windows [N,9,100] (time last, like train_cnn) + lefty mask."""
    Xs, ys, fs, lf = [], [], [], []
    for cls in CLASSES:
        d = STROKES / split / cls
        if not d.exists():
            continue
        for csv in sorted(d.glob("*.csv")):
            df = pd.read_csv(csv)
            if len(df) != 100 or any(c not in df.columns for c in RAW):
                continue
            arr = df[RAW].to_numpy(np.float64)
            if not np.isfinite(arr).all():
                continue
            Xs.append(arr.T)                       # [9,100]
            ys.append(CLS2IDX[cls])
            fs.append(csv.name)
            lf.append(WF2PLAYER.get(csv.name, "") in LEFTIES)
    return np.stack(Xs), np.array(ys), fs, np.array(lf)


def gravity_transform(X, lefty, g_tilt_deg=0.0, rng=None):
    """Batched raw [N,9,100] -> gravity-relative [N,8,100].

    g_tilt_deg > 0 perturbs the estimated gravity direction per sample
    (random tilt up to that angle) — models attitude-estimation drift.
    """
    N = len(X)
    accel = np.transpose(X[:, 0:3], (0, 2, 1))     # [N,100,3]
    gyro = np.transpose(X[:, 3:6], (0, 2, 1))
    roll, pitch, yaw = X[:, 6], X[:, 7], X[:, 8]    # [N,100]

    R = euler_to_R(roll, pitch, yaw)                # [N,100,3,3] body->world
    g_world = np.array([0.0, 0.0, -1.0])
    g_dev = np.einsum("ntij,i->ntj", R, g_world)    # R^T · g == einsum over rows
    g_hat = g_dev / (np.linalg.norm(g_dev, axis=2, keepdims=True) + 1e-9)

    if g_tilt_deg > 0 and rng is not None:
        # One small random rotation per SAMPLE (constant over the window):
        # drift is slow relative to a 1 s swing.
        Rp = _rand_rotations(N, g_tilt_deg, rng).astype(np.float64)   # [N,3,3]
        g_hat = np.einsum("nij,ntj->nti", Rp, g_hat)
        g_hat /= np.linalg.norm(g_hat, axis=2, keepdims=True) + 1e-9

    def decomp(v):
        vert = np.sum(v * g_hat, axis=2)                              # [N,100]
        hvec = v - vert[..., None] * g_hat
        return vert, np.linalg.norm(hvec, axis=2), np.linalg.norm(v, axis=2)

    a_vert, a_hmag, a_mag = decomp(accel)
    w_vert, w_hmag, w_mag = decomp(gyro)
    w_vert = np.where(lefty[:, None], -w_vert, w_vert)  # canonicalise handedness
    out = np.stack([a_vert, a_hmag, a_mag, w_vert, w_hmag, w_mag, pitch, np.cos(pitch)], axis=1)
    return out.astype(np.float32)                        # [N,8,100]


def augment_epoch(Xraw, lefty, args, rng):
    """Raw-space warp/shift -> gravity transform (perturbed g) -> scale+jitter."""
    X = Xraw.copy()
    N = len(X)

    def mask(p=0.5):
        return rng.random(N) < p

    if args.aug_warp > 0:
        m = mask()
        if m.any():
            X[m] = _time_warp(X[m], args.aug_warp, rng)

    if args.aug_shift > 0:
        m = np.where(mask())[0]
        shifts = rng.integers(-args.aug_shift, args.aug_shift + 1, size=len(m))
        for k in np.unique(shifts):
            if k == 0:
                continue
            idx = m[shifts == k]
            rolled = np.roll(X[idx], k, axis=2)
            if k > 0:
                rolled[:, :, :k] = rolled[:, :, k:k + 1]
            else:
                rolled[:, :, k:] = rolled[:, :, k - 1:k]
            X[idx] = rolled

    G = gravity_transform(X, lefty, g_tilt_deg=args.aug_gtilt, rng=rng)

    if args.aug_scale > 0:
        m = mask()
        s = rng.uniform(1 - args.aug_scale, 1 + args.aug_scale, size=(int(m.sum()), 1, 1)).astype(np.float32)
        G[m, SCALABLE] *= s

    if args.aug_jitter > 0:
        ch_sd = G.std(axis=(0, 2), keepdims=True)
        G += (rng.normal(size=G.shape) * ch_sd * args.aug_jitter).astype(np.float32)

    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", action="store_true", help="enable the augmentation combo")
    ap.add_argument("--aug-gtilt", type=float, default=None, help="max gravity-tilt deg (default 7)")
    ap.add_argument("--aug-scale", type=float, default=None, help="magnitude scale ± (default 0.2)")
    ap.add_argument("--aug-warp", type=float, default=None, help="time-warp strength (default 0.1)")
    ap.add_argument("--aug-shift", type=int, default=None, help="max time-shift samples (default 5)")
    ap.add_argument("--aug-jitter", type=float, default=None, help="noise σ × channel std (default 0.03)")
    args = ap.parse_args()

    defaults = {"aug_gtilt": 7.0, "aug_scale": 0.2, "aug_warp": 0.1, "aug_shift": 5, "aug_jitter": 0.03}
    for k, dv in defaults.items():
        v = getattr(args, k)
        setattr(args, k, (dv if args.aug else 0) if v is None else v)
    aug_on = any(getattr(args, k) > 0 for k in defaults)

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    Xtr_raw, ytr, _, ltr = load_split_raw("train")
    Xva_raw, yva, _, lva = load_split_raw("validation")
    Xte_raw, yte, fte, lte = load_split_raw("test")
    print(f"train {Xtr_raw.shape}  val {Xva_raw.shape}  test {Xte_raw.shape}  dev={dev}")
    if aug_on:
        print(f"augmentation ON  g-tilt±{args.aug_gtilt}°  scale±{args.aug_scale}  "
              f"warp {args.aug_warp}  shift±{args.aug_shift}  jitter {args.aug_jitter}")

    # Clean gravity-relative transforms; train stats come from the clean train set.
    Gtr = gravity_transform(Xtr_raw, ltr)
    Gva = gravity_transform(Xva_raw, lva)
    Gte = gravity_transform(Xte_raw, lte)
    mu = Gtr.mean(axis=(0, 2), keepdims=True)
    sd = Gtr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda G: (G - mu) / sd

    def T(G): return torch.tensor(G, dtype=torch.float32)
    ytr_t = torch.tensor(ytr).to(dev)
    Xva_t = T(norm(Gva)).to(dev); Xte_t = T(norm(Gte)).to(dev)
    Xtr_t = T(norm(Gtr)).to(dev)
    aug_rng = np.random.default_rng(args.seed)

    counts = Counter(ytr.tolist())
    w = torch.tensor([1.0 / counts[i] for i in range(len(CLASSES))], dtype=torch.float32)
    w = (w / w.sum() * len(CLASSES)).to(dev)

    class CNN(nn.Module):
        def __init__(s, ch=G_CHANNELS, n=len(CLASSES)):
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
            Xtr_t = T(norm(augment_epoch(Xtr_raw, ltr, args, aug_rng))).to(dev)
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
    tag = "GRAVITY+AUG" if aug_on else "GRAVITY-ONLY"
    print(f"\n{'='*60}\nHELD-OUT TEST ({tag}):  CNN {acc*100:.1f}%")
    print("(raw CNN 8.6% | raw CNN+aug 16.3% | gravity-only@80ep 26.7% | RF v4 37.5%)")
    print("=" * 60)
    print(f"{'class':<18}{'recall':>10}")
    for ci, c in enumerate(CLASSES):
        m = yte == ci
        r = (pred[m] == yte[m]).mean() * 100 if m.sum() else 0
        print(f"{c:<18}{int((pred[m]==yte[m]).sum()):>4}/{int(m.sum()):<4}={r:>3.0f}%")
    players = np.array([WF2PLAYER.get(f, "?") for f in fte])
    print("\nper-player:")
    for pid in sorted(set(players)):
        m = players == pid
        print(f"  {ID2NAME.get(pid, pid):10} n={m.sum():4}  acc={(pred[m]==yte[m]).mean()*100:.0f}%")

    out = ML_DIR / "models" / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    fname = "cnn_gravity_aug.pt" if aug_on else "cnn_gravity.pt"
    aug_cfg = {k: getattr(args, k) for k in defaults}
    import torch as _t
    _t.save({"state": best_state, "mu": mu, "sd": sd, "classes": CLASSES,
             "channels": "gravity-relative-8", "aug": aug_cfg if aug_on else None}, out / fname)
    print(f"\nsaved {out/fname}  (val {best_va*100:.1f}%)")
    print("GATE: ship-worthy only if player-independent held-out beats v5 RF, esp. the left-hander.")


if __name__ == "__main__":
    main()
