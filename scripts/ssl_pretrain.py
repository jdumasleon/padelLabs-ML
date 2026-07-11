#!/usr/bin/env python3
"""
ssl_pretrain.py — self-supervised pretraining on unlabeled session IMU
======================================================================
The corpus has ~6.4k labeled windows but the raw DataCollection sessions hold
millions of unlabeled samples. This pretrains the CNN trunk with masked
reconstruction (mask random time spans, reconstruct them) on sliding windows
harvested from TRAIN players only (val/test players excluded — no leakage),
then the encoder initialises the supervised classifier via
`train_cnn_gravity_aug.py --init <encoder.pt>`.

Stages (each cached, rerun is cheap):
  1. harvest   raw CSVs -> ssl-cache/unlabeled_raw.npy   [N,9,100] float32
  2. pretrain  masked-reconstruction -> models/cnn/ssl_encoder_<rep>.pt

Run in the padel-ml-eval venv:

  ../padel-ml-eval/bin/python ssl_pretrain.py                 # harvest + pretrain (gravity rep)
  ../padel-ml-eval/bin/python ssl_pretrain.py --rep raw       # raw 9-channel rep
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_cnn_gravity_aug import gravity_transform, G_CHANNELS

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent
DATA = ML_DIR.parent / "DataCollection"
CACHE = ML_DIR / "ssl-cache"

RAW_COLS = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]

# v4 split: train players only. Folder names are ground truth (CLAUDE.md §8.0);
# note the trailing spaces on some folders. Val (Victor, Carlos) and test
# (Diego, Guillermo) are excluded so the held-out evaluation stays honest.
TRAIN_FOLDERS = ["Jose", "Ela", "Enrique ", "Irene", "Javier", "Jorge"]

WIN, STRIDE = 100, 50


def harvest(min_activity: float, cap_per_session: int, rng) -> np.ndarray:
    """Sliding windows over every raw train-player session CSV, filtered to
    windows with real motion (std of |accel| above threshold)."""
    out = []
    for folder in TRAIN_FOLDERS:
        base = DATA / folder
        if not base.exists():
            print(f"  WARN missing folder: {folder!r}")
            continue
        csvs = [p for p in base.rglob("*.csv")
                if "classified" not in p.name and "validation" not in p.name]
        for p in sorted(csvs):
            try:
                df = pd.read_csv(p, usecols=RAW_COLS, dtype=np.float32)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            arr = df.to_numpy()
            if len(arr) < WIN or not np.isfinite(arr).all():
                arr = arr[np.isfinite(arr).all(axis=1)]
                if len(arr) < WIN:
                    continue
            n_win = 1 + (len(arr) - WIN) // STRIDE
            starts = np.arange(n_win) * STRIDE
            # windows [n,100,9]
            w = np.stack([arr[s:s + WIN] for s in starts])
            # activity filter: std of accel magnitude over the window
            amag = np.linalg.norm(w[:, :, 0:3], axis=2)
            keep = amag.std(axis=1) >= min_activity
            w = w[keep]
            if len(w) > cap_per_session:
                w = w[rng.choice(len(w), cap_per_session, replace=False)]
            if len(w):
                out.append(np.transpose(w, (0, 2, 1)))   # [n,9,100]
        print(f"  {folder.strip():10} sessions={len(csvs):3}  windows so far={sum(len(o) for o in out)}")
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["gravity", "raw"], default="gravity")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-activity", type=float, default=0.08,
                    help="min std of |accel| per window (drops idle segments)")
    ap.add_argument("--cap-per-session", type=int, default=4000)
    ap.add_argument("--mask-spans", type=int, default=3)
    ap.add_argument("--mask-len", type=int, default=10)
    ap.add_argument("--reharvest", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ── 1. harvest (cached) ──────────────────────────────────────────
    CACHE.mkdir(exist_ok=True)
    cache_f = CACHE / "unlabeled_raw.npy"
    if cache_f.exists() and not args.reharvest:
        X = np.load(cache_f)
        print(f"loaded cache {cache_f.name}: {X.shape}")
    else:
        print("harvesting unlabeled windows (train players only)…")
        X = harvest(args.min_activity, args.cap_per_session, rng)
        np.save(cache_f, X)
        print(f"harvested {X.shape} -> {cache_f}")

    # ── 2. representation ────────────────────────────────────────────
    if args.rep == "gravity":
        # All train players are right-handed -> lefty=False everywhere.
        X = gravity_transform(X.astype(np.float64), np.zeros(len(X), bool))
        C = G_CHANNELS
    else:
        C = 9
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    print(f"rep={args.rep}  X={X.shape}")

    # ── 3. masked-reconstruction pretraining ────────────────────────
    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Encoder = EXACT trunk of the supervised CNN (same Sequential indices
    # net.0 … net.10) so the state dict drops straight into the classifier.
    def make_trunk(ch):
        return nn.Sequential(
            nn.Conv1d(ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )

    class MaskedAE(nn.Module):
        def __init__(s, ch):
            super().__init__()
            s.net = make_trunk(ch)     # -> [128, 25]
            s.dec = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv1d(32, ch, 3, padding=1),
            )
        def forward(s, x):
            return s.dec(s.net(x))

    model = MaskedAE(C).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    Xt = torch.tensor(X)
    n = len(Xt)
    T = X.shape[2]
    print(f"pretraining on {n} windows  dev={dev}")

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch):
            xb = Xt[perm[i:i + args.batch]].to(dev)
            # mask random contiguous spans per sample
            m = torch.zeros_like(xb, dtype=torch.bool)
            for _ in range(args.mask_spans):
                starts = torch.randint(0, T - args.mask_len, (len(xb),))
                for j, st in enumerate(starts):
                    m[j, :, st:st + args.mask_len] = True
            xin = xb.masked_fill(m, 0.0)
            rec = model(xin)
            loss = ((rec - xb)[m] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  masked-MSE {tot/nb:.4f}")

    out = ML_DIR / "models" / "cnn" / f"ssl_encoder_{args.rep}.pt"
    torch.save({"trunk": model.net.state_dict(), "rep": args.rep, "channels": C,
                "n_unlabeled": n, "epochs": args.epochs,
                "mask": {"spans": args.mask_spans, "len": args.mask_len}}, out)
    print(f"saved encoder -> {out}")
    print(f"fine-tune:  ../padel-ml-eval/bin/python train_cnn_gravity_aug.py --epochs 80 --init {out.name}")


if __name__ == "__main__":
    main()
