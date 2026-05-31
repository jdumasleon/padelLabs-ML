#!/usr/bin/env python3
"""
train_cnn.py — temporal 1D-CNN baseline for the stroke classifier
==================================================================
Trains a small 1D convolutional net on the RAW 100x9 stroke windows (not the 90
hand-crafted RF features), so the model can learn the motion *shape* that
separates lobs/drives/overheads. Apples-to-apples vs the RF: same train/val/test
player-independent split.

Run in the padel-ml-eval venv (Python 3.12 + torch).

  ../padel-ml-eval/bin/python train_cnn.py --epochs 80
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("validation")
    Xte, yte, fte = load_split("test")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}  dev={dev}")

    # Per-channel standardization from TRAIN stats (over samples+time).
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: (X - mu) / sd
    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)

    def T(X): return torch.tensor(X, dtype=torch.float32)
    Xtr_t, ytr_t = T(Xtr).to(dev), torch.tensor(ytr).to(dev)
    Xva_t = T(Xva).to(dev); Xte_t = T(Xte).to(dev)

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
    torch.save({"state": best_state, "mu": mu, "sd": sd, "classes": CLASSES}, out / "cnn.pt")
    print(f"\nsaved {out/'cnn.pt'}  (val {best_va*100:.1f}%)")


if __name__ == "__main__":
    main()
