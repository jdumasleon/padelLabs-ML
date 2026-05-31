#!/usr/bin/env python3
"""
contact_segment.py — contact-aligned re-segmentation + temporal CNN test
=========================================================================
Re-extracts stroke windows from the RAW DataCollection sessions, anchored on the
ACTUAL ball-contact (sharp impact) instead of the early spike marker. Contact is
detected as the peak of high-frequency jerk energy in a forward search window from
the marker (the impact is a brief high-frequency burst; the backswing is smooth and
lower-frequency). Then trains the same 1D-CNN and evaluates on the held-out test,
to see whether clean alignment lets the temporal model beat the order-invariant RF.

Read-only w.r.t. the corpus (builds windows in memory; writes nothing under
labeled-strokes/). Run in padel-ml-eval venv.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/jldumas/Jo/PadelLabs")
DC = ROOT / "DataCollection"
ML = ROOT / "PadelLabs-ML"
sys.path.insert(0, str(ML / "scripts"))

RAW = ["accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ", "roll", "pitch", "yaw"]
CLASSES = ["backhand", "backhand_lob", "backhand_volley", "bandeja", "forehand",
           "forehand_lob", "forehand_volley", "smash", "vibora"]
C2I = {c: i for i, c in enumerate(CLASSES)}
VAL = {"14A0FF96-6B40-423D-ADB2-D94E8AD96EF6", "1DC25578-923C-4EFE-9717-237E45B71EDE"}   # Victor, Carlos
TEST = {"1880202A-848B-444A-886C-B771B2D18998", "5E354AD0-50F2-4EC7-B653-1FB9989DC793"}  # Guillermo, Diego
LEFTIES = {"1880202A-848B-444A-886C-B771B2D18998"}
ID2N = {"5E354AD0": "Diego", "1880202A": "Guillermo*LH"}
BURST_GAP = 1.2
PRE, POST = 30, 70

def bursts(markers):
    v = sorted([m for m in markers if float(m.get("timestamp", -1)) >= 0], key=lambda m: float(m["timestamp"]))
    out, cur = [], []
    for m in v:
        ts = float(m["timestamp"])
        if not cur: cur = [m]
        elif ts - float(cur[0]["timestamp"]) <= BURST_GAP: cur.append(m)
        else: out.append(cur); cur = [m]
    if cur: out.append(cur)
    return out

def winner(b):
    for m in b:
        s = str(m.get("strokeType", "")).strip()
        if s and s != "unknown": return s
    return "unknown"

def contact_index(df, anchor_ts, mode):
    """Index of ball contact: peak high-frequency jerk in [anchor, anchor+0.8s]."""
    ts = df["timestamp"].to_numpy()
    coarse = int(np.abs(ts - anchor_ts).argmin())
    a = df[["accelX", "accelY", "accelZ"]].to_numpy()
    mag = np.sqrt((a ** 2).sum(1))
    if mode == "accelpeak":
        s, e = max(1, coarse - 5), min(len(df), coarse + 90)
        return s + int(mag[s:e].argmax())
    # jerk = |d accel/dt| (central diff of the 3-axis vector), high at sharp impact
    jerk = np.zeros(len(df))
    jerk[1:-1] = np.linalg.norm(a[2:] - a[:-2], axis=1)
    s, e = max(1, coarse - 5), min(len(df) - 1, coarse + 90)   # forward search ~0.85s
    if e <= s: return None
    return s + int(jerk[s:e].argmax())

def extract(mode):
    Xtr, ytr, Xva, yva, Xte, yte, fte = [], [], [], [], [], [], []
    audit_pos = defaultdict(list)
    for csv in sorted(DC.rglob("*.csv")):
        if any(t in csv.name for t in ("_classified", "_validation")): continue
        jp = csv.with_suffix(".json")
        if not jp.exists(): continue
        try:
            df = pd.read_csv(csv); meta = json.load(open(jp))
        except Exception: continue
        if any(c not in df.columns for c in RAW): continue
        pid = str(meta.get("playerId", "")); lefty = pid in LEFTIES
        split = "test" if pid in TEST else ("validation" if pid in VAL else "train")
        for b in bursts(meta.get("markers", [])):
            st = winner(b)
            if st not in C2I: continue
            ci = contact_index(df, float(b[0]["timestamp"]), mode)
            if ci is None: continue
            s, e = ci - PRE, ci + POST
            if s < 0 or e > len(df): continue
            w = df[RAW].to_numpy(float)[s:e]
            if w.shape[0] != 100 or not np.isfinite(w).all(): continue
            # audit: where does max-accel peak now sit?
            mag = np.sqrt((w[:, :3] ** 2).sum(1)); audit_pos[split].append(int(mag.argmax()))
            arr = w.T.astype(np.float32)
            if split == "train": Xtr.append(arr); ytr.append(C2I[st])
            elif split == "validation": Xva.append(arr); yva.append(C2I[st])
            else: Xte.append(arr); yte.append(C2I[st]); fte.append(pid[:8])
    return (np.stack(Xtr), np.array(ytr), np.stack(Xva), np.array(yva),
            np.stack(Xte), np.array(yte), fte, audit_pos)

def evaluate(mode):
    import torch, torch.nn as nn
    torch.manual_seed(42); np.random.seed(42)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr, ytr, Xva, yva, Xte, yte, fte, ap = extract(mode)
    near = lambda L: 100 * np.mean([(25 <= p <= 35) for p in L])
    print(f"\n[{mode}] windows: train {len(ytr)} val {len(yva)} test {len(yte)} | "
          f"contact@30±5: train {near(ap['train']):.0f}% test {near(ap['test']):.0f}%")
    mu = Xtr.mean((0, 2), keepdims=True); sd = Xtr.std((0, 2), keepdims=True) + 1e-6
    f = lambda X: ((X - mu) / sd).astype("float32")
    Xtr, Xva, Xte = f(Xtr), f(Xva), f(Xte)
    t = lambda X: torch.tensor(X).to(dev)
    Xtr_t, ytr_t, Xva_t, Xte_t = t(Xtr), torch.tensor(ytr).to(dev), t(Xva), t(Xte)
    cnt = Counter(ytr.tolist()); w = torch.tensor([1 / cnt[i] for i in range(9)], dtype=torch.float32)
    w = (w / w.sum() * 9).to(dev)
    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(9, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 9))
        def forward(s, x): return s.net(x)
    m = CNN().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w); n = len(Xtr_t); best = -1; bs = None
    for ep in range(80):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]; opt.zero_grad(); lossf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): va = (m(Xva_t).argmax(1).cpu().numpy() == yva).mean()
        if va > best: best = va; bs = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): pred = m(Xte_t).argmax(1).cpu().numpy()
    pl = np.array(fte)
    print(f"  CNN val={best*100:.1f}%  held-out test={(pred==yte).mean()*100:.1f}%   (RF v4=37.5%, raw-CNN=8.6%)")
    for pid in sorted(set(pl)):
        mm = pl == pid; print(f"    {ID2N.get(pid,pid):12} acc={(pred[mm]==yte[mm]).mean()*100:.0f}%")
    print("    lobs:", " ".join(f"{CLASSES[ci]}={ (pred[yte==ci]==yte[yte==ci]).mean()*100:.0f}%"
          for ci in [C2I['forehand_lob'], C2I['backhand_lob'], C2I['vibora']]))

if __name__ == "__main__":
    print("baseline: current extraction had contact@30±5 ~38-60%, peak mean ~45")
    evaluate("accelpeak")   # anchor on biggest accel peak (contact ~ impact)
    evaluate("jerk")        # anchor on high-frequency jerk peak
