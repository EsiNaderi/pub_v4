"""Per-class accuracy + confusion matrix on the BPTT-trained checkpoint
with the best-performing MLP h=512 head."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import load_smnist


def main():
    cache = Path("results/feat_cache")
    Ftr_d = torch.load(cache / "trained_train.pt", weights_only=False)
    Fte_d = torch.load(cache / "trained_test.pt", weights_only=False)
    Ftr = Ftr_d["feats"]
    Fte = Fte_d["feats"]

    _, ytr, _, yte = load_smnist()

    n_classes = 10
    fdim = Ftr.shape[1]

    # Train MLP h=512 head (the winner)
    torch.manual_seed(20260507)
    mlp = nn.Sequential(
        nn.Linear(fdim, 512), nn.ReLU(),
        nn.Linear(512, n_classes),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    n = Ftr.shape[0]
    best = 0.0
    best_state = None
    print(f"Training MLP h=512 on cached features ({Ftr.shape}) ...", flush=True)
    for ep in range(120):
        order = torch.randperm(n)
        mlp.train()
        for s in range(0, n, 128):
            idx = order[s : s + 128]
            logits = mlp(Ftr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            te = float(mlp(Fte).argmax(dim=1).eq(yte).float().mean().item())
        if te > best:
            best = te
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep {ep:3d} test={te:.4f} best={best:.4f}", flush=True)

    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Fte).argmax(dim=1)

    print(f"\n=== Best test acc: {best:.4f} ===\n", flush=True)

    print("Per-class accuracy:")
    print(f"{'Class':>5} {'N':>6} {'Correct':>8} {'Acc':>8}")
    for c in range(n_classes):
        mask = yte == c
        n_c = int(mask.sum().item())
        n_corr = int(((pred == c) & mask).sum().item())
        print(f"{c:>5} {n_c:>6} {n_corr:>8} {n_corr/max(n_c,1):>8.4f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(yte.tolist(), pred.tolist()):
        cm[t, p] += 1
    print("       " + "".join(f"{c:>5}" for c in range(n_classes)))
    for r in range(n_classes):
        row = cm[r].tolist()
        print(f"  {r:>3} | " + "".join(f"{v:>5}" for v in row))

    print("\nTop confusions (off-diagonal):")
    cm_off = cm.clone()
    for c in range(n_classes):
        cm_off[c, c] = 0
    top = []
    for r in range(n_classes):
        for c in range(n_classes):
            if r != c and cm_off[r, c] > 0:
                top.append((int(cm_off[r, c]), r, c))
    top.sort(reverse=True)
    for n_pairs, t, p in top[:10]:
        print(f"  {n_pairs} times: true {t} -> pred {p}")


if __name__ == "__main__":
    main()
