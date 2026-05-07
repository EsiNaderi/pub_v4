"""Push a trained checkpoint as far as we can with offline readouts.

1) Compute features once on full SMNIST (60k+10k), cache to disk.
2) Try multiple heads on the cached features:
   - linear (3 LRs)
   - 2-layer MLP (h=128, h=512) with dropout
   - ridge regression (closed-form)
   - logistic regression with L2 (sklearn)

Compare results against the random-init baseline computed earlier
(0.5669) and the linear-head trained-features baseline (0.6524).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import load_smnist
from train_bptt import build_net


def compute_or_load_features(net, x, cache_path: Path, label: str, batch=64):
    if cache_path.exists():
        d = torch.load(cache_path, weights_only=False)
        print(f"  loaded cached {label} features from {cache_path}: shape={tuple(d['feats'].shape)}",
              flush=True)
        return d["feats"]
    cfg = net.cfg
    feats = []
    print(f"  computing {label} features ({x.shape[0]} samples) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            xb = x[s : s + batch].unsqueeze(-1)
            _, info = net(xb, return_layers=True)
            last = info["layer_spikes"][-1]
            T = last.shape[1]
            tail = max(1, int(round(T * cfg.tail_fraction)))
            feats.append(last[:, T - tail :].mean(dim=1))
    F_ = torch.cat(feats, dim=0)
    print(f"    done in {time.time() - t0:.1f}s. mean={F_.mean().item():.4f}, "
          f"std={F_.std().item():.4f}, zero_rate={(F_ == 0).float().mean().item():.4f}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": F_}, cache_path)
    print(f"  cached at {cache_path}", flush=True)
    return F_


def train_head(model, Ftr, ytr, Fte, yte, n_classes, epochs, lr, batch, wd, label, csv_writer):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = Ftr.shape[0]
    best = 0.0
    for ep in range(epochs):
        order = torch.randperm(n)
        model.train()
        for s in range(0, n, batch):
            idx = order[s : s + batch]
            logits = model(Ftr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = float(model(Ftr).argmax(dim=1).eq(ytr).float().mean().item())
            te = float(model(Fte).argmax(dim=1).eq(yte).float().mean().item())
        if te > best:
            best = te
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [{label}] ep {ep:3d} train={tr:.4f} test={te:.4f} best={best:.4f}",
                  flush=True)
    if csv_writer:
        csv_writer.writerow([label, lr, wd, batch, epochs, best])
    return best


def ridge_classify(Ftr, ytr, Fte, yte, n_classes, alpha=1.0):
    n = Ftr.shape[0]
    Y = F.one_hot(ytr, n_classes).float()
    A = Ftr.T @ Ftr + alpha * torch.eye(Ftr.shape[1])
    B = Ftr.T @ Y
    W = torch.linalg.solve(A, B)
    pred_te = (Fte @ W).argmax(dim=1)
    pred_tr = (Ftr @ W).argmax(dim=1)
    return float(pred_tr.eq(ytr).float().mean().item()), float(pred_te.eq(yte).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="results/ckpt_overnight_default_10k.pt")
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--cache_dir", type=str, default="results/feat_cache")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--csv", type=str, default="results/run_heads_on_trained.csv")
    args = p.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["label", "lr", "wd", "batch", "epochs", "best_test"])

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()

    print("Loading checkpoint ...", flush=True)
    net = build_net(args.arch, aux_head=True)
    ck = torch.load(args.ckpt, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    cache_dir = Path(args.cache_dir)
    Ftr = compute_or_load_features(net, xtr, cache_dir / "trained_train.pt", "train")
    Fte = compute_or_load_features(net, xte, cache_dir / "trained_test.pt", "test")

    n_classes = net.cfg.n_classes
    fdim = Ftr.shape[1]
    print(f"\nFeature dim = {fdim}, n_classes = {n_classes}", flush=True)

    # 1) Linear head with multiple LRs (we already know lr=3e-3 → 0.6524)
    for lr in [1e-3, 3e-3, 1e-2]:
        print(f"\n=== linear lr={lr} ===", flush=True)
        head = nn.Linear(fdim, n_classes)
        train_head(head, Ftr, ytr, Fte, yte, n_classes,
                   epochs=args.epochs, lr=lr, batch=128, wd=1e-4,
                   label=f"linear_lr{lr}", csv_writer=writer)
        f_csv.flush()

    # 2) MLP heads
    for h in [128, 512]:
        for drop in [0.0, 0.3]:
            print(f"\n=== mlp h={h} drop={drop} ===", flush=True)
            mlp = nn.Sequential(
                nn.Linear(fdim, h), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h, n_classes),
            )
            train_head(mlp, Ftr, ytr, Fte, yte, n_classes,
                       epochs=args.epochs, lr=3e-3, batch=128, wd=1e-4,
                       label=f"mlp_h{h}_drop{drop}", csv_writer=writer)
            f_csv.flush()

    # 3) Ridge regression
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        tr, te = ridge_classify(Ftr, ytr, Fte, yte, n_classes, alpha=alpha)
        print(f"\n=== ridge alpha={alpha}: train={tr:.4f} test={te:.4f} ===", flush=True)
        writer.writerow([f"ridge_alpha{alpha}", "", "", "", "closed-form", te])
        f_csv.flush()

    f_csv.close()
    print("\n=== Done. CSV at:", args.csv, "===", flush=True)


if __name__ == "__main__":
    main()
