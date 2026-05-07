"""Compute features for a checkpoint on full SMNIST, train linear and MLP heads,
report best test accuracy. Focused subset of heads_on_trained.py for speed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import load_smnist
from train_bptt import build_net


def compute_features(net, x, batch=64):
    cfg = net.cfg
    feats = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            xb = x[s : s + batch].unsqueeze(-1)
            _, info = net(xb, return_layers=True)
            last = info["layer_spikes"][-1]
            T = last.shape[1]
            tail = max(1, int(round(T * cfg.tail_fraction)))
            feats.append(last[:, T - tail :].mean(dim=1))
    return torch.cat(feats, dim=0)


def train_head(model, Ftr, ytr, Fte, yte, epochs, lr, batch, wd, label):
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
            te = float(model(Fte).argmax(dim=1).eq(yte).float().mean().item())
        if te > best:
            best = te
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  [{label}] ep {ep:3d} test={te:.4f} best={best:.4f}", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--cache_dir", type=str, default="results/feat_cache_eval")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()

    print(f"Loading {args.ckpt} ...", flush=True)
    net = build_net(args.arch, aux_head=True)
    ck = torch.load(args.ckpt, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    print(f"  ckpt epoch {ck.get('epoch', '?')}, test_acc {ck.get('test_acc', '?')}", flush=True)
    net.eval()

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "train.pt"
    test_cache = cache_dir / "test.pt"

    if train_cache.exists() and test_cache.exists():
        Ftr = torch.load(train_cache, weights_only=False)
        Fte = torch.load(test_cache, weights_only=False)
        print(f"  loaded cached features (train {Ftr.shape}, test {Fte.shape})", flush=True)
    else:
        print("Computing train features ...", flush=True)
        t0 = time.time()
        Ftr = compute_features(net, xtr)
        print(f"  done in {time.time() - t0:.1f}s. mean={Ftr.mean().item():.4f}, "
              f"zero_rate={(Ftr == 0).float().mean().item():.4f}", flush=True)
        torch.save(Ftr, train_cache)

        print("Computing test features ...", flush=True)
        t0 = time.time()
        Fte = compute_features(net, xte)
        print(f"  done in {time.time() - t0:.1f}s.", flush=True)
        torch.save(Fte, test_cache)

    n_classes = net.cfg.n_classes
    fdim = Ftr.shape[1]

    print(f"\n=== linear lr=3e-3, {args.epochs} ep ===", flush=True)
    linear = nn.Linear(fdim, n_classes)
    best_lin = train_head(linear, Ftr, ytr, Fte, yte, args.epochs, 3e-3, 128, 1e-4, "linear")
    print(f"  LINEAR best test = {best_lin:.4f}", flush=True)

    print(f"\n=== mlp h=512, no drop, {args.epochs} ep ===", flush=True)
    mlp = nn.Sequential(
        nn.Linear(fdim, 512), nn.ReLU(),
        nn.Linear(512, n_classes),
    )
    best_mlp = train_head(mlp, Ftr, ytr, Fte, yte, args.epochs, 3e-3, 128, 1e-4, "mlp512")
    print(f"  MLP h=512 best test = {best_mlp:.4f}", flush=True)

    print(f"\n=== SUMMARY for {args.ckpt} ===", flush=True)
    print(f"  feat dim = {fdim}, mean = {Ftr.mean().item():.4f}, zero_rate = {(Ftr == 0).float().mean().item():.4f}",
          flush=True)
    print(f"  linear best = {best_lin:.4f}", flush=True)
    print(f"  mlp512 best = {best_mlp:.4f}", flush=True)


if __name__ == "__main__":
    main()
