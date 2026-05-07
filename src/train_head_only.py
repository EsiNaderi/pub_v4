"""Train ONLY the readout head on top of a frozen HRN reservoir.

This is the classic reservoir-computing baseline: random recurrent
features + trained linear classifier. It establishes a lower bound on
what the architecture can do without hidden-layer plasticity.

Pipeline:
1) Forward all train samples through frozen HRN, cache last-layer tail-mean rates.
2) Train a linear classifier (or per-class pool readout) on cached features.
3) Forward all test samples, evaluate.

Caching avoids re-running the (expensive) forward each epoch.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, make_default_config
from train_bptt import build_net


def compute_tail_features(net: HierarchicalResonantNet, x: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """Forward x in batches, return (N, n_classes * out_pool_size) tail mean rates."""

    feats = []
    cfg = net.cfg
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            xb = x[s:s+batch].unsqueeze(-1)                                # (B, T, 1)
            _, info = net(xb, return_layers=True)
            last_seq = info["layer_spikes"][-1]                            # (B, T, N_out)
            T = last_seq.shape[1]
            tail = max(1, int(round(T * cfg.tail_fraction)))
            f = last_seq[:, T - tail:].mean(dim=1)                          # (B, N_out)
            feats.append(f)
    return torch.cat(feats, dim=0)


def per_class_pool_logits(features: torch.Tensor, n_classes: int, pool_size: int,
                          temperature: float, bias: torch.Tensor) -> torch.Tensor:
    """features: (B, n_classes * pool_size). Return per-class pool-rate logits."""

    B = features.shape[0]
    feats_p = features.view(B, n_classes, pool_size)
    pool_rate = feats_p.mean(dim=2)                                         # (B, n_classes)
    pool_rate_c = pool_rate - pool_rate.mean(dim=1, keepdim=True)
    return pool_rate_c * temperature + bias


def evaluate_head(features_tr, ytr, features_te, yte, n_classes, pool_size,
                   temperature, head_kind: str, lr: float, epochs: int, batch: int,
                   device: str, weight_decay: float = 0.0) -> dict:
    """head_kind: 'linear' or 'pool_rate'."""

    Ftr = features_tr.to(device); Fte = features_te.to(device)
    ytr_d = ytr.to(device); yte_d = yte.to(device)

    if head_kind == "linear":
        head = nn.Linear(Ftr.shape[1], n_classes).to(device)
        params = list(head.parameters())
    else:
        bias = nn.Parameter(torch.zeros(n_classes, device=device))
        head = None
        params = [bias]

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    n_train = Ftr.shape[0]
    best_test = 0.0
    log = []
    for ep in range(epochs):
        order = torch.randperm(n_train)
        t0 = time.time()
        for s in range(0, n_train, batch):
            idx = order[s:s+batch]
            xb = Ftr[idx]; yb = ytr_d[idx]
            if head_kind == "linear":
                logits = head(xb)
            else:
                logits = per_class_pool_logits(xb, n_classes, pool_size, temperature, bias)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # eval
        with torch.no_grad():
            if head_kind == "linear":
                pred_tr = head(Ftr).argmax(dim=1)
                pred_te = head(Fte).argmax(dim=1)
            else:
                pred_tr = per_class_pool_logits(Ftr, n_classes, pool_size, temperature, bias).argmax(dim=1)
                pred_te = per_class_pool_logits(Fte, n_classes, pool_size, temperature, bias).argmax(dim=1)
        train_acc = float((pred_tr == ytr_d).float().mean().item())
        test_acc = float((pred_te == yte_d).float().mean().item())
        elapsed = time.time() - t0
        log.append((ep, train_acc, test_acc, elapsed))
        if test_acc > best_test:
            best_test = test_acc
        print(f"  ep {ep:3d}  t={elapsed:5.1f}s  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}", flush=True)
    return {"best_test": best_test, "log": log}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--head_kind", type=str, choices=["linear", "pool_rate", "both"], default="both")
    p.add_argument("--head_epochs", type=int, default=100)
    p.add_argument("--head_lr", type=float, default=3e-3)
    p.add_argument("--head_batch", type=int, default=128)
    p.add_argument("--head_weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    print(f"loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]
    print(f"train: {xtr.shape[0]}, test: {xte.shape[0]}", flush=True)

    print(f"building net ...", flush=True)
    net = build_net(args.arch).to(device)
    net.eval()
    print(f"net params: {net.n_params()}", flush=True)

    print(f"computing train features ...", flush=True)
    t0 = time.time()
    Ftr = compute_tail_features(net, xtr.to(device), batch=64).cpu()
    print(f"  done in {time.time() - t0:.1f}s, shape: {Ftr.shape}", flush=True)
    print(f"  feature stats: mean={Ftr.mean().item():.4f}, std={Ftr.std().item():.4f}, zero_rate={(Ftr==0).float().mean().item():.4f}", flush=True)

    print(f"computing test features ...", flush=True)
    t0 = time.time()
    Fte = compute_tail_features(net, xte.to(device), batch=64).cpu()
    print(f"  done in {time.time() - t0:.1f}s, shape: {Fte.shape}", flush=True)

    cfg = net.cfg
    if args.head_kind in ("linear", "both"):
        print("\n=== Linear head ===", flush=True)
        res = evaluate_head(Ftr, ytr, Fte, yte, cfg.n_classes, cfg.out_pool_size,
                             cfg.readout_temperature, "linear",
                             args.head_lr, args.head_epochs, args.head_batch,
                             device, args.head_weight_decay)
        print(f"linear best test: {res['best_test']:.4f}", flush=True)
    if args.head_kind in ("pool_rate", "both"):
        print("\n=== Pool-rate readout (per-class pool, only bias trainable) ===", flush=True)
        res = evaluate_head(Ftr, ytr, Fte, yte, cfg.n_classes, cfg.out_pool_size,
                             cfg.readout_temperature, "pool_rate",
                             args.head_lr, args.head_epochs, args.head_batch,
                             device, args.head_weight_decay)
        print(f"pool_rate best test: {res['best_test']:.4f}", flush=True)


if __name__ == "__main__":
    main()
