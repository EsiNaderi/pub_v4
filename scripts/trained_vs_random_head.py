"""Compare BPTT-trained checkpoint vs fresh random init, both with linear-head
trained on full 60k SMNIST. The decisive test: did BPTT improve features?

For each configuration:
1. Compute tail-window features on 60k train / 10k test (single forward pass)
2. Train a linear head on those features for many epochs
3. Report best test acc

Outputs a single CSV with all results.
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


def compute_tail_features(net, x, batch=64):
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


def train_linear_head(Ftr, ytr, Fte, yte, n_classes, epochs=300, lr=3e-3, batch=128, wd=1e-4):
    head = nn.Linear(Ftr.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    n = Ftr.shape[0]
    best = 0.0
    log = []
    for ep in range(epochs):
        order = torch.randperm(n)
        for s in range(0, n, batch):
            idx = order[s : s + batch]
            logits = head(Ftr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tr = float(head(Ftr).argmax(dim=1).eq(ytr).float().mean().item())
            te = float(head(Fte).argmax(dim=1).eq(yte).float().mean().item())
        if te > best:
            best = te
        log.append((ep, tr, te))
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d}  train={tr:.4f}  test={te:.4f}  best_test={best:.4f}", flush=True)
    return best, log


def run_one(label, build_fn, train_size, test_size, epochs, csv_writer):
    print(f"\n=== {label} ===", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if train_size: xtr, ytr = xtr[:train_size], ytr[:train_size]
    if test_size: xte, yte = xte[:test_size], yte[:test_size]
    net = build_fn()
    net.eval()
    print(f"  net params: {net.n_params()['total']}", flush=True)

    print(f"  computing train features ({xtr.shape[0]} samples) ...", flush=True)
    t0 = time.time()
    Ftr = compute_tail_features(net, xtr)
    print(f"    done in {time.time() - t0:.1f}s. mean={Ftr.mean().item():.4f}, "
          f"std={Ftr.std().item():.4f}, zero_rate={(Ftr == 0).float().mean().item():.4f}", flush=True)

    print(f"  computing test features ...", flush=True)
    t0 = time.time()
    Fte = compute_tail_features(net, xte)
    print(f"    done in {time.time() - t0:.1f}s.", flush=True)

    print(f"  training linear head for {epochs} epochs ...", flush=True)
    best, log = train_linear_head(Ftr, ytr, Fte, yte, net.cfg.n_classes, epochs=epochs)
    print(f"  BEST TEST: {best:.4f}", flush=True)
    if csv_writer:
        csv_writer.writerow([label, train_size, test_size, epochs, best,
                              Ftr.shape[1], float(Ftr.mean().item()),
                              float((Ftr == 0).float().mean().item())])
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="results/ckpt_overnight_default_10k.pt")
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--train_size", type=int, default=60000)
    p.add_argument("--test_size", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--csv", type=str, default="results/run_trained_vs_random_head.csv")
    args = p.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f = open(args.csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["label", "train_size", "test_size", "head_epochs", "best_test_acc",
                     "feat_dim", "feat_mean", "feat_zero_rate"])

    # 1) Fresh random init (with our default seed for reproducibility)
    def build_random():
        torch.manual_seed(20260506)
        return build_net(args.arch, aux_head=True)
    run_one(f"random_init_seed_20260506", build_random, args.train_size, args.test_size,
            args.epochs, writer)
    f.flush()

    # 2) BPTT-trained checkpoint
    def build_trained():
        net = build_net(args.arch, aux_head=True)
        ck = torch.load(args.ckpt, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        return net
    run_one(f"bptt_trained_ckpt", build_trained, args.train_size, args.test_size,
            args.epochs, writer)
    f.flush()

    f.close()
    print("\n=== Done. CSV at:", args.csv, "===", flush=True)


if __name__ == "__main__":
    main()
