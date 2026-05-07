"""Multi-seed random feature baseline. Establishes the floor."""

from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet
from train_bptt import build_net


def compute_features(net, x, batch=64):
    feats = []
    cfg = net.cfg
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            xb = x[s:s+batch].unsqueeze(-1)
            _, info = net(xb, return_layers=True)
            last = info["layer_spikes"][-1]
            T = last.shape[1]
            tail = max(1, int(round(T * cfg.tail_fraction)))
            feats.append(last[:, T - tail:].mean(dim=1))
    return torch.cat(feats, dim=0)


def main():
    seeds = [0, 1, 2]
    archs = ["small", "default"]
    train_size = 5000
    test_size = 1000

    xtr, ytr, xte, yte = load_smnist()
    xtr, ytr = xtr[:train_size], ytr[:train_size]
    xte, yte = xte[:test_size], yte[:test_size]

    print(f"{'arch':>10} {'seed':>4}  {'feat_shape':>12} {'mean':>6} {'std':>5} {'zeros':>6}  {'ridge_tr':>8} {'ridge_te':>8}", flush=True)
    for arch in archs:
        for seed in seeds:
            torch.manual_seed(seed)
            net = build_net(arch)
            net.eval()
            t0 = time.time()
            Ftr = compute_features(net, xtr)
            Fte = compute_features(net, xte)
            wall = time.time() - t0

            n_classes = 10
            one_hot = torch.zeros(Ftr.shape[0], n_classes); one_hot.scatter_(1, ytr.unsqueeze(1), 1.0)
            Fb_tr = torch.cat([Ftr, torch.ones(Ftr.shape[0], 1)], dim=1)
            Fb_te = torch.cat([Fte, torch.ones(Fte.shape[0], 1)], dim=1)
            lam = 1e-2
            A = Fb_tr.t() @ Fb_tr + lam * torch.eye(Fb_tr.shape[1])
            b = Fb_tr.t() @ one_hot
            W = torch.linalg.solve(A, b)
            tr_acc = float((Fb_tr @ W).argmax(dim=1).eq(ytr).float().mean().item())
            te_acc = float((Fb_te @ W).argmax(dim=1).eq(yte).float().mean().item())

            print(f"{arch:>10} {seed:>4}  {tuple(Ftr.shape)} {Ftr.mean().item():>5.3f} {Ftr.std().item():>5.3f} "
                  f"{(Ftr==0).float().mean().item():>5.3f}  {tr_acc:>7.3f} {te_acc:>7.3f}  ({wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
