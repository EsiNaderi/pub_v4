"""Compare a trained checkpoint against a fresh random network.

Shows how much hidden-layer plasticity contributed beyond random features.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from smnist_data import load_smnist
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


def ridge_classify(Ftr, ytr, Fte, yte, n_classes=10, lam=1e-2):
    one_hot = torch.zeros(Ftr.shape[0], n_classes); one_hot.scatter_(1, ytr.unsqueeze(1), 1.0)
    Fb_tr = torch.cat([Ftr, torch.ones(Ftr.shape[0], 1)], dim=1)
    Fb_te = torch.cat([Fte, torch.ones(Fte.shape[0], 1)], dim=1)
    A = Fb_tr.t() @ Fb_tr + lam * torch.eye(Fb_tr.shape[1])
    b = Fb_tr.t() @ one_hot
    W = torch.linalg.solve(A, b)
    tr_acc = float((Fb_tr @ W).argmax(dim=1).eq(ytr).float().mean().item())
    te_acc = float((Fb_te @ W).argmax(dim=1).eq(yte).float().mean().item())
    return tr_acc, te_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--aux_head", action="store_true")
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    args = p.parse_args()

    print(f"Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]

    print(f"\n=== Random reservoir (fresh init) ===", flush=True)
    torch.manual_seed(20260506)
    net_rand = build_net(args.arch, aux_head=args.aux_head)
    net_rand.eval()
    Ftr_r = compute_features(net_rand, xtr)
    Fte_r = compute_features(net_rand, xte)
    print(f"  features: {Ftr_r.shape}, mean={Ftr_r.mean().item():.4f}, "
          f"zero_rate={(Ftr_r==0).float().mean().item():.4f}", flush=True)
    tr_r, te_r = ridge_classify(Ftr_r, ytr, Fte_r, yte)
    print(f"  ridge: train_acc={tr_r:.4f}, test_acc={te_r:.4f}", flush=True)

    print(f"\n=== Trained network (from {args.ckpt}) ===", flush=True)
    ckpt = torch.load(args.ckpt, weights_only=False)
    net_t = build_net(args.arch, aux_head=args.aux_head)
    net_t.load_state_dict(ckpt["state_dict"])
    net_t.eval()
    Ftr_t = compute_features(net_t, xtr)
    Fte_t = compute_features(net_t, xte)
    print(f"  features: {Ftr_t.shape}, mean={Ftr_t.mean().item():.4f}, "
          f"zero_rate={(Ftr_t==0).float().mean().item():.4f}", flush=True)
    tr_t, te_t = ridge_classify(Ftr_t, ytr, Fte_t, yte)
    print(f"  ridge: train_acc={tr_t:.4f}, test_acc={te_t:.4f}", flush=True)

    # Direct accuracy via the trained head
    print(f"\n=== Trained network's full forward (with aux_head) ===", flush=True)
    n_correct_train = 0; n_correct_test = 0
    with torch.no_grad():
        for s in range(0, xte.shape[0], 64):
            xb = xte[s:s+64].unsqueeze(-1); yb = yte[s:s+64]
            logits, _ = net_t(xb)
            n_correct_test += int((logits.argmax(dim=1) == yb).sum().item())
        for s in range(0, xtr.shape[0], 64):
            xb = xtr[s:s+64].unsqueeze(-1); yb = ytr[s:s+64]
            logits, _ = net_t(xb)
            n_correct_train += int((logits.argmax(dim=1) == yb).sum().item())
    train_acc_full = n_correct_train / xtr.shape[0]
    test_acc_full = n_correct_test / xte.shape[0]
    print(f"  full forward: train_acc={train_acc_full:.4f}, test_acc={test_acc_full:.4f}", flush=True)

    print(f"\n=== Summary ===")
    print(f"{'method':>30} {'train':>7} {'test':>7}")
    print(f"{'random reservoir + ridge':>30} {tr_r:>7.4f} {te_r:>7.4f}")
    print(f"{'trained reservoir + ridge':>30} {tr_t:>7.4f} {te_t:>7.4f}")
    print(f"{'trained network forward':>30} {train_acc_full:>7.4f} {test_acc_full:>7.4f}")


if __name__ == "__main__":
    main()
