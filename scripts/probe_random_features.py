"""Test how much information random-init HRN extracts:
   forward 1000 train + 500 test samples, train a ridge classifier on tail rates."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, make_default_config


def main():
    torch.manual_seed(0)
    device = "cpu"
    cfg = make_default_config()
    net = HierarchicalResonantNet(cfg).to(device)
    net.eval()

    xtr, ytr, xte, yte = load_smnist()
    xtr = xtr[:1000]; ytr = ytr[:1000]
    xte = xte[:500]; yte = yte[:500]

    def compute_features(x):
        feats_list = []
        with torch.no_grad():
            for s in range(0, x.shape[0], 32):
                xb = x[s:s+32].unsqueeze(-1).to(device)
                _, info = net(xb, return_layers=True)
                last_seq = info["layer_spikes"][-1]                       # (B, T, N_out)
                T = last_seq.shape[1]
                tail = max(1, int(round(T * cfg.tail_fraction)))
                feats = last_seq[:, T - tail:].mean(dim=1)                # (B, N_out)
                feats_list.append(feats.cpu())
        return torch.cat(feats_list, dim=0)

    print("computing features...")
    Ftr = compute_features(xtr)
    Fte = compute_features(xte)
    print(f"feats train: {Ftr.shape}, test: {Fte.shape}")
    print(f"feats train mean activity: {Ftr.mean().item():.4f}, std: {Ftr.std().item():.4f}")
    print(f"feats train zero rate: {(Ftr == 0).float().mean().item():.4f}")

    # Train a ridge classifier (one-vs-rest)
    n_classes = 10
    one_hot = torch.zeros(Ftr.shape[0], n_classes); one_hot.scatter_(1, ytr.unsqueeze(1), 1.0)

    # Add bias
    Ftr_b = torch.cat([Ftr, torch.ones(Ftr.shape[0], 1)], dim=1)
    Fte_b = torch.cat([Fte, torch.ones(Fte.shape[0], 1)], dim=1)

    # Closed-form ridge
    lam = 1e-2
    A = Ftr_b.t() @ Ftr_b + lam * torch.eye(Ftr_b.shape[1])
    b = Ftr_b.t() @ one_hot
    W = torch.linalg.solve(A, b)

    pred_tr = (Ftr_b @ W).argmax(dim=1)
    pred_te = (Fte_b @ W).argmax(dim=1)
    print(f"random-feature ridge: train_acc={float((pred_tr==ytr).float().mean()):.4f}, test_acc={float((pred_te==yte).float().mean()):.4f}")

    # Also try logistic via SGD
    Ftr_d = Ftr.to(device)
    Fte_d = Fte.to(device)
    head = torch.nn.Linear(Ftr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for ep in range(40):
        order = torch.randperm(Ftr.shape[0])
        for s in range(0, Ftr.shape[0], 32):
            idx = order[s:s+32]
            xb = Ftr_d[idx]; yb = ytr[idx].to(device)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_tr = head(Ftr_d).argmax(dim=1).cpu()
        pred_te = head(Fte_d).argmax(dim=1).cpu()
    print(f"random-feature logistic: train_acc={float((pred_tr==ytr).float().mean()):.4f}, test_acc={float((pred_te==yte).float().mean()):.4f}")


if __name__ == "__main__":
    main()
