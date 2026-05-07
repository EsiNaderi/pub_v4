"""Diagnostic: check gradient magnitudes layer-by-layer + spike-rate response
to different classes at random init."""

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
    cfg = make_default_config()
    cfg.aux_linear_head = True
    net = HierarchicalResonantNet(cfg)

    xtr, ytr, _, _ = load_smnist()
    # one sample per class
    by_class = []
    for c in range(10):
        idx = (ytr == c).nonzero()[0].item()
        by_class.append(xtr[idx])
    x = torch.stack(by_class).unsqueeze(-1)            # (10, 784, 1)
    y = torch.arange(10)

    print("=== Forward pass diagnostics ===")
    with torch.no_grad():
        logits, info = net(x, return_layers=True)
        print(f"logits min/max: {logits.min().item():.3f}, {logits.max().item():.3f}")
        print(f"pool_logits min/max: {info['pool_logits'].min().item():.3f}, {info['pool_logits'].max().item():.3f}")
        print(f"argmax distribution: {logits.argmax(dim=1).tolist()}")
        for i, s in enumerate(info["layer_spikes"]):
            mean_rate = s.mean().item()
            std_rate = s.mean(dim=(0, 1)).std().item()  # across-neuron std of mean rate
            print(f"  layer {i}: rate {mean_rate:.4f} std-across-neurons {std_rate:.4f}")
        # per-class output rate variance
        out = info["layer_spikes"][-1]                   # (10, 784, N_out)
        out_p = out.view(10, 784, 10, cfg.out_pool_size).mean(dim=(1, 3))  # (10, 10) class x pool
        print(f"\noutput pool rate per class:\n{out_p.numpy()}")
        # top-firing pool per class
        top_pool = out_p.argmax(dim=1)
        print(f"top-firing pool per class: {top_pool.tolist()}")
        # variance across classes per pool (high = pool differentiates classes)
        per_pool_var = out_p.var(dim=0)
        print(f"per-pool variance across classes: {per_pool_var.tolist()}")

    print("\n=== Gradient flow ===")
    net.train()
    logits, info = net(x, return_layers=True)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is None:
            print(f"  {name}: NO GRAD")
        else:
            g_norm = p.grad.norm().item()
            p_norm = p.norm().item()
            print(f"  {name:30s}: |grad|={g_norm:.4e}  |param|={p_norm:.3e}  ratio={g_norm/(p_norm+1e-12):.3e}")


if __name__ == "__main__":
    main()
