"""Test gradient flow with various truncations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, make_default_config


def main():
    xtr, ytr, _, _ = load_smnist()
    by_class = []
    for c in range(10):
        idx = (ytr == c).nonzero()[0].item()
        by_class.append(xtr[idx])
    x = torch.stack(by_class).unsqueeze(-1)
    y = torch.arange(10)

    for gtrunc in [50, 100, 200, 300, 0]:
        torch.manual_seed(0)
        cfg = make_default_config()
        cfg.aux_linear_head = True
        net = HierarchicalResonantNet(cfg)
        for layer in net.layers:
            layer.cfg.grad_truncate = gtrunc

        net.zero_grad()
        logits, info = net(x, return_layers=False)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # collect gradient norms
        nan_layers = []
        ok_grads = {}
        for name, p in net.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                nan_layers.append(name)
            else:
                ok_grads[name] = p.grad.norm().item()
        print(f"gtrunc={gtrunc:>3}: NaN layers: {nan_layers if nan_layers else '(none)'}")
        if not nan_layers:
            print(f"  layer-0 D_re grad: {ok_grads.get('layers.0.D_re', '?'):.3e}")
            print(f"  layer-0 W_re grad: {ok_grads.get('layers.0.W_re', '?'):.3e}")
            print(f"  layer-0 omega grad: {ok_grads.get('layers.0.omega', '?'):.3e}")


if __name__ == "__main__":
    main()
