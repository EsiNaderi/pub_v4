"""Hunt down the layer-0 NaN by toggling components."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, HRNConfig, LayerSpec


def cfg_no_rec(use_rec0: bool = False, T_only: int = 0) -> HRNConfig:
    return HRNConfig(
        in_dim=1, n_classes=10,
        layers=[
            LayerSpec(n_pools=4, pool_size=32, omega_lo=0.5, omega_hi=2.5,
                      use_recurrence=use_rec0, in_init_scale=4.0,
                      theta=1.0, eta=0.30),
            LayerSpec(n_pools=8, pool_size=32, omega_lo=0.10, omega_hi=1.0,
                      in_init_scale=4.0, theta=1.0, eta=0.05),
        ],
        out_pool_size=24, out_omega_lo=0.02, out_omega_hi=0.30,
        out_in_init_scale=4.0, tail_fraction=0.30,
        aux_linear_head=True,
    )


def test(label, cfg_fn, T=784):
    xtr, ytr, _, _ = load_smnist()
    by_class = []
    for c in range(10):
        idx = (ytr == c).nonzero()[0].item()
        by_class.append(xtr[idx])
    x = torch.stack(by_class).unsqueeze(-1)
    if T < 784:
        x = x[:, :T]
    y = torch.arange(10)

    torch.manual_seed(0)
    cfg = cfg_fn()
    net = HierarchicalResonantNet(cfg)
    net.zero_grad()
    logits, _ = net(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    nan_params = [n for n, p in net.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"{label}: NaN params: {nan_params if nan_params else '(none)'}")


if __name__ == "__main__":
    test("baseline (no rec layer 0)", cfg_no_rec)
    test("with rec layer 0", lambda: cfg_no_rec(use_rec0=True))
    # short T
    test("T=100", cfg_no_rec, T=100)
    test("T=300", cfg_no_rec, T=300)
    test("T=500", cfg_no_rec, T=500)
