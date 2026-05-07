"""Diagnostic: scan theta / eta / D_scale to find a healthy spike regime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, HRNConfig, LayerSpec


def make_cfg(theta: float, eta: float, in_scale: float) -> HRNConfig:
    return HRNConfig(
        in_dim=1, n_classes=10,
        layers=[
            LayerSpec(n_pools=4, pool_size=32, omega_lo=0.5, omega_hi=2.5,
                      in_init_scale=in_scale, rec_init_scale=0.30,
                      theta=theta, eta=eta),
            LayerSpec(n_pools=8, pool_size=32, omega_lo=0.10, omega_hi=1.0,
                      in_init_scale=8.0, rec_init_scale=0.30,
                      theta=theta, eta=eta),
        ],
        out_pool_size=24, out_omega_lo=0.02, out_omega_hi=0.30,
        out_in_init_scale=8.0, tail_fraction=0.30,
    )


def main():
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    xtr, ytr, xte, yte = load_smnist()
    x = xtr[:64].unsqueeze(-1).to(device)

    print(f"{'theta':>6} {'eta':>5} {'D_sc':>5}  rate0   rate1   rate_out")
    for theta in [0.5, 0.7, 1.0]:
        for eta in [0.10, 0.30, 0.50]:
            for in_scale in [2.0, 4.0, 6.0]:
                cfg = make_cfg(theta, eta, in_scale)
                net = HierarchicalResonantNet(cfg).to(device)
                with torch.no_grad():
                    logits, info = net(x, return_layers=True)
                rates = [float(s.mean().item()) for s in info["layer_spikes"]]
                print(f"{theta:>6.2f} {eta:>5.2f} {in_scale:>5.1f}  {rates[0]:.4f}  {rates[1]:.4f}  {rates[2]:.4f}")


if __name__ == "__main__":
    main()
