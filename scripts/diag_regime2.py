"""Re-scan firing regime with parameter sweep on real SMNIST."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, HRNConfig, LayerSpec


def make_cfg(theta, eta, in_scale, gamma=0.10, beta=0.20):
    return HRNConfig(
        in_dim=1, n_classes=10,
        layers=[
            LayerSpec(n_pools=4, pool_size=32, omega_lo=0.5, omega_hi=2.5,
                      in_init_scale=in_scale, theta=theta, eta=eta,
                      gamma=gamma, beta=beta),
            LayerSpec(n_pools=8, pool_size=32, omega_lo=0.10, omega_hi=1.0,
                      in_init_scale=in_scale, theta=theta, eta=eta * 0.5,
                      gamma=gamma, beta=beta),
        ],
        out_pool_size=24, out_omega_lo=0.02, out_omega_hi=0.30,
        out_in_init_scale=in_scale, out_theta=theta, out_eta=eta * 0.5,
        out_gamma=gamma, out_beta=beta,
        tail_fraction=0.30,
    )


def main():
    torch.manual_seed(0)
    xtr, _, _, _ = load_smnist()
    x = xtr[:32].unsqueeze(-1)

    print(f"{'gam':>4} {'bet':>4} {'theta':>5} {'eta':>4} {'D_sc':>4}  rate0   rate1   rate_out")
    for gamma, beta in [(0.10, 0.20), (0.15, 0.20), (0.20, 0.20), (0.05, 0.15)]:
        for theta in [0.3, 0.5, 0.7]:
            for eta in [0.10, 0.30]:
                for in_scale in [2.0, 4.0]:
                    cfg = make_cfg(theta, eta, in_scale, gamma, beta)
                    torch.manual_seed(0)
                    net = HierarchicalResonantNet(cfg)
                    with torch.no_grad():
                        _, info = net(x, return_layers=True)
                    rates = [float(s.mean().item()) for s in info["layer_spikes"]]
                    print(f"{gamma:>4.2f} {beta:>4.2f} {theta:>5.2f} {eta:>4.2f} {in_scale:>4.1f}  "
                          f"{rates[0]:.3f}  {rates[1]:.3f}  {rates[2]:.3f}")


if __name__ == "__main__":
    main()
