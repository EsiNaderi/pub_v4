"""Hierarchical Resonant Net.

Stacks K resonator pools across L layers, with spike-only inter-layer
communication (the R2 audit boundary). The readout is *per-class output
pools*: the last layer has exactly C pools (one per class), each with
P_out neurons. Logits come directly from per-pool tail-window spike
rates -- there is NO learnable head and NO fingerprint of the trajectory
beyond pool-level aggregation. This forces classes to occupy disjoint
resonant basins.

Architectural principles (from user spec):
- Hierarchical/fractal decomposition into pools.
- Within a pool, smaller submodes; across pools, weakly coupled.
- Inter-layer communication via spikes only.
- Each layer chooses a different frequency band (sensory: high; mode: mid; class: low/memory).

The forward returns (logits, info), where info contains per-layer spike
sequences and dynamic-state diagnostics. Trainable in two modes:

- "bptt": surrogate gradient through time (capacity check).
- "local": three-factor / e-prop / homeostasis rules (target).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from resonator import PoolConfig, ResonatorPool
from resonator_jit import ResonatorPoolJIT


@dataclass
class LayerSpec:
    n_pools: int
    pool_size: int
    omega_lo: float
    omega_hi: float
    use_recurrence: bool = True
    block_diag: bool = True
    rec_init_scale: float = 0.30
    in_init_scale: float = 2.0
    gamma: float = 0.10
    beta: float = 0.20
    lambda_leak: float = 0.01
    kappa: float = 0.30
    theta: float = 1.00
    eta: float = 0.10
    omega_per_pool: bool = False


@dataclass
class HRNConfig:
    in_dim: int = 1
    n_classes: int = 10
    layers: List[LayerSpec] = field(default_factory=list)
    out_pool_size: int = 24            # P for the per-class output pools
    out_omega_lo: float = 0.02
    out_omega_hi: float = 0.30
    out_use_recurrence: bool = True
    out_block_diag: bool = True
    out_rec_init_scale: float = 0.30
    out_in_init_scale: float = 8.0     # boosted because last-layer input is sparse spikes
    out_theta: float = 1.0
    out_eta: float = 0.05
    out_gamma: float = 0.10
    out_beta: float = 0.20
    out_lambda_leak: float = 0.01
    out_kappa: float = 0.30
    out_omega_per_pool: bool = False
    tail_fraction: float = 0.30        # fraction of sequence used for readout averaging
    surr_param: float = 5.0
    surr_kind: str = "fast_sigmoid"
    readout_temperature: float = 10.0  # multiplies pool-rate logits before softmax
    use_pool_bias: bool = True         # learnable per-class bias on logits (breaks symmetry)
    use_jit: bool = True               # use ResonatorPoolJIT (~3x faster)
    center_logits: bool = True         # subtract mean rate -> zero-centered logits
    bias_init_scale: float = 0.05      # small random init breaks pool-class symmetry
    # Auxiliary linear head over output spikes (for capacity verification only).
    # When enabled, outputs the linear head's logits as primary; pool-rate logits
    # become a parallel signal stored in info["pool_logits"].
    aux_linear_head: bool = False
    aux_head_features: int = 0         # 0 means use last-layer rate features only


def make_default_config() -> HRNConfig:
    """3-layer default (sensory -> mode -> class).

    Init regime: each layer fires ~5-15% on real SMNIST inputs.
    """

    return HRNConfig(
        in_dim=1,
        n_classes=10,
        layers=[
            LayerSpec(
                n_pools=4, pool_size=32, omega_lo=0.5, omega_hi=2.5,
                use_recurrence=True, block_diag=True,
                in_init_scale=4.0, rec_init_scale=0.30,
                theta=0.7, eta=0.30, gamma=0.20, beta=0.20,
            ),
            LayerSpec(
                n_pools=8, pool_size=32, omega_lo=0.10, omega_hi=1.0,
                use_recurrence=True, block_diag=True,
                in_init_scale=2.0, rec_init_scale=0.30,
                theta=0.7, eta=0.10, gamma=0.20, beta=0.20,
            ),
        ],
        out_pool_size=24,
        out_omega_lo=0.02,
        out_omega_hi=0.30,
        out_use_recurrence=True,
        out_block_diag=True,
        out_in_init_scale=4.0,
        out_theta=0.6, out_eta=0.15, out_gamma=0.20, out_beta=0.20,
        tail_fraction=0.30,
        surr_param=2.5,
    )


class HierarchicalResonantNet(nn.Module):
    def __init__(self, cfg: HRNConfig):
        super().__init__()
        self.cfg = cfg

        Pool = ResonatorPoolJIT if cfg.use_jit else ResonatorPool
        layer_pools = []
        in_dim = cfg.in_dim
        for spec in cfg.layers:
            pool_cfg = PoolConfig(
                n_pools=spec.n_pools, pool_size=spec.pool_size, in_dim=in_dim,
                omega_lo=spec.omega_lo, omega_hi=spec.omega_hi,
                gamma=spec.gamma, beta=spec.beta, lambda_leak=spec.lambda_leak,
                kappa=spec.kappa, theta=spec.theta, eta=spec.eta,
                use_recurrence=spec.use_recurrence, block_diag=spec.block_diag,
                rec_init_scale=spec.rec_init_scale, in_init_scale=spec.in_init_scale,
                surr_param=cfg.surr_param, surr_kind=cfg.surr_kind,
                omega_per_pool=spec.omega_per_pool,
            )
            layer_pools.append(Pool(pool_cfg))
            in_dim = pool_cfg.n_total

        # output layer: K = n_classes pools, P = out_pool_size, in_dim from last hidden
        out_cfg = PoolConfig(
            n_pools=cfg.n_classes, pool_size=cfg.out_pool_size, in_dim=in_dim,
            omega_lo=cfg.out_omega_lo, omega_hi=cfg.out_omega_hi,
            theta=cfg.out_theta, eta=cfg.out_eta,
            gamma=cfg.out_gamma, beta=cfg.out_beta,
            lambda_leak=cfg.out_lambda_leak, kappa=cfg.out_kappa,
            use_recurrence=cfg.out_use_recurrence, block_diag=cfg.out_block_diag,
            rec_init_scale=cfg.out_rec_init_scale, in_init_scale=cfg.out_in_init_scale,
            surr_param=cfg.surr_param, surr_kind=cfg.surr_kind,
            omega_per_pool=cfg.out_omega_per_pool,
        )
        layer_pools.append(Pool(out_cfg))
        self.layers = nn.ModuleList(layer_pools)
        self.out_layer = self.layers[-1]

        if cfg.use_pool_bias:
            # small random init to break pool-class symmetry at start
            self.pool_bias = nn.Parameter(torch.randn(cfg.n_classes) * cfg.bias_init_scale)
        else:
            self.pool_bias = None

        # Optional auxiliary linear head for capacity verification.
        if cfg.aux_linear_head:
            n_out_total = cfg.n_classes * cfg.out_pool_size
            in_feats = cfg.aux_head_features or n_out_total
            self.aux_head = nn.Linear(in_feats, cfg.n_classes)
        else:
            self.aux_head = None

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def forward(
        self,
        x: torch.Tensor,
        return_layers: bool = False,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        x: (B, T, in_dim)
        Returns (logits, info) where logits is (B, C).
        """

        B, T, _ = x.shape
        sig = x
        layer_spikes: List[torch.Tensor] = []
        states: List[dict] = []
        for layer in self.layers:
            s_seq, info = layer(sig, return_uv=False, return_qsq=False)
            layer_spikes.append(s_seq)
            states.append(info["final_state"])
            sig = s_seq

        # readout: per-class output pool tail-window spike rate
        out_seq_full = layer_spikes[-1]                                  # (B, T, n_classes * out_pool_size)
        out_seq_p = out_seq_full.view(B, T, self.cfg.n_classes, self.cfg.out_pool_size)
        tail = max(1, int(round(T * self.cfg.tail_fraction)))
        pool_rate = out_seq_p[:, T - tail :].mean(dim=(1, 3))            # (B, n_classes)
        if self.cfg.center_logits:
            pool_rate_centered = pool_rate - pool_rate.mean(dim=1, keepdim=True)
        else:
            pool_rate_centered = pool_rate
        pool_logits = pool_rate_centered * self.cfg.readout_temperature
        if self.pool_bias is not None:
            pool_logits = pool_logits + self.pool_bias

        if self.aux_head is not None:
            # use last-layer per-neuron tail rates as features
            feats = out_seq_full[:, T - tail :].mean(dim=1)               # (B, N_out_total)
            logits = self.aux_head(feats)
        else:
            logits = pool_logits

        info = {"pool_logits": pool_logits}
        if return_layers:
            info["layer_spikes"] = layer_spikes
        if return_state:
            info["states"] = states
        info["pool_rate"] = pool_rate
        return logits, info

    def n_params(self) -> dict:
        out = {"total": 0}
        for i, layer in enumerate(self.layers):
            n = sum(p.numel() for p in layer.parameters())
            out[f"layer_{i}"] = n
            out["total"] += n
        if self.pool_bias is not None:
            out["pool_bias"] = self.pool_bias.numel()
            out["total"] += self.pool_bias.numel()
        return out


if __name__ == "__main__":
    cfg = make_default_config()
    net = HierarchicalResonantNet(cfg)
    print("layers:", net.n_layers, "params:", net.n_params())
    x = torch.rand(2, 100, 1)
    logits, info = net(x, return_layers=True)
    print("logits:", logits.shape, "value range:", logits.min().item(), logits.max().item())
    for i, s in enumerate(info["layer_spikes"]):
        rate = s.mean().item()
        print(f"  layer {i}: spikes shape {tuple(s.shape)}, rate {rate:.4f}")
