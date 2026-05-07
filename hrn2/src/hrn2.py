"""HRN-v2: Hierarchical Resonant Network composing oscillator stages.

Each stage is a population of complex damped-rotation oscillators. The
inter-stage signal is the time-series of per-neuron above-field
activity from the previous stage, fed as multi-channel input to the
next stage.

For the first iteration we use *full* fan-in (each stage L+1 neuron
sees every stage L output). With moderate stage sizes (N ≤ 256) the
memory cost is acceptable. Sparse connectivity can be added later
once the architecture is validated.

No BPTT. No surrogate spikes. Inter-stage credit transport is via
fixed random feedback matrices (Lillicrap), modulating each stage's
local eligibility-trace gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from oscillator import (
    OscillatorConfig, OscillatorParams, init_params, omega_of, alpha_of,
    forward_with_eligibility,
)
from local_rules import adaptive_mean_competition


@dataclass
class StageConfig:
    n_pools: int
    m_per_pool: int
    omega_min: float
    omega_max: float
    alpha_min: float
    alpha_max: float
    input_init: float = 0.05
    target_usage: float = 0.0625


@dataclass
class HRN2Config:
    stages: list[StageConfig]
    n_classes: int = 10
    tail_fraction: float = 0.30
    label_prior: float = 2.0


class HRN2:
    """Composes oscillator stages.

    The LAST stage acts as the "output" stage. Its tail energy + per-neuron
    label tags form the prediction. Earlier stages provide multi-channel
    real-valued time-series to subsequent stages.

    Parameters are stored as plain torch.Tensor (gradients are
    hand-assembled).
    """

    def __init__(self, cfg: HRN2Config, *, seed: int = 20260507):
        self.cfg = cfg
        self.rng = torch.Generator().manual_seed(seed)
        self.stage_cfgs: list[OscillatorConfig] = []
        self.stage_params: list[OscillatorParams] = []
        self.stage_thetas: list[torch.Tensor] = []
        self.stage_usage: list[torch.Tensor] = []
        self.feedback_matrices: list[torch.Tensor] = []   # B[L] from layer L+1 to L

        prev_n = 1
        for L, sc in enumerate(cfg.stages):
            P = sc.n_pools * sc.m_per_pool
            ocfg = OscillatorConfig(
                n_neurons=P, n_input_channels=prev_n,
                omega_min=sc.omega_min, omega_max=sc.omega_max,
                alpha_min=sc.alpha_min, alpha_max=sc.alpha_max,
                input_init=sc.input_init,
            )
            self.stage_cfgs.append(ocfg)
            self.stage_params.append(init_params(ocfg, generator=self.rng))
            self.stage_thetas.append(torch.full((sc.n_pools, sc.m_per_pool), 1.0))
            self.stage_usage.append(torch.full((sc.n_pools, sc.m_per_pool), 1.0 / sc.m_per_pool))
            prev_n = P

        # Per-neuron class-tag mass on the LAST stage
        last = cfg.stages[-1]
        P_last = last.n_pools * last.m_per_pool
        self.label_mass = torch.full((P_last, cfg.n_classes), cfg.label_prior / cfg.n_classes)

        # Random fixed feedback matrices: B[L] sends credit from stage L+1 back to stage L.
        # Shape: (P_{L+1}, P_L)
        for L in range(len(cfg.stages) - 1):
            P_L = cfg.stages[L].n_pools * cfg.stages[L].m_per_pool
            P_next = cfg.stages[L + 1].n_pools * cfg.stages[L + 1].m_per_pool
            B = torch.randn(P_next, P_L, generator=self.rng) / max(P_next, 1) ** 0.5
            self.feedback_matrices.append(B)

    def n_stages(self) -> int:
        return len(self.stage_params)

    def num_neurons(self, stage: int) -> int:
        sc = self.cfg.stages[stage]
        return sc.n_pools * sc.m_per_pool

    def all_tensors(self) -> list[torch.Tensor]:
        ts = []
        for sp in self.stage_params:
            ts.extend(sp.tensors())
        return ts

    def stage_param_groups(self) -> list[list[torch.Tensor]]:
        """One list per stage, used by the optimizer when stages have different LRs."""
        return [list(sp.tensors()) for sp in self.stage_params]
