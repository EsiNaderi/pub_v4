"""Resonant ALIF: hybrid of adaptive LIF and complex-amplitude rotor.

Per neuron, state is complex z = u + iv plus adaptive threshold a.

Forward dynamics:
    q = (1-lambda) [exp(i omega) z(t-1) + W_rec @ s(t-1) + eta D x(t)]
    s = H(|q|^2 - (theta + a))
    z = (1 - kappa s) q
    a = rho a + alpha s

This is exactly the Resonance Network with adaptive theta. The SL nonlinearity
(beta*(1-|z|^2)*z) is REMOVED -- amplitude is bounded by the leak (1-lambda)
factor and the spike contraction kappa, not by limit-cycle saturation.

This makes the dynamics more linear (and more amenable to BPTT) while keeping
the resonant + spike-only structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from resonator import _SpikeFn


@dataclass
class ResonantALIFConfig:
    n_pools: int
    pool_size: int
    in_dim: int
    omega_lo: float
    omega_hi: float
    lambda_leak: float = 0.01
    kappa: float = 0.30
    theta: float = 0.5
    eta: float = 0.30
    rho: float = 0.95               # adaptive threshold decay (per timestep)
    alpha_adapt: float = 0.5         # threshold increase per spike
    use_recurrence: bool = True
    block_diag: bool = True
    rec_init_scale: float = 0.20
    in_init_scale: float = 4.0
    surr_param: float = 2.5
    omega_per_pool: bool = True
    learn_dyn_params: bool = False
    detach_recurrence: bool = True

    @property
    def n_total(self) -> int:
        return self.n_pools * self.pool_size


@torch.jit.script
def _ralif_step_block(
    u: torch.Tensor, v: torch.Tensor, s_prev: torch.Tensor,
    a: torch.Tensor,
    in_r_t: torch.Tensor, in_i_t: torch.Tensor,
    cos_om: torch.Tensor, sin_om: torch.Tensor,
    one_minus_lam: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor,
    W_re_blk: torch.Tensor, W_im_blk: torch.Tensor,
    K: int, P: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_rot = cos_om * u - sin_om * v
    v_rot = sin_om * u + cos_om * v

    B = s_prev.shape[0]
    sp = s_prev.view(B, K, P)
    rec_r = torch.einsum("bkp,kqp->bkq", sp, W_re_blk).reshape(B, K * P)
    rec_i = torch.einsum("bkp,kqp->bkq", sp, W_im_blk).reshape(B, K * P)

    u_q = one_minus_lam * (u_rot + rec_r + in_r_t)
    v_q = one_minus_lam * (v_rot + rec_i + in_i_t)
    q_sq = u_q * u_q + v_q * v_q
    arg = q_sq - (theta + a)
    return u_q, v_q, q_sq, arg


class ResonantALIFPool(nn.Module):
    def __init__(self, cfg: ResonantALIFConfig):
        super().__init__()
        self.cfg = cfg
        K, P, M = cfg.n_pools, cfg.pool_size, cfg.in_dim
        N = K * P
        self.N = N

        log_lo = math.log(max(cfg.omega_lo, 1e-4))
        log_hi = math.log(max(cfg.omega_hi, cfg.omega_lo + 1e-3))
        if cfg.omega_per_pool and K > 1:
            band_edges = torch.linspace(log_lo, log_hi, K + 1)
            pieces = []
            for k in range(K):
                lo_k, hi_k = band_edges[k].item(), band_edges[k + 1].item()
                pieces.append(torch.exp(lo_k + (hi_k - lo_k) * torch.rand(P)))
            omega_init = torch.cat(pieces)
        else:
            omega_init = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(N))
        self.omega = nn.Parameter(omega_init)
        self.eta = nn.Parameter(torch.full((N,), float(cfg.eta)))

        self.register_buffer("lambda_leak", torch.full((N,), float(cfg.lambda_leak)))
        self.register_buffer("kappa", torch.full((N,), float(cfg.kappa)))
        self.register_buffer("theta", torch.full((N,), float(cfg.theta)))

        scale = cfg.in_init_scale / max(math.sqrt(M), 1.0)
        self.D_re = nn.Parameter(torch.randn(N, M) * scale)
        self.D_im = nn.Parameter(torch.randn(N, M) * scale)

        if cfg.use_recurrence:
            if cfg.block_diag:
                Wre = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
                Wim = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
                eye = torch.eye(P).unsqueeze(0)
                Wre = Wre * (1.0 - eye); Wim = Wim * (1.0 - eye)
                self.W_re = nn.Parameter(Wre); self.W_im = nn.Parameter(Wim)
            else:
                Wre = torch.randn(N, N) * (cfg.rec_init_scale / math.sqrt(N))
                Wim = torch.randn(N, N) * (cfg.rec_init_scale / math.sqrt(N))
                Wre.fill_diagonal_(0.0); Wim.fill_diagonal_(0.0)
                self.W_re = nn.Parameter(Wre); self.W_im = nn.Parameter(Wim)
        else:
            self.W_re = nn.Parameter(torch.zeros(K if cfg.block_diag else N, P if cfg.block_diag else N, P if cfg.block_diag else 1))
            self.W_im = nn.Parameter(torch.zeros_like(self.W_re))

    @property
    def K(self): return self.cfg.n_pools
    @property
    def P(self): return self.cfg.pool_size

    def forward(self, x: torch.Tensor, prev_state: Optional[dict] = None,
                return_uv: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T, M = x.shape
        device, dtype = x.device, x.dtype
        K, P = self.K, self.P
        N = self.N

        st = prev_state or {}
        u = st.get("u", torch.zeros(B, N, device=device, dtype=dtype))
        v = st.get("v", torch.zeros(B, N, device=device, dtype=dtype))
        s_prev = st.get("s", torch.zeros(B, N, device=device, dtype=dtype))
        a = st.get("a", torch.zeros(B, N, device=device, dtype=dtype))

        Dr = self.D_re.t(); Di = self.D_im.t()
        eta = self.eta.view(1, 1, -1)
        xr = (x @ Dr) * eta
        xi = (x @ Di) * eta

        cos_om = torch.cos(self.omega); sin_om = torch.sin(self.omega)
        lam = self.lambda_leak.clamp(min=0.0, max=0.99)
        one_minus_lam = (1.0 - lam).view(1, -1)
        kappa = self.kappa; theta = self.theta
        rho = self.cfg.rho
        alpha_adapt = self.cfg.alpha_adapt
        detach_rec = self.cfg.detach_recurrence

        s_list: List[torch.Tensor] = []
        for t in range(T):
            sp_for_rec = s_prev.detach() if detach_rec else s_prev
            u_q, v_q, q_sq, arg = _ralif_step_block(
                u, v, sp_for_rec, a, xr[:, t], xi[:, t],
                cos_om, sin_om, one_minus_lam, kappa, theta,
                self.W_re, self.W_im, K, P,
            )
            s = _SpikeFn.apply(arg, self.cfg.surr_param, 1)
            scale = 1.0 - kappa * s
            u = scale * u_q; v = scale * v_q
            u = torch.clamp(u, -3.0, 3.0); v = torch.clamp(v, -3.0, 3.0)
            a = rho * a + alpha_adapt * s
            s_prev = s
            s_list.append(s)

        s_seq = torch.stack(s_list, dim=1)
        info = {"final_state": {"u": u, "v": v, "s": s_prev, "a": a}}
        return s_seq, info


if __name__ == "__main__":
    cfg = ResonantALIFConfig(n_pools=4, pool_size=8, in_dim=1,
                              omega_lo=0.1, omega_hi=2.0)
    pool = ResonantALIFPool(cfg)
    x = torch.randn(2, 100, 1)
    s, _ = pool(x)
    print("ralif:", s.shape, "rate:", s.mean().item())
