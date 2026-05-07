"""Fractal resonator pool: hierarchical block-structure within a single layer.

Rather than a single K x P pool grid, organize neurons recursively as
K_outer outer pools, each containing K_inner inner pools, each containing P
neurons. The recurrent matrix has two-level block structure:

- Within each inner pool: dense P x P coupling (normal pool W_rec).
- Within each outer pool, between inner pools: sparse P x P coupling.
- Between outer pools: zero (fully independent).

Each inner pool gets its own omega sub-band, so within an outer pool's
frequency band, inner pools cover finer-grained sub-bands. This
implements the "modes within modes" recursion the user described.

The forward step is exactly the same as ResonatorPoolJIT, but the
recurrence is computed with two-level block einsum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from resonator import _SpikeFn


@dataclass
class FractalPoolConfig:
    n_outer: int            # K_outer
    n_inner: int            # K_inner per outer
    pool_size: int          # P neurons per inner pool
    in_dim: int
    omega_lo: float
    omega_hi: float
    gamma: float = 0.20
    beta: float = 0.20
    lambda_leak: float = 0.01
    kappa: float = 0.30
    theta: float = 0.7
    eta: float = 0.30
    use_recurrence: bool = True
    inner_init_scale: float = 0.30
    outer_init_scale: float = 0.10        # outer-pool inter-inner coupling (smaller)
    in_init_scale: float = 4.0
    surr_param: float = 2.5
    learn_dyn_params: bool = False

    @property
    def n_total(self) -> int:
        return self.n_outer * self.n_inner * self.pool_size


@torch.jit.script
def _step_fractal(
    u: torch.Tensor, v: torch.Tensor, s_prev: torch.Tensor,
    in_r_t: torch.Tensor, in_i_t: torch.Tensor,
    cos_om: torch.Tensor, sin_om: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor,
    one_minus_lam: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor,
    W_re_inner: torch.Tensor, W_im_inner: torch.Tensor,        # (K_o, K_i, P, P) - dense within inner
    W_re_cross: torch.Tensor, W_im_cross: torch.Tensor,        # (K_o, K_i, K_i, P, P) - cross inner pools within outer
    K_o: int, K_i: int, P: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_rot = cos_om * u - sin_om * v
    v_rot = sin_om * u + cos_om * v
    amp_sq_raw = u * u + v * v
    amp_sq = amp_sq_raw.detach().clamp(max=10.0)
    sl = beta * (1.0 - amp_sq) - gamma
    u_sl = sl * u
    v_sl = sl * v

    B = s_prev.shape[0]
    sp = s_prev.view(B, K_o, K_i, P)
    # inner coupling: each inner pool's recurrence within itself
    rec_r_in = torch.einsum("bokp,okqp->bokq", sp, W_re_inner).reshape(B, K_o * K_i * P)
    rec_i_in = torch.einsum("bokp,okqp->bokq", sp, W_im_inner).reshape(B, K_o * K_i * P)
    # cross-inner coupling within outer: source over inner pools, target inner pool
    # W_cross[o, dst, src, q, p] : edge (src->dst) weight from neuron p to q within outer o
    rec_r_cross = torch.einsum("boip,odiqp->bodq", sp, W_re_cross).reshape(B, K_o * K_i * P)
    rec_i_cross = torch.einsum("boip,odiqp->bodq", sp, W_im_cross).reshape(B, K_o * K_i * P)

    rec_r = rec_r_in + rec_r_cross
    rec_i = rec_i_in + rec_i_cross

    u_q = one_minus_lam * (u_rot + u_sl + rec_r + in_r_t)
    v_q = one_minus_lam * (v_rot + v_sl + rec_i + in_i_t)
    q_sq = u_q * u_q + v_q * v_q
    arg = q_sq - theta
    return u_q, v_q, q_sq, arg


class FractalResonatorPool(nn.Module):
    """Two-level block recurrence, fractal omega tiling."""

    def __init__(self, cfg: FractalPoolConfig):
        super().__init__()
        self.cfg = cfg
        K_o, K_i, P, M = cfg.n_outer, cfg.n_inner, cfg.pool_size, cfg.in_dim
        N = K_o * K_i * P
        self.N = N

        # Fractal omega: outer pool's outer-band split into inner sub-bands
        log_lo = math.log(max(cfg.omega_lo, 1e-4))
        log_hi = math.log(max(cfg.omega_hi, cfg.omega_lo + 1e-3))
        outer_edges = torch.linspace(log_lo, log_hi, K_o + 1)
        omega_pieces = []
        for o in range(K_o):
            outer_lo, outer_hi = outer_edges[o].item(), outer_edges[o + 1].item()
            inner_edges = torch.linspace(outer_lo, outer_hi, K_i + 1)
            for i in range(K_i):
                inner_lo, inner_hi = inner_edges[i].item(), inner_edges[i + 1].item()
                omega_pieces.append(torch.exp(inner_lo + (inner_hi - inner_lo) * torch.rand(P)))
        omega_init = torch.cat(omega_pieces)
        self.omega = nn.Parameter(omega_init)
        self.eta = nn.Parameter(torch.full((N,), float(cfg.eta)))
        self.register_buffer("gamma", torch.full((N,), float(cfg.gamma)))
        self.register_buffer("beta", torch.full((N,), float(cfg.beta)))
        self.register_buffer("lambda_leak", torch.full((N,), float(cfg.lambda_leak)))
        self.register_buffer("kappa", torch.full((N,), float(cfg.kappa)))
        self.register_buffer("theta", torch.full((N,), float(cfg.theta)))

        scale = cfg.in_init_scale / max(math.sqrt(M), 1.0)
        self.D_re = nn.Parameter(torch.randn(N, M) * scale)
        self.D_im = nn.Parameter(torch.randn(N, M) * scale)

        # inner pool recurrence: (K_o, K_i, P, P)
        Wre_in = torch.randn(K_o, K_i, P, P) * (cfg.inner_init_scale / math.sqrt(P))
        Wim_in = torch.randn(K_o, K_i, P, P) * (cfg.inner_init_scale / math.sqrt(P))
        eye = torch.eye(P).view(1, 1, P, P)
        Wre_in = Wre_in * (1.0 - eye)
        Wim_in = Wim_in * (1.0 - eye)
        self.W_re_inner = nn.Parameter(Wre_in)
        self.W_im_inner = nn.Parameter(Wim_in)

        # cross-inner-pool recurrence: (K_o, K_i_dst, K_i_src, P, P)
        Wre_cross = torch.randn(K_o, K_i, K_i, P, P) * (cfg.outer_init_scale / math.sqrt(K_i * P))
        Wim_cross = torch.randn(K_o, K_i, K_i, P, P) * (cfg.outer_init_scale / math.sqrt(K_i * P))
        # zero same-pool entries (those are inner)
        eye_inner = torch.eye(K_i).view(1, K_i, K_i, 1, 1)
        Wre_cross = Wre_cross * (1.0 - eye_inner)
        Wim_cross = Wim_cross * (1.0 - eye_inner)
        self.W_re_cross = nn.Parameter(Wre_cross)
        self.W_im_cross = nn.Parameter(Wim_cross)

    def forward(self, x: torch.Tensor, prev_state: Optional[dict] = None,
                return_uv: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T, M = x.shape
        device, dtype = x.device, x.dtype
        K_o, K_i, P = self.cfg.n_outer, self.cfg.n_inner, self.cfg.pool_size
        N = self.N

        st = prev_state or {
            "u": torch.zeros(B, N, device=device, dtype=dtype),
            "v": torch.zeros(B, N, device=device, dtype=dtype),
            "s": torch.zeros(B, N, device=device, dtype=dtype),
        }
        u, v, s_prev = st["u"], st["v"], st["s"]

        Dr = self.D_re.t(); Di = self.D_im.t()
        eta = self.eta.view(1, 1, -1)
        xr = (x @ Dr) * eta
        xi = (x @ Di) * eta

        cos_om = torch.cos(self.omega); sin_om = torch.sin(self.omega)
        gamma = self.gamma; beta = self.beta
        lam = self.lambda_leak.clamp(min=0.0, max=0.99)
        one_minus_lam = (1.0 - lam).view(1, -1)
        kappa = self.kappa; theta = self.theta

        s_list: List[torch.Tensor] = []
        for t in range(T):
            in_r_t = xr[:, t]
            in_i_t = xi[:, t]
            u_q, v_q, q_sq, arg = _step_fractal(
                u, v, s_prev.detach(), in_r_t, in_i_t,
                cos_om, sin_om, gamma, beta, one_minus_lam, kappa, theta,
                self.W_re_inner, self.W_im_inner,
                self.W_re_cross, self.W_im_cross,
                K_o, K_i, P,
            )
            s = _SpikeFn.apply(arg, self.cfg.surr_param, 1)
            scale = 1.0 - kappa * s
            u = scale * u_q; v = scale * v_q
            u = torch.clamp(u, -3.0, 3.0); v = torch.clamp(v, -3.0, 3.0)
            s_prev = s
            s_list.append(s)

        s_seq = torch.stack(s_list, dim=1)
        info = {"final_state": {"u": u, "v": v, "s": s_prev}}
        return s_seq, info


if __name__ == "__main__":
    cfg = FractalPoolConfig(n_outer=2, n_inner=4, pool_size=8, in_dim=1,
                             omega_lo=0.1, omega_hi=2.0)
    pool = FractalResonatorPool(cfg)
    x = torch.randn(2, 50, 1)
    s, _ = pool(x)
    print("frac pool:", s.shape, "rate:", s.mean().item())
