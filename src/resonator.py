"""Stuart-Landau resonator pool, fully vectorized in real arithmetic.

Each neuron carries a complex amplitude z_i = u_i + i v_i, evolved by the
Stuart-Landau equation (rotation by omega + amplitude saturation), then
projected to a binary spike when |q|^2 > theta. Reset on spike contracts
z by factor (1 - kappa). Per-neuron parameters (omega, gamma, beta, lambda,
kappa, theta, eta) are heterogeneous and learnable.

Key design choices:
- Real-valued storage (u, v) -- complex MPS is brittle and slower.
- Recurrence applied to spikes (R2: only binary spikes cross the substrate).
- W_rec can be block-diagonal (pool structure) or dense.
- The forward returns spike sequence, optional u/v sequence, and surrogate
  derivative sequence (for local rules / surrogate-gradient BPTT).

Differentiable forward: spike emission uses an atan/rect surrogate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _fast_sigmoid_surrogate(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Zenke fast-sigmoid surrogate: 1 / (1 + (slope|x|))^2, peak slope/2 at x=0."""

    return 1.0 / (1.0 + slope * x.abs()).pow(2) * (slope / 2.0)


def _rect_surrogate(x: torch.Tensor, half_width: float) -> torch.Tensor:
    return (x.abs() < half_width).to(x.dtype) / (2.0 * half_width)


class _SpikeFn(torch.autograd.Function):
    """Heaviside on x with selectable surrogate."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope_or_w: float, kind: int):
        ctx.save_for_backward(x)
        ctx.slope_or_w = slope_or_w
        ctx.kind = kind
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        if ctx.kind == 0:  # rect
            sg = _rect_surrogate(x, ctx.slope_or_w)
        else:  # fast sigmoid
            sg = _fast_sigmoid_surrogate(x, ctx.slope_or_w)
        return grad_out * sg, None, None


def spike_fn(x: torch.Tensor, param: float = 5.0, kind: str = "fast_sigmoid") -> torch.Tensor:
    k = 0 if kind == "rect" else 1
    return _SpikeFn.apply(x, param, k)


@dataclass
class PoolConfig:
    n_pools: int           # K
    pool_size: int         # P, neurons per pool
    in_dim: int            # input feature dim per timestep
    omega_lo: float        # log-uniform omega range, low (overall band)
    omega_hi: float        # log-uniform omega range, high (overall band)
    omega_per_pool: bool = False  # if True, each pool gets its own contiguous sub-band of [lo, hi]
    gamma: float = 0.10    # leak / damping (was lambda; we keep lambda separate)
    beta: float = 0.20     # SL nonlinearity strength
    lambda_leak: float = 0.01
    kappa: float = 0.30
    theta: float = 1.00
    eta: float = 0.10
    use_recurrence: bool = True
    block_diag: bool = True  # if True, each pool has its own (P x P) W_rec
    rec_init_scale: float = 0.30
    in_init_scale: float = 2.0
    surr_param: float = 5.0           # fast-sigmoid slope (or rect half-width)
    surr_kind: str = "fast_sigmoid"   # "fast_sigmoid" or "rect"
    learn_dyn_params: bool = False    # gamma/beta/lambda/kappa/theta as buffers (stability)
    detach_recurrence: bool = True    # e-prop style: detach s_prev before W_rec
    grad_truncate: int = 0            # if > 0, truncate gradient to last K steps
    use_alif: bool = False            # adaptive threshold: ALIF-like spike-frequency adaptation
    alif_alpha: float = 0.05          # threshold increment per spike
    alif_tau: float = 100.0           # threshold adaptation decay (timesteps)

    @property
    def n_total(self) -> int:
        return self.n_pools * self.pool_size


class ResonatorPool(nn.Module):
    """A K-pool * P-neuron Stuart-Landau resonator population.

    Forward signature: (x, prev_state) -> (s_seq, info)
        x: (B, T, in_dim) real
        prev_state optional dict with u, v, prev_spike on first call (else zeros)

    Returns:
        s_seq: (B, T, N_total) binary spikes (with surrogate gradient)
        info: dict with u_seq, v_seq, q_amp_sq_seq, optionally
    """

    def __init__(self, cfg: PoolConfig):
        super().__init__()
        self.cfg = cfg
        K, P, M = cfg.n_pools, cfg.pool_size, cfg.in_dim
        N = K * P
        self.N = N

        # heterogeneous omega per neuron, log-uniform across the requested band.
        log_lo = math.log(max(cfg.omega_lo, 1e-4))
        log_hi = math.log(max(cfg.omega_hi, cfg.omega_lo + 1e-3))
        if cfg.omega_per_pool and K > 1:
            # each pool gets its own contiguous sub-band, so pools are tuned to different frequencies
            band_edges = torch.linspace(log_lo, log_hi, K + 1)
            omega_per_pool = []
            for k in range(K):
                lo_k, hi_k = band_edges[k].item(), band_edges[k + 1].item()
                omega_per_pool.append(torch.exp(lo_k + (hi_k - lo_k) * torch.rand(P)))
            omega_init = torch.cat(omega_per_pool)
        else:
            omega_init = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(N))
        gamma_init = torch.full((N,), float(cfg.gamma))
        beta_init = torch.full((N,), float(cfg.beta))
        lambda_init = torch.full((N,), float(cfg.lambda_leak))
        kappa_init = torch.full((N,), float(cfg.kappa))
        theta_init = torch.full((N,), float(cfg.theta))
        eta_init = torch.full((N,), float(cfg.eta))

        if cfg.learn_dyn_params:
            self.omega = nn.Parameter(omega_init)
            self.gamma = nn.Parameter(gamma_init)
            self.beta = nn.Parameter(beta_init)
            self.lambda_leak = nn.Parameter(lambda_init)
            self.kappa = nn.Parameter(kappa_init)
            self.theta = nn.Parameter(theta_init)
            self.eta = nn.Parameter(eta_init)
        else:
            self.register_buffer("omega", omega_init)
            self.register_buffer("gamma", gamma_init)
            self.register_buffer("beta", beta_init)
            self.register_buffer("lambda_leak", lambda_init)
            self.register_buffer("kappa", kappa_init)
            self.register_buffer("theta", theta_init)
            self.register_buffer("eta", eta_init)

        # complex input encoder D ∈ ℂ^(N×M); store as two real matrices
        scale = cfg.in_init_scale / max(math.sqrt(M), 1.0)
        self.D_re = nn.Parameter(torch.randn(N, M) * scale)
        self.D_im = nn.Parameter(torch.randn(N, M) * scale)

        # recurrent W_rec applied to spikes; complex, stored as two real matrices
        if cfg.use_recurrence:
            if cfg.block_diag:
                # (K, P, P) — each pool independent
                Wre = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
                Wim = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
                # zero diagonal of each block
                eye = torch.eye(P).unsqueeze(0)
                Wre = Wre * (1.0 - eye)
                Wim = Wim * (1.0 - eye)
                self.W_re = nn.Parameter(Wre)
                self.W_im = nn.Parameter(Wim)
            else:
                Wre = torch.randn(N, N) * (cfg.rec_init_scale / math.sqrt(N))
                Wim = torch.randn(N, N) * (cfg.rec_init_scale / math.sqrt(N))
                Wre.fill_diagonal_(0.0)
                Wim.fill_diagonal_(0.0)
                self.W_re = nn.Parameter(Wre)
                self.W_im = nn.Parameter(Wim)
        else:
            self.W_re = None
            self.W_im = None

        # for fast cos/sin during forward
        self.register_buffer("_proto_zero", torch.zeros(1))

    @property
    def K(self) -> int:
        return self.cfg.n_pools

    @property
    def P(self) -> int:
        return self.cfg.pool_size

    def init_state(self, B: int, device, dtype):
        return {
            "u": torch.zeros(B, self.N, device=device, dtype=dtype),
            "v": torch.zeros(B, self.N, device=device, dtype=dtype),
            "s": torch.zeros(B, self.N, device=device, dtype=dtype),
        }

    def _apply_rec(self, s_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (rec_re, rec_im) of shape (B, N) given previous spikes (B, N)."""

        if self.W_re is None:
            B = s_prev.shape[0]
            zero = torch.zeros(B, self.N, device=s_prev.device, dtype=s_prev.dtype)
            return zero, zero
        if self.cfg.block_diag:
            # reshape to (B, K, P), apply per-pool: (B,K,P) @ (K,P,P).T -> (B,K,P)
            B = s_prev.shape[0]
            sp = s_prev.view(B, self.K, self.P)
            re = torch.einsum("bkp,kqp->bkq", sp, self.W_re).reshape(B, self.N)
            im = torch.einsum("bkp,kqp->bkq", sp, self.W_im).reshape(B, self.N)
            return re, im
        else:
            re = s_prev @ self.W_re.t()
            im = s_prev @ self.W_im.t()
            return re, im

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[dict] = None,
        return_uv: bool = False,
        return_qsq: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        x: (B, T, M)
        Returns spikes (B, T, N) and info dict.
        """

        B, T, M = x.shape
        assert M == self.cfg.in_dim, f"in_dim mismatch: {M} vs {self.cfg.in_dim}"
        device, dtype = x.device, x.dtype

        st = prev_state or self.init_state(B, device, dtype)
        u = st["u"]
        v = st["v"]
        s_prev = st["s"]

        # precompute input projection for the whole sequence in one matmul -> (B, T, N)
        # (B, T, M) @ (M, N) = (B, T, N); transpose D
        Dr = self.D_re.t()
        Di = self.D_im.t()
        xr = x @ Dr
        xi = x @ Di
        # eta scales the input drive per neuron
        eta = self.eta.view(1, 1, -1)
        xr = xr * eta
        xi = xi * eta

        cos_om = torch.cos(self.omega)
        sin_om = torch.sin(self.omega)
        gamma = self.gamma
        beta = self.beta
        lam = self.lambda_leak.clamp(min=0.0, max=0.99)
        kappa = self.kappa
        theta = self.theta

        s_list = []
        u_list = []
        v_list = []
        qsq_list = []

        # prefactor (1 - lambda) applied to whole pre-spike linear combination
        one_minus_lam = (1.0 - lam).view(1, -1)

        for t in range(T):
            # rotation
            u_rot = cos_om * u - sin_om * v
            v_rot = sin_om * u + cos_om * v

            # SL amplitude saturation (active around limit cycle |z|^2 = 1 - gamma/beta)
            amp_sq = u * u + v * v
            sl = beta * (1.0 - amp_sq) - gamma  # combined SL+damping factor on z
            u_sl = sl * u
            v_sl = sl * v

            # recurrence on previous spikes
            rec_r, rec_i = self._apply_rec(s_prev)

            # input drive at this timestep
            in_r = xr[:, t]
            in_i = xi[:, t]

            # pre-spike state q
            u_q = one_minus_lam * (u_rot + u_sl + rec_r + in_r)
            v_q = one_minus_lam * (v_rot + v_sl + rec_i + in_i)

            # spike on |q|^2 - theta
            q_sq = u_q * u_q + v_q * v_q
            arg = q_sq - theta
            s = spike_fn(arg, self.cfg.surr_param, self.cfg.surr_kind)

            # reset (contract z by 1-kappa on spike)
            scale = 1.0 - kappa * s
            u = scale * u_q
            v = scale * v_q

            s_prev = s

            s_list.append(s)
            if return_uv:
                u_list.append(u)
                v_list.append(v)
            if return_qsq:
                qsq_list.append(q_sq)

        s_seq = torch.stack(s_list, dim=1)  # (B, T, N)
        info: dict = {
            "final_state": {"u": u, "v": v, "s": s_prev},
        }
        if return_uv:
            info["u_seq"] = torch.stack(u_list, dim=1)
            info["v_seq"] = torch.stack(v_list, dim=1)
        if return_qsq:
            info["qsq_seq"] = torch.stack(qsq_list, dim=1)
        return s_seq, info


if __name__ == "__main__":
    cfg = PoolConfig(n_pools=4, pool_size=8, in_dim=1, omega_lo=0.05, omega_hi=2.0)
    pool = ResonatorPool(cfg)
    x = torch.randn(2, 10, 1)
    s, info = pool(x, return_uv=True)
    print("spikes:", s.shape, "rate:", s.mean().item())
    print("u_seq:", info["u_seq"].shape)
