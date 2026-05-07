"""JIT-scripted resonator step for tighter Python-loop performance.

The innermost per-timestep body (rotation, SL, recurrence apply, spike,
reset) is wrapped in torch.jit.script. Outer loop and gradient propagation
remain in eager PyTorch. Recurrence is supported in dense and block-diag
forms.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from resonator import _SpikeFn, PoolConfig


@torch.jit.script
def _step_block(
    u: torch.Tensor, v: torch.Tensor, s_prev: torch.Tensor,
    in_r_t: torch.Tensor, in_i_t: torch.Tensor,
    cos_om: torch.Tensor, sin_om: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor,
    one_minus_lam: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor,
    W_re_blk: torch.Tensor, W_im_blk: torch.Tensor,
    K: int, P: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # rotation
    u_rot = cos_om * u - sin_om * v
    v_rot = sin_om * u + cos_om * v
    # SL + damping (clamp amp_sq to prevent runaway from numerical drift)
    # Detach amp_sq when computing SL: prevents cubic feedback gradient
    # explosion. The forward dynamics are unchanged; the backward path
    # treats the SL coefficient as a static gating factor at each step.
    amp_sq_raw = u * u + v * v
    amp_sq = amp_sq_raw.detach().clamp(max=10.0)
    sl = beta * (1.0 - amp_sq) - gamma
    u_sl = sl * u
    v_sl = sl * v
    # block recurrence
    B = s_prev.shape[0]
    sp = s_prev.view(B, K, P)
    rec_r = torch.einsum("bkp,kqp->bkq", sp, W_re_blk).reshape(B, K * P)
    rec_i = torch.einsum("bkp,kqp->bkq", sp, W_im_blk).reshape(B, K * P)
    u_q = one_minus_lam * (u_rot + u_sl + rec_r + in_r_t)
    v_q = one_minus_lam * (v_rot + v_sl + rec_i + in_i_t)
    q_sq = u_q * u_q + v_q * v_q
    arg = q_sq - theta
    return u_q, v_q, q_sq, arg


@torch.jit.script
def _step_dense(
    u: torch.Tensor, v: torch.Tensor, s_prev: torch.Tensor,
    in_r_t: torch.Tensor, in_i_t: torch.Tensor,
    cos_om: torch.Tensor, sin_om: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor,
    one_minus_lam: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor,
    W_re: torch.Tensor, W_im: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_rot = cos_om * u - sin_om * v
    v_rot = sin_om * u + cos_om * v
    # Detach amp_sq when computing SL: prevents cubic feedback gradient
    # explosion. The forward dynamics are unchanged; the backward path
    # treats the SL coefficient as a static gating factor at each step.
    amp_sq_raw = u * u + v * v
    amp_sq = amp_sq_raw.detach().clamp(max=10.0)
    sl = beta * (1.0 - amp_sq) - gamma
    u_sl = sl * u
    v_sl = sl * v
    rec_r = s_prev @ W_re.t()
    rec_i = s_prev @ W_im.t()
    u_q = one_minus_lam * (u_rot + u_sl + rec_r + in_r_t)
    v_q = one_minus_lam * (v_rot + v_sl + rec_i + in_i_t)
    q_sq = u_q * u_q + v_q * v_q
    arg = q_sq - theta
    return u_q, v_q, q_sq, arg


class ResonatorPoolJIT(nn.Module):
    def __init__(self, cfg: PoolConfig):
        super().__init__()
        self.cfg = cfg
        K, P, M = cfg.n_pools, cfg.pool_size, cfg.in_dim
        N = K * P
        self.N = N

        log_lo = math.log(max(cfg.omega_lo, 1e-4))
        log_hi = math.log(max(cfg.omega_hi, cfg.omega_lo + 1e-3))
        if cfg.omega_per_pool and K > 1:
            band_edges = torch.linspace(log_lo, log_hi, K + 1)
            omega_per_pool_list = []
            for k in range(K):
                lo_k, hi_k = band_edges[k].item(), band_edges[k + 1].item()
                omega_per_pool_list.append(torch.exp(lo_k + (hi_k - lo_k) * torch.rand(P)))
            omega_init = torch.cat(omega_per_pool_list)
        else:
            omega_init = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(N))
        gamma_init = torch.full((N,), float(cfg.gamma))
        beta_init = torch.full((N,), float(cfg.beta))
        lambda_init = torch.full((N,), float(cfg.lambda_leak))
        kappa_init = torch.full((N,), float(cfg.kappa))
        theta_init = torch.full((N,), float(cfg.theta))
        eta_init = torch.full((N,), float(cfg.eta))

        # Only the frequency/input/coupling params are learnable by default.
        # gamma, beta, kappa, lambda, theta are STABILITY parameters that we
        # keep as buffers unless cfg.learn_dyn_params is True. (BPTT through
        # SL with theta learnable diverges quickly.)
        self.omega = nn.Parameter(omega_init)
        self.eta = nn.Parameter(eta_init)
        if cfg.learn_dyn_params:
            self.gamma = nn.Parameter(gamma_init)
            self.beta = nn.Parameter(beta_init)
            self.lambda_leak = nn.Parameter(lambda_init)
            self.kappa = nn.Parameter(kappa_init)
            self.theta = nn.Parameter(theta_init)
        else:
            self.register_buffer("gamma", gamma_init)
            self.register_buffer("beta", beta_init)
            self.register_buffer("lambda_leak", lambda_init)
            self.register_buffer("kappa", kappa_init)
            self.register_buffer("theta", theta_init)

        scale = cfg.in_init_scale / max(math.sqrt(M), 1.0)
        self.D_re = nn.Parameter(torch.randn(N, M) * scale)
        self.D_im = nn.Parameter(torch.randn(N, M) * scale)

        if cfg.use_recurrence:
            if cfg.block_diag:
                Wre = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
                Wim = torch.randn(K, P, P) * (cfg.rec_init_scale / math.sqrt(P))
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
            zero_blk = torch.zeros(K, P, P) if cfg.block_diag else torch.zeros(N, N)
            self.register_buffer("W_re", zero_blk.clone())
            self.register_buffer("W_im", zero_blk.clone())

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

    def forward(self, x: torch.Tensor, prev_state: Optional[dict] = None,
                return_uv: bool = False, return_qsq: bool = False) -> Tuple[torch.Tensor, dict]:
        B, T, M = x.shape
        device, dtype = x.device, x.dtype

        st = prev_state or self.init_state(B, device, dtype)
        u = st["u"]; v = st["v"]; s_prev = st["s"]

        Dr = self.D_re.t(); Di = self.D_im.t()
        eta = self.eta.view(1, 1, -1)
        xr = (x @ Dr) * eta
        xi = (x @ Di) * eta

        cos_om = torch.cos(self.omega)
        sin_om = torch.sin(self.omega)
        gamma = self.gamma; beta = self.beta
        lam = self.lambda_leak.clamp(min=0.0, max=0.99)
        one_minus_lam = (1.0 - lam).view(1, -1)
        kappa = self.kappa
        theta = self.theta
        # ALIF state and decay
        theta_adapt = torch.zeros(B, self.N, device=device, dtype=dtype)
        alif_decay = math.exp(-1.0 / max(self.cfg.alif_tau, 1.0))

        s_list: List[torch.Tensor] = []
        u_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        qsq_list: List[torch.Tensor] = []

        block = self.cfg.block_diag
        K = self.K; P = self.P
        detach_rec = self.cfg.detach_recurrence
        gtrunc = self.cfg.grad_truncate
        # if grad_truncate > 0, detach state at t = T - gtrunc so backward only flows through last gtrunc steps
        cutoff = max(0, T - gtrunc) if gtrunc > 0 else -1

        for t in range(T):
            in_r_t = xr[:, t]
            in_i_t = xi[:, t]
            if cutoff >= 0 and t == cutoff:
                u = u.detach()
                v = v.detach()
                s_prev = s_prev.detach()
            sp_for_rec = s_prev.detach() if detach_rec else s_prev
            if block:
                u_q, v_q, q_sq, arg = _step_block(
                    u, v, sp_for_rec, in_r_t, in_i_t,
                    cos_om, sin_om, gamma, beta, one_minus_lam, kappa, theta,
                    self.W_re, self.W_im, K, P,
                )
            else:
                u_q, v_q, q_sq, arg = _step_dense(
                    u, v, sp_for_rec, in_r_t, in_i_t,
                    cos_om, sin_om, gamma, beta, one_minus_lam, kappa, theta,
                    self.W_re, self.W_im,
                )
            # If ALIF, adjust effective threshold by adaptive component.
            # arg already = q_sq - theta. We additionally subtract theta_adapt per neuron.
            if self.cfg.use_alif:
                arg = arg - theta_adapt
            s = _SpikeFn.apply(arg, self.cfg.surr_param, 1)
            scale = 1.0 - kappa * s
            u = scale * u_q
            v = scale * v_q
            u = torch.clamp(u, -3.0, 3.0)
            v = torch.clamp(v, -3.0, 3.0)
            s_prev = s
            # Update adaptive threshold: theta_adapt = decay * theta_adapt + alpha * s
            if self.cfg.use_alif:
                theta_adapt = alif_decay * theta_adapt + self.cfg.alif_alpha * s

            s_list.append(s)
            if return_uv:
                u_list.append(u); v_list.append(v)
            if return_qsq:
                qsq_list.append(q_sq)

        s_seq = torch.stack(s_list, dim=1)
        info: dict = {"final_state": {"u": u, "v": v, "s": s_prev}}
        if return_uv:
            info["u_seq"] = torch.stack(u_list, dim=1)
            info["v_seq"] = torch.stack(v_list, dim=1)
        if return_qsq:
            info["qsq_seq"] = torch.stack(qsq_list, dim=1)
        return s_seq, info
