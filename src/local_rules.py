"""Biologically plausible local learning rules for the HRN.

Three-factor / e-prop framework adapted to Stuart-Landau spiking populations.

Per-parameter local rule: dW = sum_t L_post(t) * eligibility(t)

For the input encoder D ∈ ℂ^{N×M} of layer ℓ:

    pre(t)        = pre-synaptic activity (input x or previous-layer spike)
    e(t)         = forward eligibility for amplitude w.r.t. D entries
                 = ∂|q(t)|² / ∂D_re,im, traced through reset
    L_post(t)    = local post-synaptic learning signal:
                   - top-local credit at output pools (CE gradient on pool rate)
                   - feedback alignment / random projection from layer ℓ+1 spike error
                   - homeostasis on rate
                   - phase coherence (resonance tuning)

For the recurrent W_rec ∈ ℂ^{N×N}:

    pre(t)       = previous-layer's output spikes
    e(t)         = same eligibility shape but with pre being s_prev

For dynamic params (omega): a separate phase-coherence rule moves omega
toward the input's dominant frequency component the neuron is receiving.

The functions below produce GRADIENTS (post, pre) shape suitable for
direct SGD. They never call .backward(), only forward eligibility products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


EPS = 1e-12


def fast_sigmoid_surrogate(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    return 1.0 / (1.0 + slope * x.abs()).pow(2) * (slope / 2.0)


@dataclass
class PoolForwardLogs:
    """Trajectory needed for local-rule gradients.

    Stored on detached tensors. Each layer fills these during a forward
    pass with `track_logs=True`.
    """

    u_seq: torch.Tensor      # (B, T, N)
    v_seq: torch.Tensor      # (B, T, N)
    qsq_seq: torch.Tensor    # (B, T, N) pre-spike |q|^2
    s_seq: torch.Tensor      # (B, T, N)
    pre_seq: torch.Tensor    # (B, T, M) input to layer
    surrog: torch.Tensor     # (B, T, N) H'(qsq - theta)


@torch.no_grad()
def input_grad_local(
    logs: PoolForwardLogs,
    L_post: torch.Tensor,
    D_re: torch.Tensor,
    D_im: torch.Tensor,
    omega: torch.Tensor,
    eta: torch.Tensor,
    lambda_leak: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local gradient for the complex input encoder D.

    Approximates ∂|q(t)|^2 / ∂D using a forward eligibility trace:
        eu_re(t) = (1-kappa s(t-1)) * (cos(omega) eu_re - sin(omega) ev_re ...) + (1-lambda) eta * pre(t)
        ev_im(t) similarly with cos/sin coupling

    Returns (dD_re, dD_im) shape (N, M).
    """

    B, T, M = logs.pre_seq.shape
    _, _, N = logs.qsq_seq.shape
    device = logs.pre_seq.device
    dtype = logs.pre_seq.dtype

    cos_o = torch.cos(omega)
    sin_o = torch.sin(omega)
    one_minus_lam = (1.0 - lambda_leak.clamp(0.0, 0.99))

    # eligibility traces for u, v components -- shape (B, N, M)
    eu_re = torch.zeros(B, N, M, device=device, dtype=dtype)
    ev_re = torch.zeros(B, N, M, device=device, dtype=dtype)
    eu_im = torch.zeros(B, N, M, device=device, dtype=dtype)
    ev_im = torch.zeros(B, N, M, device=device, dtype=dtype)

    dD_re = torch.zeros(N, M, device=device, dtype=dtype)
    dD_im = torch.zeros(N, M, device=device, dtype=dtype)

    s_prev = torch.zeros(B, N, device=device, dtype=dtype)

    # input drive multiplier: each timestep increments e by (1-lam)*eta*pre
    # eta is per neuron -> (N,)
    in_scale = (one_minus_lam * eta).view(1, N, 1)        # broadcast over batch and pre

    for t in range(T):
        # update traces with rotation-then-reset semantics:
        # eu_q = (1-lam) [cos eu_re - sin ev_re + sl * eu_re + ... + eta * pre * δ_re_kernel]
        # We approximate: e(t+1) = decay * e(t) + drive(t+1)
        # where decay = (1 - kappa s(t-1)) on neuron axis, applied with rotation.
        decay = (1.0 - kappa * s_prev).view(B, N, 1)
        # rotation between u and v components
        new_eu_re = decay * (cos_o.view(1, N, 1) * eu_re - sin_o.view(1, N, 1) * ev_re)
        new_ev_re = decay * (sin_o.view(1, N, 1) * eu_re + cos_o.view(1, N, 1) * ev_re)
        new_eu_im = decay * (cos_o.view(1, N, 1) * eu_im - sin_o.view(1, N, 1) * ev_im)
        new_ev_im = decay * (sin_o.view(1, N, 1) * eu_im + cos_o.view(1, N, 1) * ev_im)

        eu_re = new_eu_re
        ev_re = new_ev_re
        eu_im = new_eu_im
        ev_im = new_ev_im

        pre_t = logs.pre_seq[:, t]                          # (B, M)
        # contribution to e at time t: (1-lam)*eta*pre for D_re affects u; D_im affects v
        # u_q += eta * (D_re @ x).real_part = eta * D_re @ x  (real since x real)
        # v_q += eta * (D_im @ x)
        eu_re = eu_re + in_scale * pre_t.unsqueeze(1)
        ev_im = ev_im + in_scale * pre_t.unsqueeze(1)

        # ∂|q|^2/∂D_re = 2 (u_q * eu_re + v_q * ev_re)
        # Note: ∂u_q/∂D_im = 0 (D_im doesn't enter u_q), but trace propagation
        # mixes D_re and D_im over time through rotation. Hence we keep both eu/ev
        # for both D_re and D_im.
        u_q = logs.u_seq[:, t]                              # (B, N)
        v_q = logs.v_seq[:, t]
        # local credit: L_post * H'(q^2 - theta) is already encoded in L_post if caller wants
        # Here we expect L_post(t) = local post-synaptic eligibility (B,N) including surrogate
        L_t = L_post[:, t]                                  # (B, N)

        # gradient contribution for this step
        # outer product over (N, M): sum over batch
        de_re = 2.0 * (u_q.unsqueeze(-1) * eu_re + v_q.unsqueeze(-1) * ev_re)  # (B,N,M)
        de_im = 2.0 * (u_q.unsqueeze(-1) * eu_im + v_q.unsqueeze(-1) * ev_im)
        weight = L_t.unsqueeze(-1)                                              # (B,N,1)
        dD_re = dD_re + (weight * de_re).sum(dim=0)
        dD_im = dD_im + (weight * de_im).sum(dim=0)

        s_prev = logs.s_seq[:, t]

    return dD_re, dD_im


@torch.no_grad()
def rec_grad_local(
    logs: PoolForwardLogs,
    L_post: torch.Tensor,
    omega: torch.Tensor,
    lambda_leak: torch.Tensor,
    kappa: torch.Tensor,
    block_diag: bool,
    K: int,
    P: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local gradient for the complex recurrent W_rec.

    pre is `s_seq` shifted by 1 (s(t-1)).
    Returns dW_re, dW_im in either (K,P,P) (block_diag) or (N,N) shape.
    """

    B, T, N = logs.s_seq.shape
    device = logs.s_seq.device
    dtype = logs.s_seq.dtype

    cos_o = torch.cos(omega)
    sin_o = torch.sin(omega)
    one_minus_lam = (1.0 - lambda_leak.clamp(0.0, 0.99))

    if block_diag:
        # pre source is s_prev reshaped to (B, K, P) -> rec input to same K,P
        eu_re = torch.zeros(B, K, P, P, device=device, dtype=dtype)
        ev_re = torch.zeros(B, K, P, P, device=device, dtype=dtype)
        eu_im = torch.zeros(B, K, P, P, device=device, dtype=dtype)
        ev_im = torch.zeros(B, K, P, P, device=device, dtype=dtype)
        dW_re = torch.zeros(K, P, P, device=device, dtype=dtype)
        dW_im = torch.zeros(K, P, P, device=device, dtype=dtype)
    else:
        eu_re = torch.zeros(B, N, N, device=device, dtype=dtype)
        ev_re = torch.zeros(B, N, N, device=device, dtype=dtype)
        eu_im = torch.zeros(B, N, N, device=device, dtype=dtype)
        ev_im = torch.zeros(B, N, N, device=device, dtype=dtype)
        dW_re = torch.zeros(N, N, device=device, dtype=dtype)
        dW_im = torch.zeros(N, N, device=device, dtype=dtype)

    s_prev = torch.zeros(B, N, device=device, dtype=dtype)
    # We can use a simpler "post-spike-rate * pre-spike-rate" eligibility,
    # i.e. e(t) ~ sum_τ alpha^{t-τ} s_pre(τ); applied via decay constant
    # alpha = (1 - lambda) on a fast trace.
    # That's a Hebbian e-prop with eligibility alpha = 1 - lam.
    # Below we use simple Hebbian decay traces (much cheaper than full BPTT eligibility).
    if block_diag:
        ee_re = torch.zeros(B, K, P, device=device, dtype=dtype)  # presyn trace per-pool
    else:
        ee_re = torch.zeros(B, N, device=device, dtype=dtype)

    decay_pre = (1.0 - lambda_leak.clamp(0.0, 0.99)).mean().item()  # one fast trace common across neurons

    for t in range(T):
        if block_diag:
            sp = s_prev.view(B, K, P)
            ee_re = decay_pre * ee_re + sp                         # (B, K, P) presyn trace
            # post credit
            L_t = L_post[:, t].view(B, K, P)                       # (B, K, P)
            # outer product per-pool: post (P,) x pre (P,) -> (P, P)
            # treat as Hebbian on outer(L_t, ee_re)
            outer = torch.einsum("bkp,bkq->kpq", L_t, ee_re)
            dW_re = dW_re + outer
            # imaginary direction nudge: small (use 0 for now; can be added by phase term)
            # to provide a phase-aware update we use H' along imaginary direction = 0 here
        else:
            ee_re = decay_pre * ee_re + s_prev                     # (B, N) presyn trace
            L_t = L_post[:, t]                                     # (B, N)
            outer = torch.einsum("bp,bq->pq", L_t, ee_re)
            dW_re = dW_re + outer

        s_prev = logs.s_seq[:, t]

    return dW_re, dW_im


@torch.no_grad()
def omega_grad_local(
    logs: PoolForwardLogs,
    L_post: torch.Tensor,
) -> torch.Tensor:
    """Phase-coherence based omega update.

    Move omega so that the dominant frequency in the input matches.
    Uses cross-correlation between pre-synaptic drive's instantaneous phase
    and the rotor's. Approximated locally as the time-derivative of phase.

    Returns dω of shape (N,).
    """

    B, T, N = logs.qsq_seq.shape
    if T < 2:
        return torch.zeros(N, device=logs.qsq_seq.device, dtype=logs.qsq_seq.dtype)
    u = logs.u_seq
    v = logs.v_seq
    # instantaneous phase increment (signed): atan2(v(t+1), u(t+1)) - atan2(v(t), u(t))
    phase = torch.atan2(v, u + EPS)                                 # (B, T, N)
    dphi = phase[:, 1:] - phase[:, :-1]
    # wrap to (-pi, pi)
    dphi = ((dphi + math.pi) % (2 * math.pi)) - math.pi
    # weight by L_post (the post-synaptic learning signal scales the omega update)
    L_short = L_post[:, :-1]
    domega = (L_short * dphi).sum(dim=(0, 1))                        # (N,)
    return domega


@torch.no_grad()
def homeostatic_theta_update(
    s_seq: torch.Tensor,
    target_rate: float,
    eta_theta: float = 0.001,
) -> torch.Tensor:
    """Adaptive threshold to push spike rate toward target.

    Returns delta_theta of shape (N,). theta <- theta + delta_theta.
    delta_theta > 0 -> threshold raised -> rate decreases.
    """

    rate = s_seq.mean(dim=(0, 1))                                    # (N,)
    return eta_theta * (rate - target_rate)


def softmax_topdown_signal(
    pool_rate: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    pool_size: int,
    tail_T: int,
) -> torch.Tensor:
    """Top-local CE gradient on pool rate, broadcast to per-neuron per-timestep
    learning signal at the output pools.

    Returns L_post of shape (B, T_full, n_classes * pool_size). Caller should
    only fill the tail window with this signal and zero elsewhere.

    pool_rate: (B, n_classes)
    targets:   (B,)
    """

    # softmax(logits = pool_rate) -> probs
    probs = torch.softmax(pool_rate, dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    # gradient of CE w.r.t. logits = probs - one_hot (B, n_classes)
    # we want negative gradient direction (we're minimizing CE) so use -(probs - one_hot)
    delta = -(probs - one_hot)                                       # (B, n_classes)
    # broadcast across pool_size and tail timesteps; user fills outer rest
    return delta                                                     # caller expands
