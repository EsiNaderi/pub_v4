"""Forward-only training with local rules.

Key idea: eliminate BPTT entirely. The forward pass produces:
    u_seq, v_seq, q_sq_seq, s_seq, surrog_seq

Per-layer gradients use *simple Hebbian eligibility* with a single decay
constant (much cheaper than full e-prop):

    pre_trace(t) = alpha * pre_trace(t-1) + pre(t)
    dW = sum_{b, t} L_post[b, t] outer pre_trace[b, t]

L_post is a per-neuron, per-timestep "post-synaptic credit", composed of:
    - top-local CE gradient at output class pools (only the active tail
      window has nonzero credit)
    - feedback alignment: random fixed B_l projects layer-(l+1) credit to
      layer-l credit
    - homeostatic rate term: nudge to keep rate near target
    - phase coherence: adjusts omega to track input phase

All operations are fully vectorized over (B, T, N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from resonator_jit import _SpikeFn
from hrn import HierarchicalResonantNet


@dataclass
class LocalTrainCfg:
    lr_D: float = 5e-3
    lr_W: float = 5e-3
    lr_omega: float = 1e-3
    lr_theta: float = 1e-3
    eligibility_alpha: float = 0.95           # exponential trace decay
    homeostasis_target: float = 0.10
    homeostasis_coef: float = 1.0
    fb_align_scale: float = 1.0
    use_feedback_alignment: bool = True
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    train_omega: bool = True
    train_theta: bool = True
    surrogate_slope: float = 5.0


def fast_sigmoid_surrogate(x: torch.Tensor, slope: float) -> torch.Tensor:
    return 1.0 / (1.0 + slope * x.abs()).pow(2) * (slope / 2.0)


@torch.no_grad()
def hrn_forward_with_logs(net: HierarchicalResonantNet, x: torch.Tensor) -> dict:
    """Run the full HRN forward, recording per-layer logs needed for local rules.

    Returns:
        layer_logs: list of dicts (one per layer):
            "u_seq": (B, T, N)
            "v_seq": (B, T, N)
            "q_sq_seq": (B, T, N)
            "s_seq": (B, T, N)
            "pre_seq": (B, T, M)  # input to this layer
            "surrog": (B, T, N)
        pool_rate: (B, n_classes) for the output layer
        logits: (B, n_classes)
    """

    B, T, _ = x.shape
    sig = x
    layer_logs = []
    for li, layer in enumerate(net.layers):
        cfg = layer.cfg
        K, P = cfg.n_pools, cfg.pool_size
        N = layer.N
        device, dtype = sig.device, sig.dtype
        u = torch.zeros(B, N, device=device, dtype=dtype)
        v = torch.zeros(B, N, device=device, dtype=dtype)
        s_prev = torch.zeros(B, N, device=device, dtype=dtype)

        Dr = layer.D_re.t()
        Di = layer.D_im.t()
        eta = layer.eta.view(1, 1, -1)
        xr = (sig @ Dr) * eta
        xi = (sig @ Di) * eta

        cos_om = torch.cos(layer.omega)
        sin_om = torch.sin(layer.omega)
        gamma = layer.gamma; beta = layer.beta
        lam = layer.lambda_leak.clamp(min=0.0, max=0.99)
        one_minus_lam = (1.0 - lam).view(1, -1)
        kappa = layer.kappa
        theta = layer.theta

        u_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        v_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        q_sq_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        s_seq = torch.empty(B, T, N, device=device, dtype=dtype)

        for t in range(T):
            u_rot = cos_om * u - sin_om * v
            v_rot = sin_om * u + cos_om * v
            amp_sq = u * u + v * v
            sl = beta * (1.0 - amp_sq) - gamma
            u_sl = sl * u
            v_sl = sl * v
            if cfg.block_diag:
                sp = s_prev.view(B, K, P)
                rec_r = torch.einsum("bkp,kqp->bkq", sp, layer.W_re).reshape(B, N)
                rec_i = torch.einsum("bkp,kqp->bkq", sp, layer.W_im).reshape(B, N)
            else:
                rec_r = s_prev @ layer.W_re.t()
                rec_i = s_prev @ layer.W_im.t()
            u_q = one_minus_lam * (u_rot + u_sl + rec_r + xr[:, t])
            v_q = one_minus_lam * (v_rot + v_sl + rec_i + xi[:, t])
            q_sq = u_q * u_q + v_q * v_q
            arg = q_sq - theta
            s = (arg > 0).to(dtype)
            scale = 1.0 - kappa * s
            u = scale * u_q
            v = scale * v_q
            s_prev = s
            u_seq[:, t] = u_q     # store pre-spike state (q-space)
            v_seq[:, t] = v_q
            q_sq_seq[:, t] = q_sq
            s_seq[:, t] = s

        surrog = fast_sigmoid_surrogate(q_sq_seq - layer.theta.view(1, 1, -1), 5.0)

        layer_logs.append({
            "u_seq": u_seq, "v_seq": v_seq,
            "q_sq_seq": q_sq_seq, "s_seq": s_seq,
            "pre_seq": sig, "surrog": surrog,
            "K": K, "P": P, "N": N,
        })
        sig = s_seq

    out_seq = layer_logs[-1]["s_seq"]
    out_seq_view = out_seq.view(B, T, net.cfg.n_classes, net.cfg.out_pool_size)
    tail = max(1, int(round(T * net.cfg.tail_fraction)))
    pool_rate = out_seq_view[:, T - tail :].mean(dim=(1, 3))
    logits = pool_rate * net.cfg.readout_temperature

    return {
        "layer_logs": layer_logs,
        "pool_rate": pool_rate,
        "logits": logits,
        "tail": tail,
    }


@torch.no_grad()
def make_eligibility_trace(seq: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving-average trace.

    seq: (B, T, M)
    Returns trace of same shape, where trace[t] = alpha*trace[t-1] + seq[t].
    Implemented via simple loop (cheap; the heavy lifting is the per-step ops).

    Faster vectorized form: 1D recursive linear filter (lfilter), but our
    seq is small enough that the loop is fine.
    """

    B, T, M = seq.shape
    out = torch.empty_like(seq)
    tr = torch.zeros(B, M, device=seq.device, dtype=seq.dtype)
    for t in range(T):
        tr = alpha * tr + seq[:, t]
        out[:, t] = tr
    return out


@torch.no_grad()
def class_pool_l_post(
    pool_rate: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    pool_size: int,
    T: int,
    tail: int,
    surrog_tail: torch.Tensor,
    temperature: float = 5.0,
) -> torch.Tensor:
    """Output-pool L_post: CE gradient on logits, broadcast to per-neuron credit.

    surrog_tail: (B, tail, N_out) surrogate at output pools, in tail window.
    Returns L_post of shape (B, T, N_out) with zero outside tail window.
    """

    B = pool_rate.shape[0]
    N_out = n_classes * pool_size
    probs = torch.softmax(pool_rate * temperature, dim=1)        # (B, n_classes)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    delta = (one_hot - probs)                                    # negative gradient (we move w.r.t. -CE)
    # broadcast across pool_size and tail T
    delta_neuron = delta.unsqueeze(-1).expand(-1, -1, pool_size).reshape(B, N_out) / max(1, tail * pool_size)
    L_post = torch.zeros(B, T, N_out, device=pool_rate.device, dtype=pool_rate.dtype)
    L_post[:, T - tail:] = delta_neuron.unsqueeze(1)
    # multiply by surrogate so credit flows where the spike threshold is sensitive
    L_post[:, T - tail:] = L_post[:, T - tail:] * surrog_tail
    return L_post


@torch.no_grad()
def compute_layer_grads(
    net: HierarchicalResonantNet,
    fwd: dict,
    targets: torch.Tensor,
    cfg: LocalTrainCfg,
    fb_proj: List[torch.Tensor],
) -> dict:
    """Compute local gradients for all layers.

    fb_proj: random feedback alignment matrices, one per inter-layer interface.
        fb_proj[l] has shape (N_l, N_{l+1}) and projects layer-(l+1) credit
        back to layer-l credit. (Only the diagonal of credit is propagated;
        the actual credit signal is L_post per neuron.)
    """

    layer_logs = fwd["layer_logs"]
    pool_rate = fwd["pool_rate"]
    tail = fwd["tail"]
    n_layers = len(layer_logs)
    out_layer = net.layers[-1]
    cfg_net = net.cfg
    pool_size = cfg_net.out_pool_size
    n_classes = cfg_net.n_classes

    # 1) build L_post for output layer (top-local credit)
    last_logs = layer_logs[-1]
    T = last_logs["s_seq"].shape[1]
    surrog_out = last_logs["surrog"]                                 # (B, T, N_out)
    L_post_layers: List[torch.Tensor] = [None] * n_layers
    L_post_layers[-1] = class_pool_l_post(
        pool_rate, targets, n_classes, pool_size, T, tail,
        surrog_out[:, T - tail:], cfg_net.readout_temperature,
    )

    # 2) feedback alignment propagation back to earlier layers
    for l in range(n_layers - 2, -1, -1):
        if cfg.use_feedback_alignment:
            B_lp1 = fb_proj[l]                                           # (N_l, N_{l+1})
            # propagate per-timestep credit
            L_lp1 = L_post_layers[l + 1]                                  # (B, T, N_{l+1})
            L_l = L_lp1 @ B_lp1.t() * cfg.fb_align_scale                   # (B, T, N_l)
        else:
            L_l = torch.zeros_like(layer_logs[l]["surrog"])
        # multiply by surrogate of current layer
        L_l = L_l * layer_logs[l]["surrog"]
        # add homeostatic component
        rate = layer_logs[l]["s_seq"].mean(dim=(0, 1))                    # (N_l,)
        homeo = -cfg.homeostasis_coef * (rate - cfg.homeostasis_target).view(1, 1, -1)
        L_l = L_l + homeo
        L_post_layers[l] = L_l

    # 3) compute gradients per layer
    grads = {}
    for l, logs in enumerate(layer_logs):
        layer = net.layers[l]
        K, P, N = logs["K"], logs["P"], logs["N"]
        L_post = L_post_layers[l]                                          # (B, T, N)

        # Eligibility traces of inputs and recurrent spikes
        pre_seq = logs["pre_seq"]                                          # (B, T, M)
        s_seq = logs["s_seq"]                                              # (B, T, N)

        pre_trace = make_eligibility_trace(pre_seq, cfg.eligibility_alpha)
        # for recurrent, use s_seq shifted by 1 (s_prev)
        s_shift = torch.cat([torch.zeros_like(s_seq[:, :1]), s_seq[:, :-1]], dim=1)
        rec_trace = make_eligibility_trace(s_shift, cfg.eligibility_alpha)

        # gradient for input encoder D
        # Decompose D_re vs D_im with phase factor: input drive in u uses D_re, in v uses D_im.
        # Local rule splits credit by u-direction and v-direction. We project L_post via
        # post-state phase: ∂q²/∂D_re ∝ u_q ; ∂q²/∂D_im ∝ v_q.
        u_q = logs["u_seq"]; v_q = logs["v_seq"]
        # eta scales drive
        eta = layer.eta.view(1, 1, -1)
        # gradient contributions
        L_u = (L_post * 2.0 * u_q * eta)                                   # (B, T, N)
        L_v = (L_post * 2.0 * v_q * eta)
        dD_re = torch.einsum("btn,btm->nm", L_u, pre_trace)
        dD_im = torch.einsum("btn,btm->nm", L_v, pre_trace)

        # gradient for recurrent W (block-diagonal or dense)
        if layer.cfg.use_recurrence:
            rec_pre = rec_trace                                             # (B, T, N)
            if layer.cfg.block_diag:
                # reshape: pre is (B, T, K, P), post is (B, T, K, P)
                pre_blk = rec_pre.view(rec_pre.shape[0], rec_pre.shape[1], K, P)
                Lu_blk = L_u.view(L_u.shape[0], L_u.shape[1], K, P)
                Lv_blk = L_v.view(L_v.shape[0], L_v.shape[1], K, P)
                dW_re = torch.einsum("btkp,btkq->kpq", Lu_blk, pre_blk)
                dW_im = torch.einsum("btkp,btkq->kpq", Lv_blk, pre_blk)
            else:
                dW_re = torch.einsum("btn,btm->nm", L_u, rec_pre)
                dW_im = torch.einsum("btn,btm->nm", L_v, rec_pre)
        else:
            dW_re = None
            dW_im = None

        # gradient for omega: phase coherence rule
        # increment in phase: dphi(t) = atan2(v(t), u(t)) - atan2(v(t-1), u(t-1))
        # We want omega such that dphi matches the post-credit-weighted average.
        # Use a simpler form: domega ∝ <L_post, theta_imag_part_of_drive>, but we
        # approximate by: domega = sum_t L_post(t) * (u(t) v_q(t) - v(t) u_q(t)) / (q_sq + eps)
        if cfg.train_omega:
            num = u_q * v_q - v_q * u_q  # zero! degenerate. Use a different form.
            # Better: domega = -sum_t L_post * d|q|^2 / domega
            # |q|^2 depends on omega via the rotation: u_rot = cos(om) u - sin(om) v
            # ∂u_rot/∂om = -sin(om) u - cos(om) v = -v_rot
            # ∂v_rot/∂om = cos(om) u - sin(om) v = u_rot  (NOT EXACT here since we don't store u_rot)
            # We approximate using u_q and v_q:
            #   ∂q²/∂omega ≈ 2 (u_q * (-v_q) + v_q * u_q) = 0  (also degenerate)
            # The phase eligibility we want is: domega(t) = phase(q(t)) * H'(amp - theta)
            # i.e. shift omega so that the post-credit phase matches the input phase.
            # Use cross-product based update:
            # domega ∝ sum_t L_post * (u_prev * v_q - v_prev * u_q)
            # (this is the imag part of conjugate(z_prev) * q which = sin(phase_diff))
            u_prev = torch.cat([torch.zeros_like(u_q[:, :1]), u_q[:, :-1]], dim=1)
            v_prev = torch.cat([torch.zeros_like(v_q[:, :1]), v_q[:, :-1]], dim=1)
            cross = u_prev * v_q - v_prev * u_q                               # (B, T, N)
            domega = (L_post * cross).sum(dim=(0, 1))                         # (N,)
        else:
            domega = torch.zeros_like(layer.omega)

        # homeostatic theta update
        if cfg.train_theta:
            rate = s_seq.mean(dim=(0, 1))                                     # (N,)
            dtheta = (rate - cfg.homeostasis_target)                          # positive => raise threshold
        else:
            dtheta = torch.zeros_like(layer.theta)

        grads[l] = {
            "dD_re": dD_re, "dD_im": dD_im,
            "dW_re": dW_re, "dW_im": dW_im,
            "domega": domega, "dtheta": dtheta,
        }
    return grads


def apply_local_grads(net: HierarchicalResonantNet, grads: dict, cfg: LocalTrainCfg) -> None:
    """Apply per-layer gradients with simple SGD (Adam later)."""

    for l, layer in enumerate(net.layers):
        g = grads[l]
        with torch.no_grad():
            # weight gradients are *post-update direction*: we add g (positive => stronger)
            # subject to clip and weight decay
            for name, p, dp, lr in [
                ("D_re", layer.D_re, g["dD_re"], cfg.lr_D),
                ("D_im", layer.D_im, g["dD_im"], cfg.lr_D),
                ("W_re", layer.W_re if layer.cfg.use_recurrence else None, g["dW_re"], cfg.lr_W),
                ("W_im", layer.W_im if layer.cfg.use_recurrence else None, g["dW_im"], cfg.lr_W),
                ("omega", layer.omega, g["domega"], cfg.lr_omega),
                ("theta", layer.theta, g["dtheta"], cfg.lr_theta),
            ]:
                if p is None or dp is None:
                    continue
                if cfg.weight_decay > 0:
                    dp = dp - cfg.weight_decay * p
                if cfg.grad_clip_norm > 0:
                    n = dp.norm()
                    if n > cfg.grad_clip_norm:
                        dp = dp * (cfg.grad_clip_norm / n.clamp_min(1e-12))
                # take a step in the gradient *direction* (we already negated CE inside L_post)
                p.add_(lr * dp)


def init_feedback_projections(net: HierarchicalResonantNet, scale: float = 0.1) -> List[torch.Tensor]:
    """Random fixed feedback alignment matrices. fb[l]: (N_l, N_{l+1}).
    Returns a list of length n_layers - 1.
    """

    fbs = []
    n_l = len(net.layers)
    for l in range(n_l - 1):
        N_l = net.layers[l].N
        N_lp1 = net.layers[l + 1].N
        fbs.append(torch.randn(N_l, N_lp1) * (scale / math.sqrt(N_lp1)))
    return fbs
