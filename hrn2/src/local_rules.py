"""Local learning rules for HRN-v2.

All updates are per-neuron and use only locally available signals
(eligibility traces, post-spike-time activity, label tags from
neighbours, homeostatic running averages). No BPTT, no backprop
through other neurons.

Components:
- adaptive_mean_competition: pool-level WTA with no temperature.
- label_hebbian_step: per-neuron label-mass Hebbian update.
- usage_homeostasis: per-neuron threshold + usage-EMA homeostasis.
- inter_stage_credit: random-feedback alignment (Lillicrap) with
  optional Hebbian gating, used to send a credit signal from a
  later stage to an earlier stage.
- credit_for_softmax_pool: convert per-class-pool tail energies into
  a per-neuron credit signal (`d L / d E_i`) via softmax CE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-12


def adaptive_mean_competition(
    energy: torch.Tensor,                 # (B, K_pools, M_neurons)
    theta: torch.Tensor,                  # (K_pools, M_neurons)
    usage_penalty_coef: float = 0.0,
    usage_ema: torch.Tensor | None = None,  # (K_pools, M_neurons)
    target_usage: float = 0.0625,
) -> torch.Tensor:
    """Pool-level adaptive-mean WTA.

    Returns r ∈ [0, 1] per (B, K, M) where Σ_m r_{b, k, m} ≤ 1 in each
    pool. If a pool's energy is uniformly below the field, the pool
    response is uniform = 1 / M (matching pub_v3's behaviour).
    """
    if usage_penalty_coef > 0 and usage_ema is not None:
        penalty = usage_penalty_coef * torch.log((usage_ema / max(target_usage, EPS)).clamp_min(EPS))
        u = energy - theta - penalty.unsqueeze(0)
    else:
        u = energy - theta.unsqueeze(0)
    excess = (u - u.mean(dim=2, keepdim=True)).clamp_min(0.0)
    mass = excess.sum(dim=2, keepdim=True)
    M = energy.shape[2]
    uniform = torch.full_like(energy, 1.0 / M)
    return torch.where(mass > EPS, excess / mass.clamp_min(EPS), uniform)


def softmax_competition(
    energy: torch.Tensor,                 # (B, K, M)
    theta: torch.Tensor,                  # (K, M)
    beta: float,
    top_k: int = 0,
) -> torch.Tensor:
    """Pool-level softmax competition with optional top-k mask."""
    u = energy - theta.unsqueeze(0)
    resp = torch.softmax(beta * u, dim=2)
    if top_k > 0 and top_k < energy.shape[2]:
        keep = resp.topk(top_k, dim=2).indices
        mask = torch.zeros_like(resp)
        mask.scatter_(2, keep, 1.0)
        resp = resp * mask
        resp = resp / resp.sum(dim=2, keepdim=True).clamp_min(EPS)
    return resp


def global_softmax_competition(
    energy: torch.Tensor,                 # (B, P)
    theta: torch.Tensor,                  # (P,)
    beta: float,
    top_k: int = 0,
) -> torch.Tensor:
    """Single-pool softmax competition matching pub_v3's resonant_self_organizing_layer."""
    u = energy - theta.unsqueeze(0)
    resp = torch.softmax(beta * u, dim=1)
    if top_k > 0 and top_k < energy.shape[1]:
        keep = resp.topk(top_k, dim=1).indices
        mask = torch.zeros_like(resp)
        mask.scatter_(1, keep, 1.0)
        resp = resp * mask
        resp = resp / resp.sum(dim=1, keepdim=True).clamp_min(EPS)
    return resp


def label_probs(label_mass: torch.Tensor, prior: float, classes: int) -> torch.Tensor:
    """Per-neuron class distribution q_i(c) from label_mass + uniform prior.

    label_mass: (P, C). Returns (P, C) row-normalized.
    """
    q = label_mass + (prior / classes)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(EPS)
    return q


def label_hebbian_step(
    label_mass: torch.Tensor,             # (P, C)
    resp: torch.Tensor,                   # (B, P)  flattened pool dims
    y: torch.Tensor,                      # (B,) int
    classes: int,
    lr: float,
    decay: float = 0.0,
    tag_power: float = 1.0,
) -> None:
    """Per-neuron label-mass Hebbian update (in-place).

    For each neuron i and sample b: label_mass[i, y_b] += lr * resp[b, i]^p
    """
    onehot = F.one_hot(y, classes).float()
    with torch.no_grad():
        tag_resp = resp.clamp_min(EPS).pow(tag_power)
        tag_resp = tag_resp / tag_resp.sum(dim=1, keepdim=True).clamp_min(EPS)
        if decay > 0:
            label_mass.mul_(1.0 - decay)
        label_mass.add_(lr * (tag_resp.t() @ onehot))


def usage_homeostasis(
    theta: torch.Tensor,                  # (K, M)
    usage_ema: torch.Tensor,              # (K, M)
    resp: torch.Tensor,                   # (B, K, M)
    target_usage: float,
    homeo_lr: float,
    ema_lr: float,
    min_theta: float = -2.0,
    max_theta: float = 5.0,
) -> None:
    with torch.no_grad():
        usage = resp.mean(dim=0)                      # (K, M)
        usage_ema.mul_(1.0 - ema_lr).add_(ema_lr * usage)
        theta.add_(homeo_lr * (usage - target_usage))
        theta.clamp_(min_theta, max_theta)


def credit_for_class_pool_softmax(
    pool_logits: torch.Tensor,            # (B, C)  one logit per class pool
    y: torch.Tensor,                      # (B,)
    classes: int,
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """Standard softmax CE credit at the class-pool level.

    Returns dL/d_logit per (B, C). Each class-pool logit is the mean tail
    energy of that pool. Distributing this credit across pool members is
    a separate step (each member gets the same δ_i = δ_class / M).
    """
    probs = torch.softmax(softmax_temperature * pool_logits, dim=1)
    onehot = F.one_hot(y, classes).float()
    return (probs - onehot) / max(pool_logits.shape[0], 1)


def credit_for_self_organising_pool(
    resp: torch.Tensor,                   # (B, P)
    label_mass: torch.Tensor,             # (P, C)
    y: torch.Tensor,                      # (B,)
    classes: int,
    label_prior: float = 2.0,
    credit_gain: float = 1.0,
) -> torch.Tensor:
    """Credit per neuron for the self-organising (label-mass) head.

    Mirrors pub_v3's resonant_self_organizing_layer rule:
        P(c) = sum_i resp_i * q_i(c)
        loss = -log P(y)
        dL/dE_i = -dE_i_credit_scale * resp_i * (1 - q_i(y) / P(y))

    Returns (B, P).
    """
    q = label_probs(label_mass, label_prior, classes)
    probs = resp @ q                                  # (B, C)
    py = probs[torch.arange(probs.shape[0]), y].clamp_min(EPS)
    qy = q[:, y].t()                                  # (B, P)
    delta_amp = credit_gain * resp * (1.0 - qy / py.unsqueeze(1)) / max(probs.shape[0], 1)
    return delta_amp


def credit_for_class_pool_per_neuron(
    pool_logits_credit: torch.Tensor,     # (B, C)
    pool_index: torch.Tensor,             # (P,) class index per neuron
    classes: int,
    M_per_pool: int,
) -> torch.Tensor:
    """Distribute per-class-pool credit to per-neuron credit.

    Each neuron in pool c gets dL/dE_i = dL/dlogit_c / M_per_pool because
    logit_c = (1/M) * sum_{i in pool c} E_i.
    """
    B = pool_logits_credit.shape[0]
    delta = pool_logits_credit[:, pool_index] / M_per_pool   # (B, P)
    return delta


def random_feedback(
    delta_post: torch.Tensor,             # (B, P_post)
    feedback_matrix: torch.Tensor,        # (P_post, P_pre)  random fixed
    *,
    rescale: bool = True,
) -> torch.Tensor:
    """Lillicrap random feedback alignment.

    delta_post @ feedback_matrix gives a credit estimate for the upstream
    layer. Optionally rescales each pre-neuron's credit by 1/sqrt(P_post)
    for variance balance.
    """
    delta_pre = delta_post @ feedback_matrix
    if rescale:
        scale = 1.0 / max(1.0, delta_post.shape[1] ** 0.5)
        delta_pre = delta_pre * scale
    return delta_pre


def hebbian_gated_credit(
    delta_pre: torch.Tensor,              # (B, P_pre)
    pre_activity: torch.Tensor,           # (B, P_pre)
    *,
    gate_threshold: float = 0.0,
) -> torch.Tensor:
    """Multiplicative gate: only update neurons that were active.

    Useful when delta_pre comes from random feedback alignment — pre
    neurons that didn't participate this batch shouldn't get credit.
    """
    gate = (pre_activity > gate_threshold).to(delta_pre.dtype)
    return delta_pre * gate
