"""Layer-wise greedy training: train each layer's INPUT WEIGHTS (D_re, D_im)
using a local Hebbian rule, then train the output head supervisedly.

The idea is:
- Layer ℓ's D_re/D_im are tuned to maximize variance/independence of pool
  activations on real data (Oja-like or k-means-style competitive learning).
- Recurrence W_rec stays as a random reservoir (frozen).
- Once hidden layers are tuned, supervised training is only on output.

This is a fully-feedforward credit assignment scheme: no global gradient
required, just local Hebbian + competitive rules.

Concretely: use a competitive Hebbian rule on each pool's neurons. For
pool k with input pre(t):
  winner = argmax_{i in k} z_i(t)
  D_winner += eta_lr * pre(t) * z_winner(t)
  (other neurons in pool get smaller updates via soft-max)

This drives each pool's neurons to specialize in different input patterns.

For pool-of-pools, run per-pool independently.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import load_smnist
from hrn import HierarchicalResonantNet, make_default_config
from train_bptt import build_net


@torch.no_grad()
def hebbian_input_step(layer, x_seq, lr_in: float = 1e-3):
    """One pass of Hebbian update on layer.D_re/D_im using x_seq.

    For each timestep t, run forward to get u(t), v(t). Then update:
      D_re[i, m] += lr_in * (x[t,m] * u[t,i] / |z[t,i]|)
      D_im[i, m] += lr_in * (x[t,m] * v[t,i] / |z[t,i]|)

    Effectively: D moves toward the input direction the most-active neuron sees,
    weighted by amplitude. Plus normalization to prevent runaway.
    """

    B, T, M = x_seq.shape
    cfg = layer.cfg
    K, P, N = cfg.n_pools, cfg.pool_size, layer.N

    # forward to get u, v sequences
    u = torch.zeros(B, N); v = torch.zeros(B, N); s_prev = torch.zeros(B, N)
    Dr = layer.D_re.t(); Di = layer.D_im.t()
    eta = layer.eta.view(1, 1, -1)
    xr = (x_seq @ Dr) * eta
    xi = (x_seq @ Di) * eta
    cos_om = torch.cos(layer.omega); sin_om = torch.sin(layer.omega)
    gamma = layer.gamma; beta = layer.beta
    lam = layer.lambda_leak.clamp(min=0.0, max=0.99)
    one_minus_lam = (1.0 - lam).view(1, -1)
    kappa = layer.kappa; theta = layer.theta

    dD_re = torch.zeros_like(layer.D_re)
    dD_im = torch.zeros_like(layer.D_im)

    for t in range(T):
        u_rot = cos_om * u - sin_om * v
        v_rot = sin_om * u + cos_om * v
        amp_sq = (u * u + v * v).clamp(max=10.0)
        sl = beta * (1.0 - amp_sq) - gamma
        u_sl = sl * u; v_sl = sl * v
        # apply recurrence
        if cfg.use_recurrence:
            if cfg.block_diag:
                sp = s_prev.view(B, K, P)
                rec_r = torch.einsum("bkp,kqp->bkq", sp, layer.W_re).reshape(B, N)
                rec_i = torch.einsum("bkp,kqp->bkq", sp, layer.W_im).reshape(B, N)
            else:
                rec_r = s_prev @ layer.W_re.t()
                rec_i = s_prev @ layer.W_im.t()
        else:
            rec_r = torch.zeros_like(u); rec_i = torch.zeros_like(u)
        u_q = one_minus_lam * (u_rot + u_sl + rec_r + xr[:, t])
        v_q = one_minus_lam * (v_rot + v_sl + rec_i + xi[:, t])
        q_sq = u_q * u_q + v_q * v_q
        s = (q_sq > theta).float()
        scale_sp = 1.0 - kappa * s
        u = scale_sp * u_q
        v = scale_sp * v_q
        u = torch.clamp(u, -3.0, 3.0); v = torch.clamp(v, -3.0, 3.0)
        s_prev = s

        # Hebbian: dD_re ~ pre(t) * u(t)/|z(t)|; dD_im ~ pre(t) * v(t)/|z(t)|
        amp = (u_q.square() + v_q.square()).sqrt().clamp(min=1e-3)
        u_norm = u_q / amp                                          # (B, N)
        v_norm = v_q / amp
        # only for spike events: focus credit on neurons that fired
        gate = s                                                     # (B, N)
        weight = gate                                                # could include amp
        # outer product summed across batch
        dD_re = dD_re + torch.einsum("bn,bm->nm", weight * u_norm, x_seq[:, t])
        dD_im = dD_im + torch.einsum("bn,bm->nm", weight * v_norm, x_seq[:, t])

    # apply update (Oja-style)
    layer.D_re += lr_in * dD_re / (B * T)
    layer.D_im += lr_in * dD_im / (B * T)
    # normalize each row (Oja's rule keeps |D_n| stable)
    norm_re = layer.D_re.norm(dim=1, keepdim=True).clamp(min=1e-6)
    norm_im = layer.D_im.norm(dim=1, keepdim=True).clamp(min=1e-6)
    target = math.sqrt(M) * cfg.in_init_scale / max(math.sqrt(M), 1.0)
    layer.D_re *= (target / norm_re)
    layer.D_im *= (target / norm_im)
    return float(s_prev.mean().item())


@torch.no_grad()
def train_layerwise(net: HierarchicalResonantNet, x_data: torch.Tensor, n_passes: int, lr_in: float):
    """Run Hebbian passes on each layer in turn (sensory first, then deeper)."""

    B0, _, _ = x_data.shape
    sig = x_data
    for li, layer in enumerate(net.layers):
        print(f"  layer {li}: training input weights via Hebbian ({n_passes} passes)", flush=True)
        for p in range(n_passes):
            t0 = time.time()
            rate = hebbian_input_step(layer, sig, lr_in=lr_in)
            print(f"    pass {p+1}: rate={rate:.4f} t={time.time()-t0:.1f}s", flush=True)
        # advance signal through trained layer
        sig, _ = layer(sig)
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--test_size", type=int, default=500)
    p.add_argument("--n_passes", type=int, default=2)
    p.add_argument("--lr_in", type=float, default=1e-2)
    p.add_argument("--head_epochs", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=3e-3)
    p.add_argument("--head_kind", type=str, default="both")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]

    net = build_net(args.arch)
    net.eval()
    print(f"net params: {net.n_params()}", flush=True)

    # subset for hebbian training
    print(f"running layer-wise hebbian training (subset)...", flush=True)
    train_layerwise(net, xtr.unsqueeze(-1)[: min(500, xtr.shape[0])], args.n_passes, args.lr_in)

    # then evaluate via head-only
    from train_head_only import compute_tail_features, evaluate_head
    print(f"computing features after Hebbian training ...", flush=True)
    Ftr = compute_tail_features(net, xtr.to("cpu"), batch=64)
    Fte = compute_tail_features(net, xte.to("cpu"), batch=64)
    print(f"feature stats: mean={Ftr.mean().item():.4f}, std={Ftr.std().item():.4f}, zero_rate={(Ftr==0).float().mean().item():.4f}", flush=True)

    cfg = net.cfg
    if args.head_kind in ("linear", "both"):
        print("\n=== Linear head ===", flush=True)
        evaluate_head(Ftr, ytr, Fte, yte, cfg.n_classes, cfg.out_pool_size,
                       cfg.readout_temperature, "linear",
                       args.head_lr, args.head_epochs, 128, "cpu", 1e-4)


if __name__ == "__main__":
    main()
