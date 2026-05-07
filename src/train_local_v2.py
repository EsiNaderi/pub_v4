"""Improved local-rule training: sample-level credit + winner-take-all
between pools.

Key changes vs `train_local.py`:

1. **Sample-level credit**: per sample, compute one credit value per
   class pool. Apply it uniformly across the tail window timesteps and
   pool neurons. Avoids per-timestep noise.

2. **Winner-take-all between pools**: at each timestep in the tail
   window, the pool with highest activity gets positive credit if it's
   the correct class, negative if not. Other pools get zero credit. This
   creates competition that drives specialization.

3. **Pure Hebbian on pre-spike correlation**: dW = sum_t s_post(t) *
   pre_trace(t), no surrogate-gradient term. The learning signal is
   the post-synaptic spike itself; positive/negative credit modulates
   the rate.

Pipeline per batch:
- Forward through all layers, no gradient.
- Compute pool_rate (B, n_classes).
- Determine winner pools (B,) = argmax(pool_rate, dim=1).
- For each sample b:
   * If winner == target: positive credit for winner pool's neurons; nothing else.
   * If winner != target: positive credit for target pool's neurons,
     negative credit for winner pool's neurons.
- Propagate credit to hidden layers via random feedback alignment.
- Compute Hebbian gradients on D, W_rec.
- Adam-like update.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import SMNISTBatcher, load_smnist
from hrn import HierarchicalResonantNet
from train_bptt import build_net
from train_local import (hrn_forward_logged, init_fb_proj, AdamLite, _exp_trace, _surrog)


@dataclass
class LocalV2Cfg:
    lr_D: float = 5e-3
    lr_W: float = 5e-3
    eligibility_alpha: float = 0.95
    target_high: float = 1.0     # credit value for correct/winner pool
    target_low: float = -0.5     # credit value for wrong pool
    homeostasis_target: float = 0.10
    homeostasis_coef: float = 1.0
    fb_align_scale: float = 1.0
    use_feedback_alignment: bool = True
    weight_decay: float = 1e-4
    grad_clip: float = 5.0


@torch.no_grad()
def compute_grads_v2(net: HierarchicalResonantNet, fwd: dict, targets: torch.Tensor,
                     cfg: LocalV2Cfg, fb_proj: List[torch.Tensor]) -> tuple[List[dict], torch.Tensor]:
    layers = fwd["layers"]
    pool_logits = fwd["pool_logits"]
    pool_rate = fwd["pool_rate"]                  # (B, n_classes)
    tail = fwd["tail"]; T = fwd["T"]
    B = pool_rate.shape[0]
    n_classes = net.cfg.n_classes
    pool_size = net.cfg.out_pool_size

    # Identify winner pool per sample
    winners = pool_rate.argmax(dim=1)              # (B,)
    correct = (winners == targets)

    # Build credit at output layer: per (sample, neuron)
    # - Correct pool gets +target_high
    # - Winner pool (if not correct) gets target_low
    # - Other pools get 0
    out_log = layers[-1]
    N_out = out_log["s_seq"].shape[2]
    credit = torch.zeros(B, n_classes)
    credit[torch.arange(B), targets] = cfg.target_high
    wrong_mask = ~correct
    credit[wrong_mask, winners[wrong_mask]] = cfg.target_low
    # Expand to per-neuron in pool: (B, n_classes, pool_size)
    cred_neuron = credit.unsqueeze(-1).expand(-1, -1, pool_size).reshape(B, N_out)

    # Apply uniformly in tail window
    L_post: List[torch.Tensor] = [None] * len(layers)
    L_out = torch.zeros(B, T, N_out)
    L_out[:, T - tail:] = cred_neuron.unsqueeze(1) / float(tail)
    L_post[-1] = L_out

    # Feedback alignment to earlier layers
    for li in range(len(layers) - 2, -1, -1):
        if cfg.use_feedback_alignment:
            B_lp1 = fb_proj[li]
            L_l = L_post[li + 1] @ B_lp1.t() * cfg.fb_align_scale
        else:
            L_l = torch.zeros_like(layers[li]["q_sq_seq"])
        # mean-subtract
        L_l = L_l - L_l.mean(dim=2, keepdim=True)
        # add homeostasis
        rate = layers[li]["s_seq"].mean(dim=(0, 1))
        L_l = L_l - cfg.homeostasis_coef * (rate - cfg.homeostasis_target).view(1, 1, -1)
        L_post[li] = L_l

    # Compute gradients per layer
    grads = []
    for li, log in enumerate(layers):
        layer = net.layers[li]
        cfg_l = layer.cfg
        K, P, N = log["K"], log["P"], log["N"]
        L_layer = L_post[li]                        # (B, T, N)
        s_seq = log["s_seq"]                        # (B, T, N) -- post spikes
        pre_seq = log["pre_seq"]
        s_shift = torch.cat([torch.zeros_like(s_seq[:, :1]), s_seq[:, :-1]], dim=1)
        pre_trace = _exp_trace(pre_seq, cfg.eligibility_alpha)
        rec_trace = _exp_trace(s_shift, cfg.eligibility_alpha)

        # Pure Hebbian: dW ~ sum_t L_post[t] * s_post[t] * pre_trace[t]
        # (could also use surrog instead of s_post for soft credit, but s_post is more biological)
        L_eff = L_layer * s_seq                     # multiply by post spike
        dD_re = torch.einsum("btn,btm->nm", L_eff, pre_trace)
        # For D_im, use a small phase offset -- approximated by the imaginary part of q
        # (This is a hack; properly we'd track u_q vs v_q separately)
        dD_im = dD_re * 0.0                          # zero out for simplicity (D_re carries most signal)

        if cfg_l.use_recurrence:
            if cfg_l.block_diag:
                L_blk = L_eff.view(L_eff.shape[0], L_eff.shape[1], K, P)
                rec_blk = rec_trace.view(rec_trace.shape[0], rec_trace.shape[1], K, P)
                dW_re = torch.einsum("btkp,btkq->kpq", L_blk, rec_blk)
                dW_im = torch.zeros_like(dW_re)
            else:
                dW_re = torch.einsum("btn,btm->nm", L_eff, rec_trace)
                dW_im = torch.zeros_like(dW_re)
        else:
            dW_re = None; dW_im = None

        grads.append({
            "dD_re": dD_re, "dD_im": dD_im, "dW_re": dW_re, "dW_im": dW_im,
        })

    # Pool bias gradient: standard CE
    probs = torch.softmax(pool_logits, dim=1)
    one_hot = torch.zeros_like(probs); one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    db = (one_hot - probs).sum(dim=0)
    return grads, db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--test_size", type=int, default=500)
    p.add_argument("--lr_D", type=float, default=5e-3)
    p.add_argument("--lr_W", type=float, default=5e-3)
    p.add_argument("--lr_bias", type=float, default=5e-3)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--target_rate", type=float, default=0.10)
    p.add_argument("--homeo_coef", type=float, default=1.0)
    p.add_argument("--target_high", type=float, default=1.0)
    p.add_argument("--target_low", type=float, default=-0.5)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--time_budget", type=float, default=14400)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]

    net = build_net(args.arch).to("cpu")
    print(f"net params: {net.n_params()}", flush=True)

    fb_proj = init_fb_proj(net)
    cfg = LocalV2Cfg(lr_D=args.lr_D, lr_W=args.lr_W, eligibility_alpha=args.alpha,
                     homeostasis_target=args.target_rate, homeostasis_coef=args.homeo_coef,
                     target_high=args.target_high, target_low=args.target_low)

    D_re_list = [layer.D_re for layer in net.layers]
    D_im_list = [layer.D_im for layer in net.layers]
    W_re_list = [layer.W_re for layer in net.layers if layer.cfg.use_recurrence]
    W_im_list = [layer.W_im for layer in net.layers if layer.cfg.use_recurrence]

    opt_D_re = AdamLite(D_re_list, cfg.lr_D, cfg.weight_decay)
    opt_D_im = AdamLite(D_im_list, cfg.lr_D, cfg.weight_decay)
    opt_W_re = AdamLite(W_re_list, cfg.lr_W, cfg.weight_decay)
    opt_W_im = AdamLite(W_im_list, cfg.lr_W, cfg.weight_decay)
    if net.pool_bias is not None:
        opt_bias = AdamLite([net.pool_bias], args.lr_bias, 0.0)
    else:
        opt_bias = None

    train_loader = SMNISTBatcher(xtr, ytr, args.batch, "cpu", seed=args.seed)
    test_loader = SMNISTBatcher(xte, yte, args.batch * 2, "cpu", seed=args.seed + 1)

    csv_path = Path(args.csv) if args.csv else None
    writer = None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "step", "wall", "train_loss", "train_acc", "test_loss", "test_acc", "rates"])

    t0 = time.time(); step = 0; best_test = 0.0
    for epoch in range(args.epochs):
        ema_loss = None; ema_acc = None
        for xb, yb in train_loader.shuffle_iter():
            fwd = hrn_forward_logged(net, xb)
            grads, db = compute_grads_v2(net, fwd, yb, cfg, fb_proj)
            opt_D_re.step([g["dD_re"] for g in grads], cfg.grad_clip)
            opt_D_im.step([g["dD_im"] for g in grads], cfg.grad_clip)
            if W_re_list:
                opt_W_re.step([g["dW_re"] for g in grads if g["dW_re"] is not None], cfg.grad_clip)
                opt_W_im.step([g["dW_im"] for g in grads if g["dW_im"] is not None], cfg.grad_clip)
            if opt_bias is not None:
                opt_bias.step([db], cfg.grad_clip)

            with torch.no_grad():
                pred = fwd["logits"].argmax(dim=1)
                acc = float((pred == yb).float().mean().item())
                loss = float(F.cross_entropy(fwd["logits"], yb).item())
            ema_loss = loss if ema_loss is None else 0.95 * ema_loss + 0.05 * loss
            ema_acc = acc if ema_acc is None else 0.95 * ema_acc + 0.05 * acc

            if step % args.log_every == 0:
                wall = time.time() - t0
                rates = [layers["s_seq"].mean().item() for layers in fwd["layers"]]
                rates_str = "/".join(f"{r:.3f}" for r in rates)
                print(f"[ep {epoch} step {step:5d} t={wall:6.1f}s] loss={loss:.3f} (ema {ema_loss:.3f}) "
                      f"acc={acc:.3f} (ema {ema_acc:.3f}) rates={rates_str}", flush=True)
            step += 1
            if (time.time() - t0) > args.time_budget:
                break

        # eval
        n_correct = 0; n_total = 0; loss_sum = 0.0
        for xb, yb in test_loader.seq_iter():
            fwd = hrn_forward_logged(net, xb)
            pred = fwd["logits"].argmax(dim=1)
            n_correct += int((pred == yb).sum().item())
            n_total += xb.shape[0]
            loss_sum += float(F.cross_entropy(fwd["logits"], yb).item()) * xb.shape[0]
        test_acc = n_correct / max(n_total, 1)
        test_loss = loss_sum / max(n_total, 1)
        if test_acc > best_test:
            best_test = test_acc
        wall = time.time() - t0
        print(f"[ep {epoch} EVAL t={wall:6.1f}s] test_loss={test_loss:.4f} test_acc={test_acc:.4f} best={best_test:.4f}", flush=True)
        if writer:
            rates_str = "/".join(f"{r:.3f}" for r in rates)
            writer.writerow([epoch, step, f"{wall:.1f}", f"{ema_loss:.4f}", f"{ema_acc:.4f}",
                              f"{test_loss:.4f}", f"{test_acc:.4f}", rates_str])
            f_csv.flush()
        if (time.time() - t0) > args.time_budget:
            break

    print(f"best test: {best_test:.4f}", flush=True)
    if writer: f_csv.close()


if __name__ == "__main__":
    main()
