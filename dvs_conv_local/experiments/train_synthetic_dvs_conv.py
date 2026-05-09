"""Synthetic DVS moving-edge smoke test for local spiking convolution.

This is deliberately not a benchmark. It verifies that a shared
convolutional spiking oscillator can learn from local eligibility traces
and manually derived class-pool credit without autograd/backprop.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from conv_oscillator_spiking import (
    ConvOscillatorConfig,
    contracted_local_grads_conv,
    forward_spiking_conv,
    init_params,
    omega_of,
)
from local_readout import class_pool_logits, cross_entropy_credit
from optim import Adam


def make_moving_edge_dataset(n, T, H, W, classes, noise, gen):
    x = torch.zeros(n, T, 2, H, W)
    y = torch.arange(n, dtype=torch.long) % classes
    y = y[torch.randperm(n, generator=gen)]
    max_disp_x = W - 5
    max_disp_y = H - 5
    for i in range(n):
        c = int(y[i].item())
        last_row = None
        last_col = None
        for t in range(T):
            if c == 0:          # right-moving vertical edge
                row = None
                col = 2 + round(max_disp_x * t / max(1, T - 1))
            elif c == 1:        # left-moving vertical edge
                row = None
                col = W - 3 - round(max_disp_x * t / max(1, T - 1))
            elif c == 2:        # down-moving horizontal edge
                row = 2 + round(max_disp_y * t / max(1, T - 1))
                col = None
            else:               # up-moving horizontal edge
                row = H - 3 - round(max_disp_y * t / max(1, T - 1))
                col = None

            if col is not None:
                c0 = max(0, min(W - 1, col))
                x[i, t, 0, :, c0] = 1.0
                if last_col is not None:
                    x[i, t, 1, :, last_col] = 1.0
                last_col = c0
            if row is not None:
                r0 = max(0, min(H - 1, row))
                x[i, t, 0, r0, :] = 1.0
                if last_row is not None:
                    x[i, t, 1, last_row, :] = 1.0
                last_row = r0

    if noise > 0:
        mask = torch.rand(x.shape, generator=gen) < noise
        x[mask] = 1.0
    return x, y


def accuracy(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--test_size", type=int, default=256)
    p.add_argument("--time_steps", type=int, default=24)
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--classes", type=int, default=4)
    p.add_argument("--m_per_class", type=int, default=6)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--theta_init", type=float, default=0.45)
    p.add_argument("--theta_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--target_rate", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--noise", type=float, default=0.002)
    p.add_argument("--csv", type=str, default="results/synthetic_dvs_conv_local.csv")
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    xtr, ytr = make_moving_edge_dataset(
        args.train_size, args.time_steps, args.height, args.width,
        args.classes, args.noise, gen)
    xte, yte = make_moving_edge_dataset(
        args.test_size, args.time_steps, args.height, args.width,
        args.classes, args.noise, gen)

    out_channels = args.classes * args.m_per_class
    class_index = torch.repeat_interleave(torch.arange(args.classes), args.m_per_class)
    cfg = ConvOscillatorConfig(
        in_channels=2,
        out_channels=out_channels,
        kernel_size=args.kernel,
        padding=args.kernel // 2,
        input_init=0.18,
        omega_min=0.04,
        omega_max=1.25,
        alpha_min=0.86,
        alpha_max=0.996,
    )
    params = init_params(cfg, gen)
    opt = Adam(params.tensors(), args.lr)
    theta = torch.full((out_channels,), args.theta_init)
    rate_ema = torch.full((out_channels,), args.target_rate)

    def evaluate(x, y):
        n = 0
        smooth_ok = 0
        spike_ok = 0
        rate_sum = 0.0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s:s + args.batch]
            yb = y[s:s + args.batch]
            out = forward_spiking_conv(
                xb, params, cfg, theta, tail=args.time_steps,
                beta=args.beta, sample_binary=True, rng=gen)
            logits = class_pool_logits(out["rho"], class_index, args.classes, args.temperature)
            spike_logits = class_pool_logits(out["spike_rate"], class_index, args.classes, args.temperature)
            smooth_ok += int((logits.argmax(1) == yb).sum().item())
            spike_ok += int((spike_logits.argmax(1) == yb).sum().item())
            rate_sum += float(out["spike_rate"].mean().item()) * xb.shape[0]
            n += xb.shape[0]
        return smooth_ok / n, spike_ok / n, rate_sum / n

    out_csv = ROOT / args.csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(out_csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "te_smooth", "te_spike", "best_te_spike", "rate", "omega_mean"])

    best_spike = 0.0
    t0 = time.time()
    print("Synthetic DVS local spiking conv", flush=True)
    print(f"train={tuple(xtr.shape)}, test={tuple(xte.shape)}, classes={args.classes}, "
          f"out_channels={out_channels}, kernel={args.kernel}", flush=True)
    for epoch in range(args.epochs + 1):
        te_smooth, te_spike, te_rate = evaluate(xte, yte)
        best_spike = max(best_spike, te_spike)
        wall = time.time() - t0
        om = float(omega_of(params, cfg).mean().item())
        print(f"[ep {epoch:02d} t={wall:5.1f}s] smooth={te_smooth:.3f} "
              f"BINARY={te_spike:.3f} best={best_spike:.3f} "
              f"rate={te_rate:.3f} omega={om:.3f}", flush=True)
        writer.writerow([epoch, f"{wall:.1f}", f"{te_smooth:.4f}", f"{te_spike:.4f}",
                         f"{best_spike:.4f}", f"{te_rate:.4f}", f"{om:.4f}"])
        f_csv.flush()
        if epoch == args.epochs:
            break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s:s + args.batch]
            xb = xtr[idx]
            yb = ytr[idx]
            out = forward_spiking_conv(
                xb, params, cfg, theta, tail=args.time_steps,
                beta=args.beta, sample_binary=True, rng=gen)
            _, _, credit = cross_entropy_credit(
                out["rho"], yb, class_index, args.classes, args.temperature)
            grads = contracted_local_grads_conv(
                xb, params, cfg, theta, tail=args.time_steps,
                credit_rho=credit, beta=args.beta)
            opt.step(grads.tensors(), args.grad_clip)

            observed = out["rho"].mean(dim=(0, 2, 3))
            rate_ema.mul_(1.0 - args.ema_lr).add_(args.ema_lr * observed)
            theta.add_(args.theta_lr * (rate_ema - args.target_rate))

    f_csv.close()
    print(f"Best binary-spike test accuracy: {best_spike:.4f}", flush=True)


if __name__ == "__main__":
    main()
