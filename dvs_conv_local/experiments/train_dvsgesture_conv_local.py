"""DVS Gesture benchmark for the strict local spiking conv prototype.

This is intentionally a one-layer baseline:

    DVS frames -> shared spiking conv oscillator -> class-pool readout

Learning is local:

    dtheta = sum_t local_class_credit * forward_eligibility(theta, t)

There is no BPTT, no surrogate derivative through sampled spikes, no
transported downstream weight matrix, and no learned classifier head.
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
from dvsgesture_data import NUM_CLASSES, load_dvsgesture
from local_readout import class_pool_logits, cross_entropy_credit
from optim import Adam


def class_counts(y, classes):
    return [int((y == c).sum().item()) for c in range(classes)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--time_bins", type=int, default=12)
    p.add_argument("--spatial", type=int, default=32)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--test_limit", type=int, default=0)
    p.add_argument("--m_per_class", type=int, default=2)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.006)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--theta_init", type=float, default=0.55)
    p.add_argument("--theta_lr", type=float, default=0.08)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--target_rate", type=float, default=0.06)
    p.add_argument("--temperature", type=float, default=12.0)
    p.add_argument("--input_scale", type=float, default=0.15)
    p.add_argument("--input_init", type=float, default=0.08)
    p.add_argument("--csv", type=str, default="results/dvsgesture_conv_local.csv")
    p.add_argument("--cache_dir", type=str, default="data/cache")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    data_root = args.data_root if args.data_root else None
    xtr, ytr, xte, yte = load_dvsgesture(
        ROOT / args.cache_dir,
        time_bins=args.time_bins,
        spatial=args.spatial,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        seed=args.seed,
        data_root=data_root,
    )
    xtr = xtr * args.input_scale
    xte = xte * args.input_scale

    classes = NUM_CLASSES
    out_channels = classes * args.m_per_class
    class_index = torch.repeat_interleave(torch.arange(classes), args.m_per_class)
    cfg = ConvOscillatorConfig(
        in_channels=2,
        out_channels=out_channels,
        kernel_size=args.kernel,
        padding=args.kernel // 2,
        input_init=args.input_init,
        omega_min=0.03,
        omega_max=1.20,
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
                xb, params, cfg, theta, tail=args.time_bins,
                beta=args.beta, sample_binary=True, rng=gen)
            logits = class_pool_logits(out["rho"], class_index, classes, args.temperature)
            spike_logits = class_pool_logits(out["spike_rate"], class_index, classes, args.temperature)
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

    print("DVS Gesture strict local spiking conv benchmark", flush=True)
    print(f"train={tuple(xtr.shape)}, test={tuple(xte.shape)}", flush=True)
    print(f"train class counts={class_counts(ytr, classes)}", flush=True)
    print(f"test  class counts={class_counts(yte, classes)}", flush=True)
    print(f"out_channels={out_channels}, kernel={args.kernel}, input_scale={args.input_scale}", flush=True)

    best_spike = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        te_smooth, te_spike, te_rate = evaluate(xte, yte)
        best_spike = max(best_spike, te_spike)
        wall = time.time() - t0
        om = float(omega_of(params, cfg).mean().item())
        print(f"[ep {epoch:02d} t={wall:6.1f}s] smooth={te_smooth:.4f} "
              f"BINARY={te_spike:.4f} best={best_spike:.4f} "
              f"rate={te_rate:.4f} omega={om:.4f}", flush=True)
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
                xb, params, cfg, theta, tail=args.time_bins,
                beta=args.beta, sample_binary=True, rng=gen)
            _, _, credit = cross_entropy_credit(
                out["rho"], yb, class_index, classes, args.temperature)
            grads = contracted_local_grads_conv(
                xb, params, cfg, theta, tail=args.time_bins,
                credit_rho=credit, beta=args.beta)
            opt.step(grads.tensors(), args.grad_clip)

            observed = out["rho"].mean(dim=(0, 2, 3))
            rate_ema.mul_(1.0 - args.ema_lr).add_(args.ema_lr * observed)
            theta.add_(args.theta_lr * (rate_ema - args.target_rate))

    f_csv.close()
    print(f"Best binary-spike DVS Gesture accuracy: {best_spike:.4f}", flush=True)


if __name__ == "__main__":
    main()
