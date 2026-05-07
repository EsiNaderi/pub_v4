"""BPTT training loop for HRN. Capacity check.

Logs per-step train batch metrics and per-epoch test metrics. Tracks resonance
diagnostics: per-layer spike rates, pool firing variance, output-pool
discrimination (winner-correct rate per pool).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smnist_data import SMNISTBatcher, load_smnist
from hrn import HRNConfig, LayerSpec, HierarchicalResonantNet, make_default_config


def get_device(want: str = "auto") -> str:
    if want == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return want


def build_net(arch: str = "default", aux_head: bool = False) -> HierarchicalResonantNet:
    if arch == "default":
        cfg = make_default_config()
    elif arch == "small":
        cfg = HRNConfig(
            in_dim=1, n_classes=10,
            layers=[
                LayerSpec(n_pools=4, pool_size=16, omega_lo=0.5, omega_hi=2.5,
                          in_init_scale=2.0, rec_init_scale=0.30),
            ],
            out_pool_size=12, out_omega_lo=0.02, out_omega_hi=0.30,
            out_in_init_scale=8.0, tail_fraction=0.30,
        )
    elif arch == "deep":
        cfg = HRNConfig(
            in_dim=1, n_classes=10,
            layers=[
                LayerSpec(n_pools=4, pool_size=32, omega_lo=0.7, omega_hi=2.8,
                          in_init_scale=2.0, rec_init_scale=0.30),
                LayerSpec(n_pools=8, pool_size=24, omega_lo=0.20, omega_hi=1.20,
                          in_init_scale=8.0, rec_init_scale=0.30),
                LayerSpec(n_pools=8, pool_size=24, omega_lo=0.05, omega_hi=0.50,
                          in_init_scale=8.0, rec_init_scale=0.30),
            ],
            out_pool_size=24, out_omega_lo=0.02, out_omega_hi=0.20,
            out_in_init_scale=8.0, tail_fraction=0.30,
        )
    elif arch == "wide":
        cfg = HRNConfig(
            in_dim=1, n_classes=10,
            layers=[
                LayerSpec(n_pools=16, pool_size=32, omega_lo=0.3, omega_hi=3.0,
                          in_init_scale=4.0, rec_init_scale=0.30,
                          theta=0.7, eta=0.30, gamma=0.20, beta=0.20),
                LayerSpec(n_pools=16, pool_size=48, omega_lo=0.05, omega_hi=1.0,
                          in_init_scale=2.0, rec_init_scale=0.30,
                          theta=0.7, eta=0.10, gamma=0.20, beta=0.20),
            ],
            out_pool_size=48, out_omega_lo=0.01, out_omega_hi=0.30,
            out_in_init_scale=4.0, tail_fraction=0.30,
            out_theta=0.6, out_eta=0.10, out_gamma=0.20, out_beta=0.20,
            surr_param=2.5,
        )
    elif arch == "big":
        # Modest-scale hierarchical with per-pool freq tiling and ALIF.
        cfg = HRNConfig(
            in_dim=1, n_classes=10,
            layers=[
                LayerSpec(n_pools=16, pool_size=24, omega_lo=0.05, omega_hi=3.0,
                          in_init_scale=4.0, rec_init_scale=0.20,
                          theta=0.5, eta=0.30, gamma=0.20, beta=0.20,
                          omega_per_pool=True),
                LayerSpec(n_pools=16, pool_size=24, omega_lo=0.02, omega_hi=1.0,
                          in_init_scale=2.0, rec_init_scale=0.20,
                          theta=0.5, eta=0.10, gamma=0.20, beta=0.20,
                          omega_per_pool=True),
            ],
            out_pool_size=32, out_omega_lo=0.005, out_omega_hi=0.30,
            out_in_init_scale=2.0, tail_fraction=0.30,
            out_theta=0.5, out_eta=0.07, out_gamma=0.20, out_beta=0.20,
            out_omega_per_pool=True,
            surr_param=2.5,
        )
    elif arch == "tiled":
        # Per-pool frequency tiling: each pool covers a distinct sub-band.
        cfg = HRNConfig(
            in_dim=1, n_classes=10,
            layers=[
                LayerSpec(n_pools=16, pool_size=32, omega_lo=0.10, omega_hi=3.0,
                          in_init_scale=4.0, rec_init_scale=0.30,
                          theta=0.7, eta=0.30, gamma=0.20, beta=0.20),
                LayerSpec(n_pools=16, pool_size=32, omega_lo=0.02, omega_hi=1.0,
                          in_init_scale=2.0, rec_init_scale=0.30,
                          theta=0.6, eta=0.15, gamma=0.20, beta=0.20),
            ],
            out_pool_size=32, out_omega_lo=0.005, out_omega_hi=0.30,
            out_in_init_scale=4.0, tail_fraction=0.30,
            out_theta=0.6, out_eta=0.10, out_gamma=0.20, out_beta=0.20,
            surr_param=2.5,
        )
        # enable per-pool tiling
        for ls in cfg.layers:
            ls.omega_per_pool = True
        cfg.out_omega_per_pool = True
    else:
        raise ValueError(arch)
    if aux_head:
        cfg.aux_linear_head = True
    return HierarchicalResonantNet(cfg)


def evaluate(net: HierarchicalResonantNet, loader: SMNISTBatcher, max_batches: int = 0) -> dict:
    net.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    rate_sums = None
    rate_counts = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader.seq_iter()):
            logits, info = net(xb, return_layers=True)
            loss = F.cross_entropy(logits, yb)
            loss_sum += float(loss.item()) * xb.shape[0]
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.shape[0]
            rates = [float(s.mean().item()) for s in info["layer_spikes"]]
            if rate_sums is None:
                rate_sums = rates
            else:
                rate_sums = [a + b for a, b in zip(rate_sums, rates)]
            rate_counts += 1
            if max_batches and i + 1 >= max_batches:
                break
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "rates": [r / max(rate_counts, 1) for r in (rate_sums or [])],
    }


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    print(f"device: {device}", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size:
        xtr = xtr[: args.train_size]
        ytr = ytr[: args.train_size]
    if args.test_size:
        xte = xte[: args.test_size]
        yte = yte[: args.test_size]
    print(f"train: {xtr.shape[0]}, test: {xte.shape[0]}", flush=True)

    train_loader = SMNISTBatcher(xtr, ytr, args.batch, device, seed=args.seed)
    test_loader = SMNISTBatcher(xte, yte, args.eval_batch, device, seed=args.seed + 1)

    net = build_net(args.arch, aux_head=args.aux_head).to(device)
    n_params = net.n_params()
    print(f"net params: {n_params}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = Path(args.log) if args.log else None
    csv_path = Path(args.csv) if args.csv else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "step", "wall", "train_loss", "train_acc", "test_loss", "test_acc", "rates"])
    else:
        writer = None

    t0 = time.time()
    step = 0
    best_test = 0.0
    train_loss_ema = None
    train_acc_ema = None
    for epoch in range(args.epochs):
        net.train()
        for xb, yb in train_loader.shuffle_iter():
            logits, info = net(xb, return_layers=False)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_norm)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = float((pred == yb).float().mean().item())
            l_val = float(loss.item())
            train_loss_ema = l_val if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * l_val
            train_acc_ema = acc if train_acc_ema is None else 0.95 * train_acc_ema + 0.05 * acc

            if step % args.log_every == 0:
                wall = time.time() - t0
                print(
                    f"[ep {epoch} step {step:5d} t={wall:6.1f}s] "
                    f"loss={l_val:.3f} (ema {train_loss_ema:.3f}) acc={acc:.3f} (ema {train_acc_ema:.3f})",
                    flush=True,
                )

            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        # eval per epoch
        ev = evaluate(net, test_loader, max_batches=args.eval_max_batches)
        wall = time.time() - t0
        rates_str = ", ".join(f"{r:.4f}" for r in ev["rates"])
        print(
            f"[ep {epoch} EVAL t={wall:6.1f}s] test_loss={ev['loss']:.3f} test_acc={ev['acc']:.4f}  rates=[{rates_str}]",
            flush=True,
        )
        if writer:
            writer.writerow([
                epoch, step, f"{wall:.1f}", f"{train_loss_ema:.4f}",
                f"{train_acc_ema:.4f}", f"{ev['loss']:.4f}", f"{ev['acc']:.4f}",
                rates_str,
            ])
            f_csv.flush()
        if ev["acc"] > best_test:
            best_test = ev["acc"]
            if args.ckpt:
                Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": net.state_dict(), "cfg": asdict(net.cfg), "test_acc": ev["acc"], "epoch": epoch}, args.ckpt)
                print(f"  saved best ckpt to {args.ckpt} (test_acc={ev['acc']:.4f})")

        if args.max_steps and step >= args.max_steps:
            break
        if (time.time() - t0) > args.time_budget:
            print(f"time budget reached: {wall:.1f}s")
            break

    print(f"best test acc: {best_test:.4f}")
    if writer:
        f_csv.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--train_size", type=int, default=0)
    p.add_argument("--test_size", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_max_batches", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--time_budget", type=float, default=3600 * 8)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--log", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--aux_head", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
