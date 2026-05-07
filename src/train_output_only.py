"""Train ONLY the output layer (D, W_rec, omega, eta of last layer + pool_bias).

Hidden layers are frozen as random reservoir. This isolates the question:
can we make the pool-rate readout work via local supervised credit?

Uses the same surrogate-gradient BPTT but only on the output-layer parameters.
This is much faster than full BPTT because gradient only flows through one
layer.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import SMNISTBatcher, load_smnist
from hrn import HierarchicalResonantNet
from train_bptt import build_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--use_aux_head", action="store_true")
    p.add_argument("--time_budget", type=float, default=14400)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--log_every", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]

    net = build_net(args.arch, aux_head=args.use_aux_head).to("cpu")

    # Freeze hidden layers (all except last + bias + aux_head)
    for layer in net.layers[:-1]:
        for param in layer.parameters():
            param.requires_grad_(False)

    trainable = []
    for name, p_obj in net.named_parameters():
        if p_obj.requires_grad:
            trainable.append((name, p_obj))
    print(f"trainable params:", flush=True)
    n_trainable = 0
    for name, p_obj in trainable:
        n_trainable += p_obj.numel()
        print(f"  {name}: {tuple(p_obj.shape)} = {p_obj.numel()}", flush=True)
    print(f"total trainable: {n_trainable}", flush=True)

    opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)

    train_loader = SMNISTBatcher(xtr, ytr, args.batch, "cpu", seed=args.seed)
    test_loader = SMNISTBatcher(xte, yte, args.batch * 2, "cpu", seed=args.seed + 1)

    csv_path = Path(args.csv) if args.csv else None
    writer = None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "step", "wall", "train_loss", "train_acc", "test_loss", "test_acc"])

    t0 = time.time(); step = 0; best_test = 0.0
    train_loss_ema = None; train_acc_ema = None
    for epoch in range(args.epochs):
        net.train()
        for xb, yb in train_loader.shuffle_iter():
            logits, info = net(xb, return_layers=False)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.clip_norm)
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = float((pred == yb).float().mean().item())
            l_val = float(loss.item())
            train_loss_ema = l_val if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * l_val
            train_acc_ema = acc if train_acc_ema is None else 0.95 * train_acc_ema + 0.05 * acc
            if step % args.log_every == 0:
                wall = time.time() - t0
                print(f"[ep {epoch} step {step:5d} t={wall:6.1f}s] loss={l_val:.3f} "
                      f"(ema {train_loss_ema:.3f}) acc={acc:.3f} (ema {train_acc_ema:.3f})", flush=True)
            step += 1
            if (time.time() - t0) > args.time_budget:
                break
        # eval
        net.eval()
        n_correct = 0; n_total = 0; loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in test_loader.seq_iter():
                logits, _ = net(xb)
                pred = logits.argmax(dim=1)
                n_correct += int((pred == yb).sum().item())
                n_total += xb.shape[0]
                loss_sum += float(F.cross_entropy(logits, yb).item()) * xb.shape[0]
        test_acc = n_correct / max(n_total, 1)
        test_loss = loss_sum / max(n_total, 1)
        if test_acc > best_test:
            best_test = test_acc
            if args.ckpt:
                Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": net.state_dict(), "test_acc": test_acc, "epoch": epoch}, args.ckpt)
        wall = time.time() - t0
        print(f"[ep {epoch} EVAL t={wall:6.1f}s] test_loss={test_loss:.4f} "
              f"test_acc={test_acc:.4f} best={best_test:.4f}", flush=True)
        if writer:
            writer.writerow([epoch, step, f"{wall:.1f}", f"{train_loss_ema:.4f}",
                              f"{train_acc_ema:.4f}", f"{test_loss:.4f}", f"{test_acc:.4f}"])
            f_csv.flush()
        if (time.time() - t0) > args.time_budget:
            break

    print(f"best test: {best_test:.4f}", flush=True)
    if writer: f_csv.close()


if __name__ == "__main__":
    main()
