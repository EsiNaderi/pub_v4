"""Continue BPTT from an existing checkpoint.

Hypothesis: the original 4.5h BPTT plateaued at 47% in-loop because the
aux-head was undertrained relative to the rapidly changing features.
With features further along (and aux-head co-adapting), more BPTT
should still improve features. Test by training for some more time,
saving checkpoints, then offline-evaluating each via a linear head.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smnist_data import SMNISTBatcher, load_smnist
from train_bptt import build_net, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_in", type=str, default="results/ckpt_overnight_default_10k.pt")
    p.add_argument("--ckpt_out", type=str, default="results/ckpt_continue_default.pt")
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--time_budget", type=int, default=5400)  # 90 minutes
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--csv", type=str, default="results/run_continue_default.csv")
    p.add_argument("--seed", type=int, default=20260507)
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"device = {device}", flush=True)

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size:
        xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:
        xte, yte = xte[: args.test_size], yte[: args.test_size]
    print(f"train: {xtr.shape[0]}, test: {xte.shape[0]}", flush=True)

    train_loader = SMNISTBatcher(xtr, ytr, args.batch, device, seed=args.seed)
    test_loader = SMNISTBatcher(xte, yte, 128, device, seed=args.seed + 1)

    net = build_net(args.arch, aux_head=True).to(device)
    print(f"Loading {args.ckpt_in} ...", flush=True)
    ck = torch.load(args.ckpt_in, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    print(f"  resumed from epoch {ck.get('epoch', '?')} (test_acc {ck.get('test_acc', '?')})",
          flush=True)
    print(f"net params: {net.n_params()}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "step", "wall", "train_loss_ema", "train_acc_ema",
                     "test_loss", "test_acc", "rates"])

    t0 = time.time()
    step = 0
    best_test = 0.0
    train_loss_ema = None
    train_acc_ema = None
    epoch = 0
    while True:
        if (time.time() - t0) > args.time_budget:
            print("time budget reached.")
            break
        net.train()
        for xb, yb in train_loader.shuffle_iter():
            if (time.time() - t0) > args.time_budget:
                break
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
                print(f"[ep {epoch} step {step:5d} t={wall:6.1f}s] loss={l_val:.3f} "
                      f"(ema {train_loss_ema:.3f}) acc={acc:.3f} (ema {train_acc_ema:.3f})",
                      flush=True)
            step += 1

        ev = evaluate(net, test_loader, max_batches=None)
        wall = time.time() - t0
        rates_str = ", ".join(f"{r:.4f}" for r in ev["rates"])
        print(f"[ep {epoch} EVAL t={wall:6.1f}s] test_loss={ev['loss']:.3f} "
              f"test_acc={ev['acc']:.4f}  rates=[{rates_str}]", flush=True)
        writer.writerow([epoch, step, f"{wall:.1f}", f"{train_loss_ema:.4f}",
                          f"{train_acc_ema:.4f}", f"{ev['loss']:.4f}",
                          f"{ev['acc']:.4f}", rates_str])
        f_csv.flush()
        if ev["acc"] > best_test:
            best_test = ev["acc"]
            Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "cfg": asdict(net.cfg),
                        "test_acc": ev["acc"], "epoch": epoch}, args.ckpt_out)
            print(f"  saved best ckpt to {args.ckpt_out} (test_acc={ev['acc']:.4f})",
                  flush=True)
        epoch += 1

    print(f"best test acc (in-loop): {best_test:.4f}", flush=True)
    f_csv.close()

    # Always save the LAST state too
    last_path = args.ckpt_out.replace(".pt", "_last.pt")
    torch.save({"state_dict": net.state_dict(), "cfg": asdict(net.cfg),
                "test_acc": ev["acc"], "epoch": epoch - 1}, last_path)
    print(f"  saved final ckpt to {last_path}", flush=True)


if __name__ == "__main__":
    main()
