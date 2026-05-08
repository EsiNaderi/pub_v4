"""Single-stage class-pool resonant net: each oscillator pre-assigned to a class.

Architecture:
    Input scalar(t)  (B, T, 1)
       │
       ▼
    P = C × M oscillators (each with own ω, α, d, b)
    Indexed by (class c, position k); tail energy E_{c, k} = |z_{c, k}(t)|^2
    Logit_c = mean_k E_{c, k}.
    Loss = cross-entropy on logits.

Local rule: each oscillator's parameters update via eligibility-trace
gradient × per-neuron credit dL/dE_i = T * d_logit_{c(i)} / M.

This is the simplest expression of pub_v3's "fixed class-pool" scaffold
adapted to SMNIST: the architecture imposes the class-mode mapping, and
local plasticity refines each oscillator's resonance to its class. No
top-level head, no MLP, no BPTT.

Per-neuron forward and gradient is identical to the F=1 (scalar input)
case; computation cost is purely O(B × T × P), which is fast (P ≤ 1024
is fine).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from oscillator import OscillatorConfig, init_params, omega_of, alpha_of, forward_with_eligibility
from optim import Adam
from smnist_data import load_smnist


def class_pool_logits(E, class_index, classes, temperature):
    logits = torch.zeros(E.shape[0], classes, device=E.device, dtype=E.dtype)
    for c in range(classes):
        mask = (class_index == c)
        logits[:, c] = E[:, mask].mean(dim=1)
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m_per_class", type=int, default=24)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--om_min", type=float, default=0.005)
    p.add_argument("--om_max", type=float, default=1.2)
    p.add_argument("--al_min", type=float, default=0.95)
    p.add_argument("--al_max", type=float, default=0.999)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--csv", type=str, default="results/classpool_single.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = 10
    P = classes * args.m_per_class
    class_index = torch.repeat_interleave(torch.arange(classes), args.m_per_class)

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    print(f"train {xtr.shape}, test {xte.shape}, P={P}", flush=True)

    cfg = OscillatorConfig(n_neurons=P, n_input_channels=1,
                            omega_min=args.om_min, omega_max=args.om_max,
                            alpha_min=args.al_min, alpha_max=args.al_max,
                            input_init=args.input_init)
    params = init_params(cfg, generator=gen)
    opt = Adam(params.tensors(), args.lr)

    def fwd(xb, train=False):
        out = forward_with_eligibility(xb, params, cfg, args.tail,
                                         accumulate_traces=train, save_amp_seq=False)
        E = out["E"]
        logits = class_pool_logits(E, class_index, classes, args.temperature)
        return out, logits

    def evaluate(x, y):
        ls = 0.0; ac = 0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            _, logits = fwd(xb, train=False)
            ls += F.cross_entropy(logits, yb, reduction="sum").item()
            ac += (logits.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return ls / n, ac / n

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "tr_loss", "tr_acc", "te_loss", "te_acc",
                     "best_te_acc", "om_mean", "al_mean"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        loss_tr, acc_tr = evaluate(xtr, ytr)
        loss_te, acc_te = evaluate(xte, yte)
        if acc_te > best: best = acc_te
        om_mean = float(omega_of(params, cfg).mean().item())
        al_mean = float(alpha_of(params, cfg).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] tr_loss={loss_tr:.3f} tr_acc={acc_tr:.4f} "
              f"te_loss={loss_te:.3f} te_acc={acc_te:.4f} best={best:.4f} "
              f"om={om_mean:.4f} al={al_mean:.4f}",
              flush=True)
        writer.writerow([epoch, f"{wall:.1f}", f"{loss_tr:.4f}", f"{acc_tr:.4f}",
                          f"{loss_te:.4f}", f"{acc_te:.4f}", f"{best:.4f}",
                          f"{om_mean:.4f}", f"{al_mean:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out, logits = fwd(xb, train=True)
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(yb, classes).float()
            d_logit = (probs - onehot) / xb.shape[0]
            d_E = d_logit[:, class_index] * (args.temperature / args.m_per_class)

            g_d_r = (d_E.unsqueeze(-1) * out["dE_d_r"]).sum(dim=0)
            g_d_i = (d_E.unsqueeze(-1) * out["dE_d_i"]).sum(dim=0)
            g_b_r = (d_E * out["dE_b_r"]).sum(dim=0)
            g_b_i = (d_E * out["dE_b_i"]).sum(dim=0)
            g_om = (d_E * out["dE_omega_raw"]).sum(dim=0)
            g_al = (d_E * out["dE_alpha_raw"]).sum(dim=0)
            opt.step([g_d_r, g_d_i, g_b_r, g_b_i, g_om, g_al], args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
