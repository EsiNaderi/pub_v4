"""DECOLLE 2-stage HRN-v2: each stage has class-pool readout + own loss.

Stage 0: P_0 oscillators (10 class pools × M_0_per_class).
         Scalar SMNIST input. Class-pool tail-energy logits → loss L_0.
         Eligibility-trace local rule → updates own (d, b, ω, α).

Stage 1: P_1 oscillators (10 class pools × M_1_per_class).
         Sparse fan-in of K_1 from stage-0 amp_seq (random fixed indices).
         Class-pool tail-energy logits → loss L_1.
         Own eligibility-trace local rule.

Final prediction: ensemble (alpha-weighted mean of P_0 and P_1 probs).

This is biologically plausible: each stage trains via its own local
classifier and own credit. The hierarchy emerges purely through the
forward path. No BPTT, no surrogate spikes, no inter-stage backward
pass.

The K_1 sparse fan-in keeps stage-1 forward tractable when M_0 is large.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from oscillator import (
    OscillatorConfig, init_params, omega_of, alpha_of,
    forward_with_eligibility, forward_with_eligibility_sparse,
)
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
    p.add_argument("--m0_per_class", type=int, default=48)
    p.add_argument("--m1_per_class", type=int, default=48)
    p.add_argument("--k1_fanin", type=int, default=24,
                   help="number of stage-0 outputs each stage-1 neuron sees")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--tail0", type=int, default=200)
    p.add_argument("--tail1", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--lr1", type=float, default=0.005)
    p.add_argument("--lr_decay_after", type=int, default=10,
                   help="epoch after which to halve LRs")
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--om0_min", type=float, default=0.005)
    p.add_argument("--om0_max", type=float, default=1.2)
    p.add_argument("--al0_min", type=float, default=0.95)
    p.add_argument("--al0_max", type=float, default=0.999)
    p.add_argument("--om1_min", type=float, default=0.001)
    p.add_argument("--om1_max", type=float, default=0.30)
    p.add_argument("--al1_min", type=float, default=0.97)
    p.add_argument("--al1_max", type=float, default=0.9995)
    p.add_argument("--input_init0", type=float, default=0.05)
    p.add_argument("--input_init1", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ensemble_alpha", type=float, default=0.5)
    p.add_argument("--csv", type=str, default="results/decolle_2stage.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = 10
    M0 = classes * args.m0_per_class
    M1 = classes * args.m1_per_class
    class_index_0 = torch.repeat_interleave(torch.arange(classes), args.m0_per_class)
    class_index_1 = torch.repeat_interleave(torch.arange(classes), args.m1_per_class)
    K1 = args.k1_fanin

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    print(f"train {xtr.shape}, test {xte.shape}", flush=True)
    print(f"M0={M0}, M1={M1}, K1_fanin={K1}", flush=True)

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=1,
                             omega_min=args.om0_min, omega_max=args.om0_max,
                             alpha_min=args.al0_min, alpha_max=args.al0_max,
                             input_init=args.input_init0)
    cfg1 = OscillatorConfig(n_neurons=M1, n_input_channels=K1,
                             omega_min=args.om1_min, omega_max=args.om1_max,
                             alpha_min=args.al1_min, alpha_max=args.al1_max,
                             input_init=args.input_init1)
    p0 = init_params(cfg0, generator=gen)
    p1 = init_params(cfg1, generator=gen)

    # Sparse fan-in: each stage-1 neuron picks K1 random stage-0 channels.
    # For diversity, each stage-1 neuron includes some same-class stage-0 channels
    # (so it has access to its own class's stage-0 features) + some random others.
    in_idx = torch.zeros(M1, K1, dtype=torch.long)
    half_k = max(1, K1 // 2)
    for i in range(M1):
        c = int(class_index_1[i].item())
        same_class_pool = (class_index_0 == c).nonzero(as_tuple=True)[0]
        other_pool = (class_index_0 != c).nonzero(as_tuple=True)[0]
        sc_pick = same_class_pool[torch.randperm(len(same_class_pool), generator=gen)[:half_k]]
        oc_pick = other_pool[torch.randperm(len(other_pool), generator=gen)[:K1 - half_k]]
        in_idx[i] = torch.cat([sc_pick, oc_pick])

    opt0 = Adam(p0.tensors(), args.lr0)
    opt1 = Adam(p1.tensors(), args.lr1)

    def fwd(xb, train=False):
        out0 = forward_with_eligibility(xb, p0, cfg0, args.tail0,
                                         accumulate_traces=train, save_amp_seq=True)
        amp_seq_0 = out0["amp_seq"]                      # (B, T, M0)
        E0 = out0["E"]
        logits_0 = class_pool_logits(E0, class_index_0, classes, args.temperature)

        out1 = forward_with_eligibility_sparse(amp_seq_0, in_idx, p1, cfg1, args.tail1,
                                                accumulate_traces=train, save_amp_seq=False)
        E1 = out1["E"]
        logits_1 = class_pool_logits(E1, class_index_1, classes, args.temperature)
        return out0, logits_0, out1, logits_1

    def evaluate(x, y):
        l0_sum = 0.0; l1_sum = 0.0
        a0 = 0; a1 = 0; ac = 0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            _, logits_0, _, logits_1 = fwd(xb, train=False)
            l0_sum += F.cross_entropy(logits_0, yb, reduction="sum").item()
            l1_sum += F.cross_entropy(logits_1, yb, reduction="sum").item()
            a0 += (logits_0.argmax(1) == yb).sum().item()
            a1 += (logits_1.argmax(1) == yb).sum().item()
            prob_0 = F.softmax(logits_0, dim=1)
            prob_1 = F.softmax(logits_1, dim=1)
            prob_c = (1 - args.ensemble_alpha) * prob_0 + args.ensemble_alpha * prob_1
            ac += (prob_c.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return l0_sum / n, l1_sum / n, a0 / n, a1 / n, ac / n

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "tr_a0", "tr_a1", "tr_ac",
                     "te_a0", "te_a1", "te_ac", "best_te_ac",
                     "om0", "al0", "om1", "al1"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        l0_tr, l1_tr, a0_tr, a1_tr, ac_tr = evaluate(xtr, ytr)
        l0_te, l1_te, a0_te, a1_te, ac_te = evaluate(xte, yte)
        if ac_te > best: best = ac_te
        om0_m = float(omega_of(p0, cfg0).mean().item())
        al0_m = float(alpha_of(p0, cfg0).mean().item())
        om1_m = float(omega_of(p1, cfg1).mean().item())
        al1_m = float(alpha_of(p1, cfg1).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] "
              f"tr a0={a0_tr:.4f} a1={a1_tr:.4f} ac={ac_tr:.4f} | "
              f"te a0={a0_te:.4f} a1={a1_te:.4f} ac={ac_te:.4f} | best={best:.4f} | "
              f"om0={om0_m:.3f} om1={om1_m:.4f}",
              flush=True)
        writer.writerow([epoch, f"{wall:.1f}",
                          f"{a0_tr:.4f}", f"{a1_tr:.4f}", f"{ac_tr:.4f}",
                          f"{a0_te:.4f}", f"{a1_te:.4f}", f"{ac_te:.4f}",
                          f"{best:.4f}",
                          f"{om0_m:.4f}", f"{al0_m:.4f}",
                          f"{om1_m:.4f}", f"{al1_m:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        # LR decay after a configured epoch
        if epoch == args.lr_decay_after:
            opt0.lr *= args.lr_decay_factor
            opt1.lr *= args.lr_decay_factor
            print(f"  LR decay: opt0.lr={opt0.lr}, opt1.lr={opt1.lr}", flush=True)

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out0, logits_0, out1, logits_1 = fwd(xb, train=True)

            # Stage 0 update
            probs_0 = F.softmax(logits_0, dim=1)
            onehot = F.one_hot(yb, classes).float()
            d_logit_0 = (probs_0 - onehot) / xb.shape[0]
            d_E_0 = d_logit_0[:, class_index_0] * (args.temperature / args.m0_per_class)
            g0 = [(d_E_0.unsqueeze(-1) * out0["dE_d_r"]).sum(dim=0),
                  (d_E_0.unsqueeze(-1) * out0["dE_d_i"]).sum(dim=0),
                  (d_E_0 * out0["dE_b_r"]).sum(dim=0),
                  (d_E_0 * out0["dE_b_i"]).sum(dim=0),
                  (d_E_0 * out0["dE_omega_raw"]).sum(dim=0),
                  (d_E_0 * out0["dE_alpha_raw"]).sum(dim=0)]
            opt0.step(g0, args.grad_clip)

            # Stage 1 update
            probs_1 = F.softmax(logits_1, dim=1)
            d_logit_1 = (probs_1 - onehot) / xb.shape[0]
            d_E_1 = d_logit_1[:, class_index_1] * (args.temperature / args.m1_per_class)
            g1 = [(d_E_1.unsqueeze(-1) * out1["dE_d_r"]).sum(dim=0),
                  (d_E_1.unsqueeze(-1) * out1["dE_d_i"]).sum(dim=0),
                  (d_E_1 * out1["dE_b_r"]).sum(dim=0),
                  (d_E_1 * out1["dE_b_i"]).sum(dim=0),
                  (d_E_1 * out1["dE_omega_raw"]).sum(dim=0),
                  (d_E_1 * out1["dE_alpha_raw"]).sum(dim=0)]
            opt1.step(g1, args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc (ensemble): {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
