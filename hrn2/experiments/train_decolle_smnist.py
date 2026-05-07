"""DECOLLE-style multi-stage HRN-v2 trainer.

Each stage has its own class-pool readout and its own contrastive loss.
Each stage's parameters are updated by its OWN local credit (no inter-
stage gradient transport). The hierarchy emerges purely through the
forward path: stage 1 receives stage-0 amplitude trajectory as input;
stage 1 receives a richer signal because stage 0 has already extracted
class-discriminative features.

This avoids the credit-assignment problem entirely while still
benefiting from stage composition.

Final prediction: ensemble of all stages' P(c).

Architecture:
    Stage 0:    1 oscillator bank, 10 class pools × M_0_per_class
                input: scalar SMNIST(t)
                amp_seq_0 (B, T, M_0)  → stage 1 input
                tail energies → class-pool logits → loss L_0
    Stage 1:    1 oscillator bank, 10 class pools × M_1_per_class
                input: amp_seq_0 (B, T, M_0)
                amp_seq_1 (B, T, M_1)  → stage 2 input  (optional)
                tail energies → class-pool logits → loss L_1
    Stage 2:    similar (optional)

Total loss: L_0 + L_1 + ... (each updates only its own stage's params).
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

from oscillator import OscillatorConfig, init_params, omega_of, alpha_of, forward_with_eligibility
from optim import Adam
from smnist_data import load_smnist


def class_pool_logits(E, class_index, classes, temperature):
    """Compute class-pool logits from per-neuron tail energies.

    E: (B, P) tail energies
    class_index: (P,) class assignment
    Returns (B, classes), mean-centered + temperature-scaled.
    """
    logits = torch.zeros(E.shape[0], classes, device=E.device, dtype=E.dtype)
    for c in range(classes):
        mask = (class_index == c)
        logits[:, c] = E[:, mask].mean(dim=1)
    logits = logits - logits.mean(dim=1, keepdim=True)
    return logits * temperature


def class_pool_credit(probs, y, class_index, M_per_class, temperature, batch_size):
    """Per-neuron credit dL/dE for class-pool CE loss.

    Returns (B, P) where P = classes * M_per_class.
    """
    onehot = F.one_hot(y, probs.shape[1]).float()
    d_logit = (probs - onehot) / max(batch_size, 1)              # (B, classes)
    return d_logit[:, class_index] * (temperature / M_per_class)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0_per_class", type=int, default=12)         # stage 0 size = 10 * this
    p.add_argument("--m1_per_class", type=int, default=12)         # stage 1 size
    p.add_argument("--use_stage1", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--tail0", type=int, default=200)
    p.add_argument("--tail1", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--lr1", type=float, default=0.005)
    p.add_argument("--grad_clip", type=float, default=2.0)
    # stage 0 frequency band (fast)
    p.add_argument("--om0_min", type=float, default=0.005)
    p.add_argument("--om0_max", type=float, default=1.2)
    p.add_argument("--al0_min", type=float, default=0.95)
    p.add_argument("--al0_max", type=float, default=0.999)
    # stage 1 frequency band (slower)
    p.add_argument("--om1_min", type=float, default=0.001)
    p.add_argument("--om1_max", type=float, default=0.30)
    p.add_argument("--al1_min", type=float, default=0.97)
    p.add_argument("--al1_max", type=float, default=0.9995)
    p.add_argument("--input_init0", type=float, default=0.05)
    p.add_argument("--input_init1", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ensemble_alpha", type=float, default=0.5,
                   help="weight on stage 1 in ensemble; (1-alpha) on stage 0")
    p.add_argument("--csv", type=str, default="results/decolle.csv")
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

    print(f"Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    print(f"train {xtr.shape}, test {xte.shape}", flush=True)
    print(f"M0={M0}, M1={M1 if args.use_stage1 else 0}", flush=True)

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=1,
                             omega_min=args.om0_min, omega_max=args.om0_max,
                             alpha_min=args.al0_min, alpha_max=args.al0_max,
                             input_init=args.input_init0)
    p0 = init_params(cfg0, generator=gen)
    opt0 = Adam(p0.tensors(), args.lr0)

    if args.use_stage1:
        cfg1 = OscillatorConfig(n_neurons=M1, n_input_channels=M0,
                                 omega_min=args.om1_min, omega_max=args.om1_max,
                                 alpha_min=args.al1_min, alpha_max=args.al1_max,
                                 input_init=args.input_init1)
        p1 = init_params(cfg1, generator=gen)
        opt1 = Adam(p1.tensors(), args.lr1)
    else:
        cfg1 = None; p1 = None; opt1 = None

    def fwd(xb, train=False):
        out0 = forward_with_eligibility(xb, p0, cfg0, args.tail0,
                                          accumulate_traces=train,
                                          save_amp_seq=args.use_stage1)
        E0 = out0["E"]
        logits_0 = class_pool_logits(E0, class_index_0, classes, args.temperature)
        if not args.use_stage1:
            return out0, logits_0, None, None
        amp_seq_0 = out0["amp_seq"]
        out1 = forward_with_eligibility(amp_seq_0, p1, cfg1, args.tail1,
                                          accumulate_traces=train, save_amp_seq=False)
        E1 = out1["E"]
        logits_1 = class_pool_logits(E1, class_index_1, classes, args.temperature)
        return out0, logits_0, out1, logits_1

    def evaluate(x, y):
        l0_sum = 0.0; l1_sum = 0.0; lc_sum = 0.0
        a0 = 0; a1 = 0; ac = 0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            out0, logits_0, out1, logits_1 = fwd(xb, train=False)
            l0 = F.cross_entropy(logits_0, yb, reduction="sum").item()
            a0 += (logits_0.argmax(1) == yb).sum().item()
            l0_sum += l0
            if args.use_stage1:
                l1 = F.cross_entropy(logits_1, yb, reduction="sum").item()
                a1 += (logits_1.argmax(1) == yb).sum().item()
                l1_sum += l1
                # ensemble: avg softmax probs
                prob_0 = F.softmax(logits_0, dim=1)
                prob_1 = F.softmax(logits_1, dim=1)
                prob_c = (1 - args.ensemble_alpha) * prob_0 + args.ensemble_alpha * prob_1
                ac += (prob_c.argmax(1) == yb).sum().item()
                lc_sum += -torch.log(prob_c[torch.arange(prob_c.shape[0]), yb].clamp_min(1e-12)).sum().item()
            n += xb.shape[0]
        out = {"l0": l0_sum / n, "a0": a0 / n}
        if args.use_stage1:
            out.update({"l1": l1_sum / n, "a1": a1 / n, "lc": lc_sum / n, "ac": ac / n})
        return out

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    if args.use_stage1:
        writer.writerow(["epoch", "wall", "tr_a0", "tr_a1", "tr_ac",
                         "te_a0", "te_a1", "te_ac", "best_te_ac"])
    else:
        writer.writerow(["epoch", "wall", "tr_a0", "te_a0", "best_te_a0"])

    best_a = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        ev_tr = evaluate(xtr, ytr)
        ev_te = evaluate(xte, yte)
        wall = time.time() - t0
        if args.use_stage1:
            metric = ev_te["ac"]
            if metric > best_a: best_a = metric
            print(f"[ep {epoch:02d} t={wall:6.1f}s] tr a0={ev_tr['a0']:.4f} a1={ev_tr['a1']:.4f} ac={ev_tr['ac']:.4f} | "
                  f"te a0={ev_te['a0']:.4f} a1={ev_te['a1']:.4f} ac={ev_te['ac']:.4f} | best={best_a:.4f}",
                  flush=True)
            writer.writerow([epoch, f"{wall:.1f}", f"{ev_tr['a0']:.4f}", f"{ev_tr['a1']:.4f}", f"{ev_tr['ac']:.4f}",
                              f"{ev_te['a0']:.4f}", f"{ev_te['a1']:.4f}", f"{ev_te['ac']:.4f}", f"{best_a:.4f}"])
        else:
            metric = ev_te["a0"]
            if metric > best_a: best_a = metric
            print(f"[ep {epoch:02d} t={wall:6.1f}s] tr={ev_tr['a0']:.4f} te={ev_te['a0']:.4f} best={best_a:.4f}",
                  flush=True)
            writer.writerow([epoch, f"{wall:.1f}", f"{ev_tr['a0']:.4f}", f"{ev_te['a0']:.4f}", f"{best_a:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out0, logits_0, out1, logits_1 = fwd(xb, train=True)

            # Stage 0: own credit
            probs_0 = F.softmax(logits_0, dim=1)
            d_E_0 = class_pool_credit(probs_0, yb, class_index_0, args.m0_per_class,
                                       args.temperature, xb.shape[0])
            g0_d_r = (d_E_0.unsqueeze(-1) * out0["dE_d_r"]).sum(dim=0)
            g0_d_i = (d_E_0.unsqueeze(-1) * out0["dE_d_i"]).sum(dim=0)
            g0_b_r = (d_E_0 * out0["dE_b_r"]).sum(dim=0)
            g0_b_i = (d_E_0 * out0["dE_b_i"]).sum(dim=0)
            g0_om = (d_E_0 * out0["dE_omega_raw"]).sum(dim=0)
            g0_al = (d_E_0 * out0["dE_alpha_raw"]).sum(dim=0)
            opt0.step([g0_d_r, g0_d_i, g0_b_r, g0_b_i, g0_om, g0_al], args.grad_clip)

            if args.use_stage1:
                probs_1 = F.softmax(logits_1, dim=1)
                d_E_1 = class_pool_credit(probs_1, yb, class_index_1, args.m1_per_class,
                                           args.temperature, xb.shape[0])
                g1_d_r = (d_E_1.unsqueeze(-1) * out1["dE_d_r"]).sum(dim=0)
                g1_d_i = (d_E_1.unsqueeze(-1) * out1["dE_d_i"]).sum(dim=0)
                g1_b_r = (d_E_1 * out1["dE_b_r"]).sum(dim=0)
                g1_b_i = (d_E_1 * out1["dE_b_i"]).sum(dim=0)
                g1_om = (d_E_1 * out1["dE_omega_raw"]).sum(dim=0)
                g1_al = (d_E_1 * out1["dE_alpha_raw"]).sum(dim=0)
                opt1.step([g1_d_r, g1_d_i, g1_b_r, g1_b_i, g1_om, g1_al], args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc (final metric): {best_a:.4f}", flush=True)


if __name__ == "__main__":
    main()
