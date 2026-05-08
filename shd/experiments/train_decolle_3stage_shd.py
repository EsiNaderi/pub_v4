"""DECOLLE 3-stage HRN on Spiking Heidelberg Digits (SHD).

Same architecture as the 89.9% SMNIST run (3 stages of complex damped
linear oscillators, class-pool tail-energy readout per stage,
eligibility-trace local credit, DECOLLE-style local supervision per
stage, ensemble of stage probs).

Differences for SHD:

* Input is (B, T=100, F=700) cochlear spike-rate per-channel rather
  than a single scalar pixel sequence. Stage 0 therefore uses
  ``forward_with_eligibility_sparse`` with a per-neuron random fan-in
  K_0 (typically 32-64) into the 700 input channels.

* 20 classes instead of 10.

* Frequency bands re-tuned to T=100 (SHD).

No BPTT, no surrogate spikes, no inter-stage backward pass --- the
substrate is identical to the SMNIST architecture.
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

from oscillator import (
    OscillatorConfig, init_params, omega_of, alpha_of,
    forward_with_eligibility_sparse,
)
from optim import Adam
from shd_data import load_shd, N_CLASSES


def class_pool_logits(E, class_index, classes, temperature):
    logits = torch.zeros(E.shape[0], classes, device=E.device, dtype=E.dtype)
    for c in range(classes):
        logits[:, c] = E[:, class_index == c].mean(dim=1)
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def make_class_aligned_fanin(classes, m_curr_per_class, m_prev_per_class, K, gen):
    """Each curr-stage neuron gets K stage-prev inputs: half same-class + half random other."""
    P = classes * m_curr_per_class
    class_index_curr = torch.repeat_interleave(torch.arange(classes), m_curr_per_class)
    class_index_prev = torch.repeat_interleave(torch.arange(classes), m_prev_per_class)
    half_k = max(1, K // 2)
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        c = int(class_index_curr[i].item())
        same = (class_index_prev == c).nonzero(as_tuple=True)[0]
        other = (class_index_prev != c).nonzero(as_tuple=True)[0]
        sc_pick = same[torch.randperm(len(same), generator=gen)[:half_k]]
        oc_pick = other[torch.randperm(len(other), generator=gen)[:K - half_k]]
        in_idx[i] = torch.cat([sc_pick, oc_pick])
    return in_idx


def make_stage0_random_fanin(P, F_in, K, gen):
    """Random sparse fan-in for stage 0 (no class structure on the input side)."""
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        perm = torch.randperm(F_in, generator=gen)
        in_idx[i] = perm[:K]
    return in_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0_per_class", type=int, default=12)         # 20*12 = 240 stage-0 neurons
    p.add_argument("--m1_per_class", type=int, default=12)
    p.add_argument("--m2_per_class", type=int, default=12)
    p.add_argument("--k0_fanin", type=int, default=48,
                   help="how many of 700 input channels each stage-0 neuron sees")
    p.add_argument("--k1_fanin", type=int, default=12)
    p.add_argument("--k2_fanin", type=int, default=12)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=4000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--tail0", type=int, default=60)
    p.add_argument("--tail1", type=int, default=80)
    p.add_argument("--tail2", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_decay_after", type=int, default=10)
    p.add_argument("--lr_decay_after_2", type=int, default=18, help="optional second LR decay")
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    # frequency bands by stage (T=100, so bands are higher than SMNIST T=784)
    p.add_argument("--om0_min", type=float, default=0.05); p.add_argument("--om0_max", type=float, default=1.5)
    p.add_argument("--om1_min", type=float, default=0.01); p.add_argument("--om1_max", type=float, default=0.40)
    p.add_argument("--om2_min", type=float, default=0.005); p.add_argument("--om2_max", type=float, default=0.15)
    p.add_argument("--al0_min", type=float, default=0.85); p.add_argument("--al0_max", type=float, default=0.995)
    p.add_argument("--al1_min", type=float, default=0.92); p.add_argument("--al1_max", type=float, default=0.998)
    p.add_argument("--al2_min", type=float, default=0.95); p.add_argument("--al2_max", type=float, default=0.999)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ensemble_w0", type=float, default=0.2)
    p.add_argument("--ensemble_w1", type=float, default=0.4)
    p.add_argument("--ensemble_w2", type=float, default=0.4)
    p.add_argument("--csv", type=str, default="results/shd_3stage.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = N_CLASSES
    M0 = classes * args.m0_per_class
    M1 = classes * args.m1_per_class
    M2 = classes * args.m2_per_class
    class_index_0 = torch.repeat_interleave(torch.arange(classes), args.m0_per_class)
    class_index_1 = torch.repeat_interleave(torch.arange(classes), args.m1_per_class)
    class_index_2 = torch.repeat_interleave(torch.arange(classes), args.m2_per_class)

    print("Loading SHD ...", flush=True)
    xtr, ytr, xte, yte = load_shd()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    F_in = xtr.shape[2]
    print(f"train {tuple(xtr.shape)}, test {tuple(xte.shape)}, classes={classes}", flush=True)
    print(f"M0={M0}, M1={M1}, M2={M2}, K0={args.k0_fanin}, K1={args.k1_fanin}, K2={args.k2_fanin}",
          flush=True)

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=args.k0_fanin,
                             omega_min=args.om0_min, omega_max=args.om0_max,
                             alpha_min=args.al0_min, alpha_max=args.al0_max,
                             input_init=args.input_init)
    cfg1 = OscillatorConfig(n_neurons=M1, n_input_channels=args.k1_fanin,
                             omega_min=args.om1_min, omega_max=args.om1_max,
                             alpha_min=args.al1_min, alpha_max=args.al1_max,
                             input_init=args.input_init)
    cfg2 = OscillatorConfig(n_neurons=M2, n_input_channels=args.k2_fanin,
                             omega_min=args.om2_min, omega_max=args.om2_max,
                             alpha_min=args.al2_min, alpha_max=args.al2_max,
                             input_init=args.input_init)
    p0 = init_params(cfg0, generator=gen)
    p1 = init_params(cfg1, generator=gen)
    p2 = init_params(cfg2, generator=gen)
    in_idx_0 = make_stage0_random_fanin(M0, F_in, args.k0_fanin, gen)
    in_idx_1 = make_class_aligned_fanin(classes, args.m1_per_class, args.m0_per_class, args.k1_fanin, gen)
    in_idx_2 = make_class_aligned_fanin(classes, args.m2_per_class, args.m1_per_class, args.k2_fanin, gen)
    opt0 = Adam(p0.tensors(), args.lr)
    opt1 = Adam(p1.tensors(), args.lr)
    opt2 = Adam(p2.tensors(), args.lr)

    def fwd(xb, train=False):
        out0 = forward_with_eligibility_sparse(xb, in_idx_0, p0, cfg0, args.tail0,
                                                 accumulate_traces=train, save_amp_seq=True)
        amp0 = out0["amp_seq"]
        E0 = out0["E"]
        logits_0 = class_pool_logits(E0, class_index_0, classes, args.temperature)

        out1 = forward_with_eligibility_sparse(amp0, in_idx_1, p1, cfg1, args.tail1,
                                                 accumulate_traces=train, save_amp_seq=True)
        amp1 = out1["amp_seq"]
        E1 = out1["E"]
        logits_1 = class_pool_logits(E1, class_index_1, classes, args.temperature)

        out2 = forward_with_eligibility_sparse(amp1, in_idx_2, p2, cfg2, args.tail2,
                                                 accumulate_traces=train, save_amp_seq=False)
        E2 = out2["E"]
        logits_2 = class_pool_logits(E2, class_index_2, classes, args.temperature)

        return out0, logits_0, out1, logits_1, out2, logits_2

    def evaluate(x, y):
        a0 = a1 = a2 = ac = 0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch]
            yb = y[s : s + args.batch]
            _, l0, _, l1, _, l2 = fwd(xb, train=False)
            a0 += (l0.argmax(1) == yb).sum().item()
            a1 += (l1.argmax(1) == yb).sum().item()
            a2 += (l2.argmax(1) == yb).sum().item()
            p0_ = F.softmax(l0, dim=1); p1_ = F.softmax(l1, dim=1); p2_ = F.softmax(l2, dim=1)
            pc = args.ensemble_w0 * p0_ + args.ensemble_w1 * p1_ + args.ensemble_w2 * p2_
            ac += (pc.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return a0 / n, a1 / n, a2 / n, ac / n

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "tr_a0", "tr_a1", "tr_a2", "tr_ac",
                     "te_a0", "te_a1", "te_a2", "te_ac", "best_te_ac",
                     "om0", "om1", "om2"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        a0_tr, a1_tr, a2_tr, ac_tr = evaluate(xtr, ytr)
        a0_te, a1_te, a2_te, ac_te = evaluate(xte, yte)
        if ac_te > best: best = ac_te
        om0_m = float(omega_of(p0, cfg0).mean().item())
        om1_m = float(omega_of(p1, cfg1).mean().item())
        om2_m = float(omega_of(p2, cfg2).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] "
              f"tr a0={a0_tr:.3f} a1={a1_tr:.3f} a2={a2_tr:.3f} ac={ac_tr:.3f} | "
              f"te a0={a0_te:.3f} a1={a1_te:.3f} a2={a2_te:.3f} ac={ac_te:.3f} | best={best:.4f} | "
              f"om0={om0_m:.3f} om1={om1_m:.3f} om2={om2_m:.4f}",
              flush=True)
        writer.writerow([epoch, f"{wall:.1f}",
                          f"{a0_tr:.4f}", f"{a1_tr:.4f}", f"{a2_tr:.4f}", f"{ac_tr:.4f}",
                          f"{a0_te:.4f}", f"{a1_te:.4f}", f"{a2_te:.4f}", f"{ac_te:.4f}", f"{best:.4f}",
                          f"{om0_m:.4f}", f"{om1_m:.4f}", f"{om2_m:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break
        if epoch == args.lr_decay_after:
            opt0.lr *= args.lr_decay_factor
            opt1.lr *= args.lr_decay_factor
            opt2.lr *= args.lr_decay_factor
            print(f"  LR decay #1: lr={opt0.lr}", flush=True)
        if args.lr_decay_after_2 > 0 and epoch == args.lr_decay_after_2:
            opt0.lr *= args.lr_decay_factor
            opt1.lr *= args.lr_decay_factor
            opt2.lr *= args.lr_decay_factor
            print(f"  LR decay #2: lr={opt0.lr}", flush=True)

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx]
            yb = ytr[idx]
            out0, logits_0, out1, logits_1, out2, logits_2 = fwd(xb, train=True)
            onehot = F.one_hot(yb, classes).float()

            for (out, logits, ci, M_per, opt) in [
                (out0, logits_0, class_index_0, args.m0_per_class, opt0),
                (out1, logits_1, class_index_1, args.m1_per_class, opt1),
                (out2, logits_2, class_index_2, args.m2_per_class, opt2),
            ]:
                probs = F.softmax(logits, dim=1)
                d_logit = (probs - onehot) / xb.shape[0]
                d_E = d_logit[:, ci] * (args.temperature / M_per)
                g = [(d_E.unsqueeze(-1) * out["dE_d_r"]).sum(dim=0),
                     (d_E.unsqueeze(-1) * out["dE_d_i"]).sum(dim=0),
                     (d_E * out["dE_b_r"]).sum(dim=0),
                     (d_E * out["dE_b_i"]).sum(dim=0),
                     (d_E * out["dE_omega_raw"]).sum(dim=0),
                     (d_E * out["dE_alpha_raw"]).sum(dim=0)]
                opt.step(g, args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc (ensemble): {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
