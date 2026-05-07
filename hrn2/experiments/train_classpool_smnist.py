"""Class-pool output trainer for SMNIST.

Architecture:
    Input scalar(t) (B, T, 1)
       │
       ▼
    Stage 0 oscillator bank (M_0 neurons, diverse ω, amplitude trajectory)
       │
       │  amp_seq_0 (B, T, M_0)
       ▼
    Output stage: 10 class pools × M_out neurons
       Each neuron is a damped complex oscillator driven by stage-0 amp_seq.
       Tail-energy is the per-neuron readout.
       Logit_c = mean_{i in class-pool c} E_i.
       Loss = cross-entropy on logits.
       Per-neuron credit: δ_i = (prob_c − 1[c==y]) / M_out
                          (where c is the class-pool of i)
       Local rule: each neuron updates its (d, b, ω, α) by
                   eligibility-trace × δ_i.

Stage 0 parameters are also trained via the same per-neuron credit
signal *transported through random feedback alignment*. This is the
biologically plausible substitute for backprop through the inter-stage
input weights.

This is the simplest hierarchy we expect to break the single-layer
ceiling: class-pool supervision gives much stronger credit than label
tags, and the hierarchy gives temporal-feature composition.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0", type=int, default=128)
    p.add_argument("--m_out", type=int, default=24)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--tail0", type=int, default=200)
    p.add_argument("--tail_out", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--lr_out", type=float, default=0.002)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--amp0_normalize", action="store_true",
                   help="normalize stage-0 amp_seq to mean ~1 before feeding to output stage")
    # stage 0 frequency band
    p.add_argument("--om0_min", type=float, default=0.005)
    p.add_argument("--om0_max", type=float, default=1.2)
    p.add_argument("--al0_min", type=float, default=0.95)
    p.add_argument("--al0_max", type=float, default=0.999)
    # output stage frequency band (slower)
    p.add_argument("--om_out_min", type=float, default=0.001)
    p.add_argument("--om_out_max", type=float, default=0.30)
    p.add_argument("--al_out_min", type=float, default=0.97)
    p.add_argument("--al_out_max", type=float, default=0.9995)
    p.add_argument("--input_init0", type=float, default=0.05)
    p.add_argument("--input_init_out", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--center_input", action="store_true")
    p.add_argument("--use_stage0_credit", action="store_true",
                   help="if set, transport credit back to stage 0 via random feedback")
    p.add_argument("--csv", type=str, default="results/classpool_smoke.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    if args.center_input:
        m = xtr.mean(); xtr = xtr - m; xte = xte - m
    print(f"train {xtr.shape}, test {xte.shape}", flush=True)

    classes = 10
    M0 = args.m0
    Mout_per_class = args.m_out
    P_out = classes * Mout_per_class

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=1,
                            omega_min=args.om0_min, omega_max=args.om0_max,
                            alpha_min=args.al0_min, alpha_max=args.al0_max,
                            input_init=args.input_init0)
    cfg_out = OscillatorConfig(n_neurons=P_out, n_input_channels=M0,
                                omega_min=args.om_out_min, omega_max=args.om_out_max,
                                alpha_min=args.al_out_min, alpha_max=args.al_out_max,
                                input_init=args.input_init_out)
    p0 = init_params(cfg0, generator=gen)
    p_out = init_params(cfg_out, generator=gen)

    # Class index per output neuron
    class_index = torch.repeat_interleave(torch.arange(classes), Mout_per_class)  # (P_out,)

    # Random feedback for stage 0 (Lillicrap)
    B0_feedback = torch.randn(P_out, M0, generator=gen) / max(P_out, 1) ** 0.5

    opt0 = Adam(p0.tensors(), args.lr0)
    opt_out = Adam(p_out.tensors(), args.lr_out)

    def fwd(xb, train=False):
        # Stage 0
        out0 = forward_with_eligibility(xb, p0, cfg0, args.tail0,
                                          accumulate_traces=train, save_amp_seq=True)
        amp_seq_0 = out0["amp_seq"]                      # (B, T, M0)
        if args.amp0_normalize:
            # per-batch, per-channel mean (across time) — keep relative variation
            scale = amp_seq_0.mean(dim=(0, 1)).clamp_min(1e-6)
            amp_seq_0 = amp_seq_0 / scale.view(1, 1, -1)

        # Output stage: each neuron is driven by amp_seq_0
        out_stage = forward_with_eligibility(amp_seq_0, p_out, cfg_out, args.tail_out,
                                              accumulate_traces=train, save_amp_seq=False)
        E_out = out_stage["E"]                            # (B, P_out)
        # logits per class = mean of E_out over class members
        logits = torch.zeros(xb.shape[0], classes)
        for c in range(classes):
            mask = (class_index == c)
            logits[:, c] = E_out[:, mask].mean(dim=1)
        # Center per-batch to prevent scale explosion, then apply temperature.
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits * args.temperature
        return out0, out_stage, E_out, logits

    def evaluate(x, y):
        ls = 0.0; ac = 0.0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            _, _, _, logits = fwd(xb, train=False)
            loss = F.cross_entropy(logits, yb, reduction="sum").item()
            pred = logits.argmax(dim=1)
            bs = xb.shape[0]
            ls += loss; ac += (pred == yb).sum().item(); n += bs
        return ls / n, ac / n

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "train_acc", "test_acc", "best_test"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        loss_tr, acc_tr = evaluate(xtr, ytr)
        loss_te, acc_te = evaluate(xte, yte)
        if acc_te > best: best = acc_te
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] tr_loss={loss_tr:.3f} tr_acc={acc_tr:.4f} "
              f"te_loss={loss_te:.3f} te_acc={acc_te:.4f} best={best:.4f}", flush=True)
        writer.writerow([epoch, f"{wall:.1f}", f"{acc_tr:.4f}", f"{acc_te:.4f}", f"{best:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out0, out_stage, E_out, logits = fwd(xb, train=True)

            # CE gradient on logits (reduction='mean'-style) → per-class credit
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(yb, classes).float()
            d_logit = (probs - onehot) / xb.shape[0]                   # (B, classes)
            # logit_c = (T/M) sum_i E_i; dlogit/dE_i = T/M for i in pool c
            # so dloss/dE_i = T * d_logit[c(i)] / M_out
            d_E_out = d_logit[:, class_index] * (args.temperature / Mout_per_class)  # (B, P_out)

            # Output-stage gradient assembly
            go_d_r = (d_E_out.unsqueeze(-1) * out_stage["dE_d_r"]).sum(dim=0)
            go_d_i = (d_E_out.unsqueeze(-1) * out_stage["dE_d_i"]).sum(dim=0)
            go_b_r = (d_E_out * out_stage["dE_b_r"]).sum(dim=0)
            go_b_i = (d_E_out * out_stage["dE_b_i"]).sum(dim=0)
            go_om = (d_E_out * out_stage["dE_omega_raw"]).sum(dim=0)
            go_al = (d_E_out * out_stage["dE_alpha_raw"]).sum(dim=0)
            opt_out.step([go_d_r, go_d_i, go_b_r, go_b_i, go_om, go_al], args.grad_clip)

            # Optionally transport credit to stage 0 via random feedback
            if args.use_stage0_credit:
                d_E_0 = d_E_out @ B0_feedback                          # (B, M0)
                g0_d_r = (d_E_0.unsqueeze(-1) * out0["dE_d_r"]).sum(dim=0)
                g0_d_i = (d_E_0.unsqueeze(-1) * out0["dE_d_i"]).sum(dim=0)
                g0_b_r = (d_E_0 * out0["dE_b_r"]).sum(dim=0)
                g0_b_i = (d_E_0 * out0["dE_b_i"]).sum(dim=0)
                g0_om = (d_E_0 * out0["dE_omega_raw"]).sum(dim=0)
                g0_al = (d_E_0 * out0["dE_alpha_raw"]).sum(dim=0)
                opt0.step([g0_d_r, g0_d_i, g0_b_r, g0_b_i, g0_om, g0_al], args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
