"""Two-stage HRN-v2 trainer for SMNIST with DECOLLE-style local supervision.

Stage 0: 1 global pool of M_0 oscillators, scalar SMNIST input.
Stage 1: 1 global pool of M_1 oscillators, multi-channel input = stage-0
         amplitude trajectory (P_0 channels).

Each stage has its OWN per-neuron label-mass head and its OWN local
classification loss. Each stage's parameters are updated by its own
local credit signal modulating its own eligibility traces. There is
no inter-stage backward pass; the hierarchy is enforced only by the
forward path.

Final prediction: weighted combination of the two stages' P(c).
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

from oscillator import OscillatorConfig, init_params, omega_of, alpha_of, forward_with_eligibility
from optim import Adam
from local_rules import (
    global_softmax_competition, label_probs, label_hebbian_step,
    credit_for_self_organising_pool,
)
from smnist_data import load_smnist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0", type=int, default=128)
    p.add_argument("--m1", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--tail0", type=int, default=200)
    p.add_argument("--tail1", type=int, default=200)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lr1", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--wta_beta", type=float, default=3.0)
    p.add_argument("--wta_beta_start", type=float, default=0.1)
    p.add_argument("--wta_beta_warmup", type=float, default=3.0)
    p.add_argument("--target_usage", type=float, default=0.0125)
    p.add_argument("--homeo_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--label_lr", type=float, default=0.02)
    p.add_argument("--label_prior", type=float, default=2.0)
    p.add_argument("--theta_init", type=float, default=1.0)
    # stage 0 frequency band (fast)
    p.add_argument("--om0_min", type=float, default=0.005)
    p.add_argument("--om0_max", type=float, default=1.2)
    p.add_argument("--al0_min", type=float, default=0.95)
    p.add_argument("--al0_max", type=float, default=0.999)
    # stage 1 frequency band (slower; envelopes evolve slowly)
    p.add_argument("--om1_min", type=float, default=0.001)
    p.add_argument("--om1_max", type=float, default=0.30)
    p.add_argument("--al1_min", type=float, default=0.97)
    p.add_argument("--al1_max", type=float, default=0.9995)
    p.add_argument("--input_init0", type=float, default=0.05)
    p.add_argument("--input_init1", type=float, default=0.02)
    p.add_argument("--center_input", action="store_true")
    p.add_argument("--combine_alpha", type=float, default=0.5,
                   help="weight on P_1 in final prediction; (1-alpha) is on P_0")
    p.add_argument("--csv", type=str, default="results/two_stage.csv")
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
        m = xtr.mean()
        xtr = xtr - m; xte = xte - m
    print(f"train {xtr.shape}, test {xte.shape}", flush=True)

    classes = 10
    M0 = args.m0; M1 = args.m1

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=1,
                            omega_min=args.om0_min, omega_max=args.om0_max,
                            alpha_min=args.al0_min, alpha_max=args.al0_max,
                            input_init=args.input_init0)
    cfg1 = OscillatorConfig(n_neurons=M1, n_input_channels=M0,
                            omega_min=args.om1_min, omega_max=args.om1_max,
                            alpha_min=args.al1_min, alpha_max=args.al1_max,
                            input_init=args.input_init1)
    p0 = init_params(cfg0, generator=gen)
    p1 = init_params(cfg1, generator=gen)

    label_mass_0 = torch.full((M0, classes), args.label_prior / classes)
    label_mass_1 = torch.full((M1, classes), args.label_prior / classes)
    theta_0 = torch.full((M0,), args.theta_init)
    theta_1 = torch.full((M1,), args.theta_init)
    usage_ema_0 = torch.full((M0,), 1.0 / M0)
    usage_ema_1 = torch.full((M1,), 1.0 / M1)

    opt0 = Adam(p0.tensors(), args.lr0)
    opt1 = Adam(p1.tensors(), args.lr1)

    def current_beta(epoch):
        if args.wta_beta_warmup <= 0: return args.wta_beta
        frac = min(1.0, epoch / args.wta_beta_warmup)
        return args.wta_beta_start + frac * (args.wta_beta - args.wta_beta_start)

    def fwd(xb, train=False, beta=None):
        # Stage 0
        out0 = forward_with_eligibility(xb, p0, cfg0, args.tail0,
                                         accumulate_traces=train, save_amp_seq=True)
        amp_seq_0 = out0["amp_seq"]                  # (B, T, M0)
        E0 = out0["E"]                                # (B, M0)
        resp_0 = global_softmax_competition(E0, theta_0, beta or args.wta_beta, args.top_k)

        # Stage 1
        out1 = forward_with_eligibility(amp_seq_0, p1, cfg1, args.tail1,
                                         accumulate_traces=train, save_amp_seq=False)
        E1 = out1["E"]                                # (B, M1)
        resp_1 = global_softmax_competition(E1, theta_1, beta or args.wta_beta, args.top_k)
        return out0, out1, resp_0, resp_1

    def predict(resp_0, resp_1):
        q0 = label_probs(label_mass_0, args.label_prior, classes)
        q1 = label_probs(label_mass_1, args.label_prior, classes)
        prob_0 = resp_0 @ q0
        prob_1 = resp_1 @ q1
        prob_combined = (1.0 - args.combine_alpha) * prob_0 + args.combine_alpha * prob_1
        return prob_0, prob_1, prob_combined

    def evaluate(x, y, beta=None):
        ls0 = ls1 = lsc = 0.0
        ac0 = ac1 = acc_c = 0.0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            _, _, resp_0, resp_1 = fwd(xb, train=False, beta=beta)
            prob_0, prob_1, prob_c = predict(resp_0, resp_1)
            py0 = prob_0[torch.arange(prob_0.shape[0]), yb].clamp_min(1e-12)
            py1 = prob_1[torch.arange(prob_1.shape[0]), yb].clamp_min(1e-12)
            pyc = prob_c[torch.arange(prob_c.shape[0]), yb].clamp_min(1e-12)
            bs = xb.shape[0]
            ls0 += -torch.log(py0).sum().item(); ls1 += -torch.log(py1).sum().item()
            lsc += -torch.log(pyc).sum().item()
            ac0 += (prob_0.argmax(1) == yb).sum().item()
            ac1 += (prob_1.argmax(1) == yb).sum().item()
            acc_c += (prob_c.argmax(1) == yb).sum().item()
            n += bs
        return (ls0/n, ac0/n), (ls1/n, ac1/n), (lsc/n, acc_c/n)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall", "tr_acc0", "tr_acc1", "tr_acc_c",
                     "te_acc0", "te_acc1", "te_acc_c", "best_te_c", "beta"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        beta_now = current_beta(epoch)
        (l0_tr, a0_tr), (l1_tr, a1_tr), (lc_tr, ac_tr) = evaluate(xtr, ytr, beta=beta_now)
        (l0_te, a0_te), (l1_te, a1_te), (lc_te, ac_te) = evaluate(xte, yte, beta=beta_now)
        if ac_te > best: best = ac_te
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s beta={beta_now:.2f}] "
              f"train: s0={a0_tr:.4f} s1={a1_tr:.4f} c={ac_tr:.4f}  "
              f"test: s0={a0_te:.4f} s1={a1_te:.4f} c={ac_te:.4f} best_c={best:.4f}",
              flush=True)
        writer.writerow([epoch, f"{wall:.1f}",
                          f"{a0_tr:.4f}", f"{a1_tr:.4f}", f"{ac_tr:.4f}",
                          f"{a0_te:.4f}", f"{a1_te:.4f}", f"{ac_te:.4f}",
                          f"{best:.4f}", f"{beta_now:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out0, out1, resp_0, resp_1 = fwd(xb, train=True, beta=beta_now)

            # Stage 0 credit + gradient
            delta_0 = credit_for_self_organising_pool(resp_0, label_mass_0, yb, classes,
                                                      args.label_prior, credit_gain=1.0)
            g0_d_r = (delta_0.unsqueeze(-1) * out0["dE_d_r"]).sum(dim=0)
            g0_d_i = (delta_0.unsqueeze(-1) * out0["dE_d_i"]).sum(dim=0)
            g0_b_r = (delta_0 * out0["dE_b_r"]).sum(dim=0)
            g0_b_i = (delta_0 * out0["dE_b_i"]).sum(dim=0)
            g0_om = (delta_0 * out0["dE_omega_raw"]).sum(dim=0)
            g0_al = (delta_0 * out0["dE_alpha_raw"]).sum(dim=0)
            opt0.step([g0_d_r, g0_d_i, g0_b_r, g0_b_i, g0_om, g0_al], args.grad_clip)

            # Stage 1 credit + gradient
            delta_1 = credit_for_self_organising_pool(resp_1, label_mass_1, yb, classes,
                                                      args.label_prior, credit_gain=1.0)
            g1_d_r = (delta_1.unsqueeze(-1) * out1["dE_d_r"]).sum(dim=0)
            g1_d_i = (delta_1.unsqueeze(-1) * out1["dE_d_i"]).sum(dim=0)
            g1_b_r = (delta_1 * out1["dE_b_r"]).sum(dim=0)
            g1_b_i = (delta_1 * out1["dE_b_i"]).sum(dim=0)
            g1_om = (delta_1 * out1["dE_omega_raw"]).sum(dim=0)
            g1_al = (delta_1 * out1["dE_alpha_raw"]).sum(dim=0)
            opt1.step([g1_d_r, g1_d_i, g1_b_r, g1_b_i, g1_om, g1_al], args.grad_clip)

            # Hebbian on label tags
            label_hebbian_step(label_mass_0, resp_0, yb, classes, args.label_lr, decay=0.0)
            label_hebbian_step(label_mass_1, resp_1, yb, classes, args.label_lr, decay=0.0)

            # Homeostasis on theta (single-pool case)
            with torch.no_grad():
                u0 = resp_0.mean(dim=0)
                usage_ema_0.mul_(1.0 - args.ema_lr).add_(args.ema_lr * u0)
                theta_0.add_(args.homeo_lr * (u0 - args.target_usage)).clamp_(-2.0, 5.0)
                u1 = resp_1.mean(dim=0)
                usage_ema_1.mul_(1.0 - args.ema_lr).add_(args.ema_lr * u1)
                theta_1.add_(args.homeo_lr * (u1 - args.target_usage)).clamp_(-2.0, 5.0)

    f_csv.close()
    print(f"\nBest test acc (combined): {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
