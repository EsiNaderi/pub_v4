"""DECOLLE 3-stage SPIKING HRN on SHD.

Real spiking variant: every neuron emits binary spikes
    s(t) ~ Bernoulli( sigmoid( beta * (|z(t)|^2 - theta_i) ) )
which form the inter-stage signal. The smooth expected rate p(t) is
used for the (per-stage) loss readout, with gradient flowing through
sigma' (the natural derivative of the Bernoulli expected rate; not a
surrogate of a Heaviside).

Per-neuron threshold theta_i is homeostatically maintained toward
target rate r_target by an EMA + theta update.

No BPTT. No surrogate-of-Heaviside. No inter-stage backward pass.
Each stage trains by its own class-pool credit on its own tail rate.
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

from oscillator import OscillatorConfig, init_params, omega_of, alpha_of
from oscillator_spiking import forward_with_eligibility_sparse_spiking
from optim import Adam
from shd_data import load_shd, N_CLASSES


def class_pool_logits(rho, class_index, classes, temperature):
    """Class-pool tail-rate logits, mean-centered + temperature scaled."""
    logits = torch.zeros(rho.shape[0], classes, device=rho.device, dtype=rho.dtype)
    for c in range(classes):
        logits[:, c] = rho[:, class_index == c].mean(dim=1)
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def make_class_aligned_fanin(classes, m_curr_per_class, m_prev_per_class, K, gen):
    P = classes * m_curr_per_class
    class_index_curr = torch.repeat_interleave(torch.arange(classes), m_curr_per_class)
    class_index_prev = torch.repeat_interleave(torch.arange(classes), m_prev_per_class)
    half_k = max(1, K // 2)
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        c = int(class_index_curr[i].item())
        same = (class_index_prev == c).nonzero(as_tuple=True)[0]
        other = (class_index_prev != c).nonzero(as_tuple=True)[0]
        sc = same[torch.randperm(len(same), generator=gen)[:half_k]]
        oc = other[torch.randperm(len(other), generator=gen)[:K - half_k]]
        in_idx[i] = torch.cat([sc, oc])
    return in_idx


def make_stage0_random_fanin(P, F_in, K, gen):
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        perm = torch.randperm(F_in, generator=gen)
        in_idx[i] = perm[:K]
    return in_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m0_per_class", type=int, default=12)
    p.add_argument("--m1_per_class", type=int, default=12)
    p.add_argument("--m2_per_class", type=int, default=12)
    p.add_argument("--k0_fanin", type=int, default=48)
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
    p.add_argument("--lr_decay_after_2", type=int, default=18)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=2.0)
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
    # Spiking-specific knobs
    p.add_argument("--beta", type=float, default=8.0,
                   help="sharpness of spike probability sigmoid (in 1/amp^2 units)")
    p.add_argument("--theta_init", type=float, default=0.5)
    p.add_argument("--target_rate", type=float, default=0.1,
                   help="target per-step firing probability for homeostasis")
    p.add_argument("--theta_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--no_sample_binary", action="store_true",
                   help="if set, propagate smooth rates instead of binary spikes")
    p.add_argument("--csv", type=str, default="results/shd_3stage_spiking.csv")
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
    print(f"train {tuple(xtr.shape)}, test {tuple(xte.shape)}", flush=True)
    print(f"M0={M0}, M1={M1}, M2={M2}, K0={args.k0_fanin}, K1={args.k1_fanin}, K2={args.k2_fanin}",
          flush=True)
    print(f"Spiking mode: sample_binary={not args.no_sample_binary}, beta={args.beta}, "
          f"target_rate={args.target_rate}", flush=True)

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

    # Per-neuron homeostatic threshold
    theta0 = torch.full((M0,), args.theta_init)
    theta1 = torch.full((M1,), args.theta_init)
    theta2 = torch.full((M2,), args.theta_init)
    rate_ema_0 = torch.full((M0,), args.target_rate)
    rate_ema_1 = torch.full((M1,), args.target_rate)
    rate_ema_2 = torch.full((M2,), args.target_rate)

    sample_binary = not args.no_sample_binary

    def fwd(xb, train=False):
        out0 = forward_with_eligibility_sparse_spiking(
            xb, in_idx_0, p0, cfg0, args.tail0,
            threshold=theta0, beta=args.beta,
            accumulate_traces=train, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen)
        s0 = out0["spike_seq"]
        rho0 = out0["rho"]
        logits_0 = class_pool_logits(rho0, class_index_0, classes, args.temperature)

        out1 = forward_with_eligibility_sparse_spiking(
            s0, in_idx_1, p1, cfg1, args.tail1,
            threshold=theta1, beta=args.beta,
            accumulate_traces=train, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen)
        s1 = out1["spike_seq"]
        rho1 = out1["rho"]
        logits_1 = class_pool_logits(rho1, class_index_1, classes, args.temperature)

        out2 = forward_with_eligibility_sparse_spiking(
            s1, in_idx_2, p2, cfg2, args.tail2,
            threshold=theta2, beta=args.beta,
            accumulate_traces=train, save_spike_seq=False,
            sample_binary=sample_binary, rng=gen)
        rho2 = out2["rho"]
        logits_2 = class_pool_logits(rho2, class_index_2, classes, args.temperature)

        return out0, logits_0, out1, logits_1, out2, logits_2

    def evaluate(x, y):
        # Two parallel accuracies:
        #   acc_smooth: class-pool readout uses smooth tail-rate rho (the loss readout)
        #   acc_spike:  class-pool readout uses BINARY tail-spike-rate (deployment-faithful)
        a0 = a1 = a2 = ac = 0
        a0_s = a1_s = a2_s = ac_s = 0
        n = 0
        rates = [0.0, 0.0, 0.0]
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch]
            yb = y[s : s + args.batch]
            o0, l0, o1, l1, o2, l2 = fwd(xb, train=False)
            # smooth-rate accuracy
            a0 += (l0.argmax(1) == yb).sum().item()
            a1 += (l1.argmax(1) == yb).sum().item()
            a2 += (l2.argmax(1) == yb).sum().item()
            p0_ = F.softmax(l0, dim=1); p1_ = F.softmax(l1, dim=1); p2_ = F.softmax(l2, dim=1)
            pc = args.ensemble_w0 * p0_ + args.ensemble_w1 * p1_ + args.ensemble_w2 * p2_
            ac += (pc.argmax(1) == yb).sum().item()
            # binary-spike-rate accuracy (deployment-faithful; uses spike_count / tail)
            sl0 = class_pool_logits(o0["spike_rate"], class_index_0, classes, args.temperature)
            sl1 = class_pool_logits(o1["spike_rate"], class_index_1, classes, args.temperature)
            sl2 = class_pool_logits(o2["spike_rate"], class_index_2, classes, args.temperature)
            a0_s += (sl0.argmax(1) == yb).sum().item()
            a1_s += (sl1.argmax(1) == yb).sum().item()
            a2_s += (sl2.argmax(1) == yb).sum().item()
            sp0 = F.softmax(sl0, 1); sp1 = F.softmax(sl1, 1); sp2 = F.softmax(sl2, 1)
            spc = args.ensemble_w0 * sp0 + args.ensemble_w1 * sp1 + args.ensemble_w2 * sp2
            ac_s += (spc.argmax(1) == yb).sum().item()
            rates[0] += float(o0["spike_rate"].mean().item()) * xb.shape[0]
            rates[1] += float(o1["spike_rate"].mean().item()) * xb.shape[0]
            rates[2] += float(o2["spike_rate"].mean().item()) * xb.shape[0]
            n += xb.shape[0]
        return (
            (a0 / n, a1 / n, a2 / n, ac / n),          # smooth-rate
            (a0_s / n, a1_s / n, a2_s / n, ac_s / n),  # binary-spike
            [r / n for r in rates],
        )

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall",
                     "tr_ac_smooth", "tr_ac_spike",
                     "te_a0_smooth", "te_a1_smooth", "te_a2_smooth", "te_ac_smooth",
                     "te_a0_spike", "te_a1_spike", "te_a2_spike", "te_ac_spike",
                     "best_te_ac_spike", "rate0", "rate1", "rate2", "om0", "om1", "om2"])

    best_smooth = 0.0
    best_spike = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        (a0_tr, a1_tr, a2_tr, ac_tr), (a0_tr_s, a1_tr_s, a2_tr_s, ac_tr_s), _ = evaluate(xtr, ytr)
        (a0_te, a1_te, a2_te, ac_te), (a0_te_s, a1_te_s, a2_te_s, ac_te_s), te_rates = evaluate(xte, yte)
        if ac_te > best_smooth: best_smooth = ac_te
        if ac_te_s > best_spike: best_spike = ac_te_s
        om0_m = float(omega_of(p0, cfg0).mean().item())
        om1_m = float(omega_of(p1, cfg1).mean().item())
        om2_m = float(omega_of(p2, cfg2).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] "
              f"smooth te={ac_te:.3f} (best {best_smooth:.4f}) | "
              f"BINARY-SPIKE te={ac_te_s:.3f} (best {best_spike:.4f}) | "
              f"per-stage spike te {a0_te_s:.3f}/{a1_te_s:.3f}/{a2_te_s:.3f} | "
              f"rates {te_rates[0]:.3f}/{te_rates[1]:.3f}/{te_rates[2]:.3f} | "
              f"om {om0_m:.3f}/{om1_m:.3f}/{om2_m:.4f}",
              flush=True)
        writer.writerow([epoch, f"{wall:.1f}",
                          f"{ac_tr:.4f}", f"{ac_tr_s:.4f}",
                          f"{a0_te:.4f}", f"{a1_te:.4f}", f"{a2_te:.4f}", f"{ac_te:.4f}",
                          f"{a0_te_s:.4f}", f"{a1_te_s:.4f}", f"{a2_te_s:.4f}", f"{ac_te_s:.4f}",
                          f"{best_spike:.4f}",
                          f"{te_rates[0]:.4f}", f"{te_rates[1]:.4f}", f"{te_rates[2]:.4f}",
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

            for (out, logits, ci, M_per, opt, theta, rate_ema) in [
                (out0, logits_0, class_index_0, args.m0_per_class, opt0, theta0, rate_ema_0),
                (out1, logits_1, class_index_1, args.m1_per_class, opt1, theta1, rate_ema_1),
                (out2, logits_2, class_index_2, args.m2_per_class, opt2, theta2, rate_ema_2),
            ]:
                probs = F.softmax(logits, dim=1)
                d_logit = (probs - onehot) / xb.shape[0]
                d_rho = d_logit[:, ci] * (args.temperature / M_per)
                g = [(d_rho.unsqueeze(-1) * out["dRho_d_r"]).sum(dim=0),
                     (d_rho.unsqueeze(-1) * out["dRho_d_i"]).sum(dim=0),
                     (d_rho * out["dRho_b_r"]).sum(dim=0),
                     (d_rho * out["dRho_b_i"]).sum(dim=0),
                     (d_rho * out["dRho_omega_raw"]).sum(dim=0),
                     (d_rho * out["dRho_alpha_raw"]).sum(dim=0)]
                opt.step(g, args.grad_clip)

                # Homeostatic threshold update -- target a per-step firing rate
                with torch.no_grad():
                    r_obs = out["rho"].mean(dim=0)             # (P,) batch-mean rate
                    rate_ema.mul_(1.0 - args.ema_lr).add_(args.ema_lr * r_obs)
                    theta.add_(args.theta_lr * (rate_ema - args.target_rate))

    f_csv.close()
    print(f"\nBest test acc (smooth-rate ensemble): {best_smooth:.4f}", flush=True)
    print(f"Best test acc (BINARY-SPIKE ensemble): {best_spike:.4f}   <-- bio-faithful", flush=True)


if __name__ == "__main__":
    main()
