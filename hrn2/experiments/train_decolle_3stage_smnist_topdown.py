"""DECOLLE 3-stage SPIKING HRN with INTER-STAGE TOP-DOWN FEEDBACK.

Adds a predictive-coding-style top-down pathway: stages 0 and 1
receive binary spikes from stage L+1's previous-pass spike train.

Two-pass forward:
  Pass 1 (bottom-up, no traces): get spike trains s0_p1, s1_p1, s2_p1.
  Pass 2 (with top-down, with traces): each stage L receives top-down
    from stage L+1's pass-1 spike train. Eligibility traces are
    accumulated; loss and gradient come from pass 2.

Includes the same compensation mechanisms as the 3-stage spiking
trainer:
  - Subtractive reset on spike (kappa)
  - Intra-stage sparse recurrence (rec_idx + rec_params)
  - Class-pool tail-rate readout per stage with DECOLLE local loss

Spike-only inter-neuron communication. No BPTT, no surrogate of
Heaviside, no symmetric backward pass. Top-down weights are learned
by their own forward eligibility traces in pass 2.
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
from oscillator_spiking import (
    forward_with_eligibility_sparse_spiking,
    RecurrentParams, init_recurrent_params,
)
from optim import Adam
from smnist_data import load_smnist


def class_pool_logits(rho, class_index, classes, temperature):
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


def make_intra_stage_rec_idx(P, K_rec, gen):
    in_idx = torch.zeros(P, K_rec, dtype=torch.long)
    for i in range(P):
        candidates = torch.cat([torch.arange(i), torch.arange(i + 1, P)])
        perm = candidates[torch.randperm(len(candidates), generator=gen)[:K_rec]]
        in_idx[i] = perm
    return in_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m_per_class", type=int, default=24)
    p.add_argument("--k_fanin", type=int, default=12)
    p.add_argument("--rec_k", type=int, default=8)
    p.add_argument("--top_k", type=int, default=12,
                   help="top-down sparse fan-in size (each stage-L neuron sees K_top spikes from stage L+1)")
    p.add_argument("--rec_init", type=float, default=0.02)
    p.add_argument("--top_init", type=float, default=0.02)
    p.add_argument("--kappa_reset", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--tail0", type=int, default=200)
    p.add_argument("--tail1", type=int, default=300)
    p.add_argument("--tail2", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_decay_after", type=int, default=10)
    p.add_argument("--lr_decay_after_2", type=int, default=18)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--om0_min", type=float, default=0.005); p.add_argument("--om0_max", type=float, default=1.2)
    p.add_argument("--om1_min", type=float, default=0.001); p.add_argument("--om1_max", type=float, default=0.30)
    p.add_argument("--om2_min", type=float, default=0.0005); p.add_argument("--om2_max", type=float, default=0.08)
    p.add_argument("--al0_min", type=float, default=0.95); p.add_argument("--al0_max", type=float, default=0.999)
    p.add_argument("--al1_min", type=float, default=0.97); p.add_argument("--al1_max", type=float, default=0.9995)
    p.add_argument("--al2_min", type=float, default=0.98); p.add_argument("--al2_max", type=float, default=0.9999)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ensemble_w0", type=float, default=0.2)
    p.add_argument("--ensemble_w1", type=float, default=0.4)
    p.add_argument("--ensemble_w2", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--theta_init", type=float, default=0.5)
    p.add_argument("--target_rate", type=float, default=0.1)
    p.add_argument("--theta_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--no_sample_binary", action="store_true")
    p.add_argument("--csv", type=str, default="results/smnist_3stage_topdown.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0: torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = 10
    M0 = M1 = M2 = classes * args.m_per_class
    class_index = torch.repeat_interleave(torch.arange(classes), args.m_per_class)

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    print(f"train {tuple(xtr.shape)}, test {tuple(xte.shape)}", flush=True)
    print(f"M_per_class={args.m_per_class}, K_fan={args.k_fanin}, K_rec={args.rec_k}, "
          f"K_top={args.top_k}, kappa_reset={args.kappa_reset}", flush=True)

    cfg0 = OscillatorConfig(n_neurons=M0, n_input_channels=1,
                             omega_min=args.om0_min, omega_max=args.om0_max,
                             alpha_min=args.al0_min, alpha_max=args.al0_max,
                             input_init=args.input_init)
    cfg1 = OscillatorConfig(n_neurons=M1, n_input_channels=args.k_fanin,
                             omega_min=args.om1_min, omega_max=args.om1_max,
                             alpha_min=args.al1_min, alpha_max=args.al1_max,
                             input_init=args.input_init)
    cfg2 = OscillatorConfig(n_neurons=M2, n_input_channels=args.k_fanin,
                             omega_min=args.om2_min, omega_max=args.om2_max,
                             alpha_min=args.al2_min, alpha_max=args.al2_max,
                             input_init=args.input_init)
    p0 = init_params(cfg0, generator=gen)
    p1 = init_params(cfg1, generator=gen)
    p2 = init_params(cfg2, generator=gen)
    in_idx_0 = torch.zeros(M0, 1, dtype=torch.long)
    in_idx_1 = make_class_aligned_fanin(classes, args.m_per_class, args.m_per_class, args.k_fanin, gen)
    in_idx_2 = make_class_aligned_fanin(classes, args.m_per_class, args.m_per_class, args.k_fanin, gen)

    # Intra-stage recurrence
    rec_idx_0 = make_intra_stage_rec_idx(M0, args.rec_k, gen)
    rec_idx_1 = make_intra_stage_rec_idx(M1, args.rec_k, gen)
    rec_idx_2 = make_intra_stage_rec_idx(M2, args.rec_k, gen)
    rp0 = init_recurrent_params(M0, args.rec_k, args.rec_init, gen)
    rp1 = init_recurrent_params(M1, args.rec_k, args.rec_init, gen)
    rp2 = init_recurrent_params(M2, args.rec_k, args.rec_init, gen)

    # Top-down: stage 0 receives from stage 1's pass-1 spikes; stage 1 from stage 2's pass-1 spikes
    # top_idx_L picks K_top spikes from M_{L+1} stage-L+1 neurons for each stage-L neuron
    top_idx_0 = make_class_aligned_fanin(classes, args.m_per_class, args.m_per_class, args.top_k, gen)  # idx into M1
    top_idx_1 = make_class_aligned_fanin(classes, args.m_per_class, args.m_per_class, args.top_k, gen)  # idx into M2
    tp0 = init_recurrent_params(M0, args.top_k, args.top_init, gen)
    tp1 = init_recurrent_params(M1, args.top_k, args.top_init, gen)
    # stage 2 has no top-down (top stage)

    opt0 = Adam(list(p0.tensors()) + list(rp0.tensors()) + list(tp0.tensors()), args.lr)
    opt1 = Adam(list(p1.tensors()) + list(rp1.tensors()) + list(tp1.tensors()), args.lr)
    opt2 = Adam(list(p2.tensors()) + list(rp2.tensors()),                       args.lr)

    theta0 = torch.full((M0,), args.theta_init)
    theta1 = torch.full((M1,), args.theta_init)
    theta2 = torch.full((M2,), args.theta_init)
    rate_ema_0 = torch.full((M0,), args.target_rate)
    rate_ema_1 = torch.full((M1,), args.target_rate)
    rate_ema_2 = torch.full((M2,), args.target_rate)

    sample_binary = not args.no_sample_binary

    def fwd_pass1(xb):
        # Bottom-up pass, no traces, just to get spike trains for top-down feedback in pass 2.
        xb_3d = xb.unsqueeze(-1)
        out0 = forward_with_eligibility_sparse_spiking(
            xb_3d, in_idx_0, p0, cfg0, args.tail0,
            threshold=theta0, beta=args.beta,
            accumulate_traces=False, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_0, rec_params=rp0,
            top_seq=None, top_idx=None, top_params=None)
        s0 = out0["spike_seq"]
        out1 = forward_with_eligibility_sparse_spiking(
            s0, in_idx_1, p1, cfg1, args.tail1,
            threshold=theta1, beta=args.beta,
            accumulate_traces=False, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_1, rec_params=rp1,
            top_seq=None, top_idx=None, top_params=None)
        s1 = out1["spike_seq"]
        out2 = forward_with_eligibility_sparse_spiking(
            s1, in_idx_2, p2, cfg2, args.tail2,
            threshold=theta2, beta=args.beta,
            accumulate_traces=False, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_2, rec_params=rp2,
            top_seq=None, top_idx=None, top_params=None)
        s2 = out2["spike_seq"]
        return s1, s2  # we only need stage 1+2 spike trains for top-down to stages 0+1

    def fwd_pass2(xb, s1_p1, s2_p1, train=False):
        # Pass 2 with top-down feedback; this is the pass that gives us traces+gradient.
        xb_3d = xb.unsqueeze(-1)
        out0 = forward_with_eligibility_sparse_spiking(
            xb_3d, in_idx_0, p0, cfg0, args.tail0,
            threshold=theta0, beta=args.beta,
            accumulate_traces=train, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_0, rec_params=rp0,
            top_seq=s1_p1, top_idx=top_idx_0, top_params=tp0)
        s0 = out0["spike_seq"]
        out1 = forward_with_eligibility_sparse_spiking(
            s0, in_idx_1, p1, cfg1, args.tail1,
            threshold=theta1, beta=args.beta,
            accumulate_traces=train, save_spike_seq=True,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_1, rec_params=rp1,
            top_seq=s2_p1, top_idx=top_idx_1, top_params=tp1)
        s1 = out1["spike_seq"]
        out2 = forward_with_eligibility_sparse_spiking(
            s1, in_idx_2, p2, cfg2, args.tail2,
            threshold=theta2, beta=args.beta,
            accumulate_traces=train, save_spike_seq=False,
            sample_binary=sample_binary, rng=gen,
            kappa_reset=args.kappa_reset,
            rec_idx=rec_idx_2, rec_params=rp2,
            top_seq=None, top_idx=None, top_params=None)
        logits_0 = class_pool_logits(out0["rho"], class_index, classes, args.temperature)
        logits_1 = class_pool_logits(out1["rho"], class_index, classes, args.temperature)
        logits_2 = class_pool_logits(out2["rho"], class_index, classes, args.temperature)
        return out0, logits_0, out1, logits_1, out2, logits_2

    def fwd(xb, train=False):
        with torch.no_grad():
            s1_p1, s2_p1 = fwd_pass1(xb)
        return fwd_pass2(xb, s1_p1, s2_p1, train=train)

    def evaluate(x, y):
        a0 = a1 = a2 = ac = 0; a0_s = a1_s = a2_s = ac_s = 0; n = 0
        rates = [0.0, 0.0, 0.0]
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch]; yb = y[s : s + args.batch]
            o0, l0, o1, l1, o2, l2 = fwd(xb, train=False)
            a0 += (l0.argmax(1) == yb).sum().item()
            a1 += (l1.argmax(1) == yb).sum().item()
            a2 += (l2.argmax(1) == yb).sum().item()
            p0_ = F.softmax(l0, 1); p1_ = F.softmax(l1, 1); p2_ = F.softmax(l2, 1)
            pc = args.ensemble_w0 * p0_ + args.ensemble_w1 * p1_ + args.ensemble_w2 * p2_
            ac += (pc.argmax(1) == yb).sum().item()
            sl0 = class_pool_logits(o0["spike_rate"], class_index, classes, args.temperature)
            sl1 = class_pool_logits(o1["spike_rate"], class_index, classes, args.temperature)
            sl2 = class_pool_logits(o2["spike_rate"], class_index, classes, args.temperature)
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
        return ((a0/n, a1/n, a2/n, ac/n),
                (a0_s/n, a1_s/n, a2_s/n, ac_s/n),
                [r/n for r in rates])

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "wall",
                     "te_a0_smooth", "te_a1_smooth", "te_a2_smooth", "te_ac_smooth",
                     "te_a0_spike", "te_a1_spike", "te_a2_spike", "te_ac_spike",
                     "best_te_ac_spike", "rate0", "rate1", "rate2", "om0", "om1", "om2"])

    best_smooth = best_spike = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        (a0_te, a1_te, a2_te, ac_te), (a0_te_s, a1_te_s, a2_te_s, ac_te_s), te_rates = evaluate(xte, yte)
        if ac_te > best_smooth: best_smooth = ac_te
        if ac_te_s > best_spike: best_spike = ac_te_s
        om0_m = float(omega_of(p0, cfg0).mean().item())
        om1_m = float(omega_of(p1, cfg1).mean().item())
        om2_m = float(omega_of(p2, cfg2).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] smooth te={ac_te:.3f} (best {best_smooth:.4f}) | "
              f"BINARY-SPIKE te={ac_te_s:.3f} (best {best_spike:.4f}) | "
              f"per-stage spike te {a0_te_s:.3f}/{a1_te_s:.3f}/{a2_te_s:.3f} | "
              f"rates {te_rates[0]:.3f}/{te_rates[1]:.3f}/{te_rates[2]:.3f} | "
              f"om {om0_m:.3f}/{om1_m:.3f}/{om2_m:.4f}", flush=True)
        writer.writerow([epoch, f"{wall:.1f}",
                          f"{a0_te:.4f}", f"{a1_te:.4f}", f"{a2_te:.4f}", f"{ac_te:.4f}",
                          f"{a0_te_s:.4f}", f"{a1_te_s:.4f}", f"{a2_te_s:.4f}", f"{ac_te_s:.4f}",
                          f"{best_spike:.4f}", f"{te_rates[0]:.4f}", f"{te_rates[1]:.4f}", f"{te_rates[2]:.4f}",
                          f"{om0_m:.4f}", f"{om1_m:.4f}", f"{om2_m:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break
        if epoch == args.lr_decay_after:
            opt0.lr *= args.lr_decay_factor; opt1.lr *= args.lr_decay_factor; opt2.lr *= args.lr_decay_factor
            print(f"  LR decay #1: lr={opt0.lr}", flush=True)
        if args.lr_decay_after_2 > 0 and epoch == args.lr_decay_after_2:
            opt0.lr *= args.lr_decay_factor; opt1.lr *= args.lr_decay_factor; opt2.lr *= args.lr_decay_factor
            print(f"  LR decay #2: lr={opt0.lr}", flush=True)

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx]; yb = ytr[idx]
            out0, logits_0, out1, logits_1, out2, logits_2 = fwd(xb, train=True)
            onehot = F.one_hot(yb, classes).float()

            for (out, logits, opt, theta, rate_ema, has_top) in [
                (out0, logits_0, opt0, theta0, rate_ema_0, True),
                (out1, logits_1, opt1, theta1, rate_ema_1, True),
                (out2, logits_2, opt2, theta2, rate_ema_2, False),
            ]:
                probs = F.softmax(logits, dim=1)
                d_logit = (probs - onehot) / xb.shape[0]
                d_rho = d_logit[:, class_index] * (args.temperature / args.m_per_class)
                g = [(d_rho.unsqueeze(-1) * out["dRho_d_r"]).sum(dim=0),
                     (d_rho.unsqueeze(-1) * out["dRho_d_i"]).sum(dim=0),
                     (d_rho * out["dRho_b_r"]).sum(dim=0),
                     (d_rho * out["dRho_b_i"]).sum(dim=0),
                     (d_rho * out["dRho_omega_raw"]).sum(dim=0),
                     (d_rho * out["dRho_alpha_raw"]).sum(dim=0)]
                # intra-stage recurrent grads
                g.append((d_rho.unsqueeze(-1) * out["dRho_rec_d_r"]).sum(dim=0))
                g.append((d_rho.unsqueeze(-1) * out["dRho_rec_d_i"]).sum(dim=0))
                if has_top:
                    g.append((d_rho.unsqueeze(-1) * out["dRho_top_d_r"]).sum(dim=0))
                    g.append((d_rho.unsqueeze(-1) * out["dRho_top_d_i"]).sum(dim=0))
                opt.step(g, args.grad_clip)
                with torch.no_grad():
                    r_obs = out["rho"].mean(dim=0)
                    rate_ema.mul_(1.0 - args.ema_lr).add_(args.ema_lr * r_obs)
                    theta.add_(args.theta_lr * (rate_ema - args.target_rate))

    f_csv.close()
    print(f"\nBest test acc (smooth-rate ensemble): {best_smooth:.4f}", flush=True)
    print(f"Best test acc (BINARY-SPIKE ensemble): {best_spike:.4f}   <-- bio-faithful", flush=True)


if __name__ == "__main__":
    main()
