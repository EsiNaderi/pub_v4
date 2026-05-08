"""DECOLLE 5-stage SPIKING HRN on SMNIST.

Same per-neuron substrate (linear damped complex oscillator + binary
spike emission with subtractive reset + intra-stage recurrence) as the
3-stage version, scaled to FIVE stages with a strict frequency
hierarchy:

    stage 0:  fast band, pixel-rate features
    stage 1:  envelope of stage 0
    stage 2:  envelope of stage 1
    stage 3:  envelope of stage 2
    stage 4:  envelope of stage 3, very slow

Each stage has its own class-pool readout and its own local CE loss
(DECOLLE). Final prediction is a weighted ensemble of all 5 stages'
softmax outputs.

The motivation: a deeper frequency hierarchy gives the network more
"echelons" to compress the 784-step SMNIST raster sequence into
class-discriminative slow-time-scale features. Spike-only
inter-neuron communication, no BPTT, no surrogate of Heaviside, only
forward eligibility traces.
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
    p.add_argument("--m_per_class", type=int, default=24, help="neurons per class pool per stage")
    p.add_argument("--k_fanin", type=int, default=12, help="inter-stage sparse fan-in")
    p.add_argument("--rec_k", type=int, default=8, help="intra-stage recurrent fan-in")
    p.add_argument("--rec_init", type=float, default=0.02)
    p.add_argument("--kappa_reset", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_decay_after", type=int, default=10)
    p.add_argument("--lr_decay_after_2", type=int, default=18)
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--theta_init", type=float, default=0.5)
    p.add_argument("--target_rate", type=float, default=0.1)
    p.add_argument("--theta_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--no_sample_binary", action="store_true")
    p.add_argument("--csv", type=str, default="results/smnist_5stage_spiking.csv")
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0: torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = 10
    L = 5
    M_per = args.m_per_class
    M = classes * M_per
    K_fan = args.k_fanin
    Krec = args.rec_k

    # Frequency hierarchy: each stage's omega range is narrower and slower than the previous.
    # Stage 0 is fast (pixel-rate); stage L-1 is very slow (whole-image envelope).
    stage_om_min = [0.005, 0.001, 0.0005, 0.0002, 0.0001]
    stage_om_max = [1.2,   0.30,  0.10,   0.04,   0.015]
    stage_al_min = [0.95,  0.97,  0.98,   0.985,  0.99]
    stage_al_max = [0.999, 0.9995, 0.9998, 0.9999, 0.99995]
    stage_tail   = [200,   300,   400,    500,    600]

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:  xte, yte = xte[: args.test_size],  yte[: args.test_size]
    print(f"train {tuple(xtr.shape)}, test {tuple(xte.shape)}", flush=True)
    print(f"L={L}, M_per_class={M_per}, M={M}, K_fan={K_fan}, Krec={Krec}, kappa_reset={args.kappa_reset}",
          flush=True)
    for ell in range(L):
        print(f"  stage {ell}: omega [{stage_om_min[ell]}, {stage_om_max[ell]}], "
              f"alpha [{stage_al_min[ell]}, {stage_al_max[ell]}], tail {stage_tail[ell]}",
              flush=True)

    class_index = torch.repeat_interleave(torch.arange(classes), M_per)

    cfgs = []
    for ell in range(L):
        Fin = 1 if ell == 0 else K_fan
        cfgs.append(OscillatorConfig(
            n_neurons=M, n_input_channels=Fin,
            omega_min=stage_om_min[ell], omega_max=stage_om_max[ell],
            alpha_min=stage_al_min[ell], alpha_max=stage_al_max[ell],
            input_init=args.input_init,
        ))

    pps = [init_params(c, generator=gen) for c in cfgs]

    in_idxs = []
    for ell in range(L):
        if ell == 0:
            in_idxs.append(torch.zeros(M, 1, dtype=torch.long))
        else:
            in_idxs.append(make_class_aligned_fanin(classes, M_per, M_per, K_fan, gen))

    rec_idxs = [make_intra_stage_rec_idx(M, Krec, gen) for _ in range(L)] if Krec > 0 else [None] * L
    rec_paramss = [init_recurrent_params(M, Krec, args.rec_init, gen) for _ in range(L)] if Krec > 0 else [None] * L

    if Krec > 0:
        opts = [Adam(list(pps[ell].tensors()) + list(rec_paramss[ell].tensors()), args.lr) for ell in range(L)]
    else:
        opts = [Adam(pps[ell].tensors(), args.lr) for ell in range(L)]

    thetas = [torch.full((M,), args.theta_init) for _ in range(L)]
    rate_emas = [torch.full((M,), args.target_rate) for _ in range(L)]

    sample_binary = not args.no_sample_binary

    def fwd(xb, train=False):
        outs = []; logits_list = []
        x_in = xb.unsqueeze(-1)                                  # (B, T, 1) for stage 0
        for ell in range(L):
            save_spikes = (ell < L - 1)
            out = forward_with_eligibility_sparse_spiking(
                x_in, in_idxs[ell], pps[ell], cfgs[ell], stage_tail[ell],
                threshold=thetas[ell], beta=args.beta,
                accumulate_traces=train, save_spike_seq=save_spikes,
                sample_binary=sample_binary, rng=gen,
                kappa_reset=args.kappa_reset,
                rec_idx=rec_idxs[ell], rec_params=rec_paramss[ell])
            outs.append(out)
            logits_list.append(class_pool_logits(out["rho"], class_index, classes, args.temperature))
            if save_spikes:
                x_in = out["spike_seq"]
        return outs, logits_list

    def evaluate(x, y):
        a_smooth = [0] * L; a_spike = [0] * L
        ac_smooth = 0; ac_spike = 0
        n = 0; rate_sum = [0.0] * L
        # equal weights across stages for the ensemble
        w = [1.0 / L] * L
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch]; yb = y[s : s + args.batch]
            outs, logits = fwd(xb, train=False)
            for ell in range(L):
                a_smooth[ell] += (logits[ell].argmax(1) == yb).sum().item()
                sl = class_pool_logits(outs[ell]["spike_rate"], class_index, classes, args.temperature)
                a_spike[ell] += (sl.argmax(1) == yb).sum().item()
                rate_sum[ell] += float(outs[ell]["spike_rate"].mean().item()) * xb.shape[0]
            # smooth ensemble
            ps = [F.softmax(l, dim=1) for l in logits]
            pc = sum(w[ell] * ps[ell] for ell in range(L))
            ac_smooth += (pc.argmax(1) == yb).sum().item()
            # spike-rate ensemble
            sps = [F.softmax(class_pool_logits(o["spike_rate"], class_index, classes, args.temperature), dim=1)
                   for o in outs]
            spc = sum(w[ell] * sps[ell] for ell in range(L))
            ac_spike += (spc.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return ([a / n for a in a_smooth], ac_smooth / n,
                [a / n for a in a_spike], ac_spike / n,
                [r / n for r in rate_sum])

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    header = ["epoch", "wall"]
    header += [f"te_a{ell}_smooth" for ell in range(L)] + ["te_ac_smooth"]
    header += [f"te_a{ell}_spike"  for ell in range(L)] + ["te_ac_spike", "best_te_ac_spike"]
    header += [f"rate{ell}" for ell in range(L)] + [f"om{ell}" for ell in range(L)]
    writer.writerow(header)

    best_smooth = best_spike = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        a_te_smooth, ac_te_smooth, a_te_spike, ac_te_spike, te_rates = evaluate(xte, yte)
        if ac_te_smooth > best_smooth: best_smooth = ac_te_smooth
        if ac_te_spike  > best_spike:  best_spike  = ac_te_spike
        oms = [float(omega_of(pps[ell], cfgs[ell]).mean().item()) for ell in range(L)]
        wall = time.time() - t0
        per_stage_str = " ".join(f"a{ell}={a_te_spike[ell]:.3f}" for ell in range(L))
        rates_str = "/".join(f"{r:.2f}" for r in te_rates)
        oms_str = "/".join(f"{o:.3f}" for o in oms)
        print(f"[ep {epoch:02d} t={wall:6.1f}s] smooth te={ac_te_smooth:.3f} (best {best_smooth:.4f}) | "
              f"BINARY-SPIKE te={ac_te_spike:.3f} (best {best_spike:.4f}) | per-stage {per_stage_str} | "
              f"rates {rates_str} | om {oms_str}",
              flush=True)
        row = [epoch, f"{wall:.1f}"]
        row += [f"{a:.4f}" for a in a_te_smooth] + [f"{ac_te_smooth:.4f}"]
        row += [f"{a:.4f}" for a in a_te_spike]  + [f"{ac_te_spike:.4f}", f"{best_spike:.4f}"]
        row += [f"{r:.4f}" for r in te_rates] + [f"{o:.4f}" for o in oms]
        writer.writerow(row); f_csv.flush()

        if epoch == args.epochs: break
        if epoch == args.lr_decay_after:
            for o in opts: o.lr *= args.lr_decay_factor
            print(f"  LR decay #1: lr={opts[0].lr}", flush=True)
        if args.lr_decay_after_2 > 0 and epoch == args.lr_decay_after_2:
            for o in opts: o.lr *= args.lr_decay_factor
            print(f"  LR decay #2: lr={opts[0].lr}", flush=True)

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx]; yb = ytr[idx]
            outs, logits = fwd(xb, train=True)
            onehot = F.one_hot(yb, classes).float()
            for ell in range(L):
                probs = F.softmax(logits[ell], dim=1)
                d_logit = (probs - onehot) / xb.shape[0]
                d_rho = d_logit[:, class_index] * (args.temperature / M_per)
                out = outs[ell]
                g = [(d_rho.unsqueeze(-1) * out["dRho_d_r"]).sum(dim=0),
                     (d_rho.unsqueeze(-1) * out["dRho_d_i"]).sum(dim=0),
                     (d_rho * out["dRho_b_r"]).sum(dim=0),
                     (d_rho * out["dRho_b_i"]).sum(dim=0),
                     (d_rho * out["dRho_omega_raw"]).sum(dim=0),
                     (d_rho * out["dRho_alpha_raw"]).sum(dim=0)]
                if Krec > 0:
                    g.append((d_rho.unsqueeze(-1) * out["dRho_rec_d_r"]).sum(dim=0))
                    g.append((d_rho.unsqueeze(-1) * out["dRho_rec_d_i"]).sum(dim=0))
                opts[ell].step(g, args.grad_clip)
                with torch.no_grad():
                    r_obs = out["rho"].mean(dim=0)
                    rate_emas[ell].mul_(1.0 - args.ema_lr).add_(args.ema_lr * r_obs)
                    thetas[ell].add_(args.theta_lr * (rate_emas[ell] - args.target_rate))

    f_csv.close()
    print(f"\nBest test acc (smooth-rate ensemble): {best_smooth:.4f}", flush=True)
    print(f"Best test acc (BINARY-SPIKE ensemble): {best_spike:.4f}   <-- bio-faithful", flush=True)


if __name__ == "__main__":
    main()
