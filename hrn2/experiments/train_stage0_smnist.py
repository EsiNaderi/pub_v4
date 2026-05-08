"""Stage-0-only port of pub_v3's resonant_self_organizing_layer to SMNIST.

This is the *sanity port*: a single bank of complex damped-rotation
oscillators with adaptive-mean WTA over pools, per-neuron Hebbian
label tags, homeostasis on θ, eligibility-trace local credit on
{d, b, ω, α}.

If we cannot match pub_v3-style behaviour at SMNIST scale here, the
hierarchy is unlikely to help. So this run is a hard prerequisite.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))           # hrn2/src takes priority

import torch

from oscillator import OscillatorConfig, init_params, omega_of, alpha_of, forward_with_eligibility
from optim import Adam
from local_rules import (
    adaptive_mean_competition, global_softmax_competition,
    label_probs, label_hebbian_step, usage_homeostasis,
    credit_for_self_organising_pool,
)
from smnist_data import load_smnist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pools", type=int, default=1)
    p.add_argument("--m_per_pool", type=int, default=256)
    p.add_argument("--competition", choices=["adaptive_mean", "softmax"], default="softmax")
    p.add_argument("--wta_beta", type=float, default=3.0)
    p.add_argument("--wta_beta_start", type=float, default=0.05)
    p.add_argument("--wta_beta_warmup", type=float, default=4.0)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--center_input", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--time_steps", type=int, default=784)
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--label_lr", type=float, default=0.02)
    p.add_argument("--label_prior", type=float, default=2.0)
    p.add_argument("--tag_power", type=float, default=1.0)
    p.add_argument("--target_usage", type=float, default=0.0625)
    p.add_argument("--homeo_lr", type=float, default=0.1)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--theta_init", type=float, default=1.0)
    p.add_argument("--omega_min", type=float, default=0.005)
    p.add_argument("--omega_max", type=float, default=1.0)
    p.add_argument("--alpha_min", type=float, default=0.90)
    p.add_argument("--alpha_max", type=float, default=0.999)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260507)
    p.add_argument("--csv", type=str, default="results/stage0_smoke.csv")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    print("Loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size:
        xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:
        xte, yte = xte[: args.test_size], yte[: args.test_size]
    if args.center_input:
        m = xtr.mean()
        xtr = xtr - m
        xte = xte - m
        print(f"  centered input: subtracted mean {m.item():.4f}", flush=True)
    print(f"train {xtr.shape}, test {xte.shape}", flush=True)

    K = args.n_pools
    M = args.m_per_pool
    P = K * M
    classes = 10

    def current_beta(epoch):
        if args.wta_beta_warmup <= 0:
            return args.wta_beta
        frac = min(1.0, epoch / args.wta_beta_warmup)
        return args.wta_beta_start + frac * (args.wta_beta - args.wta_beta_start)
    cfg = OscillatorConfig(
        n_neurons=P, n_input_channels=1,
        omega_min=args.omega_min, omega_max=args.omega_max,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        input_init=args.input_init,
        omega_init_scale=1.5, alpha_init_scale=0.25,
    )
    params = init_params(cfg, generator=gen)
    label_mass = torch.full((P, classes), args.label_prior / classes)
    if args.competition == "softmax" and K == 1:
        theta = torch.full((P,), args.theta_init)
        usage_ema = torch.full((P,), 1.0 / P)
    else:
        theta = torch.full((K, M), args.theta_init)
        usage_ema = torch.full((K, M), 1.0 / M)
    opt = Adam(params.tensors(), args.lr)

    def forward_eval(xb, yb, train=False, beta=None):
        # xb: (B, T, 1)
        out = forward_with_eligibility(xb, params, cfg, args.tail, accumulate_traces=train)
        E = out["E"]                                    # (B, P)
        if args.competition == "adaptive_mean":
            E_pool = E.view(-1, K, M)
            resp_pool = adaptive_mean_competition(E_pool, theta)
            resp_flat = resp_pool.view(-1, P)
        else:
            assert K == 1, "softmax mode currently expects K=1 (single global pool)"
            resp_flat = global_softmax_competition(E, theta, beta or args.wta_beta, args.top_k)
            resp_pool = resp_flat.view(-1, K, M)
        # label-mass head
        q = label_probs(label_mass, args.label_prior, classes)
        probs = resp_flat @ q                           # (B, C)
        py = probs[torch.arange(probs.shape[0]), yb].clamp_min(1e-12)
        loss = -torch.log(py).mean()
        pred = probs.argmax(dim=1)
        acc = (pred == yb).float().mean()
        return loss.item(), acc.item(), out, resp_flat, resp_pool

    def evaluate(x, y, beta=None):
        nb = (x.shape[0] + args.batch - 1) // args.batch
        ls = 0.0; ac = 0.0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            l, a, _, _, _ = forward_eval(xb, yb, train=False, beta=beta)
            bs = xb.shape[0]
            ls += l * bs; ac += a * bs; n += bs
        return ls / n, ac / n

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                     "best_test", "omega_mean", "alpha_mean", "theta_mean", "usage_max"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        beta_now = current_beta(epoch)
        loss_tr, acc_tr = evaluate(xtr, ytr, beta=beta_now)
        loss_te, acc_te = evaluate(xte, yte, beta=beta_now)
        if acc_te > best: best = acc_te
        om_mean = float(omega_of(params, cfg).mean().item())
        al_mean = float(alpha_of(params, cfg).mean().item())
        th_mean = float(theta.mean().item())
        usage_max = float(usage_ema.max().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] train={acc_tr:.4f} test={acc_te:.4f} best={best:.4f} "
              f"omega_mean={om_mean:.4f} alpha_mean={al_mean:.4f} theta_mean={th_mean:.3f} usage_max={usage_max:.3f}",
              flush=True)
        writer.writerow([epoch, f"{loss_tr:.4f}", f"{acc_tr:.4f}", f"{loss_te:.4f}", f"{acc_te:.4f}",
                          f"{best:.4f}", f"{om_mean:.4f}", f"{al_mean:.4f}", f"{th_mean:.3f}",
                          f"{usage_max:.3f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            _, _, out, resp_flat, resp_pool = forward_eval(xb, yb, train=True, beta=beta_now)
            # per-neuron credit signal δ_i (B, P) for the energy-amplitude space
            delta_amp = credit_for_self_organising_pool(
                resp_flat, label_mass, yb, classes, args.label_prior, credit_gain=1.0
            )
            # accumulate over batch: gradient = sum_b delta_amp[b, i] * dE_p[b, i, ...]
            g_d_r = (delta_amp.unsqueeze(-1) * out["dE_d_r"]).sum(dim=0)
            g_d_i = (delta_amp.unsqueeze(-1) * out["dE_d_i"]).sum(dim=0)
            g_b_r = (delta_amp * out["dE_b_r"]).sum(dim=0)
            g_b_i = (delta_amp * out["dE_b_i"]).sum(dim=0)
            g_om = (delta_amp * out["dE_omega_raw"]).sum(dim=0)
            g_al = (delta_amp * out["dE_alpha_raw"]).sum(dim=0)
            opt.step([g_d_r, g_d_i, g_b_r, g_b_i, g_om, g_al], args.grad_clip)
            label_hebbian_step(label_mass, resp_flat, yb, classes,
                               args.label_lr, decay=0.0, tag_power=args.tag_power)
            if args.competition == "softmax" and K == 1:
                # one-pool homeostasis
                with torch.no_grad():
                    usage = resp_flat.mean(dim=0)         # (P,)
                    usage_ema.mul_(1.0 - args.ema_lr).add_(args.ema_lr * usage)
                    theta.add_(args.homeo_lr * (usage - args.target_usage))
                    theta.clamp_(-2.0, 5.0)
            else:
                usage_homeostasis(theta, usage_ema, resp_pool,
                                  args.target_usage, args.homeo_lr, args.ema_lr)
        # log step rate
        wall = time.time() - t0
        steps_per_epoch = (xtr.shape[0] + args.batch - 1) // args.batch
        print(f"  ep {epoch:02d}: epoch wall {wall:6.1f}s, ~{wall / max(1, (epoch + 1) * steps_per_epoch):.3f}s/step",
              flush=True)

    f_csv.close()
    print(f"\nBest test acc: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
