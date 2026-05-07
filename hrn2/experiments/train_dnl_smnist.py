"""Discover → Nudge → Lock single-stage class-pool resonant net.

Inspired by pub_v3's three-phase procedure, adapted for class-pool
supervision instead of label-mass.

Phase 1 (DISCOVER): unsupervised mode discovery. The credit signal is
   purely homeostatic: each neuron's eligibility-trace gradient is
   modulated by (E_i - mean E across pool) with sign such that a
   neuron that consistently fires in a *cluster* (rather than a
   class) is reinforced. Equivalent to anti-redundancy + diversity
   pressure. ω, α, d, b all train.

Phase 2 (NUDGE): introduce class-pool supervision but at low weight
   (0.1×), still with discovery pressure. The two signals combine
   so modes that have already specialized further sharpen toward
   their class.

Phase 3 (LOCK): full class-pool supervision (1.0×), discovery turned
   off. ω/α learning rate dropped to 1/10 (lock substrate). d/b/bias
   continue training.

Phases are scheduled by epoch (--discover_until, --nudge_until).
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
    logits = torch.zeros(E.shape[0], classes, device=E.device, dtype=E.dtype)
    for c in range(classes):
        logits[:, c] = E[:, class_index == c].mean(dim=1)
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m_per_class", type=int, default=80)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--discover_until", type=int, default=4)
    p.add_argument("--nudge_until", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=2000)
    p.add_argument("--tail", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr_lock_omega_scale", type=float, default=0.1,
                   help="ω/α LR scaled by this factor in lock phase")
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--om_min", type=float, default=0.005)
    p.add_argument("--om_max", type=float, default=1.5)
    p.add_argument("--al_min", type=float, default=0.95)
    p.add_argument("--al_max", type=float, default=0.9995)
    p.add_argument("--input_init", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--discover_strength", type=float, default=0.5)
    p.add_argument("--nudge_strength", type=float, default=0.1,
                   help="class supervision weight during nudge phase (lock = 1.0)")
    p.add_argument("--csv", type=str, default="results/dnl.csv")
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

    # Two optimizers: one for d/b (always full LR), one for ω/α (locked in phase 3)
    opt_db = Adam([params.d_r, params.d_i, params.b_r, params.b_i], args.lr)
    opt_oa = Adam([params.omega_raw, params.alpha_raw], args.lr)

    def fwd(xb, train=False):
        out = forward_with_eligibility(xb, params, cfg, args.tail,
                                         accumulate_traces=train, save_amp_seq=False)
        E = out["E"]
        logits = class_pool_logits(E, class_index, classes, args.temperature)
        return out, E, logits

    def evaluate(x, y):
        ls = 0.0; ac = 0; n = 0
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch].unsqueeze(-1)
            yb = y[s : s + args.batch]
            _, _, logits = fwd(xb, train=False)
            ls += F.cross_entropy(logits, yb, reduction="sum").item()
            ac += (logits.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return ls / n, ac / n

    def phase_of(epoch):
        if epoch < args.discover_until: return "discover"
        if epoch < args.nudge_until: return "nudge"
        return "lock"

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["epoch", "phase", "wall", "tr_loss", "tr_acc", "te_loss", "te_acc",
                     "best_te_acc", "om_mean", "om_std", "al_mean"])

    best = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        phase = phase_of(epoch)
        loss_tr, acc_tr = evaluate(xtr, ytr)
        loss_te, acc_te = evaluate(xte, yte)
        if acc_te > best: best = acc_te
        om_mean = float(omega_of(params, cfg).mean().item())
        om_std = float(omega_of(params, cfg).std().item())
        al_mean = float(alpha_of(params, cfg).mean().item())
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s phase={phase}] tr={acc_tr:.4f} te={acc_te:.4f} "
              f"best={best:.4f} om={om_mean:.3f}±{om_std:.3f} al={al_mean:.4f}",
              flush=True)
        writer.writerow([epoch, phase, f"{wall:.1f}",
                          f"{loss_tr:.4f}", f"{acc_tr:.4f}",
                          f"{loss_te:.4f}", f"{acc_te:.4f}", f"{best:.4f}",
                          f"{om_mean:.4f}", f"{om_std:.4f}", f"{al_mean:.4f}"])
        f_csv.flush()
        if epoch == args.epochs: break

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx].unsqueeze(-1)
            yb = ytr[idx]
            out, E, logits = fwd(xb, train=True)

            # class-pool credit
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(yb, classes).float()
            d_logit = (probs - onehot) / xb.shape[0]
            d_E_class = d_logit[:, class_index] * (args.temperature / args.m_per_class)

            # discover credit: encourage diversity & sparsity within each class pool.
            # For each neuron in pool c, push E_i toward target (mean E in pool c).
            # This is "neighbor-relative" — neurons that are too low or too high get
            # pulled toward the pool average. But we want diversity, so we inject
            # ANTI-redundancy by computing per-neuron pull AWAY from the class-pool
            # mean (each neuron is unique). Combined with homeostatic θ, modes
            # naturally separate.
            with torch.no_grad():
                E_pool_mean = torch.zeros(E.shape[0], classes, device=E.device)
                for c in range(classes):
                    E_pool_mean[:, c] = E[:, class_index == c].mean(dim=1)
                E_target = E_pool_mean[:, class_index]
                # diversity pressure: -(E_i - E_target_class) so neurons settle near pool mean
                # but slight anti-redundancy: subtract small overlap term
                d_E_discover = -args.discover_strength * (E - E_target) / xb.shape[0]

            # combine credit by phase
            if phase == "discover":
                d_E = d_E_discover.detach()
            elif phase == "nudge":
                d_E = (d_E_discover + args.nudge_strength * d_E_class).detach()
            else:  # lock
                d_E = d_E_class

            g_d_r = (d_E.unsqueeze(-1) * out["dE_d_r"]).sum(dim=0)
            g_d_i = (d_E.unsqueeze(-1) * out["dE_d_i"]).sum(dim=0)
            g_b_r = (d_E * out["dE_b_r"]).sum(dim=0)
            g_b_i = (d_E * out["dE_b_i"]).sum(dim=0)
            g_om = (d_E * out["dE_omega_raw"]).sum(dim=0)
            g_al = (d_E * out["dE_alpha_raw"]).sum(dim=0)
            opt_db.step([g_d_r, g_d_i, g_b_r, g_b_i], args.grad_clip)

            if phase == "lock":
                # ω/α LR scaled down
                opt_oa.lr = args.lr * args.lr_lock_omega_scale
            else:
                opt_oa.lr = args.lr
            opt_oa.step([g_om, g_al], args.grad_clip)

    f_csv.close()
    print(f"\nBest test acc: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
