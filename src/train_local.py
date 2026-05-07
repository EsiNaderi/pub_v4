"""Local-rule training for the HRN.

Three-factor / e-prop style. Forward-only. No autograd through time.

Per layer ℓ, after the forward pass we have:
    s_seq[ℓ]    spike sequence (B, T, N)
    u_q, v_q    pre-spike state (B, T, N)
    surrog      H'(q^2 - theta) (B, T, N)
    pre[ℓ]      input to layer (B, T, M)

We compute:
    L_post[L]   top-local CE gradient on output pool rates (only in tail window)
    L_post[ℓ]   feedback alignment from L_post[ℓ+1] (random fixed B matrices)

Eligibility traces (single decay alpha):
    pre_trace[ℓ] = exp_smooth(pre[ℓ], alpha)
    rec_trace[ℓ] = exp_smooth(s_prev[ℓ], alpha)

Gradient shapes:
    dD_re = sum_{b,t} (L_post * 2 * eta * u_q)[b,t,n] * pre_trace[b,t,m]
    dD_im = sum_{b,t} (L_post * 2 * eta * v_q)[b,t,n] * pre_trace[b,t,m]
    dW_re = same with rec_trace
    dW_im = same with rec_trace
    domega = phase coherence rule
    dtheta = homeostatic (rate - target) — only if learn_dyn_params

We use Adam-like updates per parameter with running first/second moment.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from smnist_data import SMNISTBatcher, load_smnist
from hrn import HierarchicalResonantNet, make_default_config
from train_bptt import build_net


@dataclass
class LocalCfg:
    lr_D: float = 5e-3
    lr_W: float = 5e-3
    lr_omega: float = 1e-4
    eligibility_alpha: float = 0.95
    homeostasis_target: float = 0.10
    homeostasis_coef: float = 1.0
    fb_align_scale: float = 1.0
    use_feedback_alignment: bool = True
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    train_omega: bool = False
    surr_slope: float = 2.5


def _surrog(q_sq, theta, slope):
    x = q_sq - theta
    return 1.0 / (1.0 + slope * x.abs()).pow(2) * (slope / 2.0)


def _exp_trace(seq: torch.Tensor, alpha: float) -> torch.Tensor:
    """Bounded exponential trace: tr = alpha * tr + (1-alpha) * seq.

    Steady-state under constant input is bounded by max(seq).
    """

    B, T, M = seq.shape
    one_minus = 1.0 - alpha
    out = torch.empty_like(seq)
    tr = torch.zeros(B, M, device=seq.device, dtype=seq.dtype)
    for t in range(T):
        tr = alpha * tr + one_minus * seq[:, t]
        out[:, t] = tr
    return out


@torch.no_grad()
def hrn_forward_logged(net: HierarchicalResonantNet, x: torch.Tensor) -> dict:
    """Forward through the HRN, logging per-layer state. NO autograd."""

    B, T, _ = x.shape
    sig = x
    layers_log: List[dict] = []
    for layer in net.layers:
        cfg_l = layer.cfg
        K, P = cfg_l.n_pools, cfg_l.pool_size
        N = layer.N
        device, dtype = sig.device, sig.dtype
        u = torch.zeros(B, N, device=device, dtype=dtype)
        v = torch.zeros(B, N, device=device, dtype=dtype)
        s_prev = torch.zeros(B, N, device=device, dtype=dtype)
        a = torch.zeros(B, N, device=device, dtype=dtype)

        Dr = layer.D_re.t(); Di = layer.D_im.t()
        eta = layer.eta.view(1, 1, -1)
        xr = (sig @ Dr) * eta
        xi = (sig @ Di) * eta

        cos_om = torch.cos(layer.omega); sin_om = torch.sin(layer.omega)
        gamma = layer.gamma; beta = layer.beta
        lam = layer.lambda_leak.clamp(min=0.0, max=0.99)
        one_minus_lam = (1.0 - lam).view(1, -1)
        kappa = layer.kappa; theta = layer.theta

        u_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        v_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        q_sq_seq = torch.empty(B, T, N, device=device, dtype=dtype)
        s_seq = torch.empty(B, T, N, device=device, dtype=dtype)

        block = cfg_l.block_diag
        alif_decay = math.exp(-1.0 / max(cfg_l.alif_tau, 1.0)) if cfg_l.use_alif else 0.0

        for t in range(T):
            u_rot = cos_om * u - sin_om * v
            v_rot = sin_om * u + cos_om * v
            amp_sq = (u * u + v * v).clamp(max=10.0)
            sl = beta * (1.0 - amp_sq) - gamma
            u_sl = sl * u; v_sl = sl * v
            if cfg_l.use_recurrence:
                if block:
                    sp = s_prev.view(B, K, P)
                    rec_r = torch.einsum("bkp,kqp->bkq", sp, layer.W_re).reshape(B, N)
                    rec_i = torch.einsum("bkp,kqp->bkq", sp, layer.W_im).reshape(B, N)
                else:
                    rec_r = s_prev @ layer.W_re.t()
                    rec_i = s_prev @ layer.W_im.t()
            else:
                rec_r = torch.zeros_like(u); rec_i = torch.zeros_like(u)
            u_q = one_minus_lam * (u_rot + u_sl + rec_r + xr[:, t])
            v_q = one_minus_lam * (v_rot + v_sl + rec_i + xi[:, t])
            q_sq = u_q * u_q + v_q * v_q
            arg = q_sq - theta
            if cfg_l.use_alif:
                arg = arg - a
            s = (arg > 0).float()
            scale_sp = 1.0 - kappa * s
            u = scale_sp * u_q; v = scale_sp * v_q
            u = torch.clamp(u, -3.0, 3.0); v = torch.clamp(v, -3.0, 3.0)
            if cfg_l.use_alif:
                a = alif_decay * a + cfg_l.alif_alpha * s
            s_prev = s

            u_seq[:, t] = u_q                                       # store pre-spike q-state
            v_seq[:, t] = v_q
            q_sq_seq[:, t] = q_sq
            s_seq[:, t] = s

        layers_log.append({
            "u_seq": u_seq, "v_seq": v_seq, "q_sq_seq": q_sq_seq,
            "s_seq": s_seq, "pre_seq": sig, "K": K, "P": P, "N": N,
        })
        sig = s_seq

    out_seq = layers_log[-1]["s_seq"]
    out_seq_p = out_seq.view(B, T, net.cfg.n_classes, net.cfg.out_pool_size)
    tail = max(1, int(round(T * net.cfg.tail_fraction)))
    pool_rate = out_seq_p[:, T - tail :].mean(dim=(1, 3))
    if net.cfg.center_logits:
        pool_rate_centered = pool_rate - pool_rate.mean(dim=1, keepdim=True)
    else:
        pool_rate_centered = pool_rate
    pool_logits = pool_rate_centered * net.cfg.readout_temperature
    if net.pool_bias is not None:
        pool_logits = pool_logits + net.pool_bias
    if net.aux_head is not None:
        feats = out_seq[:, T - tail :].mean(dim=1)
        logits = net.aux_head(feats)
    else:
        logits = pool_logits
    return {"layers": layers_log, "logits": logits, "pool_rate": pool_rate, "pool_logits": pool_logits, "tail": tail, "T": T}


@torch.no_grad()
def compute_local_grads(net: HierarchicalResonantNet, fwd: dict, targets: torch.Tensor,
                        cfg: LocalCfg, fb_proj: List[torch.Tensor]) -> List[dict]:
    """Compute per-layer local gradients. Uses pool_logits for credit (no aux head)."""

    layers = fwd["layers"]
    pool_logits = fwd["pool_logits"]
    pool_rate = fwd["pool_rate"]
    tail = fwd["tail"]; T = fwd["T"]
    n_classes = net.cfg.n_classes
    pool_size = net.cfg.out_pool_size

    # softmax + CE gradient -> -(probs - one_hot) for 'minimize CE' i.e. positive nudge for correct
    probs = torch.softmax(pool_logits, dim=1)
    one_hot = torch.zeros_like(probs); one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    delta_class = (one_hot - probs)                                 # (B, n_classes); sums to ~0 across classes
    # spread to per-neuron over output pool, only in tail window
    out_log = layers[-1]
    surrog_out = _surrog(out_log["q_sq_seq"], net.layers[-1].theta.view(1, 1, -1), cfg.surr_slope)  # (B, T, N_out)
    L_post = [None] * len(layers)
    L_out = torch.zeros(out_log["s_seq"].shape, device=pool_rate.device, dtype=pool_rate.dtype)
    # delta_class * temp / (tail*pool_size). delta_class sums to 0 across classes
    # so pool-pool credit is balanced (some up, some down).
    delta_full = (delta_class.unsqueeze(-1).expand(-1, -1, pool_size).reshape(L_out.shape[0], -1)) * \
                 (net.cfg.readout_temperature / max(tail * pool_size, 1))
    L_out[:, T - tail:] = delta_full.unsqueeze(1) * surrog_out[:, T - tail:]
    L_post[-1] = L_out

    # feedback alignment to earlier layers (use post-spike credit before surrogate)
    for li in range(len(layers) - 2, -1, -1):
        if cfg.use_feedback_alignment:
            B_lp1 = fb_proj[li]                                      # (N_l, N_{l+1})
            L_l = L_post[li + 1] @ B_lp1.t() * cfg.fb_align_scale
        else:
            L_l = torch.zeros_like(layers[li]["q_sq_seq"])
        # mean-subtract across neurons so total credit per timestep is zero (no positive bias)
        L_l = L_l - L_l.mean(dim=2, keepdim=True)
        layer_l = net.layers[li]
        surrog_l = _surrog(layers[li]["q_sq_seq"], layer_l.theta.view(1, 1, -1), cfg.surr_slope)
        L_l = L_l * surrog_l
        # add homeostasis as tonic credit toward target rate (drives rate down if too high)
        rate = layers[li]["s_seq"].mean(dim=(0, 1))                  # (N_l,)
        L_l = L_l - cfg.homeostasis_coef * (rate - cfg.homeostasis_target).view(1, 1, -1)
        L_post[li] = L_l

    grads = []
    for li, log in enumerate(layers):
        layer = net.layers[li]
        cfg_l = layer.cfg
        K, P, N = log["K"], log["P"], log["N"]
        L_layer = L_post[li]                                          # (B, T, N)
        u_q = log["u_seq"]; v_q = log["v_seq"]
        eta = layer.eta.view(1, 1, -1)

        # eligibility traces of presynaptic activity
        pre_seq = log["pre_seq"]                                     # (B, T, M)
        s_seq = log["s_seq"]
        s_shift = torch.cat([torch.zeros_like(s_seq[:, :1]), s_seq[:, :-1]], dim=1)
        pre_trace = _exp_trace(pre_seq, cfg.eligibility_alpha)
        rec_trace = _exp_trace(s_shift, cfg.eligibility_alpha)

        Lu = L_layer * 2.0 * u_q * eta
        Lv = L_layer * 2.0 * v_q * eta
        dD_re = torch.einsum("btn,btm->nm", Lu, pre_trace)
        dD_im = torch.einsum("btn,btm->nm", Lv, pre_trace)

        if cfg_l.use_recurrence:
            if cfg_l.block_diag:
                Lu_blk = Lu.view(Lu.shape[0], Lu.shape[1], K, P)
                Lv_blk = Lv.view(Lv.shape[0], Lv.shape[1], K, P)
                rec_blk = rec_trace.view(rec_trace.shape[0], rec_trace.shape[1], K, P)
                dW_re = torch.einsum("btkp,btkq->kpq", Lu_blk, rec_blk)
                dW_im = torch.einsum("btkp,btkq->kpq", Lv_blk, rec_blk)
            else:
                dW_re = torch.einsum("btn,btm->nm", Lu, rec_trace)
                dW_im = torch.einsum("btn,btm->nm", Lv, rec_trace)
        else:
            dW_re = None; dW_im = None

        if cfg.train_omega:
            u_prev = torch.cat([torch.zeros_like(u_q[:, :1]), u_q[:, :-1]], dim=1)
            v_prev = torch.cat([torch.zeros_like(v_q[:, :1]), v_q[:, :-1]], dim=1)
            cross = u_prev * v_q - v_prev * u_q                       # phase coherence
            domega = (L_layer * cross).sum(dim=(0, 1))
        else:
            domega = None

        grads.append({
            "dD_re": dD_re, "dD_im": dD_im, "dW_re": dW_re, "dW_im": dW_im,
            "domega": domega,
        })

    # also pool_bias gradient: derivative of CE wrt pool_bias
    db = (one_hot - probs).sum(dim=0)
    return grads, db


class AdamLite:
    """Manual Adam updater (we don't use autograd)."""

    def __init__(self, params: List[torch.Tensor], lr: float, weight_decay: float = 0.0,
                 betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr; self.wd = weight_decay
        self.b1, self.b2 = betas; self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self, grads: List[torch.Tensor], grad_clip: float = 0.0):
        self.t += 1
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(self.params, grads)):
                if g is None:
                    continue
                g = g.detach()
                if self.wd > 0:
                    g = g + self.wd * p
                if grad_clip > 0:
                    n = g.norm()
                    if n > grad_clip:
                        g = g * (grad_clip / n.clamp_min(1e-12))
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g.square()
                mhat = self.m[i] / (1 - self.b1 ** self.t)
                vhat = self.v[i] / (1 - self.b2 ** self.t)
                p.add_(self.lr * mhat / (vhat.sqrt() + self.eps))


def init_fb_proj(net: HierarchicalResonantNet, scale: float = 0.1) -> List[torch.Tensor]:
    fbs = []
    for li in range(len(net.layers) - 1):
        N_l = net.layers[li].N
        N_lp1 = net.layers[li + 1].N
        fbs.append(torch.randn(N_l, N_lp1) * (scale / math.sqrt(N_lp1)))
    return fbs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="default")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--lr_D", type=float, default=5e-3)
    p.add_argument("--lr_W", type=float, default=5e-3)
    p.add_argument("--lr_bias", type=float, default=5e-3)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--target_rate", type=float, default=0.10)
    p.add_argument("--homeo_coef", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--time_budget", type=float, default=14400)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading SMNIST ...", flush=True)
    xtr, ytr, xte, yte = load_smnist()
    if args.train_size: xtr, ytr = xtr[:args.train_size], ytr[:args.train_size]
    if args.test_size: xte, yte = xte[:args.test_size], yte[:args.test_size]

    net = build_net(args.arch).to("cpu")
    print(f"net params: {net.n_params()}", flush=True)

    fb_proj = init_fb_proj(net)

    cfg = LocalCfg(lr_D=args.lr_D, lr_W=args.lr_W, eligibility_alpha=args.alpha,
                   homeostasis_target=args.target_rate, homeostasis_coef=args.homeo_coef)

    # collect parameters per layer
    D_re_list = [layer.D_re for layer in net.layers]
    D_im_list = [layer.D_im for layer in net.layers]
    W_re_list = [layer.W_re for layer in net.layers if layer.cfg.use_recurrence]
    W_im_list = [layer.W_im for layer in net.layers if layer.cfg.use_recurrence]
    omega_list = [layer.omega for layer in net.layers]

    opt_D_re = AdamLite(D_re_list, cfg.lr_D, cfg.weight_decay)
    opt_D_im = AdamLite(D_im_list, cfg.lr_D, cfg.weight_decay)
    opt_W_re = AdamLite(W_re_list, cfg.lr_W, cfg.weight_decay)
    opt_W_im = AdamLite(W_im_list, cfg.lr_W, cfg.weight_decay)
    if net.pool_bias is not None:
        opt_bias = AdamLite([net.pool_bias], args.lr_bias, 0.0)
    else:
        opt_bias = None

    train_loader = SMNISTBatcher(xtr, ytr, args.batch, "cpu", seed=args.seed)
    test_loader = SMNISTBatcher(xte, yte, args.batch * 2, "cpu", seed=args.seed + 1)

    csv_path = Path(args.csv) if args.csv else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "step", "wall", "train_loss_ema", "train_acc_ema", "test_loss", "test_acc", "rates"])
    else:
        writer = None

    t0 = time.time(); step = 0; best_test = 0.0
    train_loss_ema = None; train_acc_ema = None
    for epoch in range(args.epochs):
        for xb, yb in train_loader.shuffle_iter():
            fwd = hrn_forward_logged(net, xb)
            grads, db = compute_local_grads(net, fwd, yb, cfg, fb_proj)
            # apply
            opt_D_re.step([g["dD_re"] for g in grads], cfg.grad_clip)
            opt_D_im.step([g["dD_im"] for g in grads], cfg.grad_clip)
            if W_re_list:
                W_grads_re = [g["dW_re"] for g in grads if g["dW_re"] is not None]
                W_grads_im = [g["dW_im"] for g in grads if g["dW_im"] is not None]
                opt_W_re.step(W_grads_re, cfg.grad_clip)
                opt_W_im.step(W_grads_im, cfg.grad_clip)
            if opt_bias is not None:
                opt_bias.step([db], cfg.grad_clip)

            # metrics
            with torch.no_grad():
                pred = fwd["logits"].argmax(dim=1)
                acc = float((pred == yb).float().mean().item())
                loss = float(torch.nn.functional.cross_entropy(fwd["logits"], yb).item())
            train_loss_ema = loss if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * loss
            train_acc_ema = acc if train_acc_ema is None else 0.95 * train_acc_ema + 0.05 * acc

            if step % args.log_every == 0:
                wall = time.time() - t0
                rates = [layers["s_seq"].mean().item() for layers in fwd["layers"]]
                rates_str = "/".join(f"{r:.3f}" for r in rates)
                print(f"[ep {epoch} step {step:5d} t={wall:6.1f}s] loss={loss:.3f} (ema {train_loss_ema:.3f}) "
                      f"acc={acc:.3f} (ema {train_acc_ema:.3f}) rates={rates_str}", flush=True)
            step += 1
            if (time.time() - t0) > args.time_budget:
                break

        # eval
        n_correct = 0; n_total = 0; loss_sum = 0.0
        for xb, yb in test_loader.seq_iter():
            fwd = hrn_forward_logged(net, xb)
            pred = fwd["logits"].argmax(dim=1)
            n_correct += int((pred == yb).sum().item())
            n_total += xb.shape[0]
            loss_sum += float(torch.nn.functional.cross_entropy(fwd["logits"], yb).item()) * xb.shape[0]
        test_acc = n_correct / max(n_total, 1)
        test_loss = loss_sum / max(n_total, 1)
        if test_acc > best_test:
            best_test = test_acc
            if args.ckpt:
                Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": net.state_dict(), "test_acc": test_acc, "epoch": epoch}, args.ckpt)
        wall = time.time() - t0
        print(f"[ep {epoch} EVAL t={wall:6.1f}s] test_loss={test_loss:.4f} test_acc={test_acc:.4f} best={best_test:.4f}", flush=True)
        if writer:
            rates_str = "/".join(f"{r:.3f}" for r in rates)
            writer.writerow([epoch, step, f"{wall:.1f}", f"{train_loss_ema:.4f}", f"{train_acc_ema:.4f}",
                              f"{test_loss:.4f}", f"{test_acc:.4f}", rates_str])
            f_csv.flush()
        if (time.time() - t0) > args.time_budget:
            print(f"time budget reached", flush=True)
            break

    print(f"best test: {best_test:.4f}", flush=True)
    if writer: f_csv.close()


if __name__ == "__main__":
    main()
