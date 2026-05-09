"""Spectral-geodesic 3-stage SPIKING HRN on SHD.

This ports the strict-spiking spectral-geodesic readout from SMNIST to
SHD while keeping the existing SHD architecture choices:

* stage 0 receives random sparse fan-in from the 700 cochlear channels;
* stages 1/2 receive class-aligned binary spikes from the previous stage;
* each stage keeps local spike-spectrum demodulators over its tail;
* class evidence is geodesic distance to complex spectral prototypes;
* oscillator parameters update by local forward eligibility traces only.
*
* optional fixed cochlear delay taps expand the stage-0 input so the same
  local rule can choose phase/delay-selective synapses without changing the
  recurrent math;
* optional multi-prototype class manifolds replace one class centroid by a
  small spectral atlas per class.

No BPTT, no surrogate gradient through sampled spikes, no inter-stage
backward pass. Binary spikes are the only inter-stage signal.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hrn2" / "src"))        # optimized spectral spiking kernel
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from oscillator import OscillatorConfig, init_params, omega_of
from oscillator_spiking import (
    forward_with_eligibility_sparse_spiking,
    init_recurrent_params,
)
from optim import Adam
from shd_data import load_shd, N_CLASSES


def class_pool_logits(rho, class_index, classes, temperature):
    logits = torch.zeros(rho.shape[0], classes, device=rho.device, dtype=rho.dtype)
    for c in range(classes):
        logits[:, c] = rho[:, class_index == c].mean(dim=1)
    return (logits - logits.mean(dim=1, keepdim=True)) * temperature


def make_stage0_random_fanin(P, F_in, K, gen):
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        in_idx[i] = torch.randperm(F_in, generator=gen)[:K]
    return in_idx


def make_stage0_grouped_delay_fanin(P, F_in, delay_count, K_base, gen):
    K = K_base * delay_count
    in_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        base = torch.randperm(F_in, generator=gen)[:K_base]
        chunks = [d * F_in + base for d in range(delay_count)]
        in_idx[i] = torch.cat(chunks)
    return in_idx


def parse_delay_taps(text):
    taps = sorted({int(t.strip()) for t in text.split(",") if t.strip()})
    if not taps:
        raise ValueError("--delay_taps must contain at least one integer")
    if taps[0] < 0:
        raise ValueError("--delay_taps must be nonnegative")
    return taps


def apply_delay_bank(x, delay_taps):
    if len(delay_taps) == 1 and delay_taps[0] == 0:
        return x
    parts = []
    T = x.shape[1]
    for delay in delay_taps:
        if delay == 0:
            parts.append(x)
            continue
        shifted = torch.zeros_like(x)
        if delay < T:
            shifted[:, delay:, :] = x[:, :T - delay, :]
        parts.append(shifted)
    return torch.cat(parts, dim=2).contiguous()


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


def make_recurrent_fanin(classes, m_per_class, K, same_frac, gen):
    P = classes * m_per_class
    class_index = torch.repeat_interleave(torch.arange(classes), m_per_class)
    same_k = int(round(K * same_frac))
    same_k = max(0, min(K, same_k))
    other_k = K - same_k
    rec_idx = torch.zeros(P, K, dtype=torch.long)
    for i in range(P):
        c = int(class_index[i].item())
        same = (class_index == c).nonzero(as_tuple=True)[0]
        same = same[same != i]
        other = (class_index != c).nonzero(as_tuple=True)[0]
        chunks = []
        if same_k > 0:
            if same.numel() >= same_k:
                chunks.append(same[torch.randperm(same.numel(), generator=gen)[:same_k]])
            else:
                chunks.append(same[torch.randint(same.numel(), (same_k,), generator=gen)])
        if other_k > 0:
            chunks.append(other[torch.randperm(other.numel(), generator=gen)[:other_k]])
        rec_idx[i] = torch.cat(chunks)
    return rec_idx


def project_recurrent_params_(rec_params, max_norm):
    if rec_params is None or max_norm <= 0:
        return
    with torch.no_grad():
        row_norm = (rec_params.d_r.square() + rec_params.d_i.square()).sum(dim=1).sqrt()
        scale = (max_norm / row_norm.clamp_min(1e-6)).clamp(max=1.0)
        rec_params.d_r.mul_(scale.unsqueeze(1))
        rec_params.d_i.mul_(scale.unsqueeze(1))


def recurrent_mean_norm(rec_params):
    if rec_params is None:
        return 0.0
    with torch.no_grad():
        return float((rec_params.d_r.square() + rec_params.d_i.square()).sum(dim=1).sqrt().mean().item())


def make_spectral_freqs(omega_min, omega_max, tail, q):
    if q <= 1:
        return torch.zeros(1)
    lo = max(float(omega_min), 2.0 * math.pi / max(float(tail), 1.0))
    hi = min(float(omega_max), 0.80)
    if hi < lo:
        hi = lo
    freqs = torch.zeros(q)
    freqs[1:] = torch.exp(torch.linspace(math.log(lo), math.log(hi), q - 1))
    return freqs


def normalize_complex_batch(re, im, eps=1e-6):
    norm = (re.square().sum(dim=(1, 2), keepdim=True) +
            im.square().sum(dim=(1, 2), keepdim=True)).sqrt().clamp_min(eps)
    return re / norm, im / norm


def normalize_complex_proto_(proto_re, proto_im, eps=1e-6):
    if proto_re.dim() == 3:
        sum_dims = (1, 2)
    elif proto_re.dim() == 4:
        sum_dims = (2, 3)
    else:
        raise ValueError("prototype tensors must be rank 3 or 4")
    norm = (proto_re.square().sum(dim=sum_dims, keepdim=True) +
            proto_im.square().sum(dim=sum_dims, keepdim=True)).sqrt().clamp_min(eps)
    proto_re.div_(norm)
    proto_im.div_(norm)


def init_prototypes(classes, class_index, q, prototypes_per_class, gen):
    P = class_index.numel()
    if prototypes_per_class == 1:
        proto_re = torch.zeros(classes, P, q)
        proto_im = torch.zeros_like(proto_re)
        for c in range(classes):
            proto_re[c, class_index == c, 0] = 1.0
        normalize_complex_proto_(proto_re, proto_im)
        return proto_re, proto_im
    proto_re = 0.01 * torch.randn(classes, prototypes_per_class, P, q, generator=gen)
    proto_im = torch.zeros_like(proto_re)
    for c in range(classes):
        proto_re[c, :, class_index == c, 0] += 1.0
    normalize_complex_proto_(proto_re, proto_im)
    return proto_re, proto_im


def geodesic_logits_from_spec(spec_re, spec_im, proto_re, proto_im, tau, temperature, proto_pool):
    h_re, h_im = normalize_complex_batch(spec_re, spec_im)
    m_re = proto_re.to(device=spec_re.device, dtype=spec_re.dtype)
    m_im = proto_im.to(device=spec_re.device, dtype=spec_re.dtype)
    if m_re.dim() == 3:
        m_re = m_re.unsqueeze(1)
        m_im = m_im.unsqueeze(1)
    inner_re = torch.einsum("bpq,crpq->bcr", h_re, m_re) + torch.einsum("bpq,crpq->bcr", h_im, m_im)
    inner_im = torch.einsum("bpq,crpq->bcr", h_im, m_re) - torch.einsum("bpq,crpq->bcr", h_re, m_im)
    sim = (inner_re.square() + inner_im.square()).sqrt().clamp(0.0, 1.0 - 1e-5)
    proto_logits = (-torch.acos(sim).square() / tau) * temperature
    if proto_pool == "max":
        return proto_logits.max(dim=2).values
    if proto_pool == "lse":
        return torch.logsumexp(proto_logits, dim=2) - math.log(proto_logits.shape[2])
    raise ValueError(f"unknown proto_pool: {proto_pool}")


def stage_logits_from_spec(spec_re, spec_im, proto_re, proto_im, class_index,
                           classes, tau, temperature, rate_aux_weight, proto_pool):
    logits = geodesic_logits_from_spec(
        spec_re, spec_im, proto_re, proto_im, tau, temperature, proto_pool)
    if rate_aux_weight != 0.0:
        logits = logits + rate_aux_weight * class_pool_logits(
            spec_re[:, :, 0], class_index, classes, temperature)
    return logits


def readout_credit(out, proto_re, proto_im, yb, class_index, classes, tau,
                   temperature, rate_aux_weight, target_rate, rate_reg_weight, proto_pool):
    spec_re = out["spec_re"].detach().requires_grad_(True)
    spec_im = out["spec_im"].detach().requires_grad_(True)
    logits = stage_logits_from_spec(
        spec_re, spec_im, proto_re, proto_im, class_index,
        classes, tau, temperature, rate_aux_weight, proto_pool)
    loss = F.cross_entropy(logits, yb)
    if rate_reg_weight != 0.0:
        loss = loss + rate_reg_weight * (spec_re[:, :, 0] - target_rate).square().mean()
    d_re, d_im = torch.autograd.grad(loss, (spec_re, spec_im))
    return logits.detach(), d_re.detach(), d_im.detach()


def spectral_param_grads(out, d_re, d_im):
    d_re_k = d_re.unsqueeze(2)
    d_im_k = d_im.unsqueeze(2)
    grads = [
        (d_re_k * out["dSpec_d_r_re"] + d_im_k * out["dSpec_d_r_im"]).sum(dim=(0, 3)),
        (d_re_k * out["dSpec_d_i_re"] + d_im_k * out["dSpec_d_i_im"]).sum(dim=(0, 3)),
        (d_re * out["dSpec_b_r_re"] + d_im * out["dSpec_b_r_im"]).sum(dim=(0, 2)),
        (d_re * out["dSpec_b_i_re"] + d_im * out["dSpec_b_i_im"]).sum(dim=(0, 2)),
        (d_re * out["dSpec_omega_raw_re"] + d_im * out["dSpec_omega_raw_im"]).sum(dim=(0, 2)),
        (d_re * out["dSpec_alpha_raw_re"] + d_im * out["dSpec_alpha_raw_im"]).sum(dim=(0, 2)),
    ]
    if "dSpec_rec_d_r_re" in out:
        grads.extend([
            (d_re_k * out["dSpec_rec_d_r_re"] + d_im_k * out["dSpec_rec_d_r_im"]).sum(dim=(0, 3)),
            (d_re_k * out["dSpec_rec_d_i_re"] + d_im_k * out["dSpec_rec_d_i_im"]).sum(dim=(0, 3)),
        ])
    return grads


def spectral_sensitivities(out):
    sens = [
        (out["dSpec_d_r_re"].square() + out["dSpec_d_r_im"].square()).mean(dim=(0, 3)),
        (out["dSpec_d_i_re"].square() + out["dSpec_d_i_im"].square()).mean(dim=(0, 3)),
        (out["dSpec_b_r_re"].square() + out["dSpec_b_r_im"].square()).mean(dim=(0, 2)),
        (out["dSpec_b_i_re"].square() + out["dSpec_b_i_im"].square()).mean(dim=(0, 2)),
        (out["dSpec_omega_raw_re"].square() + out["dSpec_omega_raw_im"].square()).mean(dim=(0, 2)),
        (out["dSpec_alpha_raw_re"].square() + out["dSpec_alpha_raw_im"].square()).mean(dim=(0, 2)),
    ]
    if "dSpec_rec_d_r_re" in out:
        sens.extend([
            (out["dSpec_rec_d_r_re"].square() + out["dSpec_rec_d_r_im"].square()).mean(dim=(0, 3)),
            (out["dSpec_rec_d_i_re"].square() + out["dSpec_rec_d_i_im"].square()).mean(dim=(0, 3)),
        ])
    return sens


def fisher_precondition(grads, sens, fisher, beta, eps):
    if beta <= 0:
        return grads
    out = []
    for g, s, f in zip(grads, sens, fisher):
        f.mul_(1.0 - beta).add_(beta * s.detach())
        out.append(g / (f.sqrt() + eps))
    return out


def update_prototypes(proto_re, proto_im, spec_re, spec_im, labels, lr):
    if lr <= 0:
        return
    with torch.no_grad():
        h_re, h_im = normalize_complex_batch(spec_re.detach(), spec_im.detach())
        if proto_re.dim() == 3:
            for c in labels.unique():
                mask = labels == c
                if bool(mask.any()):
                    ci = int(c.item())
                    proto_re[ci].mul_(1.0 - lr).add_(lr * h_re[mask].mean(dim=0).cpu())
                    proto_im[ci].mul_(1.0 - lr).add_(lr * h_im[mask].mean(dim=0).cpu())
        else:
            h_re_cpu = h_re.cpu()
            h_im_cpu = h_im.cpu()
            for i in range(labels.numel()):
                ci = int(labels[i].item())
                pr = proto_re[ci]
                pi = proto_im[ci]
                inner_re = (h_re_cpu[i].unsqueeze(0) * pr + h_im_cpu[i].unsqueeze(0) * pi).sum(dim=(1, 2))
                inner_im = (h_im_cpu[i].unsqueeze(0) * pr - h_re_cpu[i].unsqueeze(0) * pi).sum(dim=(1, 2))
                best = int((inner_re.square() + inner_im.square()).argmax().item())
                proto_re[ci, best].mul_(1.0 - lr).add_(lr * h_re_cpu[i])
                proto_im[ci, best].mul_(1.0 - lr).add_(lr * h_im_cpu[i])
        normalize_complex_proto_(proto_re, proto_im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m_per_class", type=int, default=12)
    p.add_argument("--k0_fanin", type=int, default=48)
    p.add_argument("--k0_base_fanin", type=int, default=0)
    p.add_argument("--k1_fanin", type=int, default=12)
    p.add_argument("--k2_fanin", type=int, default=12)
    p.add_argument("--rec0_fanin", type=int, default=0)
    p.add_argument("--rec1_fanin", type=int, default=0)
    p.add_argument("--rec2_fanin", type=int, default=0)
    p.add_argument("--rec_same_frac", type=float, default=0.5)
    p.add_argument("--rec_init_scale", type=float, default=0.015)
    p.add_argument("--rec_init_bias", type=float, default=0.0)
    p.add_argument("--rec_norm_max", type=float, default=0.20)
    p.add_argument("--rec_grad_scale", type=float, default=0.5)
    p.add_argument("--rec_centered", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train_size", type=int, default=4000)
    p.add_argument("--test_size", type=int, default=1000)
    p.add_argument("--tail0", type=int, default=60)
    p.add_argument("--tail1", type=int, default=80)
    p.add_argument("--tail2", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.002)
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
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--theta_init", type=float, default=0.5)
    p.add_argument("--target_rate", type=float, default=0.1)
    p.add_argument("--theta_lr", type=float, default=0.05)
    p.add_argument("--ema_lr", type=float, default=0.05)
    p.add_argument("--spec_q", type=int, default=4)
    p.add_argument("--geo_tau", type=float, default=0.20)
    p.add_argument("--rate_aux_weight", type=float, default=0.20)
    p.add_argument("--rate_reg_weight", type=float, default=0.02)
    p.add_argument("--proto_lr", type=float, default=0.03)
    p.add_argument("--prototypes_per_class", type=int, default=1)
    p.add_argument("--proto_pool", choices=["lse", "max"], default="lse")
    p.add_argument("--delay_taps", type=str, default="0")
    p.add_argument("--fisher_beta", type=float, default=0.01)
    p.add_argument("--fisher_eps", type=float, default=1e-3)
    p.add_argument("--inhibit_same", type=float, default=0.05)
    p.add_argument("--inhibit_global", type=float, default=0.02)
    p.add_argument("--no_sample_binary", action="store_true")
    p.add_argument("--csv", type=str, default="results/shd_spectral_geodesic_3stage_spiking.csv")
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    delay_taps = parse_delay_taps(args.delay_taps)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    classes = N_CLASSES
    L = 3
    M_per = args.m_per_class
    M = classes * M_per
    class_index = torch.repeat_interleave(torch.arange(classes), M_per)
    sample_binary = not args.no_sample_binary
    k0_effective = args.k0_fanin
    if args.k0_base_fanin > 0:
        k0_effective = args.k0_base_fanin * len(delay_taps)

    print("Loading SHD ...", flush=True)
    xtr, ytr, xte, yte = load_shd()
    if args.train_size:
        xtr, ytr = xtr[: args.train_size], ytr[: args.train_size]
    if args.test_size:
        xte, yte = xte[: args.test_size], yte[: args.test_size]
    F_in = xtr.shape[2]
    F0_in = F_in * len(delay_taps)
    print(f"train {tuple(xtr.shape)}, test {tuple(xte.shape)}", flush=True)
    print(f"spectral-geodesic strict spiking SHD: L={L}, M/class={M_per}, "
          f"K0={k0_effective}, K1={args.k1_fanin}, K2={args.k2_fanin}, spec_q={args.spec_q}, "
          f"prototypes/class={args.prototypes_per_class}, proto_pool={args.proto_pool}, "
          f"delay_taps={delay_taps}, recK={[args.rec0_fanin, args.rec1_fanin, args.rec2_fanin]}, "
          f"rec_centered={args.rec_centered}",
          flush=True)

    cfgs = [
        OscillatorConfig(M, k0_effective, args.om0_min, args.om0_max, args.al0_min, args.al0_max,
                         input_init=args.input_init),
        OscillatorConfig(M, args.k1_fanin, args.om1_min, args.om1_max, args.al1_min, args.al1_max,
                         input_init=args.input_init),
        OscillatorConfig(M, args.k2_fanin, args.om2_min, args.om2_max, args.al2_min, args.al2_max,
                         input_init=args.input_init),
    ]
    tails = [args.tail0, args.tail1, args.tail2]
    params = [init_params(c, generator=gen) for c in cfgs]
    rec_ks = [args.rec0_fanin, args.rec1_fanin, args.rec2_fanin]
    rec_params = [
        init_recurrent_params(M, rec_ks[ell], args.rec_init_scale, gen,
                              init_bias=args.rec_init_bias) if rec_ks[ell] > 0 else None
        for ell in range(L)
    ]
    opt_tensors = [
        params[ell].tensors() + (rec_params[ell].tensors() if rec_params[ell] is not None else [])
        for ell in range(L)
    ]
    opts = [Adam(opt_tensors[ell], args.lr) for ell in range(L)]
    fisher = [[torch.zeros_like(t) for t in opt_tensors[ell]] for ell in range(L)]
    freqs = [
        make_spectral_freqs(args.om0_min, args.om0_max, args.tail0, args.spec_q),
        make_spectral_freqs(args.om1_min, args.om1_max, args.tail1, args.spec_q),
        make_spectral_freqs(args.om2_min, args.om2_max, args.tail2, args.spec_q),
    ]
    in_idxs = [
        make_stage0_grouped_delay_fanin(M, F_in, len(delay_taps), args.k0_base_fanin, gen)
        if args.k0_base_fanin > 0 else make_stage0_random_fanin(M, F0_in, args.k0_fanin, gen),
        make_class_aligned_fanin(classes, M_per, M_per, args.k1_fanin, gen),
        make_class_aligned_fanin(classes, M_per, M_per, args.k2_fanin, gen),
    ]
    rec_idxs = [
        make_recurrent_fanin(classes, M_per, rec_ks[ell], args.rec_same_frac, gen)
        if rec_ks[ell] > 0 else None
        for ell in range(L)
    ]
    thetas = [torch.full((M,), args.theta_init) for _ in range(L)]
    rate_emas = [torch.full((M,), args.target_rate) for _ in range(L)]
    proto_re = []
    proto_im = []
    for _ in range(L):
        pr, pi = init_prototypes(classes, class_index, args.spec_q,
                                 args.prototypes_per_class, gen)
        proto_re.append(pr)
        proto_im.append(pi)

    def fwd(xb, train=False):
        outs = []
        x_in = apply_delay_bank(xb, delay_taps)
        for ell in range(L):
            save_spikes = ell < L - 1
            out = forward_with_eligibility_sparse_spiking(
                x_in, in_idxs[ell], params[ell], cfgs[ell], tails[ell],
                threshold=thetas[ell], beta=args.beta,
                accumulate_traces=train, save_spike_seq=save_spikes,
                sample_binary=sample_binary, rng=gen,
                spectral_freqs=freqs[ell],
                rec_idx=rec_idxs[ell],
                rec_params=rec_params[ell],
                rec_centered=args.rec_centered,
                class_index=class_index,
                inhibit_same=args.inhibit_same,
                inhibit_global=args.inhibit_global,
                accumulate_rate_traces=False)
            outs.append(out)
            if save_spikes:
                x_in = out["spike_seq"]
        return outs

    def logits_for_stage(out, ell, use_spike):
        sr = out["spike_spec_re"] if use_spike else out["spec_re"]
        si = out["spike_spec_im"] if use_spike else out["spec_im"]
        return stage_logits_from_spec(
            sr, si, proto_re[ell], proto_im[ell], class_index, classes,
            args.geo_tau, args.temperature, args.rate_aux_weight, args.proto_pool)

    def evaluate(x, y):
        per_smooth = [0] * L
        per_spike = [0] * L
        ens_smooth = 0
        ens_spike = 0
        rates = [0.0] * L
        n = 0
        w = [0.2, 0.4, 0.4]
        for s in range(0, x.shape[0], args.batch):
            xb = x[s : s + args.batch]
            yb = y[s : s + args.batch]
            outs = fwd(xb, train=False)
            ps = []
            pbs = []
            for ell, out in enumerate(outs):
                ls = logits_for_stage(out, ell, use_spike=False)
                lb = logits_for_stage(out, ell, use_spike=True)
                per_smooth[ell] += (ls.argmax(1) == yb).sum().item()
                per_spike[ell] += (lb.argmax(1) == yb).sum().item()
                ps.append(F.softmax(ls, dim=1))
                pbs.append(F.softmax(lb, dim=1))
                rates[ell] += float(out["spike_rate"].mean().item()) * xb.shape[0]
            pc = sum(w[ell] * ps[ell] for ell in range(L))
            pbc = sum(w[ell] * pbs[ell] for ell in range(L))
            ens_smooth += (pc.argmax(1) == yb).sum().item()
            ens_spike += (pbc.argmax(1) == yb).sum().item()
            n += xb.shape[0]
        return ([a / n for a in per_smooth], ens_smooth / n,
                [a / n for a in per_spike], ens_spike / n,
                [r / n for r in rates])

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.csv, "w", newline="")
    writer = csv.writer(f_csv)
    header = ["epoch", "wall"]
    header += [f"te_a{ell}_smooth" for ell in range(L)] + ["te_ac_smooth"]
    header += [f"te_a{ell}_spike" for ell in range(L)] + ["te_ac_spike", "best_te_ac_spike"]
    header += [f"rate{ell}" for ell in range(L)] + [f"om{ell}" for ell in range(L)]
    header += [f"recnorm{ell}" for ell in range(L)]
    writer.writerow(header)

    best_smooth = 0.0
    best_spike = 0.0
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        a_smooth, ac_smooth, a_spike, ac_spike, te_rates = evaluate(xte, yte)
        best_smooth = max(best_smooth, ac_smooth)
        best_spike = max(best_spike, ac_spike)
        oms = [float(omega_of(params[ell], cfgs[ell]).mean().item()) for ell in range(L)]
        rec_norms = [recurrent_mean_norm(rec_params[ell]) for ell in range(L)]
        wall = time.time() - t0
        print(f"[ep {epoch:02d} t={wall:6.1f}s] "
              f"smooth te={ac_smooth:.3f} (best {best_smooth:.4f}) | "
              f"BINARY-SPIKE spectral te={ac_spike:.3f} (best {best_spike:.4f}) | "
              f"per-stage {'/'.join(f'{a:.3f}' for a in a_spike)} | "
              f"rates {'/'.join(f'{r:.2f}' for r in te_rates)} | "
              f"om {'/'.join(f'{o:.3f}' for o in oms)} | "
              f"rec {'/'.join(f'{r:.3f}' for r in rec_norms)}",
              flush=True)
        row = [epoch, f"{wall:.1f}"]
        row += [f"{a:.4f}" for a in a_smooth] + [f"{ac_smooth:.4f}"]
        row += [f"{a:.4f}" for a in a_spike] + [f"{ac_spike:.4f}", f"{best_spike:.4f}"]
        row += [f"{r:.4f}" for r in te_rates] + [f"{o:.4f}" for o in oms]
        row += [f"{r:.4f}" for r in rec_norms]
        writer.writerow(row)
        f_csv.flush()

        if epoch == args.epochs:
            break
        if epoch == args.lr_decay_after:
            for opt in opts:
                opt.lr *= args.lr_decay_factor
            print(f"  LR decay #1: lr={opts[0].lr}", flush=True)
        if args.lr_decay_after_2 > 0 and epoch == args.lr_decay_after_2:
            for opt in opts:
                opt.lr *= args.lr_decay_factor
            print(f"  LR decay #2: lr={opts[0].lr}", flush=True)

        order = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(args.seed + epoch))
        for s in range(0, xtr.shape[0], args.batch):
            idx = order[s : s + args.batch]
            xb = xtr[idx]
            yb = ytr[idx]
            outs = fwd(xb, train=True)
            for ell, out in enumerate(outs):
                _, d_re, d_im = readout_credit(
                    out, proto_re[ell], proto_im[ell], yb, class_index,
                    classes, args.geo_tau, args.temperature, args.rate_aux_weight,
                    args.target_rate, args.rate_reg_weight, args.proto_pool)
                grads = spectral_param_grads(out, d_re, d_im)
                sens = spectral_sensitivities(out)
                if rec_params[ell] is not None and args.rec_grad_scale != 1.0:
                    grads[-2] = args.rec_grad_scale * grads[-2]
                    grads[-1] = args.rec_grad_scale * grads[-1]
                grads = fisher_precondition(grads, sens, fisher[ell], args.fisher_beta, args.fisher_eps)
                opts[ell].step(grads, args.grad_clip)
                project_recurrent_params_(rec_params[ell], args.rec_norm_max)
                update_prototypes(proto_re[ell], proto_im[ell],
                                  out["spec_re"], out["spec_im"], yb, args.proto_lr)
                with torch.no_grad():
                    r_obs = out["rho"].mean(dim=0)
                    rate_emas[ell].mul_(1.0 - args.ema_lr).add_(args.ema_lr * r_obs)
                    thetas[ell].add_(args.theta_lr * (rate_emas[ell] - args.target_rate))

    f_csv.close()
    print(f"\nBest test acc (smooth spectral ensemble): {best_smooth:.4f}", flush=True)
    print(f"Best test acc (BINARY-SPIKE spectral ensemble): {best_spike:.4f}   <-- bio-faithful", flush=True)


if __name__ == "__main__":
    main()
