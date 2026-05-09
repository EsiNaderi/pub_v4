"""Strict-spiking convolutional oscillator with local eligibility traces.

Layer state per output channel and spatial site:

    z_o,y,x(t+1) = alpha_o exp(i omega_o) z_o,y,x(t)
                   + conv2d(x(t), d_o)_y,x + b_o

Spike probability:

    p_o,y,x(t) = sigmoid(beta * (|z_o,y,x(t)|^2 - theta_o))

The layer emits binary spikes. Training uses the derivative of the
Bernoulli mean p(t), contracted with a local readout credit. There is no
BPTT, no surrogate derivative through sampled spikes, and no autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ConvOscillatorConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    omega_min: float = 0.03
    omega_max: float = 1.20
    alpha_min: float = 0.85
    alpha_max: float = 0.995
    input_init: float = 0.08
    bias_init: float = 0.0
    z_clip: float = 0.0


@dataclass
class ConvOscillatorParams:
    d_r: torch.Tensor
    d_i: torch.Tensor
    b_r: torch.Tensor
    b_i: torch.Tensor
    omega_raw: torch.Tensor
    alpha_raw: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        return [self.d_r, self.d_i, self.b_r, self.b_i, self.omega_raw, self.alpha_raw]


@dataclass
class ConvOscillatorGrads:
    d_r: torch.Tensor
    d_i: torch.Tensor
    b_r: torch.Tensor
    b_i: torch.Tensor
    omega_raw: torch.Tensor
    alpha_raw: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        return [self.d_r, self.d_i, self.b_r, self.b_i, self.omega_raw, self.alpha_raw]


def init_params(cfg: ConvOscillatorConfig,
                generator: torch.Generator | None = None) -> ConvOscillatorParams:
    if generator is None:
        generator = torch.Generator().manual_seed(20260508)
    fan = cfg.in_channels * cfg.kernel_size * cfg.kernel_size
    scale = cfg.input_init / float(max(1, fan)) ** 0.5
    shape = (cfg.out_channels, cfg.in_channels, cfg.kernel_size, cfg.kernel_size)
    return ConvOscillatorParams(
        d_r=scale * torch.randn(shape, generator=generator),
        d_i=scale * torch.randn(shape, generator=generator),
        b_r=cfg.bias_init * torch.randn(cfg.out_channels, generator=generator),
        b_i=cfg.bias_init * torch.randn(cfg.out_channels, generator=generator),
        omega_raw=1.5 * torch.randn(cfg.out_channels, generator=generator),
        alpha_raw=0.25 * torch.randn(cfg.out_channels, generator=generator),
    )


def omega_of(params: ConvOscillatorParams, cfg: ConvOscillatorConfig) -> torch.Tensor:
    sig = torch.sigmoid(params.omega_raw)
    return cfg.omega_min + (cfg.omega_max - cfg.omega_min) * sig


def alpha_of(params: ConvOscillatorParams, cfg: ConvOscillatorConfig) -> torch.Tensor:
    sig = torch.sigmoid(params.alpha_raw)
    return cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * sig


def output_hw(height: int, width: int, cfg: ConvOscillatorConfig) -> tuple[int, int]:
    k = cfg.kernel_size
    h = (height + 2 * cfg.padding - k) // cfg.stride + 1
    w = (width + 2 * cfg.padding - k) // cfg.stride + 1
    return h, w


def _unfold(x_t: torch.Tensor, cfg: ConvOscillatorConfig) -> torch.Tensor:
    return F.unfold(
        x_t,
        kernel_size=cfg.kernel_size,
        padding=cfg.padding,
        stride=cfg.stride,
    )


def forward_spiking_conv(
    x_seq: torch.Tensor,
    params: ConvOscillatorParams,
    cfg: ConvOscillatorConfig,
    threshold: torch.Tensor,
    tail: int,
    beta: float = 8.0,
    sample_binary: bool = True,
    rng: torch.Generator | None = None,
    save_spike_seq: bool = False,
    kappa_reset: float = 0.0,
) -> dict:
    """Forward pass only. No gradient graph is created by the caller."""
    B, T, C, H, W = x_seq.shape
    assert C == cfg.in_channels
    O = cfg.out_channels
    Hout, Wout = output_hw(H, W, cfg)
    L = Hout * Wout
    assert threshold.shape == (O,)

    wr = params.d_r.reshape(O, -1)
    wi = params.d_i.reshape(O, -1)
    om = omega_of(params, cfg)
    al = alpha_of(params, cfg)
    rot_r = al * torch.cos(om)
    rot_i = al * torch.sin(om)

    z_r = torch.zeros(B, O, L, dtype=x_seq.dtype, device=x_seq.device)
    z_i = torch.zeros_like(z_r)
    theta = threshold.view(1, O, 1)
    rho_sum = torch.zeros_like(z_r)
    spike_sum = torch.zeros_like(z_r)
    if save_spike_seq:
        spike_seq = torch.zeros(B, T, O, Hout, Wout, dtype=x_seq.dtype, device=x_seq.device)
    else:
        spike_seq = None

    tail_start = max(0, T - tail)
    tail_len = max(1, T - tail_start)

    for t in range(T):
        patches = _unfold(x_seq[:, t], cfg)
        drive_r = torch.einsum("bkl,ok->bol", patches, wr) + params.b_r.view(1, O, 1)
        drive_i = torch.einsum("bkl,ok->bol", patches, wi) + params.b_i.view(1, O, 1)
        zr_next = rot_r.view(1, O, 1) * z_r - rot_i.view(1, O, 1) * z_i + drive_r
        zi_next = rot_i.view(1, O, 1) * z_r + rot_r.view(1, O, 1) * z_i + drive_i
        if cfg.z_clip > 0:
            zr_next = zr_next.clamp(-cfg.z_clip, cfg.z_clip)
            zi_next = zi_next.clamp(-cfg.z_clip, cfg.z_clip)
        z_r, z_i = zr_next, zi_next

        p_t = torch.sigmoid(beta * (z_r.square() + z_i.square() - theta))
        if sample_binary:
            if rng is None:
                s_t = torch.bernoulli(p_t)
            else:
                noise = torch.empty_like(p_t).uniform_(0.0, 1.0, generator=rng)
                s_t = (noise < p_t).to(p_t.dtype)
        else:
            s_t = p_t
        if save_spike_seq:
            spike_seq[:, t] = s_t.view(B, O, Hout, Wout)
        if t >= tail_start:
            rho_sum.add_(p_t)
            spike_sum.add_(s_t)
        if kappa_reset > 0:
            keep = 1.0 - kappa_reset * s_t
            z_r.mul_(keep)
            z_i.mul_(keep)

    out = {
        "rho": (rho_sum / tail_len).view(B, O, Hout, Wout),
        "spike_rate": (spike_sum / tail_len).view(B, O, Hout, Wout),
        "z_r": z_r.view(B, O, Hout, Wout),
        "z_i": z_i.view(B, O, Hout, Wout),
    }
    if spike_seq is not None:
        out["spike_seq"] = spike_seq
    return out


def contracted_local_grads_conv(
    x_seq: torch.Tensor,
    params: ConvOscillatorParams,
    cfg: ConvOscillatorConfig,
    threshold: torch.Tensor,
    tail: int,
    credit_rho: torch.Tensor,
    beta: float = 8.0,
    kappa_reset: float = 0.0,
) -> ConvOscillatorGrads:
    """Contract local readout credit with forward eligibility traces.

    credit_rho must be dL/d rho from a local readout on this same layer.
    No downstream weight matrix is transported into this function.
    """
    B, T, C, H, W = x_seq.shape
    O = cfg.out_channels
    Hout, Wout = output_hw(H, W, cfg)
    L = Hout * Wout
    K = C * cfg.kernel_size * cfg.kernel_size
    assert credit_rho.shape == (B, O, Hout, Wout)
    credit = credit_rho.reshape(B, O, L)

    wr = params.d_r.reshape(O, K)
    wi = params.d_i.reshape(O, K)
    om = omega_of(params, cfg)
    al = alpha_of(params, cfg)
    cos_w = torch.cos(om)
    sin_w = torch.sin(om)
    rot_r = al * cos_w
    rot_i = al * sin_w
    sig_o = torch.sigmoid(params.omega_raw)
    sig_a = torch.sigmoid(params.alpha_raw)
    d_omega_d_raw = (cfg.omega_max - cfg.omega_min) * sig_o * (1.0 - sig_o)
    d_alpha_d_raw = (cfg.alpha_max - cfg.alpha_min) * sig_a * (1.0 - sig_a)

    z_r = torch.zeros(B, O, L, dtype=x_seq.dtype, device=x_seq.device)
    z_i = torch.zeros_like(z_r)
    ewr_r = torch.zeros(B, O, K, L, dtype=x_seq.dtype, device=x_seq.device)
    ewr_i = torch.zeros_like(ewr_r)
    ewi_r = torch.zeros_like(ewr_r)
    ewi_i = torch.zeros_like(ewr_r)
    ebr_r = torch.zeros(B, O, L, dtype=x_seq.dtype, device=x_seq.device)
    ebr_i = torch.zeros_like(ebr_r)
    ebi_r = torch.zeros_like(ebr_r)
    ebi_i = torch.zeros_like(ebr_r)
    eom_r = torch.zeros_like(ebr_r)
    eom_i = torch.zeros_like(ebr_r)
    eal_r = torch.zeros_like(ebr_r)
    eal_i = torch.zeros_like(ebr_r)

    g_wr = torch.zeros_like(wr)
    g_wi = torch.zeros_like(wi)
    g_br = torch.zeros_like(params.b_r)
    g_bi = torch.zeros_like(params.b_i)
    g_om = torch.zeros_like(params.omega_raw)
    g_al = torch.zeros_like(params.alpha_raw)

    theta = threshold.view(1, O, 1)
    tail_start = max(0, T - tail)
    tail_len = max(1, T - tail_start)
    tail_scale = 1.0 / float(tail_len)

    for t in range(T):
        patches = _unfold(x_seq[:, t], cfg)
        prev_r = z_r
        prev_i = z_i
        drive_r = torch.einsum("bkl,ok->bol", patches, wr) + params.b_r.view(1, O, 1)
        drive_i = torch.einsum("bkl,ok->bol", patches, wi) + params.b_i.view(1, O, 1)
        zr_next = rot_r.view(1, O, 1) * prev_r - rot_i.view(1, O, 1) * prev_i + drive_r
        zi_next = rot_i.view(1, O, 1) * prev_r + rot_r.view(1, O, 1) * prev_i + drive_i
        if cfg.z_clip > 0:
            zr_next = zr_next.clamp(-cfg.z_clip, cfg.z_clip)
            zi_next = zi_next.clamp(-cfg.z_clip, cfg.z_clip)

        add_patch = patches.unsqueeze(1)
        ewr_r_new = rot_r.view(1, O, 1, 1) * ewr_r - rot_i.view(1, O, 1, 1) * ewr_i + add_patch
        ewr_i_new = rot_i.view(1, O, 1, 1) * ewr_r + rot_r.view(1, O, 1, 1) * ewr_i
        ewi_r_new = rot_r.view(1, O, 1, 1) * ewi_r - rot_i.view(1, O, 1, 1) * ewi_i
        ewi_i_new = rot_i.view(1, O, 1, 1) * ewi_r + rot_r.view(1, O, 1, 1) * ewi_i + add_patch
        ewr_r, ewr_i = ewr_r_new, ewr_i_new
        ewi_r, ewi_i = ewi_r_new, ewi_i_new

        ebr_r_new = rot_r.view(1, O, 1) * ebr_r - rot_i.view(1, O, 1) * ebr_i + 1.0
        ebr_i_new = rot_i.view(1, O, 1) * ebr_r + rot_r.view(1, O, 1) * ebr_i
        ebi_r_new = rot_r.view(1, O, 1) * ebi_r - rot_i.view(1, O, 1) * ebi_i
        ebi_i_new = rot_i.view(1, O, 1) * ebi_r + rot_r.view(1, O, 1) * ebi_i + 1.0
        ebr_r, ebr_i = ebr_r_new, ebr_i_new
        ebi_r, ebi_i = ebi_r_new, ebi_i_new

        rz_r = cos_w.view(1, O, 1) * prev_r - sin_w.view(1, O, 1) * prev_i
        rz_i = sin_w.view(1, O, 1) * prev_r + cos_w.view(1, O, 1) * prev_i
        eom_r_new = rot_r.view(1, O, 1) * eom_r - rot_i.view(1, O, 1) * eom_i \
            - al.view(1, O, 1) * rz_i * d_omega_d_raw.view(1, O, 1)
        eom_i_new = rot_i.view(1, O, 1) * eom_r + rot_r.view(1, O, 1) * eom_i \
            + al.view(1, O, 1) * rz_r * d_omega_d_raw.view(1, O, 1)
        eom_r, eom_i = eom_r_new, eom_i_new
        eal_r_new = rot_r.view(1, O, 1) * eal_r - rot_i.view(1, O, 1) * eal_i \
            + rz_r * d_alpha_d_raw.view(1, O, 1)
        eal_i_new = rot_i.view(1, O, 1) * eal_r + rot_r.view(1, O, 1) * eal_i \
            + rz_i * d_alpha_d_raw.view(1, O, 1)
        eal_r, eal_i = eal_r_new, eal_i_new

        z_r, z_i = zr_next, zi_next
        p_t = torch.sigmoid(beta * (z_r.square() + z_i.square() - theta))

        if t >= tail_start:
            c_t = credit * tail_scale
            sig_prime = p_t * (1.0 - p_t) * beta
            base = 2.0 * c_t * sig_prime
            g_wr.add_((base.unsqueeze(2) * (z_r.unsqueeze(2) * ewr_r + z_i.unsqueeze(2) * ewr_i)).sum(dim=(0, 3)))
            g_wi.add_((base.unsqueeze(2) * (z_r.unsqueeze(2) * ewi_r + z_i.unsqueeze(2) * ewi_i)).sum(dim=(0, 3)))
            g_br.add_((base * (z_r * ebr_r + z_i * ebr_i)).sum(dim=(0, 2)))
            g_bi.add_((base * (z_r * ebi_r + z_i * ebi_i)).sum(dim=(0, 2)))
            g_om.add_((base * (z_r * eom_r + z_i * eom_i)).sum(dim=(0, 2)))
            g_al.add_((base * (z_r * eal_r + z_i * eal_i)).sum(dim=(0, 2)))

        if kappa_reset > 0:
            s_t = (torch.empty_like(p_t).uniform_(0.0, 1.0) < p_t).to(p_t.dtype)
            keep = 1.0 - kappa_reset * s_t
            z_r.mul_(keep)
            z_i.mul_(keep)

    return ConvOscillatorGrads(
        d_r=g_wr.view_as(params.d_r),
        d_i=g_wi.view_as(params.d_i),
        b_r=g_br,
        b_i=g_bi,
        omega_raw=g_om,
        alpha_raw=g_al,
    )
