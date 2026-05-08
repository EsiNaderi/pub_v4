"""Per-neuron complex damped-rotation oscillator with eligibility traces.

Model (per neuron i, generalized to multi-channel input):

    z_i(t+1) = alpha_i * exp(i*omega_i) * z_i(t) + sum_k d_ik * x_k(t) + b_i
    E_i      = mean_{t in tail} |z_i(t)|^2

Parameters per neuron:
    d_r[i, k], d_i[i, k]    complex input weights (one pair per input channel k)
    b_r[i],    b_i[i]       complex bias
    omega_raw[i]            mapped via sigmoid to [omega_min, omega_max]
    alpha_raw[i]            mapped via sigmoid to [alpha_min, alpha_max]

Forward maintains eligibility traces e_p^r, e_p^i for each parameter p.
Tail accumulates dE/dp = (2 / tail_len) * sum_{t in tail} (z_r * e_p^r + z_i * e_p^i).

This is the same substrate as pub_v3's resonant_self_organizing_layer.py,
generalized to F input channels and vectorized over (B, P) where P is
total neurons (typically arranged as K_pools * M_neurons).

Crucially: NO BPTT, NO surrogate spikes. The forward pass tracks each
parameter's effect on the tail energy purely by forward eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1e-12


@dataclass
class OscillatorConfig:
    n_neurons: int                 # P
    n_input_channels: int          # F
    omega_min: float = 0.01
    omega_max: float = 1.20
    alpha_min: float = 0.80
    alpha_max: float = 0.999
    omega_init_scale: float = 1.5
    alpha_init_scale: float = 0.25
    input_init: float = 0.1
    bias_init: float = 0.0
    z_clip: float = 0.0            # 0 disables (linear regime)


@dataclass
class OscillatorParams:
    d_r: torch.Tensor              # (P, F)
    d_i: torch.Tensor              # (P, F)
    b_r: torch.Tensor              # (P,)
    b_i: torch.Tensor              # (P,)
    omega_raw: torch.Tensor        # (P,)
    alpha_raw: torch.Tensor        # (P,)

    def tensors(self):
        return [self.d_r, self.d_i, self.b_r, self.b_i, self.omega_raw, self.alpha_raw]

    def names(self):
        return ["d_r", "d_i", "b_r", "b_i", "omega_raw", "alpha_raw"]


@dataclass
class OscillatorGrads:
    d_r: torch.Tensor
    d_i: torch.Tensor
    b_r: torch.Tensor
    b_i: torch.Tensor
    omega_raw: torch.Tensor
    alpha_raw: torch.Tensor

    def tensors(self):
        return [self.d_r, self.d_i, self.b_r, self.b_i, self.omega_raw, self.alpha_raw]


def init_params(cfg: OscillatorConfig, *, generator: torch.Generator | None = None) -> OscillatorParams:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(20260507)
    P = cfg.n_neurons
    F = cfg.n_input_channels
    return OscillatorParams(
        d_r=cfg.input_init * torch.randn(P, F, generator=generator),
        d_i=cfg.input_init * torch.randn(P, F, generator=generator),
        b_r=cfg.bias_init * torch.randn(P, generator=generator),
        b_i=cfg.bias_init * torch.randn(P, generator=generator),
        omega_raw=cfg.omega_init_scale * torch.randn(P, generator=generator),
        alpha_raw=cfg.alpha_init_scale * torch.randn(P, generator=generator),
    )


def omega_of(params: OscillatorParams, cfg: OscillatorConfig) -> torch.Tensor:
    return cfg.omega_min + (cfg.omega_max - cfg.omega_min) * torch.sigmoid(params.omega_raw)


def alpha_of(params: OscillatorParams, cfg: OscillatorConfig) -> torch.Tensor:
    return cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * torch.sigmoid(params.alpha_raw)


def forward_with_eligibility(
    x_seq: torch.Tensor,                    # (B, T, F)
    params: OscillatorParams,
    cfg: OscillatorConfig,
    tail: int,
    accumulate_traces: bool = True,
    save_amp_seq: bool = False,
) -> dict:
    """Forward pass + tail-energy + per-parameter dE/dp accumulators.

    Returns a dict with:
        z_r:  (B, P)        final real part
        z_i:  (B, P)        final imaginary part
        E:    (B, P)        tail energy
        z_seq_r: (B, T, P)  full trajectory (only if requested)
        z_seq_i: (B, T, P)
        dE_d_r: (B, P, F) gradient of tail energy wrt d_r per neuron
        dE_d_i: (B, P, F)
        dE_b_r: (B, P)
        dE_b_i: (B, P)
        dE_omega_raw: (B, P)
        dE_alpha_raw: (B, P)

    All gradient tensors are summed over the tail window (mean already
    applied; multiplied by 2 because dE/dp = 2 * Re(z* * dz/dp)).

    The per-batch-per-neuron structure means a downstream layer can
    multiply by a per-neuron credit signal then sum-reduce over batch.
    """
    B, T, F = x_seq.shape
    P = params.d_r.shape[0]
    assert F == cfg.n_input_channels

    om = omega_of(params, cfg)            # (P,)
    al = alpha_of(params, cfg)
    cos_w = torch.cos(om)
    sin_w = torch.sin(om)
    rot_r = al * cos_w
    rot_i = al * sin_w
    sig_o = torch.sigmoid(params.omega_raw)
    sig_a = torch.sigmoid(params.alpha_raw)
    d_omega_d_raw = (cfg.omega_max - cfg.omega_min) * sig_o * (1.0 - sig_o)  # (P,)
    d_alpha_d_raw = (cfg.alpha_max - cfg.alpha_min) * sig_a * (1.0 - sig_a)

    z_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    z_i = torch.zeros_like(z_r)

    # eligibility traces shaped to match parameters; per (B, P[, F])
    if accumulate_traces:
        edr_r = torch.zeros(B, P, F, dtype=x_seq.dtype, device=x_seq.device)
        edr_i = torch.zeros_like(edr_r)
        edi_r = torch.zeros_like(edr_r)
        edi_i = torch.zeros_like(edr_r)
        ebr_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
        ebr_i = torch.zeros_like(ebr_r)
        ebi_r = torch.zeros_like(ebr_r)
        ebi_i = torch.zeros_like(ebr_r)
        eom_r = torch.zeros_like(ebr_r)
        eom_i = torch.zeros_like(ebr_r)
        eal_r = torch.zeros_like(ebr_r)
        eal_i = torch.zeros_like(ebr_r)

        gE_d_r = torch.zeros_like(edr_r)        # (B, P, F)
        gE_d_i = torch.zeros_like(edr_r)
        gE_b_r = torch.zeros_like(ebr_r)        # (B, P)
        gE_b_i = torch.zeros_like(ebr_r)
        gE_om = torch.zeros_like(ebr_r)
        gE_al = torch.zeros_like(ebr_r)

    energy_sum = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)

    if save_amp_seq:
        amp_seq = torch.zeros(B, T, P, dtype=x_seq.dtype, device=x_seq.device)
    else:
        amp_seq = None

    tail_start = max(0, T - tail)
    tail_len = max(1, T - tail_start)

    for t in range(T):
        x_t = x_seq[:, t]                       # (B, F)
        prev_r = z_r
        prev_i = z_i

        # input drive: (B, F) @ (F, P) -> (B, P)
        drive_r = x_t @ params.d_r.t() + params.b_r.unsqueeze(0)
        drive_i = x_t @ params.d_i.t() + params.b_i.unsqueeze(0)

        z_r_next = rot_r.unsqueeze(0) * prev_r - rot_i.unsqueeze(0) * prev_i + drive_r
        z_i_next = rot_i.unsqueeze(0) * prev_r + rot_r.unsqueeze(0) * prev_i + drive_i

        if cfg.z_clip > 0:
            z_r_next = z_r_next.clamp(-cfg.z_clip, cfg.z_clip)
            z_i_next = z_i_next.clamp(-cfg.z_clip, cfg.z_clip)

        if accumulate_traces:
            # rotate eligibility traces: each is updated by R then add direct contribution
            def rot_e(er, ei):
                return rot_r.unsqueeze(0) * er - rot_i.unsqueeze(0) * ei, \
                       rot_i.unsqueeze(0) * er + rot_r.unsqueeze(0) * ei

            # for d_r[k]: dz_next/dd_r[k] = R * dz/dd_r[k] + (x_k, 0) (since drive = x @ d_r.T contributes to real)
            # Eligibilities have shape (B, P, F); direct contribution is x_t broadcast across P.
            # rot_e for traces with extra F dim:
            edr_r_new = rot_r.view(1, P, 1) * edr_r - rot_i.view(1, P, 1) * edr_i + x_t.unsqueeze(1)
            edr_i_new = rot_i.view(1, P, 1) * edr_r + rot_r.view(1, P, 1) * edr_i
            edi_r_new = rot_r.view(1, P, 1) * edi_r - rot_i.view(1, P, 1) * edi_i
            edi_i_new = rot_i.view(1, P, 1) * edi_r + rot_r.view(1, P, 1) * edi_i + x_t.unsqueeze(1)
            edr_r, edr_i = edr_r_new, edr_i_new
            edi_r, edi_i = edi_r_new, edi_i_new

            ebr_r_new = rot_r.unsqueeze(0) * ebr_r - rot_i.unsqueeze(0) * ebr_i + 1.0
            ebr_i_new = rot_i.unsqueeze(0) * ebr_r + rot_r.unsqueeze(0) * ebr_i
            ebi_r_new = rot_r.unsqueeze(0) * ebi_r - rot_i.unsqueeze(0) * ebi_i
            ebi_i_new = rot_i.unsqueeze(0) * ebi_r + rot_r.unsqueeze(0) * ebi_i + 1.0
            ebr_r, ebr_i = ebr_r_new, ebr_i_new
            ebi_r, ebi_i = ebi_r_new, ebi_i_new

            # omega: rot_z := alpha * R(omega) z = alpha * (cos*z_r - sin*z_i, sin*z_r + cos*z_i)
            # d(rot_z_r)/d omega = alpha * (-sin*z_r - cos*z_i)
            # d(rot_z_i)/d omega = alpha * ( cos*z_r - sin*z_i)
            # Multiplied by domega/draw chain rule.
            rz_r = cos_w.unsqueeze(0) * prev_r - sin_w.unsqueeze(0) * prev_i
            rz_i = sin_w.unsqueeze(0) * prev_r + cos_w.unsqueeze(0) * prev_i
            eom_r_new, eom_i_new = rot_e(eom_r, eom_i)
            eom_r_new = eom_r_new - al.unsqueeze(0) * rz_i * d_omega_d_raw.unsqueeze(0)
            eom_i_new = eom_i_new + al.unsqueeze(0) * rz_r * d_omega_d_raw.unsqueeze(0)
            eom_r, eom_i = eom_r_new, eom_i_new

            # alpha: d(alpha * R z)/d alpha = R z. Then chain through dalpha/draw.
            eal_r_new, eal_i_new = rot_e(eal_r, eal_i)
            eal_r_new = eal_r_new + rz_r * d_alpha_d_raw.unsqueeze(0)
            eal_i_new = eal_i_new + rz_i * d_alpha_d_raw.unsqueeze(0)
            eal_r, eal_i = eal_r_new, eal_i_new

        z_r = z_r_next
        z_i = z_i_next

        if save_amp_seq:
            amp_seq[:, t] = z_r.square() + z_i.square()

        if t >= tail_start:
            energy_sum = energy_sum + z_r.square() + z_i.square()
            if accumulate_traces:
                # dE/dp accumulates 2*(z_r*e_p^r + z_i*e_p^i) per timestep
                gE_d_r = gE_d_r + 2.0 * (z_r.unsqueeze(-1) * edr_r + z_i.unsqueeze(-1) * edr_i)
                gE_d_i = gE_d_i + 2.0 * (z_r.unsqueeze(-1) * edi_r + z_i.unsqueeze(-1) * edi_i)
                gE_b_r = gE_b_r + 2.0 * (z_r * ebr_r + z_i * ebr_i)
                gE_b_i = gE_b_i + 2.0 * (z_r * ebi_r + z_i * ebi_i)
                gE_om = gE_om + 2.0 * (z_r * eom_r + z_i * eom_i)
                gE_al = gE_al + 2.0 * (z_r * eal_r + z_i * eal_i)

    energy = energy_sum / tail_len

    out = {
        "z_r": z_r,
        "z_i": z_i,
        "E": energy,
    }
    if save_amp_seq:
        out["amp_seq"] = amp_seq
    if accumulate_traces:
        out["dE_d_r"] = gE_d_r / tail_len
        out["dE_d_i"] = gE_d_i / tail_len
        out["dE_b_r"] = gE_b_r / tail_len
        out["dE_b_i"] = gE_b_i / tail_len
        out["dE_omega_raw"] = gE_om / tail_len
        out["dE_alpha_raw"] = gE_al / tail_len
    return out


def forward_with_eligibility_sparse(
    x_seq: torch.Tensor,                    # (B, T, M_in)
    in_idx: torch.Tensor,                   # (P, K) sparse fan-in indices
    params: OscillatorParams,                # d_r, d_i shape (P, K)
    cfg: OscillatorConfig,
    tail: int,
    accumulate_traces: bool = True,
    save_amp_seq: bool = False,
) -> dict:
    """Same as forward_with_eligibility but with per-neuron sparse fan-in.

    Each output neuron i has K input weights connecting to specific
    M_in stage-output channels (chosen by in_idx[i, :]).

    cfg.n_input_channels must equal K (the per-neuron fan-in).
    params.d_r, params.d_i must have shape (P, K).
    """
    B, T, M_in = x_seq.shape
    P, K = in_idx.shape
    assert params.d_r.shape == (P, K), f"expected (P={P}, K={K}), got {params.d_r.shape}"

    om = omega_of(params, cfg)
    al = alpha_of(params, cfg)
    cos_w = torch.cos(om); sin_w = torch.sin(om)
    rot_r = al * cos_w
    rot_i = al * sin_w
    sig_o = torch.sigmoid(params.omega_raw)
    sig_a = torch.sigmoid(params.alpha_raw)
    d_omega_d_raw = (cfg.omega_max - cfg.omega_min) * sig_o * (1.0 - sig_o)
    d_alpha_d_raw = (cfg.alpha_max - cfg.alpha_min) * sig_a * (1.0 - sig_a)

    z_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    z_i = torch.zeros_like(z_r)

    if accumulate_traces:
        edr_r = torch.zeros(B, P, K, dtype=x_seq.dtype, device=x_seq.device)
        edr_i = torch.zeros_like(edr_r)
        edi_r = torch.zeros_like(edr_r)
        edi_i = torch.zeros_like(edr_r)
        ebr_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device); ebr_i = torch.zeros_like(ebr_r)
        ebi_r = torch.zeros_like(ebr_r); ebi_i = torch.zeros_like(ebr_r)
        eom_r = torch.zeros_like(ebr_r); eom_i = torch.zeros_like(ebr_r)
        eal_r = torch.zeros_like(ebr_r); eal_i = torch.zeros_like(ebr_r)
        gE_d_r = torch.zeros_like(edr_r); gE_d_i = torch.zeros_like(edr_r)
        gE_b_r = torch.zeros_like(ebr_r); gE_b_i = torch.zeros_like(ebr_r)
        gE_om = torch.zeros_like(ebr_r); gE_al = torch.zeros_like(ebr_r)

    energy_sum = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    if save_amp_seq:
        amp_seq = torch.zeros(B, T, P, dtype=x_seq.dtype, device=x_seq.device)
    else:
        amp_seq = None

    tail_start = max(0, T - tail)
    tail_len = max(1, T - tail_start)

    for t in range(T):
        x_t = x_seq[:, t]                       # (B, M_in)
        # gather per-neuron inputs: shape (B, P, K)
        gathered = x_t[:, in_idx]               # advanced indexing: (B, P, K)
        prev_r = z_r; prev_i = z_i

        drive_r = (gathered * params.d_r.unsqueeze(0)).sum(dim=2) + params.b_r.unsqueeze(0)
        drive_i = (gathered * params.d_i.unsqueeze(0)).sum(dim=2) + params.b_i.unsqueeze(0)

        z_r_next = rot_r.unsqueeze(0) * prev_r - rot_i.unsqueeze(0) * prev_i + drive_r
        z_i_next = rot_i.unsqueeze(0) * prev_r + rot_r.unsqueeze(0) * prev_i + drive_i
        if cfg.z_clip > 0:
            z_r_next = z_r_next.clamp(-cfg.z_clip, cfg.z_clip)
            z_i_next = z_i_next.clamp(-cfg.z_clip, cfg.z_clip)

        if accumulate_traces:
            edr_r_new = rot_r.view(1, P, 1) * edr_r - rot_i.view(1, P, 1) * edr_i + gathered
            edr_i_new = rot_i.view(1, P, 1) * edr_r + rot_r.view(1, P, 1) * edr_i
            edi_r_new = rot_r.view(1, P, 1) * edi_r - rot_i.view(1, P, 1) * edi_i
            edi_i_new = rot_i.view(1, P, 1) * edi_r + rot_r.view(1, P, 1) * edi_i + gathered
            edr_r, edr_i = edr_r_new, edr_i_new
            edi_r, edi_i = edi_r_new, edi_i_new

            ebr_r_new = rot_r.unsqueeze(0) * ebr_r - rot_i.unsqueeze(0) * ebr_i + 1.0
            ebr_i_new = rot_i.unsqueeze(0) * ebr_r + rot_r.unsqueeze(0) * ebr_i
            ebi_r_new = rot_r.unsqueeze(0) * ebi_r - rot_i.unsqueeze(0) * ebi_i
            ebi_i_new = rot_i.unsqueeze(0) * ebi_r + rot_r.unsqueeze(0) * ebi_i + 1.0
            ebr_r, ebr_i = ebr_r_new, ebr_i_new
            ebi_r, ebi_i = ebi_r_new, ebi_i_new

            rz_r = cos_w.unsqueeze(0) * prev_r - sin_w.unsqueeze(0) * prev_i
            rz_i = sin_w.unsqueeze(0) * prev_r + cos_w.unsqueeze(0) * prev_i
            eom_r_new = rot_r.unsqueeze(0) * eom_r - rot_i.unsqueeze(0) * eom_i \
                        - al.unsqueeze(0) * rz_i * d_omega_d_raw.unsqueeze(0)
            eom_i_new = rot_i.unsqueeze(0) * eom_r + rot_r.unsqueeze(0) * eom_i \
                        + al.unsqueeze(0) * rz_r * d_omega_d_raw.unsqueeze(0)
            eom_r, eom_i = eom_r_new, eom_i_new
            eal_r_new = rot_r.unsqueeze(0) * eal_r - rot_i.unsqueeze(0) * eal_i \
                        + rz_r * d_alpha_d_raw.unsqueeze(0)
            eal_i_new = rot_i.unsqueeze(0) * eal_r + rot_r.unsqueeze(0) * eal_i \
                        + rz_i * d_alpha_d_raw.unsqueeze(0)
            eal_r, eal_i = eal_r_new, eal_i_new

        z_r = z_r_next; z_i = z_i_next
        if save_amp_seq:
            amp_seq[:, t] = z_r.square() + z_i.square()
        if t >= tail_start:
            energy_sum = energy_sum + z_r.square() + z_i.square()
            if accumulate_traces:
                gE_d_r = gE_d_r + 2.0 * (z_r.unsqueeze(-1) * edr_r + z_i.unsqueeze(-1) * edr_i)
                gE_d_i = gE_d_i + 2.0 * (z_r.unsqueeze(-1) * edi_r + z_i.unsqueeze(-1) * edi_i)
                gE_b_r = gE_b_r + 2.0 * (z_r * ebr_r + z_i * ebr_i)
                gE_b_i = gE_b_i + 2.0 * (z_r * ebi_r + z_i * ebi_i)
                gE_om = gE_om + 2.0 * (z_r * eom_r + z_i * eom_i)
                gE_al = gE_al + 2.0 * (z_r * eal_r + z_i * eal_i)

    energy = energy_sum / tail_len
    out = {"z_r": z_r, "z_i": z_i, "E": energy}
    if save_amp_seq: out["amp_seq"] = amp_seq
    if accumulate_traces:
        out["dE_d_r"] = gE_d_r / tail_len
        out["dE_d_i"] = gE_d_i / tail_len
        out["dE_b_r"] = gE_b_r / tail_len
        out["dE_b_i"] = gE_b_i / tail_len
        out["dE_omega_raw"] = gE_om / tail_len
        out["dE_alpha_raw"] = gE_al / tail_len
    return out


def forward_with_amp_trace(
    x_seq: torch.Tensor,                    # (B, T, F)
    params: OscillatorParams,
    cfg: OscillatorConfig,
) -> dict:
    """Cheaper forward pass that returns the full amplitude trajectory,
    no eligibility tracking. Used at inference time and for upstream
    feeding of stage k+1 by stage k's r_i(t)."""
    B, T, F = x_seq.shape
    P = params.d_r.shape[0]
    assert F == cfg.n_input_channels

    om = omega_of(params, cfg)
    al = alpha_of(params, cfg)
    cos_w = torch.cos(om)
    sin_w = torch.sin(om)
    rot_r = al * cos_w
    rot_i = al * sin_w

    z_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    z_i = torch.zeros_like(z_r)
    z_seq_r = torch.zeros(B, T, P, dtype=x_seq.dtype, device=x_seq.device)
    z_seq_i = torch.zeros_like(z_seq_r)

    for t in range(T):
        x_t = x_seq[:, t]
        drive_r = x_t @ params.d_r.t() + params.b_r.unsqueeze(0)
        drive_i = x_t @ params.d_i.t() + params.b_i.unsqueeze(0)
        z_r_next = rot_r.unsqueeze(0) * z_r - rot_i.unsqueeze(0) * z_i + drive_r
        z_i_next = rot_i.unsqueeze(0) * z_r + rot_r.unsqueeze(0) * z_i + drive_i
        if cfg.z_clip > 0:
            z_r_next = z_r_next.clamp(-cfg.z_clip, cfg.z_clip)
            z_i_next = z_i_next.clamp(-cfg.z_clip, cfg.z_clip)
        z_r = z_r_next
        z_i = z_i_next
        z_seq_r[:, t] = z_r
        z_seq_i[:, t] = z_i

    amp = (z_seq_r.square() + z_seq_i.square()).sqrt()
    return {
        "z_seq_r": z_seq_r,
        "z_seq_i": z_seq_i,
        "amp_seq": amp,
    }
