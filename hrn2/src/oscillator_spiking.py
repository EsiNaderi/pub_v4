"""Spiking variant of the per-neuron damped-rotation oscillator.

Each oscillator emits BINARY spikes used as the inter-neuron signal.
Spike emission is stochastic Bernoulli with rate
    p(t) = sigmoid( beta * (|z(t)|^2 - theta) ).

After a spike, the state is subtractively reset:
    z(t) <- (1 - kappa * s(t)) * z(t)
where kappa in [0, 1) is the reset gain (kappa=0 disables reset).

OPTIONAL intra-stage recurrence: each neuron may also receive binary
spikes from a fixed sparse set of K_rec other neurons in the SAME
stage at the previous timestep. Recurrent input weights (rec_d_r,
rec_d_i) are learned via their own eligibility traces.

Inter-neuron signal is always binary s(t) in {0, 1}. Per-stage tail
readout uses the smooth rate rho_i = (1/T_tail) sum_t p_i(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from oscillator import OscillatorConfig, OscillatorParams, omega_of, alpha_of


@dataclass
class RecurrentParams:
    d_r: torch.Tensor       # (P, K_rec) recurrent real weights
    d_i: torch.Tensor       # (P, K_rec) recurrent imag weights

    def tensors(self):
        return [self.d_r, self.d_i]


def init_recurrent_params(P: int, K_rec: int, init_scale: float = 0.02,
                          generator: torch.Generator | None = None,
                          *, init_bias: float = 0.0) -> RecurrentParams:
    """init_bias < 0 makes recurrence net inhibitory (lateral inhibition / WTA).

    `init_bias` is keyword-only to keep the positional API backward-compatible.
    """
    if generator is None:
        generator = torch.Generator(); generator.manual_seed(20260507)
    return RecurrentParams(
        d_r=init_bias + init_scale * torch.randn(P, K_rec, generator=generator),
        d_i=init_scale * torch.randn(P, K_rec, generator=generator),
    )


def forward_with_eligibility_sparse_spiking(
    x_seq: torch.Tensor,                    # (B, T, M_in) -- can be binary or continuous
    in_idx: torch.Tensor,                   # (P, K) sparse fan-in indices
    params: OscillatorParams,                # d_r, d_i shape (P, K)
    cfg: OscillatorConfig,
    tail: int,
    threshold: torch.Tensor,                 # (P,) per-neuron threshold theta_i
    beta: float = 10.0,
    accumulate_traces: bool = True,
    save_spike_seq: bool = True,
    sample_binary: bool = True,              # if False, propagate smooth rate p(t) instead
    rng: torch.Generator | None = None,
    kappa_reset: float = 0.0,                # subtractive reset gain on spike (0 = no reset)
    rec_idx: torch.Tensor | None = None,     # (P, K_rec) intra-stage recurrent fan-in (optional)
    rec_params: RecurrentParams | None = None,
    top_seq: torch.Tensor | None = None,     # (B, T, M_top) top-down spike train from stage L+1 (pass 1)
    top_idx: torch.Tensor | None = None,     # (P, K_top) top-down fan-in indices
    top_params: RecurrentParams | None = None,
) -> dict:
    """Spiking forward pass with eligibility traces for the rate gradient.

    Returns a dict with:
        z_r, z_i        : (B, P) final state
        rho             : (B, P) tail mean rate (smooth) -- THIS is what loss uses
        spike_seq       : (B, T, P) binary spikes (or rates if sample_binary=False)
        dRho_*          : (B, P, [K]) parameter gradients of rho
        spike_count     : (B, P) total spikes emitted in the tail (for diagnostic)
    """
    B, T, M_in = x_seq.shape
    P, K = in_idx.shape
    assert params.d_r.shape == (P, K), f"expected ({P}, {K}), got {params.d_r.shape}"
    assert threshold.shape == (P,), f"threshold shape {threshold.shape} != ({P},)"

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
    s_prev = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)  # for recurrence

    use_rec = (rec_idx is not None) and (rec_params is not None)
    if use_rec:
        Krec = rec_idx.shape[1]
        assert rec_params.d_r.shape == (P, Krec), f"rec_params.d_r shape {rec_params.d_r.shape} != ({P},{Krec})"

    use_top = (top_seq is not None) and (top_idx is not None) and (top_params is not None)
    if use_top:
        Ktop = top_idx.shape[1]
        assert top_seq.shape[0] == B and top_seq.shape[1] == T, \
            f"top_seq shape {top_seq.shape} mismatch (B={B}, T={T})"
        assert top_params.d_r.shape == (P, Ktop), \
            f"top_params.d_r shape {top_params.d_r.shape} != ({P},{Ktop})"

    if accumulate_traces:
        edr_r = torch.zeros(B, P, K, dtype=x_seq.dtype, device=x_seq.device)
        edr_i = torch.zeros_like(edr_r); edi_r = torch.zeros_like(edr_r); edi_i = torch.zeros_like(edr_r)
        ebr_r = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device); ebr_i = torch.zeros_like(ebr_r)
        ebi_r = torch.zeros_like(ebr_r); ebi_i = torch.zeros_like(ebr_r)
        eom_r = torch.zeros_like(ebr_r); eom_i = torch.zeros_like(ebr_r)
        eal_r = torch.zeros_like(ebr_r); eal_i = torch.zeros_like(ebr_r)
        gR_d_r = torch.zeros_like(edr_r); gR_d_i = torch.zeros_like(edr_r)
        gR_b_r = torch.zeros_like(ebr_r); gR_b_i = torch.zeros_like(ebr_r)
        gR_om = torch.zeros_like(ebr_r); gR_al = torch.zeros_like(ebr_r)
        if use_rec:
            erecdr_r = torch.zeros(B, P, Krec, dtype=x_seq.dtype, device=x_seq.device)
            erecdr_i = torch.zeros_like(erecdr_r)
            erecdi_r = torch.zeros_like(erecdr_r)
            erecdi_i = torch.zeros_like(erecdr_r)
            gR_recd_r = torch.zeros_like(erecdr_r)
            gR_recd_i = torch.zeros_like(erecdr_r)
        if use_top:
            etopdr_r = torch.zeros(B, P, Ktop, dtype=x_seq.dtype, device=x_seq.device)
            etopdr_i = torch.zeros_like(etopdr_r)
            etopdi_r = torch.zeros_like(etopdr_r)
            etopdi_i = torch.zeros_like(etopdr_r)
            gR_topd_r = torch.zeros_like(etopdr_r)
            gR_topd_i = torch.zeros_like(etopdr_r)

    rho_sum = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    spike_count = torch.zeros(B, P, dtype=x_seq.dtype, device=x_seq.device)
    if save_spike_seq:
        spike_seq = torch.zeros(B, T, P, dtype=x_seq.dtype, device=x_seq.device)
    else:
        spike_seq = None

    tail_start = max(0, T - tail)
    tail_len = max(1, T - tail_start)

    theta = threshold.unsqueeze(0)              # (1, P) for broadcast

    for t in range(T):
        x_t = x_seq[:, t]                       # (B, M_in)
        gathered = x_t[:, in_idx]               # (B, P, K)
        prev_r = z_r; prev_i = z_i

        drive_r = (gathered * params.d_r.unsqueeze(0)).sum(dim=2) + params.b_r.unsqueeze(0)
        drive_i = (gathered * params.d_i.unsqueeze(0)).sum(dim=2) + params.b_i.unsqueeze(0)

        if use_rec:
            # gather recurrent input from this stage's spikes at t-1
            gathered_rec = s_prev[:, rec_idx]            # (B, P, Krec)
            drive_r = drive_r + (gathered_rec * rec_params.d_r.unsqueeze(0)).sum(dim=2)
            drive_i = drive_i + (gathered_rec * rec_params.d_i.unsqueeze(0)).sum(dim=2)

        if use_top:
            # gather top-down input from previous-pass stage L+1 spikes at this same time
            top_t = top_seq[:, t]                          # (B, M_top)
            gathered_top = top_t[:, top_idx]               # (B, P, Ktop)
            drive_r = drive_r + (gathered_top * top_params.d_r.unsqueeze(0)).sum(dim=2)
            drive_i = drive_i + (gathered_top * top_params.d_i.unsqueeze(0)).sum(dim=2)

        z_r_next = rot_r.unsqueeze(0) * prev_r - rot_i.unsqueeze(0) * prev_i + drive_r
        z_i_next = rot_i.unsqueeze(0) * prev_r + rot_r.unsqueeze(0) * prev_i + drive_i
        if cfg.z_clip > 0:
            z_r_next = z_r_next.clamp(-cfg.z_clip, cfg.z_clip)
            z_i_next = z_i_next.clamp(-cfg.z_clip, cfg.z_clip)

        if accumulate_traces:
            # Account for subtractive reset on previous spike: traces follow
            # dz_tilde(t)/dp = R(omega) * (1 - kappa*s(t-1)) * dz_tilde(t-1)/dp + new
            if kappa_reset > 0:
                reset_factor = 1.0 - kappa_reset * s_prev               # (B, P)
                rf3 = reset_factor.unsqueeze(-1)
                edr_r = edr_r * rf3; edr_i = edr_i * rf3
                edi_r = edi_r * rf3; edi_i = edi_i * rf3
                ebr_r = ebr_r * reset_factor; ebr_i = ebr_i * reset_factor
                ebi_r = ebi_r * reset_factor; ebi_i = ebi_i * reset_factor
                eom_r = eom_r * reset_factor; eom_i = eom_i * reset_factor
                eal_r = eal_r * reset_factor; eal_i = eal_i * reset_factor
                if use_rec:
                    erecdr_r = erecdr_r * rf3; erecdr_i = erecdr_i * rf3
                    erecdi_r = erecdi_r * rf3; erecdi_i = erecdi_i * rf3
                if use_top:
                    etopdr_r = etopdr_r * rf3; etopdr_i = etopdr_i * rf3
                    etopdi_r = etopdi_r * rf3; etopdi_i = etopdi_i * rf3

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

            if use_rec:
                # Recurrent input is treated as a fixed signal (s_prev not differentiated through),
                # matching e-prop's local-rule convention.
                erecdr_r_new = rot_r.view(1, P, 1) * erecdr_r - rot_i.view(1, P, 1) * erecdr_i + gathered_rec
                erecdr_i_new = rot_i.view(1, P, 1) * erecdr_r + rot_r.view(1, P, 1) * erecdr_i
                erecdi_r_new = rot_r.view(1, P, 1) * erecdi_r - rot_i.view(1, P, 1) * erecdi_i
                erecdi_i_new = rot_i.view(1, P, 1) * erecdi_r + rot_r.view(1, P, 1) * erecdi_i + gathered_rec
                erecdr_r, erecdr_i = erecdr_r_new, erecdr_i_new
                erecdi_r, erecdi_i = erecdi_r_new, erecdi_i_new

            if use_top:
                etopdr_r_new = rot_r.view(1, P, 1) * etopdr_r - rot_i.view(1, P, 1) * etopdr_i + gathered_top
                etopdr_i_new = rot_i.view(1, P, 1) * etopdr_r + rot_r.view(1, P, 1) * etopdr_i
                etopdi_r_new = rot_r.view(1, P, 1) * etopdi_r - rot_i.view(1, P, 1) * etopdi_i
                etopdi_i_new = rot_i.view(1, P, 1) * etopdi_r + rot_r.view(1, P, 1) * etopdi_i + gathered_top
                etopdr_r, etopdr_i = etopdr_r_new, etopdr_i_new
                etopdi_r, etopdi_i = etopdi_r_new, etopdi_i_new

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

        amp2 = z_r.square() + z_i.square()
        u = beta * (amp2 - theta)
        p_t = torch.sigmoid(u)                  # (B, P) spike rate at this step

        # Sample binary spike for inter-stage signal
        if sample_binary:
            if rng is None:
                s_t = torch.bernoulli(p_t)
            else:
                noise = torch.empty_like(p_t).uniform_(0.0, 1.0, generator=rng)
                s_t = (noise < p_t).to(p_t.dtype)
        else:
            s_t = p_t                            # propagate smooth rate

        if save_spike_seq:
            spike_seq[:, t] = s_t

        if t >= tail_start:
            rho_sum = rho_sum + p_t              # accumulate smooth rate for loss
            spike_count = spike_count + s_t      # binary count for diagnostics
            if accumulate_traces:
                # Bernoulli-mean derivative: dp/d|z|^2 = sigma'(u) * beta
                # (where sigma'(u) = p_t * (1 - p_t))
                sig_prime = p_t * (1.0 - p_t) * beta            # (B, P)
                gR_d_r = gR_d_r + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * edr_r + z_i.unsqueeze(-1) * edr_i)
                gR_d_i = gR_d_i + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * edi_r + z_i.unsqueeze(-1) * edi_i)
                gR_b_r = gR_b_r + sig_prime * 2.0 * (z_r * ebr_r + z_i * ebr_i)
                gR_b_i = gR_b_i + sig_prime * 2.0 * (z_r * ebi_r + z_i * ebi_i)
                gR_om = gR_om + sig_prime * 2.0 * (z_r * eom_r + z_i * eom_i)
                gR_al = gR_al + sig_prime * 2.0 * (z_r * eal_r + z_i * eal_i)
                if use_rec:
                    gR_recd_r = gR_recd_r + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * erecdr_r + z_i.unsqueeze(-1) * erecdr_i)
                    gR_recd_i = gR_recd_i + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * erecdi_r + z_i.unsqueeze(-1) * erecdi_i)
                if use_top:
                    gR_topd_r = gR_topd_r + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * etopdr_r + z_i.unsqueeze(-1) * etopdr_i)
                    gR_topd_i = gR_topd_i + sig_prime.unsqueeze(-1) * 2.0 * (z_r.unsqueeze(-1) * etopdi_r + z_i.unsqueeze(-1) * etopdi_i)

        # Subtractive reset on spike: z(t) <- (1 - kappa * s(t)) * z(t)
        if kappa_reset > 0:
            keep = 1.0 - kappa_reset * s_t       # (B, P) -- 1 if no spike, 1-kappa if spike
            z_r = z_r * keep
            z_i = z_i * keep

        # Carry s(t) for recurrent input next step
        s_prev = s_t

    rho = rho_sum / tail_len                     # (B, P) tail mean rate
    spike_rate = spike_count / tail_len          # (B, P) tail spike-count rate

    out = {
        "z_r": z_r, "z_i": z_i,
        "rho": rho,                              # smooth rate (used for loss)
        "spike_rate": spike_rate,                # binary-sampled rate (diagnostic / eval)
    }
    if save_spike_seq:
        out["spike_seq"] = spike_seq
    if accumulate_traces:
        out["dRho_d_r"] = gR_d_r / tail_len
        out["dRho_d_i"] = gR_d_i / tail_len
        out["dRho_b_r"] = gR_b_r / tail_len
        out["dRho_b_i"] = gR_b_i / tail_len
        out["dRho_omega_raw"] = gR_om / tail_len
        if use_rec:
            out["dRho_rec_d_r"] = gR_recd_r / tail_len
            out["dRho_rec_d_i"] = gR_recd_i / tail_len
        if use_top:
            out["dRho_top_d_r"] = gR_topd_r / tail_len
            out["dRho_top_d_i"] = gR_topd_i / tail_len
        out["dRho_alpha_raw"] = gR_al / tail_len
    return out
