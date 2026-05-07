# pub_v4 Experiment Log

Date: 2026-05-06 (overnight session)
Author: Claude Opus 4.7 (autonomous)

## Goal

Achieve 95%+ on SMNIST (single-channel input, 784 timesteps) with a
biologically plausible, hierarchical resonant neural architecture.

## Architecture Overview

`src/resonator.py` and `src/resonator_jit.py`:
A vectorized Stuart-Landau resonator pool. Each neuron has complex amplitude
`z = u + iv` evolved by:

```
q = (1 - lambda) * [exp(i*omega) z(t-1) + beta * z(t-1) (1 - |z|^2) - gamma z + W_rec @ s_prev + eta * D x(t)]
s = H(|q|^2 - theta)  (with surrogate gradient)
z = (1 - kappa * s) * q
```

Heterogeneous omega per neuron. Block-diagonal recurrent W_rec to enforce pool
structure. Frequency tiling option (`omega_per_pool`) gives each pool its own
sub-band.

`src/hrn.py`: Hierarchical Resonant Net composing layers of pools.
Output layer has K=10 pools (one per class) and reads tail-window per-class
pool spike rate as logits. Optional auxiliary linear head for capacity check.

`src/fractal_pool.py`: Two-level block recurrence for "modes within modes".
`src/resonant_alif.py`: Variant with adaptive threshold (ALIF-style).

## Stability Fixes Discovered

1. **Amplitude clipping** in forward: `u, v = clamp(u, -3, 3)`. Without this,
   gradients exploded to NaN through the SL nonlinearity over 784 timesteps.

2. **Detach amp_sq in SL term**: `sl = beta * (1 - amp_sq.detach()) - gamma`.
   The cubic feedback in `sl * u = (β(1-|z|²) - γ) u` causes gradient
   explosion. Treating amp_sq as a static gating factor in the backward
   path keeps dynamics correct in forward but stabilizes backward.

3. **Frozen stability params**: gamma, beta, lambda_leak, kappa, theta as
   buffers (not learnable). Their gradients are unstable through long
   sequences (theta gradient grew to 1e4 magnitude). Train only D, W_rec,
   omega, eta.

4. **Detached recurrence (e-prop style)**: `s_prev.detach()` before W_rec.
   Reduces gradient explosion through long sequences without losing
   recurrent computation.

## Firing Regime Tuning

Default config (post-tuning):
```
Layer 0 (sensory): 4 pools × 32 neurons. omega ∈ [0.5, 2.5].
                   theta=0.7, eta=0.30, in_init_scale=4.0
Layer 1 (mode):    8 pools × 32 neurons. omega ∈ [0.10, 1.0].
                   theta=0.7, eta=0.10, in_init_scale=2.0
Output (class):    10 pools × 24 neurons. omega ∈ [0.02, 0.30].
                   theta=0.6, eta=0.15, in_init_scale=4.0
gamma=beta=0.20 throughout.
```

Random-init firing rates on real SMNIST samples (32 batch):
- Layer 0: ~14% (responds to stroke pixels)
- Layer 1: ~3-15% (depending on tuning)
- Output: ~21% (modulated by Layer 1 spikes)

## Random-Feature Baselines (frozen network, trained head only)

| Subset | Head     | Train acc | Test acc |
|--------|----------|-----------|----------|
| 5k     | linear   | 0.50      | 0.47     |
| 5k     | pool-rate (only bias trainable) | 0.10 | 0.10 (chance) |
| 5k     | linear (big arch, ALIF) | 0.25 | 0.25 |

Note: pool-rate readout fails without supervision because random pools
do not naturally segregate by class. Hidden-layer plasticity is required.

## BPTT Training

`src/train_bptt.py` with auxiliary linear head:
- 5k subset, 8 epochs, lr=1e-3, batch=32: training in progress.
- Loss decreasing: 2.33 → 2.17 in first 20 steps (normal early training).
- Per-step time: ~10s on CPU, batch 32, T=784.
- Estimated full epoch: ~1 hour.

## Local-Rule Training (e-prop / 3-factor)

`src/train_local.py`:
- Forward pass logged (no autograd).
- Per-layer eligibility traces of presynaptic activity.
- Top-local CE gradient at output pools, feedback alignment to earlier layers.
- Manual Adam updates.
- ~100x faster than BPTT (0.5s/step vs 10s/step).

Status:
- Stable forward.
- Rates initially ran away upward without homeostasis.
- With homeostasis_coef=5.0, rates stabilize but classification accuracy
  remains at chance (10%). The local credit signal is not sufficiently
  selective to drive class-pool specialization.
- Open: tune the eligibility trace (currently exp_smooth without
  normalization), feedback alignment scale, and homeostatic balance.

## What Works

1. Hierarchical pool structure with block-diagonal recurrence.
2. Per-pool frequency tiling.
3. Forward dynamics stable across 784 timesteps with amplitude clipping.
4. BPTT gradient flow stable after the SL detach + amplitude clip fixes.
5. Auxiliary linear head provides a capacity check.
6. Random features have non-trivial signal (47% test acc on 5k subset).

## What Doesn't (Yet)

1. Pool-rate readout without head training: random pools do not specialize.
2. Local rules training: stable but not yet learning class structure.
3. Training rate is slow (BPTT bound by Python loop over T=784).

## Open Directions

1. **Two-stage training**: train aux_head first to establish per-pool class
   correspondence, then transfer that to per-class pool readout.
2. **Top-down feedback**: include feedback from class pools to mode pools
   for synchronization-based selectivity.
3. **Fractal pool architecture** (`fractal_pool.py`): test whether
   recursive sub-pool structure improves features.
4. **ResonantALIF** (`resonant_alif.py`): adaptive threshold + resonance.
5. **Better local rules**: balance gradient sign by mean-subtracting
   L_post; use stricter eligibility normalization.
