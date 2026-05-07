# pub_v4 Design

## Goal

Demonstrate "computation is resonance" on SMNIST with 95%+ accuracy using
a biologically plausible, hierarchical resonant neural architecture and
local learning rules.

## Hypothesis Recap

> Synapses are not static scalar weights — they are dynamical systems.
> Neural computation is fundamentally oscillatory. Oscillatory dynamics
> coexist with discrete spike events. Frequency, phase, and resonance are
> learnable representations. Learning emerges through tuning coupled
> oscillators. **Computation is resonance.**

## Core Substrate: Stuart-Landau Resonator

Each neuron is a complex-amplitude oscillator with a threshold-on-amplitude
spike rule:

```
q(t)   = (1-λ) [exp(iω) z(t-1) + β z(t-1) (1 - |z(t-1)|²) - γ z(t-1)
                + W_rec @ s(t-1) + η D x(t)]
s(t)   = H(|q(t)|² - θ)              (with surrogate gradient)
z(t)   = (1 - κ s(t)) q(t)            (spike contraction)
```

State per neuron: `z = u + iv` (complex amplitude). 7 dynamical params:
omega (resonant frequency), gamma (damping), beta (SL nonlinearity),
lambda (leak), kappa (spike contraction), theta (threshold), eta (input
gain).

The Stuart-Landau term `β z (1 - |z|²)` creates a stable limit cycle at
`|z|² = 1 - γ/β`. Without input, the neuron oscillates at amplitude
`√(1-γ/β)` with phase advancing at rate omega per timestep. With input,
amplitude exceeds the limit cycle and crosses threshold, emitting a spike.

The reset `(1 - κ s) q` contracts the rotor at spike time — this is the
non-unitary "measurement" event in the wave-particle duality reading.

## Hierarchical Pool Structure

A network is L layers of pools. Each layer is K independent pools of P
neurons. The recurrent matrix `W_rec ∈ ℂ^{N×N}` is **block-diagonal**:
each pool has its own `P × P` complex coupling, with zero coupling
between pools.

Inter-layer communication is via **spikes only** (R2 audit boundary).

```
Input x(t)  →  Layer 0 (sensory pools, high freq)
                       ↓ s_0(t)
              Layer 1 (mode pools, mid freq)
                       ↓ s_1(t)
              Output  (class pools, slow freq) — K=10, one per class
                       ↓ tail-window mean spike rate per pool
              Logits
```

Per-pool frequency tiling: each pool's `omega` band is a contiguous slice
of the layer's overall `[omega_lo, omega_hi]` range. This implements
multi-scale temporal sensitivity within a layer.

## Per-Class Output Pools (Resonant Basins)

The output layer has exactly `K = n_classes` pools (one per class), each
with `P = out_pool_size` neurons. The readout is:

```
pool_rate[c] = mean over (tail timesteps × pool neurons) of s_out
logits[c]    = (pool_rate[c] - mean_c pool_rate) * temperature + bias[c]
```

Conceptually, each class is a "resonant basin": when the network sees
a digit of class `c`, the pool of neurons indexed by `c` should fire
preferentially in the tail of the sequence. Other pools should stay
silent.

This is the **non-fingerprint** readout: information must concentrate
in *which pool* fires, not in arbitrary patterns of individual neurons.

## Stability Engineering

Three modifications were essential to make BPTT through 784 timesteps
work without NaN gradient explosion:

1. **Amplitude clamp**: After reset, `u, v` are clamped to `[-3, 3]`.
   Prevents runaway amplitude from compounding over long sequences.

2. **Detached `amp_sq` in SL term**: The Stuart-Landau saturation term
   `β z (1 - |z|²)` is computed with `|z|².detach()`. This removes the
   cubic feedback loop in the backward pass while keeping the forward
   dynamics correct. Without this, gradient through the SL term
   explodes through ~300+ timesteps.

3. **Frozen stability params**: `theta`, `gamma`, `beta`, `lambda_leak`,
   `kappa` are kept as buffers (not trainable). Their gradients
   accumulate over T timesteps to magnitudes ~1e4, destabilizing
   training. The architecture has enough capacity in `D`, `W_rec`,
   `omega`, `eta` alone.

4. **e-prop style detached recurrence**: `s_prev.detach()` before
   `W_rec @ s_prev`. This breaks the gradient through the spike
   feedback path, leaving only forward gradient through `u, v`. Reduces
   gradient explosion further.

## Local Learning (target, in progress)

Three-factor / e-prop style:
- Per-layer eligibility traces of presynaptic activity
- Top-local CE gradient at output pools
- Random feedback alignment (`fb_proj`) to deeper layers
- Mean-subtracted post-credit so gradient sum is zero (no positive bias)
- Homeostatic threshold (target rate) to prevent rate runaway

Implementation: `src/train_local.py`. ~100x faster than BPTT (no autograd
graph through 784 timesteps). Stable forward, stable rates with
homeostasis. Class-pool selectivity is the open challenge: the local
credit signal hasn't yet driven pool-class specialization.

## Files

- `src/smnist_data.py` — SMNIST loader (single-channel, 784 timesteps).
- `src/resonator.py` — pool config and reference implementation.
- `src/resonator_jit.py` — JIT-scripted forward (faster).
- `src/hrn.py` — Hierarchical Resonant Net.
- `src/fractal_pool.py` — fractal sub-pool variant.
- `src/resonant_alif.py` — Resonant + ALIF variant.
- `src/rotator_alif.py` — pure rotator + ALIF (no SL).
- `src/train_bptt.py` — BPTT trainer with optional aux head.
- `src/train_local.py` — local-rule trainer.
- `src/train_output_only.py` — output-layer only fine-tune.
- `src/train_head_only.py` — frozen reservoir + linear/pool head.
- `scripts/diag_*.py` — diagnostics (gradient flow, firing regime).
- `scripts/probe_random_features.py` — random reservoir baseline.
- `scripts/analyze_ckpt.py` — per-class diagnostic on a trained model.

## Open Directions

1. **Larger network**: pub_v1 used N=8192 for 90.8% on SMNIST. My
   current default has ~600 neurons. Scaling up should help, modulo
   training time.

2. **Multi-stage training**: pretrain layer 0 unsupervisedly (Hebbian),
   then layer 1, then output. Reduces BPTT depth and is more
   biologically plausible.

3. **Better local rule**: the e-prop credit signal needs to drive
   pool-class specialization. Possible fixes: lateral inhibition
   between pools (winner-take-all), explicit class-conditional credit,
   or direct error injection at the output pool.

4. **Phase-coupled gating**: replace simple `W_rec @ s` with a
   phase-coherent gate: `gate = cos(omega_post - omega_pre) > threshold`.
   This makes routing via synchronization, not just via spike count.

5. **Top-down feedback**: include feedback from class pools to mode
   pools to drive predictive synchronization.

## Lessons Learned

- Stuart-Landau dynamics need careful stability engineering for BPTT
  through 784 timesteps (amplitude clamp + detached SL).
- Per-class pool readout needs supervised training; random pools do
  not naturally specialize by class.
- Random reservoir features get ~47% test accuracy on SMNIST 5k
  (default 600-neuron arch). This is the floor; training improves.
- Forward computation is dominated by the 784-step Python loop; CPU
  is faster than MPS due to Python dispatch overhead. JIT helps a bit.
- Local rules are 100x faster per step than BPTT but need careful
  homeostasis to avoid rate explosion.
