# HRN-v2 Design

**Date**: 2026-05-07
**Goal**: aim for *as high as we can* on SMNIST (single-channel,
T=784) using a *biologically plausible, hierarchical resonant*
architecture, with *only* local learning rules (no BPTT, no surrogate
spikes, no backprop-through-time-disguised-as-biology). 91-92% is
acceptable; the ambition is higher.

**Approach**: pub_v3's per-neuron oscillator + eligibility-trace
substrate is the foundation, but we *blend* it with several more
powerful (but still biologically grounded) mechanisms that pub_v3
hadn't combined:

1. **Larger N**: pub_v3 used 80 neurons on T=128 burst. SMNIST needs
   ≥1024, ideally 2048+, to span the temporal-feature space.
2. **Multi-stage hierarchy** with separate frequency bands per stage,
   recursive mode decomposition.
3. **Sparse trainable intra-pool recurrence** (3-factor Hebbian) —
   gives within-pool mode binding without BPTT.
4. **Inter-stage credit via random-feedback + WCC-AF precision
   rescaling** (Lillicrap + bio95_local_rules) — lets gradient signal
   reach early stages without symmetric backward paths.
5. **DECOLLE-style per-stage auxiliary local supervision** — each
   stage gets its own loss head, so credit doesn't have to traverse
   the full hierarchy.
6. **Three-phase training (discover → nudge → lock)** — pub_v3's
   procedure adapted.
7. **Class-pool tail-energy readout** (no MLP), bigger M_out, plus a
   *trainable* per-class-pool input weight learned by local
   target-driven plasticity (not a standard linear readout).

## Core principles (non-negotiable)

1. **Computation is resonance**: every neuron is a damped complex
   oscillator with its own (ω, α, input synapses, bias).
2. **Learning is local tuning of dynamical-system parameters**:
   per-neuron eligibility traces and three-factor / Hebbian rules
   only.
3. **Memory is a stable resonant configuration** (the learned
   parameters of each oscillator + per-pool label tags).
4. **Decision-making is settling into one resonant mode over
   another**: prediction = pool whose tail energy dominates.
5. **Hierarchy by recursive decomposition**: a hard problem breaks
   into modes; each mode is handled by a pool of neurons; each pool
   recursively decomposes into sub-modes.

## What pub_v3 already showed

Per-neuron primitive (`experiments/resonant_self_organizing_layer.py`):

```
z_i(t+1) = α_i · exp(iω_i) · z_i(t) + d_i · x(t) + b_i
E_i      = mean_{t in tail} |z_i(t)|²
```

(linear damped rotation — *not* Stuart-Landau).

Layer (single resonant bank):

```
u_i = E_i − θ_i − usage_penalty_i
r_i = relu(u_i − mean_j u_j) / Σ_k relu(...)        (adaptive-mean)
P(c) = Σ_i r_i · q_i(c)                              (q learned by Hebb)
```

Update rule: per-neuron Adam on local eligibility gradients (∂E/∂{d, b,
ω, α}) modulated by a credit signal `dloss/dE`.

**Best result**: 87.0% on noisy-burst-synthetic (T=128, 10 classes).
**Open problem (pub_v3 conclusion)**: tag-mediated nudge changes
activity by 0.002–0.004 — too weak to break the single-layer ceiling.

## Why pub_v4-overnight (HRN-v1) plateaued at 67%

Wrong substrate for biology: Stuart-Landau cubic + spike emission +
W_rec + BPTT through 784 timesteps. The cubic feedback nearly
destabilizes; spike emission requires surrogate gradients to train;
BPTT is computationally expensive and biologically implausible. None
of those are necessary for the task, as pub_v3 demonstrates.

## HRN-v2 architecture

```
SMNIST input (B, T=784, 1)
        │
        ▼
Stage 0  (FAST band, ω₀ ∈ [0.05, 1.5])
   K₀ pools × M₀ neurons each
   Each neuron: linear complex oscillator, tail-energy readout
   Pool-level adaptive-mean WTA → r₀_i(t)        ∈ [0, 1]^N₀
   Per-neuron eligibility trace for (d, b, ω, α)
        │
        │ time-series r₀(t) (real-valued)
        ▼
Stage 1  (SLOW band, ω₁ ∈ [0.005, 0.20])
   K₁ pools × M₁ neurons each
   Each stage-1 neuron j receives input
       x₁_j(t) = Σ_i d₁_{ji} · r₀_i(t)
   (sparse fan-in, learned). The drive is now a slow signal because
   r₀ varies on the time-scale of pool competition (~50ms-equivalent),
   so stage 1 needs slower ω to phase-lock.
   Per-neuron eligibility for (d₁, b₁, ω₁, α₁); pool-level WTA → r₁_j(t).
        │
        ▼
Output  (CLASS band)
   N_out = 10 × M_out neurons
   Driven by r₁(t) over the full sequence (or just the tail).
   Tail-energy + per-neuron Hebbian label tags.
   P(c) = Σ_k r_k · q_k(c).
```

### Key design choices

1. **Linear complex rotation** — not Stuart-Landau. Reproduces pub_v3's
   stable substrate.
2. **Three stages, three time-scales**: ω₀ > ω₁ > ω_out. Each stage's
   "fast" input is the output of the slower stage above.
3. **Pools (intra-stage)**: K pools of M neurons. Adaptive-mean WTA
   *within* a pool. No competition across pools. This is the
   "modes within modes" structure: each pool occupies one frequency
   band, sub-tuned by its M neurons.
4. **Sparse fan-in** between stages: each stage-1 neuron sees only a
   subset (e.g., 16 of 256) of stage-0 outputs. Random initial
   sparsity, learned input weights. This implements the "modes
   become inputs to the next level".
5. **Local rule only**: per-neuron eligibility for (d, b, ω, α) +
   per-neuron Hebbian on q + homeostasis on θ. No BPTT, no shared
   gradients, no learned readout matrix.

### Inter-stage credit (the new mechanism)

The hard problem pub_v3 didn't crack: how does stage 1 send a useful
credit signal back to stage 0?

**Mechanism A — Random feedback alignment + Hebbian gating**:
- Stage-1 emits its credit δ_j (computed locally from prediction
  vs. label).
- A *random fixed* feedback matrix B (sparse, the same shape as
  d₁ᵀ) projects δ_j to stage 0: δ_i^{from-1} = Σ_j B_{ij} · δ_j.
- Stage-0 neurons whose recent activity *correlates* with their
  inherited δ get larger updates (Hebbian gating: only update if
  r₀_i(t) was high *and* δ_i was high).

**Mechanism B — Phase-coherence nudge** (the ambitious one):
- During a "nudge" phase, stage-1 neurons *project a target phase*
  back to their input stage-0 neurons:
    target_phase_i = phase(z₁_j) for the j with highest q_j(y)
    in the pool that stage-0 neuron i feeds.
- Stage-0 neurons receive a small phase-tilting drive towards this
  target during the next sequence pass. Their eligibility records
  this tilt as a positive correlation with the correct class.
- This satisfies pub_v3's observation: "the nudged phase probably
  needs to alter the recurrent phase/coherence field itself".

**Plan**: implement Mechanism A first (simpler), measure. If we
plateau, escalate to Mechanism B.

### Discover → Nudge → Lock procedure

Borrowed from pub_v3 with modification:

| Phase | Stage 0 LR | Stage 1 LR | Hebb on tags | Inter-stage credit |
|---|---|---|---|---|
| Discover | high | 0 | low | off |
| Nudge | medium | high | medium | on |
| Lock | low | medium | high | on |

The lock-phase observation in pub_v3 — "after nudge, free Hebbian
drive drops from 0.25 to 0.02" — means the discovery is *frozen* and
the nudge takes over. We replicate that.

## Experimental plan

1. **Sanity port** (Day 1): port `resonant_self_organizing_layer.py`
   to SMNIST in this workspace. Verify we hit 70-80% on SMNIST
   (subset, then full). This is the *baseline* we must match.
2. **Stage-1 added** (Day 1-2): add the slow envelope layer. Hope:
   80-87% on SMNIST.
3. **Inter-stage credit (Mechanism A)** (Day 2): add random-feedback
   credit transport. Hope: 87-90%.
4. **Inter-stage credit (Mechanism B) + lock phase** (Day 2-3): add
   phase nudge if A plateaus. Hope: 90-92%.

If 90% is reached, the system is principled, scalable, biologically
plausible, and beats pub_v3 cleanly on SMNIST.

## Files

- `src/oscillator.py` — per-neuron complex oscillator + eligibility
  traces over multi-channel input (vectorized over (B, P)).
- `src/hrn2.py` — multi-stage architecture skeleton.
- `src/local_rules.py` — adaptive-mean & softmax WTA, Hebbian tags,
  homeostasis, two credit forms (label-mass, class-pool), random
  feedback alignment.
- `src/optim.py` — pub_v3-style Adam.
- `experiments/train_stage0_smnist.py` — pub_v3 single-bank port (label-mass head).
- `experiments/train_classpool_smnist.py` — two-stage with class-pool head + random feedback to stage 0.
- `experiments/train_classpool_single.py` — single-stage with **fixed
  class-pool assignment** (each oscillator pre-assigned to a class).
  Equivalent to pub_v3's "fixed class-pool scaffold" on SMNIST.
- `experiments/train_decolle_smnist.py` — multi-stage DECOLLE-style
  (each stage has its own class-pool readout and own loss).

## Empirical findings (running ledger)

| Architecture | Train size | P_total | Epochs | Best test |
|---|---|---|---|---|
| pub_v3 label-mass head (mode-discovery + Hebbian) | 5k | 256 | 5 | 0.252 |
| Class-pool single (M_per_class=24, P=240) | 5k | 240 | 6 | 0.521 |
| Class-pool single (M_per_class=48, P=480) | 10k | 480 | 15 | 0.660 |
| Class-pool single (M_per_class=80, P=800) | 10k | 800 | 12 | 0.667 |
| DECOLLE 2-stage M=24, K1=12 | 10k | 480 | 15 | 0.764 |
| **DECOLLE 2-stage M=24, K1=12, LR decay** | **10k** | **480** | **25** | **0.829** |
| DECOLLE 3-stage M=20, K=12 | 10k | 600 | 20 | running |

Insights so far:
- **Class-pool supervision blew past label-mass mode-discovery** by
  +28pp at the same scale. The eligibility-trace credit signal from CE
  loss on class-pool tail energies is much stronger than the
  P(c)=Σ resp_i q_i(c) head used in pub_v3.
- **ω barely learns** (0.598 → 0.589 over 15 epochs) — what's training
  is mostly the input weights {d_r, d_i, b_r, b_i}, with random initial
  ω/α providing a frequency-tiled filter bank. This suggests treating
  it as a *learnable random-filter classifier* rather than expecting
  ω/α to specialize.
- **Per-step time** scales roughly linearly with P (P=480: ~17s/epoch
  on 10k; P=800: ~55s/epoch on 10k).
- **Stage-2 with full fan-in is too slow**: eligibility traces of
  shape (B, P_2, M_0) blow up. Sparse fan-in (K_1 << M_0) is
  necessary for tractable hierarchy.

## Path to higher accuracy (current plan)

1. ☑ Single-stage class-pool with bigger P (480 → 800). Establish
   single-layer ceiling.
2. ☐ DECOLLE 2-stage with K_1=24 sparse fan-in (each stage-1 neuron
   sees half same-class + half random other stage-0 outputs).
3. ☐ Increase tail window (tail=400+) so oscillator memory captures
   more of the sequence.
4. ☐ Scale to full 60k train.
5. ☐ Frequency-tiling per class pool (each class pool gets a unique
   ω sub-band) — *if* this turns out to be discriminative.

## What's not the answer

Pub_v3's resonant_self_organizing_layer IS NOT a SMNIST architecture —
its 87% was on synthetic noisy bursts where classes are
frequency-discriminable. SMNIST classes share temporal characteristics
and require **stronger credit signal** (class-pool supervision) plus
likely **larger N** and **hierarchy**.

## Path to higher accuracy (current plan)

1. Single-stage class-pool with bigger P (480 → 1024) and longer
   training. Establish single-layer ceiling.
2. Add stage-1 with sparse fan-in from stage-0 amp_seq (avoid F=N_0
   blowup). Each stage-1 neuron sees ~16-32 stage-0 outputs.
3. DECOLLE-style: each stage has its own class-pool head and own
   loss. Train each stage by its own credit only (no inter-stage
   gradient transport).
4. Add discover→nudge→lock procedure if plateau.
