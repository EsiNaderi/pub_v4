# Related Work / Background

Context for the choices in pub_v4. Each entry: what was used, what the
literature says, where to look for ideas.

## Stuart-Landau resonator as the substrate

The SL equation `dz/dt = (γ + iω) z + β z (1 - |z|²)` is the canonical
weakly-nonlinear oscillator, with a stable limit cycle at `|z|² = -γ/β`
when `γ < 0, β > 0`. It is widely used in computational neuroscience as
a normal-form model for cortical oscillations.

For a discrete-time spiking version (the "Resonance Network" in
`/Users/esi/research/resonance/DESIGN.md`):
- `z` is the rotor amplitude.
- `q` is the pre-spike state (one Euler step ahead).
- `s = H(|q|² - θ)` is the spike.
- `z ← (1 - κ s) q` is the reset.

This is the "wave-particle" interface: smooth complex evolution between
spikes; non-unitary contraction at spike events.

Reference: any cortical-oscillator paper, e.g.
- Pikovsky, Rosenblum, Kurths, *Synchronization* (CUP, 2001).
- Aoi, Lepage et al, "Influence-balanced spike-coupled oscillators..."
  Frontiers Computational Neuroscience.

## Surrogate gradient through binary spikes

Heaviside step function `H(x)` has gradient zero almost everywhere.
Surrogate gradient methods replace `H'` with a smooth approximation:
- Fast sigmoid: `1 / (1 + slope*|x|)² · (slope/2)` (Zenke & Ganguli 2018).
- Rectangular: `1{|x| < w} / (2w)`.
- Sigmoid derivative: `sigmoid(x)(1-sigmoid(x))`.

We use fast sigmoid with slope=2.5. Lower slope = wider gradient region
= more spikes contribute to the gradient.

Reference:
- Neftci, Mostafa, Zenke, "Surrogate Gradient Learning in Spiking Neural
  Networks." IEEE SPM, 2019.

## Block-diagonal recurrence as pool structure

Rather than full N×N recurrence, we use K independent P×P blocks. Each
"pool" is then a separately-coupled subnetwork. This:
1. Reduces parameters by factor K (N²/(K·P²) = N/P).
2. Enforces architectural pool boundaries.
3. Makes the einsum `(B,K,P) × (K,P,P) → (B,K,P)` very efficient.

Reference: This shows up implicitly in many SNN papers but isn't named.
The closest is the modular-recurrence/structured-sparsity literature
(e.g., Hopfield modules).

## e-prop / 3-factor / forward eligibility

Bellec et al., "A solution to the learning dilemma for recurrent networks
of spiking neurons" (Nature Communications, 2020). Eligibility traces
record local pre-post correlation; learning signals from above gate
when those traces become weight changes.

Three-factor formulation:
```
e(t)        — eligibility trace (local, pre-post product)
L(t)        — learning signal (post-synaptic credit, top-down)
ΔW(t)       = e(t) · L(t)
```

Our `train_local.py` implements this with random feedback alignment for
the inter-layer credit propagation.

## Random feedback alignment

Lillicrap et al., "Random synaptic feedback weights support error
backpropagation for deep learning" (Nature Communications, 2016). Random
fixed feedback matrices `B` can replace the transpose `W^T` in the
backward pass with surprisingly little degradation, given enough training
time.

Used in our local-rule trainer to propagate the output-layer credit to
hidden layers without needing exact transposes.

## Adaptive LIF (ALIF)

Bellec, Salaj, Subramoney et al., "Long short-term memory and learning-to-
learn in networks of spiking neurons" (NeurIPS, 2018). LSNN: LIF +
adaptive threshold (ALIF). Adaptive threshold rises with each spike and
decays with time constant tau_adapt. This produces spike-frequency
adaptation, which gives the network a "long-term memory" component
without explicit gating.

Our `src/resonant_alif.py` and `src/rotator_alif.py` add ALIF on top of
SL/rotator dynamics. Untested in tonight's session.

## Per-class pool readout (resonant basin)

The "each class is a resonant basin" framing has roots in attractor
network theory (Hopfield), where each memory is a fixed point. Modern
spiking variants:
- Diehl & Cook 2015, "Unsupervised learning of digit recognition using
  STDP" — uses competitive Hebbian + lateral inhibition + per-pool
  assignment. Reaches 95% on static MNIST (NOT sequential).

For SMNIST (sequential), the per-pool assignment must hold across the
temporal extent of the input. The "tail-window mean spike rate per pool"
readout is one way to time-average; alternatives include peak detection
or last-spike timing.

## SMNIST benchmarks

Sequential MNIST (one pixel per timestep, 784 timesteps, 10 classes):
- LSTMs: 99%
- Simple RNNs: 94-95%
- LSNN (LIF + ALIF + e-prop): 92-94%
- pSMNIST (permuted): 95% with LSTMs, 90-92% with SNNs

Most BPTT-trained SNN approaches use:
- Heterogeneous time constants
- Surrogate gradient
- ~1000-4000 neurons
- Many epochs (50+)

Our overnight run targets the same regime but is constrained by per-step
CPU cost.

## CCCP-Ω (from prior pub_v3)

Concave-Convex Procedure adapted to per-neuron rotors. The local rule
moves omega toward a class-conditional prototype frequency. Worked in
spirit for the resonance network but couldn't discriminate classes via
end-of-sequence prototype because the SL limit cycle washes out class-
specific phase by t=T.

This insight motivated our tail-window mean rate readout (more robust
to limit-cycle settling).

## Equilibrium propagation

Scellier and Bengio, "Equilibrium Propagation: Bridging the gap between
energy-based models and backpropagation" (Frontiers Computational
Neuroscience, 2017). Two-phase contrastive learning: free phase, then
nudged phase, with ΔW = (post_nudged - post_free) · pre.

Adapted in `gpt-solidification` for spiking. Could be an alternative to
e-prop for the resonance substrate. Not implemented in pub_v4.

## Predictive coding

Whittington and Bogacz 2017. Local Hebbian updates with iterative error
inference can approximate backprop. Could apply to the HRN by adding a
top-down feedback path from output to mode pools.

## What we built on, what's new in pub_v4

Built on:
- Stuart-Landau spiking substrate (from `/Users/esi/research/resonance`).
- Block-diagonal pool structure (new in pub_v4).
- Per-pool frequency tiling (new).
- Per-class output pool readout (variant of class-pool ideas in pub_v3).
- Stability fixes for BPTT through 784 timesteps (new in pub_v4).

New in pub_v4 (untested at scale):
- Fractal pool architecture (`fractal_pool.py`).
- Resonant + ALIF hybrid (`resonant_alif.py`, `rotator_alif.py`).
- e-prop-style local trainer with mean-subtracted feedback (`train_local.py`).
- Winner-take-all + Hebbian local trainer (`train_local_v2.py`).
