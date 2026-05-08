# Overnight spiking SMNIST notes (2026-05-07 → 2026-05-08)

## Findings during the night

### 1. Subtractive reset breaks oscillator dynamics

LIF/ALIF neurons benefit from a subtractive reset on spike
($v \leftarrow v - v_{\text{th}}$): it implements the "consume-and-fire"
budget that's central to integrate-and-fire models.

For our resonant oscillator $z(t+1) = \alpha\, e^{i\omega} z(t) + d\, x(t) + b$,
the same reset $z \leftarrow (1-\kappa)z$ on spike *destroys the phase*
that the oscillator carries. The resonance IS the computation, and the
phase is half of the message. Multiplicative shrink halves the
amplitude but ALSO multiplies the phase vector — which is wrong if
the goal was to "consume" only the magnitude.

Empirically: with $\kappa = 0.5$, training collapses to ~25% test
accuracy (chance ~10%). With $\kappa = 0.0$, training works as before.
Even after correctly fixing the eligibility-trace recurrence to
account for the reset (see below), behaviour did not recover.

**Lesson**: the LIF "subtract on fire" reset is incompatible with
oscillatory subthreshold dynamics. A future bio-faithful spike-
emission rule for resonators should preserve phase — perhaps a
*radial* reset $|z| \leftarrow (1-\kappa)|z|$ (shrinks amplitude,
preserves phase angle) — but I did not have time to test that
tonight.

### 2. The eligibility-trace recurrence MUST account for reset

If you do choose to use any form of reset, the eligibility-trace
recurrence needs to track it. Specifically, after an emitted spike
$s(t)$ scales the post-step state by $(1-\kappa s(t))$, the next-step
trace evolves as
$$
e_{\theta}(t+1) = R_\alpha(\omega)\,(1-\kappa s(t))\, e_{\theta}(t) + \frac{\partial}{\partial\theta}\!\left[\text{drive}(t+1)\right].
$$
Without the $(1-\kappa s)$ factor, the trace represents
$\partial \tilde z(t)/\partial\theta$ for the *un-reset* state and
no longer matches what $z(t)$ actually is. Empirically this leads
to silent gradient drift over many timesteps with high $\alpha$.

I added this factor to `oscillator_spiking.py` for the all four
parameter eligibilities and the optional intra-stage and top-down
recurrent eligibilities. Mathematically correct now; just doesn't
help on its own because the reset is the wrong primitive for
oscillators (finding 1).

### 3. Intra-stage recurrence destabilises deep, high-$\alpha$ stages

With $\alpha \approx 0.999$ the integrator gain is
$1/(1-\alpha^2) \approx 500$. Even very small recurrent input
(`rec_init = 0.005`, $K_{\text{rec}} = 6$) produces enough variance
to push amplitudes above the spike threshold over 200+ tail steps.
Stage 2 saturates to ~100% firing rate and stops carrying
information.

Negative `init_bias` (lateral inhibition) didn't help — at
initialisation no spikes have happened yet, so the bias has no
effect on first-pass dynamics. By the time spikes start arriving
the network is already in an unstable regime.

**Lesson**: intra-stage recurrence in resonant SNNs needs
either (a) very low $\alpha$ in the recurrent stage, (b) explicit
inhibitory connections that activate *immediately* (not just from
spike-driven inhibition), or (c) a homeostatic mechanism with
faster time-constant than the recurrence itself. A simple random
init plus rate-homeostasis on $\theta$ is not sufficient.

### 4. Pure depth scaling (5 stages) IS stable and helps

A 5-stage hierarchy with no reset and no intra-stage recurrence
trains cleanly. Final result on SMNIST 10k:

| Stage | Test acc alone (binary spike) |
|---|---|
| 0 (fast band) | 0.379 |
| 1 | 0.688 |
| 2 | 0.788 |
| **3** | **0.811** |
| 4 (slowest band) | 0.789 |
| Ensemble | **0.8135** |

This is the headline spiking SMNIST number for tonight: **81.35%** with
binary spike count readout, comparable to literature SNN baselines on
SMNIST and only ~8.5pp below the analog 3-stage HRN-v2 (89.9%).

Each stage's $\omega$ band is strictly slower than the one above it;
the cascade of envelope-detector stages compresses the 784-step pixel
sequence into progressively slower features. Stage 3 alone (band
$\omega \in [0.0002, 0.04]$) is the strongest individual stage.

This is the *clean* depth-scaling story: the local-rule + spike-only
architecture works at depth, no special tricks needed.

## Recommended writeup for the SMNIST spiking result

* Architecture: 5 stages of damped complex linear oscillators,
  per-stage class-pool tail-rate readout, DECOLLE-style local
  cross-entropy per stage, ensemble of stage softmaxes.
* Per-neuron forward eligibility for $\{d_r, d_i, b_r, b_i, \omega, \alpha\}$.
* Spike emission: stochastic Bernoulli of $\sigma(\beta(|z|^2 - \theta_i))$,
  homeostatic threshold targeting per-step rate ~10%.
* Inter-neuron communication: binary spikes only.
* No BPTT, no surrogate of Heaviside, no inter-stage backward pass.

## Negative results to record

* Subtractive reset $z \leftarrow (1-\kappa)z$ on spike: breaks training.
* Intra-stage recurrence with random $\pm$ small init: causes runaway
  saturation in deep high-$\alpha$ stages.
* Negative-mean recurrent init: did not stabilise (worse than zero-mean).

## Outstanding ideas not yet tested tonight

* **Radial reset** ($|z| \leftarrow (1-\kappa)|z|$, preserve phase) —
  the bio-faithful reset for an oscillator.
* **Inter-stage top-down feedback** (predictive-coding style): each
  stage's pass-2 forward receives spikes from stage L+1's pass-1
  spike train. Code is implemented; not yet run.
* **Structured inhibitory recurrence**: explicit subtraction of
  pool-mean activity instead of random recurrent weights.
* **Spectral readout**: FFT of spike train over tail window, magnitude
  at characteristic frequency as the neuron's feature.
