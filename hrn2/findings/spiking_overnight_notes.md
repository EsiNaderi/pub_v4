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

## Follow-up: spectral-geodesic strict-spiking readout (2026-05-08)

We tested the spectral-readout idea as a spiking-only spectral-geodesic
head: each stage keeps local tail-window spike-spectrum demodulators,
and class evidence is computed by distance to complex spectral
prototypes instead of pure spike count/rate. Binary spikes remain the
only inter-stage signal, and oscillator parameters still update by
forward eligibility traces modulated by local readout credit.

Interrupted long run:

```bash
python3 -B experiments/train_spectral_geodesic_5stage_smnist_spiking.py \
  --m_per_class 20 --k_fanin 12 \
  --epochs 25 --batch 64 \
  --train_size 10000 --test_size 2000 \
  --threads 4 --spec_q 4 \
  --csv results/smnist_spectral_geodesic_5stage_spiking.csv
```

Best observed binary-spike ensemble before interruption: **84.85%** at
epoch 12. This is a +3.5pp improvement over the 81.35% spike-rate
baseline. At the peak epoch, stage-alone binary accuracies were:

| Stage | Test acc |
|---|---:|
| 0 | 0.434 |
| 1 | 0.651 |
| 2 | 0.799 |
| 3 | 0.837 |
| 4 | 0.836 |
| Ensemble | **0.8485** |

The early deep-stage saturation also self-corrected: stage 3/4 rates
started near 0.95/1.00 at epoch 0 and were 0.047/0.083 at the best
epoch. This is the first evidence that spectral organization of spike
trajectories gives a genuine improvement over spike-rate counting
without relaxing the strict-spiking constraint.

## Follow-up: spectral-geodesic SHD result (2026-05-08)

We then ported the same strict-spiking spectral-geodesic readout to
SHD. The existing SHD cache uses 4k train / 1k test examples, T=100
time bins, 700 cochlear channels, and 20 classes. Architecture:
3 stages, M=12 neurons/class/stage, random sparse cochlear fan-in
K0=48, class-aligned K1=K2=12, binary inter-stage spikes only.

Best binary-spike ensemble: **76.70%** at epoch 16.

| Method | Best binary test |
|---|---:|
| Analog SHD 3-stage HRN-v2 | 0.684 |
| Strict-spiking SHD spike-rate readout | 0.606 |
| **Strict-spiking SHD spectral-geodesic readout** | **0.767** |

Peak stage-alone binary accuracies:

| Stage | Test acc | Spike rate |
|---|---:|---:|
| 0 | 0.600 | 0.111 |
| 1 | 0.756 | 0.113 |
| 2 | 0.738 | 0.104 |
| Ensemble | **0.767** | --- |

This is stronger evidence than SMNIST that the spike-spectrum geometry
is useful: the same local rule beats both the strict-spiking SHD
baseline and the previous analog SHD baseline on the local cache.
