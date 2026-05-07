# Lessons Learned (pub_v4 overnight session)

A research log of what I tried, what surprised me, and what would I do
differently next time.

## Stuart-Landau through long sequences is fragile

The cubic feedback in `β z (1 - |z|²)` produces gradient explosion
through ~300+ timesteps. Detaching `|z|²` in the SL term (so SL is
treated as a static gating in backward) is sufficient to fix it. Without
this, *all* gradients in layer 0 become NaN.

Even with detached SL, amplitude can drift unboundedly. A hard clamp
on `(u, v)` to `[-3, 3]` after the spike reset prevents runaway.

These two fixes together (detach + clamp) are enough to make BPTT
through 784 timesteps stable.

## Theta should NOT be learned

When `theta` is `nn.Parameter`, its gradient accumulates over T
timesteps. With T=784 and surrogate gradient on each step, theta's
gradient magnitude reaches `O(1e4)`. With normal LR, theta jumps from
0.7 to 50+ in one step → all neurons silent → next forward gives zero
gradient → unstable.

The simpler fix: keep theta (and gamma, beta, lambda_leak, kappa) as
*frozen buffers*. The architecture has enough capacity in the
input/recurrent matrices and omega.

## CPU is faster than MPS for tight Python loops

Per-timestep dispatch overhead dominates. CPU's lower per-op overhead
beats MPS's higher per-op overhead for this workload. ~4x speedup.

## torch.jit.script helps marginally

`torch.jit.script` on the inner step function (the rotation, SL,
recurrence apply, spike) saves ~10% per step. The Python loop overhead
remains.

`torch.compile` did not provide benefit and took >15 minutes to compile
without finishing — likely due to graph break on the autograd Function.

## Random reservoir features have signal but cap low

With 600-neuron Stuart-Landau reservoir, ridge classifier on tail-window
firing rates gets ~47% test on 5k SMNIST. Individual seeds vary
±3-5pp. The features are useful but far below what training-driven
features should give.

Pool-rate readout (no learnable head) is at chance unless trained.
Random pools do not naturally specialize by class.

## Per-class output pools is the right principle but needs training

The "each class is a resonant basin" framing is principled, but it does
NOT happen automatically. Even with a slight learnable pool_bias,
training never breaks the symmetry: the network finds it easier to fire
all pools uniformly than to specialize each pool to a class.

The auxiliary linear head is a useful capacity check. It also serves
as a "scaffold": with enough training, the network learns features
that work both for the linear head and the pool-rate readout.

For pure pool-rate readout to work, you likely need:
- Strong lateral inhibition between pools
- Class-conditional credit injection (target rate per pool per class)
- Or BPTT through enough epochs to break symmetry (probably very long)

## Local rules are 100x faster but harder to tune

Forward-only training with manual eligibility traces and Hebbian-like
updates runs at 0.5s/step vs BPTT's 11s/step. But class-pool
specialization didn't emerge in tonight's iterations.

The credit signal at the output (CE gradient on pool rates) is balanced
across classes; the feedback alignment to hidden layers via random
B matrices doesn't naturally drive specialization.

What might work:
- Sample-level credit (one update per sample, not per timestep)
- Pool-pool lateral inhibition
- Direct target rates per class

## Per-step computation cost dominates BPTT runtime

Each step has ~15 elementary ops × 784 timesteps × 5-7 cores = wall
time ~11s for batch=32. This is the floor.

To go faster: reduce ops per step, parallelize over time (impossible
for recurrence), or use a fundamentally different substrate.

## Architecture has plenty of capacity if you give it time

BPTT loss `2.33 → 1.56` over 120 batches. The trajectory is consistent
with reaching loss ~0.5 (acc ~75-80%) by epoch 5-6.

The 95% target is achievable in principle but needs ~10x more training
time than tonight's session, OR scaled-up architecture, OR a more
efficient training rule (which local rules promise but haven't
delivered yet).

## Surprises

1. Layer 0 (single recurrent layer with 1-D input) works AT ALL with
   such a tiny number of parameters (~9k). The complex projection from
   1-D pixel to 128 phases gives the network a richer representation
   than a real-valued projection would.

2. The CE gradient at the auxiliary linear head, when combined with
   the spike-only feedforward path, reaches into deep layers via
   surrogate gradient and is non-trivially informative. The spike-only
   bottleneck is limiting but not fatal.

3. ALIF (adaptive threshold) silenced the network too aggressively
   with my default settings. Tuning ALIF needs care.

## What I would build differently next time

1. **Use ALIF + simpler rotator (no SL)**: the SL nonlinearity costs
   gradient stability without obvious benefit at this scale. A rotator
   + adaptive threshold would be cleaner.

2. **Scale architecture first, then iterate**: jump straight to ~2000
   neurons, ~500k params. The current 600-neuron arch is bottlenecked.

3. **Two-stage training from the start**: fast freeze-and-train output
   first, then unfreeze for full BPTT. Avoids burning time on
   non-discriminative features.

4. **Profile + optimize forward**: 11s/step on CPU is the rate-limiter.
   Even 5s/step (2x speedup) would unlock 8 epochs/4 hours.
