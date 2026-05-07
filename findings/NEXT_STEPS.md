# Next Steps

**Updated 2026-05-07 04:10 EDT** after the trained-vs-random head
comparison: BPTT improves features by +8.55pp over random init when
both get a 200-epoch linear head on full 60k. The bottleneck is the
*readout*, not the BPTT training. This reorders the priorities.

## A. (highest priority) Replace the in-loop readout

The per-class pool-rate readout gets 50.07% on the trained net but a
linear head gets 65.24% on the *same* features. So the immediate
4-hour win is:

1. Make the readout learnable in-loop. Either a linear head replacing
   pool-rate, or train BOTH (use linear head for gradient, evaluate on
   pool-rate). With linear head, BPTT can push features further.
2. Hold trained features fixed and train a richer head offline (MLP,
   transformer, ridge). This is what `scripts/heads_on_trained.py`
   tests. Cheap upper-bound for what the features can do.

## B. (high priority) Continue BPTT with the new readout

If A.1 works, BPTT-from-checkpoint with linear-head readout for 4-8
hours should push significantly past 65%. Use the ckpt as warm-start
to avoid re-paying the stability-fix learning curve.

## Older priorities (still valid)

The architecture is operational and BPTT trains. To push toward 95% on
SMNIST single-channel, additional iterations:

## 1. Scale the network

The current default is ~600 neurons / 226k params. Prior pub_v1 reached
90.8% with N=8192 single-population reservoir. With the current
hierarchical architecture, scaling to ~2000 neurons should give a
meaningful jump:

```
LayerSpec(n_pools=16, pool_size=64, ...)  # 1024 neurons
LayerSpec(n_pools=16, pool_size=64, ...)  # 1024 neurons
out_pool_size=64                          # 640 output neurons
```

Total ~2700 neurons, ~5M params. BPTT will be slower (~30s/step on CPU).
A 6-hour overnight run on a 20k subset would reach ~720 steps = ~2.3
epochs.

## 2. Two-stage training

Stage 1 (fast): freeze hidden layers, train aux linear head + output
layer parameters. ~10x faster per step.

Stage 2 (slow): unfreeze and BPTT through the full network for fewer
steps.

This recovers a useful baseline before investing BPTT time.

## 3. Mini-batched truncated BPTT

Currently we BPTT through the full T=784 sequence. With grad_truncate=200
(implemented as `cfg.grad_truncate`) the gradient only flows through the
last 200 steps, which is where the readout signal is. This:
- Reduces memory by ~3.9x.
- Allows larger batches (more parallelism).
- Same step time (forward is unchanged).

Worth trying with grad_truncate=235 (matches tail_fraction=0.30 × 784).

## 4. Better local rules

The current `train_local.py` is stable but doesn't drive class-pool
specialization. Things to try:

- **Pool-pool lateral inhibition**: inject negative feedback from the
  top-firing pool to other pools at each timestep. WTA dynamics cause
  the pools to compete for representation.

- **Direct error injection at output pool**: instead of CE gradient on
  pool_rate, set targets per-timestep (correct pool fires at rate r_high,
  others at r_low) and train via target-driven Hebbian. This is more
  similar to the perceptron rule.

- **Eligibility = pre-trace × post-spike**: replace the current
  per-timestep credit with a Hebbian rule of the form `dW = sum_t pre(t) * s_post(t) * sign(L_post)`,
  where `L_post` is computed once per sample (sample-level error).

## 5. Curriculum: short sequences first

Train initially on T=196 (downsampled SMNIST or row-coded), then T=392,
then T=784. Each stage is faster and warm-starts the next. This is
biologically plausible (gradual exposure) and accelerates training.

## 6. Frequency-tile learning

Per-pool omega is initialized to a sub-band but is learnable. After
training, examine which pools' omega has shifted: have they specialized
to particular temporal frequencies of their input?

The phase-coherence rule in `compute_local_grads` (omega_grad_local)
implements a "move omega toward the dominant input frequency the neuron
phase-locks to" update. If this is enabled and tuned, omega should
self-organize.

## 7. Top-down feedback

Add a feedback path from the output (class) layer back to the mode
layer, so that during a sample, the network's "guess" influences the
mode-pool firing pattern. This is predictive-coding style and is known
to improve recurrent classification.

## 8. Architectural insight to revisit

The user's hypothesis emphasizes "modes within modes" — fractal
recursion. The `fractal_pool.py` module implements two-level block
recurrence (outer pools × inner pools × neurons). This wasn't trained
in tonight's session. Worth a controlled comparison vs. flat block
recurrence.

## Open empirical questions

1. With the same 226k-param network, can BPTT (longer training) reach
   80%+ on full SMNIST? The current trajectory suggests this is feasible
   with ~6-8 hours of training.

2. Does the per-pool frequency tiling (omega_per_pool=True) actually
   help, or is uniform omega per layer fine?

3. Can the per-class output pool readout match the auxiliary linear head
   after enough training? If yes, the principled "resonant basin" readout
   is empirically validated.

4. Does scaling N help linearly (more capacity) or sub-linearly (returns
   diminish)?

## Reproducing tonight's BPTT run

```bash
cd /Users/esi/research/pub_v4
python3 -B src/train_bptt.py \
    --arch default --aux_head \
    --epochs 8 --batch 32 \
    --train_size 10000 --test_size 2000 \
    --log_every 20 --lr 1e-3 --clip_norm 1.0 \
    --time_budget 14400 \
    --csv results/run_overnight_default_10k.csv \
    --ckpt results/ckpt_overnight_default_10k.pt \
    --device cpu
```

## Inspecting a checkpoint

```bash
python3 -B scripts/final_report.py --ckpt results/ckpt_overnight_default_10k.pt
python3 -B scripts/analyze_ckpt.py --ckpt results/ckpt_overnight_default_10k.pt --aux_head
```
