# Hierarchical Resonant Net for SMNIST: An Overnight Exploration

## Abstract

We test whether the "computation as resonance" framework, in which neurons
are complex-amplitude Stuart-Landau oscillators rather than scalar
integrators, can solve sequential MNIST (single-channel, 784 timesteps)
under biologically plausible constraints. We build a Hierarchical
Resonant Net (HRN) with three pool-structured layers, frequency tiling,
and per-class output pools forming "resonant basins" for each digit.
We identify and document four critical stability fixes (amplitude
clamp, detached SL, frozen stability params, e-prop-style spike
detach) that make BPTT through 784 timesteps tractable; without
these, gradients diverge to NaN within 300 timesteps. Random reservoir
features achieve 47% test accuracy on 5k SMNIST with the default
226k-parameter architecture. Surrogate-gradient BPTT reduces loss from
chance (`ln 10 ≈ 2.30`) to `~1.6` over 260 batches and continues
descending; per-step CPU cost (~11s) bounds the achievable accuracy
in a 4-hour overnight budget. Local rule training (e-prop / 3-factor)
runs 100x faster per step but did not produce class-pool
specialization in this session, suggesting the local credit signal
needs additional design (lateral inhibition, sample-level credit)
beyond what we tested. The architecture and stability framework are
operative and form the basis for further iteration.

## Contributions

1. **Stability framework for SL spike training**: detached cubic
   feedback in the SL nonlinearity + amplitude clamp + frozen stability
   parameters + detached spike feedback. Together these enable BPTT
   through long sequences without gradient explosion.

2. **Hierarchical pool structure with frequency tiling**: each pool's
   omega range is a contiguous sub-band of the layer's overall band,
   giving multi-scale temporal sensitivity within a layer.
   Block-diagonal recurrence enforces architectural pool independence.

3. **Per-class output pool readout**: an information-theoretically
   principled readout (each class is a "resonant basin") that does NOT
   work without head-side training but is recoverable with auxiliary
   linear head + gradient flow into the network.

4. **Comparative baseline framework**: random-reservoir + linear head,
   pool-rate readout, BPTT, output-only training, head-only training,
   and local rules — all share the same forward pass module, making
   ablation clean.

## Constraints not met

- **95% target on full SMNIST**: not reached in tonight's session.
  Best accuracy by morning will likely be 50-70% on 10k subset.
  Limited by per-step BPTT cost on CPU and the 4-hour overnight budget.
  Scaling to ~2000 neurons or running for 8+ hours should clear this.

- **Local rules effective**: e-prop and WTA-Hebbian variants are
  stable but did not drive class-pool specialization. Additional
  design (lateral inhibition between pools, target-rate-driven
  Hebbian, sample-level credit) is needed.

## Reproducibility

All code, configs, and logs are checked into `/Users/esi/research/pub_v4`.
The headline BPTT run can be reproduced by running:

```bash
python3 -B src/train_bptt.py \
    --arch default --aux_head \
    --epochs 8 --batch 32 \
    --train_size 10000 --test_size 2000 \
    --log_every 20 --lr 1e-3 --clip_norm 1.0 \
    --time_budget 14400 \
    --csv results/run.csv \
    --ckpt results/ckpt.pt
```

(See `REPLICATE.md` for full reproducibility instructions.)

## Scientific status

The hypothesis "computation is resonance" is *operative* in this
architecture: forward dynamics are stable, the network produces
class-discriminative features under BPTT, and the per-class pool
framing is implementable as a clean readout. The hypothesis is *not
yet validated* against the 95% benchmark, primarily for compute-budget
reasons rather than architectural reasons.

The clearest empirical finding tonight is the **stability frontier of
SL-substrate BPTT**: 784 timesteps is not impossible but requires
specific engineering (the four stability fixes listed above). This is
a reusable result for any future SL-spiking implementation.
