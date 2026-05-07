# pub_v4 — Hierarchical Resonant Net

Built overnight (2026-05-06 → 2026-05-07) for the SMNIST 95% target with
a hierarchical/fractal resonant neural architecture under biologically
plausible constraints.

> **Read [MORNING_BRIEF.md](MORNING_BRIEF.md) first** — the
> interpretation of tonight's results changed in the second half of
> the night.

## Headline numbers

| Configuration | Test acc on full SMNIST 10k |
|---|---|
| BPTT-trained features + 2-layer MLP head (h=512) | **0.6751** |
| BPTT-trained features + linear head (200 ep, 60k train) | 0.6524 |
| Random reservoir features + linear head (200 ep, 60k train) | 0.5669 |
| Continued BPTT, in-loop pool-rate readout (2k test) | 0.5455 |
| Original BPTT (4.5h), in-loop pool-rate readout | 0.4735 |
| Random reservoir, in-loop pool-rate readout | ~0.10 (chance) |

**95% target NOT reached.** Best result is **67.51%** with offline
MLP-head readout on BPTT-trained features. BPTT *did* improve
features (+10.8pp over random init at the offline-MLP head level).
Continuing BPTT for 90 more minutes recovered the in-loop readout
to 54.55% (+7.2pp) but did NOT improve features further (offline
ceiling stayed at ~67.5%). At this 226k-param, 600-neuron scale
the bottleneck is architectural, not training-time.

See `findings/RESULTS.md` for full breakdown.

## Read first

1. `findings/RESULTS.md` — TL;DR + measured numbers.
2. `findings/DESIGN.md` — architecture rationale, stability fixes.
3. `findings/LESSONS_LEARNED.md` — what was surprising, what to avoid.
4. `findings/NEXT_STEPS.md` — concrete suggestions for the next iteration.
5. `findings/EXPERIMENT_LOG.md` — chronological detail on what was tried.

## See current state at a glance

```bash
bash scripts/morning_summary.sh
```

This prints: process status, latest training log, per-epoch CSV, and (if
the checkpoint is ready) a final test-set evaluation.

## Live watching

```bash
bash scripts/watch_progress.sh
```

## What was built

1. **`HierarchicalResonantNet` (HRN)** in `src/hrn.py`: stack of complex
   Stuart-Landau resonator pools, spike-only inter-layer communication,
   per-class output pools as a "resonant basin" readout. Heterogeneous
   `omega` per neuron with optional per-pool frequency tiling. Block-diagonal
   recurrence enforces pool independence at the architectural level.

2. **`ResonatorPoolJIT`** in `src/resonator_jit.py`: `torch.jit.script`-ed
   inner step. Vectorized over (B, K, P) with einsum for block recurrence.

3. **Stability engineering**: amplitude clamp, detached SL term, frozen
   stability params, e-prop-style detached recurrence. Without these,
   gradient explodes to NaN through 784 timesteps.

4. **Variants** (untested at scale):
   - `src/fractal_pool.py` — recursive sub-pool block structure.
   - `src/resonant_alif.py` — adaptive threshold + SL.
   - `src/rotator_alif.py` — pure rotator + ALIF (no SL).

5. **Trainers**:
   - `src/train_bptt.py` — BPTT with optional auxiliary linear head.
   - `src/train_local.py` — three-factor / e-prop local rules (no autograd).
   - `src/train_output_only.py` — freeze hidden, train output layer only.
   - `src/train_head_only.py` — frozen reservoir + linear/pool readout.

6. **Diagnostics**:
   - `scripts/diag_gradients.py` — per-param gradient flow.
   - `scripts/diag_spike_regime.py` — spike rate sweep.
   - `scripts/probe_random_features.py` — random reservoir baseline.
   - `scripts/analyze_ckpt.py` — per-class confusion + pool selectivity.

## Headline results (live — updates pending)

| Setup | Train | Test | Notes |
|-------|-------|------|-------|
| default arch random features + linear head (5k subset) | 0.50 | 0.47 | feature ceiling |
| pool-rate readout, only pool_bias trainable (5k) | 0.10 | 0.10 | random pools don't specialize |
| BPTT default + aux head (10k subset, ongoing) | tbd | tbd | overnight run |

See `findings/RESULTS.md` for live numbers.

## Reproducing

```bash
cd /Users/esi/research/pub_v4

# random feature baseline
python3 -B scripts/probe_random_features.py

# head-only on 5k
python3 -B src/train_head_only.py --arch default --train_size 5000 --test_size 1000 --head_kind both

# BPTT + aux head (overnight run config)
python3 -B src/train_bptt.py --arch default --aux_head --epochs 8 --batch 32 \
    --train_size 10000 --test_size 2000 --log_every 20 --lr 1e-3 \
    --csv results/run_overnight_default_10k.csv \
    --ckpt results/ckpt_overnight_default_10k.pt
```

## Critical engineering takeaways

1. **CPU faster than MPS** for this Python-loop-bound workload (~4x).
2. **Amplitude clamp + detached SL** are necessary for stable BPTT through
   long sequences with Stuart-Landau dynamics.
3. **Random Stuart-Landau reservoir** features get ~47% test accuracy on
   SMNIST 5k subset with default config — non-trivial but well below 95%.
4. **Local rules** (e-prop) are 100x faster per step than BPTT but need
   careful homeostasis + pool-pool competition to learn class-pool
   specialization. This is where the user's intended target lives but
   requires more iteration than tonight's session afforded.

## What's missing for 95%

- Larger network (200k params, 600 neurons → too small for SMNIST).
- More compute (even good architectures need many epochs on full 60k).
- Better local rule design (current credit signal doesn't drive
  pool-class specialization).

The architecture is *operative*: forward stable, gradient flows clean,
random features have signal. Reaching 95% is a matter of scaling and
training-time budget.
