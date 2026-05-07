# Replication Guide

## Prerequisites

```bash
# Python 3.10+, PyTorch 2.6+
pip install torch torchvision
```

The data loader expects MNIST raw files at `/Users/esi/research/data/MNIST/raw/`.

## Smoke test

Verify everything imports and runs:

```bash
cd /Users/esi/research/pub_v4
python3 -B src/smnist_data.py     # loads MNIST, prints shapes
python3 -B src/resonator.py       # tests pool forward
python3 -B src/hrn.py             # tests full HRN forward
```

## Diagnostics

Check firing regime, gradient flow, and random-feature accuracy:

```bash
python3 -B scripts/diag_spike_regime.py        # spike rate sweep
python3 -B scripts/diag_gradients.py           # per-param gradient norms
python3 -B scripts/probe_random_features.py    # ridge baseline
python3 -B scripts/multiseed_random_feats.py   # multi-seed baseline
```

## Train (BPTT, default config)

```bash
python3 -B src/train_bptt.py \
    --arch default --aux_head \
    --epochs 8 --batch 32 \
    --train_size 10000 --test_size 2000 \
    --log_every 20 --lr 1e-3 --clip_norm 1.0 \
    --time_budget 14400 \
    --csv results/run.csv \
    --ckpt results/ckpt.pt \
    --device cpu
```

Available archs: `small`, `default`, `deep`, `wide`, `tiled`, `big`.

## Train (local rules, e-prop style)

```bash
python3 -B src/train_local.py \
    --arch default \
    --epochs 10 --batch 32 \
    --train_size 5000 --test_size 1000 \
    --homeo_coef 5.0 --target_rate 0.10 \
    --csv results/run_local.csv \
    --time_budget 7200
```

(Note: tonight's iterations did not converge to useful accuracy. See
`findings/LESSONS_LEARNED.md` for analysis. The trainer is fast (~0.5s/step)
and stable, but the credit signal needs more iteration.)

## Train (output layer only, hidden frozen)

```bash
python3 -B src/train_output_only.py \
    --arch default --use_aux_head \
    --epochs 30 --batch 64 \
    --train_size 10000 --test_size 2000 \
    --log_every 20 --lr 3e-3
```

Useful as a control: how good is the architecture if hidden layers are
random? Should produce ~50-60% test accuracy on 10k subset.

## Train (frozen reservoir + linear head)

```bash
python3 -B src/train_head_only.py \
    --arch default \
    --train_size 10000 --test_size 2000 \
    --head_kind both --head_epochs 100
```

Fast: caches features once, then trains the head on cached features.
Establishes the floor for what random features can achieve.

## Inspect a checkpoint

```bash
python3 -B scripts/analyze_ckpt.py \
    --ckpt results/ckpt_overnight_default_10k.pt \
    --aux_head

python3 -B scripts/final_report.py \
    --ckpt results/ckpt_overnight_default_10k.pt
```

`analyze_ckpt.py` shows per-class confusion + which output pool fires
top for each class.

`final_report.py` evaluates on the FULL 10000-sample SMNIST test set.

## Live monitoring

```bash
bash scripts/watch_progress.sh   # snapshot
bash scripts/morning_summary.sh  # summary
```

## Configuration Knobs (HRN)

In `src/hrn.py:HRNConfig`:

| Param | Effect |
|-------|--------|
| `layers[i].n_pools, pool_size` | Layer width = pools × pool_size |
| `layers[i].omega_lo, omega_hi` | Frequency band (radians/step) |
| `layers[i].omega_per_pool` | True = each pool gets its own sub-band |
| `layers[i].theta` | Spike threshold (frozen) |
| `layers[i].eta` | Input gain |
| `layers[i].in_init_scale` | D matrix scale |
| `layers[i].rec_init_scale` | W_rec matrix scale |
| `layers[i].use_recurrence` | Disable with False |
| `layers[i].block_diag` | Block-diagonal vs dense W_rec |
| `out_pool_size` | P for the K=10 class output pools |
| `tail_fraction` | Fraction of T used for output readout |
| `readout_temperature` | Multiplier on pool-rate logits |
| `use_pool_bias` | Learnable per-class bias (breaks symmetry) |
| `use_jit` | Use JIT-scripted forward (faster) |
| `aux_linear_head` | Add a learnable linear head over output features |

## Key files

```
src/
├── smnist_data.py        # SMNIST loader
├── resonator.py          # PoolConfig + reference ResonatorPool
├── resonator_jit.py      # JIT-scripted forward (faster)
├── hrn.py                # HierarchicalResonantNet
├── fractal_pool.py       # Two-level block recurrence (sub-pools within pools)
├── resonant_alif.py      # SL + adaptive threshold
├── rotator_alif.py       # Pure rotator + adaptive threshold (no SL)
├── train_bptt.py         # BPTT trainer
├── train_local.py        # e-prop / 3-factor local rule
├── train_local_v2.py     # WTA + Hebbian local rule
├── train_output_only.py  # BPTT on output layer only
├── train_head_only.py    # Frozen reservoir + linear/pool head

scripts/
├── analyze_ckpt.py            # checkpoint diagnostics
├── final_report.py            # final eval on full test
├── morning_summary.sh         # overnight summary
├── watch_progress.sh          # live progress
├── diag_*.py                  # firing regime, gradient flow
├── probe_random_features.py   # ridge baseline
├── multiseed_random_feats.py  # multi-seed baseline

findings/
├── DESIGN.md           # architecture rationale
├── EXPERIMENT_LOG.md   # what was tried, chronological
├── RESULTS.md          # measured numbers
├── LESSONS_LEARNED.md  # surprises and takeaways
├── NEXT_STEPS.md       # concrete next iteration suggestions
├── RELATED_WORK.md     # background and citations
```
