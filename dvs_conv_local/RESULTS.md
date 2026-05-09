# Results

Date: 2026-05-08

Only **binary-spike accuracy** is considered the real spiking result.
Smooth/probability accuracy is logged only as a diagnostic.

## Synthetic DVS Moving-Edge Smoke Test

Command:

```bash
python3 dvs_conv_local/experiments/train_synthetic_dvs_conv.py \
  --epochs 8 --threads 4 \
  --csv results/synthetic_dvs_conv_local.csv
```

Result:

| Epoch | Smooth test | Binary-spike test | Mean spike rate |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0078 | 0.0416 |
| 1 | 1.0000 | 1.0000 | 0.2673 |
| 8 | 1.0000 | 1.0000 | 0.1694 |

Interpretation:

- The convolutional local-rule machinery works mechanically.
- The synthetic task is intentionally easy; this is not a DVS benchmark.
- The `100%` number here must not be compared with DVS Gesture or any
  real event-camera benchmark.
- The mean spike rate is high after learning, so a real DVS experiment
  should add stronger sparsity/homeostasis before scaling.
- Training used manually computed class-pool credit and manually
  contracted convolutional eligibility traces. There was no autograd
  training path, no `.backward()`, no `torch.optim`, and no transported
  downstream weight matrix.

## DVS Gesture Benchmark

Dataset:

- DVS Gesture, 11 classes.
- Local tonic-compatible data already existed at
  `/Users/esi/research/biosuccess/data/DVSGesture`.
- Train/test split from tonic: 1077 train examples, 264 test examples.
- Frames were cached locally as binary/clipped tensors:
  `dvs_conv_local/data/cache/dvsgesture_T8_S24_binary.pt`.
- Frame shape: `(N, T=8, polarity=2, H=24, W=24)`.

Architecture:

```text
DVS frames
  -> one strict local spiking convolutional oscillator
  -> fixed class-pool readout
```

There is no learned classifier head and no second spiking stage.

Command:

```bash
python3 dvs_conv_local/experiments/train_dvsgesture_conv_local.py \
  --epochs 12 --batch 8 \
  --time_bins 8 --spatial 24 \
  --m_per_class 3 --kernel 3 \
  --input_scale 0.30 \
  --theta_init 0.40 --target_rate 0.08 --theta_lr 0.06 \
  --lr 0.008 --temperature 12 \
  --threads 4 \
  --csv results/dvsgesture_conv_local_T8S24_m3.csv
```

Result:

| Epoch | Smooth test | Binary-spike test | Best binary | Mean spike rate |
|---:|---:|---:|---:|---:|
| 0 | 0.0947 | 0.0682 | 0.0682 | 0.0397 |
| 1 | 0.1932 | 0.1856 | 0.1856 | 0.1070 |
| 4 | 0.2235 | 0.2121 | 0.2121 | 0.1291 |
| 6 | 0.2689 | 0.2689 | 0.2689 | 0.1505 |
| 9 | 0.3106 | 0.3182 | 0.3182 | 0.1617 |
| **10** | **0.3182** | **0.3258** | **0.3258** | 0.1622 |
| 12 | 0.2614 | 0.2614 | 0.3258 | 0.1742 |

Best real benchmark result:

```text
DVS Gesture binary-spike test accuracy: 32.58%
```

Interpretation:

- This is above chance for 11 classes, but not competitive.
- It is the first real benchmark result for the strict local
  convolutional oscillator prototype.
- The result is much lower than previous non-backprop DVS Gesture
  systems in other folders because this prototype intentionally uses
  only one local conv layer and no learned readout matrix.
- The late drop after epoch 10 suggests the current homeostasis/sparsity
  controls are too weak or the single-layer class-pool readout saturates.

## Current Conclusion

The strict spiking convolutional local-rule machinery works and can run
on a real DVS benchmark without backpropagation or weight transport.
The current architecture is too shallow for DVS Gesture. The next
benchmark-relevant step is a second local spiking stage or a local
spectral/geodesic readout over binary spike maps, while keeping the
binary-only evaluation.
