# DVS Local Spiking Convolution Prototype

This folder is intentionally separate from `hrn2/` and `shd/`.

Goal: test whether the HRN-v2 spiking oscillator idea can be extended to
a spatial DVS front end without introducing backpropagation, surrogate
Heaviside gradients, or weight transport.

Current contents:

- `src/conv_oscillator_spiking.py`: strict spiking convolutional
  oscillator layer with shared convolutional weights and manual
  forward eligibility traces.
- `src/local_readout.py`: manual class-pool readout and cross-entropy
  credit. No autograd.
- `src/optim.py`: manual Adam update over explicitly computed local
  gradients.
- `src/dvsgesture_data.py`: local DVS Gesture frame-cache loader using
  the already available tonic-compatible dataset on this machine.
- `experiments/train_synthetic_dvs_conv.py`: synthetic moving-edge
  DVS-like smoke experiment.
- `experiments/train_dvsgesture_conv_local.py`: first real DVS Gesture
  benchmark path using the same strict local rule.
- `METHOD.md`: mathematical rule and constraints.
- `RESULTS.md`: synthetic smoke result and real DVS Gesture benchmark.

This is now a minimal DVS benchmark implementation, but not yet a strong
architecture. It intentionally starts with a single local spiking
convolutional layer and a fixed class-pool readout.

Important distinction:

- The earlier `100%` result was only a synthetic moving-edge smoke test.
- The real DVS Gesture benchmark result from this prototype is
  **32.58% binary-spike accuracy** on the 11-class test split.

In this folder, "binary" means evaluation from sampled spikes
`s_t in {0, 1}` only, not continuous spike probabilities.
