# Method

Date: 2026-05-08

## Layer Dynamics

The convolutional spiking oscillator layer receives event frames:

```text
x(t) in R^{B x C_in x H x W}
```

Each output channel and spatial position keeps a complex oscillator:

```text
z_{o,y,x}(t+1)
  = alpha_o exp(i omega_o) z_{o,y,x}(t)
    + (W_o * x(t))_{y,x}
    + b_o
```

where `*` is a shared spatial convolution. The convolutional kernel is
shared over all spatial positions, so this is a real convolutional
layer, not flattened random fan-in.

The spike probability is:

```text
p_{o,y,x}(t) = sigmoid(beta (|z_{o,y,x}(t)|^2 - theta_o))
```

The emitted spike used for binary evaluation is sampled:

```text
s_{o,y,x}(t) ~ Bernoulli(p_{o,y,x}(t))
s_{o,y,x}(t) in {0, 1}
```

## Readout

The current benchmark uses a fixed class-pool readout. Output channels
are assigned to classes, and logits are:

```text
logit_c = temperature * mean_{o in class c, y, x} rho_{o,y,x}
rho_{o,y,x} = mean_t p_{o,y,x}(t)
```

This readout has no learned classifier matrix.

For reporting, the only score that matters is the binary-spike score:

```text
spike_rate_{o,y,x} = mean_t s_{o,y,x}(t)
binary logit_c = mean_{o in class c, y, x} spike_rate_{o,y,x}
```

The smooth/probability score is logged only as a diagnostic.

## Learning Rule

The local update is a three-factor eligibility rule:

```text
Delta theta = -eta sum_{b,o,y,x} credit_{b,o,y,x}
                            * d rho_{b,o,y,x} / d theta
```

The eligibility trace is computed forward in time:

```text
e_theta(t+1)
  = alpha_o exp(i omega_o) e_theta(t)
    + local input term
```

For a shared convolutional kernel weight, the local input term is the
corresponding image patch value. The shared kernel gradient sums local
credit times local eligibility across batch and spatial positions.

The readout credit is derived analytically from local class-pool
cross-entropy:

```text
credit = d L_class_pool / d rho
```

This is supervised local gradient credit. It is not unsupervised
Hebbian learning.

## What Is Excluded

The implementation deliberately excludes:

- BPTT through time.
- Backpropagation through multiple layers.
- Surrogate derivative through sampled binary spikes.
- Transported downstream learned weights.
- Learned classifier/readout matrix.
- `torch.optim` or `.backward()` based training.

The important mathematical exclusion is not the absence of `.backward()`
alone. The important exclusion is that no term of the form

```text
delta_l = W_{l+1}^T delta_{l+1}
```

is used, and no downstream learned weights are carried back to earlier
layers.

## Current Limitation

The current real benchmark is only:

```text
DVS Gesture frames
  -> one strict local spiking convolutional oscillator
  -> fixed class-pool readout
```

That is intentionally conservative. It proves that the local
convolutional rule can run on a real DVS benchmark, but it is not yet a
competitive DVS architecture.
