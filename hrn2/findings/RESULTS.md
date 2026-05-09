# HRN-v2 Results — SMNIST 10-way

Date: 2026-05-07

## Headline result

**89.90% test accuracy** on SMNIST 10-way (10k train / 2k test) with a
fully biologically plausible 3-stage hierarchical resonant network
trained by *local* eligibility-trace rules — NO BPTT, NO surrogate
gradients, NO inter-stage backward pass.

This is a 65pp improvement over the pub_v3-style label-mass head
on the same dataset (25%), and demonstrates that the recursive
resonance approach can solve SMNIST competitively under bio-plausible
constraints.

## Headline numbers (10k train / 2k test SMNIST 10-way)

| Architecture | Local rule | Best test | Notes |
|---|---|---|---|
| Pub_v3 label-mass head (mode-discovery + Hebbian) | full pub_v3 recipe | 0.252 | Matches pub_v3's smnist_class_pool_contrast (22-25%) |
| Class-pool single, P=480 | eligibility on (d, b, ω, α) | 0.660 | 15 epochs, no hierarchy |
| Class-pool single, P=800 | same | 0.667 | scaling N alone gives marginal gain |
| DECOLLE 2-stage, M=24/class, K1=12, 15 ep | per-stage local credit | 0.764 | hierarchy unlocks > 70% |
| **DECOLLE 2-stage**, **25 ep with LR decay at 14** | | **0.829** | LR decay stabilizes stage 1 |
| **DECOLLE 3-stage**, **M=20/class, 20 ep with LR decay at 12** | | **0.866** | 3rd stage adds +3.7pp |
| **DECOLLE 3-stage**, **M=24/class, 35 ep, 2 LR decays at 12 & 24** | | **0.899** | second LR decay added +3pp |
| **Strict-spiking 5-stage**, **binary spike-rate readout** | spike Bernoulli + local eligibility | **0.8135** | no reset, no recurrence; previous strict-spiking headline |
| **Strict-spiking spectral-geodesic 5-stage**, **interrupted at ep 13** | spike-spectrum prototypes + local eligibility | **0.8485** | binary-spike ensemble peak at ep 12; +3.5pp over spike-rate readout |
| **Strict-spiking spectral-geodesic 5-stage MIXTURE K=4**, **stopped at ep 23** | mixture-prototype geodesic + local eligibility | **0.8630** | best at ep 15; +1.45pp over single-prototype baseline |
| **SHD strict-spiking 3-stage**, **binary spike-rate readout** | spike Bernoulli + local eligibility | **0.606** | 4k/1k SHD cache |
| **SHD strict-spiking spectral-geodesic 3-stage** | spike-spectrum prototypes + local eligibility | **0.767** | 4k/1k SHD cache; +16.1pp over spike-rate SHD |

## What works

The single-stage class-pool resonant net plateaus near 67%. **Hierarchy
is the key mechanism** that unlocks higher accuracy:

1. Each stage is a population of damped complex linear oscillators
   (pub_v3's substrate, no Stuart-Landau cubic, no spike emission).
2. Each stage has 10 class-pools × M oscillators per class.
3. Each stage gets its own class-pool tail-energy logits and its own
   cross-entropy loss (DECOLLE-style local supervision).
4. Each stage's parameters update via eligibility-trace local credit
   (no BPTT, no surrogate gradients).
5. Stage L+1's input is stage L's amplitude trajectory, with sparse
   fan-in (each stage L+1 neuron sees K of the M_L outputs from
   stage L; half from the same class pool, half from random other
   classes).
6. Each stage operates at a slower frequency band (ω) than the one
   below it.
7. LR decay (epoch 12-14) is critical to stabilize the slower stages,
   which oscillate before decay.
8. Final prediction is an ensemble (weighted softmax average of the
   stages).

## What's NOT used (intentionally)

- BPTT through time
- Surrogate gradients for spike emission
- Stuart-Landau cubic feedback
- Inter-stage backward gradient transport
- Learned readout matrices
- Recurrent connections within a pool

## Why this matters scientifically

The result demonstrates that **a hierarchical resonant network with
fully local learning rules can solve SMNIST competitively**. The
recursive decomposition vision — each level extracts longer-time-scale
features from the level below — is operative:
- Stage 0 (ω ∈ [0.005, 1.2]) filters the raw pixel sequence.
- Stage 1 (ω ∈ [0.001, 0.30]) resonates with envelopes of stage-0
  amplitudes.
- Stage 2 (ω ∈ [0.0005, 0.08]) resonates with envelopes of stage-1
  amplitudes.

Each stage's ω band gets *learned* (e.g., stage-1 ω moved from 0.156
to 0.183 over 25 epochs) — the substrate itself is plastic.

## Per-stage contributions (3-stage M=20 final)

| Stage | Test acc alone |
|---|---|
| Stage 0 (raw pixel) | 0.665 |
| Stage 1 (envelope of stage 0) | 0.847 |
| Stage 2 (envelope of stage 1) | 0.790 |
| Ensemble | **0.866** |

## Per-stage contributions (3-stage M=24, 35 epochs final)

| Stage | Test acc alone |
|---|---|
| Stage 0 (raw pixel) | 0.712 |
| Stage 1 (envelope of stage 0) | 0.873 |
| Stage 2 (envelope of stage 1) | 0.860 |
| Ensemble | **0.898** (peak 0.899 at ep 31) |

ω trajectory across training:
- Stage 0: ω 0.603 → 0.591 (small change, near-random filter bank)
- Stage 1: ω 0.156 → 0.182 (+17% — substrate is learning)
- Stage 2: ω 0.0394 → 0.0493 (+25% — substrate is learning at slower scale)

Each higher stage's substrate genuinely adapts at its characteristic
time-scale.

## Strict-spiking spectral-geodesic follow-up

Date: 2026-05-08

A spiking-only spectral-geodesic readout improves the strict-spiking
SMNIST result from **81.35%** to **84.85%** on the same 10k train /
2k test protocol. The run was interrupted after epoch 13 to optimize
the implementation, but the best binary-spike ensemble accuracy was
already **0.8485** at epoch 12:

| Epoch | Binary ensemble | Stage 2 | Stage 3 | Stage 4 | Rates 0/1/2/3/4 |
|---:|---:|---:|---:|---:|---|
| 3 | 0.8285 | 0.7360 | 0.8240 | 0.8230 | 0.112/0.113/0.084/0.102/0.171 |
| 7 | 0.8395 | 0.7785 | 0.8255 | 0.8210 | 0.114/0.117/0.093/0.063/0.117 |
| **12** | **0.8485** | **0.7985** | **0.8370** | **0.8360** | 0.116/0.116/0.111/0.047/0.083 |

The mechanism differs from the spike-rate baseline:

- Inter-stage communication remains binary spikes only.
- Each stage computes local tail-window spike-spectrum coefficients.
- Class evidence is a geodesic/prototype score in complex spectral
  space, with a small rate scaffold for stability.
- The local readout credit is multiplied by forward eligibility traces;
  no BPTT, no surrogate gradient through sampled spikes, and no
  inter-stage backward pass are introduced.
- Initial deep-stage saturation self-corrected during training: stage
  3/4 rates moved from roughly 0.95/1.00 at epoch 0 to 0.047/0.083 at
  the peak epoch.

Reproducibility of the interrupted run:

```bash
cd /Users/esi/research/pub_v4/hrn2
python3 -B experiments/train_spectral_geodesic_5stage_smnist_spiking.py \
  --m_per_class 20 --k_fanin 12 \
  --epochs 25 --batch 64 \
  --train_size 10000 --test_size 2000 \
  --threads 4 --spec_q 4 \
  --csv results/smnist_spectral_geodesic_5stage_spiking.csv
```

## Strict-spiking mixture-prototype geodesic (K=4)

Date: 2026-05-08

Replaces the single per-class prototype $\xi_c \in \mathbb{C}^{P\times q}$
with **four prototypes per class** $\{\xi_{c,k}\}_{k=1..4}$. The class
logit is the log-sum-exp over within-class geodesic distances:
$$\text{logit}_c = \log \sum_{k=1}^{4} \exp\!\big(-d(\text{spec}, \xi_{c,k})^2 / \tau\big)$$
The Hebbian update is **winner-take-all within class**: each batch sample
moves only its nearest within-class prototype. Captures intra-class
variation (e.g.\ class 7 with vs.\ without horizontal stroke) that a
single mean-prototype necessarily averages out.

Results on the same 10k train / 2k test protocol:

| Epoch | Binary ensemble | Stage 2 | Stage 3 | Stage 4 | Rates 0/1/2/3/4 |
|---:|---:|---:|---:|---:|---|
| 3 | 0.8245 | 0.7450 | 0.8220 | 0.8210 | 0.112/0.113/0.072/0.135/0.232 |
| 7 | 0.8385 | 0.7300 | 0.8310 | 0.8340 | 0.114/0.118/0.090/0.108/0.171 |
| 12 | 0.8435 | 0.7690 | 0.8210 | 0.8360 | 0.115/0.114/0.116/0.083/0.092 |
| 13 | 0.8570 | 0.7930 | 0.8340 | 0.8570 | 0.116/0.118/0.121/0.085/0.084 |
| **15** | **0.8630** | **0.7955** | **0.8405** | **0.8635** | 0.116/0.116/0.129/0.097/0.082 |

Best binary-spike ensemble: **0.8630** at epoch 15. The run was stopped
at epoch 23 (of 25 planned) after plateauing in the 84-86% range past
LR decay #2. Stage 4 alone reached 0.8635 — the slowest band benefits
most from mixture prototypes.

Hyperparameters (only `--k_mix 4` and `--proto_init_jitter 0.05` differ
from the single-prototype variant):

```bash
cd /Users/esi/research/pub_v4/hrn2
python3 -B experiments/train_spectral_geodesic_mixture_5stage_smnist_spiking.py \
  --m_per_class 20 --k_fanin 12 --rec_k 0 \
  --k_mix 4 --proto_init_jitter 0.05 \
  --epochs 25 --batch 64 \
  --train_size 10000 --test_size 2000 --threads 4 \
  --lr 0.002 --lr_decay_after 10 --lr_decay_after_2 18 \
  --csv results/smnist_spectral_geodesic_mixture_K4.csv
```

Net spiking-SMNIST progression in this thread:
0.8135 (no spectral readout) → 0.8485 (single-prototype geodesic) →
**0.8630** (mixture-prototype geodesic, K=4). Total improvement of
**+5.0pp** while preserving binary-spike inter-neuron communication,
forward-only eligibility, no surrogate of Heaviside.

## SHD spectral-geodesic follow-up

Date: 2026-05-08

The same spectral-geodesic readout transfers cleanly to SHD. On the
existing 4k train / 1k test cache (`T=100`, 700 cochlear channels,
20 classes), the 3-stage strict-spiking spectral run reached
**76.70%** binary-spike ensemble accuracy at epoch 16:

| Method | Best binary test |
|---|---:|
| Analog SHD 3-stage HRN-v2 | 0.684 |
| Strict-spiking SHD 3-stage spike-rate readout | 0.606 |
| **Strict-spiking SHD spectral-geodesic readout** | **0.767** |

Peak epoch details:

| Epoch | Binary ensemble | Stage 0 | Stage 1 | Stage 2 | Rates 0/1/2 |
|---:|---:|---:|---:|---:|---|
| **16** | **0.7670** | 0.6000 | 0.7560 | 0.7380 | 0.111/0.113/0.104 |

Reproducibility:

```bash
cd /Users/esi/research/pub_v4/shd
python3 -B experiments/train_spectral_geodesic_3stage_shd_spiking.py \
  --m_per_class 12 --k0_fanin 48 --k1_fanin 12 --k2_fanin 12 \
  --epochs 25 --batch 64 \
  --train_size 4000 --test_size 1000 \
  --threads 4 --spec_q 4 \
  --csv results/shd_spectral_geodesic_3stage_spiking.csv
```

### SHD ablations toward 90-95%

Date: 2026-05-08

We then tested several principled extensions aimed at the >90% SHD target.
All retained strict binary inter-stage spikes and local spectral
eligibility updates.

| Variant | CSV | Stopped at | Best binary test | Read |
|---|---|---:|---:|---|
| Fixed delay bank + 4 spectral prototypes/class | `shd/results/shd_spectral_geodesic_delay_proto_3stage_spiking.csv` | epoch 8 | 0.737 | Added capacity but did not beat 0.767 |
| All-stage learned recurrent weights, row-norm constrained | `shd/results/shd_spectral_geodesic_rec_3stage_spiking.csv` | epoch 3 | 0.664 | Stabilized rates but slowed separation |
| Stage-2-only learned recurrence | `shd/results/shd_spectral_geodesic_rec2_3stage_spiking.csv` | epoch 2 | 0.550 | Raw recurrence disrupted the final manifold |
| Grouped learnable delay taps per cochlear fan-in | `shd/results/shd_spectral_geodesic_grouped_delay_3stage_spiking.csv` | epoch 3 | 0.688 | Faster early learning, worse later trajectory |
| 4-prototype spectral atlas, no delay/recurrence | `shd/results/shd_spectral_geodesic_proto4_max_3stage_spiking.csv` | epoch 3 | 0.675 | Better epoch 1, worse later trajectory |

Interpretation: simply adding delay capacity, recurrent weights, or
more spectral prototypes is not enough. The successful baseline appears
to rely on a delicate stage-wise manifold formation process: stage 1 is
the strongest separator and stage 2 must refine without injecting
unstructured temporal inertia. The next recurrence attempt should be
more constrained than arbitrary sparse recurrent weights, e.g. a
structured class-pool Laplacian / contractive transport operator with
learned scalar gains rather than free recurrent synapses.

## Reproducibility

Best 2-stage:
```bash
cd /Users/esi/research/pub_v4/hrn2
python3 -B experiments/train_decolle_2stage.py \
  --m0_per_class 24 --m1_per_class 24 --k1_fanin 12 \
  --epochs 25 --batch 64 \
  --train_size 10000 --test_size 2000 --threads 4 \
  --tail0 200 --tail1 300 \
  --lr0 0.005 --lr1 0.005 --lr_decay_after 14 --lr_decay_factor 0.5
```

Best 3-stage:
```bash
python3 -B experiments/train_decolle_3stage.py \
  --m0_per_class 24 --m1_per_class 24 --m2_per_class 24 \
  --k1_fanin 12 --k2_fanin 12 \
  --epochs 35 --batch 64 \
  --train_size 10000 --test_size 2000 --threads 4 \
  --tail0 200 --tail1 300 --tail2 400 \
  --lr 0.005 --lr_decay_after 12 --lr_decay_after_2 24 --lr_decay_factor 0.5
```

## Open questions / next steps

1. Does scaling N per stage continue to help? (M=24 vs M=48 vs M=96)
2. Does deeper hierarchy (4-stage, 5-stage) keep adding?
3. Does training on full 60k SMNIST (vs 10k) close more gap to 91%?
4. Can ensembling multiple seeds push higher?
5. Can "discover-nudge-lock" replacing the LR decay schedule give a
   smoother/better trajectory?
