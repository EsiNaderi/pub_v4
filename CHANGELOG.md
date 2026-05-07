# pub_v4 Changelog (overnight 2026-05-06)

A record of decisions made during the overnight session.

## 22:40 EDT — Session start

User went to sleep. Goal: 95%+ on SMNIST single-channel using
hierarchical resonant net + biologically plausible local learning.

## Initial setup

- Created workspace `/Users/esi/research/pub_v4` with src/, scripts/,
  findings/, logs/, results/.
- Loaded SMNIST raw bytes to PyTorch tensors, cached.

## Architecture v0 — vanilla Stuart-Landau

- Implemented `ResonatorPool` with complex amplitude `z = u + iv`,
  rotation by omega, Stuart-Landau saturation, threshold-on-amplitude
  spike, contraction reset.
- Block-diagonal recurrence for pool independence.
- Hierarchical Resonant Net (`HRN`) composing pools across layers.
- Per-class output pools as the readout principle.

## Stability crisis & fixes

Discovered NaN gradients in layer 0 within ~40 BPTT steps:

1. Tried surrogate gradient variants (rect, fast sigmoid). Slope tuning
   didn't help.
2. Disabled `theta` learning (huge gradients accumulated to ~1e4).
3. Detached `s_prev` before W_rec (e-prop style). Layer 1+ stabilized.
4. Layer 0 still NaN — gradient through cubic SL feedback.
5. **Detached amp_sq in SL term** → no more NaN at T=300.
6. **Amplitude clamp on (u, v) ∈ [-3, 3]** → stable through full T=784.

After these four fixes, BPTT runs cleanly.

## Spike rate tuning

Sweep over `(theta, eta, in_init_scale, gamma, beta)` to find an init
that gives healthy 5-30% rates per layer. Settled on:
- Layer 0: theta=0.7, eta=0.30, in_init_scale=4.0, gamma=beta=0.20
- Layer 1: theta=0.7, eta=0.10, in_init_scale=2.0
- Output: theta=0.6, eta=0.15, in_init_scale=4.0

## Performance benchmarks

- CPU forward (default arch, batch=32, T=784): ~10s/step
- MPS forward: 4x slower than CPU (Python loop overhead)
- torch.compile: failed to finish compile in 15+ min
- torch.jit.script on inner step: ~10% speedup
- Conclusion: stuck at ~10s/step on CPU

## Training experiments

- Smoke tests on 1k subset → loss not moving (chance acc).
- Diagnosed: pool bias was zero, all logits equal → uniform softmax →
  no useful gradient.
- Added: small random pool_bias init (breaks symmetry); centered logits
  (logits = (rate - mean(rate)) * temp + bias).
- After fix: training started but slow.

## Random feature baselines

- Default arch (226k params, ~600 neurons): 47% test acc with linear
  head on 5k subset.
- Small arch: 31% ± 3% across 3 seeds on 5k subset.
- Pool-rate readout (no learnable head): 10% chance.

## Architecture variants

- Tried "tiled" (1.0M params, frequency-tiled per pool, ALIF on): 25%
  test (over-saturated, sparse features).
- Tried "big" (3.2M params): too slow to compute features in budget.
- Defaulted to "default" arch as the working configuration.

## ALIF added (then turned off by default)

Added adaptive threshold via spike-frequency adaptation. With default
alpha=0.05, tau=100, the network became too sparse. Default
`use_alif=False`. Available via `resonant_alif.py`, `rotator_alif.py`.

## Local rules attempt

- Implemented e-prop / 3-factor rule: forward-only, eligibility traces,
  random feedback alignment, manual Adam.
- 100x faster per step than BPTT (0.5s vs 10s).
- BUT did NOT learn class structure (stuck at chance).
- Iterated v2 with mean-subtraction + winner-take-all — still chance.
- Documented as "stable but not effective" in lessons learned.

## Main BPTT run (started 22:40)

- arch=default, aux_head=True, 10k subset, batch=32, lr=1e-3, time
  budget 4 hours.
- Loss EMA decreased from 2.327 to 1.654 in first 200 steps.
- Train acc EMA 0.06 → 0.37.
- Trajectory ongoing.

## Auto-chain (queued)

After BPTT finishes (02:40 ETA), chain runs:
- Phase 2: Frozen-reservoir + linear head on full 60k SMNIST.
- Phase 3: Multi-seed random feature baseline.
- Phase 4: Morning summary script.

## Documentation produced

- README.md (top-level)
- REPLICATE.md (step-by-step)
- RUNNING_OVERNIGHT.md (what's running tonight)
- findings/DESIGN.md (architecture rationale)
- findings/RESULTS.md (measured numbers)
- findings/EXPERIMENT_LOG.md (chronological)
- findings/LESSONS_LEARNED.md (surprises)
- findings/NEXT_STEPS.md (concrete suggestions)
- findings/RELATED_WORK.md (citations)
- CHANGELOG.md (this)

## Conclusion at session checkpoint (23:18)

Architecture is operational. BPTT trains it slowly but steadily on a 10k
subset. 95% target NOT reached in tonight's session — limited by per-step
BPTT cost on CPU (~10s/step) and a 4-hour overnight budget. The
architecture, stability fixes, and trainers are reusable for next
iteration. Local rules are stable but need more design iterations.
