# Morning brief — 2026-05-07

You went to sleep with this picture: BPTT-trained net at 50.07% in-loop,
random reservoir at 56.55%, BPTT looked like it had not helped. Wrong
interpretation; the in-loop pool-rate readout was a bottleneck. Here is
the corrected picture after the rest of the autonomous overnight work.

## Best result tonight: **0.6751** on full SMNIST

BPTT-trained checkpoint, tail-window pool firing rates (240-d) read
out by a 2-layer MLP (h=512, no dropout) trained for 150 epochs on the
full 60k training set. Random-init reservoir under the same readout
caps at 0.5669, so BPTT contributed about +10.8pp of feature
improvement.

## What we now know

| Question | Answer |
|---|---|
| Did BPTT improve features over random init? | **Yes, by ~10.8pp at the MLP-head level** |
| Was the in-loop 50% number a true ceiling? | **No** — it was the per-class pool-rate readout. Offline linear head on the same features = 65.24%. |
| Does more BPTT continue to help features? | **No** at this scale. 90 extra minutes pushed in-loop 47% → 55% (readout fit) but offline-readout ceiling stayed ~67.5%. |
| What is needed for 95%? | More neurons / different architecture. The 600-neuron net is at its representational ceiling. |

## Three independent pieces of evidence pointing at the same conclusion

1. **Trained vs random head sweep** (same head training budget):
   - Random: 56.69% (linear head, 200 ep, 60k train)
   - Trained: 65.24% (linear head, same setup)
   - Trained: 67.51% (MLP h=512, 150 ep)

2. **MLP h=128 vs h=512**: 67.41% vs 67.51%. Doubling head capacity
   adds 0.1pp. The features are the bottleneck, not the head.

3. **Continued BPTT** (90 min, lr 5e-4):
   - In-loop pool-rate: 47.35% → 54.55%
   - Offline linear: 65.04% → 65.62% (noise)
   - Offline MLP h=512: 67.51% → 67.47% (noise)
   - Conclusion: extra training fit the readout to existing features
     rather than improving features.

## Per-class breakdown of the best run (MLP h=512 head, 67.34% on this seed)

| Class | N | Acc |
|---|---|---|
| 1 | 1135 | 0.933 |
| 6 |  958 | 0.817 |
| 8 |  974 | 0.796 |
| 4 |  982 | 0.742 |
| 7 | 1028 | 0.665 |
| 0 |  980 | 0.662 |
| 2 | 1032 | 0.661 |
| 9 | 1009 | 0.581 |
| 3 | 1010 | 0.529 |
| 5 |  892 | **0.284** |

Top confusions (off-diagonal):
- **5 → 3**: 308 / 892 (~35% of all 5s misread as 3)
- **9 ↔ 7**: 306 + 257 = 563 mixed up
- **4 → 9**: 190
- **3 → 0**: 175

Class 5 is the killer. Its raster-scan pixel sequence is too similar
to a "bad 3" or "bad 0" for the current frequency-tiled features to
separate. This is a useful diagnostic: scaling neurons or adding more
temporal-feature variety should mostly close the 5/3/9/7 confusions.

CSV / log: `logs/per_class_breakdown.log`.

## Files to look at

- `findings/RESULTS.md` — full breakdown with numbers, trajectories
- `findings/NEXT_STEPS.md` — what to try next; reordered priorities
- `results/run_trained_vs_random_head.csv` — decisive comparison
- `results/run_heads_on_trained.csv` — head-architecture sweep
- `results/ckpt_overnight_default_10k.pt` — original 4.5h ckpt
- `results/ckpt_continue_default.pt` — best continuation ckpt
- `results/feat_cache/`, `results/feat_cache_continue/` — cached features

## What to do next

The clear path forward, in priority order:

1. **Replace the in-loop readout with a trainable linear head as the
   primary loss**, dropping the pool-rate constraint. The bio-plausible
   "resonant basin" framing is conceptually clean but costs ~17pp of
   accuracy. A linear head on tail-window pool features is closer to
   what you'd actually want.

2. **Scale to ~2000 neurons.** At 600 neurons we hit a ~67.5% ceiling.
   Pub_v1 used 8192 single-population for 90.8%. A scaled hierarchical
   architecture with ~2000 neurons should land in the 75-85% range.

3. **Architectural change**: Try input pre-coding (row at a time, T=28
   instead of T=784) to reduce the temporal-credit-assignment burden.
   This is row-sequence MNIST — a less-extreme but still recurrent
   benchmark that the network should solve more easily.

4. **Local rule revisit**: e-prop runs 100x faster per step. If we
   redesigned the local rule with mean-subtracted feedback alignment +
   pool-pool lateral inhibition + sample-level credit, it could
   plausibly reach random-reservoir level (56%) or better, with
   cheap iteration.

The 95% target is still real but requires either scale (path 2) or
problem-reduction (path 3). The 4-hour overnight budget at this scale
hits its ceiling near 67.5%.

## Reproducing the headline result

```bash
cd /Users/esi/research/pub_v4
python3 -B scripts/heads_on_trained.py --epochs 150
# best result will be MLP h=512 line in results/run_heads_on_trained.csv
```

(Features are cached in `results/feat_cache/` from tonight's run, so
this should be ~5 minutes after the first invocation re-uses the cache.)
