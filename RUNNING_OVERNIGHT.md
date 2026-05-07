# What's Running Overnight (started 2026-05-06 22:40 EDT)

## Phase 1: BPTT main run (in progress)

```
PID: 19686
Cmd: python3 -B src/train_bptt.py --arch default --aux_head --epochs 8
     --batch 32 --train_size 10000 --test_size 2000
     --log_every 20 --lr 1e-3 --clip_norm 1.0 --time_budget 14400
     --csv results/run_overnight_default_10k.csv
     --ckpt results/ckpt_overnight_default_10k.pt
```

- Started 22:40
- Will self-terminate at 02:40 (4-hour budget) or 8 epochs (whichever first)
- Logs: `logs/run_overnight_default_10k.log`
- CSV: `results/run_overnight_default_10k.csv` (per-epoch eval)
- Best ckpt: `results/ckpt_overnight_default_10k.pt`

Expected per-epoch wall: ~58 min. Realistic ~4 epochs in budget.

## Phase 2-4: Auto-chain (queued, waits for Phase 1 completion)

```
PID: 24290 (bash run_overnight_chain.sh)
Logs: logs/overnight_chain/
```

After BPTT finishes:

- Phase 2: `train_head_only.py` on full 60k SMNIST. Uses the trained HRN
  as a frozen feature extractor and trains a linear head. ~15 min.
  → `logs/overnight_chain/phase2_head_full.log`
  → `results/run_chain_head_full.csv`
- Phase 3: `multiseed_random_feats.py` for a clean multi-seed baseline
  on the trained network. ~5 min.
  → `logs/overnight_chain/phase3_multiseed.log`
- Phase 4: Run `morning_summary.sh` to produce the final report.
  → `logs/overnight_chain/phase4_summary.txt`

## To check progress at any time

```bash
bash scripts/watch_progress.sh        # live snapshot
bash scripts/morning_summary.sh       # comprehensive summary
tail -F logs/run_overnight_default_10k.log
ls -lt results/ logs/                  # latest files
```

## Final status check (in the morning)

```bash
ps aux | grep -E "train_bptt|run_overnight_chain" | grep -v grep
ls -la results/ckpt_overnight_default_10k.pt
cat logs/overnight_chain/chain.log
python3 -B scripts/final_report.py --ckpt results/ckpt_overnight_default_10k.pt
```
