#!/bin/bash
# Sequential training chain. Designed to keep the CPU busy through the night.
# Each phase runs after the previous completes (or hits its time budget).
#
# This is a wrap-around for the existing main BPTT run. Once that finishes,
# we run several follow-ups in sequence.

set -e
cd /Users/esi/research/pub_v4

LOGDIR=logs/overnight_chain
mkdir -p $LOGDIR

# Wait for the main BPTT run to finish (it self-terminates at time_budget)
echo "[chain] waiting for main BPTT to finish..." | tee -a $LOGDIR/chain.log
while ps aux | grep "train_bptt.py" | grep -v grep > /dev/null; do
    sleep 60
done
echo "[chain] main BPTT done at $(date)" | tee -a $LOGDIR/chain.log

# Phase 2: head-only finetune on the BPTT-trained features (full SMNIST)
echo "[chain] phase 2: head-only on full SMNIST" | tee -a $LOGDIR/chain.log
python3 -B src/train_head_only.py \
    --arch default \
    --train_size 60000 --test_size 10000 \
    --head_kind both --head_epochs 200 \
    --csv results/run_chain_head_full.csv \
    > $LOGDIR/phase2_head_full.log 2>&1
echo "[chain] phase 2 done at $(date)" | tee -a $LOGDIR/chain.log

# Phase 3: random-feature multi-seed (default) on full SMNIST
echo "[chain] phase 3: multi-seed random feats" | tee -a $LOGDIR/chain.log
python3 -B scripts/multiseed_random_feats.py \
    > $LOGDIR/phase3_multiseed.log 2>&1
echo "[chain] phase 3 done at $(date)" | tee -a $LOGDIR/chain.log

# Phase 4: write a final markdown summary of all results
echo "[chain] phase 4: morning summary" | tee -a $LOGDIR/chain.log
bash scripts/morning_summary.sh > $LOGDIR/phase4_summary.txt 2>&1
echo "[chain] all phases done at $(date)" | tee -a $LOGDIR/chain.log
