#!/bin/bash
# Morning summary — autonomous overnight 2026-05-06 → 2026-05-07
# Run: bash scripts/morning_summary.sh

cd /Users/esi/research/pub_v4

echo "=========================================="
echo "  pub_v4 OVERNIGHT TRAINING SUMMARY"
echo "  $(date)"
echo "=========================================="

echo ""
echo "## ► Read MORNING_BRIEF.md FIRST"
echo "   (interpretation has changed since you went to sleep —"
echo "    BPTT did help; the in-loop pool-rate readout was a bottleneck)"
echo ""
ls -la MORNING_BRIEF.md 2>/dev/null

echo ""
echo "## Headline numbers"
echo "   BPTT-trained features + MLP h=512 head:  0.6751  <- best"
echo "   BPTT-trained features + linear head:     0.6524"
echo "   Random reservoir + linear head:          0.5669"
echo "   Continued BPTT, in-loop pool-rate (2k):  0.5455  (was 0.4735)"

echo ""
echo "## Process status (any still running?)"
ps aux | grep -E "(train_bptt|continue_bptt|heads_on_trained|quick_heads)" | grep -v grep \
  | awk '{print "  PID", $2, "CPU%", $3, "MEM%", $4, "ELAPSED", $10, "CMD", $11" "$12" "$13}'
if [ -z "$(ps aux | grep -E '(train_bptt|continue_bptt|heads_on_trained|quick_heads)' | grep -v grep)" ]; then
    echo "  (no training processes running)"
fi

echo ""
echo "## Trained vs random head comparison (decisive test)"
if [ -f results/run_trained_vs_random_head.csv ]; then
    cat results/run_trained_vs_random_head.csv
fi

echo ""
echo "## Head-architecture sweep on trained features"
if [ -f results/run_heads_on_trained.csv ]; then
    cat results/run_heads_on_trained.csv
fi

echo ""
echo "## Continuation BPTT trajectory (90 min, lr 5e-4)"
if [ -f results/run_continue_default.csv ]; then
    cat results/run_continue_default.csv
fi

echo ""
echo "## Original BPTT trajectory (4.5h)"
if [ -f results/run_overnight_default_10k.csv ]; then
    cat results/run_overnight_default_10k.csv
fi

echo ""
echo "## Checkpoints"
ls -la results/ckpt_*.pt 2>/dev/null

echo ""
echo "## Cached features (for fast offline head training)"
ls -la results/feat_cache* 2>/dev/null

echo ""
echo "## Findings docs"
ls -la findings/

echo ""
echo "## Logs from this overnight"
ls -la logs/ 2>/dev/null | tail -20

echo ""
echo "=========================================="
echo "  See MORNING_BRIEF.md for the corrected"
echo "  interpretation and recommended next steps."
echo "=========================================="
