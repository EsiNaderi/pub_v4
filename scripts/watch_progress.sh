#!/bin/bash
# Watch BPTT progress in real time. Usage: bash scripts/watch_progress.sh
#
# Tails the overnight training log and the CSV file, parses the latest values.

LOG=/Users/esi/research/pub_v4/logs/run_overnight_default_10k.log
CSV=/Users/esi/research/pub_v4/results/run_overnight_default_10k.csv

echo "=== Latest log entries ==="
tail -10 "$LOG" 2>/dev/null

echo ""
echo "=== CSV entries (per-epoch) ==="
if [ -f "$CSV" ]; then
    head -1 "$CSV"
    tail -10 "$CSV"
else
    echo "(no CSV yet — first epoch eval pending)"
fi

echo ""
echo "=== Process status ==="
ps aux | grep "train_bptt" | grep -v grep | head -3 | awk '{print "PID", $2, "CPU%", $3, "MEM%", $4, "ELAPSED", $10}'
