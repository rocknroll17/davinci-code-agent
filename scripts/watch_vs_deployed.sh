#!/usr/bin/env bash
# Every INTERVAL seconds: snapshot checkpoints/latest.pt, play it against the deployed
# model, and keep the snapshot with the best win rate as checkpoints/best_vs_deployed.pt.
cd "$(dirname "$0")/.."
CKPT_DIR=${CKPT_DIR:-checkpoints}
LOG=${LOG:-logs/vs_deployed.log}
INTERVAL=${INTERVAL:-1800}
GAMES=${GAMES:-1000}
best=0
mkdir -p "$CKPT_DIR/vs_deployed"
while true; do
    snap=$CKPT_DIR/vs_deployed/$(date +%m%d_%H%M).pt
    cp "$CKPT_DIR/latest.pt" "$snap"
    line=$(venv/bin/python scripts/vs_deployed.py --ckpt "$snap" --games "$GAMES" --device cuda 2>/dev/null | tail -1)
    echo "$(date '+%F %T') $line" | tee -a "$LOG"
    rate=$(echo "$line" | sed -n 's/.*win rate \([0-9.]*\)%.*/\1/p')
    if [ -n "$rate" ] && awk "BEGIN{exit !($rate > $best)}"; then
        best=$rate
        cp "$snap" "$CKPT_DIR/best_vs_deployed.pt"
        echo "  -> new best_vs_deployed.pt ($rate%)" | tee -a "$LOG"
    fi
    sleep "$INTERVAL"
done
