#!/usr/bin/env bash
# Every INTERVAL seconds: snapshot checkpoints/latest.pt, play it against the deployed
# model, and keep the snapshot with the best win rate as checkpoints/best_vs_deployed.pt.
cd "$(dirname "$0")/.."
INTERVAL=${INTERVAL:-1800}
GAMES=${GAMES:-1000}
best=0
mkdir -p checkpoints/vs_deployed
while true; do
    snap=checkpoints/vs_deployed/$(date +%m%d_%H%M).pt
    cp checkpoints/latest.pt "$snap"
    line=$(venv/bin/python scripts/vs_deployed.py --ckpt "$snap" --games "$GAMES" --device cuda 2>/dev/null | tail -1)
    echo "$(date '+%F %T') $line" | tee -a logs/vs_deployed.log
    rate=$(echo "$line" | sed -n 's/.*win rate \([0-9.]*\)%.*/\1/p')
    if [ -n "$rate" ] && awk "BEGIN{exit !($rate > $best)}"; then
        best=$rate
        cp "$snap" checkpoints/best_vs_deployed.pt
        echo "  -> new best_vs_deployed.pt ($rate%)" | tee -a logs/vs_deployed.log
    fi
    sleep "$INTERVAL"
done
