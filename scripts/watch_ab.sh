#!/usr/bin/env bash
# Every 10 min: snapshot both A/B runs and play them head-to-head (1000 games).
cd "$(dirname "$0")/.."
A=ab_shaped; B=ab_outcome
while true; do
    snap=experiments/_snap/$(date +%m%d_%H%M)
    for e in $A $B; do
        mkdir -p "$snap/$e/checkpoints"
        cp "experiments/$e/checkpoints/latest.pt" "$snap/$e/checkpoints/latest.pt"
    done
    venv/bin/python arena_fast.py --root "$snap" --games 1000 --n-envs 100 --n-workers 2 --device cuda:0 2>/dev/null \
        | grep " vs " | sed "s|^ *|$(date '+%F %T') |" | tee -a logs/ab.log
    rm -rf "$snap"
    sleep 600
done
