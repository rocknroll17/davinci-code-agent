#!/usr/bin/env bash
# heads/layers architecture sweep — scale both up, arena to find the best setting.
# 4 configs, one per GPU, trained to STEPS (default 4M), separate dirs under experiments/.
#
#   GPU 0  sw_h16l4   heads=16 layers=4   (push heads)
#   GPU 1  sw_h4l8    heads=4  layers=8   (push layers)
#   GPU 2  sw_h8l6    heads=8  layers=6   (moderate both)
#   GPU 3  sw_h16l8   heads=16 layers=8   (max both)
#
# token_dim=128 → valid head counts are powers of 2 (…,8,16,32). 16 → head_dim 8.
#
# Usage:  bash scripts/launch_sweep.sh         (STEPS=4000000 default)
#         bash scripts/launch_sweep.sh stop
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
STEPS="${STEPS:-4000000}"
RUN_DIR="experiments"
PIDS="$RUN_DIR/sweep_pids.txt"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "stop" ]]; then
    [[ -f "$PIDS" ]] && while read -r pid name; do kill "$pid" 2>/dev/null && echo "stopped $name"; done < "$PIDS"
    rm -f "$PIDS"; exit 0
fi

: > "$PIDS"
launch() {  # name gpu port heads layers
    local name=$1 gpu=$2 port=$3 heads=$4 layers=$5
    mkdir -p "$RUN_DIR/$name"   # keep checkpoints → run_experiment.py auto-resumes from latest.pt
    echo "Launching $name on GPU $gpu (h=$heads l=$layers, :$port)"
    PYTHONUNBUFFERED=1 nohup "$PY" run_experiment.py --exp "$name" --device-id "$gpu" \
        --total-timesteps "$STEPS" --dashboard-port "$port" \
        --n-heads "$heads" --n-layers "$layers" --batch-size 4096 \
        > "$RUN_DIR/$name/run.out" 2>&1 &
    echo "$! $name" >> "$PIDS"
    sleep 2
}

launch sw_h16l4 0 6006 16 4
launch sw_h4l8  1 6007 4  8
launch sw_h8l6  2 6008 8  6
launch sw_h16l8 3 6009 16 8

echo ""; echo "Sweep launched (to ${STEPS} steps):"; cat "$PIDS"
echo "Watch:  $PY scripts/watch_experiments.py --total $STEPS"
echo "Arena:  $PY compare_experiments.py --ckpt latest.pt --games 120 --eval-episodes 0"
echo "Stop:   bash scripts/launch_sweep.sh stop"
