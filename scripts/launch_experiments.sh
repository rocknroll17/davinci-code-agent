#!/usr/bin/env bash
# Launch the 4-way comparison study: one experiment per GPU, all to 10M timesteps.
#
#   GPU 0  baseline   — original model, original reward
#   GPU 1  monotone   — monotone reward (no win/lose, only per-guess reward)
#   GPU 2  heads      — 8 attention heads (token_dim=128 is not divisible by 6, so
#                       the next valid head count above 4 is 8)
#   GPU 3  layers     — 5 transformer layers (4 -> 5)
#
# Everything else is identical across runs.
#
# Usage:  bash scripts/launch_experiments.sh
# Logs:   experiments/<name>/run.out   (training stdout)
# Stop:   bash scripts/launch_experiments.sh stop

set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
STEPS="${STEPS:-10000000}"
RUN_DIR="experiments"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "stop" ]]; then
    if [[ -f "$RUN_DIR/pids.txt" ]]; then
        while read -r pid name; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $name (pid $pid)"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$RUN_DIR/pids.txt"
        rm -f "$RUN_DIR/pids.txt"
    else
        echo "No pids.txt found."
    fi
    exit 0
fi

: > "$RUN_DIR/pids.txt"

launch() {
    local name="$1"; shift
    local gpu="$1"; shift
    local port="$1"; shift
    mkdir -p "$RUN_DIR/$name"
    echo "Launching '$name' on GPU $gpu  (dashboard :$port) ..."
    # PYTHONUNBUFFERED=1 → logs stream to run.out live (tail -f works in real time)
    PYTHONUNBUFFERED=1 nohup "$PY" run_experiment.py --exp "$name" --device-id "$gpu" \
        --total-timesteps "$STEPS" --dashboard-port "$port" "$@" \
        > "$RUN_DIR/$name/run.out" 2>&1 &
    echo "$! $name" >> "$RUN_DIR/pids.txt"
    sleep 2  # stagger startup so worker forks don't thundering-herd
}

#      name      gpu  dashboard-port
launch baseline  0    6006
launch monotone  1    6007 --reward-mode monotone
launch heads     2    6008 --n-heads 8
launch layers    3    6009 --n-layers 5

echo ""
echo "All 4 experiments launched. PIDs:"
cat "$RUN_DIR/pids.txt"
echo ""
echo "Dashboards:     baseline :6006  monotone :6007  heads :6008  layers :6009"
echo "Follow a run:   tail -f experiments/baseline/run.out"
echo "Stop all:       bash scripts/launch_experiments.sh stop"
