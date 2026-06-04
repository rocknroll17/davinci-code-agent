#!/usr/bin/env bash
# Multi-GPU (4x) data-parallel training via torchrun.
#   bash scripts/train_ddp.sh h8l6_200M 8 6 200000000
# args: <exp> <n_heads> <n_layers> <total_timesteps>
set -uo pipefail
cd "$(dirname "$0")/.."

EXP="${1:-h8l6_200M}"
HEADS="${2:-8}"
LAYERS="${3:-6}"
STEPS="${4:-200000000}"
NPROC="${NPROC:-4}"
NENVS="${NENVS:-440}"
# 4 ranks share 96 cores -> ~22 workers/rank
NWORKERS="${NWORKERS:-22}"
BATCH="${BATCH:-8192}"

mkdir -p "experiments/$EXP"
# torch.compile (inductor) needs Python.h at gcc time; venv ships them under include/.
export CPATH="$PWD/.venv/include/python3.10${CPATH:+:$CPATH}"
COMPILE_FLAG="${COMPILE:+--compile}"   # COMPILE=1 to enable torch.compile
PYTHONUNBUFFERED=1 nohup .venv/bin/torchrun --nproc_per_node="$NPROC" --standalone train_ddp.py \
    --exp "$EXP" --n-heads "$HEADS" --n-layers "$LAYERS" \
    --total-timesteps "$STEPS" --n-envs "$NENVS" --n-workers "$NWORKERS" \
    --batch-size "$BATCH" --fp16 $COMPILE_FLAG \
    > "experiments/$EXP/run.out" 2>&1 &
echo "$!" > "experiments/$EXP/ddp.pid"
echo "launched DDP training '$EXP' (${NPROC} GPU, h=$HEADS l=$LAYERS, ${STEPS} steps, fp16)"
echo "  log:  tail -f experiments/$EXP/run.out"
echo "  stop: kill \$(cat experiments/$EXP/ddp.pid)  (or pkill -f train_ddp.py)"
