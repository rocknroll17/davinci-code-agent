#!/usr/bin/env python3
"""
Multi-GPU data-parallel training entry point (manual-DDP PPO).

Launch with torchrun (one process per GPU):

  torchrun --nproc_per_node=4 train_ddp.py --exp h8l6_200M \
      --n-heads 8 --n-layers 6 --total-timesteps 200000000 --fp16

Each rank runs its own SubprocVecEnv (per-rank seed) and PPO update; gradients
are averaged across ranks (manual all-reduce in PPOTrainer.update). Only rank 0
logs, evaluates, and writes checkpoints. Single-GPU still works:
`python train_ddp.py ...` (world_size=1, no dist) or the original run_experiment.py.

NOTE: unlike run_experiment.py this does NOT set CUDA_VISIBLE_DEVICES — torchrun
assigns each rank its GPU via LOCAL_RANK.
"""
import argparse
import os

import torch
import torch.distributed as dist

import src.utils.logger  # noqa: F401  (configures root logger)
from src.hooks import NaNDetector
from src.trainer import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DaVinci Code DDP training")
    p.add_argument("--exp", required=True)
    p.add_argument("--reward-mode", choices=["normal", "monotone"], default="normal")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--zero-init", action="store_true")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=440)
    p.add_argument("--n-workers", type=int, default=22)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dashboard-port", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # torchrun env (defaults make this work as a plain single-process run too).
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    if args.reward_mode == "monotone":
        os.environ["DVC_MONOTONE_REWARD"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(args.torch_threads)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_dist:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            device_id=device if torch.cuda.is_available() else None,
        )

    exp_dir = os.path.join("experiments", args.exp)
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_workers=args.n_workers,
        episodes_per_update=args.n_envs,
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        zero_init=args.zero_init,
        fp16=args.fp16,
        compile=args.compile,
        monotone_reward=(args.reward_mode == "monotone"),
        save_dir=os.path.join(exp_dir, "checkpoints"),
        log_dir=os.path.join(exp_dir, "logs"),
    )

    if rank == 0:
        print("=" * 70)
        print(f"DDP training: {args.exp}")
        print(f"  world_size       : {world_size} | rank0 device cuda:{local_rank}")
        print(f"  n_heads/n_layers : {config.n_heads} / {config.n_layers}")
        print(f"  fp16/compile     : {config.fp16} / {config.compile}")
        print(f"  n_envs/rank      : {config.n_envs} (effective {config.n_envs * world_size})")
        print(f"  total_timesteps  : {config.total_timesteps:,} (global)")
        print("=" * 70)

    trainer = PPOTrainer(config, device, rank=rank, world_size=world_size)

    if rank == 0:
        trainer.register_hook(NaNDetector())
        if args.dashboard_port > 0:
            try:
                from src.dashboard.server import DashboardServer
                from src.hooks import DashboardHook
                dash = DashboardServer(host="0.0.0.0", port=args.dashboard_port)
                dash.start()
                trainer.register_hook(DashboardHook(dash))
                print(f"[Dashboard] {args.exp}: http://localhost:{args.dashboard_port}")
            except Exception as e:
                print(f"[Dashboard] disabled ({e})")

    latest = os.path.join(config.save_dir, "latest.pt")
    if os.path.exists(latest):
        trainer.load(latest)
        if rank == 0:
            print(f"Resumed from global timestep {trainer._last_global_timesteps:,}")

    try:
        trainer.train()
    finally:
        if is_dist:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
