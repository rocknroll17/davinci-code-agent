#!/usr/bin/env python3
"""
Train one named experiment. Checkpoints and logs go to experiments/<name>/, and the
run resumes from experiments/<name>/checkpoints/latest.pt if it exists.

  python run_experiment.py --exp my_run --n-layers 2 --reward-mode outcome

Several GPUs (one process per GPU, gradients averaged across them):

  torchrun --nproc_per_node=4 run_experiment.py --exp my_run
"""

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Da Vinci Code experiment runner")
    p.add_argument("--exp", required=True, help="experiment name (output dir)")
    p.add_argument("--reward-mode", choices=["normal", "monotone", "outcome"], default="normal",
                   help="outcome = win/lose only + potential-based shaping")
    p.add_argument("--n-heads", type=int, default=4, help="encoder attention heads")
    p.add_argument("--n-layers", type=int, default=4, help="encoder transformer layers")
    p.add_argument("--hidden-dim", type=int, default=512, help="width after the encoder (fusion, head inputs)")
    p.add_argument("--ff-dim", type=int, default=512, help="width inside each encoder layer's feed-forward")
    p.add_argument("--zero-init", action="store_true",
                   help="initialize all default weights/biases to 0 (designated inits kept)")
    p.add_argument("--device-id", type=int, default=0, help="CUDA device index")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # Throughput knobs (defaults tuned for ~24 CPU cores / GPU on this 96-core box)
    p.add_argument("--n-envs", type=int, default=440)
    p.add_argument("--n-workers", type=int, default=22)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="entropy bonus; the draw-color head gets 2x (same ratio as the default)")
    p.add_argument("--torch-threads", type=int, default=4,
                   help="intra-op torch threads per process (avoid 4-way oversubscription)")
    p.add_argument("--gpu-mem-gb", type=float, default=0,
                   help="if >0, cap this process's PyTorch GPU memory (to share one GPU)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Pin GPU + reward mode BEFORE importing torch / building env workers ---
    # The reward flag must be in the environment before SubprocVecEnv forks workers.
    # Under torchrun each process gets RANK / LOCAL_RANK / WORLD_SIZE and its own GPU.
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    if args.reward_mode == "monotone":
        os.environ["DVC_MONOTONE_REWARD"] = "1"
    # Limit BLAS / OpenMP threads so 4 concurrent runs don't each grab all cores.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))

    import torch

    import src.utils.logger  # noqa: F401  (configures root logger)
    from src.hooks import NaNDetector
    from src.reward_config import RewardConfig
    from src.trainer import PPOConfig, PPOTrainer

    # Win/lose (+-10, same scale as the default) + potential shaping; every other shaping term off.
    OUTCOME_REWARD = RewardConfig(
        guess_success=0.0, joker_success=0.0, guess_fail=0.0, guess_order_violation=0.0,
        streak_bonus_multiplier=0.0, streak_break=0.0, stop_with_determined=0.0,
        stop_with_near_determined=0.0, draw_win=0.0, draw_lose=0.0,
        continue_success=0.0, continue_fail=0.0, potential=0.5,
    )

    torch.set_num_threads(args.torch_threads)
    if args.gpu_mem_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_gb / total)

    # Single process: cuda:0 is the GPU picked by --device-id (masked above).
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    exp_dir = os.path.join("experiments", args.exp)
    save_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_workers=args.n_workers,
        episodes_per_update=args.n_envs,  # ~1 collection round per update
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        color_ent_coef=2 * args.ent_coef,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        ff_dim=args.ff_dim,
        zero_init=args.zero_init,
        monotone_reward=(args.reward_mode == "monotone"),
        reward_config=OUTCOME_REWARD if args.reward_mode == "outcome" else RewardConfig(),
        save_dir=save_dir,
        log_dir=log_dir,
    )

    if rank == 0:
        print_summary(args, config, exp_dir, world_size)

    trainer = PPOTrainer(config, device, rank=rank, world_size=world_size)
    if rank == 0:
        trainer.register_hook(NaNDetector())

    # Resume if a checkpoint already exists for this experiment.
    latest = os.path.join(save_dir, "latest.pt")
    if os.path.exists(latest):
        trainer.load(latest)
        if rank == 0:
            print(f"Resumed from timestep {trainer.timesteps * world_size:,}")

    trainer.train()
    if world_size > 1:
        torch.distributed.destroy_process_group()


def print_summary(args, config, exp_dir: str, world_size: int) -> None:
    print("=" * 70)
    print(f"Experiment: {args.exp}")
    print(f"  processes (GPUs) : {world_size}")
    print(f"  reward_mode      : {args.reward_mode} (monotone={config.monotone_reward})")
    print(f"  n_heads/n_layers : {config.n_heads} / {config.n_layers}")
    print(f"  zero_init        : {config.zero_init}")
    print(f"  n_envs/n_workers : {config.n_envs} / {config.n_workers}")
    print(f"  batch_size       : {config.batch_size}")
    print(f"  total_timesteps  : {config.total_timesteps:,}")
    print(f"  output dir       : {exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
