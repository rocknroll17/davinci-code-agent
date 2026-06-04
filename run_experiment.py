#!/usr/bin/env python3
"""
Single-experiment training runner for the 4-way comparison study.

Each experiment is one independent self-play training run, pinned to one GPU,
writing its own checkpoints/logs under experiments/<name>/. The four variants:

  1. baseline   — original model, original reward
  2. monotone   — monotone reward (no win/lose signal, only per-guess reward)
  3. heads      — encoder transformer with more attention heads
  4. layers     — encoder transformer with an extra layer

Nothing else is changed between runs.

Example
-------
  python run_experiment.py --exp baseline --device-id 0
  python run_experiment.py --exp monotone --reward-mode monotone --device-id 1
  python run_experiment.py --exp heads --n-heads 8 --device-id 2
  python run_experiment.py --exp layers --n-layers 5 --device-id 3
"""

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Da Vinci Code experiment runner")
    p.add_argument("--exp", required=True, help="experiment name (output dir)")
    p.add_argument("--reward-mode", choices=["normal", "monotone"], default="normal")
    p.add_argument("--n-heads", type=int, default=4, help="encoder attention heads")
    p.add_argument("--n-layers", type=int, default=4, help="encoder transformer layers")
    p.add_argument("--zero-init", action="store_true",
                   help="initialize all default weights/biases to 0 (designated inits kept)")
    p.add_argument("--device-id", type=int, default=0, help="CUDA device index")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # Throughput knobs (defaults tuned for ~24 CPU cores / GPU on this 96-core box)
    p.add_argument("--n-envs", type=int, default=440)
    p.add_argument("--n-workers", type=int, default=22)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--torch-threads", type=int, default=4,
                   help="intra-op torch threads per process (avoid 4-way oversubscription)")
    p.add_argument("--dashboard-port", type=int, default=0,
                   help="if >0, serve the live web dashboard on this port")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Pin GPU + reward mode BEFORE importing torch / building env workers ---
    # The reward flag must be in the environment before SubprocVecEnv forks workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    if args.reward_mode == "monotone":
        os.environ["DVC_MONOTONE_REWARD"] = "1"
    # Limit BLAS / OpenMP threads so 4 concurrent runs don't each grab all cores.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))

    import torch

    import src.utils.logger  # noqa: F401  (configures root logger)
    from src.hooks import NaNDetector
    from src.trainer import PPOConfig, PPOTrainer

    torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 0 == the masked device

    exp_dir = os.path.join("experiments", args.exp)
    save_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_workers=args.n_workers,
        episodes_per_update=args.n_envs,  # ~1 collection round per update
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        zero_init=args.zero_init,
        monotone_reward=(args.reward_mode == "monotone"),
        save_dir=save_dir,
        log_dir=log_dir,
    )

    print("=" * 70)
    print(f"Experiment: {args.exp}")
    print(f"  device           : cuda:{args.device_id}")
    print(f"  reward_mode      : {args.reward_mode} (monotone={config.monotone_reward})")
    print(f"  n_heads/n_layers : {config.n_heads} / {config.n_layers}")
    print(f"  zero_init        : {config.zero_init}")
    print(f"  n_envs/n_workers : {config.n_envs} / {config.n_workers}")
    print(f"  batch_size       : {config.batch_size}")
    print(f"  total_timesteps  : {config.total_timesteps:,}")
    print(f"  output dir       : {exp_dir}")
    print("=" * 70)

    trainer = PPOTrainer(config, device)
    trainer.register_hook(NaNDetector())

    # Live web dashboard (per-experiment port) so training is watchable in a browser.
    if args.dashboard_port > 0:
        try:
            from src.dashboard.server import DashboardServer
            from src.hooks import DashboardHook
            dashboard = DashboardServer(host="0.0.0.0", port=args.dashboard_port)
            dashboard.start()
            trainer.register_hook(DashboardHook(dashboard))
            print(f"[Dashboard] {args.exp}: http://localhost:{args.dashboard_port}")
        except Exception as e:
            print(f"[Dashboard] disabled ({e})")

    # Resume if a checkpoint already exists for this experiment.
    latest = os.path.join(save_dir, "latest.pt")
    if os.path.exists(latest):
        trainer.load(latest)
        print(f"Resumed from timestep {trainer.timesteps:,}")

    trainer.train()


if __name__ == "__main__":
    main()
