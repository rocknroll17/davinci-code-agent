#!/usr/bin/env python3
"""
Da Vinci Code world-model (DreamerV3-style) training.

    python train_wm.py                     # 기본 설정으로 학습
    python train_wm.py --rounds 500        # 수집 라운드 수 지정
    python train_wm.py --resume            # checkpoints_wm/dreamer_latest.pt 이어서
    python train_wm.py --small             # CPU 스모크용 소형 모델
    python train_wm.py --large             # 24GB GPU용 논문 크기 (32x32 latents)

Multi-GPU (one process per GPU, gradients averaged across ranks):

    torchrun --standalone --nproc_per_node=2 train_wm.py --large --envs 64

Each rank collects with its own envs/replay (per-rank seeds) and the models
stay synchronized via gradient all-reduce (same convention as train_ddp.py).
Pin GPUs with CUDA_VISIBLE_DEVICES=2,3 if some devices are busy.

The agent is trained entirely from imagined rollouts inside a learned RSSM;
the real (joker_control) environment is only used to collect replay data.
"""

import argparse
import os

import torch
import torch.distributed as dist

import src.utils.logger  # noqa: F401
from src.wm.dreamer import DreamerConfig, DreamerTrainer
from src.wm.nets import WMConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--small", action="store_true", help="tiny nets for CPU smoke")
    ap.add_argument("--large", action="store_true",
                    help="paper-size nets (32x32 latents) + bigger batches; for 24GB GPUs")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--updates", type=int, default=None,
                    help="WM and AC updates per collection round")
    ap.add_argument("--episodes-per-round", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=None,
                    help="parallel environment worker processes per rank")
    ap.add_argument("--entropy-scale", type=float, default=None,
                    help="actor entropy bonus (default 3e-4; raise to fight collapse)")
    ap.add_argument("--prefill", type=int, default=None,
                    help="random prefill episodes per rank")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="rounds between head-to-head evals vs the deployed model (0=off)")
    ap.add_argument("--eval-games", type=int, default=None)
    ap.add_argument("--eval-opponent", type=str, default=None,
                    help="legacy PPO checkpoint to evaluate against")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # torchrun env (defaults keep plain `python train_wm.py` single-process)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # All ranks must agree on device kind + backend: use GPUs only when every
    # rank can have its own (otherwise fall back to CPU/gloo, e.g. local tests).
    use_cuda = torch.cuda.is_available() and world_size <= torch.cuda.device_count()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        # DDP + seed: give ranks different data but deterministic per rank
        if args.seed is None:
            args.seed = 0

    wm_cfg = WMConfig()
    if args.small:
        wm_cfg = WMConfig(deter_dim=128, stoch_discrete=8, stoch_classes=8,
                          hidden=128, embed_dim=128)
    elif args.large:
        # DreamerV3 paper latents (32x32) + wider nets — ~24GB GPU territory
        wm_cfg = WMConfig(deter_dim=1024, stoch_discrete=32, stoch_classes=32,
                          hidden=1024, embed_dim=1024)

    cfg = DreamerConfig(n_envs=args.envs, seed=args.seed, wm=wm_cfg)
    if args.large:
        cfg.batch_size = 32
        cfg.seq_len = 64
        cfg.episodes_per_round = 64
        cfg.wm_updates_per_round = 100
        cfg.ac_updates_per_round = 100
        cfg.prefill_episodes = 256
        cfg.replay_capacity = 20000
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.updates is not None:
        cfg.wm_updates_per_round = args.updates
        cfg.ac_updates_per_round = args.updates
    if args.episodes_per_round is not None:
        cfg.episodes_per_round = args.episodes_per_round
    if args.n_workers is not None:
        cfg.n_workers = args.n_workers
    if args.entropy_scale is not None:
        cfg.entropy_scale = args.entropy_scale
    if args.prefill is not None:
        cfg.prefill_episodes = args.prefill
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games
    if args.eval_opponent is not None:
        cfg.eval_opponent = args.eval_opponent
    trainer = DreamerTrainer(cfg, device, rank=rank, world_size=world_size)

    ckpt = os.path.join(cfg.save_dir, "dreamer_latest.pt")
    if args.resume and os.path.exists(ckpt):
        trainer.load(ckpt)
        if trainer.is_main:
            print(f"Resumed from {ckpt} (steps={world_size * trainer.total_env_steps:,})")

    if trainer.is_main:
        print(f"Device: {device} (world_size={world_size}) | params: "
              f"wm={sum(p.numel() for p in trainer.wm.parameters()):,}, "
              f"actor={sum(p.numel() for p in trainer.actor.parameters()):,}")
    try:
        trainer.train(rounds=args.rounds)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
