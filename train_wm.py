#!/usr/bin/env python3
"""
Da Vinci Code world-model (DreamerV3-style) training.

    python train_wm.py                     # 기본 설정으로 학습
    python train_wm.py --rounds 500        # 수집 라운드 수 지정
    python train_wm.py --resume            # checkpoints_wm/dreamer_latest.pt 이어서
    python train_wm.py --small             # CPU 스모크용 소형 모델

The agent is trained entirely from imagined rollouts inside a learned RSSM;
the real (joker_control) environment is only used to collect replay data.
"""

import argparse
import os

import torch

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
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    trainer = DreamerTrainer(cfg, device)

    ckpt = os.path.join(cfg.save_dir, "dreamer_latest.pt")
    if args.resume and os.path.exists(ckpt):
        trainer.load(ckpt)
        print(f"Resumed from {ckpt} (steps={trainer.total_env_steps:,})")

    print(f"Device: {device} | params: "
          f"wm={sum(p.numel() for p in trainer.wm.parameters()):,}, "
          f"actor={sum(p.numel() for p in trainer.actor.parameters()):,}")
    trainer.train(rounds=args.rounds)


if __name__ == "__main__":
    main()
