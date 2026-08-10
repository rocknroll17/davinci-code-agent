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
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wm_cfg = WMConfig()
    if args.small:
        wm_cfg = WMConfig(deter_dim=128, stoch_discrete=8, stoch_classes=8,
                          hidden=128, embed_dim=128)

    cfg = DreamerConfig(n_envs=args.envs, seed=args.seed, wm=wm_cfg)
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
