#!/usr/bin/env python3
"""
Da Vinci Code Self-Play Training
python main.py 실행하면 바로 학습 시작 (checkpoints/latest.pt 가 있으면 이어서 학습)
python main.py --reset-optimizer 로 optimizer 초기화 후 학습
"""

import os
import sys

import torch

import src.utils.logger  # noqa: F401  (configures root logger on import)
from src.hooks import NaNDetector
from src.trainer import PPOConfig, PPOTrainer

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


def main():
    reset_optimizer = "--reset-optimizer" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = PPOConfig(save_dir=CHECKPOINT_DIR, reset_optimizer_on_load=reset_optimizer)
    trainer = PPOTrainer(config, device)
    trainer.register_hook(NaNDetector())

    print(f"Device: {device}")
    if os.path.exists(CHECKPOINT_PATH):
        trainer.load(CHECKPOINT_PATH)
        print(f"Resumed from timestep {trainer.timesteps}")
    else:
        print("Starting fresh training")

    trainer.train()


if __name__ == "__main__":
    main()
