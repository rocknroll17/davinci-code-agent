#!/usr/bin/env python3
"""
Da Vinci Code Self-Play Training
python main.py 실행하면 바로 학습 시작
python main.py --no-viz 로 시각화 없이 학습
python main.py --reset-optimizer 로 optimizer 초기화 후 학습
"""

import os
import sys

import torch

import src.utils.logger  # noqa: F401  (configures root logger on import)
from src.hooks import NaNDetector
from src.trainer import PPOConfig, PPOTrainer
from src.visualizer import create_visualizer

# ============== 설정 ==============
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


def main():
    # 옵션 파싱
    use_viz = "--no-viz" not in sys.argv
    reset_optimizer = "--reset-optimizer" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = PPOConfig(
        save_dir=CHECKPOINT_DIR,
        reset_optimizer_on_load=reset_optimizer
    )
    
    trainer = PPOTrainer(config, device)
    
    # 체크포인트 있으면 불러오기
    if os.path.exists(CHECKPOINT_PATH):
        trainer.load(CHECKPOINT_PATH)
        msg = f"Resumed from timestep {trainer.timesteps}"
    else:
        msg = "Starting fresh training"
    
    optimizer_msg = "🔄 Optimizer reset: ON" if reset_optimizer else ""

    trainer.register_hook(NaNDetector())

    # 시각화 시작
    viz = None
    if use_viz:
        try:
            viz = create_visualizer()
            viz.start()
            viz.add_log(msg)
            viz.add_log(f"Device: {device}")
            if optimizer_msg:
                viz.add_log(optimizer_msg)
        except Exception as e:
            print(f"Warning: Could not start visualizer: {e}")
            use_viz = False
    
    if not use_viz:
        print(f"Device: {device}")
        print(msg)
        if optimizer_msg:
            print(optimizer_msg)
    
    try:
        # 학습
        trainer.train()
    finally:
        # 시각화 종료
        if viz:
            viz.stop()


if __name__ == "__main__":
    main()
