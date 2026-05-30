#!/usr/bin/env python3
"""
Da Vinci Code Self-Play Training
python main.py 실행하면 바로 학습 시작
python main.py --no-viz 로 시각화 없이 학습
python main.py --finetune 로 특수 케이스 파인튜닝 학습
python main.py --reset-optimizer 로 optimizer 초기화 후 학습 (파인튜닝 후 일반학습 전환 시 사용)
python main.py --dashboard 로 브라우저 대시보드 활성화 (기본: 포트 6006)
python main.py --dashboard-port 8080 로 포트 변경
"""

import os
import sys
import torch

import src.utils.logger  # configure root logger early
from src.trainer import PPOTrainer, PPOConfig
from src.visualizer import create_visualizer, get_visualizer

# ============== 설정 ==============
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


def main():
    # 옵션 파싱
    use_viz = "--no-viz" not in sys.argv
    use_finetune = "--finetune" in sys.argv
    reset_optimizer = "--reset-optimizer" in sys.argv
    use_dashboard = "--dashboard" in sys.argv

    # --dashboard-port <n>
    dashboard_port = 6006
    if "--dashboard-port" in sys.argv:
        idx = sys.argv.index("--dashboard-port")
        if idx + 1 < len(sys.argv):
            dashboard_port = int(sys.argv[idx + 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = PPOConfig(
        save_dir=CHECKPOINT_DIR,
        finetune=use_finetune,
        reset_optimizer_on_load=reset_optimizer
    )
    
    trainer = PPOTrainer(config, device)
    
    # 체크포인트 있으면 불러오기
    if os.path.exists(CHECKPOINT_PATH):
        trainer.load(CHECKPOINT_PATH)
        msg = f"Resumed from timestep {trainer.timesteps}"
    else:
        msg = "Starting fresh training"
    
    # Finetune 모드 메시지
    finetune_msg = "🎯 Finetune mode: ON (특수 케이스 학습)" if use_finetune else "Finetune mode: OFF"
    optimizer_msg = "🔄 Optimizer reset: ON" if reset_optimizer else ""

    # ── 대시보드 ──────────────────────────────────────────────────────────────
    if use_dashboard:
        try:
            from src.dashboard.server import DashboardServer
            from src.hooks import DashboardHook, NaNDetector

            dashboard = DashboardServer(host="0.0.0.0", port=dashboard_port)
            dashboard.start()
            trainer.register_hook(DashboardHook(dashboard))
            trainer.register_hook(NaNDetector())
            print(f"[Dashboard] http://localhost:{dashboard_port}")
        except ImportError as e:
            print(f"Warning: Could not start dashboard (missing dependencies: {e}). "
                  "Install with: pip install fastapi uvicorn[standard]")
            use_dashboard = False
    else:
        # Even without dashboard, register NaN detector by default
        from src.hooks import NaNDetector
        trainer.register_hook(NaNDetector())

    # 시각화 시작
    viz = None
    if use_viz:
        try:
            viz = create_visualizer()
            viz.start()
            viz.add_log(msg)
            viz.add_log(f"Device: {device}")
            viz.add_log(finetune_msg)
            if optimizer_msg:
                viz.add_log(optimizer_msg)
            if use_dashboard:
                viz.add_log(f"Dashboard: http://localhost:{dashboard_port}")
        except Exception as e:
            print(f"Warning: Could not start visualizer: {e}")
            use_viz = False
    
    if not use_viz:
        print(f"Device: {device}")
        print(msg)
        print(finetune_msg)
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
