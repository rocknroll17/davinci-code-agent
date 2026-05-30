"""CI smoke test: run one PPO update on CPU with a tiny config.

Exercises env + model forward + belief module + GAE + backward in a single
update. No checkpoint or GPU required, so it runs on a stock CI runner.
"""
import math
import os
import sys

# Make the repo root importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.trainer import PPOTrainer, PPOConfig


def main() -> None:
    torch.manual_seed(0)
    cfg = PPOConfig(
        n_envs=1,
        episodes_per_update=2,
        batch_size=64,
        n_epochs=1,
        eval_interval=10**9,
        save_interval=10**9,
        log_interval=10**9,
        save_dir="/tmp/ci_ckpt",
        log_dir="/tmp/ci_log",
    )
    trainer = PPOTrainer(cfg, torch.device("cpu"))
    trainer.collect_rollouts()
    stats = trainer.update()

    total = stats["total_loss"]
    assert not math.isnan(total), "total_loss is NaN"
    print(
        f"[smoke] one PPO update OK — "
        f"total_loss={total:.4f}, belief_loss={stats.get('belief_loss')}"
    )


if __name__ == "__main__":
    main()
