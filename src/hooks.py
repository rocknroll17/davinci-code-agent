"""
Training hooks — pluggable callbacks for the PPO training loop.

Usage
-----
from src.hooks import NaNDetector

trainer.register_hook(NaNDetector())
trainer.train()

Built-in hooks
--------------
NaNDetector
    Saves an emergency checkpoint and raises RuntimeError when any
    loss component is NaN or ±Inf.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrainingHook(ABC):
    """
    Abstract base for training callbacks.

    The trainer calls ``on_update_end`` after every PPO update with a
    ``stats`` dict containing at least:

        update_count       int
        timesteps          int
        episodes           int
        mean_reward        float
        mean_episode_length float
        policy_loss        float
        value_loss         float
        belief_loss        float
        entropy            float
        total_loss         float
        learning_rate      float
    """

    @abstractmethod
    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        """Called once per PPO update after gradients have been applied."""

    def on_eval_end(self, trainer: Any, eval_stats: Dict[str, Any]) -> None:
        """Called after each evaluation run (optional override)."""

    def on_training_end(self, trainer: Any) -> None:
        """Called when ``trainer.train()`` finishes (optional override)."""


# ---------------------------------------------------------------------------
# NaNDetector
# ---------------------------------------------------------------------------

class NaNDetector(TrainingHook):
    """
    Halt training and save an emergency checkpoint when any loss is NaN/Inf.
    """

    def __init__(self, save_dir: str = "checkpoints/emergency") -> None:
        self.save_dir = save_dir

    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        bad_keys = [
            k for k in ("policy_loss", "value_loss", "belief_loss", "entropy", "total_loss")
            if not np.isfinite(stats.get(k, 0.0))
        ]
        if bad_keys:
            os.makedirs(self.save_dir, exist_ok=True)
            ckpt = os.path.join(self.save_dir, f"nan_at_{stats['update_count']}.pt")
            try:
                trainer.save(ckpt)
                msg = f"Emergency checkpoint saved to {ckpt}"
            except Exception as e:
                msg = f"Emergency checkpoint save failed: {e}"
            raise RuntimeError(
                f"[NaNDetector] NaN/Inf detected in {bad_keys} at update "
                f"{stats['update_count']}. {msg}"
            )
