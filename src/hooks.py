"""
Training hooks — pluggable callbacks for the PPO training loop.

Usage
-----
from src.hooks import NaNDetector, ActionDiversityMonitor

trainer.register_hook(NaNDetector())
trainer.register_hook(ActionDiversityMonitor(window=100))
trainer.train()

Built-in hooks
--------------
NaNDetector
    Saves an emergency checkpoint and raises RuntimeError when any
    loss component is NaN or ±Inf.

ActionDiversityMonitor
    Warns when the policy collapses to near-deterministic behaviour
    (entropy < threshold for ``window`` consecutive updates).

LossSpike
    Logs a warning when any loss jumps by more than ``factor`` * its
    rolling mean.

CheckpointOnImprovement
    Saves an extra named checkpoint whenever a chosen metric improves
    beyond ``min_delta``.

DashboardHook
    Emits per-update metrics to the ``DashboardServer`` for live
    browser visualisation.  Instantiate after calling
    ``DashboardServer.start()``.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# ActionDiversityMonitor
# ---------------------------------------------------------------------------

class ActionDiversityMonitor(TrainingHook):
    """
    Warn when policy entropy collapses for ``window`` consecutive updates.
    """

    def __init__(self, threshold: float = 0.01, window: int = 50) -> None:
        self.threshold = threshold
        self.window = window
        self._low_entropy_streak = 0

    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        entropy = stats.get("entropy", float("inf"))
        if entropy < self.threshold:
            self._low_entropy_streak += 1
            if self._low_entropy_streak >= self.window:
                logger.warning(
                    "[ActionDiversityMonitor] Policy entropy %.4f < %.4f for %d "
                    "consecutive updates. Possible policy collapse at update %d.",
                    entropy, self.threshold, self._low_entropy_streak,
                    stats["update_count"],
                )
        else:
            self._low_entropy_streak = 0


# ---------------------------------------------------------------------------
# LossSpike
# ---------------------------------------------------------------------------

class LossSpike(TrainingHook):
    """
    Log a warning when a loss component spikes relative to its rolling mean.
    """

    def __init__(self, factor: float = 5.0, window: int = 50) -> None:
        self.factor = factor
        self.window = window
        self._history: Dict[str, deque] = {}

    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        keys = ("policy_loss", "value_loss", "belief_loss")
        for k in keys:
            v = stats.get(k)
            if v is None or not np.isfinite(v):
                continue
            h = self._history.setdefault(k, deque(maxlen=self.window))
            if len(h) >= 5:
                mean_val = float(np.mean(h))
                if mean_val > 0 and v > mean_val * self.factor:
                    logger.warning(
                        "[LossSpike] %s spiked to %.4f (%.1fx rolling mean %.4f) "
                        "at update %d.",
                        k, v, v / mean_val, mean_val, stats["update_count"],
                    )
            h.append(v)


# ---------------------------------------------------------------------------
# CheckpointOnImprovement
# ---------------------------------------------------------------------------

class CheckpointOnImprovement(TrainingHook):
    """
    Save an extra checkpoint whenever *metric* improves by at least *min_delta*.

    Parameters
    ----------
    metric : str
        Key in ``eval_stats`` (e.g. ``"player0_win_rate"``).
    save_path : str
        Destination path for the improved checkpoint.
    higher_is_better : bool
        Set to ``False`` to treat lower values as improvement (e.g. loss).
    min_delta : float
        Minimum change to count as improvement.
    """

    def __init__(
        self,
        metric: str = "player0_win_rate",
        save_path: str = "checkpoints/best_improved.pt",
        higher_is_better: bool = True,
        min_delta: float = 0.001,
    ) -> None:
        self.metric = metric
        self.save_path = save_path
        self.higher_is_better = higher_is_better
        self.min_delta = min_delta
        self._best: Optional[float] = None

    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        pass  # operates only on eval data

    def on_eval_end(self, trainer: Any, eval_stats: Dict[str, Any]) -> None:
        value = eval_stats.get(self.metric)
        if value is None:
            return
        improved = (
            self._best is None
            or (self.higher_is_better and value > self._best + self.min_delta)
            or (not self.higher_is_better and value < self._best - self.min_delta)
        )
        if improved:
            self._best = value
            trainer.save(self.save_path)
            logger.info(
                "[CheckpointOnImprovement] %s improved to %.4f — saved to %s",
                self.metric, value, self.save_path,
            )


# ---------------------------------------------------------------------------
# DashboardHook
# ---------------------------------------------------------------------------

class DashboardHook(TrainingHook):
    """
    Emit per-update metrics to the web ``DashboardServer``.

    Parameters
    ----------
    server : DashboardServer
        A started ``DashboardServer`` instance (from ``src.dashboard.server``).
    """

    def __init__(self, server: Any) -> None:
        self.server = server

    def on_update_end(self, trainer: Any, stats: Dict[str, Any]) -> None:
        self.server.emit({
            "type":                "update",
            "update":              stats.get("update_count"),
            "timesteps":           stats.get("timesteps"),
            "episodes":            stats.get("episodes"),
            "mean_reward":         stats.get("mean_reward"),
            "mean_episode_length": stats.get("mean_episode_length"),
            "policy_loss":         stats.get("policy_loss"),
            "value_loss":          stats.get("value_loss"),
            "belief_loss":         stats.get("belief_loss"),
            "entropy":             stats.get("entropy"),
            "total_loss":          stats.get("total_loss"),
            "learning_rate":       stats.get("learning_rate"),
        })

    def on_eval_end(self, trainer: Any, eval_stats: Dict[str, Any]) -> None:
        self.server.emit({
            "type":               "eval",
            "update":             trainer.updates,
            "timesteps":          trainer.timesteps,
            "player0_win_rate":   eval_stats.get("player0_win_rate"),
            "player1_win_rate":   eval_stats.get("player1_win_rate"),
            "mean_episode_length": eval_stats.get("mean_episode_length"),
        })

    def on_training_end(self, trainer: Any) -> None:
        self.server.emit({"type": "done", "timesteps": trainer.timesteps})
