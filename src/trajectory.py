"""
Trajectory — record, save, load, and replay full game episodes.

Usage
-----
Recording during play
~~~~~~~~~~~~~~~~~~~~~
from src.trajectory import TrajectoryRecorder

recorder = TrajectoryRecorder()
obs, info = env.reset()
while not done:
    action_mask = env.get_action_mask()
    action, probs = agent.act(obs, action_mask)
    next_obs, _, reward, terminated, truncated, info, result = env.step(action)
    recorder.record(obs, action_mask, action, probs, reward, result, info)
    obs = next_obs
    done = terminated or truncated
recorder.finish(winner=info.get("winner"))
recorder.save("trajectories/episode_001.json")

Replay and analysis
~~~~~~~~~~~~~~~~~~~
from src.trajectory import Trajectory

traj = Trajectory.load("trajectories/episode_001.json")
print(traj.summary())

for step in traj.steps:
    print(step.phase, step.action, step.reward)

# Replay with a different model
for step in traj.steps:
    new_action, _ = new_agent.act(step.obs, step.action_mask)
    if not np.array_equal(new_action, step.action):
        print(f"Step {step.index}: old={step.action}  new={new_action}")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Step data
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """All data captured at a single environment step."""

    index: int
    phase: str                          # "DRAW" / "GUESS" / "DECISION"
    current_player: int
    obs: Dict[str, np.ndarray]
    action_mask: Optional[Dict[str, np.ndarray]]
    action: np.ndarray                  # shape (4,): [color, pos, value, decision]
    action_probs: Dict[str, float]      # per-head prob of chosen action
    reward: float
    result_type: str                    # "DrawResult" / "GuessResult" / etc.
    result_info: Dict[str, Any]         # extra fields from the result object

    # Serialization helpers
    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "phase": self.phase,
            "current_player": self.current_player,
            "obs": {k: v.tolist() for k, v in self.obs.items()},
            "action_mask": (
                {k: v.tolist() for k, v in self.action_mask.items()}
                if self.action_mask else None
            ),
            "action": self.action.tolist(),
            "action_probs": self.action_probs,
            "reward": self.reward,
            "result_type": self.result_type,
            "result_info": self.result_info,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(
            index=d["index"],
            phase=d["phase"],
            current_player=d["current_player"],
            obs={k: np.array(v, dtype=np.int8) for k, v in d["obs"].items()},
            action_mask=(
                {k: np.array(v) for k, v in d["action_mask"].items()}
                if d.get("action_mask") else None
            ),
            action=np.array(d["action"], dtype=np.int64),
            action_probs=d.get("action_probs", {}),
            reward=d["reward"],
            result_type=d.get("result_type", ""),
            result_info=d.get("result_info", {}),
        )


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """
    Accumulates steps during a live episode and produces a ``Trajectory``.

    Call ``record()`` after each ``env.step()``, then ``finish()`` when the
    episode ends, then ``save()`` / ``get()`` as needed.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._steps: List[TrajectoryStep] = []
        self._metadata: Dict[str, Any] = metadata or {}
        self._start_time = datetime.utcnow().isoformat()
        self._winner: Optional[int] = None

    def record(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]],
        action: np.ndarray,
        action_probs: Optional[Dict[str, float]],
        reward: float,
        result: Any,
        info: Dict[str, Any],
    ) -> None:
        """Append one step to the recording buffer."""
        phase_onehot = obs.get("phase", np.array([1, 0, 0]))
        phase_names = ["DRAW", "GUESS", "DECISION"]
        phase = phase_names[int(np.argmax(phase_onehot))]

        result_type = type(result).__name__ if result is not None else ""
        result_info: Dict[str, Any] = {}
        if result is not None:
            for attr in ("is_invalid", "is_correct", "is_continue",
                         "position", "guessed_value", "player_id"):
                if hasattr(result, attr):
                    v = getattr(result, attr)
                    # Convert enums / numpy types to plain Python
                    result_info[attr] = int(v) if hasattr(v, "__int__") and not isinstance(v, bool) else v

        step = TrajectoryStep(
            index=len(self._steps),
            phase=phase,
            current_player=info.get("current_player", -1),
            obs={k: v.copy() for k, v in obs.items()},
            action_mask=(
                {k: v.copy() for k, v in action_mask.items()} if action_mask else None
            ),
            action=np.array(action, dtype=np.int64),
            action_probs=action_probs or {},
            reward=float(reward),
            result_type=result_type,
            result_info=result_info,
        )
        self._steps.append(step)

    def finish(self, winner: Optional[int] = None) -> None:
        """Mark the episode as finished and record the winner."""
        self._winner = winner

    def get(self) -> "Trajectory":
        """Return the completed ``Trajectory`` object."""
        return Trajectory(
            steps=list(self._steps),
            winner=self._winner,
            metadata={
                **self._metadata,
                "start_time": self._start_time,
                "end_time": datetime.utcnow().isoformat(),
                "n_steps": len(self._steps),
            },
        )

    def save(self, path: str) -> str:
        """Save directly to ``path`` and return the path."""
        traj = self.get()
        traj.save(path)
        return path

    def reset(self) -> None:
        """Clear all recorded steps (reuse the recorder for the next episode)."""
        self._steps.clear()
        self._winner = None
        self._start_time = datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class Trajectory:
    """
    Immutable record of a complete game episode.

    Attributes
    ----------
    steps : list[TrajectoryStep]
    winner : int | None
    metadata : dict
    """

    def __init__(
        self,
        steps: List[TrajectoryStep],
        winner: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.steps = steps
        self.winner = winner
        self.metadata = metadata or {}

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """Serialise to JSON at ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "winner": self.winner,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        """Deserialise from a JSON file produced by ``save()``."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            steps=[TrajectoryStep.from_dict(s) for s in data["steps"]],
            winner=data.get("winner"),
            metadata=data.get("metadata", {}),
        )

    # ---------- analysis helpers ----------

    def summary(self) -> str:
        """Human-readable episode summary."""
        n = len(self.steps)
        total_r = sum(s.reward for s in self.steps)
        phases = [s.phase for s in self.steps]
        guess_steps = [s for s in self.steps if s.phase == "GUESS"]
        correct = sum(1 for s in guess_steps if s.result_info.get("is_correct"))
        lines = [
            f"Trajectory: {n} steps, winner={self.winner}, total_reward={total_r:.3f}",
            f"  DRAW={phases.count('DRAW')}  GUESS={phases.count('GUESS')}  DECISION={phases.count('DECISION')}",
            f"  Guess success: {correct}/{len(guess_steps)}",
        ]
        if self.metadata:
            lines.append(f"  Meta: {self.metadata}")
        return "\n".join(lines)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def guess_accuracy(self) -> float:
        guesses = [s for s in self.steps if s.phase == "GUESS"]
        if not guesses:
            return 0.0
        correct = sum(1 for s in guesses if s.result_info.get("is_correct"))
        return correct / len(guesses)

    def filter_phase(self, phase: str) -> List[TrajectoryStep]:
        """Return only steps in the given phase (``'DRAW'``, ``'GUESS'``, ``'DECISION'``)."""
        return [s for s in self.steps if s.phase == phase.upper()]

    def compare_with_agent(self, agent) -> List[dict]:
        """
        Re-run every step through ``agent`` and return diffs where the new
        agent would have chosen a different action.

        Parameters
        ----------
        agent : ModelAgent

        Returns
        -------
        list[dict]  — one entry per diverging step with keys:
            ``index``, ``phase``, ``old_action``, ``new_action``
        """
        diffs = []
        for step in self.steps:
            new_action, _ = agent.act(step.obs, step.action_mask)
            if not np.array_equal(new_action, step.action):
                diffs.append({
                    "index": step.index,
                    "phase": step.phase,
                    "old_action": step.action.tolist(),
                    "new_action": new_action.tolist(),
                })
        return diffs
