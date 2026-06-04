"""
run_episode — the single canonical "play one full game" loop.

Before this, the reset→while-not-done→pick-action→step loop was copy-pasted in
``PPOTrainer.evaluate``, ``EvalSuite.run_policy`` and ``compare_experiments.play_match``.
They differed only in: single-agent self-play vs two-agent head-to-head, and how
much per-step statistics they collected. ``run_episode`` unifies the loop and
exposes an ``on_step`` callback so callers inject their own stats without
duplicating control flow.

(``play.py`` keeps its own interactive loop — different concern.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from src.interfaces import Agent


@dataclass
class StepContext:
    """Everything a stats hook might need about one executed move."""
    env: Any
    player: int                 # who acted (current_player before the move)
    phase: int                  # Phase value of the move
    obs_before: Dict[str, np.ndarray]
    info_before: Dict[str, Any]
    action: np.ndarray
    reward: float
    result: Any                 # Result subclass from env.step (or None)
    info_after: Dict[str, Any]
    done: bool


@dataclass
class EpisodeResult:
    winner: Optional[int]
    length: int
    rewards: List[float]        # [player0_total, player1_total]


def run_episode(
    env: Any,
    agents: Union[Agent, Sequence[Agent]],
    *,
    deterministic: bool = True,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[StepContext], None]] = None,
    max_steps: int = 1000,
) -> EpisodeResult:
    """Play one full game.

    Args:
        env: a DaVinciCodeEnv.
        agents: a single Agent (self-play, used for both players) OR a 2-sequence
                ``[agent_p0, agent_p1]`` for head-to-head.
        deterministic: argmax vs sampling for each agent's ``act``.
        seed: passed to ``env.reset(seed=...)`` if given.
        on_step: optional callback invoked once per executed move with a
                 ``StepContext`` — used by rich evaluators to accumulate stats.
        max_steps: safety cap.

    Returns:
        EpisodeResult(winner, length, [reward_p0, reward_p1]).
    """
    # Normalize to a per-player pair: a single agent plays both seats (self-play).
    pair: Sequence[Agent] = agents if isinstance(agents, (list, tuple)) else (agents, agents)

    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    rewards = [0.0, 0.0]
    length = 0
    done = False

    while not done and length < max_steps:
        player = info.get("current_player", 0)
        agent = pair[player]
        mask = env.get_action_mask()
        action, _ = agent.act(obs, mask, deterministic=deterministic)

        phase = int(np.argmax(obs["phase"]))
        obs_before, info_before = obs, info

        # env.step → (obs, render_obs, reward, terminated, truncated, info, result)
        obs, _render, reward, terminated, truncated, info, result = env.step(action)
        done = terminated or truncated

        rewards[player] += float(reward)
        length += 1

        if on_step is not None:
            on_step(StepContext(
                env=env, player=player, phase=phase,
                obs_before=obs_before, info_before=info_before,
                action=np.asarray(action), reward=float(reward),
                result=result, info_after=info, done=done,
            ))

    return EpisodeResult(winner=info.get("winner"), length=length, rewards=rewards)
