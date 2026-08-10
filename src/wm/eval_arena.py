"""
Head-to-head evaluation: world-model agent vs the deployed (legacy) PPO model.

The legacy checkpoint was trained under the OLD rules — random joker placement,
3-dim phase one-hot, 4-component actions — so it cannot act in the JOKER phase
of a joker_control game. ``LegacyOpponentAdapter`` bridges the gap:

- JOKER phase on the legacy side → a RANDOM valid insert index. From the legacy
  model's perspective this reproduces exactly the rule it was trained under.
- Other phases → the 4-dim phase one-hot is sliced to the legacy 3 dims and the
  legacy 4-component action is padded to 5.

The world-model agent keeps its full joker control, so the measured win rate
includes whatever edge deliberate joker placement provides.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np

from src.env import DaVinciCodeEnv
from src.runner import run_episode


class LegacyOpponentAdapter:
    """Wraps a legacy ``ModelAgent`` so it can play in a joker_control env."""

    def __init__(self, inner, rng: Optional[random.Random] = None) -> None:
        self.inner = inner
        self.rng = rng or random.Random(0)

    def act(self, obs, action_mask=None, deterministic: bool = False):
        if int(np.argmax(obs["phase"])) == 3:  # JOKER phase → random placement
            valid = np.flatnonzero(action_mask["joker"]) if action_mask else [0]
            joker_pos = int(self.rng.choice(list(valid)))
            return np.array([0, 0, 0, 0, joker_pos], dtype=np.int64), {}

        legacy_obs = dict(obs)
        legacy_obs["phase"] = np.asarray(obs["phase"])[:3]
        legacy_mask = {k: v for k, v in (action_mask or {}).items() if k != "joker"}
        action, probs = self.inner.act(legacy_obs, legacy_mask or None,
                                       deterministic=deterministic)
        action5 = np.zeros(5, dtype=np.int64)
        action5[:4] = np.asarray(action)[:4]
        return action5, probs


def evaluate_vs_legacy(
    wm_agent,
    legacy_agent,
    n_games: int = 200,
    seed0: int = 0,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> Dict[str, float]:
    """Play n_games with alternating seats and per-game seeds.

    Returns win rates from the WORLD MODEL's perspective (draws count as
    losses for both — they are rare and excluded from the winner field).
    """
    env = DaVinciCodeEnv(viewer=None, joker_control=True)
    wins = 0
    seat_wins = [0, 0]     # wins when WM sits as P0 / P1
    seat_games = [0, 0]
    lengths = []

    for g in range(n_games):
        wm_seat = g % 2
        if hasattr(wm_agent, "reset"):
            wm_agent.reset()
        pair = [None, None]
        pair[wm_seat] = wm_agent
        pair[1 - wm_seat] = legacy_agent

        res = run_episode(env, pair, deterministic=deterministic,
                          seed=seed0 + g, max_steps=max_steps)
        seat_games[wm_seat] += 1
        if res.winner == wm_seat:
            wins += 1
            seat_wins[wm_seat] += 1
        lengths.append(res.length)

    return {
        "eval/win_rate": wins / n_games,
        "eval/win_rate_p0": seat_wins[0] / max(1, seat_games[0]),
        "eval/win_rate_p1": seat_wins[1] / max(1, seat_games[1]),
        "eval/mean_length": float(np.mean(lengths)),
        "eval/n_games": n_games,
    }
