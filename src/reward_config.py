"""
RewardConfig — all reward/penalty magnitudes in one injectable object.

Previously these lived as 16 module-level constants in ``src.constants`` and were
imported directly by the environment and the Episode class. That made the reward
scheme global and un-tunable per experiment. ``RewardConfig`` groups them so they
can be passed into ``DaVinciCodeEnv`` and ``Episode`` (and through ``PPOConfig``),
enabling reward / discount experiments without editing globals.

Backward compatibility: the default values are exactly the current constants
(``src/constants.py``), and every consumer falls back to ``RewardConfig()`` when
none is supplied — so existing behaviour is unchanged unless a config is given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Reward magnitudes (defaults == current src.constants values)."""

    # --- terminal / outcome ---
    win: float = 10.0                     # REWARD_WIN (env, winning guess)
    lose: float = -10.0                   # REWARD_LOSE (episode, loser's last move)

    # --- per-guess (immediate, env) ---
    guess_success: float = 0.5           # REWARD_GUESS_SUCCESS
    joker_success: float = 1.0           # REWARD_JOKER_SUCCESS
    guess_fail: float = -0.5             # REWARD_GUESS_FAIL
    guess_order_violation: float = -0.5  # REWARD_GUESS_ORDER_VIOLATION
    streak_bonus_multiplier: float = 0.2  # REWARD_STREAK_BONUS_MULTIPLIER
    streak_break: float = -0.1           # REWARD_STREAK_BREAK
    invalid_action: float = -1.0         # REWARD_INVALID_ACTION

    # --- decision phase (env) ---
    stop_decision: float = 0.0           # REWARD_STOP_DECISION
    stop_with_determined: float = -0.5   # REWARD_STOP_WITH_DETERMINED
    stop_with_near_determined: float = -0.15  # REWARD_STOP_WITH_NEAR_DETERMINED

    # --- retroactive (episode, applied at game end) ---
    draw_win: float = 0.1                # REWARD_DRAW_WIN
    draw_lose: float = -0.1              # REWARD_DRAW_LOSE
    continue_success: float = 0.4        # REWARD_CONTINUE_SUCCESS
    continue_fail: float = -0.25         # REWARD_CONTINUE_FAIL

    def to_dict(self) -> dict:
        """Plain-dict form for serialization (checkpoint config / JSON logs)."""
        return asdict(self)
