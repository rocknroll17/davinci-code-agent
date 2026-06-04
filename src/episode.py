"""
Episode — collects every move of a SINGLE game and assigns the end-of-game
(retroactive) rewards in one place.

Design goal: keep all *reward attribution that depends on the game outcome* in
one cohesive, testable object instead of scattering it across the trainer's
rollout loop. The environment (``DaVinciCodeEnv``) remains the single source of
truth for per-step (immediate) rewards; ``Episode`` only adds the rewards that
can only be known once the whole game is over:

    - DRAW move      → +REWARD_DRAW_WIN if the mover won, else REWARD_DRAW_LOSE
    - CONTINUE move  → +REWARD_CONTINUE_SUCCESS if that player's NEXT guess was
                       correct, else REWARD_CONTINUE_FAIL   (computed at game end)
    - loser's last   → +REWARD_LOSE

Usage (per env, inside the rollout loop)::

    ep = Episode(env_id)
    ...
    ep.record(transition, phase, result)     # every step (transition is mutable)
    ...
    ep.finalize(winner)                       # once, when the game ends

``transition`` is whatever object the buffer stores; ``Episode`` only reads
``.player_id`` and mutates ``.reward`` (the buffer keeps the same object, so the
mutation is visible to GAE). This keeps the class decoupled from the concrete
``Transition`` type, so it can be reused by other trainers/back-ends later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from src.constants import Phase
from src.reward_config import RewardConfig
from src.result.guess_result import GuessResult
from src.result.streak_result import StreakResult


@dataclass
class GameMove:
    """One recorded move within an episode."""
    transition: Any           # buffer transition; we only touch .reward / .player_id
    phase: int                # Phase value at which the action was taken
    player_id: int
    is_continue: bool = False  # DECISION phase, chose CONTINUE
    is_guess: bool = False     # a valid GUESS move
    guess_correct: Optional[bool] = None  # for guess moves: was it right?


class Episode:
    """Accumulates the moves of one game and applies outcome-based rewards."""

    def __init__(self, env_id: int, reward_config: Optional[RewardConfig] = None) -> None:
        self.env_id = env_id
        self.moves: List[GameMove] = []
        # None → defaults identical to the old global constants (behaviour unchanged).
        self._rc = reward_config if reward_config is not None else RewardConfig()

    def record(self, transition: Any, phase: int, result: Any) -> None:
        """Record one executed move (called every step for this env)."""
        is_continue = (
            isinstance(result, StreakResult)
            and not result.is_invalid
            and result.is_continue
        )
        is_guess = isinstance(result, GuessResult) and not result.is_invalid
        guess_correct = result.is_correct if is_guess else None
        self.moves.append(GameMove(
            transition=transition,
            phase=int(phase),
            player_id=int(getattr(transition, "player_id", 0)),
            is_continue=is_continue,
            is_guess=is_guess,
            guess_correct=guess_correct,
        ))

    def finalize(self, winner: Optional[int]) -> None:
        """Apply all end-of-game (retroactive) rewards. No-op if no winner."""
        if winner is None:
            return
        loser = 1 - winner

        # 1) DRAW moves: small win/lose shaping spread over every card drawn.
        for m in self.moves:
            if m.phase == Phase.DRAW.value:
                m.transition.reward += (
                    self._rc.draw_win if m.player_id == winner else self._rc.draw_lose
                )

        # 2) CONTINUE moves: graded by that player's NEXT guess outcome.
        n = len(self.moves)
        for idx in range(n):
            m = self.moves[idx]
            if not m.is_continue:
                continue
            for nxt in self.moves[idx + 1:]:
                if nxt.is_guess and nxt.player_id == m.player_id:
                    m.transition.reward += (
                        self._rc.continue_success if nxt.guess_correct
                        else self._rc.continue_fail
                    )
                    break

        # 3) Loser's last move: the big terminal penalty.
        for m in reversed(self.moves):
            if m.player_id == loser:
                m.transition.reward += self._rc.lose
                break

    def __len__(self) -> int:
        return len(self.moves)
