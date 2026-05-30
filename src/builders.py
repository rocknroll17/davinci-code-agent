"""
GameStateBuilder — controllable environment factory for Da Vinci Code.

Lets you construct an environment pre-loaded with specific hands, deck state,
constraint matrix, and phase, without going through the random env.reset() flow.

Usage
-----
from src.builders import GameStateBuilder

env = (
    GameStateBuilder()
    .set_my_hand(["B0", "B5", "W3", "J"])
    .set_opponent_hand(["B1", "W2", "B7"])
    .set_phase("GUESS")
    .set_current_player(0)
    .build()
)

obs, info = env.reset()   # Returns the injected state as first obs
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Union

from src.env import DaVinciCodeEnv
from src.constants import Phase, Color, CardValue, MAX_HAND_SIZE, NUM_VALUES
from src.cards.card import Card
from src.phase import PhaseCycle


# ---------------------------------------------------------------------------
# Card spec parser helpers
# ---------------------------------------------------------------------------

def _parse_card_spec(spec: str) -> Card:
    """
    Parse a human-readable card spec string into a Card instance.

    Format: ``"<Color><Value>"``
    - Color: ``B`` = BLACK, ``W`` = WHITE
    - Value: ``0``–``11`` for normal cards, ``J`` or ``-`` for Joker
    - Examples: ``"B0"`` ``"W11"`` ``"J"`` ``"-"``

    Cards are created as *revealed* (fully known) by default.
    """
    spec = spec.strip().upper()
    if not spec:
        raise ValueError("Empty card spec")

    # Joker
    if spec in ("J", "-", "BJ", "WJ"):
        color = Color.BLACK if spec.startswith("B") else Color.WHITE
        from src.cards.black_card import BlackCard
        from src.cards.white_card import WhiteCard
        cls = BlackCard if color == Color.BLACK else WhiteCard
        card = cls(CardValue.JOKER)
        card._revealed = True
        return card

    color_char = spec[0]
    if color_char == "B":
        color = Color.BLACK
    elif color_char == "W":
        color = Color.WHITE
    else:
        raise ValueError(f"Unknown color char '{color_char}' in spec '{spec}'")

    value_str = spec[1:]
    try:
        value = int(value_str)
    except ValueError:
        raise ValueError(f"Cannot parse value '{value_str}' in spec '{spec}'")

    if not (0 <= value <= 11):
        raise ValueError(f"Card value must be 0-11, got {value} in spec '{spec}'")

    from src.cards.black_card import BlackCard
    from src.cards.white_card import WhiteCard
    cls = BlackCard if color == Color.BLACK else WhiteCard
    card = cls(value)
    card._revealed = True
    return card


# ---------------------------------------------------------------------------
# GameStateBuilder
# ---------------------------------------------------------------------------

class GameStateBuilder:
    """
    Fluent builder for constructing a DaVinciCodeEnv in a specific state.

    All methods return ``self`` for chaining.
    ``build()`` returns a configured env whose first ``reset()`` call
    will restore the injected state.
    """

    def __init__(self) -> None:
        self._my_hand: Optional[List[str]] = None           # card specs for player 0
        self._opponent_hand: Optional[List[str]] = None     # card specs for player 1
        self._constraint_matrix: Optional[np.ndarray] = None
        self._phase: Phase = Phase.DRAW
        self._current_player: int = 0
        self._seed: Optional[int] = None
        self._remaining_deck: Optional[tuple[int, int]] = None  # (black, white)

    # ---------- fluent setters ----------

    def set_my_hand(self, specs: List[str]) -> "GameStateBuilder":
        """Set player 0's hand using card spec strings (e.g. ``['B0', 'W3', 'J']``)."""
        self._my_hand = list(specs)
        return self

    def set_opponent_hand(self, specs: List[str]) -> "GameStateBuilder":
        """Set player 1's hand using card spec strings."""
        self._opponent_hand = list(specs)
        return self

    def set_constraint_matrix(self, matrix: np.ndarray) -> "GameStateBuilder":
        """Inject a pre-built constraint matrix (shape MAX_HAND_SIZE × NUM_VALUES)."""
        if matrix.shape != (MAX_HAND_SIZE, NUM_VALUES):
            raise ValueError(
                f"constraint_matrix must be ({MAX_HAND_SIZE}, {NUM_VALUES}), got {matrix.shape}"
            )
        self._constraint_matrix = matrix.copy()
        return self

    def set_phase(self, phase: Union[str, Phase]) -> "GameStateBuilder":
        """Set the starting phase: ``'DRAW'``, ``'GUESS'``, ``'DECISION'``, or a ``Phase`` enum."""
        if isinstance(phase, str):
            phase = Phase[phase.upper()]
        self._phase = phase
        return self

    def set_current_player(self, player_id: int) -> "GameStateBuilder":
        """Set which player acts first (0 or 1)."""
        if player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        self._current_player = player_id
        return self

    def set_seed(self, seed: int) -> "GameStateBuilder":
        """Seed for any remaining randomness (deck, unused slots)."""
        self._seed = seed
        return self

    def set_remaining_deck(self, black: int, white: int) -> "GameStateBuilder":
        """Override the remaining deck counts (cosmetic — affects obs only)."""
        self._remaining_deck = (black, white)
        return self

    # ---------- build ----------

    def build(self) -> "_InjectedEnv":
        """Construct and return the configured environment."""
        return _InjectedEnv(
            my_hand_specs=self._my_hand,
            opponent_hand_specs=self._opponent_hand,
            constraint_matrix=self._constraint_matrix,
            phase=self._phase,
            current_player=self._current_player,
            seed=self._seed,
            remaining_deck=self._remaining_deck,
        )


# ---------------------------------------------------------------------------
# _InjectedEnv — DaVinciCodeEnv subclass that overrides reset()
# ---------------------------------------------------------------------------

class _InjectedEnv(DaVinciCodeEnv):
    """
    DaVinciCodeEnv that restores a specific game state on reset().

    Fields not set by the builder are filled with sensible defaults
    (empty hand / fresh deck).
    """

    def __init__(
        self,
        my_hand_specs: Optional[List[str]],
        opponent_hand_specs: Optional[List[str]],
        constraint_matrix: Optional[np.ndarray],
        phase: Phase,
        current_player: int,
        seed: Optional[int],
        remaining_deck: Optional[tuple[int, int]],
    ) -> None:
        super().__init__(seed=seed)
        self._injected_my_hand = my_hand_specs
        self._injected_opp_hand = opponent_hand_specs
        self._injected_constraint = constraint_matrix
        self._injected_phase = phase
        self._injected_current_player = current_player
        self._injected_remaining_deck = remaining_deck

    def reset(self, *, seed=None, options=None):
        # Run base reset to initialise all internal structures with random state,
        # then overwrite with whatever the builder specified.
        obs, info = super().reset(seed=seed, options=options)

        # --- Override hand for player 0 (Hand is a list subclass) ---
        if self._injected_my_hand is not None:
            hand0 = self.players[0]._hand
            hand0.clear()
            for spec in self._injected_my_hand:
                hand0.append(_parse_card_spec(spec))
            self.players[0].update_initial_constraint(len(self._injected_my_hand))

        # --- Override hand for player 1 ---
        if self._injected_opp_hand is not None:
            hand1 = self.players[1]._hand
            hand1.clear()
            for spec in self._injected_opp_hand:
                hand1.append(_parse_card_spec(spec))
            self.players[1].update_initial_constraint(len(self._injected_opp_hand))

        # --- Override constraint matrix ---
        if self._injected_constraint is not None:
            self.players[0]._constraint_matrix = self._injected_constraint.copy()

        # --- Override phase ---
        self._phase = PhaseCycle(start=self._injected_phase)

        # --- Override current player ---
        self._current_player = self._injected_current_player

        return self._get_observation(), self._get_info()
