from ..cards.card import Card
from .result import Result


class JokerPlaceResult(Result):
    """Result of a JOKER phase action: the player chose where to insert a joker."""

    def __init__(self, player_id: int, reward: float, card: Card, position: int,
                 is_initial: bool, is_invalid: bool = False) -> None:
        super().__init__(player_id, reward, is_invalid)
        self.card = card
        self.position = position
        self.is_initial = is_initial  # True for initial-hand jokers, False for drawn

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_invalid:
            return f"Player {self.player_id} made an invalid joker placement."
        kind = "initial" if self.is_initial else "drawn"
        return f"Player {self.player_id} placed {kind} joker at position {self.position}"
