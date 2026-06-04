from ..cards.card import Card
from .result import Result


class DrawResult(Result):
    """Class representing a draw result in a game."""
    def __init__(self, player_id: int, reward: float, drawn_card: Card, position: int, is_invalid: bool = False) -> None:
        super().__init__(player_id, reward, is_invalid)
        self.drawn_card: Card | None = drawn_card if not is_invalid else None
        self.position: int | None = position if not is_invalid else None

    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        if self.is_invalid:
            return f"Player {self.player_id} made an invalid draw."
        return f"Player {self.player_id} drew {self.drawn_card} at position {self.position}"