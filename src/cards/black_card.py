from src.constants import Color

from .card import Card


class BlackCard(Card):
    """
    Represents a black card in Da Vinci Code game.
    Inherits from Card and sets color to BLACK by default.
    """
    def __init__(self, value: int, is_revealed: bool = False) -> None:
        super().__init__(color=Color.BLACK, value=value, is_revealed=is_revealed)