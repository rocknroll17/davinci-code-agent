from src.constants import Color
from .card import Card

class WhiteCard(Card):
    """
    Represents a white card in Da Vinci Code game.
    Inherits from Card and sets color to WHITE by default.
    """
    def __init__(self, value: int, is_revealed: bool = False) -> None:
        super().__init__(color=Color.WHITE, value=value, is_revealed=is_revealed)