"""Card class for Da Vinci Code Game."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.constants import Color, CardValue


@dataclass
class Card(ABC):
    """
    Represents a single card in Da Vinci Code game.
    
    Attributes:
        color: Card color (BLACK=0, WHITE=1)
        value: Card value (0-11 for numbers, 12 for joker)
        is_revealed: Whether the card value is revealed to opponent
    """
    color: Color
    value: CardValue = field(compare=False)
    is_revealed: bool = False
    
    def __post_init__(self) -> None:
        """Validate card properties after initialization."""
        if isinstance(self.value, int):
            self.value = CardValue(self.value)
        if self.color not in (Color.BLACK, Color.WHITE):
            raise ValueError(f"Invalid color: {self.color}")
        if not (0 <= self.value <= CardValue.JOKER):
            raise ValueError(f"Invalid value: {self.value}")
    
    def __hash__(self) -> int:
        """Hash based on color and value."""
        return hash((self.color, self.value))
    
    def __eq__(self, other: "Card") -> bool:
        """Equality based on color and value."""
        return (self.color, self.value) == (other.color, other.value)
    
    def __lt__(self, other: "Card") -> bool:
        """Less-than comparison for sorting cards."""
        if self.is_joker or other.is_joker:
            return True

        return (self.value, self.color) < (other.value, other.color)

    @property
    def is_joker(self) -> bool:
        """Check if this card is a joker."""
        return self.value == CardValue.JOKER
    
    @property
    def is_number(self) -> bool:
        """Check if this card is a numbered card (not a joker)."""
        return self.value != CardValue.JOKER
    
    def sort_key(self) -> tuple[int, int]:
        """
        Return sort key for hand ordering.
        
        Cards are sorted by value first, then by color (black before white).
        
        Returns:
            Tuple of (value, color) for sorting
        """
        return (self.value, self.color)
    
    def to_observation(self, hidden: bool = False) -> list[int]:
        """
        Convert card to observation format.
        
        Args:
            hidden: If True, return hidden representation
            
        Returns:
            [color, value] or [color, HIDDEN] if hidden
        """
        if hidden and not self.is_revealed:
            return [self.color, CardValue.HIDDEN]
        return [self.color, self.value]
    
    def reveal(self) -> None:
        """Reveal the card's value to the opponent."""
        self.is_revealed = True
    
    def __repr__(self) -> str:
        """String representation of the card."""
        color_str = "B" if self.color == Color.BLACK else "W"
        value_str = "J" if self.is_joker else str(self.value)
        revealed_str = "*" if self.is_revealed else ""
        return f"{color_str}{value_str}{revealed_str}"

    def to_string(self) -> str:
        """String representation of the card."""
        return f"{self.color.to_string()} {self.value.to_string()}"
    
    def to_ui_string(self) -> str:
        """String representation for UI display."""
        color = "검정" if self.color == Color.BLACK else "하양"
        if self.is_joker:
            return f"{color} Joker"
        return f"{color} {self.value.to_string()}"
    
class OpponentCard(Card):
    """
    Represents a card from opponent's perspective.
    Hides value if not revealed.
    """
    def __init__(self, card: Card) -> None:
        super().__init__(color=card.color, value=card.value if card.is_revealed else CardValue.HIDDEN, is_revealed=card.is_revealed)

    def __post_init__(self) -> None:
        """Validate card properties after initialization."""
        if isinstance(self.value, int):
            self.value = CardValue(self.value)
        if self.color not in (Color.BLACK, Color.WHITE):
            raise ValueError(f"Invalid color: {self.color}")
        if not (CardValue.HIDDEN <= self.value <= CardValue.JOKER):
            raise ValueError(f"Invalid value: {self.value}")