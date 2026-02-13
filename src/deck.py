"""Deck management for Da Vinci Code Game."""

import random
from typing import List, Optional

from src.cards.card import Card
from src.cards.black_card import BlackCard
from src.cards.white_card import WhiteCard
from src.constants import Color, CardValue


class Deck:
    """
    Manages the draw deck for Da Vinci Code game.
    
    The deck contains:
    - Black cards: 0-11 (12 cards) + 1 joker = 13 cards
    - White cards: 0-11 (12 cards) + 1 joker = 13 cards
    Total: 26 cards
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the deck.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._rng = random.Random(seed)
        self._black_cards: list[BlackCard] = []
        self._white_cards: list[WhiteCard] = []
        self._initialize_deck()
    
    def _initialize_deck(self) -> None:
        """Create and shuffle the initial deck."""
        # Create black cards (0-11 + joker)
        self._black_cards = [BlackCard(v) for v in range(CardValue.JOKER + 1)]
        # Create white cards (0-11 + joker)
        self._white_cards = [WhiteCard(v) for v in range(CardValue.JOKER + 1)]
        
        # Shuffle each pile separately
        self._rng.shuffle(self._black_cards)
        self._rng.shuffle(self._white_cards)
    
    def draw(self, color: Color) -> Optional[Card]:
        """
        Draw a card of specified color from the deck.
        
        Args:
            color: Color of card to draw (Color.BLACK or Color.WHITE)
            
        Returns:
            Drawn card, or None if no cards of that color remain
        """
        if color == Color.BLACK:
            if self._black_cards:
                return self._black_cards.pop()
            return None
        elif color == Color.WHITE:
            if self._white_cards:
                return self._white_cards.pop()
            return None
        return None
    
    def initial_draw(self, count: int = 1) -> Optional[List[Card]]:
        """
        Draw initial cards from both colors.
        각 카드를 50% 확률로 검정/흰색 중 선택 (독립 시행)

        Args:
            count: Number of cards to draw
        Returns:
            List of drawn cards, or None if insufficient cards
        """
        drawn_cards = []
        for _ in range(count):
            # 각 카드마다 50% 확률로 색상 결정
            if self._rng.random() < 0.5:
                color = Color.BLACK
            else:
                color = Color.WHITE
            
            # 해당 색이 없으면 다른 색에서 뽑기
            card = self.draw(color)
            if card is None:
                other_color = Color.WHITE if color == Color.BLACK else Color.BLACK
                card = self.draw(other_color)
            
            if card is None:
                return None  # 덱이 완전히 비었음
            drawn_cards.append(card)
        
        return drawn_cards
    
    # def draw_random(self) -> Optional[Card]:
    #     """
    #     Draw a random card from the deck.
        
    #     Returns:
    #         Randomly drawn card, or None if deck is empty
    #     """
    #     available = []
    #     if self._black_cards:
    #         available.append(Color.BLACK)
    #     if self._white_cards:
    #         available.append(Color.WHITE)
        
    #     if not available:
    #         return None
        
    #     chosen_color = self._rng.choice(available)
    #     return self.draw(chosen_color)
    
    @property
    def black_count(self) -> int:
        """Number of black cards remaining."""
        return len(self._black_cards)
    
    @property
    def white_count(self) -> int:
        """Number of white cards remaining."""
        return len(self._white_cards)
    
    @property
    def total_count(self) -> int:
        """Total number of cards remaining."""
        return self.black_count + self.white_count
    
    def is_empty(self) -> bool:
        """Check if deck is empty."""
        return self.total_count == 0
    
    def is_one_empty(self) -> bool:
        """Check if one color is empty."""
        return (self.black_count == 0) != (self.white_count == 0)
    
    def has_color(self, color: Color) -> bool:
        """Check if deck has cards of specified color."""
        if color == Color.BLACK:
            return self.black_count > 0
        elif color == Color.WHITE:
            return self.white_count > 0
        return False
    
    def get_remaining(self) -> list[int]:
        """
        Get remaining deck counts for observation.
        
        Returns:
            [black_count, white_count]
        """
        return [self.black_count, self.white_count]
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the deck to initial state.
        
        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self._rng = random.Random(seed)
        self._initialize_deck()