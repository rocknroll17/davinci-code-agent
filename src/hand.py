import random
import sys
from typing import List, Optional, Tuple
from typing_extensions import override

import numpy as np
from src.cards.card import Card, OpponentCard
from src.constants import MAX_HAND_SIZE, CardValue, Color

class Hand(list[Card]):
    """
    Hand is now a subclass of list[Card].
    - 초기 카드: 정렬 모드 (조커 랜덤)
    - 이후 카드: 가능한 위치 계산 후 랜덤 삽입
    """

    def __init__(self) -> None:
        """Initialize an empty hand."""
        super().__init__()
        self.last_drawn_card: Card | None = None

    @override
    def add_card(self, card: "Card", joker_position: Optional[int] = None) -> int:
        """
        Add a card to the hand following game rules.
        - Joker: can go anywhere (if joker_position provided, use it)
        - Number card: valid positions calculated, one randomly chosen
        Returns the index where the card was inserted.
        """
        self.last_drawn_card = card
        
        # 조커이고 위치가 지정된 경우
        if card.is_joker and joker_position is not None:
            valid_positions = self.get_valid_joker_positions()
            if joker_position in valid_positions:
                insert_pos = joker_position
            else:
                insert_pos = self._get_insert_position(card)
        else:
            insert_pos = self._get_insert_position(card)
        
        self.insert(insert_pos, card)
        return insert_pos
    
    @override
    def add_pending_card(self, card: "Card", position: int) -> int:
        """
        Add a card to the hand at a specific position.
        Used for initial card setup where position is predetermined.
        Returns the index where the card was inserted.
        """
        self.last_drawn_card = card
        self.insert(position, card)
        return position
    
    def get_valid_joker_positions(self) -> List[int]:
        """
        조커가 삽입될 수 있는 모든 유효한 위치 반환.
        조커는 어디든 갈 수 있으므로 0부터 len(self)까지 모든 위치 반환.
        """
        return list(range(len(self) + 1))

    def _get_insert_position(self, card: "Card") -> int:
        """Determine valid insertion position for a card."""
        if card.is_joker or not self:
            return random.randint(0, len(self))

        # Find left and right boundaries among non-joker cards
        left_idx = -1
        right_idx = len(self)
        for i, c in enumerate(self):
            if not c.is_joker and c < card:
                left_idx = i
        for i, c in enumerate(self):
            if not c.is_joker and card < c:
                right_idx = i
                break

        # Candidate positions between boundaries
        candidate_positions = list(range(left_idx + 1, right_idx + 1))
        return random.choice(candidate_positions)
            
    def add_initial_cards(self, cards: List[Card]) -> None:
        """
        초기 카드 세트를 핸드에 추가한다.
        일반 카드(조커 제외)는 정렬해서 먼저 넣고, 조커는 분리해서 랜덤 위치에 삽입한다.
        """
        normal_cards = [c for c in cards if not c.is_joker]
        jokers = [c for c in cards if c.is_joker]

        # 일반 카드 정렬 후 삽입
        normal_cards.sort(key=lambda c: (c.value, c.color))
        for card in normal_cards:
            self.add_card(card)

        # 조커는 랜덤 위치에 삽입
        for joker in jokers:
            self.add_card(joker)

        self.last_drawn_card = None

    def _find_sorted_position_exclude_joker(self, card: Card) -> int:
        """
        조커를 제외하고 정렬된 위치를 찾는다.
        """
        pos = 0
        for i, c in enumerate(self):
            if c.is_joker:
                continue
            if card.sort_key() < c.sort_key():
                return i
            pos = i + 1
        return pos

    def _find_valid_positions(self, new_card: Card) -> list[int]:
        """
        새 카드가 들어갈 수 있는 모든 유효한 위치 찾기.
        - 조커가 주변에 있으면 위치 제약 완화
        - 오름차순, 같은 값이면 검정이 왼쪽
        """
        if new_card.is_joker:
            # 조커는 0 ~ len(self) 모든 위치 가능
            return list(range(len(self) + 1))
        n = len(self)
        if n == 0:
            return [0]
        valid = []
        for pos in range(n + 1):
            if self._is_valid_position(new_card, pos):
                valid.append(pos)
        return valid

    def _is_valid_position(self, new_card: Card, pos: int) -> bool:
        """새 카드가 pos 위치에 들어갈 수 있는지 확인"""
        n = len(self)
        
        # 왼쪽 범위 결정 (왼쪽에서 가장 가까운 비조커 카드)
        left_bound = None  # (value, color) or None
        for i in range(pos - 1, -1, -1):
            if not self[i].is_joker:
                left_bound = (self[i].value, self[i].color)
                break
        
        # 오른쪽 범위 결정 (오른쪽에서 가장 가까운 비조커 카드)
        right_bound = None  # (value, color) or None
        for i in range(pos, n):
            if not self[i].is_joker:
                right_bound = (self[i].value, self[i].color)
                break
        
        new_val = new_card.value
        new_col = new_card.color.value  # Color enum to int
        
        # 왼쪽 조건: new_card > left_bound
        if left_bound:
            lv, lc = left_bound
            if new_val < lv:
                return False
            if new_val == lv and new_col <= lc:
                return False
        
        # 오른쪽 조건: new_card < right_bound
        if right_bound:
            rv, rc = right_bound
            if new_val > rv:
                return False
            if new_val == rv and new_col >= rc:
                return False
        
        return True

    def _can_be_left_of(self, left: Card, right: Card) -> bool:
        """left가 right의 왼쪽에 올 수 있는지 (조커 미포함 일반 비교)"""
        # 조커가 포함되면 별도 로직 (실제로는 _is_valid_position에서 처리)
        if left.is_joker or right.is_joker:
            return True
        # 숫자 비교
        if left.value < right.value:
            return True
        if left.value == right.value and left.color < right.color:
            return True
        return False
    
    def to_observation(self, hidden: bool = False) -> np.ndarray:
        """
        Convert hand to observation format.

        Args:
            hidden: If True, hide unrevealed cards

        Returns:
            numpy array of shape (MAX_HAND_SIZE, 2)
        """
        obs = np.full((MAX_HAND_SIZE, 2), [Color.NONE, CardValue.NONE], dtype=np.int8)
        for i, card in enumerate(self):
            card_obs = card.to_observation(hidden=hidden)
            obs[i] = card_obs

        return obs
    
    def to_string(self) -> str:
        """String representation of the hand."""
        return ", ".join(card.to_string() for card in self)
    
    def to_opponent_view(self) -> "Hand":
        """
        상대방이 보는 내 손패 표현.
        공개된 카드는 값과 색상, 숨겨진 카드는 "??"로 표시.
        """
        view = Hand()
        last_drawn_view_card = None
        for card in self:
            if card.is_revealed:
                view.append(card)
                if card is self.last_drawn_card:
                    last_drawn_view_card = card
            else:
                opp_card = OpponentCard(card)
                view.append(opp_card)
                if card is self.last_drawn_card:
                    last_drawn_view_card = opp_card
        view.last_drawn_card = last_drawn_view_card
        return view
    
    @property
    def size(self) -> int:
        """Number of cards in hand."""
        return len(self)
    
    def get_position(self, card: "Card") -> int:
        """
        Get the position of a specific card in hand.
        
        Args:
            card: Card to find
        Returns:
            Index of the card, or -1 if not found
        """
        try:
            return self.index(card)
        except ValueError:
            return -1
    
    def get_card(self, position: int) -> Card | None:
        """
        Get card at specific position.
        
        Args:
            position: Index of card
            
        Returns:
            Card at position or None if invalid
        """
        if 0 <= position < len(self):
            return self[position]
        return None
    
    def all_revealed(self) -> bool:
        """Check if all cards are revealed (player lost)."""
        return len(self) > 0 and all(card.is_revealed for card in self)
    
    def reveal_card(self, position: int) -> bool:
        """
        Reveal card at specific position.
        
        Args:
            position: Index of card to reveal
            
        Returns:
            True if card was revealed, False if invalid position
        """
        if position < 0 or position >= len(self):
            return False
        card = self[position]
        if card.is_revealed:
            return False
        card.is_revealed = True
        return True
    
    def end_turn(self) -> None:
        """Clear last drawn card at end of turn."""
        self.last_drawn_card = None
    
    def reveal_drawn_card(self) -> None:
        """
        Reveal the most recently added hidden card.
        
        Returns:
            True if a card was revealed, False if none hidden
        """
        # Reveal last drawn card if it's hidden
        if self.last_drawn_card is not None and not self.last_drawn_card.is_revealed:
            self.last_drawn_card.reveal()
            return self.index(self.last_drawn_card)

        # No cards in Deck. Reveal a random hidden card.
        return self.reveal_random_hidden()

        

    def reveal_random_hidden(self) -> int:
        """
        Reveal a random hidden card from the hand.

        Returns True if a card was revealed, False otherwise.
        """
        hidden_indices = [i for i, c in enumerate(self) if not getattr(c, "is_revealed", False)]
        if not hidden_indices:
            sys.exit("Error: No hidden cards to reveal.")
        idx = random.choice(hidden_indices)
        self[idx].is_revealed = True
        return idx
        
    # ============================================================
    # The code below is for finetune to find specific conditions of deck
    # ============================================================

    def is_joker_between(self) -> Tuple[List[int], Optional[int]]:
        """
        Check if there's a hidden joker that is definitively identifiable.
        
        A joker's position can be confirmed when:
        - Same numbered Black and White cards are both revealed
        - All cards between them are jokers
        - Returns indices of hidden jokers in that range
        
        Example:
        ... B5(revealed) ??(hidden) W5(revealed) ...
        -> The hidden card is definitely a joker
            
        Returns:
            Tuple of (List of indices of hidden jokers, value of surrounding cards)
            빈 리스트면 조커 확정 없음
        """
        n = len(self)
        if n < 3:
            return [], None
        
        # Find all revealed black number cards
        for i, left in enumerate(self):
            if not (left.is_revealed and left.is_number and left.color == Color.BLACK):
                continue
            
            # Look for matching revealed white card with same value
            for j in range(i + 2, n):
                right = self[j]
                
                # Check if right is the matching white card
                if not (right.is_revealed and right.is_number and 
                        right.color == Color.WHITE and right.value == left.value):
                    # If we hit a non-joker, stop searching further
                    if not self[j].is_joker:
                        break
                    continue
                
                # Check if all cards between are jokers
                middle_cards = self[i + 1:j]
                if all(card.is_joker for card in middle_cards):
                    # Return indices of hidden jokers and the surrounding value
                    hidden_jokers = [
                        i + 1 + idx
                        for idx, card in enumerate(middle_cards) 
                        if not card.is_revealed
                    ]
                    if hidden_jokers:
                        return hidden_jokers, int(left.value)
                break
        
        return [], None

