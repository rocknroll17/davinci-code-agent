import numpy as np
from src.constants import MAX_HAND_SIZE, NUM_VALUES, Color
from src.hand import Hand


# Constraint matrix cell values
_SLOT_EMPTY    = -1  # slot does not exist (opponent has no card here)
_VALUE_UNKNOWN =  0  # slot exists, value not yet ruled out or confirmed
_VALUE_RULED_OUT = 1  # slot exists, this value has been tried and failed (or card is revealed)


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self._hand: Hand = Hand()
        # shape (MAX_HAND_SIZE, NUM_VALUES): _SLOT_EMPTY → _VALUE_UNKNOWN → _VALUE_RULED_OUT
        self._constraint_matrix: np.ndarray = np.full(
            (MAX_HAND_SIZE, NUM_VALUES), _SLOT_EMPTY, dtype=np.int8
        )

    def __repr__(self) -> str:
        return f"Player({self.player_id})"
    
    def __str__(self) -> str:
        return f"Player {self.player_id}"
    
    def reset(self) -> None:
        self._hand.clear()
        self._constraint_matrix.fill(_SLOT_EMPTY)
        

    def _update_constraint_revealed(self, position: int) -> None:
        """
        Update constraint matrix when card is revealed.
        
        Args:
            position: Position of revealed card
            value: Value of revealed card
        """
        # Mark entire row as ruled-out: card is revealed, its exact value is now known
        self._constraint_matrix[position, :] = _VALUE_RULED_OUT
    
    def _update_constraint_failed(self, position: int, value: int, color: Color) -> None:
        """
        Update constraint matrix when guess fails.
        
        Args:
            position: Position that was guessed
            value: Value that was wrong
            color: Color of the card at that position (known, visible)
        """
        # Column index = value (slot color is already visible via opp_hand)
        col = value
        self._constraint_matrix[position, col] = _VALUE_RULED_OUT

    def get_own_observation(self) -> dict[str, np.ndarray]:
        """Get observation from this player's own perspective (full hand + constraint matrix)."""
        return {
            "my_hand": self._hand.to_observation(hidden=False),
            "constraint_matrix": self._constraint_matrix.copy()
        }

    def get_opponent_observation(self) -> dict[str, np.ndarray]:
        """Get observation of this player as seen by the opponent (hidden hand)."""
        return {
            "opponent_hand": self._hand.to_observation(hidden=True)
        }
    
    
    def end_turn(self) -> None:
        """Actions to perform at the end of the player's turn."""
        # Currently no specific actions needed
        self._hand.end_turn()

    def guess_success(self, position: int) -> None:
        self._update_constraint_revealed(position)

    def guess_fail(self, position: int, guessed_value: int, color: Color) -> None:
        """
        Handle actions when a guess fails.
        
        Args:
            position: Position that was guessed
            guessed_value: Value that was incorrectly guessed
            color: Color of the card at that position (visible to guesser)
        """
        # Update constraint matrix - mark this (color, value) as wrong for this position
        self._update_constraint_failed(position, guessed_value, color)
        
        # Reveal own card (the one just drawn)
        return self._hand.reveal_drawn_card()

    def update_constraint(self, position: int) -> None:
        """
        Update constraint matrix when opponent draws a card.
        
        Args:
            position: Position of the drawn card
        """
        # Insert a fresh _VALUE_UNKNOWN row at `position`; drop the last row to keep shape constant
        nrows, ncols = self._constraint_matrix.shape
        position = max(0, min(position, nrows))
        self._constraint_matrix = np.insert(
            self._constraint_matrix,
            position,
            values=np.zeros(ncols, dtype=self._constraint_matrix.dtype),
            axis=0
        )
        if self._constraint_matrix.shape[0] > nrows:
            self._constraint_matrix = self._constraint_matrix[:nrows, :]

    def update_initial_constraint(self, initial_hand_size: int) -> None:
        """
        Update constraint matrix at the start of the game.
        
        Marks all positions as unknown (0) for the initial hand size.
        """
        self._constraint_matrix[:initial_hand_size, :] = _VALUE_UNKNOWN