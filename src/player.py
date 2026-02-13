import sys
import numpy as np
from src.constants import MAX_HAND_SIZE, NUM_VALUES
from src.hand import Hand


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self._hand: Hand = Hand()
        self._constraint_matrix: np.ndarray = np.zeros((MAX_HAND_SIZE, NUM_VALUES), dtype=np.int8)

    def __repr__(self) -> str:
        return f"Player({self.player_id})"
    
    def __str__(self) -> str:
        return f"Player {self.player_id}"
    
    def reset(self) -> None:
        self._hand.clear()
        self._constraint_matrix.fill(-1)
        

    def _update_constraint_revealed(self, position: int) -> None:
        """
        Update constraint matrix when card is revealed.
        
        Args:
            position: Position of revealed card
            value: Value of revealed card
        """
        # Mark the entire row as known (set to 1)
        self._constraint_matrix[position, :] = 1
    
    def _update_constraint_failed(self, position: int, value: int) -> None:
        """
        Update constraint matrix when guess fails.
        
        Args:
            position: Position that was guessed
            value: Value that was wrong
        """
        self._constraint_matrix[position, value] = 1

    def _get_observation(self, is_mine: bool) -> dict[str, np.ndarray]:
        """
        Get current observation from this player's perspective.
        
        Returns:
            Observation dictionary
        """
        if is_mine:
            return {
                "my_hand": self._hand.to_observation(hidden=False),
                "constraint_matrix": self._constraint_matrix.copy()
            }

        else:
            return {
                "opponent_hand": self._hand.to_observation(hidden=True)
            }
    
    
    def end_turn(self) -> None:
        """Actions to perform at the end of the player's turn."""
        # Currently no specific actions needed
        self._hand.end_turn()

    def guess_success(self, position: int) -> None:
        self._update_constraint_revealed(position)

    def guess_fail(self, position: int, guessed_value: int) -> None:
        """
        Handle actions when a guess fails.
        
        Args:
            position: Position that was guessed
            guessed_value: Value that was incorrectly guessed
        """
        # Update constraint matrix - mark this value as wrong for this position
        self._update_constraint_failed(position, guessed_value)
        
        # Reveal own card (the one just drawn)
        return self._hand.reveal_drawn_card()

    def update_constraint(self, position: int) -> None:
        """
        Update constraint matrix when opponent draws a card.
        
        Args:
            position: Position of the drawn card
        """
        # Insert an unknown row at `position`, shifting subsequent rows to the
        # right. Keep the matrix length constant by dropping the last row.
        nrows, ncols = self._constraint_matrix.shape
        # Clamp position
        if position < 0:
            position = 0
        if position > nrows:
            position = nrows

        # Use numpy.insert to insert a zero row, then truncate to original size
        self._constraint_matrix = np.insert(
            self._constraint_matrix,
            position,
            values=np.zeros(ncols, dtype=self._constraint_matrix.dtype),
            axis=0
        )

        # Truncate if grew by one
        if self._constraint_matrix.shape[0] > nrows:
            self._constraint_matrix = self._constraint_matrix[:nrows, :]

    def update_initial_constraint(self, initial_hand_size: int) -> None:
        """
        Update constraint matrix at the start of the game.
        
        Marks all positions as unknown (0) for the initial hand size.
        """
        self._constraint_matrix[:initial_hand_size, :] = 0