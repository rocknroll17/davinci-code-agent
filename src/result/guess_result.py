from .result import Result


class GuessResult(Result):
    """Class representing a guess result in a game."""
    def __init__(self, player_id: int, reward: float, position: int, guessed_value: str, is_correct: bool, is_invalid: bool = False) -> None:
        super().__init__(player_id, reward, is_invalid)
        self.position = position
        self.guessed_value = guessed_value
        self.is_correct = is_correct

    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        if self.is_invalid:
            return f"Player {self.player_id} made an invalid guess."
        return f"Player {self.player_id} guessed '{self.guessed_value}' at position {self.position} which is {'O' if self.is_correct else 'X'}"
        