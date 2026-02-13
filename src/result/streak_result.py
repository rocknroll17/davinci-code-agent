from .result import Result

class StreakResult(Result):
    """Class representing a streak result in a game."""
    def __init__(self, player_id: int, reward: float, is_continue: bool, is_invalid: bool = False) -> None:
        super().__init__(player_id, reward, is_invalid)
        if not is_invalid:
            self.is_continue = is_continue

    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        if self.is_invalid:
            return f"Player {self.player_id} made an invalid streak decision."
        return f"Player {self.player_id} {'Continues' if self.is_continue else 'Stops'}."