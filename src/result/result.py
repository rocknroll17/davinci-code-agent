from abc import ABC, abstractmethod


class Result(ABC):
    """Abstract base class for representing the result of a game or operation."""
    def __init__(self, player_id: int, reward: float, is_invalid: bool = False) -> None:
        self._player_id = player_id
        self._reward = reward
        self.is_invalid = is_invalid

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def invalid(self) -> bool:
        """Check if the result is invalid."""
        return self.is_invalid
    
    @property
    def reward(self) -> float:
        """Get the reward associated with the result."""
        return self._reward
    
    @property
    def player_id(self) -> int:
        """Get the player ID associated with the result."""
        return self._player_id
