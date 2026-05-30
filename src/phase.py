from src.constants import Phase


class InvalidPhaseTransitionError(RuntimeError):
    """Raised when a phase transition is called in the wrong state."""
    pass


class PhaseCycle:
    """
    Controls the lifecycle of a Da Vinci Code game phase.

    Phase transitions are explicitly modeled as methods,
    ensuring only valid actions can be performed in each phase.
    """

    def __init__(self, start: Phase = Phase.DRAW) -> None:
        self._phase: Phase = start

    @property
    def phase(self) -> Phase:
        return self._phase
    
    @property
    def name(self) -> Phase:
        return self._phase.name

    @property
    def value(self) -> int:
        return self._phase.value

    # ===== DRAW =====

    def draw_done(self) -> None:
        self._require(Phase.DRAW)
        self._phase = Phase.GUESS

    # ===== GUESS =====

    def guess_correct(self) -> None:
        self._require(Phase.GUESS)
        self._phase = Phase.DECISION

    def guess_wrong(self) -> None:
        self._require(Phase.GUESS)
        self._phase = Phase.DRAW

    # ===== DECISION =====

    def continue_streak(self) -> None:
        self._require(Phase.DECISION)
        self._phase = Phase.GUESS

    def end_streak(self) -> None:
        self._require(Phase.DECISION)
        self._phase = Phase.DRAW  # END phase 생기면 여기만 수정

    # ===== internal =====

    def _require(self, expected: Phase) -> None:
        if self._phase is not expected:
            raise InvalidPhaseTransitionError(
                f"Invalid transition: expected {expected.name}, current {self._phase.name}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase):
            return NotImplemented
        return self._phase == other