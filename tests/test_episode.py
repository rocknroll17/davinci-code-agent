"""Unit tests for Episode.finalize — retroactive reward attribution."""
from dataclasses import dataclass

import numpy as np

from src.episode import Episode
from src.reward_config import RewardConfig
from src.constants import Phase
from src.result.guess_result import GuessResult
from src.result.streak_result import StreakResult


@dataclass
class FakeT:
    """Minimal transition stand-in: Episode only touches .reward / .player_id."""
    player_id: int
    reward: float = 0.0


def _draw_result(pid):
    # a DRAW move's result type doesn't matter to Episode (phase drives draw reward)
    return None


def test_draw_reward_winner_loser():
    rc = RewardConfig()
    ep = Episode(0, rc)
    # p0 draws twice, p1 draws once; then p1 makes a (losing) guess so the -10
    # loser penalty lands on the guess, isolating the draw rewards.
    t0a, t0b, t1draw, t1last = FakeT(0), FakeT(0), FakeT(1), FakeT(1)
    ep.record(t0a, Phase.DRAW.value, None)
    ep.record(t1draw, Phase.DRAW.value, None)
    ep.record(t0b, Phase.DRAW.value, None)
    ep.record(t1last, Phase.GUESS.value, GuessResult(1, 0.0, 0, 0, False, is_invalid=False))
    ep.finalize(winner=0)
    assert t0a.reward == rc.draw_win
    assert t0b.reward == rc.draw_win
    assert t1draw.reward == rc.draw_lose            # draw-loss only
    assert t1last.reward == rc.lose                 # loser's last move gets -10


def test_loser_gets_lose_penalty_once():
    rc = RewardConfig()
    ep = Episode(0, rc)
    # p1 is loser; only their LAST move gets -10
    t1a = FakeT(1); t1b = FakeT(1); t0 = FakeT(0)
    ep.record(t1a, Phase.GUESS.value, GuessResult(1, 0.0, 0, 0, False, is_invalid=False))
    ep.record(t0, Phase.GUESS.value, GuessResult(0, 0.0, 0, 0, True, is_invalid=False))
    ep.record(t1b, Phase.GUESS.value, GuessResult(1, 0.0, 0, 0, False, is_invalid=False))
    ep.finalize(winner=0)
    assert t1a.reward == 0.0
    assert t1b.reward == rc.lose      # only the last p1 move
    assert t0.reward == 0.0


def test_continue_success_and_fail():
    rc = RewardConfig()
    ep = Episode(0, rc)
    # p0 CONTINUE then a CORRECT guess -> continue_success
    c_ok = FakeT(0)
    ep.record(c_ok, Phase.DECISION.value, StreakResult(0, 0.0, True, is_invalid=False))
    ep.record(FakeT(0), Phase.GUESS.value, GuessResult(0, 0.0, 0, 0, True, is_invalid=False))
    # p0 CONTINUE then a WRONG guess -> continue_fail
    c_bad = FakeT(0)
    ep.record(c_bad, Phase.DECISION.value, StreakResult(0, 0.0, True, is_invalid=False))
    ep.record(FakeT(0), Phase.GUESS.value, GuessResult(0, 0.0, 0, 0, False, is_invalid=False))
    ep.finalize(winner=0)
    assert abs(c_ok.reward - rc.continue_success) < 1e-9
    assert abs(c_bad.reward - rc.continue_fail) < 1e-9


def test_no_winner_is_noop():
    rc = RewardConfig()
    ep = Episode(0, rc)
    t = FakeT(0)
    ep.record(t, Phase.DRAW.value, None)
    ep.finalize(winner=None)
    assert t.reward == 0.0


def test_custom_reward_config_used():
    rc = RewardConfig(draw_win=7.0, draw_lose=-3.0, lose=-99.0)
    ep = Episode(0, rc)
    tw, tl = FakeT(0), FakeT(1)
    ep.record(tw, Phase.DRAW.value, None)
    ep.record(tl, Phase.GUESS.value, GuessResult(1, 0.0, 0, 0, False, is_invalid=False))
    ep.finalize(winner=0)
    assert tw.reward == 7.0
    assert tl.reward == -99.0   # loser's last move
