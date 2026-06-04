"""Core game-engine invariants (deck, dealing, masking, termination)."""

from src.constants import Color
from src.deck import Deck
from src.runner import run_episode


def test_initial_deal_and_deck_count(env):
    obs, info = env.reset()
    # 4 cards each, 26 - 8 = 18 left in deck
    assert env.players[0]._hand.size == 4
    assert env.players[1]._hand.size == 4
    assert env._deck.total_count == 18


def test_deck_refills_on_reset():
    d = Deck()
    assert d.total_count == 26
    for _ in range(10):
        d.draw(Color.BLACK); d.draw(Color.WHITE)
    assert d.total_count == 6
    d.reset()
    assert d.total_count == 26          # reset rebuilds the deck


def test_following_mask_never_invalid(env, agent):
    """An agent that always picks from the mask should never trigger an invalid action."""
    total_steps = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            mask = env.get_action_mask()
            action, _ = agent.act(obs, mask, deterministic=False)
            obs, _r, _rew, term, trunc, info, result = env.step(action)
            assert not getattr(result, "is_invalid", False), "mask-following action was invalid"
            done = term or trunc
            steps += 1
        total_steps += steps
    assert total_steps > 0, "no steps were actually executed"


def test_games_terminate_with_winner(env, agent):
    for _ in range(10):
        res = run_episode(env, agent, deterministic=False)
        assert res.length < 500, "game did not terminate"
        # Da Vinci Code always has a winner (someone reveals all opponent cards / self-destructs)
        assert res.winner in (0, 1)


def test_action_mask_shapes(env):
    env.reset()
    m = env.get_action_mask()
    assert m["color"].shape == (2,)
    assert m["position"].shape == (13,)
    assert m["value"].shape == (13, 13)
    assert m["decision"].shape == (2,)
