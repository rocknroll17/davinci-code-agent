"""run_episode contract: single-agent and head-to-head."""
import numpy as np

from src.runner import run_episode, EpisodeResult


def test_single_agent_episode(env, agent):
    res = run_episode(env, agent, deterministic=False)
    assert isinstance(res, EpisodeResult)
    assert res.winner in (0, 1, None)
    assert res.length > 0
    assert len(res.rewards) == 2


def test_two_agent_head_to_head(env, make_agent):
    a, b = make_agent(1), make_agent(2)
    res = run_episode(env, [a, b], deterministic=False)
    assert res.winner in (0, 1, None)
    assert res.length > 0


def test_on_step_called_once_per_move(env, agent):
    calls = []
    res = run_episode(env, agent, on_step=lambda ctx: calls.append(ctx))
    assert len(calls) == res.length
    # context sanity
    c0 = calls[0]
    assert c0.player in (0, 1)
    assert "phase" in c0.obs_before
    assert c0.action.shape == (4,)
