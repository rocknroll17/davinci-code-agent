"""Monotone reward mode: the terminal win bonus must be dropped at env level."""
import numpy as np

from src.env import DaVinciCodeEnv
from src.runner import run_episode


def _max_step_reward(env, agent, n_games=15):
    hi = -1e9
    for _ in range(n_games):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            mask = env.get_action_mask()
            action, _ = agent.act(obs, mask, deterministic=False)
            obs, _r, reward, term, trunc, info, _ = env.step(action)
            hi = max(hi, float(reward))
            done = term or trunc
            steps += 1
    return hi


def test_monotone_drops_win_bonus(monkeypatch, make_agent):
    win = DaVinciCodeEnv()._rc.win  # 10.0 by default
    # monotone ON: no single step reward should reach the +win terminal bonus
    monkeypatch.setenv("DVC_MONOTONE_REWARD", "1")
    env_mono = DaVinciCodeEnv()
    assert env_mono._monotone_reward is True
    hi_mono = _max_step_reward(env_mono, make_agent(1))
    assert hi_mono < win, f"monotone should drop +{win} win bonus, saw {hi_mono}"


def test_normal_mode_includes_win_bonus(monkeypatch, make_agent):
    monkeypatch.delenv("DVC_MONOTONE_REWARD", raising=False)
    env = DaVinciCodeEnv()
    assert env._monotone_reward is False
    win = env._rc.win
    hi = _max_step_reward(env, make_agent(1), n_games=20)
    assert hi >= win, f"normal mode should include the +{win} win bonus somewhere, max={hi}"
