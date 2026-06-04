"""Retroactive rewards in the trainer buffer (draw/continue/lose) via Episode."""
from collections import defaultdict

import numpy as np
import pytest
import torch

from src.trainer import PPOTrainer, PPOConfig
from src.constants import Phase

EPS = 1e-4


@pytest.fixture(scope="module")
def buffer_transitions():
    cfg = PPOConfig(n_envs=6, episodes_per_update=6, n_workers=2)
    tr = PPOTrainer(cfg, torch.device("cpu"))
    tr.collect_rollouts()
    return tr.buffer.transitions, tr.config.reward_config


def _episodes(transitions):
    per_env = defaultdict(list)
    for t in transitions:
        per_env[t.env_id].append(t)
    eps = []
    for ts in per_env.values():
        cur = []
        for t in ts:
            cur.append(t)
            if t.done:
                eps.append(cur); cur = []
    return eps


def test_draw_rewards_are_plus_minus(buffer_transitions):
    transitions, rc = buffer_transitions
    for ep in _episodes(transitions):
        for t in ep:
            if np.argmax(t.obs["phase"]) == Phase.DRAW.value:
                assert abs(t.reward - rc.draw_win) < EPS or abs(t.reward - rc.draw_lose) < EPS


def test_continue_rewards_in_expected_set(buffer_transitions):
    transitions, rc = buffer_transitions
    allowed = (0.0, rc.continue_success, rc.continue_fail)
    for ep in _episodes(transitions):
        for t in ep:
            if np.argmax(t.obs["phase"]) == Phase.DECISION.value and int(t.action[3]) == 1:
                assert any(abs(t.reward - a) < EPS for a in allowed), f"continue reward {t.reward}"


def test_exactly_one_loser_penalty_per_episode(buffer_transitions):
    transitions, rc = buffer_transitions
    eps = _episodes(transitions)
    assert len(eps) >= 1
    for ep in eps:
        big_neg = [t for t in ep if t.reward <= rc.lose + 1.0]
        assert len(big_neg) == 1, f"expected 1 loser(-10) move, got {len(big_neg)}"
