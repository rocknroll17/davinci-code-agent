"""Shared pytest fixtures/helpers for the Da Vinci Code test suite.

Tests avoid trained checkpoints — they use a seeded RandomAgent that always
picks a *valid* action from the action masks, so they exercise the real game
engine + reward path without needing a model.
"""
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from src.env import DaVinciCodeEnv
from src.constants import Phase


class RandomAgent:
    """Picks a uniformly-random VALID action from the masks. Satisfies `Agent`."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def _pick(self, mask_row) -> int:
        valid = [i for i, ok in enumerate(np.asarray(mask_row).ravel()) if ok]
        return self.rng.choice(valid) if valid else 0

    def act(self, obs: Dict[str, np.ndarray],
            action_mask: Optional[Dict[str, np.ndarray]] = None,
            deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        m = action_mask
        color = self._pick(m["color"])
        position = self._pick(m["position"])
        # value mask is per-position (13,13): use the chosen position's row
        vmask = np.asarray(m["value"])
        vrow = vmask[position] if vmask.ndim == 2 else vmask
        value = self._pick(vrow)
        decision = self._pick(m["decision"])
        return np.array([color, position, value, decision], dtype=np.int64), {}


@pytest.fixture
def env():
    return DaVinciCodeEnv()


@pytest.fixture
def agent():
    return RandomAgent(seed=123)


@pytest.fixture
def make_agent():
    """Factory for seeded RandomAgents (avoids cross-module imports in tests)."""
    return lambda seed=0: RandomAgent(seed=seed)


@pytest.fixture(autouse=True)
def _seed_global_random():
    """Seed the global `random` (used by hand.py card placement) for stability.

    NOTE: this does NOT fully fix game determinism — the deck uses its own
    `random.Random(seed)` (src/deck.py) seeded from OS entropy when the env is
    built with seed=None, so deck deals still vary run to run. These tests
    therefore assert structural invariants, never specific card values/order.
    """
    random.seed(20240601)
    yield
