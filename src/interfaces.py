"""
Typed interfaces (structural Protocols) for pluggable agents and policies.

These let any object that *behaves* like an agent — a trained model wrapper, a
rule-based bot, a human-input shim — be used interchangeably by the evaluation
and head-to-head machinery, without forcing a shared base class. Conformance is
structural (duck typing checked by the type checker), so existing classes such
as ``src.agent.ModelAgent`` already satisfy ``Agent`` with no changes.

Usage::

    from src.interfaces import Agent

    def run_episode(env, agents: Agent | list[Agent], ...): ...
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Anything that can pick a single action for a single observation.

    Matches ``src.agent.ModelAgent.act`` exactly, so ModelAgent is an ``Agent``
    structurally (no inheritance needed).
    """

    def act(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Return ``(action[4], per-head-prob dict)`` for one (unbatched) obs."""
        ...


@runtime_checkable
class BatchAgent(Protocol):
    """An agent that can also act on a batch of observations at once."""

    def act_batch(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return ``actions[N, 4]`` for a batched obs dict."""
        ...


@runtime_checkable
class Policy(Protocol):
    """The lower-level network contract used during training/inference.

    Matches ``src.model.DaVinciCodePolicy.get_action``.
    """

    def get_action(
        self,
        obs: Dict[str, "np.ndarray"],
        action_mask: Optional[Dict[str, "np.ndarray"]] = None,
        deterministic: bool = False,
    ): ...
