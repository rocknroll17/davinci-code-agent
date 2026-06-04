"""
ModelAgent — clean inference wrapper around DaVinciCodePolicy.

Removes the obs→tensor/mask→tensor boilerplate from every caller
and exposes a single ``act()`` / ``act_batch()`` interface.

Usage
-----
from src.agent import ModelAgent

agent = ModelAgent.from_checkpoint("checkpoints/best_model.pt")

# Single step (unbatched obs dict from env.step)
action, probs = agent.act(obs, action_mask)

# Batched (from VecEnv)
actions = agent.act_batch(obs_batch, masks_batch, deterministic=True)

# Logits for analysis
logits = agent.logits(obs, action_mask)
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from src.model import DaVinciCodePolicy, obs_to_tensor, action_mask_to_tensor
from src.constants import ACTION_KEYS, CHECKPOINT_BEST


class ModelAgent:
    """
    Thin wrapper around ``DaVinciCodePolicy`` for convenient inference.

    Parameters
    ----------
    policy : DaVinciCodePolicy
        Already-loaded policy network.
    device : torch.device
        Target device.
    """

    def __init__(self, policy: DaVinciCodePolicy, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.policy.eval()

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str = CHECKPOINT_BEST,
        device: Optional[torch.device] = None,
        hidden_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
    ) -> "ModelAgent":
        """
        Load a ``ModelAgent`` directly from a ``.pt`` checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint (defaults to ``checkpoints/best_model.pt``).
        device : torch.device | None
            Target device (auto-detect if ``None``).
        hidden_dim : int
            Must match the saved model's hidden dimension.
        n_heads, n_layers : int
            Encoder transformer shape — must match the saved model. If the
            checkpoint stored its config, these are auto-read from it.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device)
        # Prefer the architecture recorded in the checkpoint's config (added by the
        # experiment runner) so heads/layers variants load with the right shape.
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        n_heads = cfg.get("n_heads", n_heads)
        n_layers = cfg.get("n_layers", n_layers)
        policy = DaVinciCodePolicy(hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers).to(device)

        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        # strict=False tolerates architecture additions (e.g. slot_pos_embed) when
        # loading older checkpoints; missing params keep their fresh init.
        policy.load_state_dict(state_dict, strict=False)

        agent = cls(policy, device)
        agent._checkpoint_path = path
        agent._timesteps = checkpoint.get("timesteps", None)
        return agent

    @classmethod
    def from_trainer(cls, trainer) -> "ModelAgent":
        """
        Borrow the policy from a live ``PPOTrainer`` (no copy — shares weights).
        """
        return cls(trainer.policy, trainer.device)

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def act(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Select an action for a **single** obs dict (no batch dimension).

        Parameters
        ----------
        obs : dict
            Observation dict from ``env.step()`` or ``env.reset()``.
        action_mask : dict | None
            Action masks from ``env.get_action_mask()``.
        deterministic : bool
            If ``True``, take the argmax action; otherwise sample.

        Returns
        -------
        action : np.ndarray shape (4,)
            ``[color, position, value, decision]``
        probs : dict[str, float]
            Per-head probability of the chosen action.
        """
        obs_t = obs_to_tensor(obs, self.device)
        mask_t = action_mask_to_tensor(action_mask, self.device) if action_mask else None

        with torch.no_grad():
            actions, log_probs, _ = self.policy.get_action(obs_t, mask_t, deterministic=deterministic)

        action = actions[0]
        probs = {k: float(np.exp(log_probs[k].cpu().numpy()[0])) for k in ACTION_KEYS if k in log_probs}
        return action, probs

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------

    def act_batch(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select actions for a **batched** obs dict (leading batch dimension).

        Parameters
        ----------
        obs : dict
            Batched obs, e.g. from SubprocVecEnv (each value shaped ``(N, ...)``)

        Returns
        -------
        actions : np.ndarray shape (N, 4)
        """
        obs_t = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
        mask_t = (
            {k: torch.from_numpy(v).bool().to(self.device) for k, v in action_mask.items()}
            if action_mask else None
        )

        with torch.no_grad():
            actions, _, _ = self.policy.get_action(obs_t, mask_t, deterministic=deterministic)

        return actions

    # ------------------------------------------------------------------
    # Logits / probability inspection
    # ------------------------------------------------------------------

    def logits(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return **raw logits** for each action head (no masking applied).

        Useful for debugging policy confidence and bias analysis.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"color"``, ``"position"``, ``"value"``, ``"decision"``
        """
        obs_t = obs_to_tensor(obs, self.device)

        with torch.no_grad():
            action_logits, _, _ = self.policy.forward(obs_t, action_mask=None)

        return {k: v.cpu().numpy()[0] for k, v in action_logits.items()}

    def probs(
        self,
        obs: Dict[str, np.ndarray],
        action_mask: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return softmax probabilities for each action head.

        Returns
        -------
        dict[str, np.ndarray]  — same keys as ``logits()``
        """
        import torch.nn.functional as F
        raw = self.logits(obs, action_mask)
        return {k: F.softmax(torch.tensor(v), dim=-1).numpy() for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Value estimate
    # ------------------------------------------------------------------

    def value(self, obs: Dict[str, np.ndarray]) -> float:
        """Return the value-head estimate for a single obs."""
        obs_t = obs_to_tensor(obs, self.device)
        with torch.no_grad():
            features, _, _ = self.policy.encoder(obs_t)
            v = self.policy.value_head(features)
        return float(v.cpu().item())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        path = getattr(self, "_checkpoint_path", "in-memory")
        ts = getattr(self, "_timesteps", "?")
        return f"ModelAgent(checkpoint={path!r}, timesteps={ts})"
