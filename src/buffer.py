"""
Rollout Buffer for PPO Training.

Stores trajectories from environment interactions for batch updates.
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

import numpy as np
import torch

from src.constants import ACTION_KEYS


@dataclass
class Transition:
    """Single transition in the environment."""
    obs: Dict[str, np.ndarray]
    player_id: int
    action: np.ndarray
    reward: float
    done: bool
    log_probs: Dict[str, float]
    value: float
    action_mask: Optional[Dict[str, np.ndarray]] = None
    env_id: int = 0  # Which parallel env this transition came from
    hidden_values: Optional[np.ndarray] = None  # True opponent hidden card values for belief loss
    stream_end: bool = False  # this player's last move in the game (ends their GAE stream)


class RolloutBuffer:
    """
    Buffer to store rollout trajectories for PPO training.
    
    Implements Generalized Advantage Estimation (GAE) for
    computing advantages and returns.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """
        Initialize rollout buffer.
        
        에피소드 기반 수집이므로 크기 제한 없이 동적으로 증가.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Target device for tensors
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.transitions: List[Transition] = []
        self.episode_start_idx = 0
        self._cache: dict = {}  # holds pre-computed tensors; keyed by attribute name
        self._cached_n: int = 0
        
    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.transitions.append(transition)
    
    def clear(self) -> None:
        """Clear all stored transitions and cached tensors."""
        self.transitions.clear()
        self.episode_start_idx = 0
        self._cache.clear()
        self._cached_n = 0
    
    @property
    def size(self) -> int:
        """Number of stored transitions."""
        return len(self.transitions)
    
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        last_values_per_env: Optional[Dict[tuple, float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantages and returns for every stored transition.

        Self-play: both seats of a game are stored in one buffer, but each
        transition's obs, value and reward are from the *acting* player's point
        of view. So GAE runs separately on each (env_id, player_id) stream — for a
        player, the opponent's turns are just part of the environment.

        A stream ends on `done` (the move that ended the game) or `stream_end`
        (this player's own last move of the game). A stream that does not end
        bootstraps from last_values_per_env[(env_id, player_id)], else last_value.

        Returns (returns, advantages), each (N,) float32 in buffer order.
        NaN/Inf rewards, values and advantages are replaced by 0.
        """
        advantages = np.zeros(len(self.transitions), dtype=np.float32)

        streams: Dict[tuple, List[int]] = {}  # (env_id, player_id) -> buffer indices, in order
        for idx, t in enumerate(self.transitions):
            streams.setdefault((t.env_id, t.player_id), []).append(idx)

        for key, indices in streams.items():
            bootstrap = last_value
            if last_values_per_env is not None and key in last_values_per_env:
                bootstrap = last_values_per_env[key]
            stream_adv = self._gae([self.transitions[i] for i in indices], _finite_or_zero(bootstrap))
            advantages[indices] = stream_adv

        values = np.array([_finite_or_zero(t.value) for t in self.transitions], dtype=np.float32)
        returns = advantages + values  # what V(s) is trained towards
        returns = np.where(np.isnan(returns) | np.isinf(returns), 0.0, returns)
        return returns, advantages

    def _gae(self, stream: List[Transition], bootstrap: float) -> np.ndarray:
        """GAE over one player's transitions (in time order) → advantages (T,) float32.

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)        (V after an ending move = 0)
            A_t     = delta_t + gamma * lambda * A_{t+1}
        """
        rewards = np.array([_finite_or_zero(t.reward) for t in stream], dtype=np.float32)
        values = np.array([_finite_or_zero(t.value) for t in stream], dtype=np.float32)
        ends = np.array([t.done or t.stream_end for t in stream], dtype=bool)
        next_values = np.append(values, bootstrap)  # V(s_{t+1}); the last one is the bootstrap

        advantages = np.zeros(len(stream), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(stream))):  # backwards: A_t needs A_{t+1}
            if ends[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = next_values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = _finite_or_zero(last_gae)
        return advantages

    def get_batches(
        self,
        batch_size: int,
        returns: np.ndarray,
        advantages: np.ndarray
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generate mini-batches for PPO update.
        
        Optimized version that pre-computes tensors for faster batching.
        
        Args:
            batch_size: Size of each mini-batch
            returns: Computed returns array
            advantages: Computed advantages array
            
        Yields:
            Dictionary containing batch data
        """
        n = len(self.transitions)
        
        # Pre-compute all arrays once (avoid repeated list comprehensions)
        if not hasattr(self, '_cache') or self._cached_n != n:
            self._cache_tensors()
            self._cached_n = n
        
        indices = np.random.permutation(n)
        
        for start_idx in range(0, n, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Use pre-computed tensors with indexing
            batch_obs = {
                key: self._cache['obs'][key][batch_indices]
                for key in self._cache['obs']
            }
            
            batch_actions = {
                key: self._cache['actions'][key][batch_indices]
                for key in self._cache['actions']
            }
            
            batch_old_log_probs = {
                key: self._cache['log_probs'][key][batch_indices]
                for key in self._cache['log_probs']
            }
            
            batch_action_mask = None
            if self._cache.get('masks') is not None:
                batch_action_mask = {
                    key: self._cache['masks'][key][batch_indices]
                    for key in self._cache['masks']
                }
            
            batch_old_values = self._cache['values'][batch_indices]
            batch_returns = torch.from_numpy(returns[batch_indices]).float().to(self.device)
            batch_advantages = torch.from_numpy(advantages[batch_indices]).float().to(self.device)
            
            batch_hidden_values = None
            if self._cache.get('hidden_values') is not None:
                batch_hidden_values = self._cache['hidden_values'][batch_indices]
            
            yield {
                "obs": batch_obs,
                "player_id": self._cache['player_ids'][batch_indices],
                "actions": batch_actions,
                "old_log_probs": batch_old_log_probs,
                "action_mask": batch_action_mask,
                "old_values": batch_old_values,
                "returns": batch_returns,
                "advantages": batch_advantages,
                "hidden_values": batch_hidden_values
            }
    
    def _cache_tensors(self) -> None:
        """Pre-compute tensors from transitions for efficient batching."""
        n = len(self.transitions)
        
        # Cache observations
        self._cache['obs'] = {
            key: torch.from_numpy(
                np.stack([self.transitions[i].obs[key] for i in range(n)])
            ).float().to(self.device)
            for key in self.transitions[0].obs.keys()
        }
        
        # Cache player IDs
        self._cache['player_ids'] = torch.from_numpy(
            np.array([t.player_id for t in self.transitions])
        ).long().to(self.device)
        
        # Cache actions — ACTION_KEYS defines index ↔ name mapping
        self._cache['actions'] = {
            key: torch.from_numpy(
                np.array([self.transitions[i].action[k_idx] for i in range(n)])
            ).long().to(self.device)
            for k_idx, key in enumerate(ACTION_KEYS)
        }
        
        # Cache old log probs
        self._cache['log_probs'] = {
            key: torch.from_numpy(
                np.array([self.transitions[i].log_probs.get(key, 0.0) for i in range(n)])
            ).float().to(self.device)
            for key in ACTION_KEYS
        }
        
        # Cache action masks
        if self.transitions[0].action_mask is not None:
            self._cache['masks'] = {
                key: torch.from_numpy(
                    np.stack([self.transitions[i].action_mask[key] for i in range(n)])
                ).bool().to(self.device)
                for key in self.transitions[0].action_mask.keys()
            }
        else:
            self._cache['masks'] = None
        
        # Cache old values
        self._cache['values'] = torch.from_numpy(
            np.array([t.value for t in self.transitions])
        ).float().to(self.device)
        
        # Cache hidden values for belief loss
        if self.transitions[0].hidden_values is not None:
            self._cache['hidden_values'] = torch.from_numpy(
                np.stack([t.hidden_values for t in self.transitions])
            ).long().to(self.device)
        else:
            self._cache['hidden_values'] = None


def _finite_or_zero(x: float) -> float:
    """x, or 0.0 if x is NaN or Inf."""
    return x if not np.isnan(x) and not np.isinf(x) else 0.0
