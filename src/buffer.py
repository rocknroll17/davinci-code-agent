"""
Rollout Buffer for PPO Training.

Stores trajectories from environment interactions for batch updates.
"""

import torch
import numpy as np
from typing import Dict, List, Generator, Optional
from dataclasses import dataclass, field

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
        last_values_per_env: Optional[Dict[int, float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE, separated per environment.
        
        Each env's trajectory is processed independently to prevent
        cross-env value leakage in GAE computation.
        
        Args:
            last_value: Default bootstrap value for incomplete episodes
            last_values_per_env: Per-env bootstrap values {env_id: value}
            
        Returns:
            Tuple of (returns, advantages) arrays
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        
        # Group transition indices by env_id
        env_groups: Dict[int, List[int]] = {}
        for idx, t in enumerate(self.transitions):
            eid = t.env_id
            if eid not in env_groups:
                env_groups[eid] = []
            env_groups[eid].append(idx)
        
        # Compute GAE separately for each env's trajectory
        for eid, indices in env_groups.items():
            # Get bootstrap value for this env
            if last_values_per_env is not None and eid in last_values_per_env:
                env_last_value = last_values_per_env[eid]
            else:
                env_last_value = last_value
            env_last_value = float(env_last_value) if not np.isnan(env_last_value) and not np.isinf(env_last_value) else 0.0
            
            env_n = len(indices)
            env_rewards = np.array([
                self.transitions[i].reward if not np.isnan(self.transitions[i].reward) and not np.isinf(self.transitions[i].reward) else 0.0
                for i in indices
            ], dtype=np.float32)
            env_values = np.array([
                self.transitions[i].value if not np.isnan(self.transitions[i].value) and not np.isinf(self.transitions[i].value) else 0.0
                for i in indices
            ], dtype=np.float32)
            env_dones = np.array([self.transitions[i].done for i in indices], dtype=bool)
            
            # Append last value for bootstrapping
            env_values_ext = np.append(env_values, env_last_value)
            
            last_gae = 0.0
            env_advantages = np.zeros(env_n, dtype=np.float32)
            
            for t in reversed(range(env_n)):
                if env_dones[t]:
                    next_value = 0.0
                    last_gae = 0.0
                else:
                    next_value = env_values_ext[t + 1]
                delta = env_rewards[t] + self.gamma * next_value - env_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                env_advantages[t] = last_gae if not np.isnan(last_gae) and not np.isinf(last_gae) else 0.0
            
            # Write back to global advantages array
            for local_idx, global_idx in enumerate(indices):
                advantages[global_idx] = env_advantages[local_idx]
        
        # Compute returns
        values = np.array([
            t.value if not np.isnan(t.value) and not np.isinf(t.value) else 0.0
            for t in self.transitions
        ], dtype=np.float32)
        returns = advantages + values
        returns = np.where(np.isnan(returns) | np.isinf(returns), 0.0, returns)

        return returns, advantages
    
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
