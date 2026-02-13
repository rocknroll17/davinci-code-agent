"""
Rollout Buffer for PPO Training.

Stores trajectories from environment interactions for batch updates.
"""

import torch
import numpy as np
from typing import Dict, List, Generator, Optional
from dataclasses import dataclass, field


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
        
    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.transitions.append(transition)
    
    def clear(self) -> None:
        """Clear all stored transitions and cached tensors."""
        self.transitions.clear()
        self.episode_start_idx = 0
        # Clear cached tensors
        self._cached_n = 0
        if hasattr(self, '_cached_obs'):
            del self._cached_obs
        if hasattr(self, '_cached_actions'):
            del self._cached_actions
        if hasattr(self, '_cached_log_probs'):
            del self._cached_log_probs
        if hasattr(self, '_cached_masks'):
            del self._cached_masks
        if hasattr(self, '_cached_values'):
            del self._cached_values
        if hasattr(self, '_cached_player_ids'):
            del self._cached_player_ids
    
    @property
    def size(self) -> int:
        """Number of stored transitions."""
        return len(self.transitions)
    
    def is_full(self) -> bool:
        """Deprecated: 에피소드 기반 수집에서는 사용하지 않음."""
        return False
    
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
        if not hasattr(self, '_cached_tensors') or self._cached_n != n:
            self._cache_tensors()
            self._cached_n = n
        
        indices = np.random.permutation(n)
        
        for start_idx in range(0, n, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Use pre-computed tensors with indexing
            batch_obs = {
                key: self._cached_obs[key][batch_indices]
                for key in self._cached_obs.keys()
            }
            
            batch_actions = {
                key: self._cached_actions[key][batch_indices]
                for key in self._cached_actions.keys()
            }
            
            batch_old_log_probs = {
                key: self._cached_log_probs[key][batch_indices]
                for key in self._cached_log_probs.keys()
            }
            
            batch_action_mask = None
            if self._cached_masks is not None:
                batch_action_mask = {
                    key: self._cached_masks[key][batch_indices]
                    for key in self._cached_masks.keys()
                }
            
            batch_old_values = self._cached_values[batch_indices]
            batch_returns = torch.from_numpy(returns[batch_indices]).float().to(self.device)
            batch_advantages = torch.from_numpy(advantages[batch_indices]).float().to(self.device)
            
            yield {
                "obs": batch_obs,
                "player_id": self._cached_player_ids[batch_indices],
                "actions": batch_actions,
                "old_log_probs": batch_old_log_probs,
                "action_mask": batch_action_mask,
                "old_values": batch_old_values,
                "returns": batch_returns,
                "advantages": batch_advantages
            }
    
    def _cache_tensors(self) -> None:
        """Pre-compute tensors from transitions for efficient batching."""
        n = len(self.transitions)
        
        # Cache observations
        self._cached_obs = {
            key: torch.from_numpy(
                np.stack([self.transitions[i].obs[key] for i in range(n)])
            ).float().to(self.device)
            for key in self.transitions[0].obs.keys()
        }
        
        # Cache player IDs
        self._cached_player_ids = torch.from_numpy(
            np.array([t.player_id for t in self.transitions])
        ).long().to(self.device)
        
        # Cache actions
        self._cached_actions = {
            key: torch.from_numpy(
                np.array([self.transitions[i].action[k_idx] for i in range(n)])
            ).long().to(self.device)
            for k_idx, key in enumerate(["color", "position", "value", "decision"])
        }
        
        # Cache old log probs
        self._cached_log_probs = {
            key: torch.from_numpy(
                np.array([self.transitions[i].log_probs.get(key, 0.0) for i in range(n)])
            ).float().to(self.device)
            for key in ["color", "position", "value", "decision"]
        }
        
        # Cache action masks
        if self.transitions[0].action_mask is not None:
            self._cached_masks = {
                key: torch.from_numpy(
                    np.stack([self.transitions[i].action_mask[key] for i in range(n)])
                ).bool().to(self.device)
                for key in self.transitions[0].action_mask.keys()
            }
        else:
            self._cached_masks = None
        
        # Cache old values
        self._cached_values = torch.from_numpy(
            np.array([t.value for t in self.transitions])
        ).float().to(self.device)
