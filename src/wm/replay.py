"""
Sequence replay buffer for world-model training.

Stores whole episodes (one per finished game per env) and samples fixed-length
subsequences (B, L) for RSSM training, padding short episodes and marking
episode starts with `is_first` so the RSSM resets its state mid-batch.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

OBS_KEYS = ("phase", "my_hand", "opponent_hand", "remaining_deck", "constraint_matrix")
MASK_KEYS = ("color", "position", "value", "decision", "joker")


class EpisodeAccumulator:
    """Collects one env's transitions until the episode ends."""

    def __init__(self) -> None:
        self.obs: Dict[str, List[np.ndarray]] = {k: [] for k in OBS_KEYS}
        self.masks: Dict[str, List[np.ndarray]] = {k: [] for k in MASK_KEYS}
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.continues: List[float] = []

    def add(self, obs, action, reward, done, masks) -> None:
        for k in OBS_KEYS:
            self.obs[k].append(np.asarray(obs[k]))
        for k in MASK_KEYS:
            self.masks[k].append(np.asarray(masks[k]))
        self.actions.append(np.asarray(action))
        self.rewards.append(float(reward))
        self.continues.append(0.0 if done else 1.0)

    def __len__(self) -> int:
        return len(self.actions)

    def pack(self) -> Dict:
        return {
            "obs": {k: np.stack(v) for k, v in self.obs.items()},
            "masks": {k: np.stack(v) for k, v in self.masks.items()},
            "actions": np.stack(self.actions).astype(np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "continues": np.asarray(self.continues, dtype=np.float32),
        }


class SequenceReplay:
    """Episode store with uniform subsequence sampling."""

    def __init__(self, capacity_episodes: int = 5000, seq_len: int = 32,
                 seed: Optional[int] = None) -> None:
        self.episodes: deque = deque(maxlen=capacity_episodes)
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def add_episode(self, episode: Dict) -> None:
        if len(episode["actions"]) >= 2:
            self.episodes.append(episode)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def n_steps(self) -> int:
        return sum(len(e["actions"]) for e in self.episodes)

    def sample(self, batch_size: int, device: torch.device) -> Dict:
        """Sample (B, L) subsequences. Short episodes are zero-padded; padded
        steps carry continue=0 and a fresh is_first so they don't leak."""
        L = self.seq_len
        obs_batch = {k: [] for k in OBS_KEYS}
        mask_batch = {k: [] for k in MASK_KEYS}
        act_b, rew_b, cont_b, first_b, valid_b = [], [], [], [], []

        for _ in range(batch_size):
            ep = self.rng.choice(self.episodes)
            T = len(ep["actions"])
            start = self.rng.randint(0, max(0, T - L))
            end = min(start + L, T)
            n = end - start
            pad = L - n

            def take(arr):
                seg = arr[start:end]
                if pad:
                    seg = np.concatenate(
                        [seg, np.zeros((pad,) + seg.shape[1:], dtype=seg.dtype)])
                return seg

            for k in OBS_KEYS:
                obs_batch[k].append(take(ep["obs"][k]))
            for k in MASK_KEYS:
                mask_batch[k].append(take(ep["masks"][k]))
            act_b.append(take(ep["actions"]))
            rew_b.append(take(ep["rewards"]))
            cont = take(ep["continues"])
            if pad:
                cont[n:] = 0.0
            cont_b.append(cont)
            first = np.zeros(L, dtype=np.float32)
            if start == 0:
                first[0] = 1.0
            first_b.append(first)
            valid = np.zeros(L, dtype=np.float32)
            valid[:n] = 1.0
            valid_b.append(valid)

        to = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x)).to(device=device, dtype=dt)
        return {
            "obs": {k: to(v) for k, v in obs_batch.items()},
            "masks": {k: to(v, torch.bool) for k, v in mask_batch.items()},
            "actions": to(act_b, torch.int64),
            "rewards": to(rew_b),
            "continues": to(cont_b),
            "is_first": to(first_b),
            "valid": to(valid_b),
        }
