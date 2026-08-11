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
    """Collects one env's transitions until the episode ends.

    Per step t it stores (obs_t, a_t, r_t, cont_t, flip_t) where flip_t marks
    that the acting player CHANGED between step t-1 and t (0 at episode start).

    ``set_terminal`` records the post-game observation; ``pack`` appends it as
    one extra frame (dummy action, r=0, cont=0). The reward/continue heads are
    trained on the ARRIVING state s_{t+1}, so without this frame the terminal
    win/lose reward of the last action would have no state to be predicted from.
    """

    def __init__(self) -> None:
        self.obs: Dict[str, List[np.ndarray]] = {k: [] for k in OBS_KEYS}
        self.masks: Dict[str, List[np.ndarray]] = {k: [] for k in MASK_KEYS}
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.continues: List[float] = []
        self.flips: List[float] = []
        self.hidden_values: List[np.ndarray] = []   # true opp values, -1 = n/a
        self.terminal_obs: Optional[Dict[str, np.ndarray]] = None

    def add(self, obs, action, reward, done, masks, flip: float = 0.0,
            hidden_values: Optional[np.ndarray] = None) -> None:
        for k in OBS_KEYS:
            self.obs[k].append(np.asarray(obs[k]))
        for k in MASK_KEYS:
            self.masks[k].append(np.asarray(masks[k]))
        self.actions.append(np.asarray(action))
        self.rewards.append(float(reward))
        self.continues.append(0.0 if done else 1.0)
        self.flips.append(float(flip))
        self.hidden_values.append(
            np.full(13, -1, dtype=np.int8) if hidden_values is None
            else np.asarray(hidden_values, dtype=np.int8))

    def set_terminal(self, obs) -> None:
        self.terminal_obs = {k: np.asarray(obs[k]) for k in OBS_KEYS}

    def __len__(self) -> int:
        return len(self.actions)

    def pack(self) -> Dict:
        obs = {k: list(v) for k, v in self.obs.items()}
        masks = {k: list(v) for k, v in self.masks.items()}
        actions = list(self.actions)
        rewards = list(self.rewards)
        continues = list(self.continues)
        flips = list(self.flips)
        hidden = list(self.hidden_values)
        if self.terminal_obs is not None:
            for k in OBS_KEYS:
                obs[k].append(self.terminal_obs[k])
            for k in MASK_KEYS:
                masks[k].append(masks[k][-1])   # masks are meaningless post-game
            actions.append(np.zeros_like(actions[-1]))
            rewards.append(0.0)
            continues.append(0.0)
            flips.append(0.0)
            hidden.append(np.full(13, -1, dtype=np.int8))
        return {
            "obs": {k: np.stack(v) for k, v in obs.items()},
            "masks": {k: np.stack(v) for k, v in masks.items()},
            "actions": np.stack(actions).astype(np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "continues": np.asarray(continues, dtype=np.float32),
            "flips": np.asarray(flips, dtype=np.float32),
            "hidden_values": np.stack(hidden),
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
        act_b, rew_b, cont_b, first_b, valid_b, flip_b, hid_b = [], [], [], [], [], [], []

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
            flip_b.append(take(ep.get("flips", np.zeros(T, dtype=np.float32))))
            hid = ep.get("hidden_values")
            if hid is None:
                hid = np.full((T, 13), -1, dtype=np.int8)
            seg = hid[start:end]
            if pad:
                # pad with -1 (no belief target), NOT zeros
                seg = np.concatenate(
                    [seg, np.full((pad, 13), -1, dtype=seg.dtype)])
            hid_b.append(seg)
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
            "flips": to(flip_b),
            "hidden_values": to(hid_b, torch.int64),
            "is_first": to(first_b),
            "valid": to(valid_b),
        }
