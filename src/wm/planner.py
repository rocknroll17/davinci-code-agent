"""
Pure world-model planner: the world model plays by itself, no learned policy.

At every real move it enumerates ALL legal actions for the current phase,
rolls each candidate forward inside the RSSM (prior only, M stochastic samples
per candidate to average over the model's uncertainty), scores each rollout by
the discounted sum of predicted rewards with the negamax perspective sign, and
plays the argmax first action. Model-predictive control — the actor/critic
networks are not used at all.

Candidate spaces are small enough for exhaustive enumeration:
  DRAW ≤ 2 colors, GUESS ≤ 12×13 (position × masked values),
  DECISION = 2, JOKER ≤ 13 insert slots.
Beyond the first action the rollout continues with uniformly sampled legal
actions (mask head), so the score estimates "how good does the future look
after this move, under neutral continuation".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.model import obs_to_tensor
from src.wm.nets import Actor, WMConfig, WorldModel


class WMPlannerAgent:
    """Model-predictive-control agent over a trained WorldModel.

    Compatible with the run_episode Agent interface; keeps the recurrent
    posterior state across the moves of one game (call reset() between games).
    """

    def __init__(self, wm: WorldModel, device: Optional[torch.device] = None,
                 horizon: int = 8, n_samples: int = 8, gamma: float = 0.99) -> None:
        self.device = device or torch.device("cpu")
        self.wm = wm.to(self.device).eval()
        self.horizon = horizon
        self.n_samples = n_samples
        self.gamma = gamma
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None,
                        **kwargs) -> "WMPlannerAgent":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        wm = WorldModel(WMConfig(**dict(ckpt["config"]["wm"])))
        wm.load_state_dict(ckpt["wm"], strict=False)
        return cls(wm, device, **kwargs)

    def reset(self) -> None:
        self._h, self._z = self.wm.rssm.initial(1, self.device)
        self._prev_action = torch.zeros(1, 5, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------

    @staticmethod
    def _candidates(phase: int, mask: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """All legal 5-component actions for the current phase."""
        out = []
        if phase == 0:      # DRAW: color
            for c in np.flatnonzero(mask["color"]):
                out.append(np.array([c, 0, 0, 0, 0]))
        elif phase == 1:    # GUESS: position x value (per-position value mask)
            for p in np.flatnonzero(mask["position"]):
                for v in np.flatnonzero(mask["value"][p]):
                    out.append(np.array([0, p, v, 0, 0]))
        elif phase == 2:    # DECISION: stop / continue
            for d in (0, 1):
                out.append(np.array([0, 0, 0, d, 0]))
        else:               # JOKER: insert index
            for j in np.flatnonzero(mask["joker"]):
                out.append(np.array([0, 0, 0, 0, j]))
        return out or [np.zeros(5, dtype=np.int64)]

    @torch.no_grad()
    def _rollout_scores(self, state0: torch.Tensor, candidates: List[np.ndarray]) -> np.ndarray:
        """Score = E[discounted negamax reward sum] per candidate first action."""
        N, M, H = len(candidates), self.n_samples, self.horizon
        cfg = self.wm.cfg
        # tile: each candidate replicated M times → batch N*M
        first = torch.as_tensor(np.stack(candidates), dtype=torch.long,
                                device=self.device).repeat_interleave(M, 0)
        h = state0[:, :cfg.deter_dim].expand(N * M, -1).contiguous()
        z = state0[:, cfg.deter_dim:].expand(N * M, -1).contiguous()

        scores = torch.zeros(N * M, device=self.device)
        sign = torch.ones(N * M, device=self.device)      # negamax perspective
        alive = torch.ones(N * M, device=self.device)     # continue mass
        action = first
        discount = 1.0
        for _ in range(H):
            h = self.wm.rssm.step_deter(h, z, action)
            _, z = self.wm.rssm.prior(h)
            state = torch.cat([h, z], dim=-1)
            # arriving-state heads: reward of the action just taken, game-alive,
            # perspective-flip → all conditioned on the state the action created
            r = self.wm.twohot.decode(self.wm.reward_head(state))
            cont = torch.sigmoid(self.wm.continue_head(state)).squeeze(-1)
            flip = torch.sigmoid(self.wm.flip_head(state)).squeeze(-1)
            scores = scores + discount * alive * sign * r
            sign = sign * (1.0 - 2.0 * flip)
            alive = alive * cont
            discount *= self.gamma
            # neutral continuation: uniform over predicted-legal component values
            action = self._sample_neutral(state)
        return scores.view(N, M).mean(1).cpu().numpy()

    def _sample_neutral(self, state: torch.Tensor) -> torch.Tensor:
        """Uniform legal action per imagined state (mask head + decoded phase)."""
        masks = self.wm.predict_masks(state)
        phase = self.wm.decode_phase(state)
        B = state.shape[0]
        comps = []
        specs = [("color", 2), ("position", 13), ("value", 13), ("decision", 2), ("joker", 13)]
        for key, n in specs:
            m = masks[key]
            if key == "value":
                m = m.view(B, 13, 13).any(1)
            m = m[:, :n].float()
            m = torch.where(m.sum(-1, keepdim=True) > 0, m, torch.ones_like(m))
            comps.append(torch.multinomial(m, 1).squeeze(-1))
        del phase  # phase gating is implicit: env-irrelevant components are ignored
        return torch.stack(comps, dim=-1)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs, action_mask=None, deterministic: bool = True):
        obs_t = obs_to_tensor(obs, self.device)
        embed = self.wm.encoder(obs_t)
        self._h = self.wm.rssm.step_deter(self._h, self._z, self._prev_action)
        _, self._z = self.wm.rssm.posterior(self._h, embed)
        state = torch.cat([self._h, self._z], dim=-1)

        phase = int(np.argmax(obs["phase"]))
        candidates = self._candidates(phase, action_mask)
        if len(candidates) == 1:
            best = candidates[0]
        else:
            scores = self._rollout_scores(state, candidates)
            best = candidates[int(np.argmax(scores))]

        best = np.asarray(best, dtype=np.int64)
        self._prev_action = torch.as_tensor(best[None], device=self.device)
        return best, {}
