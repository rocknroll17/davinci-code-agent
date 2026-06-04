"""
EvalSuite — deep evaluation metrics for DaVinci Code policy.

Usage
-----
from src.eval_suite import EvalSuite

# from a live trainer:
report = EvalSuite.run(trainer, n_episodes=200)
# or from a loaded agent / policy:
report = EvalSuite.run_agent(agent, n_episodes=200)
report = EvalSuite.run_policy(policy, device, n_episodes=200)

print(report.win_rate_p0, report.guess_accuracy, report.belief_accuracy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.constants import MAX_HAND_SIZE, CardValue, Phase
from src.env import DaVinciCodeEnv
from src.model import DaVinciCodePolicy, obs_to_tensor

# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Rich evaluation metrics for a single policy."""

    n_episodes: int = 0

    # ── win rates ──
    player0_wins: int = 0
    player1_wins: int = 0
    draws: int = 0

    # ── episode length ──
    episode_lengths: List[int] = field(default_factory=list)

    # ── rewards ──
    episode_rewards_p0: List[float] = field(default_factory=list)
    episode_rewards_p1: List[float] = field(default_factory=list)

    # ── guess accuracy ──
    guess_total: int = 0
    guess_correct: int = 0
    joker_guess_total: int = 0
    joker_guess_correct: int = 0

    # per-value[0-12] and per-position[0-12] accumulators
    guess_correct_by_value: List[int] = field(default_factory=lambda: [0] * 13)
    guess_total_by_value: List[int] = field(default_factory=lambda: [0] * 13)
    guess_correct_by_pos: List[int] = field(default_factory=lambda: [0] * 13)
    guess_total_by_pos: List[int] = field(default_factory=lambda: [0] * 13)

    # ── invalid actions ──
    invalid_action_count: int = 0
    total_steps: int = 0

    # ── streak ──
    streak_lengths: List[int] = field(default_factory=list)  # per-episode max streak

    # ── action distributions ──
    color_dist: List[int] = field(default_factory=lambda: [0, 0])         # [BLACK, WHITE]
    value_dist: List[int] = field(default_factory=lambda: [0] * 13)       # 0-12
    decision_dist: List[int] = field(default_factory=lambda: [0, 0])      # [STOP, CONTINUE]
    position_dist: List[int] = field(default_factory=lambda: [0] * 13)    # 0-12

    # ── belief accuracy (optional) ──
    belief_total_positions: int = 0
    belief_correct_positions: int = 0

    # ── metadata ──
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def win_rate_p0(self) -> float:
        return self.player0_wins / self.n_episodes if self.n_episodes else 0.0

    @property
    def win_rate_p1(self) -> float:
        return self.player1_wins / self.n_episodes if self.n_episodes else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.n_episodes if self.n_episodes else 0.0

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_reward_p0(self) -> float:
        return float(np.mean(self.episode_rewards_p0)) if self.episode_rewards_p0 else 0.0

    @property
    def mean_reward_p1(self) -> float:
        return float(np.mean(self.episode_rewards_p1)) if self.episode_rewards_p1 else 0.0

    @property
    def guess_accuracy(self) -> float:
        return self.guess_correct / self.guess_total if self.guess_total else 0.0

    @property
    def joker_accuracy(self) -> float:
        return (self.joker_guess_correct / self.joker_guess_total
                if self.joker_guess_total else 0.0)

    @property
    def invalid_action_rate(self) -> float:
        return self.invalid_action_count / self.total_steps if self.total_steps else 0.0

    @property
    def mean_streak_length(self) -> float:
        return float(np.mean(self.streak_lengths)) if self.streak_lengths else 0.0

    @property
    def belief_accuracy(self) -> Optional[float]:
        if self.belief_total_positions == 0:
            return None
        return self.belief_correct_positions / self.belief_total_positions

    @property
    def accuracy_by_value(self) -> Dict[int, float]:
        out = {}
        for v in range(13):
            t = self.guess_total_by_value[v]
            out[v] = self.guess_correct_by_value[v] / t if t > 0 else None
        return out

    @property
    def accuracy_by_position(self) -> Dict[int, float]:
        out = {}
        for p in range(13):
            t = self.guess_total_by_pos[p]
            out[p] = self.guess_correct_by_pos[p] / t if t > 0 else None
        return out


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------

class EvalSuite:
    """
    Runs deep evaluation and returns a rich ``EvalReport``.

    Class methods
    -------------
    run(trainer, n_episodes, seed)
        Evaluate a live ``PPOTrainer`` (uses its existing ``.env`` + policy).
    run_agent(agent, n_episodes, device, seed)
        Evaluate a ``ModelAgent`` using a fresh env.
    run_policy(policy, device, n_episodes, seed)
        Low-level: evaluate any ``DaVinciCodePolicy`` directly.
    """

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    @classmethod
    def run(cls, trainer, n_episodes: int = 200, seed: Optional[int] = None) -> EvalReport:
        """Evaluate a live ``PPOTrainer``."""
        return cls.run_policy(trainer.policy, trainer.device,
                              n_episodes=n_episodes, seed=seed)

    @classmethod
    def run_agent(
        cls,
        agent,                          # ModelAgent
        n_episodes: int = 200,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> EvalReport:
        """Evaluate a ``ModelAgent``."""
        dev = device or agent.device
        return cls.run_policy(agent.policy, dev,
                              n_episodes=n_episodes, seed=seed)

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------

    @classmethod
    def run_policy(
        cls,
        policy: DaVinciCodePolicy,
        device: torch.device,
        n_episodes: int = 200,
        seed: Optional[int] = None,
    ) -> EvalReport:
        """Run full evaluation and return an ``EvalReport``."""
        from src.agent import ModelAgent
        from src.runner import run_episode

        env = DaVinciCodeEnv(seed=seed)
        report = EvalReport(n_episodes=n_episodes)
        agent = ModelAgent(policy, device)

        policy.eval()
        with torch.no_grad():
            for ep in range(n_episodes):
                ep_seed = (seed + ep) if seed is not None else None
                ep_state = {"max_streak": 0}

                def on_step(ctx, ep_state=ep_state):
                    phase_idx = ctx.phase
                    action_np = ctx.action
                    result = ctx.result

                    # ── belief accuracy: prediction (from obs before move) vs hidden values ──
                    hidden_vals = ctx.info_before.get("hidden_values")
                    if hidden_vals is not None:
                        obs_t = obs_to_tensor(ctx.obs_before, device)
                        feats, _cp, opp_per_pos = policy.encoder(obs_t)
                        global_exp = feats.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
                        combined = torch.cat([global_exp, opp_per_pos], dim=-1)
                        belief_logits = policy.belief_head(combined)  # (1, 13, 13)
                        predicted = belief_logits[0].argmax(dim=-1).cpu().numpy()
                        hidden_np = np.asarray(hidden_vals, dtype=np.int8)
                        visible_mask = hidden_np >= 0
                        if visible_mask.any():
                            report.belief_total_positions += int(visible_mask.sum())
                            report.belief_correct_positions += int(
                                (predicted[visible_mask] == hidden_np[visible_mask]).sum()
                            )

                    report.total_steps += 1

                    color, pos, val, dec = (int(action_np[0]), int(action_np[1]),
                                            int(action_np[2]), int(action_np[3]))
                    if phase_idx == Phase.DRAW.value:
                        report.color_dist[color] += 1
                    elif phase_idx == Phase.GUESS.value:
                        report.position_dist[min(pos, 12)] += 1
                        report.value_dist[min(val, 12)] += 1
                    elif phase_idx == Phase.DECISION.value:
                        report.decision_dist[dec] += 1

                    if result is not None and getattr(result, "is_invalid", False):
                        report.invalid_action_count += 1

                    if phase_idx == Phase.GUESS.value and result is not None and hasattr(result, "is_correct"):
                        if not result.is_invalid:
                            report.guess_total += 1
                            correct = bool(result.is_correct)
                            if correct:
                                report.guess_correct += 1
                            target_v = min(int(action_np[2]), 12)
                            report.guess_total_by_value[target_v] += 1
                            if correct:
                                report.guess_correct_by_value[target_v] += 1
                            target_p = min(int(action_np[1]), 12)
                            report.guess_total_by_pos[target_p] += 1
                            if correct:
                                report.guess_correct_by_pos[target_p] += 1
                            if val == CardValue.JOKER:
                                report.joker_guess_total += 1
                                if correct:
                                    report.joker_guess_correct += 1

                    streak = getattr(ctx.env, "_streak", 0)
                    if streak > ep_state["max_streak"]:
                        ep_state["max_streak"] = streak

                res = run_episode(env, agent, deterministic=True, seed=ep_seed, on_step=on_step)

                if res.winner == 0:
                    report.player0_wins += 1
                elif res.winner == 1:
                    report.player1_wins += 1
                else:
                    report.draws += 1

                report.episode_lengths.append(res.length)
                report.episode_rewards_p0.append(res.rewards[0])
                report.episode_rewards_p1.append(res.rewards[1])
                report.streak_lengths.append(ep_state["max_streak"])

        policy.train()
        env.close()
        return report
