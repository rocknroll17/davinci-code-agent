"""
PPO Trainer for Da Vinci Code Self-Play.

Implements Proximal Policy Optimization algorithm for training
the policy network through adversarial self-play.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from src.buffer import RolloutBuffer, Transition
from src.constants import MASK_VALUE, Phase
from src.episode import Episode
from src.model import DaVinciCodePolicy
from src.result.result import Result
from src.reward_config import RewardConfig
from src.vec_env import SubprocVecEnv, VectorDaVinciEnv
from src.visualizer import DaVinciVisualizer, get_visualizer

logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Training
    total_timesteps: int = 1000000000
    learning_rate: float =8e-5  # CNN 시절 1.25e-4에서 소폭 낮춤 (Transformer 안정성)
    n_envs: int =300  # 수집 병목 해소: epu=20 채우는 시간 최소화
    n_workers: int = 6  # rollout subproc 워커 수 = CPU 코어 수 (과구독 방지)
    episodes_per_update: int = 300 # n_envs와 동일 → ~1라운드로 수집 완료
    batch_size: int = 4096  # 1096→4096: minibatch 오버헤드↓, GPU/VRAM 활용↑ (CPU 병목 머신)
    n_epochs: int = 8  # 에포크 늘려 데이터 재사용 극대화
    
    # PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2   # 0.15→0.2: policy 업데이트 폭 확대, plateau 탈출
    clip_range_vf: float = 10.0  # 리턴 스케일 [-30,+30]에 맞춤 — 0.2는 절대값 clip이라 value 업데이트 과도 제한
    ent_coef: float = 0.01  # 0.03→0.01: plateau 단계에서 entropy 너무 높으면 수렴 방해
    color_ent_coef: float = 0.02  # ent_coef 2x 비율 유지
    vf_coef: float = 0.5  # Value function coefficient
    belief_coef: float = 0.2   # 0.5→0.2: 인코더가 RL/belief 양방향 gradient 동시 수신 → value feature 훼손 방지
    max_grad_norm: float = 0.5
    lr_end: float = 3e-5  # 1e-5→3e-5: plateau 단계에서 LR floor 상향
    
    # Model
    hidden_dim: int = 512
    n_heads: int = 4   # encoder transformer attention heads (token_dim=128 must be divisible)
    n_layers: int = 4  # encoder transformer layer count
    zero_init: bool = False  # if True, init all default weights/biases to 0 (designated inits kept)

    # Performance
    fp16: bool = False    # mixed-precision (autocast fp16 + GradScaler). Turing→fp16, not bf16.
    compile: bool = False  # torch.compile the encoder (skips .any()-branchy get_action)

    # Reward policy
    # When True: monotone reward — no win/lose game-outcome signal. Only per-guess
    # rewards (correct → +, wrong → −). Disables REWARD_WIN (env), REWARD_LOSE and
    # REWARD_DRAW_WIN/LOSE (trainer retroactive).
    monotone_reward: bool = False
    # Reward magnitudes (injected into env + Episode). Default == historical constants.
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Logging
    log_interval: int = 1
    save_interval: int = 1
    eval_interval: int = 50
    n_eval_episodes: int = 50
    
    # Paths
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    reset_optimizer_on_load: bool = False  # Reset optimizer state when loading checkpoint
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (nested RewardConfig → plain dict)."""
        d = {k: v for k, v in self.__dict__.items()}
        if isinstance(d.get("reward_config"), RewardConfig):
            d["reward_config"] = d["reward_config"].to_dict()
        return d


class PPOTrainer:
    """
    PPO Trainer for Da Vinci Code self-play.
    
    Implements the training loop where a single policy network
    plays against itself, learning from both perspectives.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize PPO trainer.

        Args:
            config: Training configuration
            device: Target device (auto-detect if None)
            rank: this process's DDP rank (0 for single-GPU)
            world_size: number of DDP ranks (1 for single-GPU)
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        # Initialize policy network
        self.policy = DaVinciCodePolicy(
            config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            zero_init=config.zero_init,
        ).to(self.device)

        # Mixed precision (fp16 + GradScaler). Disabled => no-op.
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)

        # Optional: compile the encoder (forward is branch-free; get_action has
        # data-dependent .any() branches so we leave it uncompiled).
        if config.compile:
            try:
                self.policy.encoder = torch.compile(self.policy.encoder, mode="reduce-overhead")
            except Exception as e:
                if rank == 0:
                    print(f"[compile] disabled ({e})")

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # LR Scheduler: linear decay from learning_rate to lr_end
        # 에피소드당 평균 ~60 스텝 추정
        avg_steps_per_episode = 45  # 실측 평균 ~45 steps/ep (이전 60은 과다 추정)
        total_updates = config.total_timesteps // (config.episodes_per_update * avg_steps_per_episode)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=config.lr_end / config.learning_rate,
            total_iters=total_updates
        )
        
        # Vectorized Environment (multiple games in parallel).
        # Per-rank seed offset so DDP ranks collect different games (not duplicates).
        self.n_envs = config.n_envs
        self._use_subproc = config.n_envs > 1
        env_seed = (1 + rank) * 100000 if world_size > 1 else None
        if self._use_subproc:
            self.vec_env = SubprocVecEnv(n_envs=config.n_envs, n_workers=config.n_workers,
                                         seed=env_seed, reward_config=config.reward_config)
        else:
            self.vec_env = VectorDaVinciEnv(n_envs=config.n_envs, seed=env_seed,
                                            reward_config=config.reward_config)
        self.env = self.vec_env.get_viz_env()  # For visualization compatibility
        
        # Rollout buffer (에피소드 기반 수집이라 size 제한 없음)
        self.buffer = RolloutBuffer(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device
        )
        
        # Tracking
        self.timesteps = 0
        self.episodes = 0
        self.updates = 0
        self.best_win_rate = 0.0
        
        # Logging
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "belief_loss": [],
            "entropy": [],  # Changed from entropy_loss
            "total_loss": []
        }

        # Training hooks (see src/hooks.py)
        self._hooks: List = []

        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def register_hook(self, hook) -> None:
        """Register a ``TrainingHook`` callback (see ``src.hooks``)."""
        self._hooks.append(hook)

    def _call_hooks_update(self, stats: Dict[str, Any]) -> None:
        for h in self._hooks:
            h.on_update_end(self, stats)

    def _call_hooks_eval(self, eval_stats: Dict[str, Any]) -> None:
        for h in self._hooks:
            h.on_eval_end(self, eval_stats)

    def _call_hooks_end(self) -> None:
        for h in self._hooks:
            h.on_training_end(self)

    def _batch_obs_to_tensor(
        self,
        obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert batched numpy obs dict to tensor dict."""
        return {
            key: torch.from_numpy(val).to(self.device)
            for key, val in obs.items()
        }
    
    def _batch_mask_to_tensor(
        self,
        mask: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert batched numpy mask dict to boolean tensor dict."""
        return {
            key: torch.from_numpy(val).bool().to(self.device)
            for key, val in mask.items()
        }
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout data using the current policy with vectorized envs.

        Single loop for both backends. Backend differences are only at episode
        close: SubprocVecEnv auto-resets inside workers and reports the winner +
        reset obs via the info dict; VectorDaVinciEnv (threaded) reads the winner
        from the env and resets it directly, and drives the live visualizer.

        Returns:
            Dictionary of rollout statistics
        """
        self.buffer.clear()
        obs, _ = self.vec_env.reset()
        backend = "multiprocess" if self._use_subproc else "threaded"
        logger.info(f"Starting rollout collection with {self.n_envs} parallel envs ({backend})")

        episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        completed_rewards: List[float] = []
        completed_lengths: List[float] = []
        per_env_tracking = [Episode(i, self.config.reward_config) for i in range(self.n_envs)]
        steps_collected = 0
        episodes_collected = 0
        target_episodes = self.config.episodes_per_update

        while episodes_collected < target_episodes:
            action_masks_np = self.vec_env.get_action_masks()
            obs_tensor = self._batch_obs_to_tensor(obs)
            action_mask_tensor = self._batch_mask_to_tensor(action_masks_np)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.config.fp16):
                actions, log_probs, values = self.policy.get_action(obs_tensor, action_mask_tensor)

            next_obs, rewards, terminated, truncated, next_infos, results = self.vec_env.step(actions)
            dones = terminated | truncated

            log_probs_np_all = {k: v.cpu().numpy() for k, v in log_probs.items()}
            values_np_all = values.cpu().numpy().flatten()

            for i in range(self.n_envs):
                self._add_transition_and_track(
                    i, obs, actions, rewards, dones, results, next_infos,
                    log_probs_np_all, values_np_all, action_masks_np,
                    per_env_tracking, episode_rewards, episode_lengths,
                )
                steps_collected += 1

                if dones[i]:
                    if self._use_subproc:
                        # Worker already auto-reset; winner + reset obs come via info.
                        winner = next_infos[i].get("_winner")
                        reset_obs = next_infos[i].get("_reset_obs")
                    else:
                        # Thread backend: read winner from env, reset it here.
                        winner = self.vec_env.envs[i]._winner
                        reset_obs, next_infos[i] = self.vec_env.reset_single(i)
                    self._close_episode(
                        i, obs, next_obs, per_env_tracking,
                        episode_rewards, episode_lengths,
                        completed_rewards, completed_lengths,
                        winner, reset_obs,
                    )
                    episodes_collected += 1

            # Live visualizer is only driven by the threaded backend.
            if not self._use_subproc:
                viz = get_visualizer()
                if viz is not None:
                    self._update_viz_from_vec(viz, rewards[0], episode_rewards[0], actions[0], results[0])

            obs = next_obs

        return self._finalize_rollouts(obs, completed_rewards, completed_lengths, steps_collected)

    # -------------------------------------------------------------------------
    # Rollout helpers
    # -------------------------------------------------------------------------

    def _add_transition_and_track(
        self,
        i: int,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        results: List,
        next_infos: List[Dict],
        log_probs_np_all: Dict[str, np.ndarray],
        values_np_all: np.ndarray,
        action_masks_np: Dict[str, np.ndarray],
        per_env_tracking: List[Dict],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
    ) -> int:
        """Build a Transition, add it to the buffer, and update per-step tracking.

        Returns the buffer index of the added transition.
        """
        def _safe_float(x: float) -> float:
            return float(x) if not (np.isnan(x) or np.isinf(x)) else 0.0

        obs_copy = {k: np.array(v[i], copy=True) for k, v in obs.items()}
        mask_copy = {k: np.array(v[i], copy=True) for k, v in action_masks_np.items()}
        hv = next_infos[i].get("hidden_values")
        transition = Transition(
            obs=obs_copy,
            player_id=int(results[i].player_id) if hasattr(results[i], "player_id") else 0,
            action=np.array(actions[i], copy=True),
            reward=_safe_float(rewards[i]),
            done=bool(dones[i]),
            log_probs={k: _safe_float(v[i]) for k, v in log_probs_np_all.items()},
            value=_safe_float(values_np_all[i]),
            action_mask=mask_copy,
            env_id=i,
            hidden_values=np.array(hv, copy=True) if hv is not None else None,
        )
        self.buffer.add(transition)
        buffer_idx = self.buffer.size - 1
        self.timesteps += 1

        # Record the move into this env's Episode. All outcome-based (retroactive)
        # rewards — draw win/lose, continue success/fail, loser penalty — are now
        # computed in one place at game end (Episode.finalize), not inline here.
        per_env_tracking[i].record(transition, int(np.argmax(obs["phase"][i])), results[i])

        episode_rewards[i] += rewards[i]
        episode_lengths[i] += 1
        return buffer_idx

    def _close_episode(
        self,
        i: int,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        per_env_tracking: List[Dict],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        completed_rewards: List[float],
        completed_lengths: List[float],
        winner: Optional[int],
        reset_obs: Optional[Dict[str, np.ndarray]],
    ) -> None:
        """Apply end-of-episode rewards, flush tracking state, and reset counters.

        *reset_obs* is applied to next_obs[i] when provided (threaded passes the
        result of reset_single(); subproc passes the obs from info['_reset_obs']).
        """
        # End-of-game (retroactive) rewards are computed in one place by the
        # Episode object. Monotone reward mode skips all win/lose shaping.
        if not self.config.monotone_reward:
            per_env_tracking[i].finalize(winner)
        # Reset this env's collector for the next game.
        per_env_tracking[i] = Episode(i, self.config.reward_config)

        completed_rewards.append(episode_rewards[i])
        completed_lengths.append(episode_lengths[i])
        self.episodes += 1
        if reset_obs is not None:
            for k in obs:
                next_obs[k][i] = reset_obs[k]
        episode_rewards[i] = 0.0
        episode_lengths[i] = 0

    def _finalize_rollouts(
        self,
        obs: Dict[str, np.ndarray],
        completed_rewards: List[float],
        completed_lengths: List[int],
        steps_collected: int
    ) -> Dict[str, float]:
        """Compute GAE, normalize advantages, log stats. Shared by both collect methods."""
        obs_tensor = self._batch_obs_to_tensor(obs)
        with torch.no_grad():
            _, _, last_values = self.policy.get_action(obs_tensor)
        
        last_values_np = last_values.cpu().numpy().flatten()
        last_values_per_env = {i: float(last_values_np[i]) for i in range(self.n_envs)}
        
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=0.0,
            last_values_per_env=last_values_per_env
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self._returns = returns
        self._advantages = advantages
        
        self.episode_rewards.extend(completed_rewards)
        self.episode_lengths.extend(completed_lengths)
        
        logger.info(f"Collected {len(completed_rewards)} episodes ({steps_collected} steps)")
        
        return {
            "mean_reward": np.mean(completed_rewards) if completed_rewards else 0.0,
            "mean_length": np.mean(completed_lengths) if completed_lengths else 0.0,
            "n_episodes": len(completed_rewards)
        }
    
    def _update_viz_from_vec(
        self,
        viz: DaVinciVisualizer,
        reward: float,
        episode_reward: float,
        action: np.ndarray,
        result: Result
    ) -> None:
        """Update visualization from vectorized env."""
        env = self.vec_env.get_viz_env()
        render_obs, render_info = env.render_info()
        phase_idx = int(np.argmax(render_obs["phase"]))
        phase_name = Phase(phase_idx).name
        viz.update_game_state(
            my_hand=render_obs["my_hand"],
            opponent_hand=render_obs["opponent_hand"],
            phase=phase_name,
            current_player=env._current_player,
            streak=env._streak,
            deck_black=env._deck.black_count,
            deck_white=env._deck.white_count,
            episode=self.episodes,
            timesteps=self.timesteps,
            reward=reward,
            total_reward=episode_reward,
            action=action,
            result=result
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollouts.
        
        Returns:
            Dictionary of loss values
        """
        policy_losses = []
        value_losses = []
        belief_losses = []
        entropy_losses = []
        
        # DDP: every rank must run the SAME number of minibatches per epoch,
        # otherwise the per-batch grad all_reduce counts diverge (ranks collect
        # different #steps → different ceil(n/batch_size)) → NCCL deadlock.
        # Agree on the per-epoch minibatch count (min across ranks).
        n_minibatches = (self.buffer.size + self.config.batch_size - 1) // self.config.batch_size
        if self.world_size > 1:
            t = torch.tensor([n_minibatches], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            n_minibatches = max(1, int(t.item()))

        for epoch in range(self.config.n_epochs):
            for bi, batch in enumerate(self.buffer.get_batches(
                self.config.batch_size,
                self._returns,
                self._advantages
            )):
                if bi >= n_minibatches:
                    break  # keep all ranks at the same #minibatches (DDP collective match)
                # Evaluate current policy on batch (heavy forward in fp16 under autocast).
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.config.fp16):
                    log_probs, values, entropies, belief_logits = self.policy.evaluate_actions(
                        batch["obs"],
                        batch["actions"],
                        batch["action_mask"]
                    )
                # Upcast outputs to fp32 so the (cheap) loss math is numerically stable.
                if self.config.fp16:
                    values = values.float()
                    belief_logits = belief_logits.float()
                    log_probs = {k: v.float() for k, v in log_probs.items()}
                    entropies = {k: v.float() for k, v in entropies.items()}

                # Compute policy loss for each action head
                policy_loss = torch.tensor(0.0, device=self.device)
                entropy_sum = torch.tensor(0.0, device=self.device)
                num_active_heads = 0
                
                phase = batch["obs"]["phase"]
                
                for key in ["color", "position", "value", "decision"]:
                    # Get phase mask for this head
                    if key == "color":
                        head_mask = phase[:, 0].bool()
                    elif key in ["position", "value"]:
                        head_mask = phase[:, 1].bool()
                    else:
                        head_mask = phase[:, 2].bool()
                    
                    if not head_mask.any():
                        continue
                    
                    num_active_heads += 1
                    
                    # Compute ratio
                    old_log_prob = batch["old_log_probs"][key][head_mask]
                    new_log_prob = log_probs[key][head_mask]
                    
                    # Clamp log prob difference for numerical stability
                    log_ratio = new_log_prob - old_log_prob
                    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    
                    # Clipped surrogate loss
                    adv = batch["advantages"][head_mask]
                    
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 
                        1 - self.config.clip_range, 
                        1 + self.config.clip_range
                    ) * adv
                    
                    policy_loss = policy_loss - torch.min(surr1, surr2).mean()
                    # Entropy bonus (positive entropy = good exploration)
                    # Color head gets separate (higher) entropy weight for draw diversity
                    if key == "color":
                        entropy_sum = entropy_sum + (self.config.color_ent_coef / self.config.ent_coef) * entropies[key][head_mask].mean()
                    else:
                        entropy_sum = entropy_sum + entropies[key][head_mask].mean()
                
                # Average entropy over active heads
                if num_active_heads > 0:
                    entropy_mean = entropy_sum / num_active_heads
                else:
                    entropy_mean = torch.tensor(0.0, device=self.device)
                
                # Value loss
                values = values.squeeze(-1)
                if self.config.clip_range_vf is not None:
                    # Clipped value loss - get old values from batch
                    old_values = batch["old_values"] if "old_values" in batch else values.detach()
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    vf_loss1 = (values - batch["returns"]) ** 2
                    vf_loss2 = (values_clipped - batch["returns"]) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - batch["returns"]) ** 2).mean()
                
                # Total loss: policy_loss - entropy_bonus + value_loss + belief_loss
                # We want to MAXIMIZE entropy, so we SUBTRACT it from loss (below).

                # Auxiliary belief loss: predict opponent's hidden card values
                # Impossible-value masking: values that are structurally impossible for a given
                # opponent slot (same-color cards already in my hand, or already revealed by opp)
                # are masked to -inf before CE so the model is penalised for assigning them prob.
                belief_loss = torch.tensor(0.0, device=self.device)
                if batch["hidden_values"] is not None:
                    hidden_vals = batch["hidden_values"]  # (B, 13) with -1 for non-hidden
                    # Only compute loss for positions that are actually hidden (value >= 0)
                    hidden_mask = hidden_vals >= 0  # (B, 13)
                    if hidden_mask.any():
                        B = hidden_vals.shape[0]
                        # Build impossible-value mask from observation
                        obs_my  = batch["obs"]["my_hand"].long()         # (B, 13, 2)
                        obs_opp = batch["obs"]["opponent_hand"].long()   # (B, 13, 2)
                        my_c = obs_my[:, :, 0]   # (B, 13) color
                        my_v = obs_my[:, :, 1]   # (B, 13) value; -2=empty slot
                        my_valid = (my_v >= 0) & (my_v < 13)
                        opp_c = obs_opp[:, :, 0]  # (B, 13) color always visible
                        opp_v = obs_opp[:, :, 1]  # (B, 13) -1=hidden
                        opp_rev_valid = (opp_v >= 0) & (opp_v < 13)
                        # color-value bitmaps: [b, color(0/1), value(0-12)]
                        my_bmp  = torch.zeros(B, 2, 13, dtype=torch.bool, device=self.device)
                        opp_bmp = torch.zeros(B, 2, 13, dtype=torch.bool, device=self.device)
                        b_range = torch.arange(B, device=self.device)
                        for slot in range(13):
                            vld = my_valid[:, slot]
                            vb = b_range[vld]
                            if vb.numel() > 0:
                                my_bmp[vb,
                                       my_c[:, slot][vld].clamp(0, 1),
                                       my_v[:, slot][vld].clamp(0, 12)] = True
                            vld = opp_rev_valid[:, slot]
                            vb = b_range[vld]
                            if vb.numel() > 0:
                                opp_bmp[vb,
                                        opp_c[:, slot][vld].clamp(0, 1),
                                        opp_v[:, slot][vld].clamp(0, 12)] = True
                        # For each opp slot pos: impossible[b,pos,v] = bitmap[b, opp_c[b,pos], v]
                        # gather along dim=1 (color dim) of (B,2,13) → (B,13,13)
                        color_idx = opp_c.clamp(0, 1).unsqueeze(2).expand(B, 13, 13)
                        impossible = (
                            my_bmp.gather(1, color_idx) |
                            opp_bmp.gather(1, color_idx)
                        )  # (B, 13, 13)
                        # Flatten to active hidden positions
                        flat_logits  = belief_logits[hidden_mask]  # (N, 13)
                        flat_targets = hidden_vals[hidden_mask]     # (N,)
                        flat_imp     = impossible[hidden_mask]      # (N, 13)
                        # Safety: never mask the true label itself
                        flat_imp.scatter_(1, flat_targets.clamp(0, 12).unsqueeze(1), False)
                        # Apply: impossible values → very negative before softmax.
                        # MASK_VALUE (finite -1e4) instead of -inf so this stays NaN-safe
                        # under fp16 autocast (softmax(-1e4) ~ 0, same effect as -inf).
                        flat_logits = flat_logits.masked_fill(flat_imp, MASK_VALUE)
                        belief_loss = nn.functional.cross_entropy(flat_logits, flat_targets)
                
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.belief_coef * belief_loss
                    - self.config.ent_coef * entropy_mean  # Subtract to maximize entropy
                )
                
                # NaN/Inf: do NOT `continue` (that would skip this rank's grad
                # all_reduce → DDP deadlock). Proceed — GradScaler.step auto-skips
                # the optimizer step when grads are inf/nan, and all_reduce stays matched.
                if self.is_main and (torch.isnan(loss) or torch.isinf(loss)):
                    logger.warning("Loss is NaN/Inf this batch — GradScaler will skip the step")

                # Backprop (GradScaler is a no-op when fp16 is disabled).
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # DDP: average gradients across ranks (manual — evaluate_actions is
                # a custom method, so the DDP wrapper's forward hooks don't apply).
                # Zero-fill missing grads so EVERY rank all-reduces the SAME param
                # set in the SAME order: phase-gated heads may get no grad in a given
                # minibatch on one rank but not another → otherwise NCCL shape/seq
                # mismatch → deadlock. None grad ⇒ this rank contributes zeros.
                if self.world_size > 1:
                    for p in self.policy.parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(self.world_size)
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                belief_losses.append(belief_loss.item())
                entropy_losses.append(entropy_mean.item())  # Store actual entropy
        
        self.updates += 1
        
        # Track losses
        mean_losses = {
            "policy_loss": np.mean(policy_losses) if policy_losses else 0.0,
            "value_loss": np.mean(value_losses) if value_losses else 0.0,
            "belief_loss": np.mean(belief_losses) if belief_losses else 0.0,
            "entropy": np.mean(entropy_losses) if entropy_losses else 0.0,
            "total_loss": (np.mean(policy_losses) if policy_losses else 0.0) + 
                         self.config.vf_coef * (np.mean(value_losses) if value_losses else 0.0) +
                         self.config.belief_coef * (np.mean(belief_losses) if belief_losses else 0.0) -
                         self.config.ent_coef * (np.mean(entropy_losses) if entropy_losses else 0.0)
        }
        
        for key, value in mean_losses.items():
            self.losses[key].append(value)
        
        return mean_losses
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()

        # Self-play with the current policy via the shared episode loop.
        from src.agent import ModelAgent
        from src.runner import run_episode
        agent = ModelAgent(self.policy, self.device)

        wins = [0, 0]
        total_rewards = [0.0, 0.0]
        episode_lengths = []

        for ep in range(n_episodes):
            res = run_episode(self.env, agent, deterministic=True)
            if res.winner is not None:
                wins[res.winner] += 1
            total_rewards[0] += res.rewards[0]
            total_rewards[1] += res.rewards[1]
            episode_lengths.append(res.length)

        self.policy.train()

        return {
            "player0_win_rate": wins[0] / n_episodes,
            "player1_win_rate": wins[1] / n_episodes,
            "mean_reward_p0": total_rewards[0] / n_episodes,
            "mean_reward_p1": total_rewards[1] / n_episodes,
            "mean_episode_length": np.mean(episode_lengths)
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            path: Save path (auto-generate if None)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            path = os.path.join(
                self.config.save_dir,
                f"checkpoint_{self.timesteps}.pt"
            )
        
        # Store GLOBAL counts so a checkpoint resumes consistently regardless of
        # the world_size it was trained / is resumed with.
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "timesteps": self.world_size * self.timesteps,
            "episodes": self.world_size * self.episodes,
            "updates": self.updates,
            "config": self.config.to_dict()
        }, path)
        
        return path
    
    def load(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Filter out keys whose shape changed (e.g., belief_head after architecture update)
        # strict=False only handles missing/unexpected keys, not shape mismatches
        saved_state = checkpoint["policy_state_dict"]
        model_state = self.policy.state_dict()
        compatible = {k: v for k, v in saved_state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        shape_mismatch = [k for k, v in saved_state.items()
                          if k in model_state and v.shape != model_state[k].shape]
        if shape_mismatch:
            print(f"Shape mismatch — re-initialized: {shape_mismatch}")

        missing, unexpected = self.policy.load_state_dict(compatible, strict=False)
        if missing:
            print(f"New parameters initialized randomly: {missing}")
        
        if self.config.reset_optimizer_on_load:
            # Reset optimizer with fresh state (useful after finetune→normal transition)
            self.optimizer = optim.Adam(
                [p for p in self.policy.parameters() if p.requires_grad],
                lr=self.config.learning_rate
            )
            print("Optimizer state reset (fresh Adam)")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                print(f"⚠️ Optimizer load failed ({e}), keeping current optimizer state")
        
        # Checkpoints store GLOBAL counts → back to per-rank local.
        self.timesteps = checkpoint["timesteps"] // self.world_size
        self.episodes = checkpoint["episodes"] // self.world_size
        self.updates = checkpoint["updates"]
        
        # Always apply config LR (checkpoint may have old LR)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.learning_rate
        # Re-create scheduler from current point
        avg_steps_per_episode = 45  # 실측 평균 ~45 steps/ep
        total_updates = self.config.total_timesteps // (self.config.episodes_per_update * avg_steps_per_episode)
        remaining = max(total_updates - self.updates, 1)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.config.lr_end / self.config.learning_rate,
            total_iters=remaining
        )

    def _global_timesteps(self) -> int:
        """True global timestep count, identical on every rank.

        Single-GPU: just the local count. DDP: collective SUM across ranks
        (must be called by ALL ranks each iteration — it's a collective op).
        """
        if self.world_size <= 1:
            return self.timesteps
        t = torch.tensor([self.timesteps], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def train(self) -> None:
        """
        Main training loop.
        
        Runs until total_timesteps is reached, alternating between
        collecting rollouts and performing PPO updates.
        """
        if self.is_main:
            print(f"Starting training on {self.device} (world_size={self.world_size})")
            print(f"Config: {self.config}")
            print("-" * 60)

        update_count = 0
        gts = 0  # global timesteps, agreed across ranks (set after each update)

        # Loop driven by `update_count` so EVERY rank does the same number of
        # collect+update iterations (the per-update grad all_reduce stays matched —
        # no deadlock from ranks disagreeing on a per-rank timestep condition).
        # `gts` (collective SUM of all ranks' local timesteps) is the true global
        # step count, identical on all ranks, used to decide when to stop.
        while True:
            # Collect rollouts (each rank on its own envs) + synced PPO update.
            rollout_stats = self.collect_rollouts()
            update_stats = self.update()
            update_count += 1
            gts = self._global_timesteps()   # collective; identical on all ranks

            # Step LR scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # --- rank-0-only: logging, hooks, eval, checkpoints ---
            if self.is_main and update_count % self.config.log_interval == 0:
                print(f"Update {update_count} | "
                      f"Timesteps: {gts:,} | "
                      f"Episodes: {self.world_size * self.episodes} | "
                      f"Mean Reward: {rollout_stats['mean_reward']:5.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Belief Loss: {update_stats['belief_loss']:.4f}")
                viz = get_visualizer()
                if viz is not None:
                    viz.update_training_stats(
                        mean_reward=rollout_stats['mean_reward'],
                        policy_loss=update_stats['policy_loss'],
                        value_loss=update_stats['value_loss']
                    )
                    viz.add_log(f"Update {update_count}: R={rollout_stats['mean_reward']:.2f}")

            if self.is_main:
                self._call_hooks_update({
                    "update_count":        update_count,
                    "timesteps":           gts,
                    "episodes":            self.world_size * self.episodes,
                    "mean_reward":         rollout_stats["mean_reward"],
                    "mean_episode_length": rollout_stats.get("mean_length", 0.0),
                    "policy_loss":         update_stats["policy_loss"],
                    "value_loss":          update_stats["value_loss"],
                    "belief_loss":         update_stats["belief_loss"],
                    "entropy":             update_stats["entropy"],
                    "total_loss":          update_stats["total_loss"],
                    "learning_rate":       current_lr,
                })

            if self.is_main and update_count % self.config.eval_interval == 0:
                eval_stats = self.evaluate(self.config.n_eval_episodes)
                print(f"\n[Eval] P0 Win Rate: {eval_stats['player0_win_rate']:.2%} | "
                      f"P1 Win Rate: {eval_stats['player1_win_rate']:.2%} | "
                      f"Mean Length: {eval_stats['mean_episode_length']:.1f}\n")
                if eval_stats['player0_win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_stats['player0_win_rate']
                    self.save(os.path.join(self.config.save_dir, "best_model.pt"))
                self._call_hooks_eval(eval_stats)

            if self.is_main and update_count % self.config.save_interval == 0:
                self.save(os.path.join(self.config.save_dir, "latest.pt"))

            # Collective stop decision (gts identical on all ranks) → no rank
            # leaves the loop early while others wait on the next all_reduce.
            if gts >= self.config.total_timesteps:
                break

        # Final: barrier so all ranks finish before rank-0 writes.
        if self.world_size > 1:
            dist.barrier()
        if self.is_main:
            self.save(os.path.join(self.config.save_dir, "latest.pt"))
            print(f"\nTraining complete! Model saved to {self.config.save_dir}/latest.pt")
            self._call_hooks_end()
            log_path = os.path.join(
                self.config.log_dir,
                f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(log_path, 'w') as f:
                json.dump({
                    "config": self.config.to_dict(),
                    "final_timesteps": gts,
                    "final_episodes": self.world_size * self.episodes,
                    "final_updates": self.updates,
                    "episode_rewards": self.episode_rewards[-1000:],
                    "losses": {k: v[-1000:] for k, v in self.losses.items()}
                }, f, indent=2, default=float)
            print(f"Training log saved to {log_path}")
