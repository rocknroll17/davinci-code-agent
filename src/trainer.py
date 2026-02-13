"""
PPO Trainer for Da Vinci Code Self-Play.

Implements Proximal Policy Optimization algorithm for training
the policy network through adversarial self-play.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
import json
from datetime import datetime

from src.model import DaVinciCodePolicy, obs_to_tensor, action_mask_to_tensor
from src.buffer import RolloutBuffer, Transition
from src.env import DaVinciCodeEnv
from src.vec_env import VectorDaVinciEnv, SubprocVecEnv
from src.constants import (
    CardValue, Phase, Color,
    REWARD_DRAW_WIN, REWARD_DRAW_LOSE,
    REWARD_CONTINUE_SUCCESS, REWARD_CONTINUE_FAIL
)
from src.result.result import Result
from src.result.guess_result import GuessResult
from src.result.streak_result import StreakResult
from src.visualizer import DaVinciVisualizer, get_visualizer

@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Training
    total_timesteps: int = 1000000000
    learning_rate: float = 0.5e-5  # Reduced LR for fine-grained optimization
    n_envs: int = 1000  # 많을수록 step당 수집량 ↑ → pipe 왕복 줄임
    episodes_per_update: int = 1000  # n_envs와 동일 → ~1라운드로 수집 완료
    batch_size: int = 2048  # GPU 효율적 배치
    n_epochs: int = 8  # More passes for thorough optimization
    
    # PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.07  # Tighter clip for precise updates
    clip_range_vf: float = 0.07  # Match policy clip range
    ent_coef: float = 0.002  # Minimal entropy → full exploitation
    color_ent_coef: float = 0.05  # Higher entropy for color head → diverse draw exploration
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    lr_end: float = 5e-6  # Lower floor for gradual decay
    
    # Model
    hidden_dim: int = 512
    
    # Logging
    log_interval: int = 1
    save_interval: int = 1
    eval_interval: int = 50
    n_eval_episodes: int = 50
    
    # Paths
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Finetune mode: 특수 케이스 학습 (조커 확정, 잘못된 예측 페널티 등)
    finetune: bool = False
    freeze_value_on_finetune: bool = False  # Value head learns from mixed normal+finetune data
    reset_optimizer_on_load: bool = False  # Reset optimizer state when loading (useful after finetune)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class PPOTrainer:
    """
    PPO Trainer for Da Vinci Code self-play.
    
    Implements the training loop where a single policy network
    plays against itself, learning from both perspectives.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration
            device: Target device (auto-detect if None)
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize policy network
        self.policy = DaVinciCodePolicy(config.hidden_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # LR Scheduler: linear decay from learning_rate to lr_end
        # 에피소드당 평균 ~60 스텝 추정
        avg_steps_per_episode = 60
        total_updates = config.total_timesteps // (config.episodes_per_update * avg_steps_per_episode)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=config.lr_end / config.learning_rate,
            total_iters=total_updates
        )
        
        # Finetune mode: freeze value head early so optimizer param groups match saved checkpoint
        if config.finetune and config.freeze_value_on_finetune:
            for param in self.policy.value_head.parameters():
                param.requires_grad = False
            trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
            ft_total_updates = config.total_timesteps // (config.episodes_per_update * avg_steps_per_episode)
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.lr_end / config.learning_rate,
                total_iters=ft_total_updates
            )
            print("Value head FROZEN for finetune (policy heads only)")
        
        # Vectorized Environment (multiple games in parallel)
        self.n_envs = config.n_envs
        self._use_subproc = config.n_envs > 1
        if self._use_subproc:
            self.vec_env = SubprocVecEnv(n_envs=config.n_envs)
        else:
            self.vec_env = VectorDaVinciEnv(n_envs=config.n_envs)
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
            "entropy": [],  # Changed from entropy_loss
            "total_loss": []
        }
        
        # FINETUNE MODE: 케이스별 통계
        self.finetune_stats = {
            'joker_by_value': {i: 0 for i in range(12)},  # 0-11 사이의 조커 케이스
            'impossible_guess_penalty': 0,  # 불가능한 예측 (내가 가진/상대 공개)
            'joker_correct': 0,         # 조커 정확히 맞춤
            'joker_intervention': 0,    # 조커 개입
            'joker_miss': 0,            # 조커 놓침
        }
        
        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def _save_obs_tensor(self, obs_tensor: Dict[str, torch.Tensor], player: int) -> None:
        """
        Save observation tensors to human-readable .txt for debugging.
        - 의미 없는 슬롯 제거 (-1, -2)
        - phase는 문자열로 표시
        """
        import numpy as np

        if not obs_tensor:
            return  # 비어있으면 아무것도 하지 않음

        # 파일 경로 및 이름
        path_dir = os.path.join(self.config.log_dir, "tensors")
        os.makedirs(path_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"tensor_p{player}_{ts}_{self.timesteps}"
        txt_path = os.path.join(path_dir, f"{base}.txt")

        # CPU 복사
        cpu_dict = {k: v.detach().cpu() for k, v in obs_tensor.items()}

        # 카드/값 문자열 변환 함수
        def card_to_str(card_arr, is_hidden: bool = False) -> str:
            color, value = card_arr
            if value == -1:
                if is_hidden:
                    return "???"
            elif value == -2:
                return ""
            val_str = "JOKER" if value == CardValue.JOKER else str(value)
            color_str = "W" if color == 1 else "B"
            
            return f"{color_str} {val_str}"
        
        def constraint_to_str(mat: np.ndarray) -> str:
            """
            -1만 있는 행은 제거
            """
            # 행 중 -1만 있는 것 제거
            valid_rows = ~np.all(mat == -1, axis=1)
            trimmed_mat = mat[valid_rows]

            if trimmed_mat.size == 0:
                return "<empty constraint>"

            return np.array2string(trimmed_mat, threshold=1000, max_line_width=200)
        
        # Phase 문자열 매핑
        PHASE_MAP = {0: "DRAW", 1: "GUESS", 2: "DECISION"}

        with open(txt_path, "w") as f:
            f.write(f"# Observation tensors for player {player} at {ts}, timestep={self.timesteps}\n")

            for key, tensor in cpu_dict.items():
                try:
                    arr = tensor.numpy()
                except Exception:
                    arr = "<unserializable tensor>"

                f.write(f"\n## {key} (shape={getattr(arr, 'shape', 'unknown')})\n")

                # 특별 처리: phase
                if key == "phase" and isinstance(arr, np.ndarray):
                    phase_idx = int(np.argmax(arr))
                    phase_str = PHASE_MAP.get(phase_idx, f"unknown({phase_idx})")
                    f.write(f"Phase: {phase_str}\n")
                    continue

                # 카드 텐서 처리
                if key in ("my_hand", "opponent_hand") and isinstance(arr, np.ndarray):
                    cards = [card_to_str(c, is_hidden=(key == "opponent_hand")) for c in arr[0] if card_to_str(c)]
                    f.write(f"{', '.join(cards)}\n")
                    continue

                # constraint_matrix는 필요한 부분만 요약
                if key == "constraint_matrix" and isinstance(arr, np.ndarray):
                    f.write(constraint_to_str(arr[0]) + "\n")
                    continue

                # 나머지 일반 텐서
                if isinstance(arr, np.ndarray):
                    f.write(np.array2string(arr, threshold=1000, max_line_width=200) + "\n")
                else:
                    f.write(str(arr) + "\n")

    def _save_finetune_stats_graph(self) -> None:
        """
        Finetune 통계를 그래프로 저장합니다.
        - 조커 케이스 숫자별 분포 (0-11)
        - 케이스별 총계
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 조커 케이스 숫자별 분포
        ax1 = axes[0]
        values = list(range(12))
        counts = [self.finetune_stats['joker_by_value'][v] for v in values]
        
        bars = ax1.bar(values, counts, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Surrounding Card Value', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Joker Cases by Surrounding Value', fontsize=14)
        ax1.set_xticks(values)
        ax1.set_xticklabels([str(v) for v in values])
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)
        
        # 2. 케이스별 총계
        ax2 = axes[1]
        case_names = ['Joker\nCorrect', 'Joker\nIntervention', 'Joker\nMiss', 
                      'Impossible\nGuess']
        case_counts = [
            self.finetune_stats['joker_correct'],
            self.finetune_stats['joker_intervention'],
            self.finetune_stats['joker_miss'],
            self.finetune_stats['impossible_guess_penalty']
        ]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars2 = ax2.bar(case_names, case_counts, color=colors, edgecolor='black')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Finetune Cases Summary', fontsize=14)
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars2, case_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.config.log_dir, 'finetune_stats.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Finetune stats graph saved to {save_path}")

    def _check_custom_training_cases(
        self,
        obs: Dict[str, np.ndarray],
        action_masks: Dict[str, np.ndarray],
        policy_actions: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Check for custom training cases and return overrides.
        
        커스텀 학습 케이스를 감지하고 action/reward 오버라이드를 반환합니다.
        
        케이스:
        1. 조커 위치 확정 시 → 확률적으로 맞추고 약간의 보너스
        2. 확정 상황에서 정책이 틀리면 → 페널티 부여
        3. 자기가 가진 카드를 상대방이 갖고 있다고 예측 → 페널티
        
        Args:
            obs: Batched observations (n_envs, ...)
            action_masks: Batched action masks
            policy_actions: Actions chosen by policy (n_envs, 4)
            
        Returns:
            Dictionary mapping env_idx -> {'action': ..., 'reward': ...}
            빈 딕셔너리면 오버라이드 없음
        """
        import random
        
        overrides = {}
        
        # 커스텀 학습 파라미터
        INTERVENTION_PROB = 0.7       # 70% 확률로 커스텀 action 사용
        CORRECT_BONUS = 0.3           # 확정 조커 맞추면 추가 보상
        MISS_PENALTY = -1.0           # 확정 상황에서 틀리면 페널티
        IMPOSSIBLE_GUESS_PENALTY = -2.0  # 불가능한 예측 (내가 가진/상대 공개한 카드)
        
        # 각 환경을 순회하며 커스텀 케이스 체크
        for env_idx in range(self.n_envs):
            env = self.vec_env.envs[env_idx]
            
            # 현재 Phase 확인 (GUESS phase = 1)
            phase = np.argmax(obs['phase'][env_idx])
            if phase != Phase.GUESS.value:
                continue
            
            current_player = env._current_player
            my_hand = env.players[current_player]._hand
            opponent_idx = 1 - current_player
            opponent_hand = env.players[opponent_idx]._hand
            
            # 정책이 선택한 action 확인
            policy_pos = int(policy_actions[env_idx][1])
            policy_val = int(policy_actions[env_idx][2])
            
            # 상대방 카드 리스트 (position으로 접근 가능하도록)
            opponent_cards = list(opponent_hand)
            
            # policy_pos가 유효한지 체크
            if policy_pos >= len(opponent_cards):
                continue
            
            # 예측하려는 상대방 카드의 색깔
            target_card = opponent_cards[policy_pos]
            target_color = target_card.color
            
            # ============================================================
            # 케이스 3+4: 불가능한 값 예측
            # - 내가 가진 (색+값) 예측 → 상대가 가질 수 없음
            # - 상대가 이미 공개한 (색+값) 예측 → 또 있을 수 없음
            # (조커는 2장 있으니까 제외)
            # ============================================================
            if policy_val != CardValue.JOKER:
                # 1) 내가 가진 같은 색+값
                my_same_color_values = {
                    card.value for card in my_hand 
                    if not card.is_joker and card.color == target_color
                }
                # 2) 상대가 공개한 같은 색+값
                opponent_revealed_same_color_values = {
                    card.value for card in opponent_hand 
                    if card.is_revealed and not card.is_joker and card.color == target_color
                }
                # 합집합: 불가능한 값들
                impossible_values = my_same_color_values | opponent_revealed_same_color_values
                
                if policy_val in impossible_values:
                    # 멍청한 예측! → 15% 확률로 샘플링
                    if random.random() > 0.15:
                        continue  # 85% 스킵
                    
                    if env_idx not in overrides:
                        overrides[env_idx] = {}
                    overrides[env_idx]['reward_bonus'] = overrides[env_idx].get('reward_bonus', 0) + IMPOSSIBLE_GUESS_PENALTY
                    self.finetune_stats['impossible_guess_penalty'] += 1
                    continue  # 다른 케이스 체크 스킵
            
            # ============================================================
            # 케이스 1, 2: 조커 위치 확정
            # ============================================================
            joker_positions, surrounding_value = opponent_hand.is_joker_between()
            
            if not joker_positions:
                continue
            
            # 조커 위치가 확정됨!
            # 정책이 이미 올바른 선택을 했는지 체크
            is_correct_guess = (
                policy_pos in joker_positions and 
                policy_val == CardValue.JOKER
            )
            
            if is_correct_guess:
                # 정책이 이미 맞춤 → 약간의 보너스만
                overrides[env_idx] = {
                    'reward_bonus': CORRECT_BONUS
                }
                self.finetune_stats['joker_correct'] += 1
                if surrounding_value is not None:
                    self.finetune_stats['joker_by_value'][surrounding_value] += 1
            else:
                # 정책이 틀림 → 확률적으로 개입
                if random.random() < INTERVENTION_PROB:
                    # 조커가 여러 개면 랜덤 선택
                    target_pos = random.choice(joker_positions)
                    
                    custom_action = np.array(
                        [0, target_pos, CardValue.JOKER, 0], 
                        dtype=np.int64
                    )
                    
                    overrides[env_idx] = {
                        'action': custom_action,
                        'reward_bonus': CORRECT_BONUS
                    }
                    self.finetune_stats['joker_intervention'] += 1
                    if surrounding_value is not None:
                        self.finetune_stats['joker_by_value'][surrounding_value] += 1
                else:
                    # 개입 안함 → 틀린 거에 대한 페널티
                    overrides[env_idx] = {
                        'reward_bonus': MISS_PENALTY
                    }
                    self.finetune_stats['joker_miss'] += 1
        
        return overrides

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
        Collect rollout data using current policy with vectorized environments.
        
        Supports both VectorDaVinciEnv (thread-based) and SubprocVecEnv (multiprocessing).
        SubprocVecEnv uses auto-reset and parallel finetune checking.
        
        Returns:
            Dictionary of rollout statistics
        """
        if self._use_subproc:
            return self._collect_rollouts_subproc()
        else:
            return self._collect_rollouts_threaded()
    
    def _collect_rollouts_threaded(self) -> Dict[str, float]:
        """Original thread-based rollout collection (VectorDaVinciEnv)."""
        self.buffer.clear()
        import logging
        logger = logging.getLogger()
        
        obs, infos = self.vec_env.reset()
        logger.info(f"Starting rollout collection with {self.n_envs} parallel envs (threaded)")
        
        episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        completed_rewards = []
        completed_lengths = []
        
        per_env_reward_tracking = [{
            'draw_indices': [],
            'last_continue_idx': None
        } for _ in range(self.n_envs)]
        
        steps_collected = 0
        episodes_collected = 0
        target_episodes = self.config.episodes_per_update
        
        finetune_pbar = None
        if self.config.finetune:
            from tqdm import tqdm
            finetune_pbar = tqdm(total=target_episodes, desc="🎯 Finetune episodes", unit="ep", ncols=80)
        
        while episodes_collected < target_episodes:
            action_masks_np = self.vec_env.get_action_masks()
            obs_tensor = self._batch_obs_to_tensor(obs)
            action_mask_tensor = self._batch_mask_to_tensor(action_masks_np)
            
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action(obs_tensor, action_mask_tensor)
            
            custom_overrides = {}
            if self.config.finetune:
                custom_overrides = self._check_custom_training_cases(obs, action_masks_np, actions)
                for env_idx, override in custom_overrides.items():
                    if 'action' in override:
                        actions[env_idx] = override['action']
            
            next_obs, rewards, terminated, truncated, next_infos, results = self.vec_env.step(actions)
            dones = terminated | truncated
            
            if self.config.finetune:
                for env_idx, override in custom_overrides.items():
                    if 'reward' in override:
                        rewards[env_idx] = override['reward']
                    elif 'reward_bonus' in override:
                        rewards[env_idx] += override['reward_bonus']
            
            log_probs_np_all = {k: v.cpu().numpy() for k, v in log_probs.items()}
            values_np_all = values.cpu().numpy().flatten()
            
            # Recompute log_probs for overridden actions (finetune action intervention)
            # Without this, PPO ratio = π_new(a_override) / π_old(a_original) which is wrong
            if self.config.finetune and custom_overrides:
                override_action_indices = [idx for idx, ov in custom_overrides.items() if 'action' in ov]
                if override_action_indices:
                    with torch.no_grad():
                        ov_idx = np.array(override_action_indices)
                        ov_obs = {k: obs_tensor[k][ov_idx] for k in obs_tensor.keys()}
                        ov_masks = {k: action_mask_tensor[k][ov_idx] for k in action_mask_tensor.keys()}
                        ov_actions_np = actions[ov_idx]
                        ov_actions = {
                            key: torch.from_numpy(ov_actions_np[:, k_idx].copy()).long().to(self.device)
                            for k_idx, key in enumerate(["color", "position", "value", "decision"])
                        }
                        new_log_probs, _, _ = self.policy.evaluate_actions(ov_obs, ov_actions, ov_masks)
                        for local_idx, global_idx in enumerate(override_action_indices):
                            for k in log_probs_np_all:
                                log_probs_np_all[k][global_idx] = new_log_probs[k][local_idx].cpu().item()
            
            for i in range(self.n_envs):
                obs_copy = {k: np.array(v[i], copy=True) for k, v in obs.items()}
                mask_copy = {k: np.array(v[i], copy=True) for k, v in action_masks_np.items()}
                action_copy = np.array(actions[i], copy=True)
                rew = float(rewards[i]) if not np.isnan(rewards[i]) and not np.isinf(rewards[i]) else 0.0
                done_flag = bool(dones[i])
                log_probs_np = {k: float(v[i]) if not np.isnan(v[i]) and not np.isinf(v[i]) else 0.0 for k, v in log_probs_np_all.items()}
                value_np = float(values_np_all[i]) if not np.isnan(values_np_all[i]) and not np.isinf(values_np_all[i]) else 0.0
                transition = Transition(
                    obs=obs_copy,
                    player_id=int(results[i].player_id) if hasattr(results[i], 'player_id') else 0,
                    action=action_copy,
                    reward=rew,
                    done=done_flag,
                    log_probs=log_probs_np,
                    value=value_np,
                    action_mask=mask_copy,
                    env_id=i
                )
                self.buffer.add(transition)
                buffer_idx = self.buffer.size - 1
                steps_collected += 1
                
                phase_idx = np.argmax(obs['phase'][i])
                if phase_idx == Phase.DRAW.value:
                    per_env_reward_tracking[i]['draw_indices'].append(buffer_idx)
                
                result_i = results[i]
                if result_i is not None and isinstance(result_i, StreakResult) and not result_i.is_invalid:
                    if result_i.is_continue:
                        per_env_reward_tracking[i]['last_continue_idx'] = buffer_idx
                
                if result_i is not None and isinstance(result_i, GuessResult) and not result_i.is_invalid:
                    cont_idx = per_env_reward_tracking[i]['last_continue_idx']
                    if cont_idx is not None:
                        if result_i.is_correct:
                            self.buffer.transitions[cont_idx].reward += REWARD_CONTINUE_SUCCESS
                        else:
                            self.buffer.transitions[cont_idx].reward += REWARD_CONTINUE_FAIL
                        per_env_reward_tracking[i]['last_continue_idx'] = None
                
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                self.timesteps += 1
                
                if dones[i]:
                    winner = self.vec_env.envs[i]._winner
                    if winner is not None:
                        for buf_idx in per_env_reward_tracking[i]['draw_indices']:
                            if buf_idx < len(self.buffer.transitions):
                                t = self.buffer.transitions[buf_idx]
                                if t.player_id == winner:
                                    t.reward += REWARD_DRAW_WIN
                                else:
                                    t.reward += REWARD_DRAW_LOSE
                    per_env_reward_tracking[i]['draw_indices'] = []
                    per_env_reward_tracking[i]['last_continue_idx'] = None
                    
                    completed_rewards.append(episode_rewards[i])
                    completed_lengths.append(episode_lengths[i])
                    self.episodes += 1
                    episodes_collected += 1
                    
                    if finetune_pbar is not None:
                        finetune_pbar.update(1)
                    
                    new_obs, new_info = self.vec_env.reset_single(i)
                    for k in obs.keys():
                        next_obs[k][i] = new_obs[k]
                    next_infos[i] = new_info
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
            viz = get_visualizer()
            if viz is not None:
                self._update_viz_from_vec(viz, rewards[0], episode_rewards[0], actions[0], results[0])
            
            obs = next_obs
            infos = next_infos
        
        if finetune_pbar is not None:
            finetune_pbar.close()
        
        return self._finalize_rollouts(obs, completed_rewards, completed_lengths, steps_collected)
    
    def _collect_rollouts_subproc(self) -> Dict[str, float]:
        """
        Multiprocessing-based rollout collection (SubprocVecEnv).
        
        Key differences from threaded version:
        - env.step() runs in parallel worker processes (true CPU parallelism)
        - Finetune case checking runs inside workers (no GIL bottleneck)
        - Auto-reset: done envs reset inside workers (no extra round-trip)
        - Winner info comes from info dict (not direct env access)
        """
        self.buffer.clear()
        import logging
        logger = logging.getLogger()
        
        obs, infos = self.vec_env.reset()
        logger.info(f"Starting rollout collection with {self.n_envs} parallel envs (multiprocess)")
        
        episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        completed_rewards = []
        completed_lengths = []
        
        per_env_reward_tracking = [{
            'draw_indices': [],
            'last_continue_idx': None
        } for _ in range(self.n_envs)]
        
        steps_collected = 0
        episodes_collected = 0
        target_episodes = self.config.episodes_per_update
        
        finetune_pbar = None
        if self.config.finetune:
            from tqdm import tqdm
            finetune_pbar = tqdm(total=target_episodes, desc="🎯 Finetune episodes", unit="ep", ncols=80)
        
        while episodes_collected < target_episodes:
            action_masks_np = self.vec_env.get_action_masks()
            obs_tensor = self._batch_obs_to_tensor(obs)
            action_mask_tensor = self._batch_mask_to_tensor(action_masks_np)
            
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action(obs_tensor, action_mask_tensor)
            
            # Step environments (finetune checks run inside workers)
            if self.config.finetune:
                next_obs, rewards, terminated, truncated, next_infos, results, custom_overrides = \
                    self.vec_env.step_finetune(actions)
                # Sync overridden actions back to trainer (critical: buffer must store actual executed action)
                for env_idx, override in custom_overrides.items():
                    if 'action' in override:
                        actions[env_idx] = override['action']
                # Update finetune stats from worker overrides
                for env_idx, override in custom_overrides.items():
                    case_type = override.get('case_type', '')
                    if case_type == 'impossible_guess':
                        self.finetune_stats['impossible_guess_penalty'] += 1
                    elif case_type == 'joker_correct':
                        self.finetune_stats['joker_correct'] += 1
                        sv = override.get('surrounding_value')
                        if sv is not None:
                            self.finetune_stats['joker_by_value'][sv] += 1
                    elif case_type == 'joker_intervention':
                        self.finetune_stats['joker_intervention'] += 1
                        sv = override.get('surrounding_value')
                        if sv is not None:
                            self.finetune_stats['joker_by_value'][sv] += 1
                    elif case_type == 'joker_miss':
                        self.finetune_stats['joker_miss'] += 1
            else:
                next_obs, rewards, terminated, truncated, next_infos, results = \
                    self.vec_env.step(actions)
                custom_overrides = {}
            
            dones = terminated | truncated
            
            log_probs_np_all = {k: v.cpu().numpy() for k, v in log_probs.items()}
            values_np_all = values.cpu().numpy().flatten()
            
            # Recompute log_probs for overridden actions (finetune action intervention)
            # Without this, PPO ratio = π_new(a_override) / π_old(a_original) which is wrong
            if self.config.finetune and custom_overrides:
                override_action_indices = [idx for idx, ov in custom_overrides.items() if 'action' in ov]
                if override_action_indices:
                    with torch.no_grad():
                        ov_idx = np.array(override_action_indices)
                        ov_obs = {k: obs_tensor[k][ov_idx] for k in obs_tensor.keys()}
                        ov_masks = {k: action_mask_tensor[k][ov_idx] for k in action_mask_tensor.keys()}
                        ov_actions_np = actions[ov_idx]
                        ov_actions = {
                            key: torch.from_numpy(ov_actions_np[:, k_idx].copy()).long().to(self.device)
                            for k_idx, key in enumerate(["color", "position", "value", "decision"])
                        }
                        new_log_probs, _, _ = self.policy.evaluate_actions(ov_obs, ov_actions, ov_masks)
                        for local_idx, global_idx in enumerate(override_action_indices):
                            for k in log_probs_np_all:
                                log_probs_np_all[k][global_idx] = new_log_probs[k][local_idx].cpu().item()
            
            for i in range(self.n_envs):
                obs_copy = {k: np.array(v[i], copy=True) for k, v in obs.items()}
                mask_copy = {k: np.array(v[i], copy=True) for k, v in action_masks_np.items()}
                action_copy = np.array(actions[i], copy=True)
                rew = float(rewards[i]) if not np.isnan(rewards[i]) and not np.isinf(rewards[i]) else 0.0
                done_flag = bool(dones[i])
                log_probs_np = {k: float(v[i]) if not np.isnan(v[i]) and not np.isinf(v[i]) else 0.0 for k, v in log_probs_np_all.items()}
                value_np = float(values_np_all[i]) if not np.isnan(values_np_all[i]) and not np.isinf(values_np_all[i]) else 0.0
                transition = Transition(
                    obs=obs_copy,
                    player_id=int(results[i].player_id) if hasattr(results[i], 'player_id') else 0,
                    action=action_copy,
                    reward=rew,
                    done=done_flag,
                    log_probs=log_probs_np,
                    value=value_np,
                    action_mask=mask_copy,
                    env_id=i
                )
                self.buffer.add(transition)
                buffer_idx = self.buffer.size - 1
                steps_collected += 1
                
                phase_idx = np.argmax(obs['phase'][i])
                if phase_idx == Phase.DRAW.value:
                    per_env_reward_tracking[i]['draw_indices'].append(buffer_idx)
                
                result_i = results[i]
                if result_i is not None and isinstance(result_i, StreakResult) and not result_i.is_invalid:
                    if result_i.is_continue:
                        per_env_reward_tracking[i]['last_continue_idx'] = buffer_idx
                
                if result_i is not None and isinstance(result_i, GuessResult) and not result_i.is_invalid:
                    cont_idx = per_env_reward_tracking[i]['last_continue_idx']
                    if cont_idx is not None:
                        if result_i.is_correct:
                            self.buffer.transitions[cont_idx].reward += REWARD_CONTINUE_SUCCESS
                        else:
                            self.buffer.transitions[cont_idx].reward += REWARD_CONTINUE_FAIL
                        per_env_reward_tracking[i]['last_continue_idx'] = None
                
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                self.timesteps += 1
                
                if dones[i]:
                    # Get winner from info dict (auto-reset, env is in worker process)
                    winner = next_infos[i].get('_winner')
                    if winner is not None:
                        for buf_idx in per_env_reward_tracking[i]['draw_indices']:
                            if buf_idx < len(self.buffer.transitions):
                                t = self.buffer.transitions[buf_idx]
                                if t.player_id == winner:
                                    t.reward += REWARD_DRAW_WIN
                                else:
                                    t.reward += REWARD_DRAW_LOSE
                    per_env_reward_tracking[i]['draw_indices'] = []
                    per_env_reward_tracking[i]['last_continue_idx'] = None
                    
                    completed_rewards.append(episode_rewards[i])
                    completed_lengths.append(episode_lengths[i])
                    self.episodes += 1
                    episodes_collected += 1
                    
                    if finetune_pbar is not None:
                        finetune_pbar.update(1)
                    
                    # Auto-reset: use reset obs from info dict
                    if '_reset_obs' in next_infos[i]:
                        reset_obs = next_infos[i]['_reset_obs']
                        for k in obs.keys():
                            next_obs[k][i] = reset_obs[k]
                    
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
            
            viz = get_visualizer()
            if viz is not None and not self._use_subproc:
                self._update_viz_from_vec(viz, rewards[0], episode_rewards[0], actions[0], results[0])
            
            obs = next_obs
            infos = next_infos
        
        if finetune_pbar is not None:
            finetune_pbar.close()
        
        return self._finalize_rollouts(obs, completed_rewards, completed_lengths, steps_collected)
    
    def _finalize_rollouts(
        self,
        obs: Dict[str, np.ndarray],
        completed_rewards: List[float],
        completed_lengths: List[int],
        steps_collected: int
    ) -> Dict[str, float]:
        """Compute GAE, normalize advantages, log stats. Shared by both collect methods."""
        import logging
        logger = logging.getLogger()
        
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
        
        if self.config.finetune:
            logger.info(f"[FINETUNE] Collected {len(completed_rewards)} episodes ({steps_collected} steps)")
        else:
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
    
    def update_render_stats(self, reward: float, episode_reward: float, action: np.ndarray, result: Result) -> None:
        """
        Update visualization with current game state.
        Args:
            render_obs: Render observation dictionary
            phase_name: Current phase name
            reward: Reward obtained in last action
            episode_reward: Total reward in current episode
            action: Last action taken
            result: Result object from last action
        """
        render_obs, render_info = self.env.render_info()
        # 시각화 업데이트
        viz = get_visualizer()
        if viz is not None:
            phase_idx = int(np.argmax(render_obs["phase"]))
            phase_name = Phase(phase_idx).name
            viz.update_game_state(
                my_hand=render_obs["my_hand"],
                opponent_hand=render_obs["opponent_hand"],
                phase=phase_name,
                current_player=self.env._current_player,
                streak=self.env._streak,
                deck_black=self.env._deck.black_count,
                deck_white=self.env._deck.white_count,
                episode=self.episodes,
                timesteps=self.timesteps,
                reward=reward,
                total_reward=episode_reward,
                action=action[0],
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
        entropy_losses = []
        
        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(
                self.config.batch_size,
                self._returns,
                self._advantages
            ):
                # Evaluate current policy on batch
                log_probs, values, entropies = self.policy.evaluate_actions(
                    batch["obs"],
                    batch["actions"],
                    batch["action_mask"]
                )
                
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
                
                # Total loss: policy_loss - entropy_bonus + value_loss
                # We want to MAXIMIZE entropy, so we SUBTRACT it from loss
                entropy_loss = -entropy_mean  # For logging (negative = good)
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_mean  # Subtract to maximize entropy
                )
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Loss is {loss.item()}, skipping batch")
                    continue
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_mean.item())  # Store actual entropy
        
        self.updates += 1
        
        # Track losses
        mean_losses = {
            "policy_loss": np.mean(policy_losses) if policy_losses else 0.0,
            "value_loss": np.mean(value_losses) if value_losses else 0.0,
            "entropy": np.mean(entropy_losses) if entropy_losses else 0.0,
            "total_loss": (np.mean(policy_losses) if policy_losses else 0.0) + 
                         self.config.vf_coef * (np.mean(value_losses) if value_losses else 0.0) -
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
        
        wins = [0, 0]
        total_rewards = [0.0, 0.0]
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_rewards = [0.0, 0.0]
            ep_length = 0
            
            while not done:
                current_player = info["current_player"]
                action_mask = self.env.get_action_mask()
                
                obs_tensor = obs_to_tensor(obs, self.device)
                action_mask_tensor = action_mask_to_tensor(action_mask, self.device)
                
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(
                        obs_tensor, action_mask_tensor, deterministic=True
                    )
                
                obs, _, reward, terminated, truncated, info, _ = self.env.step(action[0])
                done = terminated or truncated
                
                ep_rewards[current_player] += reward
                ep_length += 1
            
            # Track winner
            if info.get("winner") is not None:
                wins[info["winner"]] += 1
            
            for i in range(2):
                total_rewards[i] += ep_rewards[i]
            episode_lengths.append(ep_length)
        
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
        
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "timesteps": self.timesteps,
            "episodes": self.episodes,
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
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        
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
        
        self.timesteps = checkpoint["timesteps"]
        self.episodes = checkpoint["episodes"]
        self.updates = checkpoint["updates"]
        
        # Always apply config LR (checkpoint may have old LR)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.config.learning_rate
        # Re-create scheduler from current point
        avg_steps_per_episode = 60
        total_updates = self.config.total_timesteps // (self.config.episodes_per_update * avg_steps_per_episode)
        remaining = max(total_updates - self.updates, 1)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.config.lr_end / self.config.learning_rate,
            total_iters=remaining
        )
    
    def train(self) -> None:
        """
        Main training loop.
        
        Runs until total_timesteps is reached, alternating between
        collecting rollouts and performing PPO updates.
        """
        print(f"Starting training on {self.device}")
        print(f"Config: {self.config}")
        print("-" * 60)
        
        update_count = 0
        
        while self.timesteps < self.config.total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # Update policy
            update_stats = self.update()
            update_count += 1
            
            # Step LR scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            if update_count % self.config.log_interval == 0:
                print(f"Update {update_count} | "
                      f"Timesteps: {self.timesteps:,} | "
                      f"Episodes: {self.episodes} | "
                      f"Mean Reward: {rollout_stats['mean_reward']:5.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f}")
                
                # 시각화 학습 통계 업데이트
                viz = get_visualizer()
                if viz is not None:
                    viz.update_training_stats(
                        mean_reward=rollout_stats['mean_reward'],
                        policy_loss=update_stats['policy_loss'],
                        value_loss=update_stats['value_loss']
                    )
                    viz.add_log(f"Update {update_count}: R={rollout_stats['mean_reward']:.2f}")
                    print(f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                                f"Value Loss: {update_stats['value_loss']:.4f}")
            
            # Evaluation
            if update_count % self.config.eval_interval == 0:
                eval_stats = self.evaluate(self.config.n_eval_episodes)
                print(f"\n[Eval] P0 Win Rate: {eval_stats['player0_win_rate']:.2%} | "
                      f"P1 Win Rate: {eval_stats['player1_win_rate']:.2%} | "
                      f"Mean Length: {eval_stats['mean_episode_length']:.1f}\n")
                
                # Save best model
                if eval_stats['player0_win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_stats['player0_win_rate']
                    self.save(os.path.join(self.config.save_dir, "best_model.pt"))
            
            # Save checkpoint
            if update_count % self.config.save_interval == 0:
                self.save(os.path.join(self.config.save_dir, "latest.pt"))
                
                # Finetune 모드일 때 통계 그래프 저장
                if self.config.finetune:
                    self._save_finetune_stats_graph()
        
        # Final save
        self.save(os.path.join(self.config.save_dir, "latest.pt"))
        print(f"\nTraining complete! Model saved to {self.config.save_dir}/latest.pt")
        
        # Finetune 모드 최종 통계 저장
        if self.config.finetune:
            self._save_finetune_stats_graph()
            print("\n📊 Final Finetune Statistics:")
            print(f"  Joker Correct: {self.finetune_stats['joker_correct']}")
            print(f"  Joker Intervention: {self.finetune_stats['joker_intervention']}")
            print(f"  Joker Miss: {self.finetune_stats['joker_miss']}")
            print(f"  Impossible Guess Penalty: {self.finetune_stats['impossible_guess_penalty']}")
            print(f"  Joker by Value: {dict(self.finetune_stats['joker_by_value'])}")
        
        # Save training log
        log_path = os.path.join(
            self.config.log_dir,
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_path, 'w') as f:
            json.dump({
                "config": self.config.to_dict(),
                "final_timesteps": self.timesteps,
                "final_episodes": self.episodes,
                "final_updates": self.updates,
                "episode_rewards": self.episode_rewards[-1000:],  # Last 1000
                "losses": {k: v[-1000:] for k, v in self.losses.items()}
            }, f, indent=2)
        print(f"Training log saved to {log_path}")
