"""
Dreamer-style training loop for Da Vinci Code:

    real env (joker_control) ── collect ──▶ sequence replay
    replay ── (B,L) batches ──▶ world model (RSSM) training
    posterior states ── imagine H steps ──▶ actor-critic training
    actor ── acts in real env ──▶ collect ...

The actor-critic never sees a real transition during its update — it learns
entirely from rollouts imagined by the RSSM (DreamerV3, arXiv:2301.04104).

Self-play convention: identical to the PPO trainer — one policy, one
alternating-perspective stream per env. Rewards are credited to the acting
player and the stream is treated as a single-agent trajectory (the world model
learns the perspective flip as part of the dynamics).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model import obs_to_tensor, action_mask_to_tensor
from src.vec_env import VectorDaVinciEnv
from src.wm.nets import Actor, Critic, WMConfig, WorldModel
from src.wm.replay import EpisodeAccumulator, SequenceReplay


@dataclass
class DreamerConfig:
    # environment / collection
    n_envs: int = 16
    seed: Optional[int] = None
    # replay
    replay_capacity: int = 5000       # episodes
    seq_len: int = 32
    batch_size: int = 16
    prefill_episodes: int = 64        # random-actor episodes before training
    # world model
    wm: WMConfig = field(default_factory=WMConfig)
    wm_lr: float = 1e-4
    # actor-critic (imagination)
    horizon: int = 15
    gamma: float = 0.99               # episodes are ~45 steps (paper uses 0.997)
    lam: float = 0.95
    entropy_scale: float = 3e-4
    ac_lr: float = 3e-5
    critic_ema_decay: float = 0.98
    return_norm_decay: float = 0.99   # EMA on the 5-95 percentile scale S
    # schedule: per collection round
    episodes_per_round: int = 16
    wm_updates_per_round: int = 50
    ac_updates_per_round: int = 50
    max_grad_norm: float = 100.0
    # io
    save_dir: str = "checkpoints_wm"


class DreamerTrainer:
    def __init__(self, config: DreamerConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wm = WorldModel(config.wm).to(self.device)
        self.actor = Actor(config.wm).to(self.device)
        self.critic = Critic(config.wm).to(self.device)
        self.critic_ema = copy.deepcopy(self.critic)
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=config.wm_lr)
        self.ac_opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.ac_lr)

        self.replay = SequenceReplay(config.replay_capacity, config.seq_len, seed=config.seed)
        self.vec_env = VectorDaVinciEnv(n_envs=config.n_envs, seed=config.seed,
                                        joker_control=True)

        # persistent collection state (per env): recurrent state + accumulators
        self._h = self._z = None
        self._prev_action = None
        self._accs = [EpisodeAccumulator() for _ in range(config.n_envs)]
        self._obs = None

        self._return_scale = 1.0
        self.total_env_steps = 0
        self.total_episodes = 0
        os.makedirs(config.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Collection (real environment)
    # ------------------------------------------------------------------

    def _reset_collection_state(self) -> None:
        n = self.cfg.n_envs
        self._h, self._z = self.wm.rssm.initial(n, self.device)
        self._prev_action = torch.zeros(n, 5, dtype=torch.long, device=self.device)
        self._obs, _ = self.vec_env.reset()
        self._accs = [EpisodeAccumulator() for _ in range(n)]

    @torch.no_grad()
    def collect(self, n_episodes: int, random_actor: bool = False) -> Dict[str, float]:
        """Run the actor in the real vectorized env until n_episodes finish."""
        if self._obs is None:
            self._reset_collection_state()

        done_episodes = 0
        ep_rewards = []
        while done_episodes < n_episodes:
            obs_t = {k: torch.as_tensor(v).to(self.device) for k, v in self._obs.items()}
            masks_np = self.vec_env.get_action_masks()

            # posterior state from real observation
            embed = self.wm.encoder(obs_t)
            self._h = self.wm.rssm.step_deter(self._h, self._z, self._prev_action)
            _, self._z = self.wm.rssm.posterior(self._h, embed)
            state = torch.cat([self._h, self._z], dim=-1)

            phase_idx = obs_t["phase"].float().argmax(-1)
            if random_actor:
                action = self._random_masked_action(masks_np)
            else:
                masks_t = {k: torch.as_tensor(v).to(self.device) for k, v in masks_np.items()}
                action, _, _ = self.actor.sample(state, masks_t, phase_idx)
            action_np = action.cpu().numpy()

            next_obs, rewards, terminated, truncated, infos, _ = self.vec_env.step(action_np)
            dones = terminated | truncated
            self.total_env_steps += self.cfg.n_envs

            for i in range(self.cfg.n_envs):
                self._accs[i].add(
                    {k: self._obs[k][i] for k in self._obs},
                    action_np[i], rewards[i], bool(dones[i]),
                    {k: masks_np[k][i] for k in masks_np},
                )
                if dones[i]:
                    ep = self._accs[i]
                    ep_rewards.append(float(sum(ep.rewards)))
                    self.replay.add_episode(ep.pack())
                    self._accs[i] = EpisodeAccumulator()
                    done_episodes += 1
                    self.total_episodes += 1
                    # reset env + recurrent state for this slot
                    reset_obs, _ = self.vec_env.reset_single(i)
                    for k in next_obs:
                        next_obs[k][i] = reset_obs[k]
                    self._h[i] = 0.0
                    self._z[i] = 0.0
                    action_np[i] = 0

            self._prev_action = torch.as_tensor(action_np, device=self.device)
            self._obs = next_obs

        return {"collect/mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "collect/episodes": done_episodes}

    def _random_masked_action(self, masks_np) -> torch.Tensor:
        n = self.cfg.n_envs
        out = np.zeros((n, 5), dtype=np.int64)
        rng = np.random
        for i in range(n):
            out[i, 0] = rng.choice(np.flatnonzero(masks_np["color"][i])) if masks_np["color"][i].any() else 0
            pos_valid = np.flatnonzero(masks_np["position"][i])
            out[i, 1] = rng.choice(pos_valid) if len(pos_valid) else 0
            row = masks_np["value"][i][out[i, 1]]
            out[i, 2] = rng.choice(np.flatnonzero(row)) if row.any() else 0
            out[i, 3] = rng.randint(0, 2)
            jk = np.flatnonzero(masks_np["joker"][i])
            out[i, 4] = rng.choice(jk) if len(jk) else 0
        return torch.as_tensor(out, device=self.device)

    # ------------------------------------------------------------------
    # World model update
    # ------------------------------------------------------------------

    def train_world_model(self) -> Dict[str, float]:
        batch = self.replay.sample(self.cfg.batch_size, self.device)
        loss, states, metrics = self.wm.loss(
            batch["obs"], batch["actions"], batch["rewards"], batch["continues"],
            batch["is_first"], mask_seq=batch["masks"], valid=batch["valid"],
        )
        self.wm_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.max_grad_norm)
        self.wm_opt.step()
        self._last_states = (states, batch["valid"])   # reused as imagination starts
        return metrics

    # ------------------------------------------------------------------
    # Actor-critic update (pure imagination)
    # ------------------------------------------------------------------

    def train_actor_critic(self) -> Dict[str, float]:
        cfg = self.cfg
        if not hasattr(self, "_last_states"):
            self.train_world_model()
        states, valid = self._last_states
        start = states.flatten(0, 1)[valid.flatten() > 0]          # (N, S)
        if start.shape[0] == 0:
            return {}

        with torch.no_grad():
            img = self.wm.imagine(self.actor, start, cfg.horizon)
            s_all = img["states"]                                   # (H+1, N, S)
            rewards, conts = img["rewards"], img["continues"]

            v_ema = self.wm.twohot.decode(self.critic_ema(s_all))   # (H+1, N)

            # λ-returns (bootstrapped with the EMA critic)
            H = cfg.horizon
            returns = torch.zeros_like(v_ema)
            returns[H] = v_ema[H]
            for t in reversed(range(H)):
                boot = (1 - cfg.lam) * v_ema[t + 1] + cfg.lam * returns[t + 1]
                returns[t] = rewards[t + 1] + cfg.gamma * conts[t + 1] * boot

            # trajectory weights: stop crediting after predicted episode end
            w = torch.cumprod(
                torch.cat([torch.ones_like(conts[:1]), conts[:-1] * cfg.gamma], 0), 0)

            # return normalization S = Per(R,95) − Per(R,5), EMA-smoothed
            scale = torch.quantile(returns[:-1].flatten(), 0.95) - \
                torch.quantile(returns[:-1].flatten(), 0.05)
            self._return_scale = (cfg.return_norm_decay * self._return_scale
                                  + (1 - cfg.return_norm_decay) * float(scale))
            advantage = (returns[:-1] - v_ema[:-1]) / max(1.0, self._return_scale)

            # actor inputs recomputed with gradients below
            phase_idx = self.wm.decode_phase(s_all[:-1].flatten(0, 1))
            masks = self.wm.predict_masks(s_all[:-1].flatten(0, 1))

        # --- actor loss (REINFORCE with normalized advantage + entropy) ---
        flat_states = s_all[:-1].flatten(0, 1).detach()
        flat_actions = img["actions"].flatten(0, 1)
        log_prob, entropy = self.actor.evaluate(flat_states, flat_actions, masks, phase_idx)
        log_prob = log_prob.view(cfg.horizon, -1)
        entropy = entropy.view(cfg.horizon, -1)
        actor_loss = -(w[:-1] * (advantage.detach() * log_prob
                                 + cfg.entropy_scale * entropy)).mean()

        # --- critic loss (two-hot CE toward sg(λ-returns)) ---
        critic_logits = self.critic(flat_states)
        critic_loss = (w[:-1].flatten()
                       * self.wm.twohot.loss(critic_logits, returns[:-1].flatten().detach())
                       ).mean()

        loss = actor_loss + critic_loss
        self.ac_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            cfg.max_grad_norm)
        self.ac_opt.step()

        # EMA critic update
        with torch.no_grad():
            for p, p_ema in zip(self.critic.parameters(), self.critic_ema.parameters()):
                p_ema.lerp_(p, 1 - cfg.critic_ema_decay)

        return {
            "ac/actor_loss": float(actor_loss.detach()),
            "ac/critic_loss": float(critic_loss.detach()),
            "ac/return_mean": float(returns.mean()),
            "ac/return_scale": float(self._return_scale),
            "ac/entropy": float(entropy.mean().detach()),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, rounds: int, log_every: int = 1) -> None:
        # prefill with masked-random play so the world model sees diverse states
        if self.replay.n_episodes < self.cfg.prefill_episodes:
            print(f"[dreamer] prefilling replay with "
                  f"{self.cfg.prefill_episodes - self.replay.n_episodes} random episodes...")
            self.collect(self.cfg.prefill_episodes - self.replay.n_episodes,
                         random_actor=True)

        for r in range(1, rounds + 1):
            cstats = self.collect(self.cfg.episodes_per_round)
            wm_m = {}
            for _ in range(self.cfg.wm_updates_per_round):
                wm_m = self.train_world_model()
            ac_m = {}
            for _ in range(self.cfg.ac_updates_per_round):
                ac_m = self.train_actor_critic()

            if r % log_every == 0:
                print(f"[dreamer] round {r} | steps {self.total_env_steps:,} "
                      f"| eps {self.total_episodes} "
                      f"| R {cstats['collect/mean_ep_reward']:.2f} "
                      f"| wm {wm_m.get('wm/loss', 0):.2f} "
                      f"(obs {wm_m.get('wm/obs', 0):.2f}, kl {wm_m.get('wm/kl_dyn', 0):.2f}) "
                      f"| actor {ac_m.get('ac/actor_loss', 0):.4f} "
                      f"| critic {ac_m.get('ac/critic_loss', 0):.2f} "
                      f"| ent {ac_m.get('ac/entropy', 0):.2f}")
            self.save(os.path.join(self.cfg.save_dir, "dreamer_latest.pt"))

    # ------------------------------------------------------------------
    # Persistence / inference
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ema": self.critic_ema.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "ac_opt": self.ac_opt.state_dict(),
            "total_env_steps": self.total_env_steps,
            "total_episodes": self.total_episodes,
            "config": self.cfg.__dict__ | {"wm": self.cfg.wm.__dict__},
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_ema.load_state_dict(ckpt["critic_ema"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.ac_opt.load_state_dict(ckpt["ac_opt"])
        self.total_env_steps = ckpt.get("total_env_steps", 0)
        self.total_episodes = ckpt.get("total_episodes", 0)


class WMAgent:
    """Inference wrapper: acts in a real env through the world model's posterior.

    Keeps the recurrent state across steps of ONE game — call reset() between
    games. Compatible with src.runner.run_episode's Agent interface.
    """

    def __init__(self, wm: WorldModel, actor: Actor,
                 device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")
        self.wm = wm.to(self.device).eval()
        self.actor = actor.to(self.device).eval()
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "WMAgent":
        device = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        wm_cfg = WMConfig(**{k: v for k, v in ckpt["config"]["wm"].items()})
        wm = WorldModel(wm_cfg)
        wm.load_state_dict(ckpt["wm"])
        actor = Actor(wm_cfg)
        actor.load_state_dict(ckpt["actor"])
        return cls(wm, actor, device)

    def reset(self) -> None:
        self._h, self._z = self.wm.rssm.initial(1, self.device)
        self._prev_action = torch.zeros(1, 5, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def act(self, obs, action_mask=None, deterministic: bool = False):
        obs_t = obs_to_tensor(obs, self.device)
        embed = self.wm.encoder(obs_t)
        self._h = self.wm.rssm.step_deter(self._h, self._z, self._prev_action)
        _, self._z = self.wm.rssm.posterior(self._h, embed)
        state = torch.cat([self._h, self._z], dim=-1)

        phase_idx = obs_t["phase"].float().argmax(-1)
        masks_t = action_mask_to_tensor(action_mask, self.device) if action_mask else None
        if deterministic:
            dists = self.actor.dist(state, masks_t)
            comps = [dists[k].probs.argmax(-1)
                     for k in ("color", "position", "value", "decision", "joker")]
            action = torch.stack(comps, dim=-1)
        else:
            action, _, _ = self.actor.sample(state, masks_t, phase_idx)
        self._prev_action = action
        return action.cpu().numpy()[0], {}
