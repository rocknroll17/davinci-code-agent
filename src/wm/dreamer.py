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
import torch.distributed as dist
import torch.nn.functional as F

from src.model import obs_to_tensor, action_mask_to_tensor
from src.vec_env import SubprocVecEnv as VectorDaVinciEnv
from src.wm.nets import Actor, Critic, WMConfig, WorldModel
from src.wm.replay import EpisodeAccumulator, SequenceReplay


@dataclass
class DreamerConfig:
    # environment / collection
    n_envs: int = 16
    n_workers: Optional[int] = None
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
    imag_batch_size: Optional[int] = 1024
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
    # evaluation vs the deployed legacy model (rank 0 only; 0 → disabled)
    eval_every: int = 20              # rounds between head-to-head evals
    eval_games: int = 200             # ±3.5%p standard error at 50%
    eval_opponent: str = "checkpoints/best_model_control.pt"
    # io
    save_dir: str = "checkpoints_wm"


class DreamerTrainer:
    """Single-GPU by default; multi-GPU data-parallel when constructed with
    rank/world_size under an initialized torch.distributed process group
    (manual gradient all-reduce, same convention as PPOTrainer)."""

    def __init__(self, config: DreamerConfig, device: Optional[torch.device] = None,
                 rank: int = 0, world_size: int = 1) -> None:
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        if world_size > 1 and not dist.is_initialized():
            raise RuntimeError("world_size>1 requires torch.distributed to be initialized")

        self.wm = WorldModel(config.wm).to(self.device)
        self.actor = Actor(config.wm).to(self.device)
        self.critic = Critic(config.wm).to(self.device)

        # DDP: every rank must start from IDENTICAL weights (grad averaging only
        # keeps replicas in sync if they begin in sync) → broadcast rank 0's.
        if world_size > 1:
            for module in (self.wm, self.actor, self.critic):
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)

        self.critic_ema = copy.deepcopy(self.critic)
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=config.wm_lr)
        self.ac_opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.ac_lr)

        # Per-rank seed offset so DDP ranks collect DIFFERENT games.
        rank_seed = None if config.seed is None else config.seed + rank * 100000
        self.replay = SequenceReplay(config.replay_capacity, config.seq_len, seed=rank_seed)
        self.vec_env = VectorDaVinciEnv(n_envs=config.n_envs, seed=rank_seed,
                                        joker_control=True, n_workers=config.n_workers)

        # persistent collection state (per env): recurrent state + accumulators
        self._h = self._z = None
        self._prev_action = None
        self._accs = [EpisodeAccumulator() for _ in range(config.n_envs)]
        self._obs = None
        self._last_player = [None] * config.n_envs   # for perspective-flip flags

        self._return_scale = 1.0
        self.total_env_steps = 0
        self.total_episodes = 0
        self.best_win_rate = 0.0
        self._legacy_opponent = None   # lazily loaded on first eval
        if self.is_main:
            os.makedirs(config.save_dir, exist_ok=True)

    def _all_reduce_grads(self, params) -> None:
        """Average gradients across ranks. Zero-fill missing grads so every
        rank reduces the same tensor set in the same order (no deadlock)."""
        if self.world_size <= 1:
            return
        for p in params:
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)

    # ------------------------------------------------------------------
    # Collection (real environment)
    # ------------------------------------------------------------------

    def _reset_collection_state(self) -> None:
        n = self.cfg.n_envs
        self._h, self._z = self.wm.rssm.initial(n, self.device)
        self._prev_action = torch.zeros(n, 5, dtype=torch.long, device=self.device)
        self._obs, _ = self.vec_env.reset()
        self._accs = [EpisodeAccumulator() for _ in range(n)]
        self._last_player = [None] * n

    @torch.no_grad()
    def collect(self, n_episodes: int, random_actor: bool = False) -> Dict[str, float]:
        """Run the actor in the real vectorized env until n_episodes finish."""
        if self._obs is None:
            self._reset_collection_state()

        done_episodes = 0
        ep_rewards = []
        guess_hits = guess_total = 0
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

            next_obs, rewards, terminated, truncated, infos, results = self.vec_env.step(action_np)
            dones = terminated | truncated
            self.total_env_steps += self.cfg.n_envs

            phase_np = self._obs["phase"].argmax(-1)
            for i in range(self.cfg.n_envs):
                # actor skill proxy: accuracy of GUESS-phase actions (visible
                # long before the win rate moves off 0%)
                if phase_np[i] == 1:
                    guess_total += 1
                    if rewards[i] > 0:
                        guess_hits += 1
                # perspective flip = the acting player changed vs the previous step
                pid = int(results[i].player_id) if (
                    results[i] is not None and hasattr(results[i], "player_id")) else 0
                flip = 0.0 if self._last_player[i] is None else float(pid != self._last_player[i])
                self._last_player[i] = pid

                self._accs[i].add(
                    {k: self._obs[k][i] for k in self._obs},
                    action_np[i], rewards[i], bool(dones[i]),
                    {k: masks_np[k][i] for k in masks_np},
                    flip=flip,
                    hidden_values=infos[i].get("hidden_values"),
                )
                if dones[i]:
                    ep = self._accs[i]
                    # terminal observation BEFORE the reset obs overwrites it —
                    # the reward head learns the win/lose reward from this frame
                    ep.set_terminal({k: np.array(next_obs[k][i], copy=True) for k in next_obs})
                    ep_rewards.append(float(sum(ep.rewards)))
                    self.replay.add_episode(ep.pack())
                    self._accs[i] = EpisodeAccumulator()
                    self._last_player[i] = None
                    done_episodes += 1
                    self.total_episodes += 1
                    # reset env + recurrent state for this slot
                    if infos[i] and "_reset_obs" in infos[i]:
                        reset_obs = infos[i]["_reset_obs"]
                    else:
                        reset_obs, _ = self.vec_env.reset_single(i)
                    for k in next_obs:
                        next_obs[k][i] = reset_obs[k]
                    self._h[i] = 0.0
                    self._z[i] = 0.0
                    action_np[i] = 0

            self._prev_action = torch.as_tensor(action_np, device=self.device)
            self._obs = next_obs

        return {"collect/mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "collect/guess_acc": guess_hits / max(1, guess_total),
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
            flip_seq=batch["flips"], hidden_seq=batch["hidden_values"],
        )
        self.wm_opt.zero_grad()
        loss.backward()
        self._all_reduce_grads(self.wm.parameters())
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
        if cfg.imag_batch_size is not None and start.shape[0] > cfg.imag_batch_size:
            idx = torch.randperm(start.shape[0], device=states.device)[:cfg.imag_batch_size]
            start = start[idx]
        if start.shape[0] == 0:
            # practically unreachable (every sampled episode has ≥2 real steps);
            # DDP-safe fallback: still run backward so grad all-reduce stays matched
            start = torch.zeros(1, states.shape[-1], device=states.device)

        with torch.no_grad():
            img = self.wm.imagine(self.actor, start, cfg.horizon)
            s_all = img["states"]                                   # (H+1, N, S)
            rewards, conts = img["rewards"], img["continues"]

            v_ema = self.wm.twohot.decode(self.critic_ema(s_all))   # (H+1, N)

            # Zero-sum (negamax) λ-returns. Every state is "current player's
            # perspective"; the critic values it for the player to act there.
            # When the perspective flips between s_t and s_{t+1}, the future
            # value/return is the OPPONENT's and must be negated — otherwise
            # the actor maximizes both players' rewards summed together.
            # rewards[t+1] is r(a_t), already from s_t's actor's perspective.
            # sign is soft: E[±1] = 1 − 2·P(flip).
            sign = 1.0 - 2.0 * img["flips"]                         # (H+1, N)
            H = cfg.horizon
            returns = torch.zeros_like(v_ema)
            returns[H] = v_ema[H]
            for t in reversed(range(H)):
                boot = (1 - cfg.lam) * v_ema[t + 1] + cfg.lam * returns[t + 1]
                returns[t] = rewards[t + 1] + cfg.gamma * conts[t + 1] * sign[t + 1] * boot

            # trajectory weights: stop crediting after predicted episode end
            w = torch.cumprod(
                torch.cat([torch.ones_like(conts[:1]), conts[:-1] * cfg.gamma], 0), 0)

            # return normalization S = Per(R,95) − Per(R,5), EMA-smoothed
            scale = torch.quantile(returns[:-1].flatten(), 0.95) - \
                torch.quantile(returns[:-1].flatten(), 0.05)
            if self.world_size > 1:
                # keep the scale identical on every rank
                dist.all_reduce(scale, op=dist.ReduceOp.SUM)
                scale = scale / self.world_size
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
        self._all_reduce_grads(
            list(self.actor.parameters()) + list(self.critic.parameters()))
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
            "ac/adv_std": float(advantage.std()),
            "ac/entropy": float(entropy.mean().detach()),
        }

    # ------------------------------------------------------------------
    # Evaluation vs deployed legacy model
    # ------------------------------------------------------------------

    def evaluate(self, n_games: Optional[int] = None, seed0: int = 0) -> Dict[str, float]:
        """Head-to-head vs the deployed legacy checkpoint (win rate from the
        world model's perspective, alternating seats, per-game seeds)."""
        from src.wm.eval_arena import LegacyOpponentAdapter, evaluate_vs_legacy

        if self._legacy_opponent is None:
            if not os.path.exists(self.cfg.eval_opponent):
                if self.is_main:
                    print(f"[dreamer] eval opponent not found: {self.cfg.eval_opponent} "
                          f"— skipping eval")
                return {}
            from src.agent import ModelAgent
            inner = ModelAgent.from_checkpoint(self.cfg.eval_opponent, device=self.device)
            self._legacy_opponent = LegacyOpponentAdapter(inner)

        wm_agent = WMAgent(self.wm, self.actor, self.device)
        stats = evaluate_vs_legacy(
            wm_agent, self._legacy_opponent,
            n_games=n_games or self.cfg.eval_games, seed0=seed0)
        # WMAgent shares the live modules — restore train mode
        self.wm.train()
        self.actor.train()
        return stats

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, rounds: int, log_every: int = 1) -> None:
        # prefill with masked-random play so the world model sees diverse states
        # (every rank prefills its own replay — DDP ranks hold disjoint data)
        if self.replay.n_episodes < self.cfg.prefill_episodes:
            if self.is_main:
                print(f"[dreamer] prefilling replay with "
                      f"{self.cfg.prefill_episodes - self.replay.n_episodes} "
                      f"random episodes/rank...")
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

            if self.is_main and r % log_every == 0:
                ws = self.world_size
                print(f"[dreamer] round {r} | steps {ws * self.total_env_steps:,} "
                      f"| eps {ws * self.total_episodes} "
                      f"| R {cstats['collect/mean_ep_reward']:.2f} "
                      f"| acc {cstats.get('collect/guess_acc', 0):.1%} "
                      f"| wm {wm_m.get('wm/loss', 0):.2f} "
                      f"(obs {wm_m.get('wm/obs', 0):.2f}, "
                      f"rew {wm_m.get('wm/reward', 0):.2f}, "
                      f"bel {wm_m.get('wm/belief_acc', 0):.1%}, "
                      f"flip {wm_m.get('wm/flip', 0):.3f}, "
                      f"kl {wm_m.get('wm/kl_dyn', 0):.2f}) "
                      f"| actor {ac_m.get('ac/actor_loss', 0):.4f} "
                      f"| critic {ac_m.get('ac/critic_loss', 0):.2f} "
                      f"| ret {ac_m.get('ac/return_mean', 0):.2f}"
                      f"/adv {ac_m.get('ac/adv_std', 0):.3f} "
                      f"| ent {ac_m.get('ac/entropy', 0):.2f}")
            if self.is_main:
                self.save(os.path.join(self.cfg.save_dir, "dreamer_latest.pt"))

            # periodic head-to-head vs the deployed model (rank 0 only; other
            # ranks proceed and simply wait at the next grad all-reduce)
            if (self.is_main and self.cfg.eval_every > 0
                    and r % self.cfg.eval_every == 0):
                estats = self.evaluate(seed0=r * 10000)
                if estats:
                    print(f"[eval] vs {os.path.basename(self.cfg.eval_opponent)}: "
                          f"win {estats['eval/win_rate']:.1%} "
                          f"(P0 {estats['eval/win_rate_p0']:.1%} / "
                          f"P1 {estats['eval/win_rate_p1']:.1%}, "
                          f"{int(estats['eval/n_games'])} games)")
                    if estats["eval/win_rate"] > self.best_win_rate:
                        self.best_win_rate = estats["eval/win_rate"]
                        self.save(os.path.join(self.cfg.save_dir, "dreamer_best.pt"))
                        print(f"[eval] new best ({self.best_win_rate:.1%}) → dreamer_best.pt")

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
            "total_env_steps": self.world_size * self.total_env_steps,
            "total_episodes": self.world_size * self.total_episodes,
            "best_win_rate": self.best_win_rate,
            "config": self.cfg.__dict__ | {"wm": self.cfg.wm.__dict__},
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # strict=False: checkpoints predating a head (e.g. flip_head) still load;
        # the missing head starts fresh while everything else warm-starts.
        missing, unexpected = self.wm.load_state_dict(ckpt["wm"], strict=False)
        if missing and self.is_main:
            print(f"[dreamer] new WM parameters initialized fresh: {missing}")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_ema.load_state_dict(ckpt["critic_ema"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.ac_opt.load_state_dict(ckpt["ac_opt"])
        self.total_env_steps = ckpt.get("total_env_steps", 0) // max(1, self.world_size)
        self.total_episodes = ckpt.get("total_episodes", 0) // max(1, self.world_size)
        self.best_win_rate = ckpt.get("best_win_rate", 0.0)


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
