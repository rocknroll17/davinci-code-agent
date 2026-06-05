#!/usr/bin/env python3
"""
Arena — same-data multi-variant ablation harness.

ONE self-play rollout is generated per update (by variant 0's policy on CPU). Every variant
then trains on that IDENTICAL data with its own config (loss weights / feature flags / arch),
each on its own GPU. After N updates the variants are pitted head-to-head (round-robin) and
ranked by win rate.

Why same data: it removes the data-variance + extra-training confounds that make separate
runs hard to compare. Differences come purely from the config.

Caveat (honest): variants 1..N train OFF variant-0's data (off-policy). Fine for short,
warm-started comparisons (policies stay close); the head-to-head at the end is ground truth.
mean_reward is identical across variants (same data), so it does NOT rank them — only h2h does.

Edit CONFIGS below to define your variants. Each is a dict of PPOConfig-field overrides
(+ `separate_my_slot_pe` or any model flag you add behind a config field).

Env note (this box): export LD_LIBRARY_PATH=/home/temp/py310/lib
Run:  PYTHONPATH=$PWD python scripts/arena.py --checkpoint checkpoints/best_model_control.pt --updates 100
"""
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.logger  # noqa: F401  (configure logging)
from src.trainer import PPOTrainer, PPOConfig
from src.model import DaVinciCodePolicy, obs_to_tensor, action_mask_to_tensor
from src.env import DaVinciCodeEnv
from src.constants import MAX_HAND_SIZE, NUM_VALUES, MASK_VALUE

# ─────────────────────────────────────────────────────────────────────────────
# DEFINE YOUR VARIANTS HERE.  variant 0 is the data generator (trained on-policy).
# Each entry: name + dict of overrides applied on top of the base config.
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS = [
    {"name": "baseline",     "overrides": {}},
    {"name": "no_belief",    "overrides": {"belief_coef": 0.0}},
    {"name": "high_entropy", "overrides": {"ent_coef": 0.03}},
    {"name": "low_clip",     "overrides": {"clip_range": 0.1}},
    # To compare a NEW feature/loss: add a field to PPOConfig (and use it in the model/loss),
    # then add a variant here toggling it, e.g. {"name": "rope", "overrides": {"use_rope": True}}.
]

ACTION_KEYS = ["color", "position", "value", "decision"]


def make_config(base: PPOConfig, overrides: dict) -> PPOConfig:
    # deepcopy keeps reward_config as a RewardConfig object — base.to_dict() would flatten it
    # to a plain dict, which then breaks `env._rc.guess_fail` (attribute access) in workers.
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def load_into(model: nn.Module, ckpt_path: str):
    """Warm-start a variant model from a checkpoint (copy slot_pos_embed -> my_slot if separate)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, ms = raw["policy_state_dict"], model.state_dict()
    comp = {k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape}
    missing, _ = model.load_state_dict(comp, strict=False)
    enc = model.encoder
    if getattr(enc, "separate_my_slot_pe", False) and any("my_slot_pos_embed" in k for k in missing):
        with torch.no_grad():
            enc.my_slot_pos_embed.weight.copy_(enc.slot_pos_embed.weight)


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant PPO update on a SHARED rollout buffer (faithful to PPOTrainer.update).
# ─────────────────────────────────────────────────────────────────────────────
def build_shared_tensors(buffer, device):
    """Materialize the shared rollout's transitions as tensors on `device` (once per variant)."""
    n = buffer.size
    obs_keys = buffer.transitions[0].obs.keys()
    obs = {k: torch.from_numpy(np.stack([buffer.transitions[i].obs[k] for i in range(n)])).float().to(device)
           for k in obs_keys}
    actions = {key: torch.from_numpy(np.array([buffer.transitions[i].action[ki] for i in range(n)])).long().to(device)
               for ki, key in enumerate(ACTION_KEYS)}
    masks = None
    if buffer.transitions[0].action_mask is not None:
        masks = {k: torch.from_numpy(np.stack([buffer.transitions[i].action_mask[k] for i in range(n)])).bool().to(device)
                 for k in buffer.transitions[0].action_mask.keys()}
    hv = None
    if buffer.transitions[0].hidden_values is not None:
        hv = torch.from_numpy(np.stack([t.hidden_values for t in buffer.transitions])).long().to(device)
    return obs, actions, masks, hv


def variant_update(model, optimizer, cfg, shared, returns_np, adv_np, device):
    """One PPO update for a variant on the shared data. Returns mean losses."""
    obs, actions, masks, hidden_values = shared
    n = returns_np.shape[0]
    returns = torch.from_numpy(returns_np).float().to(device)
    advantages = torch.from_numpy(adv_np).float().to(device)

    # Per-variant old log-probs / old values (this variant is the "behavior" for its own ratio).
    with torch.no_grad():
        old_lp, old_v, _, _ = model.evaluate_actions(obs, actions, masks)
        old_lp = {k: v.detach() for k, v in old_lp.items()}
        old_v = old_v.squeeze(-1).detach()

    pol_losses, val_losses, bel_losses, ent_losses = [], [], [], []
    bs = cfg.batch_size
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            b_obs = {k: v[idx] for k, v in obs.items()}
            b_act = {k: v[idx] for k, v in actions.items()}
            b_mask = {k: v[idx] for k, v in masks.items()} if masks is not None else None
            log_probs, values, entropies, belief_logits = model.evaluate_actions(b_obs, b_act, b_mask)
            values = values.squeeze(-1)
            phase = b_obs["phase"]

            policy_loss = torch.zeros((), device=device)
            entropy_sum = torch.zeros((), device=device)
            n_active = 0
            for key in ACTION_KEYS:
                head_mask = phase[:, 0].bool() if key == "color" else \
                            phase[:, 1].bool() if key in ("position", "value") else phase[:, 2].bool()
                if not head_mask.any():
                    continue
                n_active += 1
                old = old_lp[key][idx][head_mask]
                new = log_probs[key][head_mask]
                ratio = torch.exp(torch.clamp(new - old, -20.0, 20.0))
                adv = advantages[idx][head_mask]
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv
                policy_loss = policy_loss - torch.min(s1, s2).mean()
                w = (cfg.color_ent_coef / cfg.ent_coef) if (key == "color" and cfg.ent_coef) else 1.0
                entropy_sum = entropy_sum + w * entropies[key][head_mask].mean()
            entropy_mean = entropy_sum / n_active if n_active else torch.zeros((), device=device)

            ov = old_v[idx]
            v_clip = ov + torch.clamp(values - ov, -cfg.clip_range_vf, cfg.clip_range_vf)
            value_loss = 0.5 * torch.max((values - returns[idx]) ** 2, (v_clip - returns[idx]) ** 2).mean()

            belief_loss = torch.zeros((), device=device)
            if hidden_values is not None and cfg.belief_coef > 0:
                hv = hidden_values[idx]
                hmask = hv >= 0
                if hmask.any():
                    fl = belief_logits[hmask]
                    ft = hv[hmask].clamp(0, NUM_VALUES - 1)
                    belief_loss = F.cross_entropy(fl, ft)

            loss = (policy_loss + cfg.vf_coef * value_loss
                    + cfg.belief_coef * belief_loss - cfg.ent_coef * entropy_mean)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            pol_losses.append(policy_loss.item()); val_losses.append(value_loss.item())
            bel_losses.append(float(belief_loss.detach())); ent_losses.append(entropy_mean.item())
    return {"policy": np.mean(pol_losses or [0]), "value": np.mean(val_losses or [0]),
            "belief": np.mean(bel_losses or [0]), "entropy": np.mean(ent_losses or [0])}


# ─────────────────────────────────────────────────────────────────────────────
# Round-robin head-to-head (GPU-batched, randomized starting side).
# ─────────────────────────────────────────────────────────────────────────────
def _cur_players(infos, n):
    return np.fromiter((infos[i]["current_player"] for i in range(n)), dtype=np.int64, count=n)


def _act_split(model_a, model_b, ot, mt, useA, n, device):
    """Run A only on the envs where A is to move and B only on B's — halves forward work."""
    actions = np.zeros((n, 4), dtype=np.int64)
    useA_t = torch.from_numpy(useA).to(device)
    for model, sel in ((model_a, useA_t), (model_b, ~useA_t)):
        if not bool(sel.any()):
            continue
        idx = sel.nonzero(as_tuple=True)[0]
        sub_o = {k: v[idx] for k, v in ot.items()}
        sub_m = {k: v[idx] for k, v in mt.items()}
        with torch.no_grad():
            a, _, _ = model.get_action(sub_o, sub_m, deterministic=True)
        actions[idx.cpu().numpy()] = a
    return actions


def h2h(model_a, model_b, device, episodes=1000, n_envs=400, n_workers=None):
    """GPU-batched head-to-head on a MULTIPROCESS vec-env (true parallel CPU stepping — env
    step/mask was ~70% of the old VectorDaVinciEnv version's time). Forward is split so each
    model only runs on the envs where it is to move."""
    from src.vec_env import SubprocVecEnv
    env = SubprocVecEnv(n_envs=n_envs, n_workers=n_workers or os.cpu_count())
    n = env.n_envs
    rng = np.random.default_rng(0)
    obs, infos = env.reset()
    cur = _cur_players(infos, n)
    a_seat0 = rng.random(n) < 0.5
    a_wins = b = done = 0
    bt = lambda o: {k: torch.from_numpy(v).to(device) for k, v in o.items()}
    bm = lambda m: {k: torch.from_numpy(v).bool().to(device) for k, v in m.items()}
    while done < episodes:
        masks = env.get_action_masks()
        ot, mt = bt(obs), bm(masks)
        useA = (a_seat0 & (cur == 0)) | (~a_seat0 & (cur == 1))
        acts = _act_split(model_a, model_b, ot, mt, useA, n, device)
        next_obs, _, term, trunc, infos, _ = env.step(acts)   # auto-reset inside workers
        cur = _cur_players(infos, n)
        for i in np.nonzero(term | trunc)[0]:
            w = infos[i].get("_winner")
            if w is not None:
                a_won = (w == 0) if a_seat0[i] else (w == 1)
                a_wins += int(a_won); b += int(not a_won); done += 1
            ro = infos[i]["_reset_obs"]
            for k in next_obs:
                next_obs[k][i] = ro[k]
            cur[i] = infos[i]["_reset_info"]["current_player"]
            a_seat0[i] = rng.random() < 0.5
        obs = next_obs
    return a_wins, b, done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="warm-start all variants from here (recommended)")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=600, help="shared rollout size per update")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--h2h-episodes", type=int, default=1000)
    p.add_argument("--save-dir", default="checkpoints_arena")
    args = p.parse_args()

    ngpu = torch.cuda.device_count()
    devs = [torch.device(f"cuda:{i % ngpu}") if ngpu else torch.device("cpu") for i in range(len(CONFIGS))]
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"{len(CONFIGS)} variants over {ngpu or 'CPU'} GPU(s): "
          + ", ".join(f"{c['name']}->{d}" for c, d in zip(CONFIGS, devs)))

    base = PPOConfig(learning_rate=args.lr, lr_end=args.lr, reset_optimizer_on_load=True)
    if args.workers is not None:
        base.n_workers = args.workers
    base.n_envs = args.n_envs
    base.episodes_per_update = args.n_envs

    # variant 0 = data generator (full PPOTrainer: collects + trains on-policy)
    cfg0 = make_config(base, CONFIGS[0]["overrides"])
    gen = PPOTrainer(cfg0, devs[0])
    if args.checkpoint:
        load_into(gen.policy, args.checkpoint)   # own loader (weights_only=False); avoids main's broken trainer.load
    # variants 1..N: model + optimizer on their own GPU
    variants = []
    for c, d in zip(CONFIGS[1:], devs[1:]):
        cfg = make_config(base, c["overrides"])
        m = DaVinciCodePolicy(cfg.hidden_dim).to(d)
        if args.checkpoint:
            load_into(m, args.checkpoint)
        opt = torch.optim.Adam(m.parameters(), lr=cfg.learning_rate)
        variants.append({"name": c["name"], "model": m, "opt": opt, "cfg": cfg, "dev": d,
                         "shared": None})

    for u in range(1, args.updates + 1):
        gen.collect_rollouts()                       # ONE shared rollout (variant 0's policy)
        gen.update()                                 # variant 0 trains on-policy
        for v in variants:                           # everyone else trains on the SAME data
            shared = build_shared_tensors(gen.buffer, v["dev"])
            variant_update(v["model"], v["opt"], v["cfg"], shared,
                           gen._returns, gen._advantages, v["dev"])
        if u % 10 == 0 or u == args.updates:
            print(f"update {u}/{args.updates}  (shared rollout: {gen.buffer.size} transitions)")

    # save all
    models = {CONFIGS[0]["name"]: gen.policy}
    torch.save({"policy_state_dict": gen.policy.state_dict()}, f"{args.save_dir}/{CONFIGS[0]['name']}.pt")
    for v in variants:
        models[v["name"]] = v["model"]
        torch.save({"policy_state_dict": v["model"].state_dict()}, f"{args.save_dir}/{v['name']}.pt")

    # round-robin head-to-head on one device
    print("\n=== round-robin head-to-head (win rate of row vs column) ===")
    hd = devs[0]
    for m in models.values():
        m.to(hd).eval()
    names = list(models)
    wins = {n: 0.0 for n in names}
    games = {n: 0 for n in names}
    line = "          " + "".join(f"{n[:9]:>10}" for n in names)
    print(line)
    for a in names:
        row = f"{a[:9]:>10}"
        for b in names:
            if a == b:
                row += f"{'—':>10}"; continue
            aw, bw, tot = h2h(models[a], models[b], hd, episodes=args.h2h_episodes,
                              n_envs=500, n_workers=args.workers)
            row += f"{aw/tot:>9.1%} "
            wins[a] += aw; games[a] += tot
        print(row)
    print("\n=== overall win rate (vs all others) ===")
    for n in sorted(names, key=lambda x: -wins[x] / max(games[x], 1)):
        print(f"  {n:<16} {wins[n]/max(games[n],1):.1%}  ({int(wins[n])}/{games[n]})")


if __name__ == "__main__":
    main()
