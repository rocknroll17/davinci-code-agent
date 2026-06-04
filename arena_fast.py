#!/usr/bin/env python3
"""
Vectorized batch arena — play hundreds of games in parallel per pairing.

The per-game arena (compare_experiments) plays ONE game at a time => batch-1
GPU forwards, GPU ~idle, CPU/launch-latency bound. This runs N games at once on
a SubprocVecEnv and does ONE batched forward per model per step, so the GPU is
actually used. Engine is untouched — only the eval loop is batched. ~10-50x.

Two models A, B: half the env slots have A as player0, half have B as player0
(colour-swap fairness baked in per slot). Auto-reset keeps slots playing fresh
games until each pairing reaches the target game count.

Usage:
  python arena_fast.py --root experiments --games 2000 --n-envs 256 --device cuda:0
"""
import argparse
import itertools
import math
import os

import numpy as np
import torch

from src.agent import ModelAgent
from src.vec_env import SubprocVecEnv


def vectorized_match(A, B, n_games, vec, n_envs):
    """Play >= n_games of A vs B on a running SubprocVecEnv. Returns (a_wins, b_wins, draws)."""
    obs, infos = vec.reset()
    cur = np.array([i.get("current_player", 0) for i in infos])
    a_is_p0 = np.arange(n_envs) < (n_envs // 2)   # half the slots: A plays player 0

    a_wins = b_wins = draws = done = 0
    steps = 0
    while done < n_games and steps < 100000:
        masks = vec.get_action_masks()
        act_a = A.act_batch(obs, masks, deterministic=True)   # (N,4)
        act_b = B.act_batch(obs, masks, deterministic=True)   # (N,4)
        # Which slots should use A's action this step?
        use_a = ((cur == 0) & a_is_p0) | ((cur == 1) & ~a_is_p0)
        actions = np.where(use_a[:, None], act_a, act_b)

        obs, _r, term, trunc, next_infos, _res = vec.step(actions)
        dones = term | trunc
        for i in np.nonzero(dones)[0]:
            w = next_infos[i].get("_winner")
            if w is None:
                draws += 1
            else:
                a_won = (w == 0 and a_is_p0[i]) or (w == 1 and not a_is_p0[i])
                a_wins += int(a_won); b_wins += int(not a_won)
            done += 1
        cur = np.array([ni.get("current_player", 0) for ni in next_infos])
        steps += 1
    return a_wins, b_wins, draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--ckpt", default="latest.pt")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-workers", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exps = sorted(
        e for e in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, e, "checkpoints", args.ckpt))
    )
    agents = {e: ModelAgent.from_checkpoint(os.path.join(args.root, e, "checkpoints", args.ckpt), device=dev)
              for e in exps}
    for e in exps:
        print(f"loaded {e}")

    vec = SubprocVecEnv(n_envs=args.n_envs, n_workers=args.n_workers)
    wins = {e: 0 for e in exps}
    games = {e: 0 for e in exps}
    grid = {}
    print("\n=== vectorized round-robin ===", flush=True)
    for a, b in itertools.combinations(exps, 2):
        aw, bw, dr = vectorized_match(agents[a], agents[b], args.games, vec, args.n_envs)
        tot = aw + bw + dr
        wins[a] += aw; wins[b] += bw
        games[a] += tot; games[b] += tot
        grid[(a, b)] = aw / tot; grid[(b, a)] = bw / tot
        print(f"  {a:9s} vs {b:9s}: {aw:5d} - {bw:5d} (draws {dr})  -> {a} {aw/tot:.1%}", flush=True)
    vec.close()

    rank = sorted(exps, key=lambda e: wins[e] / games[e], reverse=True)
    print("\n" + "=" * 64)
    print("FINAL RANKING (head-to-head win rate, with 95% CI)")
    print("=" * 64)
    for i, e in enumerate(rank, 1):
        pr = wins[e] / games[e]; n = games[e]
        ci = 1.96 * math.sqrt(pr * (1 - pr) / n)
        print(f"  {i}. {e:9s} {pr*100:5.1f}%  (n={n}, ±{ci*100:.1f}%p)")
    print(f"\n>>> BEST: {rank[0]}  {wins[rank[0]]/games[rank[0]]:.1%}")


if __name__ == "__main__":
    main()
