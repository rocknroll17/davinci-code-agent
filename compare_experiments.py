#!/usr/bin/env python3
"""
Compare the 4 trained experiments and decide which is best.

Two complementary views:
  1. Head-to-head round-robin — each model plays every other model. The env
     alternates turns, so at each step we let whichever side is "current_player"
     act with its own policy. Colour-swap halves the games to cancel first-move
     advantage. This is the fairest "who actually wins" signal and is reward-shape
     independent (it only counts game outcomes), so the monotone-reward model is
     compared on equal footing.
  2. Per-model EvalSuite metrics — guess accuracy, joker accuracy, belief
     accuracy, mean episode length (self-play). Diagnostic, not the ranking.

Usage:
    python compare_experiments.py                       # all under experiments/
    python compare_experiments.py --games 400 --device cuda:0
    python compare_experiments.py --ckpt latest.pt      # use latest instead of best
"""

import argparse
import itertools
import os
from typing import Dict

import torch

from src.agent import ModelAgent
from src.env import DaVinciCodeEnv
from src.eval_suite import EvalSuite
from src.runner import run_episode

EXPERIMENTS = ["baseline", "monotone", "heads", "layers"]


def play_match(agent_a: ModelAgent, agent_b: ModelAgent, n_games: int,
               device: torch.device, seed0: int = 0) -> Dict[str, int]:
    """Play n_games of A(player0) vs B(player1), deterministic. Returns win counts."""
    env = DaVinciCodeEnv()
    wins = {"a": 0, "b": 0, "draw": 0}
    for g in range(n_games):
        res = run_episode(env, [agent_a, agent_b], deterministic=True, seed=seed0 + g)
        if res.winner == 0:
            wins["a"] += 1
        elif res.winner == 1:
            wins["b"] += 1
        else:
            wins["draw"] += 1
    return wins


def head_to_head(agents: Dict[str, ModelAgent], n_games: int,
                 device: torch.device) -> Dict[str, Dict]:
    """Round-robin. Each unordered pair plays n_games split across colour swap."""
    half = max(1, n_games // 2)
    score = {name: 0.0 for name in agents}      # 1 per win, 0.5 per draw
    games = {name: 0 for name in agents}
    grid: Dict[str, Dict[str, float]] = {a: {} for a in agents}

    for a, b in itertools.combinations(agents, 2):
        # A as player0
        r1 = play_match(agents[a], agents[b], half, device, seed0=1000)
        # swap colours: B as player0
        r2 = play_match(agents[b], agents[a], half, device, seed0=5000)
        a_wins = r1["a"] + r2["b"]
        b_wins = r1["b"] + r2["a"]
        draws = r1["draw"] + r2["draw"]
        total = a_wins + b_wins + draws
        score[a] += a_wins + 0.5 * draws
        score[b] += b_wins + 0.5 * draws
        games[a] += total
        games[b] += total
        grid[a][b] = a_wins / total if total else 0.0
        grid[b][a] = b_wins / total if total else 0.0
        print(f"  {a:9s} vs {b:9s}:  {a_wins:4d} - {b_wins:4d}  (draws {draws})  "
              f"-> {a} winrate {a_wins/total:.1%}")

    winrate = {name: score[name] / games[name] if games[name] else 0.0 for name in agents}
    return {"winrate": winrate, "grid": grid, "score": score, "games": games}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--ckpt", default="best_model.pt", help="checkpoint filename inside <exp>/checkpoints/")
    p.add_argument("--games", type=int, default=400, help="games per pairing (split across colour swap)")
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Auto-discover every experiment dir under root that has the requested checkpoint.
    discovered = sorted(
        d for d in os.listdir(args.root)
        if os.path.exists(os.path.join(args.root, d, "checkpoints", args.ckpt))
    )
    exps = discovered or EXPERIMENTS
    agents: Dict[str, ModelAgent] = {}
    for exp in exps:
        path = os.path.join(args.root, exp, "checkpoints", args.ckpt)
        if os.path.exists(path):
            agents[exp] = ModelAgent.from_checkpoint(path, device=device)
            ts = getattr(agents[exp], "_timesteps", None)
            print(f"loaded {exp:10s} <- {path}  (timesteps={ts})")
        else:
            print(f"SKIP   {exp:10s}: not found ({path})")

    if len(agents) < 2:
        print("\nNeed at least 2 trained checkpoints to compare. Exiting.")
        return

    print("\n=== Head-to-head round-robin ===")
    h2h = head_to_head(agents, args.games, device)

    diag = {n: {"guess_acc": 0.0} for n in agents}
    if args.eval_episodes > 0:
        print("\n=== Per-model diagnostics (self-play EvalSuite) ===")
        for name, ag in agents.items():
            rep = EvalSuite.run_policy(ag.policy, device, n_episodes=args.eval_episodes, seed=12345)
            diag[name] = {
                "guess_acc": rep.guess_accuracy,
                "joker_acc": rep.joker_accuracy,
                "belief_acc": rep.belief_accuracy,
                "mean_len": rep.mean_episode_length,
                "invalid_rate": rep.invalid_action_rate,
            }
            print(f"  {name:9s}  guess={rep.guess_accuracy:.1%}  joker={rep.joker_accuracy:.1%}  "
                  f"belief={(rep.belief_accuracy or 0):.1%}  len={rep.mean_episode_length:.1f}  "
                  f"invalid={rep.invalid_action_rate:.2%}")

    # Final ranking by head-to-head win rate.
    ranking = sorted(agents, key=lambda n: h2h["winrate"][n], reverse=True)
    print("\n" + "=" * 60)
    print("FINAL RANKING (by head-to-head win rate)")
    print("=" * 60)
    for i, name in enumerate(ranking, 1):
        print(f"  {i}. {name:9s}  win rate {h2h['winrate'][name]:.1%}   "
              f"guess_acc {diag[name]['guess_acc']:.1%}")
    print(f"\n>>> BEST: {ranking[0]}  (head-to-head win rate {h2h['winrate'][ranking[0]]:.1%})")


if __name__ == "__main__":
    main()
