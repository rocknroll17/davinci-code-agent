#!/usr/bin/env python3
"""
Evaluate a trained world model against the deployed legacy PPO model.

Two ways the world model can play:
  --mode planner  (default) pure world-model MPC: enumerate legal moves, roll
                  each inside the RSSM, play the best predicted line. No
                  learned policy involved.
  --mode actor    the Dreamer actor network (imagination-trained policy).

    python eval_wm.py --ckpt checkpoints_wm/dreamer_latest.pt --games 200
    python eval_wm.py --mode actor --games 200
    python eval_wm.py --horizon 10 --samples 16   # heavier planning
"""

import argparse

import torch

from src.agent import ModelAgent
from src.wm.dreamer import WMAgent
from src.wm.eval_arena import LegacyOpponentAdapter, evaluate_vs_legacy
from src.wm.planner import WMPlannerAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_wm/dreamer_latest.pt")
    ap.add_argument("--opponent", default="checkpoints/best_model_control.pt")
    ap.add_argument("--mode", choices=["planner", "actor"], default="planner")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=8, help="planner rollout depth")
    ap.add_argument("--samples", type=int, default=8, help="rollouts per candidate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "planner":
        agent = WMPlannerAgent.from_checkpoint(
            args.ckpt, device, horizon=args.horizon, n_samples=args.samples)
    else:
        agent = WMAgent.from_checkpoint(args.ckpt, device)

    opponent = LegacyOpponentAdapter(ModelAgent.from_checkpoint(args.opponent, device))
    stats = evaluate_vs_legacy(agent, opponent, n_games=args.games, seed0=args.seed)
    print(f"[{args.mode}] vs {args.opponent}: "
          f"win {stats['eval/win_rate']:.1%} "
          f"(P0 {stats['eval/win_rate_p0']:.1%} / P1 {stats['eval/win_rate_p1']:.1%}, "
          f"{int(stats['eval/n_games'])} games, mean length {stats['eval/mean_length']:.1f})")


if __name__ == "__main__":
    main()
