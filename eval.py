#!/usr/bin/env python3
"""Evaluation script for trained Da Vinci Code policy."""
import argparse
import os

import torch

from src.trainer import PPOConfig, PPOTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    config = PPOConfig()
    # For evaluation we only need 1 environment
    config.n_envs = 1
    config.n_steps = 1

    device = torch.device(args.device) if args.device else None

    trainer = PPOTrainer(config, device=device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    trainer.load(args.checkpoint)

    print(f"Evaluating for {args.episodes} episodes on {trainer.device}")
    stats = trainer.evaluate(n_episodes=args.episodes)

    print("\n--- Evaluation Results ---")
    print(f"Player0 win rate: {stats['player0_win_rate']:.2%}")
    print(f"Player1 win rate: {stats['player1_win_rate']:.2%}")
    print(f"Mean reward P0: {stats['mean_reward_p0']:.3f}")
    print(f"Mean reward P1: {stats['mean_reward_p1']:.3f}")
    print(f"Mean episode length: {stats['mean_episode_length']:.2f}")


if __name__ == "__main__":
    main()
