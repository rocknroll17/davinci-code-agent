#!/usr/bin/env python3
"""Run ONE head-to-head pairing (A vs B), colour-swapped, and print the result.

Used by the parallel arena launcher to spread the 6 pairings across GPUs.
Reuses the verified play_match (which uses run_episode).
"""
import argparse
import os

import torch

from compare_experiments import play_match
from src.agent import ModelAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("--ckpt", default="latest.pt")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=1000)   # total, split across colour swap
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    A = ModelAgent.from_checkpoint(os.path.join(args.root, args.a, "checkpoints", args.ckpt), device=dev)
    B = ModelAgent.from_checkpoint(os.path.join(args.root, args.b, "checkpoints", args.ckpt), device=dev)

    half = max(1, args.games // 2)
    r1 = play_match(A, B, half, dev, seed0=1000)   # A as player0
    r2 = play_match(B, A, half, dev, seed0=5000)   # B as player0
    a_wins = r1["a"] + r2["b"]
    b_wins = r1["b"] + r2["a"]
    draws = r1["draw"] + r2["draw"]
    with open(args.out, "w") as f:
        f.write(f"{args.a} {args.b} {a_wins} {b_wins} {draws}\n")
    print(f"{args.a} {a_wins} - {b_wins} {args.b} (draws {draws})")


if __name__ == "__main__":
    main()
