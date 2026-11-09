#!/usr/bin/env python3
"""
Head-to-head win rate of a checkpoint against the deployed (browser demo) model.

The deployed model is the ONNX file served by davinci-code-server. Its actions are
decoded exactly like the browser (docs/engine.js getAction): masked argmax for
color / position / decision, then a 2nd run conditioned on the chosen position
for the value.

Usage:
  python scripts/vs_deployed.py --ckpt checkpoints/latest.pt --games 1000
"""
import argparse
import math
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena_fast import vectorized_match  # noqa: E402
from src.agent import ModelAgent  # noqa: E402
from src.vec_env import SubprocVecEnv  # noqa: E402

DEFAULT_ONNX = os.path.expanduser("~/upbox/davinci-code-server/docs/model.onnx")
MASK_VALUE = -1e4


def masked_argmax(logits, mask):
    return int(np.argmax(np.where(mask, logits, MASK_VALUE)))


class OnnxAgent:
    """Deployed model. Its obs predates the my_hand_constraint input."""

    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def _run(self, obs, i, position):
        feeds = {
            "phase": obs["phase"][i:i + 1].astype(np.float32),
            "my_hand": obs["my_hand"][i:i + 1].astype(np.float32),
            "opponent_hand": obs["opponent_hand"][i:i + 1].astype(np.float32),
            "constraint_matrix": obs["opp_hand_constraint"][i:i + 1].astype(np.float32),
            "remaining_deck": obs["remaining_deck"][i:i + 1].astype(np.float32),
            "selected_position": np.array([position], dtype=np.int64),
        }
        color, pos, value, decision = self.session.run(
            ["color_logits", "position_logits", "value_logits", "decision_logits"], feeds)
        return color[0], pos[0], value[0], decision[0]

    def act_batch(self, obs, masks, deterministic=True):
        actions = np.zeros((len(obs["phase"]), 4), dtype=np.int64)
        for i in range(len(actions)):
            color, pos, _, decision = self._run(obs, i, 0)
            position = masked_argmax(pos, masks["position"][i])
            _, _, value, _ = self._run(obs, i, position)
            actions[i] = [
                masked_argmax(color, masks["color"][i]),
                position,
                masked_argmax(value, masks["value"][i][position]),
                masked_argmax(decision, masks["decision"][i]),
            ]
        return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/latest.pt")
    p.add_argument("--onnx", default=DEFAULT_ONNX)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--n-envs", type=int, default=100)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    agent = ModelAgent.from_checkpoint(args.ckpt, device=torch.device(args.device))
    deployed = OnnxAgent(args.onnx)
    vec = SubprocVecEnv(n_envs=args.n_envs, n_workers=args.n_workers)
    wins, losses, draws = vectorized_match(agent, deployed, args.games, vec, args.n_envs)
    vec.close()

    n = wins + losses + draws
    rate = wins / n
    ci = 1.96 * math.sqrt(rate * (1 - rate) / n)
    ts = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("timesteps", 0)
    print(f"{args.ckpt} ({ts:,} steps) vs deployed: {wins}-{losses} (draws {draws}) "
          f"-> win rate {rate:.1%} ±{ci:.1%} (n={n})", flush=True)


if __name__ == "__main__":
    main()
