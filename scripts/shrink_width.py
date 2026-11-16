#!/usr/bin/env python3
"""
Build a narrower copy of a trained model as the starting point for fine-tuning.

  fusion / head-input width  512 -> --hidden-dim   (cannot be copied: learned by distillation)
  encoder feed-forward width 512 -> --ff-dim       (keep the most used neurons, copy their weights)

Everything else (embeddings, attention, later head layers) is copied unchanged.
Distillation = make the small model reproduce the big model's logits on states
from the big model's own self-play. No games are played by the small model here.

Usage (repo root):
  venv/bin/python scripts/shrink_width.py --src checkpoints/pruned/model_2layer_286M.pt \
      --dst experiments/narrow/checkpoints/latest.pt --hidden-dim 128 --ff-dim 256
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import ModelAgent  # noqa: E402
from src.env import DaVinciCodeEnv  # noqa: E402
from src.model import DaVinciCodePolicy, load_weights  # noqa: E402

SLOTS = 13


def play(agent, n_games, seed, deterministic):
    """States visited by `agent` playing itself: list of (obs, mask, action)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = DaVinciCodeEnv(seed=seed)
    states = []
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action, _ = agent.act(obs, mask, deterministic=deterministic)
            states.append(({k: np.array(v) for k, v in obs.items()},
                           {k: np.array(v) for k, v in mask.items()}, np.array(action)))
            obs, _r, _rw, term, trunc, _i, _res = env.step(action)
            done = term or trunc
    return states


def to_batch(states, device):
    obs = {k: torch.from_numpy(np.stack([s[0][k] for s in states])).to(device) for k in states[0][0]}
    mask = {k: torch.from_numpy(np.stack([s[1][k] for s in states])).bool().to(device) for k in states[0][1]}
    return obs, mask


def outputs(policy, obs, mask, position):
    """Everything the small model has to reproduce, as one dict of tensors."""
    features, _, enriched, belief = policy._encode(obs)
    out = policy.action_heads.logits_without_value(features, enriched, obs["phase"], mask)
    out["value"] = policy.action_heads.value_logits(features, enriched, obs["phase"], mask, position)
    out["belief"] = belief
    out["critic"] = policy.value_head(features)
    return out


def most_used_neurons(teacher, obs, keep):
    """Per encoder layer: indices of the `keep` feed-forward neurons that contribute most,
    and the mean activation of all neurons (to fold the dropped ones into the bias)."""
    layers = teacher.encoder.transformer.layers
    acts = {}
    hooks = [l.linear2.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0]))
             for i, l in enumerate(layers)]
    teacher.train()        # dropout is 0: same outputs, only turns off the fused fast path so hooks fire
    with torch.no_grad():
        teacher.encoder(obs)
    teacher.eval()
    for h in hooks:
        h.remove()
    chosen = []
    for i, l in enumerate(layers):
        a = acts[i].reshape(-1, acts[i].shape[-1])                           # (states * tokens, 512)
        score = (a ** 2).mean(0) * (l.linear2.weight ** 2).sum(0)            # energy this neuron adds to the output
        chosen.append((score.topk(keep).indices.sort().values, a.mean(0)))
    return chosen


def copy_feed_forward(teacher, student, chosen):
    with torch.no_grad():
        for t, s, (idx, mean_act) in zip(teacher.encoder.transformer.layers,
                                         student.encoder.transformer.layers, chosen):
            dropped = torch.ones_like(mean_act, dtype=torch.bool)
            dropped[idx] = False
            s.linear1.weight.copy_(t.linear1.weight[idx])
            s.linear1.bias.copy_(t.linear1.bias[idx])
            s.linear2.weight.copy_(t.linear2.weight[:, idx])
            # dropped neurons are replaced by their average contribution
            s.linear2.bias.copy_(t.linear2.bias + t.linear2.weight[:, dropped] @ mean_act[dropped])


def agreement(teacher_agent, student, states, device):
    """Share of states where the small model picks the same action as the big one."""
    agent = ModelAgent(student, device)
    same = sum(bool(np.array_equal(np.array(agent.act(o, m, deterministic=True)[0]), a)) for o, m, a in states)
    return same / len(states)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--games", type=int, default=1500, help="self-play games for distillation states")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    teacher_agent = ModelAgent.from_checkpoint(args.src, device=torch.device("cpu"))
    cfg = ckpt["config"]

    train_states = play(teacher_agent, args.games, seed=0, deterministic=False)    # sampled, like training
    test_states = play(teacher_agent, 150, seed=1, deterministic=True)
    print(f"states: {len(train_states)} for distillation, {len(test_states)} for the agreement check", flush=True)

    teacher = teacher_agent.policy.to(device).eval()
    student = DaVinciCodePolicy(hidden_dim=args.hidden_dim, n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                                ff_dim=args.ff_dim).to(device)
    fresh = load_weights(student, ckpt["policy_state_dict"])
    obs, mask = to_batch(train_states, device)
    if args.ff_dim < teacher.encoder.transformer.layers[0].linear1.out_features:
        sample = {k: v[:4096] for k, v in obs.items()}         # enough to rank neurons, fits in GPU memory
        copy_feed_forward(teacher, student, most_used_neurons(teacher, sample, args.ff_dim))
    print(f"parameters: {sum(x.numel() for x in teacher.parameters()):,} -> "
          f"{sum(x.numel() for x in student.parameters()):,}", flush=True)
    print(f"same action before distillation: {agreement(teacher_agent, student, test_states, device):.1%}", flush=True)

    # New layers learn fast, copied layers only adjust.
    new = [x for n, x in student.named_parameters() if n in fresh and "transformer" not in n]
    old = [x for n, x in student.named_parameters() if not (n in fresh and "transformer" not in n)]
    opt = torch.optim.Adam([{"params": new, "lr": 1e-3}, {"params": old, "lr": 1e-4}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    n = len(train_states)
    for step in range(1, args.steps + 1):
        rows = torch.randint(0, n, (args.batch,), device=device)
        o, m = {k: v[rows] for k, v in obs.items()}, {k: v[rows] for k, v in mask.items()}
        position = torch.randint(0, SLOTS, (args.batch,), device=device)
        with torch.no_grad():
            target = outputs(teacher, o, m, position)
        pred = outputs(student, o, m, position)
        loss = sum(F.mse_loss(pred[k], target[k]) * (0.01 if k == "critic" else 1.0) for k in target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.4f}, same action "
                  f"{agreement(teacher_agent, student, test_states, device):.1%}", flush=True)

    ckpt["policy_state_dict"] = student.state_dict()
    ckpt.pop("optimizer_state_dict", None)             # shapes changed: the trainer starts a fresh Adam
    ckpt["config"] = dict(cfg, hidden_dim=args.hidden_dim, ff_dim=args.ff_dim)
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
