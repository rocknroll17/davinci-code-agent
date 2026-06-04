# Da Vinci Code — Self-Play RL Agent

[![CI](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml)
[![CodeQL](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/codeql.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A reinforcement-learning agent that learns the deduction game **Da Vinci Code** from
scratch through **PPO self-play** — with a **transformer policy network** that explicitly
reasons about the opponent's hidden cards (a *belief* auxiliary head).

### Try it

- **[Play against the trained agent in your browser →](https://rocknroll17.github.io/davinci-code-server/)**
  This same network, exported to ONNX and running 100% client-side — no install.
  Served by [**davinci-code-server**](https://github.com/rocknroll17/davinci-code-server).

<details>
<summary><b>Table of contents</b></summary>

- [The game](#the-game)
- [How it works](#how-it-works)
- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Reward shaping](#reward-shaping)
- [Observation & action spaces](#observation--action-spaces)
- [PPO hyperparameters](#ppo-hyperparameters)
- [Repository layout](#repository-layout)
- [Docker](#docker)
- [License](#license)

</details>

## The game

Da Vinci Code is a deduction board game: you guess your opponent's hidden number tiles
before they guess yours.

- Tiles are black or white, valued **0–11** plus a **Joker (12)**.
- Each hand is kept **sorted ascending** (ties: black before white) — leaking ordering info.
- Guess right → the tile flips face-up and you may keep guessing; guess wrong → one of
  **your** tiles flips up. First to reveal all of the opponent's tiles **wins**.

## How it works

A **single policy network plays both seats** (adversarial self-play) and is optimized with
**PPO**. One game step is one of three phases — **DRAW**, **GUESS**, **DECISION** — and the
network routes to the matching action head (*phase gating*).

```
Observation (hands, deck, constraints, phase)
      │
      ▼
Transformer encoder  ──►  CLS = global state   ─┐
(42 tokens, self-attention)   per-slot opponent ─┤
      │                                          │
      ▼                                          ▼
Belief module  ── predicts each hidden           Phase-gated heads
opponent tile's value distribution (aux loss),   ├─ DRAW    → color
fed back into the opponent representation        ├─ GUESS   → position → value (autoregressive)
                                                 └─ DECISION→ stop / continue
                                          + Value head V(s) for the PPO critic
```

The **belief module** is the core idea: a self-supervised auxiliary task predicts the
opponent's hidden values; its gradient shapes the encoder toward representations that are
good for *deduction*, and its (detached) prediction enriches the per-slot features the
GUESS head attends to.

> **Full I/O spec, tokenization, and the belief→action pipeline:** see
> [`src/docs/model.md`](src/docs/model.md).

## Features

- **PPO self-play** — one network learns offense and defense simultaneously.
- **Transformer policy** with a learnable CLS token and constraint-as-token encoding.
- **Belief auxiliary head** — explicit opponent-card-value prediction (CrossEntropy aux loss).
- **Phase-gated multi-head** action space with legal-move masking.
- **Vectorized self-play** (`src/vec_env.py`) for fast rollout collection.
- **Training hooks** (`src/hooks.py`) — NaN/loss-spike/action-diversity monitors, auto-checkpointing.
- **Live dashboard** (`src/dashboard/`) and a Rich TUI visualizer for monitoring runs.
- **Resumable checkpoints**, fine-tune mode, and an evaluation suite.

## Install

Requires **Python 3.10** and (for training) a CUDA-capable GPU.

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# Train (auto-resumes from checkpoints/latest.pt). Rich visualization by default.
python main.py
python main.py --no-viz            # headless (servers/CI)
python main.py --dashboard         # browser dashboard on :6006
python main.py --reset-optimizer   # reset optimizer state on resume

# Evaluate a checkpoint's win rate
python eval.py --checkpoint checkpoints/best_model.pt --episodes 500 --device cuda

# Watch the model play (Rich)
python play.py --auto --delay 0.5
python play.py --episodes 5 --deterministic
```

## Reward shaping

Outcome-dominated, with small shaping terms (exact values in [`src/constants.py`](src/constants.py)):

| Event | Reward |
|-------|--------|
| Win / Lose | **+10.0** / **−10.0** |
| Correct guess (normal / Joker) | +0.5 / +1.0 |
| Wrong guess | −0.5 |
| Guess outside the sort-order range | −0.5 |
| Streak bonus / break | +0.2 × streak / −0.1 |
| Invalid action | −1.0 |
| Continue → correct / wrong | +0.4 / −0.25 |
| Stop while a card is fully / nearly determined | −0.5 / −0.15 per card |
| Draw in a winning / losing game | +0.1 / −0.1 |

## Observation & action spaces

**Observation** (dict):

| Key | Shape | Description |
|-----|-------|-------------|
| `phase` | (3,) | one-hot [DRAW, GUESS, DECISION] |
| `my_hand` | (13, 2) | my tiles `[color, value]` |
| `opponent_hand` | (13, 2) | opponent tiles (hidden value = −1) |
| `remaining_deck` | (2,) | remaining [black, white] |
| `constraint_matrix` | (13, 13) | per-slot ruled-out values (rule-derived) |

**Actions** (phase-gated heads):

| Head | Size | Phase | Meaning |
|------|------|-------|---------|
| color | 2 | DRAW | BLACK / WHITE |
| position | 13 | GUESS | which opponent slot to attack |
| value | 13 | GUESS | guessed value (conditioned on position) |
| decision | 2 | DECISION | STOP / CONTINUE |

## PPO hyperparameters

Defaults from `PPOConfig` in [`src/trainer.py`](src/trainer.py) (override via `src/experiment_config.py`):

| Parameter | Value | | Parameter | Value |
|-----------|-------|-|-----------|-------|
| learning_rate | 8e-5 → 3e-5 | | clip_range | 0.2 |
| n_envs | 300 | | clip_range_vf | 10.0 |
| episodes_per_update | 300 | | ent_coef | 0.01 |
| batch_size | 1096 | | color_ent_coef | 0.02 |
| n_epochs | 8 | | vf_coef | 0.5 |
| gamma / gae_lambda | 0.99 / 0.95 | | belief_coef | 0.2 |
| hidden_dim | 512 | | max_grad_norm | 0.5 |

## Repository layout

```
main.py            Training entry point (--no-viz / --reset-optimizer / --dashboard)
run_experiment.py  Single-GPU experiment runner   train_ddp.py  Multi-GPU (torchrun) trainer
eval.py            Win-rate evaluation            play.py       Watch the model play (Rich)
compare_experiments.py / arena_fast.py            Head-to-head arenas
src/
  model.py         Transformer policy + belief module + phase-gated heads
  trainer.py       PPO trainer (GAE, losses, hooks)
  env.py           Gymnasium environment   vec_env.py  Vectorized self-play
  buffer.py        Rollout buffer          agent.py    Inference wrapper
  runner.py        Single-game loop        episode.py  Retroactive-reward episode
  hooks.py         Training monitors       eval_suite.py  Deep eval metrics
  reward_config.py / interfaces.py         visualizer.py  Rich TUI
  dashboard/       Live web dashboard      docs/model.md  Model spec
  deck.py hand.py player.py phase.py constants.py  cards/  result/  utils/
checkpoints/       Saved models (gitignored)    logs/  Training logs
```

## Docker

A reproducible training image (no image is published — training isn't a deployed service):

```bash
docker build -t davinci-agent .
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/logs:/app/logs" \
    davinci-agent            # runs `python main.py --no-viz`
```

## License

[MIT](LICENSE)
