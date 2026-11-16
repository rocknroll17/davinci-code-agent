# Da Vinci Code — Self-Play RL Agent

[![CI](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An agent that learns the deduction game Da Vinci Code from scratch with PPO self-play.
The policy is a small transformer over card tokens.

**[Play against it in your browser →](https://davinci-code.rocknroll17.com/)**

## Results

Win rate against the baseline model released with [Zhang et al. 2025](https://arxiv.org/abs/2506.12801),
under the paper's rules:

| Agent | Win rate | Games |
|---|---|---|
| **This agent** (2 layers, 0.48M parameters) | **60.0% ±1.0%** | 10,000 |
| This agent before shrinking (4 layers, 1.60M parameters) | 59.0% ±1.0% | 10,000 |
| PPO agent, as reported in the paper | 58.5% ±1.0% | 10,000 |
| Human experts, as reported in the paper | 50.0% – 64.0% | 20 – 25 |

The browser model is 0.68 MB (was 6.52 MB) and takes 0.65 ms per action (was 3.16 ms).

## How it works

Players guess each other's hidden, sorted number tiles; a wrong guess reveals one of your own.
One network plays both seats. Each turn is a DRAW, GUESS or DECISION step with its own action head.

```
hands, deck, guess records  ─►  Transformer encoder (55 tokens)  ─►  DRAW      color
                                         │                           GUESS     slot → value
                                 belief over hidden tiles            DECISION  stop / continue
```

## Usage

Python 3.10; a CUDA GPU for training.

```bash
pip install -r requirements.txt

python main.py                                   # train (resumes from checkpoints/latest.pt)
python scripts/vs_deployed.py --ckpt checkpoints/latest.pt   # win rate vs. the browser model
python play.py --auto                            # watch it play
```

The model is in [`src/model.py`](src/model.py), the PPO update in [`src/trainer.py`](src/trainer.py).
The browser demo lives in [davinci-code-server](https://github.com/rocknroll17/davinci-code-server).

## License

[MIT](LICENSE)
