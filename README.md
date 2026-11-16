# Da Vinci Code — Self-Play RL Agent

[![CI](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rocknroll17/davinci-code-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An agent that learns the deduction game Da Vinci Code from scratch with PPO self-play.
The policy is a small transformer over card tokens.

**[Play against it in your browser →](https://davinci-code.rocknroll17.com/)**

## Results

| | |
|---|---|
| vs. the released baseline of [Zhang et al. 2025](https://arxiv.org/abs/2506.12801) | 57.6% ±1.0% (10,000 games) |
| Model size, same win rate | 1.60M → 0.48M parameters (50.0% ±1.8%, 3,006 games) |
| Browser model | 6.52 MB → 0.68 MB, 3.16 ms → 0.65 ms per action |

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
