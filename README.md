# Da Vinci Code Self-Play RL Agent

A **reinforcement learning self-play agent** for the Da Vinci Code board game.  
Uses a **Phase-Gated Multi-Head Policy Network** trained via  
**Adversarial Self-Play** with PPO.

## About the Game

Da Vinci Code is a deduction-based board game where players try to guess their opponent's tiles.
- Each player starts with black and white tiles
- Tiles range from 0–11 plus a Joker (12)
- Tiles are sorted in ascending order in hand
- The first player to reveal all opponent tiles wins

## Project Structure

```
davinci-agent/
├── main.py                 # Training entry point
├── eval.py                 # Model evaluation (win rate measurement)
├── play.py                 # Watch model play (Rich visualization)
├── requirements.txt        # Training dependencies
├── src/
│   ├── constants.py        # Constants and enums
│   ├── env.py              # Gymnasium Environment
│   ├── model.py            # Phase-Gated Policy Network
│   ├── buffer.py           # Rollout Buffer (GAE)
│   ├── trainer.py          # PPO Trainer
│   ├── deck.py             # Deck management
│   ├── hand.py             # Hand management
│   ├── player.py           # Player class
│   ├── phase.py            # Phase management
│   ├── vec_env.py          # Vectorized environment (parallel training)
│   ├── visualizer.py       # Rich-based training visualizer
│   ├── cards/
│   │   ├── card.py         # Base Card class
│   │   ├── black_card.py   # Black card
│   │   └── white_card.py   # White card
│   ├── result/
│   │   ├── result.py       # Base Result class
│   │   ├── guess_result.py # Guess result
│   │   ├── draw_result.py  # Draw result
│   │   └── streak_result.py# Streak result
│   ├── utils/
│   │   ├── game_logic.py   # Game logic utilities
│   │   ├── logger.py       # Logging configuration
│   │   └── utils.py        # General utilities
│   ├── docs/
│   │   └── model.md        # Model I/O design specification
│   └── tools/
│       └── validate.py     # Hand validation debug tool
├── checkpoints/            # Model checkpoints
└── logs/                   # Training logs (JSON)
```

## Setup

### 1. Create and activate virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Training
```bash
# Default training (with Rich visualization)
python main.py

# Training without visualization
python main.py --no-viz

# Fine-tuning mode (edge case training)
python main.py --finetune

# Reset optimizer (when switching from fine-tuning to normal training)
python main.py --reset-optimizer
```

Training automatically resumes from `checkpoints/latest.pt` if it exists.

### 4. Evaluation
```bash
# Default evaluation (200 games)
python eval.py

# Custom options
python eval.py --checkpoint checkpoints/best_model.pt --episodes 500 --device cuda
```

### 5. Watch Model Play
```bash
# Step-by-step (press Enter each step)
python play.py

# Auto-play
python play.py --auto --delay 0.5

# Multiple episodes
python play.py --episodes 5 --deterministic
```

## Model Architecture

### Phase-Gated Multi-Head Policy Network

```
Observation → Encoder → Features
                           ↓
                    ┌──────┴──────┐
                    │   Phase     │
                    │   Gating    │
                    └──────┬──────┘
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
    Color Head      Position/Value      Decision Head
    (DRAW Phase)    (GUESS Phase)      (DECISION Phase)
```

- **Observation Encoder**: Converts dict observation into a unified feature vector
- **Phase Gating**: Activates only the head corresponding to the current phase
- **Value Head**: State value estimation for Actor-Critic training

### Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `phase` | (3,) | One-hot: [DRAW, GUESS, DECISION] |
| `my_hand` | (13, 2) | My hand: [color, value] × 13 |
| `opponent_hand` | (13, 2) | Opponent hand (hidden cards = -1) |
| `remaining_deck` | (2,) | [black, white] remaining count |
| `constraint_matrix` | (13, 13) | Failed guess history |

### Action Space

| Head | Size | Phase | Description |
|------|------|-------|-------------|
| color | 2 | DRAW | BLACK(0) / WHITE(1) |
| position | 13 | GUESS | Opponent hand position (0-12) |
| value | 13 | GUESS | Guessed value (0-12) |
| decision | 2 | DECISION | STOP(0) / CONTINUE(1) |

## Reward Structure

| Event | Reward |
|-------|--------|
| Win | +10.0 |
| Lose | -10.0 |
| Correct guess | +0.5 |
| Correct joker guess | +1.0 |
| Wrong guess | -0.5 |
| Streak bonus | +(0.2 × streak) |
| Streak break | -0.1 |
| Invalid action | -1.0 |
| Stop decision | 0.0 |
| Draw (winning game) | +0.1 |
| Draw (losing game) | -0.1 |
| Continue → correct | +0.2 |
| Continue → wrong | -0.2 |
| Stop with determined cards | -0.3 per card |

## PPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 0.5e-5 | Learning rate |
| n_envs | 1000 | Parallel environments |
| episodes_per_update | 1000 | Episodes per update |
| batch_size | 2048 | Mini-batch size |
| n_epochs | 8 | PPO epochs |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_range | 0.07 | PPO clip range |
| ent_coef | 0.002 | Entropy coefficient |
| color_ent_coef | 0.05 | Color head entropy coefficient |
| vf_coef | 0.5 | Value function coefficient |

## Self-Play

A single policy network controls both players:

```python
while not done:
    action = policy.get_action(obs, action_mask)
    obs, reward, done, info = env.step(action)
    buffer.add(transition)
```

This approach enables:
- Simultaneous learning of offense and defense
- Self-identification and correction of weaknesses
- Gradual convergence toward stronger strategies

## License

MIT License
