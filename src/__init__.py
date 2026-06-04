"""Da Vinci Code self-play RL — public API.

Typical use::

    from src import DaVinciCodeEnv, ModelAgent, run_episode
    from src import PPOTrainer, PPOConfig, RewardConfig

Game engine + spaces:  DaVinciCodeEnv, Phase, Color, CardValue
Model / agent:         DaVinciCodePolicy, ModelAgent, Agent (protocol)
Training:              PPOTrainer, PPOConfig, RewardConfig, RolloutBuffer, Transition, Episode
Play / evaluate:       run_episode, EpisodeResult, EvalSuite, EvalReport
"""

from src.agent import ModelAgent
from src.buffer import RolloutBuffer, Transition
from src.constants import CardValue, Color, Phase
from src.env import DaVinciCodeEnv
from src.episode import Episode
from src.eval_suite import EvalReport, EvalSuite
from src.interfaces import Agent, BatchAgent, Policy
from src.model import DaVinciCodePolicy, action_mask_to_tensor, obs_to_tensor
from src.reward_config import RewardConfig
from src.runner import EpisodeResult, run_episode
from src.trainer import PPOConfig, PPOTrainer

__all__ = [
    # game engine
    "DaVinciCodeEnv", "Phase", "Color", "CardValue",
    # model / agent
    "DaVinciCodePolicy", "obs_to_tensor", "action_mask_to_tensor",
    "ModelAgent", "Agent", "BatchAgent", "Policy",
    # training
    "PPOTrainer", "PPOConfig", "RewardConfig", "RolloutBuffer", "Transition", "Episode",
    # play / evaluate
    "run_episode", "EpisodeResult", "EvalSuite", "EvalReport",
]
