"""Da Vinci Code self-play RL — public API.

Typical use::

    from src import DaVinciCodeEnv, ModelAgent, run_episode
    from src import PPOTrainer, PPOConfig, RewardConfig

Game engine + spaces:  DaVinciCodeEnv, Phase, Color, CardValue
Model / agent:         DaVinciCodePolicy, ModelAgent, Agent (protocol)
Training:              PPOTrainer, PPOConfig, RewardConfig, RolloutBuffer, Transition, Episode
Play / evaluate:       run_episode, EpisodeResult, EvalSuite, EvalReport
"""

from src.env import DaVinciCodeEnv
from src.constants import Phase, Color, CardValue
from src.model import DaVinciCodePolicy, obs_to_tensor, action_mask_to_tensor
from src.agent import ModelAgent
from src.interfaces import Agent, BatchAgent, Policy
from src.reward_config import RewardConfig
from src.episode import Episode
from src.runner import run_episode, EpisodeResult
from src.trainer import PPOTrainer, PPOConfig
from src.buffer import RolloutBuffer, Transition
from src.eval_suite import EvalSuite, EvalReport

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
