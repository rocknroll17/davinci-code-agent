"""
ExperimentConfig — YAML-based configuration for PPO training experiments.

Usage
-----
Create a config file
~~~~~~~~~~~~~~~~~~~~
from src.experiment_config import ExperimentConfig
from src.trainer import PPOConfig

cfg = PPOConfig(learning_rate=1e-4, n_envs=500)
ExperimentConfig.to_yaml(cfg, "experiments/fast_lr.yaml")

Load and run
~~~~~~~~~~~~
from src.experiment_config import ExperimentConfig

config = ExperimentConfig.from_yaml("experiments/fast_lr.yaml")
trainer = PPOTrainer(config)
trainer.train()

Programmatic override
~~~~~~~~~~~~~~~~~~~~~
config = ExperimentConfig.from_yaml("base.yaml")
config.learning_rate = 2e-4   # override after load
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.trainer import PPOConfig


class ExperimentConfig:
    """
    Serialiser / deserialiser for ``PPOConfig`` using YAML.

    Requires ``pyyaml`` (``pip install pyyaml``).
    """

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> PPOConfig:
        """
        Load a ``PPOConfig`` from a YAML file.

        Parameters
        ----------
        path : str
            Path to the ``.yaml`` file.
        overrides : dict, optional
            Key-value pairs that override values from the file.

        Returns
        -------
        PPOConfig
        """
        import yaml  # deferred so callers without PyYAML still work

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        if overrides:
            raw.update(overrides)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PPOConfig:
        """
        Build a ``PPOConfig`` from a plain dictionary.

        Unknown keys are ignored; type coercion is applied for common
        numeric fields so that YAML strings ("1e-4") become floats.
        """
        defaults = PPOConfig()
        valid_fields = {f for f in defaults.__dataclass_fields__}

        # Type coercion map (field name → target Python type)
        _float_fields = {
            "learning_rate", "gamma", "gae_lambda", "clip_range",
            "clip_range_vf", "ent_coef", "color_ent_coef", "vf_coef",
            "belief_coef", "max_grad_norm", "lr_end",
        }
        _int_fields = {
            "total_timesteps", "n_envs", "episodes_per_update",
            "batch_size", "n_epochs", "hidden_dim", "n_heads", "n_layers",
            "n_workers",
            "log_interval", "save_interval", "eval_interval",
            "n_eval_episodes",
        }
        _bool_fields = {
            "reset_optimizer_on_load", "monotone_reward", "zero_init",
            "fp16", "compile",
        }

        unknown = sorted(k for k in d if k not in valid_fields)
        if unknown:
            import warnings
            warnings.warn(
                f"ExperimentConfig: unknown YAML keys will be ignored: {unknown}",
                stacklevel=2,
            )

        kwargs: Dict[str, Any] = {}
        for key, val in d.items():
            if key not in valid_fields:
                continue
            if key in _float_fields:
                val = float(val)
            elif key in _int_fields:
                val = int(val)
            elif key in _bool_fields:
                val = bool(val)
            elif key == "reward_config" and isinstance(val, dict):
                from src.reward_config import RewardConfig
                val = RewardConfig(**val)
            kwargs[key] = val

        return PPOConfig(**kwargs)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @classmethod
    def to_yaml(cls, config: PPOConfig, path: str) -> None:
        """
        Save a ``PPOConfig`` to a YAML file.

        Parameters
        ----------
        config : PPOConfig
        path : str
            Destination path (parent directories are created automatically).
        """
        import yaml  # deferred

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = config.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def default_yaml(cls) -> str:
        """Return the default ``PPOConfig`` as a YAML string (no file I/O)."""
        import yaml

        return yaml.dump(PPOConfig().to_dict(), default_flow_style=False, sort_keys=True)

    @classmethod
    def diff(cls, a: PPOConfig, b: PPOConfig) -> Dict[str, tuple]:
        """
        Return fields that differ between two configs.

        Returns
        -------
        dict
            ``{field_name: (a_value, b_value)}`` for every field where
            ``a.field != b.field``.
        """
        da, db = a.to_dict(), b.to_dict()
        return {k: (da[k], db[k]) for k in da if da[k] != db.get(k)}
