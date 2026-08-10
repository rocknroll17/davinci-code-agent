"""Head-to-head eval: legacy adapter in a joker_control game, win-rate arena."""

import numpy as np
import torch

from src.agent import ModelAgent
from src.constants import Phase
from src.env import DaVinciCodeEnv
from src.model import DaVinciCodePolicy
from src.wm.dreamer import DreamerConfig, DreamerTrainer, WMAgent
from src.wm.eval_arena import LegacyOpponentAdapter, evaluate_vs_legacy
from src.wm.nets import WMConfig

TINY = WMConfig(deter_dim=32, stoch_discrete=4, stoch_classes=4,
                hidden=32, embed_dim=32, n_bins=63)


def _legacy_agent() -> LegacyOpponentAdapter:
    torch.manual_seed(0)
    policy = DaVinciCodePolicy(hidden_dim=64, n_heads=4, n_layers=1)  # legacy arch
    return LegacyOpponentAdapter(ModelAgent(policy, torch.device("cpu")))


def test_adapter_plays_full_joker_game():
    """Legacy (3-phase) model completes joker_control games via the adapter."""
    adapter = _legacy_agent()
    finished = 0
    for seed in range(5):
        env = DaVinciCodeEnv(seed=seed, viewer=None, joker_control=True)
        obs, _ = env.reset()
        for _ in range(600):
            mask = env.get_action_mask()
            phase = int(np.argmax(obs["phase"]))
            action, _ = adapter.act(obs, mask)
            if phase == Phase.JOKER.value:
                assert mask["joker"][action[4]]  # random placement is legal
            obs, _, _, done, _, _, _ = env.step(action)
            if done:
                finished += 1
                break
    assert finished == 5


def test_evaluate_vs_legacy_alternates_seats():
    torch.manual_seed(1)
    tr = DreamerTrainer(
        DreamerConfig(n_envs=2, wm=TINY, save_dir="/tmp/wm_eval_test"),
        torch.device("cpu"))
    wm_agent = WMAgent(tr.wm, tr.actor, torch.device("cpu"))
    stats = evaluate_vs_legacy(wm_agent, _legacy_agent(), n_games=6, seed0=0)
    assert stats["eval/n_games"] == 6
    assert 0.0 <= stats["eval/win_rate"] <= 1.0
    # both seats actually played (3 games each)
    assert 0.0 <= stats["eval/win_rate_p0"] <= 1.0
    assert 0.0 <= stats["eval/win_rate_p1"] <= 1.0
    assert stats["eval/mean_length"] > 0


def test_trainer_evaluate_uses_checkpoint(tmp_path):
    """DreamerTrainer.evaluate loads the opponent checkpoint and reports stats."""
    torch.manual_seed(2)
    # fabricate a legacy checkpoint (fresh policy) for the opponent slot
    policy = DaVinciCodePolicy(hidden_dim=64, n_heads=4, n_layers=1)
    opp_path = str(tmp_path / "legacy.pt")
    torch.save({"policy_state_dict": policy.state_dict(),
                "config": {"hidden_dim": 64, "n_heads": 4, "n_layers": 1}}, opp_path)

    cfg = DreamerConfig(n_envs=2, wm=TINY, eval_games=4,
                        eval_opponent=opp_path, save_dir="/tmp/wm_eval_test2")
    tr = DreamerTrainer(cfg, torch.device("cpu"))
    stats = tr.evaluate(n_games=4)
    assert stats and stats["eval/n_games"] == 4

    # missing opponent → graceful skip
    cfg2 = DreamerConfig(n_envs=2, wm=TINY, eval_opponent="/nonexistent.pt",
                         save_dir="/tmp/wm_eval_test3")
    tr2 = DreamerTrainer(cfg2, torch.device("cpu"))
    assert tr2.evaluate(n_games=2) == {}


def test_planner_agent_plays_legal_moves():
    """Pure-MPC planner completes real games with only legal actions."""
    torch.manual_seed(3)
    from src.wm.nets import WorldModel
    from src.wm.planner import WMPlannerAgent
    wm = WorldModel(TINY)
    agent = WMPlannerAgent(wm, torch.device("cpu"), horizon=3, n_samples=2)
    env = DaVinciCodeEnv(seed=5, viewer=None, joker_control=True)
    obs, _ = env.reset()
    agent.reset()
    for _ in range(200):
        mask = env.get_action_mask()
        phase = int(np.argmax(obs["phase"]))
        action, _ = agent.act(obs, mask)
        if phase == Phase.GUESS.value:
            assert mask["position"][action[1]] and mask["value"][action[1]][action[2]]
        elif phase == Phase.JOKER.value:
            assert mask["joker"][action[4]]
        obs, _, _, done, _, _, _ = env.step(action)
        if done:
            break
    assert done
