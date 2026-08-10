"""joker_control mode: agent-chosen joker placement (JOKER phase + 5th action).

Covers:
- env invariants over many random games (jokers only enter hands via JOKER
  actions, masks expose exactly the legal insert indices, obs/action shapes)
- chosen insert index is honored
- legacy mode is untouched (3-dim phase, 4-dim action, no "joker" mask)
- model emits/evaluates the "joker" head so PPO can train it
"""

import numpy as np
import pytest
import torch

from src.constants import MAX_HAND_SIZE, Phase
from src.env import DaVinciCodeEnv
from src.model import DaVinciCodePolicy, action_mask_to_tensor, obs_to_tensor


def random_action(env: DaVinciCodeEnv, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random VALID action from the env's masks."""
    masks = env.get_action_mask()
    color = rng.choice(np.flatnonzero(masks["color"])) if masks["color"].any() else 0
    position = rng.choice(np.flatnonzero(masks["position"])) if masks["position"].any() else 0
    row = masks["value"][position]
    value = rng.choice(np.flatnonzero(row)) if row.any() else 0
    decision = rng.integers(0, 2)
    action = [color, position, value, decision]
    if env.joker_control:
        action.append(rng.choice(np.flatnonzero(masks["joker"])))
    return np.array(action, dtype=np.int64)


def play_random_game(seed: int) -> DaVinciCodeEnv:
    env = DaVinciCodeEnv(seed=seed, viewer=None, joker_control=True)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    assert obs["phase"].shape == (4,)

    for _ in range(2000):
        phase_before = Phase(int(np.argmax(obs["phase"])))
        if phase_before == Phase.JOKER:
            hand = env.players[env._current_player]._hand
            masks = env.get_action_mask()
            # mask exposes exactly insert indices 0..hand.size
            expected = np.zeros(MAX_HAND_SIZE, dtype=bool)
            expected[:min(hand.size + 1, MAX_HAND_SIZE)] = True
            assert (masks["joker"] == expected).all()
        obs, _, _, terminated, _, _, _ = env.step(random_action(env, rng))
        if terminated:
            break
    return env


def test_random_games_terminate_and_place_all_jokers():
    for seed in range(30):
        env = play_random_game(seed)
        assert env._done, f"game {seed} did not finish"
        # every joker ended up in some hand, none pending
        assert not env._pending_jokers
        jokers = sum(1 for p in env.players for c in p._hand if c.is_joker)
        drawn_jokers = 2 - (len([c for c in env._deck._black_cards if c.is_joker])
                            + len([c for c in env._deck._white_cards if c.is_joker]))
        assert jokers == drawn_jokers


def test_initial_jokers_enter_joker_phase():
    """Find a seed whose initial deal contains a joker → reset starts in JOKER phase."""
    found = False
    for seed in range(200):
        env = DaVinciCodeEnv(seed=seed, viewer=None, joker_control=True)
        obs, _ = env.reset()
        if env._pending_jokers:
            found = True
            assert int(np.argmax(obs["phase"])) == Phase.JOKER.value
            owner = env._pending_jokers[0][0]
            assert env._current_player == owner
            # owner's hand is missing the joker(s) pre-placement
            n_pending = sum(1 for p, _ in env._pending_jokers if p == owner)
            assert env.players[owner]._hand.size == 4 - n_pending
            break
    assert found, "no seed with an initial joker in 200 tries"


def test_chosen_insert_position_is_honored():
    for seed in range(200):
        env = DaVinciCodeEnv(seed=seed, viewer=None, joker_control=True)
        env.reset()
        if not env._pending_jokers:
            continue
        owner, _ = env._pending_jokers[0]
        size_before = env.players[owner]._hand.size
        target = size_before  # append at end
        env.step(np.array([0, 0, 0, 0, target], dtype=np.int64))
        hand = env.players[owner]._hand
        assert hand.size == size_before + 1
        assert hand[target].is_joker
        return
    pytest.skip("no seed with an initial joker in 200 tries")


def test_legacy_mode_unchanged():
    env = DaVinciCodeEnv(seed=0, viewer=None)
    obs, _ = env.reset()
    assert obs["phase"].shape == (3,)
    assert "joker" not in env.get_action_mask()
    assert len(env.action_space.nvec) == 4
    # jokers place themselves (never a JOKER phase)
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert int(np.argmax(obs["phase"])) != Phase.JOKER.value
        obs, _, _, terminated, _, _, _ = env.step(random_action(env, rng))
        if terminated:
            break


def test_model_joker_head_end_to_end():
    torch.manual_seed(0)
    env = DaVinciCodeEnv(seed=3, viewer=None, joker_control=True)
    obs, _ = env.reset()
    policy = DaVinciCodePolicy(hidden_dim=64, n_heads=4, n_layers=1, joker_control=True)

    obs_t = obs_to_tensor(obs)
    mask_t = action_mask_to_tensor(env.get_action_mask())
    action, log_probs, value = policy.get_action(obs_t, mask_t)

    assert action.shape == (1, 5)
    assert "joker" in log_probs

    # evaluate_actions returns joker log-probs/entropy for PPO
    actions_dict = {
        k: torch.tensor([int(action[0, i])])
        for i, k in enumerate(["color", "position", "value", "decision", "joker"])
    }
    lp, v, ent, belief = policy.evaluate_actions(obs_t, actions_dict, mask_t)
    assert "joker" in lp and "joker" in ent
    assert belief.shape == (1, 13, 13)

    # In JOKER phase the sampled insert index must respect the mask
    if int(np.argmax(obs["phase"])) == Phase.JOKER.value:
        assert mask_t["joker"][0, int(action[0, 4])]


def test_legacy_model_signature_unchanged():
    policy = DaVinciCodePolicy(hidden_dim=64, n_heads=4, n_layers=1)
    env = DaVinciCodeEnv(seed=0, viewer=None)
    obs, _ = env.reset()
    out = policy.encoder(obs_to_tensor(obs))
    assert len(out) == 3  # analysis scripts unpack a 3-tuple
    action, log_probs, _ = policy.get_action(obs_to_tensor(obs))
    assert action.shape == (1, 4)
    assert "joker" not in log_probs
