"""my_hand_constraint: values the opponent has ruled out on my slots."""
import numpy as np
import torch

from src.model import DaVinciCodePolicy, obs_to_tensor
from src.result.guess_result import GuessResult


def test_opponent_failed_guess_shows_up_in_my_observation(env, agent):
    checked = 0
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.act(obs, env.get_action_mask())
            obs, _r, _rew, term, trunc, _info, result = env.step(action)
            done = term or trunc
            # A failed guess ends the turn: the next obs is the victim's, before they draw.
            if not done and isinstance(result, GuessResult) and not result.is_correct and not result.is_invalid:
                assert obs["my_hand_constraint"][result.position, result.guessed_value] == 1
                checked += 1
    assert checked > 0


def test_mirrors_opponents_own_constraint_matrix(env, agent):
    obs, _ = env.reset()
    for _ in range(30):
        action, _ = agent.act(obs, env.get_action_mask())
        obs, _r, _rew, term, trunc, _info, _res = env.step(action)
        if term or trunc:
            break
        opponent = env.players[1 - env._current_player]
        np.testing.assert_array_equal(obs["my_hand_constraint"], opponent._constraint_matrix)


def test_new_input_is_a_no_op_at_init(env):
    torch.manual_seed(0)
    policy = DaVinciCodePolicy().eval()
    obs, _ = env.reset()
    t = obs_to_tensor(obs)
    with torch.no_grad():
        _, v1, _ = policy(t)
        t["my_hand_constraint"] = torch.ones_like(t["my_hand_constraint"])
        _, v2, _ = policy(t)
    assert torch.equal(v1, v2)
