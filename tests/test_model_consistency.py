"""The probabilities used to collect data must equal the ones PPO trains on."""
import numpy as np
import torch

from src.model import DaVinciCodePolicy


def _states(env, agent, n=200):
    obs, _ = env.reset()
    states = []
    while len(states) < n:
        mask = env.get_action_mask()
        states.append((obs, mask))
        action, _ = agent.act(obs, mask)
        obs, _r, _rew, term, trunc, _i, _res = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    obs = {k: torch.from_numpy(np.stack([s[0][k] for s in states])) for k in states[0][0]}
    mask = {k: torch.from_numpy(np.stack([s[1][k] for s in states])).bool() for k in states[0][1]}
    return obs, mask


def test_get_action_and_evaluate_actions_agree(env, agent):
    torch.manual_seed(0)
    policy = DaVinciCodePolicy().eval()
    obs, mask = _states(env, agent)
    with torch.no_grad():
        actions, log_probs, value = policy.get_action(obs, mask)
        taken = {k: torch.from_numpy(actions[:, i]) for i, k in enumerate(["color", "position", "value", "decision"])}
        new_log_probs, new_value, _, _ = policy.evaluate_actions(obs, taken, mask)
    for key in log_probs:
        torch.testing.assert_close(new_log_probs[key], log_probs[key])
    torch.testing.assert_close(new_value, value)


def test_sampled_actions_are_legal(env, agent):
    torch.manual_seed(0)
    policy = DaVinciCodePolicy().eval()
    obs, mask = _states(env, agent)
    with torch.no_grad():
        actions, _, _ = policy.get_action(obs, mask)
    rows = torch.arange(len(actions))
    position, value = torch.from_numpy(actions[:, 1]), torch.from_numpy(actions[:, 2])
    guess = obs["phase"][:, 1].bool()
    assert mask["position"][rows, position][guess].all()
    assert mask["value"][rows, position, value][guess].all()
