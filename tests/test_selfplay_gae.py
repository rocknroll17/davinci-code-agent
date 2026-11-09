"""Self-play GAE: advantages are computed per (env, player) stream."""
from collections import defaultdict

import numpy as np
import pytest
import torch

from src.buffer import RolloutBuffer, Transition
from src.constants import Phase
from src.episode import Episode
from src.trainer import PPOConfig, PPOTrainer

GAMMA, LAM = 0.9, 0.8


def _t(pid, reward, value, done=False, env_id=0):
    return Transition(
        obs={}, player_id=pid, action=np.zeros(4), reward=reward, done=done,
        log_probs={}, value=value, env_id=env_id,
    )


def _gae(rewards, values):
    """Reference GAE for one trajectory ending in a terminal state."""
    adv, last = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + GAMMA * nxt - values[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv


def test_gae_runs_per_player_not_per_env():
    # Interleaved game: p0, p0, p1, p1, p0, p1(final, env done)
    seq = [(0, 1.0, 0.5), (0, 0.0, 0.2), (1, 2.0, -0.3),
           (1, 0.0, 0.1), (0, -1.0, 0.4), (1, 5.0, 0.7)]
    buf = RolloutBuffer(gamma=GAMMA, gae_lambda=LAM)
    ts = [_t(p, r, v) for p, r, v in seq]
    ts[-1].done = True
    ep = Episode(0)
    for t in ts:
        ep.record(t, Phase.GUESS.value, None)
        buf.add(t)
    ep.mark_terminal()

    returns, adv = buf.compute_returns_and_advantages()

    for pid in (0, 1):
        idx = [i for i, (p, _, _) in enumerate(seq) if p == pid]
        exp = _gae([seq[i][1] for i in idx], [seq[i][2] for i in idx])
        np.testing.assert_allclose(adv[idx], exp, rtol=1e-5)
        np.testing.assert_allclose(returns[idx], np.array(exp) + [seq[i][2] for i in idx], rtol=1e-5)


def test_player_stream_does_not_bootstrap_into_next_game():
    # p1 ends game 1, but p0's last move must still end p0's stream.
    buf = RolloutBuffer(gamma=GAMMA, gae_lambda=LAM)
    g1 = [_t(0, 1.0, 0.0), _t(1, 0.0, 0.0, done=True)]
    g2 = [_t(0, 0.0, 100.0), _t(1, 0.0, 0.0, done=True)]
    for game in (g1, g2):
        ep = Episode(0)
        for t in game:
            ep.record(t, Phase.GUESS.value, None)
            buf.add(t)
        ep.mark_terminal()
    _, adv = buf.compute_returns_and_advantages()
    assert adv[0] == pytest.approx(1.0)  # r=1, terminal: no 0.9*100 from game 2


def test_mark_terminal_flags_each_players_last_move():
    ts = [_t(0, 0, 0), _t(1, 0, 0), _t(0, 0, 0), _t(1, 0, 0), _t(1, 0, 0)]
    ep = Episode(0)
    for t in ts:
        ep.record(t, Phase.GUESS.value, None)
    ep.mark_terminal()
    assert [t.stream_end for t in ts] == [False, False, True, False, True]
    assert all(not t.done for t in ts)  # env meaning of `done` untouched


@pytest.fixture(scope="module")
def rollout_buffer():
    cfg = PPOConfig(n_envs=6, episodes_per_update=6, n_workers=2)
    tr = PPOTrainer(cfg, torch.device("cpu"))
    tr.collect_rollouts()
    return tr.buffer.transitions


def test_rollout_keeps_only_finished_games(rollout_buffer):
    streams = defaultdict(list)
    for t in rollout_buffer:
        streams[(t.env_id, t.player_id)].append(t)
    assert streams
    for ts in streams.values():
        last = ts[-1]
        assert last.done or last.stream_end, "stream ends mid-game (unfinished game kept)"


def test_rollout_every_game_ends_both_streams(rollout_buffer):
    per_env = defaultdict(list)
    for t in rollout_buffer:
        per_env[t.env_id].append(t)
    for ts in per_env.values():
        game = []
        for t in ts:
            game.append(t)
            if t.done:
                ends = {g.player_id for g in game if g.done or g.stream_end}
                assert ends == {g.player_id for g in game}
                game = []
        assert not game, "transitions after the last finished game"
