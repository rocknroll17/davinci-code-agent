"""World model (src/wm): RSSM shapes, losses, imagination, and a full tiny loop."""

import numpy as np
import pytest
import torch

from src.wm.nets import (
    ACTION_SIZES, Actor, Critic, TwoHot, WMConfig, WorldModel, symexp, symlog,
)
from src.wm.dreamer import DreamerConfig, DreamerTrainer, WMAgent

TINY = WMConfig(deter_dim=32, stoch_discrete=4, stoch_classes=4,
                hidden=32, embed_dim=32, n_bins=63)


def test_symlog_roundtrip():
    x = torch.tensor([-25.0, -1.0, 0.0, 0.5, 30.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_twohot_roundtrip():
    th = TwoHot(n_bins=255)
    y = torch.tensor([-12.0, -0.7, 0.0, 3.3, 18.0])
    target = th.encode(y)
    assert torch.allclose(target.sum(-1), torch.ones(5), atol=1e-5)
    # expectation over the two-hot weights recovers y exactly in symlog space
    recovered = symexp((target * th.bins).sum(-1))
    assert torch.allclose(recovered, y, atol=1e-3)


def _fake_batch(B=3, T=6):
    obs = {
        "phase": torch.eye(4)[torch.randint(0, 4, (B, T))],
        "my_hand": torch.randint(-2, 13, (B, T, 13, 2)).float(),
        "opponent_hand": torch.randint(-2, 13, (B, T, 13, 2)).float(),
        "remaining_deck": torch.randint(0, 13, (B, T, 2)).float(),
        "constraint_matrix": torch.randint(-1, 2, (B, T, 13, 13)).float(),
    }
    actions = torch.stack(
        [torch.randint(0, n, (B, T)) for n in ACTION_SIZES], dim=-1)
    masks = {
        "color": torch.ones(B, T, 2, dtype=torch.bool),
        "position": torch.ones(B, T, 13, dtype=torch.bool),
        "value": torch.ones(B, T, 13, 13, dtype=torch.bool),
        "decision": torch.ones(B, T, 2, dtype=torch.bool),
        "joker": torch.ones(B, T, 13, dtype=torch.bool),
    }
    rewards = torch.randn(B, T)
    conts = torch.ones(B, T)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    valid = torch.ones(B, T)
    flips = (torch.rand(B, T) < 0.3).float()
    return obs, actions, rewards, conts, is_first, masks, valid, flips


def test_world_model_loss_finite_and_backward():
    torch.manual_seed(0)
    wm = WorldModel(TINY)
    obs, actions, rewards, conts, is_first, masks, valid, flips = _fake_batch()
    loss, states, metrics = wm.loss(obs, actions, rewards, conts, is_first,
                                    mask_seq=masks, valid=valid, flip_seq=flips)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in wm.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert states.shape == (3, 6, TINY.state_dim)
    assert metrics["wm/kl_dyn"] >= TINY.free_bits - 1e-5  # free bits floor


def test_imagination_shapes_and_actor_grad():
    torch.manual_seed(0)
    wm = WorldModel(TINY)
    actor = Actor(TINY)
    start = torch.randn(5, TINY.state_dim)
    with torch.no_grad():
        img = wm.imagine(actor, start, horizon=4)
    assert img["states"].shape == (5, 5, TINY.state_dim)
    assert img["actions"].shape == (4, 5, 5)
    assert img["rewards"].shape == (5, 5)
    assert img["flips"].shape == (5, 5)
    assert ((img["flips"] >= 0) & (img["flips"] <= 1)).all()
    # actions respect component ranges
    for i, n in enumerate(ACTION_SIZES):
        assert int(img["actions"][..., i].max()) < n

    # evaluate() attaches gradient to the actor
    lp, ent = actor.evaluate(img["states"][:-1].flatten(0, 1),
                             img["actions"].flatten(0, 1))
    lp.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in actor.parameters())


def test_replay_terminal_frame_and_flips():
    """Episodes carry flip flags and one terminal frame (dummy action, cont=0)."""
    from src.wm.replay import EpisodeAccumulator
    acc = EpisodeAccumulator()
    obs = {"phase": np.zeros(4), "my_hand": np.zeros((13, 2)),
           "opponent_hand": np.zeros((13, 2)), "remaining_deck": np.zeros(2),
           "constraint_matrix": np.zeros((13, 13))}
    masks = {"color": np.ones(2, bool), "position": np.ones(13, bool),
             "value": np.ones((13, 13), bool), "decision": np.ones(2, bool),
             "joker": np.ones(13, bool)}
    acc.add(obs, np.zeros(5), 0.5, False, masks, flip=0.0)
    acc.add(obs, np.zeros(5), -0.5, False, masks, flip=1.0)
    acc.add(obs, np.zeros(5), 10.0, True, masks, flip=0.0)
    acc.set_terminal(obs)
    ep = acc.pack()
    assert len(ep["actions"]) == 4                 # 3 steps + terminal frame
    assert ep["continues"][-1] == 0.0 and ep["rewards"][-1] == 0.0
    assert list(ep["flips"][:3]) == [0.0, 1.0, 0.0]
    # the terminal reward (win) sits at index -2, predicted from the terminal state
    assert ep["rewards"][-2] == 10.0


def test_full_dreamer_loop_tiny():
    """One real collect → WM update → imagination AC update, end to end."""
    torch.manual_seed(0)
    cfg = DreamerConfig(
        n_envs=2, seed=7, wm=TINY, seq_len=16, batch_size=4,
        prefill_episodes=0, episodes_per_round=2,
        wm_updates_per_round=2, ac_updates_per_round=2, horizon=5,
        save_dir="/tmp/wm_ckpt_test",
    )
    tr = DreamerTrainer(cfg, torch.device("cpu"))
    stats = tr.collect(2, random_actor=True)
    assert stats["collect/episodes"] == 2
    assert tr.replay.n_episodes >= 1

    wm_m = tr.train_world_model()
    assert np.isfinite(wm_m["wm/loss"])
    ac_m = tr.train_actor_critic()
    assert np.isfinite(ac_m["ac/actor_loss"]) and np.isfinite(ac_m["ac/critic_loss"])

    # save → load → WMAgent plays a real game step
    path = "/tmp/wm_ckpt_test/dreamer_test.pt"
    tr.save(path)
    agent = WMAgent.from_checkpoint(path)
    from src.env import DaVinciCodeEnv
    env = DaVinciCodeEnv(seed=1, viewer=None, joker_control=True)
    obs, _ = env.reset()
    for _ in range(30):
        action, _ = agent.act(obs, env.get_action_mask())
        obs, _, _, done, _, _, _ = env.step(action)
        if done:
            break


TINY_TF = WMConfig(deter_dim=32, stoch_discrete=4, stoch_classes=4,
                   hidden=32, embed_dim=32, n_bins=63,
                   encoder="transformer", token_dim=16, enc_layers=1, enc_heads=2)


def test_transformer_encoder_shapes_and_batch_time():
    from src.wm.nets import TransformerObsEncoder
    torch.manual_seed(0)
    enc = TransformerObsEncoder(TINY_TF)
    obs, *_ = _fake_batch(B=2, T=3)
    out = enc(obs)                       # (B, T, ...) input
    assert out.shape == (2, 3, TINY_TF.embed_dim)
    flat_obs = {k: v[:, 0] for k, v in obs.items()}
    out2 = enc(flat_obs)                 # (B, ...) input
    assert out2.shape == (2, TINY_TF.embed_dim)
    assert torch.allclose(out[:, 0], out2, atol=1e-5)


def test_belief_supervision_and_transformer_loss():
    torch.manual_seed(0)
    wm = WorldModel(TINY_TF)
    obs, actions, rewards, conts, is_first, masks, valid, flips = _fake_batch()
    hidden = torch.randint(-1, 13, (3, 6, 13))
    loss, states, metrics = wm.loss(obs, actions, rewards, conts, is_first,
                                    mask_seq=masks, valid=valid, flip_seq=flips,
                                    hidden_seq=hidden)
    assert torch.isfinite(loss)
    assert metrics["wm/belief"] > 0 and 0.0 <= metrics["wm/belief_acc"] <= 1.0
    loss.backward()
    # belief gradient reaches the encoder (deduction is forced into the latent)
    from src.wm.nets import TransformerObsEncoder
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in wm.encoder.parameters())


def test_full_loop_with_transformer_encoder():
    torch.manual_seed(0)
    cfg = DreamerConfig(
        n_envs=2, seed=9, wm=TINY_TF, seq_len=16, batch_size=4,
        prefill_episodes=0, episodes_per_round=2,
        wm_updates_per_round=2, ac_updates_per_round=2, horizon=5,
        save_dir="/tmp/wm_tf_test", eval_every=0,
    )
    tr = DreamerTrainer(cfg, torch.device("cpu"))
    tr.collect(2, random_actor=True)
    m = tr.train_world_model()
    assert np.isfinite(m["wm/loss"]) and m["wm/belief_acc"] >= 0.0
    ac = tr.train_actor_critic()
    assert np.isfinite(ac["ac/actor_loss"])
    # checkpoint roundtrip reconstructs the transformer encoder from config
    path = "/tmp/wm_tf_test/ck.pt"
    tr.save(path)
    agent = WMAgent.from_checkpoint(path)
    from src.wm.nets import TransformerObsEncoder
    assert isinstance(agent.wm.encoder, TransformerObsEncoder)
