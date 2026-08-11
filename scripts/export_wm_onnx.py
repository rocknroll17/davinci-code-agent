#!/usr/bin/env python3
"""
Export the trained world model to ONNX for the zero-backend browser demo.

Two graphs (the categorical z sampling between them happens in JS):

  wm_step.onnx   (obs_vec, h, z_prev, a_prev_onehot) -> (h_new, post_probs)
      obs_vec: the 643-float encoder input, built in JS:
        [ my_hand one-hot 13*(color3+value15) | opp_hand same | constraint 13*13
          raw (-1/0/1) | phase one-hot 4 | symlog(deck) 2 ]
      GRUCell is decomposed into explicit gate math (no aten::gru_cell in ONNX).

  wm_actor.onnx  (h, z) -> (color2, position13, value13, decision2, joker13)

  wm_meta.json   dims + layout doc for the JS side.

Run:  PYTHONPATH=$PWD python scripts/export_wm_onnx.py \
          --ckpt ~/dreamer_latest.pt --out ~/davinci-code-server/docs
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wm.nets import ACTION_DIM, Actor, ObsEncoder, WMConfig, WorldModel


class StepGraph(nn.Module):
    """(obs_vec, h, z_prev, a_prev_onehot) -> (h_new, post_probs)."""

    def __init__(self, wm: WorldModel) -> None:
        super().__init__()
        self.encoder_net = wm.encoder.net
        self.gru_in = wm.rssm.gru_in
        self.post_net = wm.rssm.post_net
        self.cfg = wm.cfg
        gru = wm.rssm.gru
        self.w_ih = nn.Parameter(gru.weight_ih.data.clone(), requires_grad=False)
        self.w_hh = nn.Parameter(gru.weight_hh.data.clone(), requires_grad=False)
        self.b_ih = nn.Parameter(gru.bias_ih.data.clone(), requires_grad=False)
        self.b_hh = nn.Parameter(gru.bias_hh.data.clone(), requires_grad=False)

    def forward(self, obs_vec, h, z_prev, a_prev):
        embed = self.encoder_net(obs_vec)
        x = self.gru_in(torch.cat([z_prev, a_prev], dim=-1))
        # GRUCell decomposed (matches torch.nn.GRUCell exactly)
        gi = x @ self.w_ih.t() + self.b_ih
        gh = h @ self.w_hh.t() + self.b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        u = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1.0 - u) * n + u * h

        logits = self.post_net(torch.cat([h_new, embed], dim=-1))
        logits = logits.view(-1, self.cfg.stoch_discrete, self.cfg.stoch_classes)
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.cfg.stoch_classes
        probs = (1 - self.cfg.unimix) * probs + self.cfg.unimix * uniform
        return h_new, probs


class ActorGraph(nn.Module):
    """(h, z) -> five per-head logit vectors."""

    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self.trunk = actor.trunk
        self.heads = actor.heads

    def forward(self, h, z):
        f = F.silu(self.trunk(torch.cat([h, z], dim=-1)))
        return (self.heads["color"](f), self.heads["position"](f),
                self.heads["value"](f), self.heads["decision"](f),
                self.heads["joker"](f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="~/dreamer_latest.pt")
    ap.add_argument("--out", default="~/davinci-code-server/docs")
    args = ap.parse_args()

    ckpt_path = os.path.expanduser(args.ckpt)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = WMConfig(**dict(ckpt["config"]["wm"]))
    if getattr(cfg, "encoder", "mlp") != "mlp":
        raise SystemExit(
            "This exporter currently supports the MLP encoder only; the "
            "transformer-encoder browser export is a separate follow-up "
            "(needs a tokenized-input graph + demo JS feed changes).")
    wm = WorldModel(cfg)
    missing, _ = wm.load_state_dict(ckpt["wm"], strict=False)
    if missing:
        print(f"warning — fresh params (old checkpoint): {missing}")
    actor = Actor(cfg)
    actor.load_state_dict(ckpt["actor"])
    wm.eval(), actor.eval()

    step = StepGraph(wm).eval()
    act = ActorGraph(actor).eval()

    obs_vec = torch.zeros(1, ObsEncoder.IN_DIM)
    h = torch.zeros(1, cfg.deter_dim)
    z = torch.zeros(1, cfg.stoch_dim)
    a = torch.zeros(1, ACTION_DIM)

    torch.onnx.export(
        step, (obs_vec, h, z, a), os.path.join(out_dir, "wm_step.onnx"),
        input_names=["obs_vec", "h", "z_prev", "a_prev"],
        output_names=["h_new", "post_probs"], opset_version=18)
    torch.onnx.export(
        act, (h, z), os.path.join(out_dir, "wm_actor.onnx"),
        input_names=["h", "z"],
        output_names=["color_logits", "position_logits", "value_logits",
                      "decision_logits", "joker_logits"], opset_version=18)

    # onnxruntime-web cannot read external-data files ("Module.MountedFiles is
    # not available") — re-save each graph with the weights embedded in the
    # single .onnx file and drop the sidecar *.data.
    import onnx
    for name in ("wm_step.onnx", "wm_actor.onnx"):
        p = os.path.join(out_dir, name)
        m = onnx.load(p)  # pulls external data into memory
        onnx.save(m, p, save_as_external_data=False)
        sidecar = p + ".data"
        if os.path.exists(sidecar):
            os.remove(sidecar)

    meta = {
        "obs_dim": ObsEncoder.IN_DIM,
        "deter_dim": cfg.deter_dim,
        "stoch_discrete": cfg.stoch_discrete,
        "stoch_classes": cfg.stoch_classes,
        "stoch_dim": cfg.stoch_dim,
        "action_dim": ACTION_DIM,
        "action_sizes": [2, 13, 13, 2, 13],
        "env_steps": int(ckpt.get("total_env_steps", 0)),
        "obs_layout": "my13*(c3+v15) | opp13*(c3+v15) | cm13*13 raw | phase4 | symlog(deck)2",
    }
    with open(os.path.join(out_dir, "wm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- numerical verification vs torch ----
    try:
        import onnxruntime as ort_rt
    except ImportError:
        print("onnxruntime not installed — skipping numeric check")
        return
    torch.manual_seed(0)
    ov = torch.randn(1, ObsEncoder.IN_DIM)
    hv = torch.randn(1, cfg.deter_dim)
    zv = torch.zeros(1, cfg.stoch_dim)
    zv[0, ::cfg.stoch_classes] = 1.0  # a valid one-hot-per-categorical z
    av = torch.zeros(1, ACTION_DIM)
    av[0, 0] = 1.0
    with torch.no_grad():
        th, tp = step(ov, hv, zv, av)
        ta = act(th, tp.flatten(1))
    s1 = ort_rt.InferenceSession(os.path.join(out_dir, "wm_step.onnx"))
    oh, op = s1.run(None, {"obs_vec": ov.numpy(), "h": hv.numpy(),
                           "z_prev": zv.numpy(), "a_prev": av.numpy()})
    s2 = ort_rt.InferenceSession(os.path.join(out_dir, "wm_actor.onnx"))
    oa = s2.run(None, {"h": oh, "z": op.reshape(1, -1)})
    import numpy as np
    err_h = float(np.abs(oh - th.numpy()).max())
    err_p = float(np.abs(op - tp.numpy()).max())
    err_a = max(float(np.abs(o - t.numpy()).max()) for o, t in zip(oa, ta))
    print(f"max abs err — h: {err_h:.2e}, post_probs: {err_p:.2e}, actor: {err_a:.2e}")
    assert err_h < 1e-4 and err_p < 1e-4 and err_a < 1e-4
    sizes = {f: os.path.getsize(os.path.join(out_dir, f)) // 1024
             for f in ("wm_step.onnx", "wm_actor.onnx")}
    print(f"✓ exported to {out_dir} ({sizes})")


if __name__ == "__main__":
    main()
