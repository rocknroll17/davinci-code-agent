#!/usr/bin/env python3
"""
Export the trained world model to ONNX for the zero-backend browser demo.

v2 layout — three graphs (categorical z sampling between step/actor happens
in JS):

  wm_encoder.onnx  observation -> embed
      mlp encoder:         obs_vec (1, 643) float
      transformer encoder: my_color/my_value/opp_color/opp_value (1,13) int64
                           (remapped in JS: color NONE->2; value HIDDEN->13,
                           NONE->14), cm (1,13,13) f32 raw, phase (1,4) f32,
                           deck_symlog (1,2) f32
  wm_step.onnx     (embed, h, z_prev, a_prev_onehot) -> (h_new, post_probs)
                   GRUCell decomposed into explicit gate math.
  wm_actor.onnx    (h, z) -> five per-head logit vectors
  wm_meta.json     {"version": 2, "encoder": ..., dims...}

Weights are embedded in each .onnx (onnxruntime-web cannot read external
data sidecars).

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

from src.wm.nets import (ACTION_DIM, Actor, ObsEncoder, TransformerObsEncoder,
                         WMConfig, WorldModel)


class EncoderMLPGraph(nn.Module):
    def __init__(self, enc: ObsEncoder) -> None:
        super().__init__()
        self.net = enc.net

    def forward(self, obs_vec):
        return self.net(obs_vec)


class EncoderTFGraph(nn.Module):
    """TransformerObsEncoder with pre-remapped index inputs (JS does the remap
    and the deck symlog; everything else — embeddings, padding mask from
    color==NONE, self-attention, CLS readout — stays in-graph)."""

    def __init__(self, enc: TransformerObsEncoder) -> None:
        super().__init__()
        self.enc = enc

    def forward(self, my_c, my_v, opp_c, opp_v, cm, phase, deck_symlog):
        e = self.enc
        N = my_c.shape[0]
        slots = e.slot_embed.weight                      # (13, td)
        seg = e.segment_embed.weight                     # (5, td)
        my_tok = e.color_embed(my_c) + e.value_embed(my_v) + slots + seg[0]
        opp_tok = e.color_embed(opp_c) + e.value_embed(opp_v) + slots + seg[1]
        con_tok = e.constraint_proj(cm) + slots + seg[2]
        phase_tok = (e.phase_proj(phase) + seg[3]).unsqueeze(1)
        deck_tok = (e.deck_proj(deck_symlog) + seg[4]).unsqueeze(1)
        cls = e.cls_token.expand(N, -1, -1)
        tokens = torch.cat([cls, my_tok, opp_tok, con_tok, phase_tok, deck_tok], dim=1)
        no_pad = torch.zeros(N, 1, dtype=torch.bool, device=my_c.device)
        pad = torch.cat([no_pad, my_c == 2, opp_c == 2, opp_c == 2, no_pad, no_pad], dim=1)
        out = e.transformer(tokens, src_key_padding_mask=pad)
        return e.out(out[:, 0])


class StepGraph(nn.Module):
    """(embed, h, z_prev, a_prev_onehot) -> (h_new, post_probs)."""

    def __init__(self, wm: WorldModel) -> None:
        super().__init__()
        self.gru_in = wm.rssm.gru_in
        self.post_net = wm.rssm.post_net
        self.cfg = wm.cfg
        gru = wm.rssm.gru
        self.w_ih = nn.Parameter(gru.weight_ih.data.clone(), requires_grad=False)
        self.w_hh = nn.Parameter(gru.weight_hh.data.clone(), requires_grad=False)
        self.b_ih = nn.Parameter(gru.bias_ih.data.clone(), requires_grad=False)
        self.b_hh = nn.Parameter(gru.bias_hh.data.clone(), requires_grad=False)

    def forward(self, embed, h, z_prev, a_prev):
        x = self.gru_in(torch.cat([z_prev, a_prev], dim=-1))
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
    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self.trunk = actor.trunk
        self.heads = actor.heads

    def forward(self, h, z):
        f = F.silu(self.trunk(torch.cat([h, z], dim=-1)))
        return (self.heads["color"](f), self.heads["position"](f),
                self.heads["value"](f), self.heads["decision"](f),
                self.heads["joker"](f))


def _embed_weights(out_dir: str, names) -> None:
    """Re-save each graph with weights embedded (no *.data sidecars)."""
    import onnx
    for name in names:
        p = os.path.join(out_dir, name)
        m = onnx.load(p)
        onnx.save(m, p, save_as_external_data=False)
        sidecar = p + ".data"
        if os.path.exists(sidecar):
            os.remove(sidecar)


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
    wm = WorldModel(cfg)
    missing, _ = wm.load_state_dict(ckpt["wm"], strict=False)
    if missing:
        print(f"warning — fresh params (old checkpoint): {missing}")
    actor = Actor(cfg)
    actor.load_state_dict(ckpt["actor"])
    wm.eval(), actor.eval()

    step = StepGraph(wm).eval()
    act = ActorGraph(actor).eval()

    h = torch.zeros(1, cfg.deter_dim)
    z = torch.zeros(1, cfg.stoch_dim)
    a = torch.zeros(1, ACTION_DIM)
    embed = torch.zeros(1, cfg.embed_dim)

    if cfg.encoder == "transformer":
        enc_graph = EncoderTFGraph(wm.encoder).eval()
        # NOTE: each example input must be a DISTINCT tensor — the exporter
        # deduplicates identical tensor objects and scrambles the input mapping
        enc_inputs = (torch.zeros(1, 13, dtype=torch.long),
                      torch.ones(1, 13, dtype=torch.long),
                      torch.full((1, 13), 2, dtype=torch.long),
                      torch.full((1, 13), 3, dtype=torch.long),
                      torch.zeros(1, 13, 13), torch.zeros(1, 4), torch.zeros(1, 2))
        enc_input_names = ["my_color", "my_value", "opp_color", "opp_value",
                           "cm", "phase", "deck_symlog"]
    else:
        enc_graph = EncoderMLPGraph(wm.encoder).eval()
        enc_inputs = (torch.zeros(1, ObsEncoder.IN_DIM),)
        enc_input_names = ["obs_vec"]

    torch.onnx.export(enc_graph, enc_inputs, os.path.join(out_dir, "wm_encoder.onnx"),
                      input_names=enc_input_names, output_names=["embed"],
                      opset_version=18)
    torch.onnx.export(step, (embed, h, z, a), os.path.join(out_dir, "wm_step.onnx"),
                      input_names=["embed", "h", "z_prev", "a_prev"],
                      output_names=["h_new", "post_probs"], opset_version=18)
    torch.onnx.export(act, (h, z), os.path.join(out_dir, "wm_actor.onnx"),
                      input_names=["h", "z"],
                      output_names=["color_logits", "position_logits", "value_logits",
                                    "decision_logits", "joker_logits"], opset_version=18)
    _embed_weights(out_dir, ("wm_encoder.onnx", "wm_step.onnx", "wm_actor.onnx"))

    meta = {
        "version": 2,
        "encoder": cfg.encoder,
        "obs_dim": ObsEncoder.IN_DIM,
        "embed_dim": cfg.embed_dim,
        "deter_dim": cfg.deter_dim,
        "stoch_discrete": cfg.stoch_discrete,
        "stoch_classes": cfg.stoch_classes,
        "stoch_dim": cfg.stoch_dim,
        "action_dim": ACTION_DIM,
        "action_sizes": [2, 13, 13, 2, 13],
        "env_steps": int(ckpt.get("total_env_steps", 0)),
    }
    with open(os.path.join(out_dir, "wm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- numerical verification vs torch ----
    import numpy as np
    import onnxruntime as ort_rt
    torch.manual_seed(0)

    # a realistic random observation
    my_hand = torch.randint(-2, 13, (1, 13, 2)).float()
    opp_hand = torch.randint(-2, 13, (1, 13, 2)).float()
    obs = {"phase": torch.eye(4)[1:2], "my_hand": my_hand, "opponent_hand": opp_hand,
           "constraint_matrix": torch.randint(-1, 2, (1, 13, 13)).float(),
           "remaining_deck": torch.tensor([[7.0, 9.0]])}
    with torch.no_grad():
        t_embed = wm.encoder(obs)

    s0 = ort_rt.InferenceSession(os.path.join(out_dir, "wm_encoder.onnx"))
    if cfg.encoder == "transformer":
        from src.wm.nets import _hand_indices, symlog
        mc, mv = _hand_indices(my_hand)
        oc, ov = _hand_indices(opp_hand)
        feeds = {"my_color": mc.numpy(), "my_value": mv.numpy(),
                 "opp_color": oc.numpy(), "opp_value": ov.numpy(),
                 "cm": obs["constraint_matrix"].numpy(),
                 "phase": obs["phase"].numpy(),
                 "deck_symlog": symlog(obs["remaining_deck"]).numpy()}
    else:
        from src.wm.nets import _hand_onehot, symlog
        vec = torch.cat([_hand_onehot(my_hand), _hand_onehot(opp_hand),
                         obs["constraint_matrix"].flatten(1),
                         obs["phase"], symlog(obs["remaining_deck"])], -1)
        feeds = {"obs_vec": vec.numpy()}
    (o_embed,) = s0.run(None, feeds)
    err_e = float(np.abs(o_embed - t_embed.numpy()).max())

    hv = torch.randn(1, cfg.deter_dim)
    zv = torch.zeros(1, cfg.stoch_dim)
    zv[0, ::cfg.stoch_classes] = 1.0
    av = torch.zeros(1, ACTION_DIM); av[0, 0] = 1.0
    with torch.no_grad():
        th, tp = step(t_embed, hv, zv, av)
        ta = act(th, tp.flatten(1))
    s1 = ort_rt.InferenceSession(os.path.join(out_dir, "wm_step.onnx"))
    oh, op = s1.run(None, {"embed": o_embed, "h": hv.numpy(),
                           "z_prev": zv.numpy(), "a_prev": av.numpy()})
    s2 = ort_rt.InferenceSession(os.path.join(out_dir, "wm_actor.onnx"))
    oa = s2.run(None, {"h": oh, "z": op.reshape(1, -1)})
    err_h = float(np.abs(oh - th.numpy()).max())
    err_a = max(float(np.abs(o - t.numpy()).max()) for o, t in zip(oa, ta))
    print(f"max abs err — embed: {err_e:.2e}, h: {err_h:.2e}, actor: {err_a:.2e}")
    assert err_e < 1e-3 and err_h < 1e-3 and err_a < 1e-3
    sizes = {f: os.path.getsize(os.path.join(out_dir, f)) // 1024
             for f in ("wm_encoder.onnx", "wm_step.onnx", "wm_actor.onnx")}
    print(f"✓ exported ({cfg.encoder} encoder, {meta['env_steps']:,} steps) → {out_dir} {sizes}")


if __name__ == "__main__":
    main()
