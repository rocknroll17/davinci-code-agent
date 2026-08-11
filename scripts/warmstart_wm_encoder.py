#!/usr/bin/env python3
"""
Warm-start the world model's transformer encoder from the deployed PPO policy.

The deployed policy's encoder (src/model.py ObservationEncoder) already learned
constraint deduction — its transformer layers, shared slot embeddings and
constraint projection are the circuitry we want the world model to inherit.
The tokenizers differ (concat-proj vs sum-of-embeddings), so this transfers
exactly the compatible, deduction-carrying parts:

  transformer layers (all 4)      -> wm.encoder.transformer
  slot_pos_embed                  -> wm.encoder.slot_embed
  segment_embed (rows reordered)  -> wm.encoder.segment_embed
  constraint_proj first Linear    -> wm.encoder.constraint_proj
  phase_proj first Linear (3->4)  -> wm.encoder.phase_proj (JOKER column fresh)
  deck_proj first Linear          -> wm.encoder.deck_proj
  cls_token                       -> wm.encoder.cls_token

and leaves everything else (card embeddings, output proj, RSSM, heads,
actor/critic) freshly initialized. Output: a full Dreamer checkpoint that
train_wm.py --resume picks up.

Run (server):
  python scripts/warmstart_wm_encoder.py --large \
      --policy checkpoints/best_model_control.pt \
      --out checkpoints_wm/dreamer_latest.pt
  torchrun ... train_wm.py --large --encoder transformer --resume ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.wm.dreamer import DreamerConfig, DreamerTrainer
from src.wm.nets import WMConfig

# WM segment order: my, opp, con, phase, deck
# policy segment ids: my=0, opp=1, phase=2, deck=3, constraint=4
SEGMENT_PERM = [0, 1, 4, 2, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="checkpoints/best_model_control.pt")
    ap.add_argument("--out", default="checkpoints_wm/dreamer_latest.pt")
    ap.add_argument("--large", action="store_true",
                    help="use the --large WM dims (must match the training flags)")
    args = ap.parse_args()

    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False)
    pol = ckpt.get("policy_state_dict", ckpt)
    pol = {k.replace("._orig_mod.", "."): v for k, v in pol.items()}

    # Policy encoder geometry (fixed by the deployed architecture)
    token_dim = pol["encoder.slot_pos_embed.weight"].shape[1]
    n_pol_layers = 1 + max(int(k.split(".")[3]) for k in pol
                           if k.startswith("encoder.transformer.layers."))
    print(f"policy encoder: token_dim={token_dim}, layers={n_pol_layers}")

    if args.large:
        wm_cfg = WMConfig(deter_dim=1024, stoch_discrete=32, stoch_classes=32,
                          hidden=1024, embed_dim=1024,
                          encoder="transformer", token_dim=token_dim,
                          enc_layers=n_pol_layers, enc_heads=4)
    else:
        wm_cfg = WMConfig(encoder="transformer", token_dim=token_dim,
                          enc_layers=n_pol_layers, enc_heads=4)

    cfg = DreamerConfig(n_envs=2, wm=wm_cfg)  # n_envs only affects this process
    trainer = DreamerTrainer(cfg, torch.device("cpu"))
    enc = trainer.wm.encoder

    transferred = []

    def copy_(dst: torch.Tensor, src: torch.Tensor, name: str):
        assert dst.shape == src.shape, f"{name}: {dst.shape} vs {src.shape}"
        with torch.no_grad():
            dst.copy_(src)
        transferred.append(name)

    # transformer layers — identical module structure, direct load
    tf_state = {k[len("encoder.transformer."):]: v for k, v in pol.items()
                if k.startswith("encoder.transformer.")}
    enc.transformer.load_state_dict(tf_state)
    transferred.append(f"transformer ({n_pol_layers} layers)")

    copy_(enc.slot_embed.weight, pol["encoder.slot_pos_embed.weight"], "slot_embed")
    copy_(enc.segment_embed.weight,
          pol["encoder.segment_embed.weight"][SEGMENT_PERM], "segment_embed(reordered)")
    copy_(enc.cls_token, pol["encoder.cls_token"], "cls_token")
    copy_(enc.constraint_proj.weight, pol["encoder.constraint_proj.0.weight"],
          "constraint_proj.w")
    copy_(enc.constraint_proj.bias, pol["encoder.constraint_proj.0.bias"],
          "constraint_proj.b")
    # phase: policy is 3-dim, WM is 4-dim (JOKER) — copy the 3 shared columns
    with torch.no_grad():
        enc.phase_proj.weight[:, :3].copy_(pol["encoder.phase_proj.0.weight"])
        enc.phase_proj.bias.copy_(pol["encoder.phase_proj.0.bias"])
    transferred.append("phase_proj(3/4 cols)")
    copy_(enc.deck_proj.weight, pol["encoder.deck_proj.0.weight"], "deck_proj.w")
    copy_(enc.deck_proj.bias, pol["encoder.deck_proj.0.bias"], "deck_proj.b")

    # sanity forward
    obs = {
        "phase": torch.eye(4)[:1], "my_hand": torch.zeros(1, 13, 2),
        "opponent_hand": torch.zeros(1, 13, 2),
        "constraint_matrix": torch.zeros(1, 13, 13),
        "remaining_deck": torch.tensor([[13.0, 13.0]]),
    }
    out = enc(obs)
    assert torch.isfinite(out).all()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)
    trainer.vec_env.close()
    print("transferred:", ", ".join(transferred))
    print(f"✓ warm-started Dreamer checkpoint written to {args.out}")
    print("  resume with: train_wm.py " + ("--large " if args.large else "")
          + "--encoder transformer --resume")


if __name__ == "__main__":
    main()
