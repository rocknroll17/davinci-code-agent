#!/usr/bin/env python3
"""
Audit the *retroactive* rewards the trainer adds in the rollout buffer:
  - DRAW transitions    → +0.1 winner / -0.1 loser   (REWARD_DRAW_WIN/LOSE)
  - CONTINUE decision   → +0.4 if next guess correct, -0.25 if wrong
                          (REWARD_CONTINUE_SUCCESS/FAIL)
  - loser's last move   → -10                          (REWARD_LOSE)

Strategy: run ONE real trainer rollout (CPU, tiny), then read buffer.transitions.
DRAW and CONTINUE have env base reward 0, so the value left in the buffer IS the
retroactive reward → directly checkable.
"""
import numpy as np
import torch

from src.constants import (
    REWARD_CONTINUE_FAIL,
    REWARD_CONTINUE_SUCCESS,
    REWARD_DRAW_LOSE,
    REWARD_DRAW_WIN,
    REWARD_LOSE,
    Phase,
)
from src.trainer import PPOConfig, PPOTrainer

EPS = 1e-4


def main():
    cfg = PPOConfig(n_envs=8, episodes_per_update=8, n_workers=2)
    trainer = PPOTrainer(cfg, torch.device("cpu"))
    print("Collecting one rollout (8 episodes, random-init policy)…")
    trainer.collect_rollouts()
    tr = trainer.buffer.transitions
    print(f"buffer has {len(tr)} transitions\n")

    # reconstruct episodes per env_id (split on done)
    from collections import defaultdict
    per_env = defaultdict(list)
    for t in tr:
        per_env[t.env_id].append(t)
    episodes = []
    for env_id, ts in per_env.items():
        cur = []
        for t in ts:
            cur.append(t)
            if t.done:
                episodes.append(cur); cur = []
    print(f"reconstructed {len(episodes)} complete episodes\n")

    issues = []
    draw_ok = cont_ok = lose_ok = 0
    draw_vals, cont_vals = [], []

    for ei, ep in enumerate(episodes):
        # ---- DRAW retro: every draw transition must be +0.1 or -0.1 ----
        draws = [t for t in ep if np.argmax(t.obs["phase"]) == Phase.DRAW.value]
        by_player = defaultdict(list)
        for t in draws:
            draw_vals.append(round(t.reward, 4))
            if abs(t.reward - REWARD_DRAW_WIN) < EPS or abs(t.reward - REWARD_DRAW_LOSE) < EPS:
                draw_ok += 1
            else:
                issues.append(f"ep{ei}: DRAW reward {t.reward:+.3f} not ±0.1")
            by_player[t.player_id].append(round(t.reward, 3))
        # winner draws should be +0.1, loser -0.1 (opposite signs across players)
        signs = {p: (np.sign(np.mean(v)) if v else 0) for p, v in by_player.items()}
        if len(signs) == 2 and len(set(signs.values())) == 1 and 0 not in signs.values():
            issues.append(f"ep{ei}: both players have same-sign draw reward {signs}")

        # ---- CONTINUE retro: decision==CONTINUE → +0.4 or -0.25 ----
        for t in ep:
            if np.argmax(t.obs["phase"]) == Phase.DECISION.value and int(t.action[3]) == 1:
                cont_vals.append(round(t.reward, 4))
                if (abs(t.reward - REWARD_CONTINUE_SUCCESS) < EPS or
                        abs(t.reward - REWARD_CONTINUE_FAIL) < EPS or
                        abs(t.reward) < EPS):
                    cont_ok += 1
                else:
                    issues.append(f"ep{ei}: CONTINUE reward {t.reward:+.3f} not in {{+0.4,-0.25,0}}")

        # ---- LOSE -10: exactly one transition deep-negative (loser's last) ----
        big_neg = [t for t in ep if t.reward <= REWARD_LOSE + 1.0]      # <= -9
        big_pos = [t for t in ep if t.reward >= 9.5]                    # winner's +10 win move
        if len(big_neg) == 1:
            lose_ok += 1
        else:
            issues.append(f"ep{ei}: expected exactly 1 loser(-10) transition, found {len(big_neg)}")
        if len(big_pos) < 1:
            issues.append(f"ep{ei}: no winning(+10) transition found")

    print("=== retroactive reward audit ===")
    print(f"DRAW   transitions checked: {draw_ok} all ±0.1   | sample values: {sorted(set(draw_vals))[:6]}")
    print(f"CONTINUE transitions checked: {cont_ok}          | sample values: {sorted(set(cont_vals))}")
    print(f"LOSE   episodes with exactly one -10: {lose_ok}/{len(episodes)}")
    print()
    print(f"constants: DRAW_WIN={REWARD_DRAW_WIN} DRAW_LOSE={REWARD_DRAW_LOSE} "
          f"CONT_OK={REWARD_CONTINUE_SUCCESS} CONT_FAIL={REWARD_CONTINUE_FAIL} LOSE={REWARD_LOSE}")
    print()
    if not issues:
        print(f"✅ ALL RETROACTIVE REWARDS CORRECT across {len(episodes)} episodes (0 issues).")
    else:
        print(f"❌ {len(issues)} issue(s):")
        for s in issues[:30]:
            print("  -", s)


if __name__ == "__main__":
    main()
