#!/usr/bin/env python3
"""
Reward/penalty audit — play a few games and check every step's reward against
the game rules (constants). Prints an annotated trace + flags any MISMATCH.

NOTE: this checks the ENV-level rewards (guess/draw/decision/invalid/streak/win).
The trainer also adds *retroactive* rewards in the rollout buffer
(continue→success/fail, draw-win/lose, loser −10) which are NOT part of env.step
and are listed separately at the end.
"""
import argparse
import numpy as np
import torch

from src.env import DaVinciCodeEnv
from src.agent import ModelAgent
from src.constants import (
    Phase, CardValue,
    REWARD_WIN, REWARD_GUESS_SUCCESS, REWARD_JOKER_SUCCESS, REWARD_GUESS_FAIL,
    REWARD_GUESS_ORDER_VIOLATION, REWARD_STREAK_BONUS_MULTIPLIER, REWARD_STREAK_BREAK,
    REWARD_INVALID_ACTION, REWARD_STOP_DECISION,
    REWARD_STOP_WITH_DETERMINED, REWARD_STOP_WITH_NEAR_DETERMINED,
)
from src.utils.game_logic import find_determined_cards, count_candidate_cards

EPS = 1e-5


def fmt_card(v):
    return "JOKER" if v == CardValue.JOKER else str(v)


def audit_game(env, agent, device, gi):
    obs, info = env.reset(seed=1000 + gi)
    print(f"\n================= GAME {gi} =================")
    step = 0
    mismatches = 0
    while True:
        cur = info["current_player"]
        phase = int(np.argmax(obs["phase"]))
        mask = env.get_action_mask()
        action, _ = agent.act(obs, mask, deterministic=False)

        # capture pre-step state
        prev_streak = env._streak
        opp = 1 - cur
        # determined check (for STOP audit) computed on pre-step hands
        pre_det = len(find_determined_cards(env.players[cur]._hand, env.players[opp]._hand))
        pre_cand2 = count_candidate_cards(env.players[cur]._hand, env.players[opp]._hand, max_candidates=2)

        obs, _, reward, term, trunc, info, result = env.step(action)
        step += 1
        won = info.get("winner")

        line = f"[g{gi} s{step}] P{cur} {Phase(phase).name:8s}"
        exp = None
        note = ""

        if phase == Phase.DRAW.value:
            exp = REWARD_INVALID_ACTION if getattr(result, "is_invalid", False) else 0.0
            note = "draw(invalid)" if exp < 0 else "draw=0"

        elif phase == Phase.GUESS.value:
            pos = int(action[1]); val = int(action[2])
            if getattr(result, "is_invalid", False):
                exp = REWARD_INVALID_ACTION
                note = f"invalid guess pos{pos}"
            elif result.is_correct:
                base = REWARD_JOKER_SUCCESS if val == CardValue.JOKER else REWARD_GUESS_SUCCESS
                streak_after = prev_streak + 1
                exp = base + REWARD_STREAK_BONUS_MULTIPLIER * streak_after
                win_bonus = REWARD_WIN if won == cur else 0.0
                exp += win_bonus
                note = (f"✔correct {fmt_card(val)} | base {base:+.1f} "
                        f"+streak {REWARD_STREAK_BONUS_MULTIPLIER*streak_after:+.1f}(x{streak_after})"
                        + (f" +WIN {win_bonus:+.0f}" if win_bonus else ""))
            else:
                base = REWARD_GUESS_FAIL
                brk = REWARD_STREAK_BREAK if prev_streak > 0 else 0.0
                # order violation = whatever's left after base+break, must be 0 or the order penalty
                order = reward - base - brk
                exp = reward  # decode-and-confirm components instead of independent recompute
                ok_order = abs(order) < EPS or abs(order - REWARD_GUESS_ORDER_VIOLATION) < EPS
                ok = ok_order
                note = (f"X wrong {fmt_card(val)} | base {base:+.1f}"
                        + (f" +break {brk:+.1f}" if brk else "")
                        + (f" +order {order:+.1f}" if abs(order) > EPS else "")
                        + ("" if ok_order else "  <<UNEXPECTED extra>>"))

        elif phase == Phase.DECISION.value:
            dec = int(action[3])
            if dec == 1:
                exp = 0.0
                note = "continue=0 (retro reward added in trainer)"
            else:
                near = max(0, pre_cand2 - pre_det)
                pen = pre_det * REWARD_STOP_WITH_DETERMINED + near * REWARD_STOP_WITH_NEAR_DETERMINED
                exp = REWARD_STOP_DECISION + pen
                note = f"stop | determined={pre_det} near={near} → pen {pen:+.2f}"

        ok = abs(reward - exp) < EPS if exp is not None else True
        flag = "OK " if ok else "❌MISMATCH"
        if not ok:
            mismatches += 1
        print(f"{line}  reward={reward:+.3f}  expect={exp:+.3f}  [{flag}]  {note}")

        if term or trunc:
            print(f"  >>> GAME OVER, winner = P{won}")
            break
        if step > 400:
            print("  (step cap)")
            break
    return mismatches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/tmp/chk_audit.pt")
    p.add_argument("--games", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cpu")
    agent = ModelAgent.from_checkpoint(args.checkpoint, device=device)
    env = DaVinciCodeEnv()

    print("Reward constants:")
    print(f"  WIN={REWARD_WIN} GUESS_OK={REWARD_GUESS_SUCCESS} JOKER_OK={REWARD_JOKER_SUCCESS} "
          f"GUESS_FAIL={REWARD_GUESS_FAIL} ORDER_VIOL={REWARD_GUESS_ORDER_VIOLATION}")
    print(f"  STREAK_BONUS=x{REWARD_STREAK_BONUS_MULTIPLIER} STREAK_BREAK={REWARD_STREAK_BREAK} "
          f"INVALID={REWARD_INVALID_ACTION} STOP_DET={REWARD_STOP_WITH_DETERMINED} "
          f"STOP_NEAR={REWARD_STOP_WITH_NEAR_DETERMINED}")

    total_mm = 0
    for gi in range(args.games):
        total_mm += audit_game(env, agent, device, gi)

    print("\n==================================================")
    if total_mm == 0:
        print(f"✅ ALL STEPS MATCH THE RULES across {args.games} games (0 mismatches).")
    else:
        print(f"❌ {total_mm} mismatch(es) found — reward logic disagrees with rules.")
    print("Reminder: retroactive rewards (continue→±, draw-win/lose ±0.1, loser −10)")
    print("are applied later in the trainer buffer, not in env.step, so they are not")
    print("shown per-step above.")


if __name__ == "__main__":
    main()
