"""Every env step's reward must match the rule re-derivation (RewardConfig)."""
import numpy as np
import pytest

from src.env import DaVinciCodeEnv
from src.reward_config import RewardConfig
from src.constants import Phase, CardValue
from src.utils.game_logic import find_determined_cards, count_candidate_cards

EPS = 1e-6


def _play_and_check(env, agent, n_games=8):
    rc = env._rc
    for g in range(n_games):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            cur = info["current_player"]
            phase = int(np.argmax(obs["phase"]))
            mask = env.get_action_mask()
            action, _ = agent.act(obs, mask, deterministic=False)
            prev_streak = env._streak
            opp = 1 - cur
            pre_det = len(find_determined_cards(env.players[cur]._hand, env.players[opp]._hand))
            pre_cand2 = count_candidate_cards(env.players[cur]._hand, env.players[opp]._hand, max_candidates=2)

            obs, _r, reward, term, trunc, info, result = env.step(action)
            done = term or trunc
            steps += 1
            won = info.get("winner")

            if phase == Phase.DRAW.value:
                exp = rc.invalid_action if getattr(result, "is_invalid", False) else 0.0
                assert abs(reward - exp) < EPS, f"DRAW reward {reward} != {exp}"

            elif phase == Phase.GUESS.value:
                val = int(action[2])
                if getattr(result, "is_invalid", False):
                    assert abs(reward - rc.invalid_action) < EPS
                elif result.is_correct:
                    base = rc.joker_success if val == CardValue.JOKER else rc.guess_success
                    exp = base + rc.streak_bonus_multiplier * (prev_streak + 1)
                    if won == cur:
                        exp += rc.win
                    assert abs(reward - exp) < EPS, f"correct guess {reward} != {exp}"
                else:
                    extra = reward - rc.guess_fail
                    brk = rc.streak_break if prev_streak > 0 else 0.0
                    order = extra - brk
                    assert abs(order) < EPS or abs(order - rc.guess_order_violation) < EPS, \
                        f"wrong-guess extra {order} not in {{0, {rc.guess_order_violation}}}"

            elif phase == Phase.DECISION.value:
                dec = int(action[3])
                if dec == 1:
                    assert abs(reward) < EPS, "CONTINUE step reward must be 0 (retro added later)"
                else:
                    near = max(0, pre_cand2 - pre_det)
                    exp = rc.stop_decision + pre_det * rc.stop_with_determined + near * rc.stop_with_near_determined
                    assert abs(reward - exp) < EPS, f"STOP reward {reward} != {exp}"


def test_env_rewards_match_rules_default(make_agent):
    _play_and_check(DaVinciCodeEnv(), make_agent(1), n_games=8)


def test_env_rewards_match_rules_custom_config(make_agent):
    rc = RewardConfig(guess_success=2.0, guess_fail=-1.5, win=42.0, joker_success=3.0)
    _play_and_check(DaVinciCodeEnv(reward_config=rc), make_agent(2), n_games=8)
