"""The ordering-violation penalty must actually FIRE for an out-of-range guess
(test_rewards only checks it's 0 OR the penalty; this proves it triggers)."""
from src.cards.black_card import BlackCard
from src.constants import Color
from src.env import DaVinciCodeEnv
from src.hand import Hand


def _opp_hand_with_revealed_neighbors():
    """Opponent hand: B3(revealed) | B7(hidden, pos1) | B11(revealed)."""
    h = Hand()
    h.add_initial_cards([BlackCard(3), BlackCard(7), BlackCard(11)])
    # reveal the two neighbors (positions 0 and 2 after sorted insert 3,7,11)
    h.reveal_card(0)
    h.reveal_card(2)
    return h


def test_order_violation_fires_for_out_of_range_guess():
    env = DaVinciCodeEnv()
    env.reset()
    cur = env._current_player
    opp = 1 - cur
    env.players[opp]._hand = _opp_hand_with_revealed_neighbors()
    env._phase.draw_done()   # advance DRAW -> GUESS so the handler's phase step is valid
    rc = env._rc

    # Guess the hidden middle card (pos 1) with value 1 (< left neighbor 3) → out of range.
    reward = env._handle_guess_failure(position=1, guessed_value=1, card_color=Color.BLACK)
    # streak was 0, so reward = guess_fail + order_violation
    assert abs(reward - (rc.guess_fail + rc.guess_order_violation)) < 1e-9, \
        f"out-of-range guess should incur order violation; got {reward}"


def test_no_order_violation_for_in_range_guess():
    env = DaVinciCodeEnv()
    env.reset()
    cur = env._current_player
    opp = 1 - cur
    env.players[opp]._hand = _opp_hand_with_revealed_neighbors()
    env._phase.draw_done()   # advance DRAW -> GUESS so the handler's phase step is valid
    rc = env._rc

    # Guess value 9 (between 3 and 11) → in range, no violation.
    reward = env._handle_guess_failure(position=1, guessed_value=9, card_color=Color.BLACK)
    assert abs(reward - rc.guess_fail) < 1e-9, f"in-range wrong guess should be plain fail; got {reward}"
