"""Direct unit tests for the determined/candidate logic (breaks the circularity
where test_rewards re-derives STOP penalty from the same functions it tests)."""
from src.hand import Hand
from src.cards.black_card import BlackCard
from src.cards.white_card import WhiteCard
from src.utils.game_logic import find_determined_cards, count_candidate_cards


def _hand(cards):
    h = Hand()
    h.add_initial_cards(cards)
    return h


def test_forced_value_is_determined():
    """If I hold every BLACK number 0..11, the opponent's one hidden BLACK card
    can only be the BLACK joker — must be reported determined as value 12."""
    my = _hand([BlackCard(v) for v in range(12)])     # black 0..11 (all known to me)
    opp = _hand([BlackCard(12)])                       # one hidden black card (the joker)
    det = find_determined_cards(my, opp)
    assert det == [(0, 12)], f"expected the hidden card forced to joker, got {det}"


def test_open_value_is_not_determined():
    """With almost no information, the opponent's hidden card is NOT determined."""
    my = _hand([BlackCard(0), WhiteCard(5)])
    opp = _hand([BlackCard(7)])                        # could be many black values
    det = find_determined_cards(my, opp)
    assert det == [], f"should not be determined, got {det}"


def test_count_candidates_determined_le_one():
    """A forced card has <=1 candidate; count_candidate_cards must count it."""
    my = _hand([BlackCard(v) for v in range(12)])
    opp = _hand([BlackCard(12)])
    n = count_candidate_cards(my, opp, max_candidates=2)
    assert n >= 1, "the forced (determined) card should be counted among <=2-candidate cards"


def test_revealed_card_not_counted_as_hidden():
    """A revealed opponent card is not a hidden/unknown position."""
    my = _hand([BlackCard(0)])
    opp = _hand([WhiteCard(3)])
    opp.reveal_card(0)
    det = find_determined_cards(my, opp)
    assert det == [], "revealed cards must not appear as determined hidden cards"
