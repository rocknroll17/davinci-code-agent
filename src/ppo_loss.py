"""
PPO loss terms, as pure functions (no trainer state).

    total = policy_loss + vf_coef * value_loss + belief_coef * belief_loss - ent_coef * entropy

B = minibatch size. Each action head only counts on the rows of its own phase.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.constants import MASK_VALUE, NUM_VALUES

# Phase column (DRAW=0, GUESS=1, DECISION=2) in which each action head acts
HEAD_PHASE = {"color": 0, "position": 1, "value": 1, "decision": 2}


def policy_loss(
    log_probs: Dict[str, torch.Tensor],
    old_log_probs: Dict[str, torch.Tensor],
    entropies: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    phase: torch.Tensor,
    clip_range: float,
    entropy_weight: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    """PPO clipped surrogate loss, summed over the action heads.

    log_probs / old_log_probs / entropies: {head: (B,)} — now vs. when the data was collected
    advantages: (B,), phase: (B, 3) one-hot
    entropy_weight: per-head multiplier for the entropy bonus (default 1)

    Returns (loss, mean entropy over active heads, approx_kl per head, clipfrac per head).
    """
    device = advantages.device
    loss = torch.tensor(0.0, device=device)
    entropy_sum = torch.tensor(0.0, device=device)
    n_heads = 0
    kls, clipfracs = [], []

    for key, phase_col in HEAD_PHASE.items():
        rows = phase[:, phase_col].bool()                                   # (B,)
        if not rows.any():
            continue
        n_heads += 1

        # ratio = pi_new / pi_old, computed in log space for stability
        log_ratio = log_probs[key][rows] - old_log_probs[key][rows]         # (N,)
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        # Take the pessimistic (smaller) of the unclipped and clipped objectives
        adv = advantages[rows]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
        loss = loss - torch.min(unclipped, clipped).mean()

        # Diagnostics (same as CleanRL): KL estimate and how often the clip was hit
        kls.append(((ratio - 1) - log_ratio).mean().item())
        clipfracs.append(((ratio - 1).abs() > clip_range).float().mean().item())

        head_entropy = entropies[key][rows].mean()
        entropy_sum = entropy_sum + entropy_weight.get(key, 1.0) * head_entropy

    if n_heads > 0:
        entropy = entropy_sum / n_heads
    else:
        entropy = torch.tensor(0.0, device=device)
    return loss, entropy, kls, clipfracs


def value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
               clip_range_vf=None) -> torch.Tensor:
    """0.5 * (V(s) - return)^2.

    values: (B,) now, old_values: (B,) when collected, returns: (B,) GAE targets.
    With clip_range_vf, V is also kept close to old_values (PPO-style); the larger
    of the two errors is used.
    """
    if clip_range_vf is None:
        return 0.5 * ((values - returns) ** 2).mean()
    clipped = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
    unclipped_error = (values - returns) ** 2
    clipped_error = (clipped - returns) ** 2
    return 0.5 * torch.max(unclipped_error, clipped_error).mean()


def belief_loss(belief_logits: torch.Tensor, hidden_values: torch.Tensor,
                my_hand: torch.Tensor, opponent_hand: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the belief head against the opponent's true hidden values.

    belief_logits: (B, 13 slots, 13 values)
    hidden_values: (B, 13) true value of each hidden opponent card, -1 if not hidden
    my_hand, opponent_hand: (B, 13, 2) [color, value] as the player sees them

    Every (color, value) card exists once. A value is impossible for a hidden slot if
    the card with that slot's color and that value is already visible (in my hand or
    revealed by the opponent); those values are masked so the head only ranks
    values that could really be there.
    """
    is_hidden = hidden_values >= 0                                          # (B, 13)
    if not is_hidden.any():
        return torch.tensor(0.0, device=belief_logits.device)

    visible = _visible_cards(my_hand, opponent_hand)                        # (B, 2 colors, 13 values)

    # impossible[b, slot, value] = visible[b, color of that opponent slot, value]
    batch_size = hidden_values.size(0)
    slot_color = opponent_hand[:, :, 0].long().clamp(0, 1)                  # (B, 13)
    slot_color = slot_color.unsqueeze(2).expand(batch_size, -1, NUM_VALUES)  # (B, 13, 13)
    impossible = visible.gather(1, slot_color)                              # (B, 13, 13)

    # Keep only hidden slots: N = number of hidden slots in the minibatch
    logits = belief_logits[is_hidden]                                       # (N, 13)
    targets = hidden_values[is_hidden]                                      # (N,)
    impossible = impossible[is_hidden]                                      # (N, 13)
    impossible.scatter_(1, targets.clamp(0, NUM_VALUES - 1).unsqueeze(1), False)  # never mask the true value

    return F.cross_entropy(logits.masked_fill(impossible, MASK_VALUE), targets)


def _visible_cards(my_hand: torch.Tensor, opponent_hand: torch.Tensor) -> torch.Tensor:
    """visible[b, color, value] = True if that card is in my hand or revealed by the opponent.

    my_hand, opponent_hand: (B, 13, 2). Returns (B, 2, 13) bool.
    """
    batch_size = my_hand.size(0)
    n_cards = 2 * NUM_VALUES          # flat index of a card = color * 13 + value
    unused = n_cards                  # extra column that absorbs empty / hidden slots
    visible = torch.zeros(batch_size, n_cards + 1, dtype=torch.bool, device=my_hand.device)

    for hand in (my_hand, opponent_hand):
        color = hand[:, :, 0].long()                                        # (B, 13)
        value = hand[:, :, 1].long()                                        # (B, 13), <0 = empty or hidden
        is_card = (value >= 0) & (value < NUM_VALUES)
        card_index = color.clamp(0, 1) * NUM_VALUES + value.clamp(0, NUM_VALUES - 1)
        card_index = torch.where(is_card, card_index, torch.full_like(card_index, unused))
        visible.scatter_(1, card_index, True)

    return visible[:, :n_cards].view(batch_size, 2, NUM_VALUES)
