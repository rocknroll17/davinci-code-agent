"""
PPO loss terms (pure functions, no trainer state).

    total = policy_loss + vf_coef * value_loss + belief_coef * belief_loss - ent_coef * entropy
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.constants import MASK_VALUE, NUM_VALUES

# Which phase column (DRAW=0, GUESS=1, DECISION=2) each action head acts in
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
    """Clipped surrogate loss summed over heads, each head on the rows of its own phase.

    Returns (loss, mean entropy over active heads, approx_kl per head, clipfrac per head).
    """
    loss = torch.tensor(0.0, device=advantages.device)
    entropy_sum = torch.tensor(0.0, device=advantages.device)
    n_heads = 0
    kls, clipfracs = [], []

    for key, col in HEAD_PHASE.items():
        rows = phase[:, col].bool()
        if not rows.any():
            continue
        n_heads += 1

        log_ratio = torch.clamp(log_probs[key][rows] - old_log_probs[key][rows], min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)  # pi_new / pi_old
        adv = advantages[rows]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
        loss = loss - torch.min(unclipped, clipped).mean()

        kls.append(((ratio - 1) - log_ratio).mean().item())
        clipfracs.append(((ratio - 1).abs() > clip_range).float().mean().item())
        entropy_sum = entropy_sum + entropy_weight.get(key, 1.0) * entropies[key][rows].mean()

    entropy = entropy_sum / n_heads if n_heads > 0 else torch.tensor(0.0, device=advantages.device)
    return loss, entropy, kls, clipfracs


def value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
               clip_range_vf=None) -> torch.Tensor:
    """0.5 * (V - return)^2, optionally PPO-style clipped around the rollout value."""
    if clip_range_vf is None:
        return 0.5 * ((values - returns) ** 2).mean()
    clipped = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
    return 0.5 * torch.max((values - returns) ** 2, (clipped - returns) ** 2).mean()


def belief_loss(belief_logits: torch.Tensor, hidden_values: torch.Tensor,
                my_hand: torch.Tensor, opponent_hand: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the belief head on the opponent's hidden slots.

    belief_logits (B, 13 slots, 13 values); hidden_values (B, 13), -1 where not hidden.
    Values that cannot be there (same color already in my hand or already revealed
    by the opponent) are masked out first, so the head only ranks possible values.
    """
    hidden = hidden_values >= 0
    if not hidden.any():
        return torch.tensor(0.0, device=belief_logits.device)

    # known[b, color, value]: this card is visible to me (my hand or opponent's revealed cards)
    B = hidden_values.size(0)
    my_color, my_value = my_hand[:, :, 0].long(), my_hand[:, :, 1].long()
    opp_color, opp_value = opponent_hand[:, :, 0].long(), opponent_hand[:, :, 1].long()
    trash = 2 * NUM_VALUES  # scatter target for slots that are not a visible card
    known = torch.zeros(B, trash + 1, dtype=torch.bool, device=belief_logits.device)
    for color, value in ((my_color, my_value), (opp_color, opp_value)):
        visible = (value >= 0) & (value < NUM_VALUES)
        index = color.clamp(0, 1) * NUM_VALUES + value.clamp(0, NUM_VALUES - 1)
        known.scatter_(1, torch.where(visible, index, torch.full_like(index, trash)), True)
    known = known[:, :trash].view(B, 2, NUM_VALUES)

    # impossible[b, slot, value] = known[b, color of that opponent slot, value]
    slot_color = opp_color.clamp(0, 1).unsqueeze(2).expand(B, -1, NUM_VALUES)
    impossible = known.gather(1, slot_color)

    logits, targets, impossible = belief_logits[hidden], hidden_values[hidden], impossible[hidden]
    impossible.scatter_(1, targets.clamp(0, NUM_VALUES - 1).unsqueeze(1), False)  # never mask the label
    return F.cross_entropy(logits.masked_fill(impossible, MASK_VALUE), targets)
