"""
Policy network for Da Vinci Code.

    observation ─► ObservationEncoder ─► features (global state, CLS token)
                                      └► opp_per_pos (one vector per opponent slot)
    opp_per_pos ─► belief head (predicts each hidden opponent value) ─► added back
    features + opp_per_pos ─► action heads (color / position / value / decision)
    features ─► ValueHead ─► V(s)

One step of the game is one of three phases, and only that phase's head acts:
    DRAW     → color      (which color to draw)
    GUESS    → position, then value given that position
    DECISION → decision   (stop / keep guessing)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.constants import MASK_VALUE, MAX_HAND_SIZE, NUM_VALUES

# Token segments (which group a token belongs to)
SEG_MY, SEG_OPP, SEG_PHASE, SEG_DECK, SEG_CONSTRAINT = 0, 1, 2, 3, 4

ACTION_KEYS = ["color", "position", "value", "decision"]


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Push illegal actions to MASK_VALUE (finite, so fp16 stays NaN-free)."""
    if mask is None:
        return logits
    return logits.masked_fill(~mask.to(logits.device), MASK_VALUE)


def gate(logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Mask every action of rows whose phase does not use this head. active: (B,)."""
    active = active.bool().unsqueeze(-1).expand_as(logits)
    return torch.where(active, logits, torch.full_like(logits, MASK_VALUE))


def masked_categorical(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Categorical]:
    """Distribution over the rows that have at least one legal action.

    Returns (valid_rows, safe_logits, dist); dist covers only logits[valid_rows].
    A tiny 1e-6 floor keeps log_prob finite.
    """
    valid = ~torch.all(logits <= MASK_VALUE + 1, dim=-1)
    safe_logits = torch.clamp(logits[valid], min=MASK_VALUE)
    probs = F.softmax(safe_logits, dim=-1) + 1e-6
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return valid, safe_logits, Categorical(probs)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module):
    """Turn the observation into 42 tokens and run a Transformer over them.

    Tokens: [CLS | my cards (13) | opponent cards (13) | constraints (13) | phase | deck]
    - constraint token i = the values I ruled out for opponent slot i
    - my card token i also carries the values the opponent ruled out for it
    All slot-i tokens share one slot embedding so attention can line them up.
    """

    def __init__(self, hidden_dim: int = 512, token_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim

        # (Layer creation order is kept as-is: it fixes the random init for a given seed.)
        # Card = color (BLACK/WHITE/none) + value (0-12/hidden/none) + position
        self.color_embed = nn.Embedding(3, 16)
        self.value_embed = nn.Embedding(15, 32)
        self.position_embed = nn.Embedding(MAX_HAND_SIZE, 16)
        self.segment_embed = nn.Embedding(5, token_dim)
        self.slot_pos_embed = nn.Embedding(MAX_HAND_SIZE, token_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.card_proj = nn.Sequential(nn.Linear(16 + 32 + 16 + 1, token_dim),
                                       nn.LayerNorm(token_dim), nn.ReLU())
        self.phase_proj = nn.Sequential(nn.Linear(3, token_dim), nn.LayerNorm(token_dim))
        self.deck_proj = nn.Sequential(nn.Linear(2, token_dim), nn.LayerNorm(token_dim))

        layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads,
                                           dim_feedforward=token_dim * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.constraint_proj = nn.Sequential(nn.Linear(NUM_VALUES, token_dim),
                                             nn.LayerNorm(token_dim), nn.ReLU())
        # Zero-initialized in DaVinciCodePolicy so loaded checkpoints start unchanged.
        self.my_hand_constraint_proj = nn.Linear(NUM_VALUES, token_dim)

        self.opp_per_pos_proj = nn.Linear(token_dim, 64)
        self.fusion = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def _tokenize_hand(self, hand: torch.Tensor) -> torch.Tensor:
        """hand (B, 13, 2) = [color, value] per slot → card tokens (B, 13, token_dim)."""
        colors = hand[:, :, 0].long().clamp(-1, 1)
        colors = torch.where(colors < 0, torch.full_like(colors, 2), colors)  # none → 2
        values = hand[:, :, 1].long().clamp(-2, 12)
        values = torch.where(values == -2, torch.full_like(values, 14), values)  # none → 14
        values = torch.where(values == -1, torch.full_like(values, 13), values)  # hidden → 13

        positions = torch.arange(MAX_HAND_SIZE, device=hand.device).expand(hand.size(0), -1)
        pos_scalar = (positions.float() / (MAX_HAND_SIZE - 1)).unsqueeze(-1)  # left < right
        token = torch.cat([self.color_embed(colors), self.value_embed(values),
                           self.position_embed(positions), pos_scalar], dim=-1)
        return self.card_proj(token)

    def _segment(self, batch_size: int, n: int, segment: int, device) -> torch.Tensor:
        return self.segment_embed(torch.full((batch_size, n), segment, dtype=torch.long, device=device))

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (features (B, hidden_dim), constraint_per_pos (B, 13, token_dim),
        opponent_per_pos (B, 13, 64))."""
        my_hand = obs["my_hand"].float()
        opponent_hand = obs["opponent_hand"].float()
        B, device = my_hand.size(0), my_hand.device

        slot_pe = self.slot_pos_embed(torch.arange(MAX_HAND_SIZE, device=device).expand(B, -1))
        constraint_per_pos = self.constraint_proj(obs["opp_hand_constraint"].float())

        my_tokens = self._tokenize_hand(my_hand) + self._segment(B, MAX_HAND_SIZE, SEG_MY, device) + slot_pe
        my_tokens = my_tokens + self.my_hand_constraint_proj(obs["my_hand_constraint"].float())
        opp_tokens = self._tokenize_hand(opponent_hand) + self._segment(B, MAX_HAND_SIZE, SEG_OPP, device) + slot_pe
        con_tokens = constraint_per_pos + self._segment(B, MAX_HAND_SIZE, SEG_CONSTRAINT, device) + slot_pe
        phase_token = self.phase_proj(obs["phase"].float()).unsqueeze(1) + self._segment(B, 1, SEG_PHASE, device)
        deck_token = self.deck_proj(obs["remaining_deck"].float()).unsqueeze(1) + self._segment(B, 1, SEG_DECK, device)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, my_tokens, opp_tokens, con_tokens, phase_token, deck_token], dim=1)

        # Empty card slots (color < 0) are ignored by attention; constraints share the opponent's padding.
        my_pad, opp_pad = my_hand[:, :, 0] < 0, opponent_hand[:, :, 0] < 0
        never = torch.zeros(B, 1, dtype=torch.bool, device=device)
        padding = torch.cat([never, my_pad, opp_pad, opp_pad, never, never], dim=1)

        tokens = self.transformer(tokens, src_key_padding_mask=padding)

        features = self.fusion(tokens[:, 0])
        opponent_per_pos = self.opp_per_pos_proj(tokens[:, 1 + MAX_HAND_SIZE:1 + 2 * MAX_HAND_SIZE])
        return features, constraint_per_pos, opponent_per_pos


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class PhaseGatedActionHead(nn.Module):
    """color / position / value / decision logits.

    The value head is conditioned on the chosen position (autoregressive GUESS).
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.color_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.position_head = nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(), nn.Linear(128, 1))
        self.position_embedding = nn.Embedding(MAX_HAND_SIZE, 32)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim + 32 + 64, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, NUM_VALUES))
        self.decision_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def first_stage_logits(self, features: torch.Tensor, opponent_per_pos: torch.Tensor,
                           phase: torch.Tensor, action_mask: Optional[Dict[str, torch.Tensor]]
                           ) -> Dict[str, torch.Tensor]:
        """Masked + phase-gated logits of every head except value."""
        mask = action_mask or {}
        slots = features.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
        position = self.position_head(torch.cat([slots, opponent_per_pos], dim=-1)).squeeze(-1)
        return {
            "color": gate(apply_mask(self.color_head(features), mask.get("color")), phase[:, 0]),
            "position": gate(apply_mask(position, mask.get("position")), phase[:, 1]),
            "decision": gate(apply_mask(self.decision_head(features), mask.get("decision")), phase[:, 2]),
        }

    def value_logits(self, features: torch.Tensor, opponent_per_pos: torch.Tensor, phase: torch.Tensor,
                     action_mask: Optional[Dict[str, torch.Tensor]], position: torch.Tensor) -> torch.Tensor:
        """Masked + phase-gated value logits for the given (chosen) position."""
        rows = torch.arange(features.size(0), device=features.device)
        x = torch.cat([features, self.position_embedding(position), opponent_per_pos[rows, position]], dim=-1)
        mask = (action_mask or {}).get("value")
        if mask is not None and mask.dim() == 3:  # per-position mask (B, 13, 13)
            mask = mask.to(features.device)[rows, position]
        return gate(apply_mask(self.value_head(x), mask), phase[:, 1])

    def forward(self, features: torch.Tensor, phase: torch.Tensor,
                action_mask: Optional[Dict[str, torch.Tensor]] = None,
                selected_position: Optional[torch.Tensor] = None,
                opponent_per_pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """All four heads; value is conditioned on selected_position (0 if not given)."""
        if opponent_per_pos is None:
            opponent_per_pos = torch.zeros(features.size(0), MAX_HAND_SIZE, 64, device=features.device)
        if selected_position is None:
            selected_position = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        logits = self.first_stage_logits(features, opponent_per_pos, phase, action_mask)
        logits["value"] = self.value_logits(features, opponent_per_pos, phase, action_mask, selected_position)
        return {k: logits[k] for k in ACTION_KEYS}


class ValueHead(nn.Module):
    """V(s) for the PPO critic."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.value_net = nn.Sequential(nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                                       nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DaVinciCodePolicy(nn.Module):
    """Encoder + belief + action heads + value head."""

    def __init__(self, hidden_dim: int = 512, n_heads: int = 4, n_layers: int = 4,
                 zero_init: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self._zero_init = zero_init

        self.encoder = ObservationEncoder(hidden_dim, n_heads=n_heads, n_layers=n_layers)
        self.action_heads = PhaseGatedActionHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)
        # Belief: distribution over each opponent slot's hidden value (trained with an auxiliary CE loss)
        self.belief_head = nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(), nn.Linear(128, NUM_VALUES))
        self.belief_to_opp_proj = nn.Linear(NUM_VALUES, 64)

        self.apply(self._init_weights)
        # These two start as no-ops, so adding them never changes a loaded checkpoint's behaviour.
        for layer in (self.belief_to_opp_proj, self.encoder.my_hand_constraint_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _init_weights(self, module: nn.Module) -> None:
        """Orthogonal Linear/Conv, normal(0, 0.02) Embedding; all zeros if zero_init."""
        if self._zero_init:
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                nn.init.zeros_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            return
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _enrich_opp_with_belief(self, features: torch.Tensor, opponent_per_pos: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict hidden opponent values and add them back onto the opponent slot features.

        Gradient routing:
        - features is detached: the belief loss shapes the per-slot features only,
          not the global features that the value head relies on.
        - belief_probs is detached: the RL loss does not train the belief head.
        Returns (enriched opponent_per_pos (B, 13, 64), belief_logits (B, 13, 13)).
        """
        global_ctx = features.detach().unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
        belief_logits = self.belief_head(torch.cat([global_ctx, opponent_per_pos], dim=-1))
        belief_probs = belief_logits.detach().softmax(dim=-1)
        return opponent_per_pos + self.belief_to_opp_proj(belief_probs), belief_logits

    def _encode(self, obs: Dict[str, torch.Tensor]):
        features, constraint_per_pos, opponent_per_pos = self.encoder(obs)
        enriched_opp, belief_logits = self._enrich_opp_with_belief(features, opponent_per_pos)
        return features, constraint_per_pos, enriched_opp, belief_logits

    def forward(self, obs: Dict[str, torch.Tensor],
                action_mask: Optional[Dict[str, torch.Tensor]] = None,
                selected_position: Optional[torch.Tensor] = None
                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Returns (action logits, V(s), constraint_per_pos)."""
        features, constraint_per_pos, enriched_opp, _ = self._encode(obs)
        logits = self.action_heads(features, obs["phase"].float(), action_mask,
                                   selected_position, opponent_per_pos=enriched_opp)
        return logits, self.value_head(features), constraint_per_pos

    def get_action(self, obs: Dict[str, torch.Tensor],
                   action_mask: Optional[Dict[str, torch.Tensor]] = None,
                   deterministic: bool = False
                   ) -> Tuple[np.ndarray, Dict[str, torch.Tensor], torch.Tensor]:
        """Sample an action (rollouts / play). Position is chosen first, then value given it.

        Returns (actions (B, 4) = [color, position, value, decision], log_probs per head, V(s)).
        """
        features, _, enriched_opp, _ = self._encode(obs)
        phase = obs["phase"]
        heads = self.action_heads

        logits = heads.first_stage_logits(features, enriched_opp, phase, action_mask)
        actions, log_probs = {}, {}
        actions["position"], log_probs["position"] = self._sample(logits["position"], deterministic)
        logits["value"] = heads.value_logits(features, enriched_opp, phase, action_mask, actions["position"])
        for key in ["color", "value", "decision"]:
            actions[key], log_probs[key] = self._sample(logits[key], deterministic)

        action_array = torch.stack([actions[k] for k in ACTION_KEYS], dim=-1).cpu().numpy()
        return action_array, log_probs, self.value_head(features)

    def _sample(self, logits: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or argmax) one head. Rows with no legal action get action 0, log_prob 0."""
        action = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        log_prob = torch.zeros(logits.size(0), device=logits.device)
        valid, safe_logits, dist = masked_categorical(logits)
        if valid.any():
            chosen = safe_logits.argmax(dim=-1) if deterministic else dist.sample()
            action[valid] = chosen
            log_prob[valid] = dist.log_prob(chosen)
        return action, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor],
                         action_mask: Optional[Dict[str, torch.Tensor]] = None):
        """log_prob / entropy of stored actions (PPO update).

        Returns (log_probs per head, V(s), entropies per head, belief_logits (B, 13, 13)).
        """
        features, _, enriched_opp, belief_logits = self._encode(obs)
        logits = self.action_heads(features, obs["phase"].float(), action_mask,
                                   selected_position=actions["position"], opponent_per_pos=enriched_opp)
        log_probs, entropies = {}, {}
        for key, head_logits in logits.items():
            log_probs[key] = torch.zeros(head_logits.size(0), device=head_logits.device)
            entropies[key] = torch.zeros(head_logits.size(0), device=head_logits.device)
            valid, _, dist = masked_categorical(head_logits)
            if valid.any():
                log_probs[key][valid] = dist.log_prob(actions[key][valid])
                entropies[key][valid] = dist.entropy()
        return log_probs, self.value_head(features), entropies, belief_logits


def obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """Single numpy observation → batched (1, ...) tensors."""
    return {key: torch.from_numpy(val).unsqueeze(0).to(device) for key, val in obs.items()}


def action_mask_to_tensor(mask: Dict[str, np.ndarray],
                          device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """Single numpy action mask → batched (1, ...) bool tensors."""
    return {key: torch.from_numpy(val).unsqueeze(0).bool().to(device) for key, val in mask.items()}
