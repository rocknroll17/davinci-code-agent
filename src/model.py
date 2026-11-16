"""
Policy network for Da Vinci Code.

Data flow (B = batch size, 13 = hand slots, 13 = card values 0-12):

    observation
      └─► ObservationEncoder (Transformer over 55 tokens)
            ├─► features          (B, 512)      summary of the whole game state
            └─► opponent_slots    (B, 13, 64)   one vector per opponent card slot
    opponent_slots ─► belief head ─► "what value is each hidden opponent card?"  (B, 13, 13)
                   └─ the belief is added back onto opponent_slots
    features + opponent_slots ─► action heads ─► logits for color / position / value / decision
    features ─► ValueHead ─► V(s)  (B, 1)

A game step is in exactly one phase, and only that phase's heads choose an action:
    DRAW     → color      which color to draw
    GUESS    → position   which opponent card to guess, then
               value      which number it is (conditioned on the chosen position)
    DECISION → decision   stop, or keep guessing
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.constants import MASK_VALUE, MAX_HAND_SIZE, NUM_VALUES

# Token groups, used as segment-embedding ids
SEG_MY, SEG_OPP, SEG_PHASE, SEG_DECK, SEG_CONSTRAINT, SEG_MY_CONSTRAINT = 0, 1, 2, 3, 4, 5

# Columns of the phase one-hot
DRAW, GUESS, DECISION = 0, 1, 2

ACTION_KEYS = ["color", "position", "value", "decision"]


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def mask_illegal(logits: torch.Tensor, legal: Optional[torch.Tensor]) -> torch.Tensor:
    """Set the logits of illegal actions to MASK_VALUE.

    logits: (B, A), legal: (B, A) bool or None (= everything legal).
    MASK_VALUE is a large negative *finite* number, not -inf: under fp16,
    softmax over a row of -inf gives NaN.
    """
    if legal is None:
        return logits
    legal = legal.to(logits.device)
    return logits.masked_fill(~legal, MASK_VALUE)


def mask_inactive_phase(logits: torch.Tensor, phase_is_active: torch.Tensor) -> torch.Tensor:
    """Mask every action in rows where this head's phase is not the current phase.

    logits: (B, A), phase_is_active: (B,) — e.g. phase[:, GUESS] for the position head.
    """
    active = phase_is_active.bool().unsqueeze(-1).expand_as(logits)  # (B, A)
    masked = torch.full_like(logits, MASK_VALUE)
    return torch.where(active, logits, masked)


def legal_action_distribution(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Categorical]:
    """Categorical distribution over the rows that have at least one legal action.

    logits: (B, A) already masked.
    Returns:
        has_legal:   (B,) bool — rows where some action is legal
        safe_logits: (N, A)    — logits of those rows (N = has_legal.sum())
        dist:        Categorical over those N rows
    A 1e-6 floor on every probability keeps log_prob finite even for masked actions.
    """
    has_legal = ~torch.all(logits <= MASK_VALUE + 1, dim=-1)       # (B,)
    safe_logits = torch.clamp(logits[has_legal], min=MASK_VALUE)   # (N, A)
    probs = F.softmax(safe_logits, dim=-1) + 1e-6                  # (N, A)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return has_legal, safe_logits, Categorical(probs)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module):
    """Turn an observation into 55 tokens and run a Transformer over them.

    Token layout (index: content):
        0       CLS — its output becomes `features`
        1-13    my cards
        14-26   opponent cards       (hidden values show up as "hidden")
        27-39   opponent constraints (what I ruled out for each opponent card)
        40-52   my constraints       (what the opponent ruled out for each of my cards)
        53      phase
        54      deck                 (cards left per color)
    Slot i of every 13-token group shares one slot embedding, so attention can tell
    that "opponent constraint i" belongs to "opponent card i" (and likewise for mine).
    """

    def __init__(self, hidden_dim: int = 512, token_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 4, ff_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim

        # NOTE: keep the layer creation order — it fixes the random init for a given seed.

        # A card = color (BLACK / WHITE / no card) + value (0-12 / hidden / no card) + slot
        self.color_embed = nn.Embedding(3, 16)
        self.value_embed = nn.Embedding(15, 32)
        self.position_embed = nn.Embedding(MAX_HAND_SIZE, 16)
        self.segment_embed = nn.Embedding(6, token_dim)
        self.slot_pos_embed = nn.Embedding(MAX_HAND_SIZE, token_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.card_proj = nn.Sequential(nn.Linear(16 + 32 + 16 + 1, token_dim),
                                       nn.LayerNorm(token_dim), nn.ReLU())
        self.phase_proj = nn.Sequential(nn.Linear(3, token_dim), nn.LayerNorm(token_dim))
        self.deck_proj = nn.Sequential(nn.Linear(2, token_dim), nn.LayerNorm(token_dim))

        layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads,
                                           dim_feedforward=ff_dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # One 13-value row per slot → one token (opponent's cards / my cards)
        self.constraint_proj = nn.Sequential(nn.Linear(NUM_VALUES, token_dim),
                                             nn.LayerNorm(token_dim), nn.ReLU())
        self.my_constraint_proj = nn.Sequential(nn.Linear(NUM_VALUES, token_dim),
                                                nn.LayerNorm(token_dim), nn.ReLU())

        self.opp_per_pos_proj = nn.Linear(token_dim, 64)
        self.fusion = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features:           (B, hidden_dim)  from the CLS token
            constraint_per_pos: (B, 13, token_dim) opponent-constraint tokens before attention
            opponent_per_pos:   (B, 13, 64)      opponent card tokens after attention
        """
        my_hand = obs["my_hand"].float()               # (B, 13, 2) [color, value]
        opponent_hand = obs["opponent_hand"].float()   # (B, 13, 2), value -1 = hidden
        batch_size = my_hand.size(0)
        device = my_hand.device

        slots = torch.arange(MAX_HAND_SIZE, device=device).expand(batch_size, -1)  # (B, 13)
        slot_pe = self.slot_pos_embed(slots)                                        # (B, 13, D)

        my_tokens = self._tokenize_hand(my_hand)                                    # (B, 13, D)
        my_tokens = my_tokens + self._segment(batch_size, MAX_HAND_SIZE, SEG_MY, device) + slot_pe

        opp_tokens = self._tokenize_hand(opponent_hand)                             # (B, 13, D)
        opp_tokens = opp_tokens + self._segment(batch_size, MAX_HAND_SIZE, SEG_OPP, device) + slot_pe

        # What I have ruled out for each opponent card
        constraint_per_pos = self.constraint_proj(obs["opp_hand_constraint"].float())  # (B, 13, D)
        con_tokens = constraint_per_pos + self._segment(batch_size, MAX_HAND_SIZE, SEG_CONSTRAINT, device) + slot_pe

        # What the opponent has ruled out for each of my cards
        my_con_tokens = self.my_constraint_proj(obs["my_hand_constraint"].float())  # (B, 13, D)
        my_con_tokens = my_con_tokens + self._segment(batch_size, MAX_HAND_SIZE, SEG_MY_CONSTRAINT, device) + slot_pe

        phase_token = self.phase_proj(obs["phase"].float()).unsqueeze(1)           # (B, 1, D)
        phase_token = phase_token + self._segment(batch_size, 1, SEG_PHASE, device)
        deck_token = self.deck_proj(obs["remaining_deck"].float()).unsqueeze(1)     # (B, 1, D)
        deck_token = deck_token + self._segment(batch_size, 1, SEG_DECK, device)

        cls = self.cls_token.expand(batch_size, -1, -1)                             # (B, 1, D)
        tokens = torch.cat([cls, my_tokens, opp_tokens, con_tokens, my_con_tokens,
                            phase_token, deck_token], dim=1)                        # (B, 55, D)

        # Attention ignores empty card slots (color < 0); a constraint token is empty
        # exactly when its card slot is.
        my_empty = my_hand[:, :, 0] < 0                                             # (B, 13)
        opp_empty = opponent_hand[:, :, 0] < 0                                      # (B, 13)
        never = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)        # (B, 1)
        padding = torch.cat([never, my_empty, opp_empty, opp_empty, my_empty, never, never], dim=1)  # (B, 55)

        tokens = self.transformer(tokens, src_key_padding_mask=padding)             # (B, 55, D)

        features = self.fusion(tokens[:, 0])                                        # (B, hidden_dim)
        opp_out = tokens[:, 1 + MAX_HAND_SIZE:1 + 2 * MAX_HAND_SIZE]                # (B, 13, D)
        opponent_per_pos = self.opp_per_pos_proj(opp_out)                           # (B, 13, 64)
        return features, constraint_per_pos, opponent_per_pos

    def _tokenize_hand(self, hand: torch.Tensor) -> torch.Tensor:
        """hand (B, 13, 2) [color, value] → card tokens (B, 13, D)."""
        colors = hand[:, :, 0].long().clamp(-1, 1)                                  # (B, 13)
        colors = torch.where(colors < 0, torch.full_like(colors, 2), colors)        # no card → 2
        values = hand[:, :, 1].long().clamp(-2, 12)                                 # (B, 13)
        values = torch.where(values == -2, torch.full_like(values, 14), values)     # no card → 14
        values = torch.where(values == -1, torch.full_like(values, 13), values)     # hidden → 13

        positions = torch.arange(MAX_HAND_SIZE, device=hand.device).expand(hand.size(0), -1)  # (B, 13)
        position_scalar = (positions.float() / (MAX_HAND_SIZE - 1)).unsqueeze(-1)   # (B, 13, 1), keeps left < right

        card = torch.cat([self.color_embed(colors),          # (B, 13, 16)
                          self.value_embed(values),          # (B, 13, 32)
                          self.position_embed(positions),    # (B, 13, 16)
                          position_scalar], dim=-1)          # (B, 13, 65)
        return self.card_proj(card)

    def _segment(self, batch_size: int, n_tokens: int, segment: int, device) -> torch.Tensor:
        """Segment embedding for n_tokens tokens of one group → (B, n_tokens, D)."""
        ids = torch.full((batch_size, n_tokens), segment, dtype=torch.long, device=device)
        return self.segment_embed(ids)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class PhaseGatedActionHead(nn.Module):
    """Logits for the four action parts. Value is conditioned on the chosen position."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.color_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.position_head = nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(), nn.Linear(128, 1))
        self.position_embedding = nn.Embedding(MAX_HAND_SIZE, 32)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim + 32 + 64, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, NUM_VALUES))
        self.decision_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def logits_without_value(self, features: torch.Tensor, opponent_per_pos: torch.Tensor,
                             phase: torch.Tensor, legal: Optional[Dict[str, torch.Tensor]]
                             ) -> Dict[str, torch.Tensor]:
        """Masked logits of color (B, 2), position (B, 13) and decision (B, 2).

        Value is left out because it needs the position to be chosen first.
        """
        legal = legal or {}

        # Score each opponent slot from (global features, that slot's features)
        features_per_slot = features.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)       # (B, 13, H)
        slot_input = torch.cat([features_per_slot, opponent_per_pos], dim=-1)         # (B, 13, H + 64)
        position = self.position_head(slot_input).squeeze(-1)                         # (B, 13)

        color = self.color_head(features)                                             # (B, 2)
        decision = self.decision_head(features)                                       # (B, 2)
        return {
            "color": mask_inactive_phase(mask_illegal(color, legal.get("color")), phase[:, DRAW]),
            "position": mask_inactive_phase(mask_illegal(position, legal.get("position")), phase[:, GUESS]),
            "decision": mask_inactive_phase(mask_illegal(decision, legal.get("decision")), phase[:, DECISION]),
        }

    def value_logits(self, features: torch.Tensor, opponent_per_pos: torch.Tensor, phase: torch.Tensor,
                     legal: Optional[Dict[str, torch.Tensor]], position: torch.Tensor) -> torch.Tensor:
        """Masked value logits (B, 13) for the card at `position` (B,)."""
        rows = torch.arange(features.size(0), device=features.device)              # (B,)
        chosen_slot = opponent_per_pos[rows, position]                              # (B, 64)
        value_input = torch.cat([features, self.position_embedding(position), chosen_slot], dim=-1)  # (B, H + 32 + 64)
        value = self.value_head(value_input)                                        # (B, 13)

        legal_value = (legal or {}).get("value")
        if legal_value is not None and legal_value.dim() == 3:
            # Legal values depend on the slot (B, 13 slots, 13 values): take the chosen slot's row
            legal_value = legal_value.to(features.device)[rows, position]           # (B, 13)
        return mask_inactive_phase(mask_illegal(value, legal_value), phase[:, GUESS])

    def forward(self, features: torch.Tensor, phase: torch.Tensor,
                action_mask: Optional[Dict[str, torch.Tensor]] = None,
                selected_position: Optional[torch.Tensor] = None,
                opponent_per_pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """All four heads. Value uses selected_position (slot 0 if not given)."""
        batch_size, device = features.size(0), features.device
        if opponent_per_pos is None:
            opponent_per_pos = torch.zeros(batch_size, MAX_HAND_SIZE, 64, device=device)
        if selected_position is None:
            selected_position = torch.zeros(batch_size, dtype=torch.long, device=device)

        logits = self.logits_without_value(features, opponent_per_pos, phase, action_mask)
        logits["value"] = self.value_logits(features, opponent_per_pos, phase, action_mask, selected_position)
        return {key: logits[key] for key in ACTION_KEYS}


class ValueHead(nn.Module):
    """V(s): expected return from this state, for the PPO critic."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.value_net = nn.Sequential(nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(),
                                       nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (B, H) → V(s) (B, 1)."""
        return self.value_net(features)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DaVinciCodePolicy(nn.Module):
    """Encoder + belief + action heads + value head.

    get_action:       pick actions while playing (collect rollouts)
    evaluate_actions: log-probs of stored actions (PPO update)
    Both go through the same masking and distribution code, so the
    probabilities used to collect data are the ones PPO trains on.
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 4, n_layers: int = 4,
                 zero_init: bool = False, ff_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self._zero_init = zero_init

        self.encoder = ObservationEncoder(hidden_dim, n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim)
        self.action_heads = PhaseGatedActionHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)
        # Belief: for every opponent slot, a distribution over its 13 possible values.
        # Trained by an auxiliary cross-entropy loss against the true hidden values.
        self.belief_head = nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(), nn.Linear(128, NUM_VALUES))
        self.belief_to_opp_proj = nn.Linear(NUM_VALUES, 64)

        self.apply(self._init_weights)
        # Start the belief residual as a no-op so it does not disturb early training.
        nn.init.zeros_(self.belief_to_opp_proj.weight)
        nn.init.zeros_(self.belief_to_opp_proj.bias)

    def _init_weights(self, module: nn.Module) -> None:
        """Orthogonal init for Linear/Conv, normal(0, 0.02) for Embedding (all zeros if zero_init)."""
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

    # -- encoding ------------------------------------------------------------

    def _enrich_opp_with_belief(self, features: torch.Tensor, opponent_per_pos: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict each hidden opponent value and add that belief onto the slot features.

        features (B, H), opponent_per_pos (B, 13, 64)
        Returns (enriched opponent_per_pos (B, 13, 64), belief_logits (B, 13, 13)).

        Two detaches decide which loss trains what:
        - features.detach(): the belief loss may shape the per-slot features, but not
          the global features that V(s) is computed from.
        - belief_logits.detach(): the RL loss does not train the belief head.
        """
        global_per_slot = features.detach().unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)   # (B, 13, H)
        belief_input = torch.cat([global_per_slot, opponent_per_pos], dim=-1)           # (B, 13, H + 64)
        belief_logits = self.belief_head(belief_input)                                  # (B, 13, 13)
        belief_probs = belief_logits.detach().softmax(dim=-1)                           # (B, 13, 13)
        enriched = opponent_per_pos + self.belief_to_opp_proj(belief_probs)             # (B, 13, 64)
        return enriched, belief_logits

    def _encode(self, obs: Dict[str, torch.Tensor]):
        """Encoder + belief → (features, constraint_per_pos, enriched opponent_per_pos, belief_logits)."""
        features, constraint_per_pos, opponent_per_pos = self.encoder(obs)
        enriched_opp, belief_logits = self._enrich_opp_with_belief(features, opponent_per_pos)
        return features, constraint_per_pos, enriched_opp, belief_logits

    def forward(self, obs: Dict[str, torch.Tensor],
                action_mask: Optional[Dict[str, torch.Tensor]] = None,
                selected_position: Optional[torch.Tensor] = None
                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Returns (logits per head, V(s) (B, 1), constraint_per_pos (B, 13, D))."""
        features, constraint_per_pos, enriched_opp, _ = self._encode(obs)
        logits = self.action_heads(features, obs["phase"].float(), action_mask,
                                   selected_position, opponent_per_pos=enriched_opp)
        return logits, self.value_head(features), constraint_per_pos

    # -- acting ---------------------------------------------------------------

    def get_action(self, obs: Dict[str, torch.Tensor],
                   action_mask: Optional[Dict[str, torch.Tensor]] = None,
                   deterministic: bool = False
                   ) -> Tuple[np.ndarray, Dict[str, torch.Tensor], torch.Tensor]:
        """Choose an action for each row (used while collecting rollouts and playing).

        Returns:
            actions:   (B, 4) numpy [color, position, value, decision]
            log_probs: {head: (B,)} — 0 for heads not used in that row's phase
            value:     V(s) (B, 1)
        """
        features, _, enriched_opp, _ = self._encode(obs)
        phase = obs["phase"]
        heads = self.action_heads

        logits = heads.logits_without_value(features, enriched_opp, phase, action_mask)
        actions, log_probs = {}, {}

        # GUESS is two steps: pick a position, then pick a value for that position.
        actions["position"], log_probs["position"] = self._choose(logits["position"], deterministic)
        logits["value"] = heads.value_logits(features, enriched_opp, phase, action_mask, actions["position"])

        for key in ["color", "value", "decision"]:
            actions[key], log_probs[key] = self._choose(logits[key], deterministic)

        action_array = torch.stack([actions[k] for k in ACTION_KEYS], dim=-1).cpu().numpy()  # (B, 4)
        return action_array, log_probs, self.value_head(features)

    def _choose(self, logits: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or take the argmax of) one head.

        logits (B, A) → action (B,), log_prob (B,). Rows with no legal action get 0 and 0.
        """
        action = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        log_prob = torch.zeros(logits.size(0), device=logits.device)
        has_legal, safe_logits, dist = legal_action_distribution(logits)
        if has_legal.any():
            chosen = safe_logits.argmax(dim=-1) if deterministic else dist.sample()   # (N,)
            action[has_legal] = chosen
            log_prob[has_legal] = dist.log_prob(chosen)
        return action, log_prob

    # -- training -------------------------------------------------------------

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor],
                         action_mask: Optional[Dict[str, torch.Tensor]] = None):
        """Log-probs and entropies of already-taken actions, for the PPO loss.

        actions: {head: (B,)}
        Returns (log_probs {head: (B,)}, V(s) (B, 1), entropies {head: (B,)}, belief_logits (B, 13, 13)).
        """
        features, _, enriched_opp, belief_logits = self._encode(obs)
        logits = self.action_heads(features, obs["phase"].float(), action_mask,
                                   selected_position=actions["position"], opponent_per_pos=enriched_opp)

        log_probs, entropies = {}, {}
        for key, head_logits in logits.items():
            batch_size, device = head_logits.size(0), head_logits.device
            log_probs[key] = torch.zeros(batch_size, device=device)
            entropies[key] = torch.zeros(batch_size, device=device)
            has_legal, _, dist = legal_action_distribution(head_logits)
            if has_legal.any():
                log_probs[key][has_legal] = dist.log_prob(actions[key][has_legal])
                entropies[key][has_legal] = dist.entropy()
        return log_probs, self.value_head(features), entropies, belief_logits


def load_weights(policy: nn.Module, state_dict: Dict[str, torch.Tensor]) -> list:
    """Load a checkpoint's weights into `policy`, also from older architectures.

    - torch.compile's "._orig_mod." key wrapper is ignored on both sides.
    - Same shape: copied.
    - An embedding table that has grown (more rows, e.g. a new token segment):
      the old rows are copied, the new rows keep their fresh init.
    - Anything else that does not fit keeps its fresh init.
    Returns the parameter names that kept (some of) their fresh init.
    """
    norm = lambda k: k.replace("._orig_mod.", ".")
    saved = {norm(k): v for k, v in state_dict.items()}
    own = policy.state_dict()
    fresh = []
    with torch.no_grad():
        for name, param in own.items():
            old = saved.get(norm(name))
            if old is not None and old.shape == param.shape:
                param.copy_(old)
            elif old is not None and old.dim() == 2 and old.shape[1] == param.shape[1] and old.shape[0] < param.shape[0]:
                param[:old.shape[0]].copy_(old)
                fresh.append(name)
            else:
                fresh.append(name)
    return fresh


def obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """One numpy observation → batched (1, ...) tensors."""
    return {key: torch.from_numpy(val).unsqueeze(0).to(device) for key, val in obs.items()}


def action_mask_to_tensor(mask: Dict[str, np.ndarray],
                          device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """One numpy action mask → batched (1, ...) bool tensors."""
    return {key: torch.from_numpy(val).unsqueeze(0).bool().to(device) for key, val in mask.items()}
