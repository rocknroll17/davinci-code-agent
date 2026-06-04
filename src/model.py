"""
Phase-Gated Multi-Head Policy Network for Da Vinci Code.

This module implements the neural network architecture described in model.md:
- Input: Dict observation space with phase, hands, deck info, constraint matrix
- Output: Multi-Head action space [color, position, value, decision]
- Phase-based gating to activate only relevant action heads
- Autoregressive value head conditioned on position selection

Key improvements:
- Position-conditioned value prediction for GUESS phase
- Better constraint matrix utilization
- Numerically stable softmax operations
- Layer normalization for training stability
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.constants import MASK_VALUE, MAX_HAND_SIZE, NUM_VALUES


class ObservationEncoder(nn.Module):
    """
    Unified Transformer Encoder for Da Vinci Code.
    
    42개 토큰이 모두 서로를 attend하는 Full Self-Attention:
    - 13 내 카드 토큰
    - 13 상대 카드 토큰
    - 13 constraint 토큰 (별도 토큰, opp와 1:1 대응)
    - 1 PHASE 토큰
    - 1 DECK 토큰
    
    핵심: 4-layer full self-attention으로 다단계 연쇄 추론이 가능:
    Layer 1: "내 검정5가 있다" → 상대 검정 포지션에 전달
    Layer 2: "정렬 제약: 상대 pos3 < pos4" → pos3 범위 축소
    Layer 3: "pos3이 확정되면 pos2도 좁아짐" → 연쇄 소거
    Layer 4: "가장 후보가 적은 포지션 = 공격 우선순위" → 전략 결정
    
    분리형 Cross-Attention과 달리:
    - 내 카드끼리, 상대 카드끼리, 내↔상대 모두 동시에 attend
    - PHASE/DECK 토큰이 전략 컨텍스트를 all cards에 broadcast
    - 4 layer 반복으로 1-hop이 아닌 multi-hop reasoning 가능
    
    Output interface (기존과 동일):
    - features: (batch, hidden_dim) global state features
    - constraint_per_pos: (batch, 13, token_dim) linear constraint features per opponent position
    - opponent_per_pos: (batch, 13, 64) attention-enriched opponent features
    """
    
    NUM_TOKENS = MAX_HAND_SIZE * 3 + 2 + 1  # 13+13+13+1+1+1(CLS) = 42
    
    def __init__(
        self,
        hidden_dim: int = 512,
        token_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4
    ) -> None:
        """
        Initialize the Unified Transformer encoder.
        
        Args:
            hidden_dim: Output feature dimension (must match action heads)
            token_dim: Per-token embedding dimension
            n_heads: Number of attention heads (head_dim = token_dim / n_heads)
            n_layers: Number of full self-attention layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        
        # === Card Tokenizer ===
        # Color: BLACK=0, WHITE=1, NONE(-1)→2
        self.color_embed = nn.Embedding(3, 16)
        # Value: 0-12(cards), HIDDEN(-1)→13, NONE(-2)→14
        self.value_embed = nn.Embedding(15, 32)
        # Hand position: 0-12
        self.position_embed = nn.Embedding(MAX_HAND_SIZE, 16)
        
        # Segment type: my_card=0, opp_card=1, phase=2, deck=3, constraint=4
        self.segment_embed = nn.Embedding(5, token_dim)

        # Shared SLOT position embedding (token_dim), added to my / opp / constraint
        # tokens alike so that my_i, opp_i and constraint_i share the SAME slot
        # identity in the attention space. This is what lets attention bind a
        # constraint token to its 1:1 opponent slot (con_i <-> opp_i); without it
        # constraint tokens are positionally anonymous and the binding is
        # structurally unlearnable. Uses the standard embedding init (normal 0.02)
        # so slots carry distinct identity from the start of fresh training.
        self.slot_pos_embed = nn.Embedding(MAX_HAND_SIZE, token_dim)
        
        # [CLS] token for global state aggregation (BERT-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        
        # Card feature projection: 16(color)+32(value)+16(pos)+1(continuous_pos) = 65 → token_dim
        self.card_proj = nn.Sequential(
            nn.Linear(65, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU()
        )
        
        # Constraint row projection: REMOVED — constraint is now injected via CNN.
        # See constraint_cnn_proj below.
        

        
        # Special token projections
        self.phase_proj = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.LayerNorm(token_dim)
        )
        self.deck_proj = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.LayerNorm(token_dim)
        )
        
        # === Unified Transformer: n_layers of full self-attention ===
        # 28개 토큰 모두가 서로 attend → multi-hop 추론 가능
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=token_dim * 4, dropout=0.0,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        
        # Constraint tokenizer: (batch, 13, 13) → (batch, 13, token_dim)
        # 13 = value candidates 0-12 per slot (color is conveyed via opp_hand token)
        # Injected directly into opp_tokens before Transformer for Self-Attention.
        self.constraint_proj = nn.Sequential(
            nn.Linear(NUM_VALUES, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU()
        )
        
        # === Output projections ===
        # Opponent per-position: token_dim → 64 (position head 호환)
        self.opp_per_pos_proj = nn.Linear(token_dim, 64)
        
        # Global fusion: mean pool 28 tokens → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def _tokenize_hand(self, hand: torch.Tensor) -> torch.Tensor:
        """
        Convert hand (batch, MAX_HAND_SIZE, 2) to token embeddings.
        
        Args:
            hand: (batch, 13, 2) where [:,:,0]=color, [:,:,1]=value
            
        Returns:
            Token embeddings (batch, 13, token_dim)
        """
        batch_size = hand.size(0)
        device = hand.device
        
        # Remap colors: BLACK=0→0, WHITE=1→1, NONE=-1→2
        colors = hand[:, :, 0].long()
        colors = torch.clamp(colors, min=-1, max=1)
        colors = torch.where(colors < 0, torch.full_like(colors, 2), colors)
        
        # Remap values: 0-12→0-12, HIDDEN=-1→13, NONE=-2→14
        values = hand[:, :, 1].long()
        values = torch.clamp(values, min=-2, max=12)
        values = torch.where(values == -2, torch.full_like(values, 14), values)
        values = torch.where(values == -1, torch.full_like(values, 13), values)
        
        # Position indices
        positions = torch.arange(MAX_HAND_SIZE, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Continuous position scalar: [0.0, 0.083, ..., 1.0] encodes left<right ordering
        pos_continuous = (positions.float() / (MAX_HAND_SIZE - 1)).unsqueeze(-1)  # (batch, 13, 1)
        
        # Embed and concatenate: (batch, 13, 16+32+16+1=65)
        token = torch.cat([
            self.color_embed(colors),       # (batch, 13, 16)
            self.value_embed(values),        # (batch, 13, 32)
            self.position_embed(positions),  # (batch, 13, 16)
            pos_continuous                   # (batch, 13, 1)
        ], dim=-1)
        
        return self.card_proj(token)  # (batch, 13, token_dim)
    
    def _get_padding_mask(self, hand: torch.Tensor) -> torch.Tensor:
        """
        Get padding mask for empty card slots.
        Empty slots have color == -1 (NONE).
        
        Returns:
            (batch, 13) bool tensor where True = empty/pad (ignored by attention)
        """
        return hand[:, :, 0] < 0
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation through Unified Transformer.
        
        All 28 tokens attend to each other → multi-step chained reasoning.
        
        Args:
            obs: Dictionary with keys: phase, my_hand, opponent_hand,
                 remaining_deck, constraint_matrix
                 
        Returns:
            Tuple of:
                - features: (batch, hidden_dim) global state features
                - constraint_per_pos: (batch, 13, token_dim) linear constraint features per opponent position
                - opponent_per_pos: (batch, 13, 64) attention-enriched features
        """
        phase = obs["phase"].float()
        my_hand = obs["my_hand"].float()             # (batch, 13, 2)
        opponent_hand = obs["opponent_hand"].float()  # (batch, 13, 2)
        remaining_deck = obs["remaining_deck"].float()
        constraint_matrix = obs["constraint_matrix"].float()  # (batch, 13, 13)
        
        batch_size = my_hand.size(0)
        device = my_hand.device
        
        # === Constraint tokenization: Linear projection per position ===
        # (batch, 13, 13) → (batch, 13, token_dim) — one token per opponent position
        constraint_per_pos = self.constraint_proj(constraint_matrix)  # (batch, 13, token_dim)
        
        # === Build 29 tokens ===
        
        # Shared slot-position embedding (batch, 13, token_dim) added to my/opp/con
        # so slot i has the same positional identity across all three token groups.
        slot_positions = torch.arange(MAX_HAND_SIZE, device=device).unsqueeze(0).expand(batch_size, -1)
        slot_pe = self.slot_pos_embed(slot_positions)  # (batch, 13, token_dim)

        # [1:14] My card tokens
        my_tokens = self._tokenize_hand(my_hand)  # (batch, 13, token_dim)
        my_seg = torch.zeros(batch_size, MAX_HAND_SIZE, dtype=torch.long, device=device)
        my_tokens = my_tokens + self.segment_embed(my_seg) + slot_pe

        # [14:27] Opponent card tokens
        opp_tokens = self._tokenize_hand(opponent_hand)  # (batch, 13, token_dim)
        opp_seg = torch.ones(batch_size, MAX_HAND_SIZE, dtype=torch.long, device=device)
        opp_tokens = opp_tokens + self.segment_embed(opp_seg) + slot_pe

        # [27:40] Constraint tokens — 별도 토큰, opp 슬롯과 1:1 대응
        # slot_pe를 공유하므로 con_i와 opp_i가 같은 슬롯 좌표를 가짐 → attention이
        # 1:1 결합을 학습 가능. (이전엔 위치 신호가 없어 구조적으로 불가능했음)
        constraint_tokens = constraint_per_pos  # (batch, 13, token_dim)
        constraint_seg = torch.full((batch_size, MAX_HAND_SIZE), 4, dtype=torch.long, device=device)
        constraint_tokens = constraint_tokens + self.segment_embed(constraint_seg) + slot_pe

        # [40] PHASE token
        phase_token = self.phase_proj(phase).unsqueeze(1)  # (batch, 1, token_dim)
        phase_seg = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        phase_token = phase_token + self.segment_embed(phase_seg)
        
        # [28] DECK token
        deck_token = self.deck_proj(remaining_deck).unsqueeze(1)  # (batch, 1, token_dim)
        deck_seg = torch.full((batch_size, 1), 3, dtype=torch.long, device=device)
        deck_token = deck_token + self.segment_embed(deck_seg)
        
        # [0] CLS token — prepended for global state aggregation
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, token_dim)
        
        # Concatenate: [CLS(1) | my(13) | opp(13) | constraint(13) | phase(1) | deck(1)] = 42 tokens
        all_tokens = torch.cat([cls_tokens, my_tokens, opp_tokens, constraint_tokens, phase_token, deck_token], dim=1)

        # Padding mask: CLS never masked; constraint shares opp's padding mask
        my_pad = self._get_padding_mask(my_hand)         # (batch, 13)
        opp_pad = self._get_padding_mask(opponent_hand)   # (batch, 13)
        cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        special_pad = torch.zeros(batch_size, 2, dtype=torch.bool, device=device)
        all_pad = torch.cat([cls_pad, my_pad, opp_pad, opp_pad, special_pad], dim=1)  # (batch, 42)
        
        # === Full Self-Attention (4 layers) ===
        # All 42 tokens attend to each other → multi-hop chained reasoning
        all_tokens = self.transformer(all_tokens, src_key_padding_mask=all_pad)
        
        # === Extract outputs ===
        
        # [CLS] token output → global features (no dilution via pooling)
        global_feat = all_tokens[:, 0, :]  # (batch, token_dim)
        features = self.fusion(global_feat)  # (batch, hidden_dim)
        
        # Opponent per-position: tokens[14:27] (shifted by 1 for CLS) → (batch, 13, 64)
        opp_out = all_tokens[:, MAX_HAND_SIZE + 1:2 * MAX_HAND_SIZE + 1, :]
        opponent_per_pos = self.opp_per_pos_proj(opp_out)
        
        return features, constraint_per_pos, opponent_per_pos


class PhaseGatedActionHead(nn.Module):
    """
    Multi-head action output with phase-based gating.
    
    Key improvements:
    - Position-conditioned value head (autoregressive)
    - Uses constraint features for better value prediction
    - Numerically stable masking with MASK_VALUE instead of -inf
    """
    
    def __init__(self, hidden_dim: int = 512) -> None:
        """
        Initialize action heads.
        
        Args:
            hidden_dim: Input feature dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Color head for DRAW phase (2 outputs: BLACK, WHITE)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Position head for GUESS phase - per-position features
        # Input: global features + opponent_per_pos (constraint baked in via Transformer)
        # (hidden_dim + 64) per position → 1 logit per position
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value head for GUESS phase - CONDITIONED on position
        # Takes features + position embedding + opponent_per_pos for that position
        # opponent_per_pos already encodes constraint info (injected before Transformer)
        self.position_embedding = nn.Embedding(MAX_HAND_SIZE, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 64, 128),  # features + pos_embed + opp_per_pos
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_VALUES)
        )
        
        # Decision head for DECISION phase (2 outputs: STOP, CONTINUE)
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        phase: torch.Tensor,
        action_mask: Optional[Dict[str, torch.Tensor]] = None,
        selected_position: Optional[torch.Tensor] = None,
        opponent_per_pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action logits for all heads with phase gating.
        
        Args:
            features: Encoded observation features (batch, hidden_dim)
            phase: One-hot phase vector (batch, 3)
            action_mask: Optional masks for each action head
            selected_position: Position selected for value conditioning (batch,)
            opponent_per_pos: Per-position opponent features (batch, 13, 64)
                              Encodes constraint info injected before Transformer.
                              Used by both position head and value head.
            
        Returns:
            Dictionary of action logits for each head
        """
        batch_size = features.size(0)
        device = features.device
        
        # Compute raw logits for heads that don't need conditioning
        color_logits = self.color_head(features)
        
        # Position head with per-position features (constraint baked into opp_per_pos via Transformer)
        features_expanded = features.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)  # (batch, 13, hidden_dim)
        if opponent_per_pos is not None:
            position_input = torch.cat([features_expanded, opponent_per_pos], dim=-1)
        else:
            dummy_opp = torch.zeros(batch_size, MAX_HAND_SIZE, 64, device=device)
            position_input = torch.cat([features_expanded, dummy_opp], dim=-1)
        position_logits = self.position_head(position_input).squeeze(-1)  # (batch, 13)
        
        decision_logits = self.decision_head(features)
        
        # Value head is conditioned on position; uses opponent_per_pos (carries constraint info)
        if selected_position is not None:
            pos_embed = self.position_embedding(selected_position)  # (batch, 32)
            batch_indices = torch.arange(batch_size, device=device)
            if opponent_per_pos is not None:
                pos_opp = opponent_per_pos[batch_indices, selected_position]  # (batch, 64)
            else:
                pos_opp = torch.zeros(batch_size, 64, device=device)
            value_input = torch.cat([features, pos_embed, pos_opp], dim=-1)
            value_logits = self.value_head(value_input)
        else:
            # Default: use position 0 (will be masked anyway if not in GUESS phase)
            default_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
            pos_embed = self.position_embedding(default_pos)
            if opponent_per_pos is not None:
                pos_opp = opponent_per_pos[:, 0]
            else:
                pos_opp = torch.zeros(batch_size, 64, device=device)
            value_input = torch.cat([features, pos_embed, pos_opp], dim=-1)
            value_logits = self.value_head(value_input)
        
        # Apply action masks if provided (use MASK_VALUE for numerical stability)
        if action_mask is not None:
            if "color" in action_mask:
                mask = action_mask["color"].to(device)
                color_logits = color_logits.masked_fill(~mask, MASK_VALUE)
            if "position" in action_mask:
                mask = action_mask["position"].to(device)
                position_logits = position_logits.masked_fill(~mask, MASK_VALUE)
            if "value" in action_mask:
                mask = action_mask["value"].to(device)
                if mask.dim() == 3 and selected_position is not None:  # per-position mask
                    batch_indices = torch.arange(batch_size, device=device)
                    mask = mask[batch_indices, selected_position]  # (batch, 13)
                value_logits = value_logits.masked_fill(~mask, MASK_VALUE)
            if "decision" in action_mask:
                mask = action_mask["decision"].to(device)
                decision_logits = decision_logits.masked_fill(~mask, MASK_VALUE)
        
        # Phase gating: mask out irrelevant heads based on current phase
        # phase[:, 0] = DRAW, phase[:, 1] = GUESS, phase[:, 2] = DECISION
        draw_active = phase[:, 0:1].bool()  # (batch, 1)
        guess_active = phase[:, 1:2].bool()
        decision_active = phase[:, 2:3].bool()
        
        # Apply phase gating (use MASK_VALUE for numerical stability)
        color_logits = torch.where(
            draw_active.expand_as(color_logits),
            color_logits,
            torch.full_like(color_logits, MASK_VALUE)
        )
        position_logits = torch.where(
            guess_active.expand_as(position_logits),
            position_logits,
            torch.full_like(position_logits, MASK_VALUE)
        )
        value_logits = torch.where(
            guess_active.expand_as(value_logits),
            value_logits,
            torch.full_like(value_logits, MASK_VALUE)
        )
        decision_logits = torch.where(
            decision_active.expand_as(decision_logits),
            decision_logits,
            torch.full_like(decision_logits, MASK_VALUE)
        )
        
        return {
            "color": color_logits,
            "position": position_logits,
            "value": value_logits,
            "decision": decision_logits
        }


class ValueHead(nn.Module):
    """
    Value function head for actor-critic methods.
    
    Estimates the state value V(s) for variance reduction in policy gradient.
    Uses LayerNorm for training stability.
    """
    
    def __init__(self, hidden_dim: int = 512) -> None:
        """
        Initialize value head.
        
        Args:
            hidden_dim: Input feature dimension
        """
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            features: Encoded observation features
            
        Returns:
            State value tensor (batch, 1)
        """
        return self.value_net(features)


class DaVinciCodePolicy(nn.Module):
    """
    Complete policy network for Da Vinci Code self-play.
    
    Architecture:
    - Observation Encoder: Processes dict observation into features
    - Phase-Gated Action Heads: Multi-head output with phase gating
    - Position-Conditioned Value Head: For autoregressive action selection
    - Value Head: For actor-critic training
    
    Key improvements:
    - Autoregressive: Value prediction is conditioned on selected position
    - Better constraint utilization through per-position features
    - Numerically stable operations (no -inf, uses MASK_VALUE)
    - LayerNorm throughout for training stability
    """
    
    def __init__(self, hidden_dim: int = 512, n_heads: int = 4, n_layers: int = 4,
                 zero_init: bool = False) -> None:
        """
        Initialize the policy network.

        Args:
            hidden_dim: Hidden layer dimension throughout the network
            n_heads: Number of attention heads in the encoder transformer
            n_layers: Number of transformer encoder layers
            zero_init: If True, every default-initialized weight/bias starts at 0
                       instead of orthogonal/normal. Explicitly designated inits
                       (belief_to_opp_proj zeros, cls_token) are kept as-is.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self._zero_init = zero_init

        self.encoder = ObservationEncoder(hidden_dim, n_heads=n_heads, n_layers=n_layers)
        self.action_heads = PhaseGatedActionHead(hidden_dim)
        self.value_head = ValueHead(hidden_dim)
        
        # Belief head: predict opponent's hidden card values
        # Input: global features (hidden_dim) + per-position features (64d) concat → (batch, 13, 13)
        # Global context enables cross-card reasoning: "B-5 in my hand → opp hidden(B) ≠ 5"
        self.belief_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_VALUES)
        )
        # Projects belief probability distribution (13d) to opponent feature space (64d) as residual
        # Zero-initialized: enrichment starts at 0, grows as belief quality improves
        self.belief_to_opp_proj = nn.Linear(NUM_VALUES, 64)

        # Initialize weights with orthogonal initialization for better training
        self.apply(self._init_weights)
        # Override: zero-init belief_to_opp_proj so enrichment starts neutral → preserves checkpoint
        nn.init.zeros_(self.belief_to_opp_proj.weight)
        nn.init.zeros_(self.belief_to_opp_proj.bias)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights.

        Default: orthogonal (Linear/Conv) + normal(0, 0.02) (Embedding).
        If ``self._zero_init``: every weight and bias starts at 0 instead — unless
        a specific value is designated elsewhere (e.g. belief_to_opp_proj, cls_token).
        """
        if self._zero_init:
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                nn.init.zeros_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            return
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _enrich_opp_with_belief(
        self, features: torch.Tensor, opponent_per_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute belief predictions and inject into opponent features as a residual.

        Gradient routing (the belief CE loss is applied to belief_logits elsewhere):
          - The global `features` input is DETACHED, so the CE gradient does NOT flow
            into the encoder through the global-context path.
          - `opponent_per_pos` is NOT detached, so the CE gradient DOES reach the encoder,
            but only through the per-slot opponent features (a deliberate "half-detach").
          This shapes the per-slot representations for belief while shielding the global
          features (which value_head depends on) from the competing belief signal.

        belief_probs are detached before being passed to action heads so the RL
        (policy/value) gradient cannot back-propagate through belief_head.

        belief_to_opp_proj is zero-initialized: enrichment starts at 0 and grows
        naturally as belief quality improves. This preserves loaded checkpoint behaviour.

        Returns:
            enriched_opp:  (batch, 13, 64) — same shape as opponent_per_pos
            belief_logits: (batch, 13, 13) — raw logits for CE auxiliary loss
        """
        # Detach the GLOBAL features before belief_head: the belief CE gradient must not
        # reach the encoder through the global-context path. The global features are shared
        # with value_head; competing gradients (game outcome vs. card distribution) would
        # push them in conflicting directions and inflate value_loss.
        # Detaching here lets belief still USE global context in the forward pass
        # (prediction quality unchanged) while protecting the global features.
        global_exp = features.detach().unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)
        # Combine with per-position features (NOT detached): (batch, 13, hidden_dim+64)
        combined = torch.cat([global_exp, opponent_per_pos], dim=-1)
        # Belief logits. CE gradient flows through belief_head AND into the encoder via
        # opponent_per_pos only (the global path above is detached) — a deliberate half-detach.
        belief_logits = self.belief_head(combined)          # (batch, 13, 13)
        # Detach before action heads: RL signal must not contaminate belief head
        belief_probs = belief_logits.detach().softmax(dim=-1)  # (batch, 13, 13)
        # Project to opponent feature space and add as residual
        belief_residual = self.belief_to_opp_proj(belief_probs)  # (batch, 13, 64)
        return opponent_per_pos + belief_residual, belief_logits
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[Dict[str, torch.Tensor]] = None,
        selected_position: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation dictionary
            action_mask: Optional action masks
            selected_position: Position for value head conditioning
            
        Returns:
            Tuple of (action_logits_dict, state_value, constraint_per_pos)
        """
        # Encode observation
        features, constraint_per_pos, opponent_per_pos = self.encoder(obs)

        # Enrich opponent features with belief predictions
        enriched_opp, _ = self._enrich_opp_with_belief(features, opponent_per_pos)

        # Get action logits with phase gating
        action_logits = self.action_heads(
            features, 
            obs["phase"].float(), 
            action_mask,
            selected_position,
            opponent_per_pos=enriched_opp
        )
        
        # Get state value
        value = self.value_head(features)
        
        return action_logits, value, constraint_per_pos
    
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Sample action from policy with autoregressive value selection.
        
        For GUESS phase: First sample position, then sample value conditioned on position.
        Optimized to minimize redundant forward passes.
        
        Args:
            obs: Observation dictionary
            action_mask: Optional action masks
            deterministic: If True, take argmax instead of sampling
            
        Returns:
            Tuple of (action_array, log_probs_dict, state_value)
        """
        batch_size = obs["phase"].size(0)
        device = obs["phase"].device
        phase = obs["phase"]
        
        # Encode observation (only once)
        features, constraint_per_pos, opponent_per_pos = self.encoder(obs)

        # Enrich opponent features with belief predictions
        enriched_opp, _ = self._enrich_opp_with_belief(features, opponent_per_pos)

        # Get state value (only uses features, no need to recompute)
        value = self.value_head(features)
        
        actions = {}
        log_probs = {}
        
        # Check if we're in GUESS phase (need autoregressive sampling)
        guess_active = phase[:, 1].bool()
        any_guess = guess_active.any()
        
        # Compute heads that don't need position conditioning
        color_logits = self.action_heads.color_head(features)
        
        # Position head with per-position features (using belief-enriched opponent)
        features_expanded = features.unsqueeze(1).expand(-1, MAX_HAND_SIZE, -1)  # (batch, 13, hidden_dim)
        position_input = torch.cat([features_expanded, enriched_opp], dim=-1)
        position_logits = self.action_heads.position_head(position_input).squeeze(-1)  # (batch, 13)
        
        decision_logits = self.action_heads.decision_head(features)
        
        # Apply action masks
        if action_mask is not None:
            if "color" in action_mask:
                mask = action_mask["color"].to(device)
                color_logits = color_logits.masked_fill(~mask, MASK_VALUE)
            if "position" in action_mask:
                mask = action_mask["position"].to(device)
                position_logits = position_logits.masked_fill(~mask, MASK_VALUE)
            if "decision" in action_mask:
                mask = action_mask["decision"].to(device)
                decision_logits = decision_logits.masked_fill(~mask, MASK_VALUE)
        
        # Apply phase gating
        draw_active = phase[:, 0:1].bool()
        decision_active = phase[:, 2:3].bool()
        
        color_logits = torch.where(
            draw_active.expand_as(color_logits),
            color_logits,
            torch.full_like(color_logits, MASK_VALUE)
        )
        position_logits = torch.where(
            guess_active.unsqueeze(-1).expand_as(position_logits),
            position_logits,
            torch.full_like(position_logits, MASK_VALUE)
        )
        decision_logits = torch.where(
            decision_active.expand_as(decision_logits),
            decision_logits,
            torch.full_like(decision_logits, MASK_VALUE)
        )
        
        # Sample position first (for value conditioning)
        position_action, position_log_prob = self._sample_from_logits(
            position_logits, deterministic
        )
        actions["position"] = position_action
        log_probs["position"] = position_log_prob
        
        # Compute value logits with position conditioning (only if in GUESS phase)
        if any_guess:
            pos_embed = self.action_heads.position_embedding(position_action)
            batch_indices = torch.arange(batch_size, device=device)
            pos_opp = enriched_opp[batch_indices, position_action]  # (batch, 64)
            value_input = torch.cat([features, pos_embed, pos_opp], dim=-1)
            value_logits = self.action_heads.value_head(value_input)
            
            # Apply value mask (per-position: select row for chosen position)
            if action_mask is not None and "value" in action_mask:
                mask = action_mask["value"].to(device)
                if mask.dim() == 3:  # (batch, 13, 13) per-position mask
                    batch_indices = torch.arange(batch_size, device=device)
                    mask = mask[batch_indices, position_action]  # (batch, 13)
                value_logits = value_logits.masked_fill(~mask, MASK_VALUE)
            
            # Apply phase gating
            value_logits = torch.where(
                guess_active.unsqueeze(-1).expand_as(value_logits),
                value_logits,
                torch.full_like(value_logits, MASK_VALUE)
            )
        else:
            # Not in guess phase - create dummy value logits (will be masked)
            value_logits = torch.full((batch_size, NUM_VALUES), MASK_VALUE, device=device)
        
        # Sample remaining actions
        color_action, color_log_prob = self._sample_from_logits(color_logits, deterministic)
        value_action, value_log_prob = self._sample_from_logits(value_logits, deterministic)
        decision_action, decision_log_prob = self._sample_from_logits(decision_logits, deterministic)
        
        actions["color"] = color_action
        actions["value"] = value_action
        actions["decision"] = decision_action
        log_probs["color"] = color_log_prob
        log_probs["value"] = value_log_prob
        log_probs["decision"] = decision_log_prob
        
        # Combine into action array [color, position, value, decision]
        action_array = torch.stack([
            actions["color"],
            actions["position"],
            actions["value"],
            actions["decision"]
        ], dim=-1).cpu().numpy()
        
        return action_array, log_probs, value
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from logits with numerical stability.
        
        Args:
            logits: Action logits (batch, num_actions)
            deterministic: If True, take argmax
            
        Returns:
            Tuple of (action, log_prob)
        """
        batch_size = logits.size(0)
        device = logits.device
        
        action = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch_size, device=device)
        
        # Check which samples have valid (non-masked) logits
        valid_mask = ~torch.all(logits <= MASK_VALUE + 1, dim=-1)
        
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            # Clamp for numerical stability
            safe_logits = torch.clamp(valid_logits, min=MASK_VALUE)
            probs = F.softmax(safe_logits, dim=-1)
            
            # Add small epsilon to prevent log(0)
            probs = probs + 1e-6
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            dist = Categorical(probs)
            
            if deterministic:
                valid_action = safe_logits.argmax(dim=-1)
            else:
                valid_action = dist.sample()
            
            action[valid_mask] = valid_action
            log_prob[valid_mask] = dist.log_prob(valid_action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        action_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Used during PPO update step. Uses autoregressive conditioning.
        
        Args:
            obs: Observation dictionary
            actions: Dictionary of actions taken
            action_mask: Optional action masks
            
        Returns:
            Tuple of (log_probs_dict, state_value, entropy_dict, belief_logits)
            belief_logits: (batch, 13, 13) predicted value distribution per opponent position
        """
        phase = obs["phase"]
        
        # Encode and get features
        features, constraint_per_pos, opponent_per_pos = self.encoder(obs)
        
        # Enrich opponent features with belief predictions (also returns logits for CE loss)
        enriched_opp, belief_logits = self._enrich_opp_with_belief(features, opponent_per_pos)

        # Get action logits with position conditioning for value head
        action_logits = self.action_heads(
            features,
            phase.float(),
            action_mask,
            selected_position=actions["position"],  # Use actual position taken
            opponent_per_pos=enriched_opp
        )

        # Get state value
        value = self.value_head(features)
        
        log_probs = {}
        entropies = {}
        
        for key, logits in action_logits.items():
            batch_size = logits.size(0)
            device = logits.device
            
            log_prob = torch.zeros(batch_size, device=device)
            entropy = torch.zeros(batch_size, device=device)
            
            # Check which samples have valid (non-masked) logits
            valid_mask = ~torch.all(logits <= MASK_VALUE + 1, dim=-1)
            
            if valid_mask.any():
                valid_logits = logits[valid_mask]
                # Clamp for numerical stability
                safe_logits = torch.clamp(valid_logits, min=MASK_VALUE)
                probs = F.softmax(safe_logits, dim=-1)
                
                # Add small epsilon to prevent log(0)
                probs = probs + 1e-6
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                dist = Categorical(probs)
                
                valid_actions = actions[key][valid_mask]
                log_prob[valid_mask] = dist.log_prob(valid_actions)
                entropy[valid_mask] = dist.entropy()
            
            log_probs[key] = log_prob
            entropies[key] = entropy
        
        return log_probs, value, entropies, belief_logits


def obs_to_tensor(
    obs: Dict[str, np.ndarray],
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy observation dict to tensor dict.
    
    Args:
        obs: Observation dictionary with numpy arrays
        device: Target device
        
    Returns:
        Dictionary with tensors
    """
    return {
        key: torch.from_numpy(val).unsqueeze(0).to(device)
        for key, val in obs.items()
    }


def action_mask_to_tensor(
    mask: Dict[str, np.ndarray],
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Convert numpy action mask dict to tensor dict.
    
    Args:
        mask: Action mask dictionary
        device: Target device
        
    Returns:
        Dictionary with boolean tensors
    """
    return {
        key: torch.from_numpy(val).unsqueeze(0).bool().to(device)
        for key, val in mask.items()
    }
