"""
DreamerV3-style world model networks for Da Vinci Code (joker_control variant).

Follows Hafner et al. 2023 ("Mastering Diverse Domains through World Models",
arXiv:2301.04104):

    sequence model:   h_t = f(h_{t-1}, z_{t-1}, a_{t-1})          (GRU)
    encoder (post):   z_t ~ q(z_t | h_t, x_t)                     (categorical)
    dynamics (prior): ẑ_t ~ p(ẑ_t | h_t)                          (categorical)
    heads:            decoder x̂_t, reward r̂_t, continue ĉ_t | h_t,z_t

    L = β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep
    β_pred=1.0, β_dyn=0.5, β_rep=0.1, free bits = 1 nat
    latents: categorical with straight-through gradients, 1% unimix
    reward/critic: symlog two-hot discrete regression (255 bins in [-20, 20])

Deviations from the paper (documented, deliberate):
- Latent size defaults to 16×16 (vs 32×32) — the game state is far smaller
  than Atari pixels; configurable via WMConfig.
- The decoder reconstructs the structured dict observation with per-field
  cross-entropy/Bernoulli losses instead of image MSE.
- An extra MASK head predicts the per-component action masks so that the actor
  can respect legality during imagination (the real env is not queryable there).
- The observation stream alternates player perspectives exactly like the
  existing PPO trainer's single-stream convention; the world model learns the
  perspective flip as part of the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import MAX_HAND_SIZE, NUM_VALUES

# Action component sizes: [color, position, value, decision, joker]
ACTION_SIZES = (2, 13, 13, 2, 13)
ACTION_DIM = sum(ACTION_SIZES)  # 43

# Mask component sizes (value mask is per-position 13x13, flattened)
MASK_SIZES = {"color": 2, "position": 13, "value": 13 * 13, "decision": 2, "joker": 13}


# ---------------------------------------------------------------------------
# symlog / two-hot (paper §"critic learning" — 255 bins over [-20, 20])
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHot:
    """Two-hot discrete regression over symlog space."""

    def __init__(self, n_bins: int = 255, low: float = -20.0, high: float = 20.0,
                 device: torch.device = torch.device("cpu")) -> None:
        self.bins = torch.linspace(low, high, n_bins, device=device)
        self.n_bins = n_bins

    def to(self, device: torch.device) -> "TwoHot":
        self.bins = self.bins.to(device)
        return self

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """y: (...,) raw target → (..., n_bins) two-hot weights in symlog space."""
        x = symlog(y).clamp(self.bins[0], self.bins[-1])
        idx_hi = torch.searchsorted(self.bins, x.detach())          # first bin >= x
        idx_hi = idx_hi.clamp(1, self.n_bins - 1)
        idx_lo = idx_hi - 1
        lo, hi = self.bins[idx_lo], self.bins[idx_hi]
        w_hi = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        target = torch.zeros(*x.shape, self.n_bins, device=x.device)
        target.scatter_(-1, idx_lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
        target.scatter_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., n_bins) logits → (...,) expected raw value (through symexp)."""
        probs = F.softmax(logits, dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the two-hot target. Returns per-element loss."""
        target = self.encode(y)
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WMConfig:
    deter_dim: int = 512          # GRU deterministic state h_t
    stoch_discrete: int = 16      # number of categorical distributions
    stoch_classes: int = 16       # classes per distribution
    hidden: int = 512             # MLP width for heads/encoder
    embed_dim: int = 512          # observation embedding
    unimix: float = 0.01          # 1% uniform mixture on categorical probs
    free_bits: float = 1.0        # nats, clip KL below this
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    n_bins: int = 255             # two-hot bins

    @property
    def stoch_dim(self) -> int:
        return self.stoch_discrete * self.stoch_classes

    @property
    def state_dim(self) -> int:
        return self.deter_dim + self.stoch_dim


# ---------------------------------------------------------------------------
# Observation encoder / decoder (structured dict obs, not pixels)
# ---------------------------------------------------------------------------

def _hand_onehot(hand: torch.Tensor) -> torch.Tensor:
    """(B, 13, 2) int hand → (B, 13*(3+15)) one-hot [color(3), value(15)].

    Color: BLACK=0, WHITE=1, NONE(-1)→2. Value: 0-12, HIDDEN(-1)→13, NONE(-2)→14.
    """
    colors = hand[..., 0].long().clamp(-1, 1)
    colors = torch.where(colors < 0, torch.full_like(colors, 2), colors)
    values = hand[..., 1].long().clamp(-2, 12)
    values = torch.where(values == -2, torch.full_like(values, 14), values)
    values = torch.where(values == -1, torch.full_like(values, 13), values)
    c = F.one_hot(colors, 3).float()
    v = F.one_hot(values, 15).float()
    return torch.cat([c, v], dim=-1).flatten(-2)


class ObsEncoder(nn.Module):
    """Dict observation → embedding vector (B, embed_dim)."""

    # 2 hands * 13*(3+15) + constraint 169 + phase 4 + deck 2
    IN_DIM = 2 * MAX_HAND_SIZE * 18 + MAX_HAND_SIZE * NUM_VALUES + 4 + 2

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.net = mlp(self.IN_DIM, cfg.hidden, cfg.embed_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            _hand_onehot(obs["my_hand"]),
            _hand_onehot(obs["opponent_hand"]),
            obs["constraint_matrix"].float().flatten(-2),
            obs["phase"].float(),
            symlog(obs["remaining_deck"].float()),
        ]
        return self.net(torch.cat(parts, dim=-1))


class ObsDecoder(nn.Module):
    """Model state (h, z) → reconstruction logits for every obs field."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.trunk = mlp(cfg.state_dim, cfg.hidden, cfg.hidden, layers=1)
        self.my_color = nn.Linear(cfg.hidden, MAX_HAND_SIZE * 3)
        self.my_value = nn.Linear(cfg.hidden, MAX_HAND_SIZE * 15)
        self.opp_color = nn.Linear(cfg.hidden, MAX_HAND_SIZE * 3)
        self.opp_value = nn.Linear(cfg.hidden, MAX_HAND_SIZE * 15)
        self.constraint = nn.Linear(cfg.hidden, MAX_HAND_SIZE * NUM_VALUES)
        self.phase = nn.Linear(cfg.hidden, 4)
        self.deck = nn.Linear(cfg.hidden, 2)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = F.silu(self.trunk(state))
        B = state.shape[:-1]
        return {
            "my_color": self.my_color(f).view(*B, MAX_HAND_SIZE, 3),
            "my_value": self.my_value(f).view(*B, MAX_HAND_SIZE, 15),
            "opp_color": self.opp_color(f).view(*B, MAX_HAND_SIZE, 3),
            "opp_value": self.opp_value(f).view(*B, MAX_HAND_SIZE, 15),
            "constraint": self.constraint(f).view(*B, MAX_HAND_SIZE, NUM_VALUES),
            "phase": self.phase(f),
            "deck": self.deck(f),
        }

    @staticmethod
    def loss(pred: Dict[str, torch.Tensor], obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """-ln p(x|h,z) summed over fields. Returns (B, T) per-step loss."""
        # Targets remapped exactly like _hand_onehot
        def hand_targets(hand):
            colors = hand[..., 0].long().clamp(-1, 1)
            colors = torch.where(colors < 0, torch.full_like(colors, 2), colors)
            values = hand[..., 1].long().clamp(-2, 12)
            values = torch.where(values == -2, torch.full_like(values, 14), values)
            values = torch.where(values == -1, torch.full_like(values, 13), values)
            return colors, values

        my_c, my_v = hand_targets(obs["my_hand"])
        op_c, op_v = hand_targets(obs["opponent_hand"])

        def ce(logits, target):
            return F.cross_entropy(
                logits.flatten(0, -2), target.flatten(), reduction="none"
            ).view(target.shape).sum(-1)

        loss = ce(pred["my_color"], my_c) + ce(pred["my_value"], my_v)
        loss = loss + ce(pred["opp_color"], op_c) + ce(pred["opp_value"], op_v)
        con = obs["constraint_matrix"].float().clamp(0, 1)  # -1(empty)→0
        loss = loss + F.binary_cross_entropy_with_logits(
            pred["constraint"], con, reduction="none").sum((-1, -2))
        phase_t = obs["phase"].float().argmax(-1)
        loss = loss + F.cross_entropy(
            pred["phase"].flatten(0, -2), phase_t.flatten(), reduction="none"
        ).view(phase_t.shape)
        loss = loss + 0.5 * ((pred["deck"] - symlog(obs["remaining_deck"].float())) ** 2).sum(-1)
        return loss


class MaskHead(nn.Module):
    """Predict per-component action masks from model state (Bernoulli logits).

    Needed because imagination cannot query env.get_action_mask(); trained with
    BCE on the masks recorded during real interaction.
    """

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.total = sum(MASK_SIZES.values())
        self.net = mlp(cfg.state_dim, cfg.hidden, self.total, layers=1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(state)
        masks, i = {}, 0
        for key, size in MASK_SIZES.items():
            masks[key] = out[..., i:i + size]
            i += size
        return masks

    @staticmethod
    def loss(pred: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for key in MASK_SIZES:
            target = masks[key].float().flatten(-2) if masks[key].dim() > pred[key].dim() \
                else masks[key].float()
            loss = loss + F.binary_cross_entropy_with_logits(
                pred[key], target, reduction="none").sum(-1)
        return loss


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------

def action_to_onehot(action: torch.Tensor) -> torch.Tensor:
    """(..., 5) int action → (..., 43) concatenated one-hot."""
    parts = [F.one_hot(action[..., i].long().clamp(0, n - 1), n).float()
             for i, n in enumerate(ACTION_SIZES)]
    return torch.cat(parts, dim=-1)


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical latents (DreamerV3)."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru_in = mlp(cfg.stoch_dim + ACTION_DIM, cfg.hidden, cfg.hidden, layers=1)
        self.gru = nn.GRUCell(cfg.hidden, cfg.deter_dim)
        self.prior_net = mlp(cfg.deter_dim, cfg.hidden,
                             cfg.stoch_discrete * cfg.stoch_classes, layers=1)
        self.post_net = mlp(cfg.deter_dim + cfg.embed_dim, cfg.hidden,
                            cfg.stoch_discrete * cfg.stoch_classes, layers=1)

    # -- categorical latent helpers -------------------------------------

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, D*C) → (B, D, C) probs with 1% unimix."""
        logits = logits.view(*logits.shape[:-1], self.cfg.stoch_discrete, self.cfg.stoch_classes)
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.cfg.stoch_classes
        return (1 - self.cfg.unimix) * probs + self.cfg.unimix * uniform

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through categorical sample → flattened one-hot (B, D*C)."""
        idx = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.cfg.stoch_classes).float()
        onehot = onehot + probs - probs.detach()  # straight-through
        return onehot.flatten(-2)

    @staticmethod
    def _kl(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
        """KL[p || q] summed over the D categorical distributions → (B,)."""
        kl = (p_probs * (torch.log(p_probs + 1e-8) - torch.log(q_probs + 1e-8))).sum(-1)
        return kl.sum(-1)

    # -- state transitions ----------------------------------------------

    def initial(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.cfg.deter_dim, device=device)
        z = torch.zeros(batch, self.cfg.stoch_dim, device=device)
        return h, z

    def step_deter(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """h_t = f(h_{t-1}, z_{t-1}, a_{t-1}). action: (B, 5) ints."""
        a = action_to_onehot(action)
        return self.gru(self.gru_in(torch.cat([z, a], dim=-1)), h)

    def posterior(self, h: torch.Tensor, embed: torch.Tensor):
        probs = self._logits_to_probs(self.post_net(torch.cat([h, embed], dim=-1)))
        return probs, self._sample(probs)

    def prior(self, h: torch.Tensor):
        probs = self._logits_to_probs(self.prior_net(h))
        return probs, self._sample(probs)

    def kl_losses(self, post_probs: torch.Tensor, prior_probs: torch.Tensor):
        """Returns (L_dyn, L_rep) with free-bits clipping, per sample."""
        fb = self.cfg.free_bits
        l_dyn = self._kl(post_probs.detach(), prior_probs).clamp(min=fb)
        l_rep = self._kl(post_probs, prior_probs.detach()).clamp(min=fb)
        return l_dyn, l_rep


# ---------------------------------------------------------------------------
# Actor / Critic over model states
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Policy over the 5-component action space, conditioned on (h, z).

    The active component is selected by the phase (decoded or real); inactive
    components are still sampled (the env ignores them) but only the active
    one contributes to the log-prob / entropy, mirroring the PPO trainer.
    """

    PHASE_TO_KEYS = {0: ("color",), 1: ("position", "value"), 2: ("decision",), 3: ("joker",)}
    KEY_INDEX = {"color": 0, "position": 1, "value": 2, "decision": 3, "joker": 4}

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.trunk = mlp(cfg.state_dim, cfg.hidden, cfg.hidden, layers=1)
        self.heads = nn.ModuleDict({
            "color": nn.Linear(cfg.hidden, 2),
            "position": nn.Linear(cfg.hidden, 13),
            "value": nn.Linear(cfg.hidden, 13),
            "decision": nn.Linear(cfg.hidden, 2),
            "joker": nn.Linear(cfg.hidden, 13),
        })

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = F.silu(self.trunk(state))
        return {k: head(f) for k, head in self.heads.items()}

    def dist(
        self,
        state: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.distributions.Categorical]:
        """Per-component categorical distributions, mask-aware.

        masks values are boolean (B, n) — for "value" a (B, 13, 13) per-position
        mask is reduced with `any` over positions (the position is sampled first
        in the real pipeline; inside imagination we use the coarse union).
        """
        logits = self(state)
        dists = {}
        for key, logit in logits.items():
            if masks is not None and key in masks:
                m = masks[key]
                if key == "value" and m.shape[-1] == 13 * 13:
                    # flat (B, 169) from MaskHead → (B, 13, 13)
                    m = m.view(*m.shape[:-1], 13, 13)
                if m.dim() == logit.dim() + 1:  # (B, 13, 13) value mask → union
                    m = m.any(dim=-2)
                logit = logit.masked_fill(~m.bool(), -1e4)
                # safety: if a row is fully masked, unmask it (uniform)
                dead = (~m.bool()).all(-1, keepdim=True)
                logit = torch.where(dead, torch.zeros_like(logit), logit)
            dists[key] = torch.distributions.Categorical(logits=logit)
        return dists

    def sample(self, state, masks=None, phase_idx: Optional[torch.Tensor] = None):
        """Sample a full (B, 5) action; log-prob/entropy from the ACTIVE head only."""
        dists = self.dist(state, masks)
        comps = []
        for key in ("color", "position", "value", "decision", "joker"):
            comps.append(dists[key].sample())
        action = torch.stack(comps, dim=-1)

        if phase_idx is None:
            phase_idx = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
        log_prob = torch.zeros(state.shape[0], device=state.device)
        entropy = torch.zeros(state.shape[0], device=state.device)
        for p, keys in self.PHASE_TO_KEYS.items():
            sel = phase_idx == p
            if not sel.any():
                continue
            for key in keys:
                i = self.KEY_INDEX[key]
                log_prob = log_prob + torch.where(
                    sel, dists[key].log_prob(action[..., i]), torch.zeros_like(log_prob))
                entropy = entropy + torch.where(
                    sel, dists[key].entropy(), torch.zeros_like(entropy))
        return action, log_prob, entropy

    def evaluate(self, state, action, masks=None, phase_idx: Optional[torch.Tensor] = None):
        """Log-prob/entropy of a GIVEN (B, 5) action — used to attach gradients
        to actions that were sampled under no_grad during imagination."""
        dists = self.dist(state, masks)
        if phase_idx is None:
            phase_idx = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
        log_prob = torch.zeros(state.shape[0], device=state.device)
        entropy = torch.zeros(state.shape[0], device=state.device)
        for p, keys in self.PHASE_TO_KEYS.items():
            sel = phase_idx == p
            if not sel.any():
                continue
            for key in keys:
                i = self.KEY_INDEX[key]
                log_prob = log_prob + torch.where(
                    sel, dists[key].log_prob(action[..., i]), torch.zeros_like(log_prob))
                entropy = entropy + torch.where(
                    sel, dists[key].entropy(), torch.zeros_like(entropy))
        return log_prob, entropy


class Critic(nn.Module):
    """Two-hot symlog value head (255 bins)."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.net = mlp(cfg.state_dim, cfg.hidden, cfg.n_bins)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Full world model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """RSSM + encoder + all heads, with the DreamerV3 training loss."""

    def __init__(self, cfg: WMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ObsEncoder(cfg)
        self.rssm = RSSM(cfg)
        self.decoder = ObsDecoder(cfg)
        self.reward_head = mlp(cfg.state_dim, cfg.hidden, cfg.n_bins)
        self.continue_head = mlp(cfg.state_dim, cfg.hidden, 1)
        self.mask_head = MaskHead(cfg)
        # Perspective-flip head: P(acting player changed between s_{t-1} and s_t).
        # Needed for zero-sum (negamax) returns inside imagination, where the
        # real env can't tell us whose turn it is.
        self.flip_head = mlp(cfg.state_dim, cfg.hidden, 1)
        self.twohot = TwoHot(cfg.n_bins)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.twohot.to(device)
        return module

    def observe(
        self,
        obs_seq: Dict[str, torch.Tensor],   # each (B, T, ...)
        action_seq: torch.Tensor,           # (B, T, 5) — action TAKEN AT step t
        is_first: torch.Tensor,             # (B, T) bool — episode starts
    ):
        """Run the posterior along a real sequence.

        Convention: h_t is computed from (h_{t-1}, z_{t-1}, a_{t-1}) where
        a_{t-1} is the action taken at the previous step; at episode starts the
        state (and previous action) is reset to zeros.
        """
        B, T = action_seq.shape[:2]
        device = action_seq.device
        h, z = self.rssm.initial(B, device)
        prev_action = torch.zeros(B, 5, dtype=torch.long, device=device)

        embeds = self.encoder({k: v for k, v in obs_seq.items()})  # (B, T, E)

        posts, priors, states = [], [], []
        for t in range(T):
            first = is_first[:, t].float().unsqueeze(-1)
            h = h * (1 - first)
            z = z * (1 - first)
            prev_action = (prev_action.float() * (1 - first)).long()

            h = self.rssm.step_deter(h, z, prev_action)
            prior_probs, _ = self.rssm.prior(h)
            post_probs, z = self.rssm.posterior(h, embeds[:, t])

            posts.append(post_probs)
            priors.append(prior_probs)
            states.append(torch.cat([h, z], dim=-1))
            prev_action = action_seq[:, t]

        return (torch.stack(posts, 1), torch.stack(priors, 1), torch.stack(states, 1))

    def loss(
        self,
        obs_seq: Dict[str, torch.Tensor],
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,        # (B, T)
        continue_seq: torch.Tensor,      # (B, T) 1.0 while game continues
        is_first: torch.Tensor,          # (B, T)
        mask_seq: Optional[Dict[str, torch.Tensor]] = None,
        valid: Optional[torch.Tensor] = None,   # (B, T) 1.0 for real steps
        flip_seq: Optional[torch.Tensor] = None,  # (B, T) perspective-change flags
    ):
        post, prior, states = self.observe(obs_seq, action_seq, is_first)
        B, T = action_seq.shape[:2]
        if valid is None:
            valid = torch.ones(B, T, device=states.device)

        def wmean(x, w):
            return (x * w).sum() / w.sum().clamp(min=1.0)

        dec = self.decoder(states)
        obs_m = wmean(ObsDecoder.loss(dec, obs_seq), valid)

        # Reward/continue are consequences of a_t, but a_t is only encoded in
        # the NEXT state (h_{t+1} = f(h_t, z_t, a_t)) — training them on s_t
        # would ask the model to predict an outcome of an action it hasn't
        # seen. Train on the arriving state instead; the terminal frame the
        # replay appends makes the final win/lose reward learnable too.
        valid2 = valid[:, 1:] * valid[:, :-1]
        states_next = states[:, 1:]
        l_rew = self.twohot.loss(self.reward_head(states_next), reward_seq[:, :-1])
        l_cont = F.binary_cross_entropy_with_logits(
            self.continue_head(states_next).squeeze(-1), continue_seq[:, :-1],
            reduction="none")
        rew_m = wmean(l_rew, valid2)
        cont_m = wmean(l_cont, valid2)

        pred_m = obs_m + rew_m + cont_m
        if mask_seq is not None:
            pred_m = pred_m + wmean(MaskHead.loss(self.mask_head(states), mask_seq), valid)
        flip_m = torch.tensor(0.0, device=states.device)
        if flip_seq is not None:
            l_flip = F.binary_cross_entropy_with_logits(
                self.flip_head(states).squeeze(-1), flip_seq, reduction="none")
            flip_m = wmean(l_flip, valid)
            pred_m = pred_m + flip_m

        l_dyn, l_rep = self.rssm.kl_losses(post, prior)
        dyn_m = wmean(l_dyn, valid)
        rep_m = wmean(l_rep, valid)

        total = (self.cfg.beta_pred * pred_m
                 + self.cfg.beta_dyn * dyn_m
                 + self.cfg.beta_rep * rep_m)
        metrics = {
            "wm/loss": float(total.detach()),
            "wm/obs": float(obs_m.detach()),
            "wm/reward": float(rew_m.detach()),
            "wm/continue": float(cont_m.detach()),
            "wm/flip": float(flip_m.detach()),
            "wm/kl_dyn": float(dyn_m.detach()),
            "wm/kl_rep": float(rep_m.detach()),
        }
        return total, states.detach(), metrics

    # -- imagination -----------------------------------------------------

    @torch.no_grad()
    def decode_phase(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)["phase"].argmax(-1)

    @torch.no_grad()
    def predict_masks(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mask_head(state)
        return {k: (v > 0) for k, v in logits.items()}

    def imagine(self, actor: "Actor", start_states: torch.Tensor, horizon: int):
        """Roll the dynamics forward with the actor (no real env).

        start_states: (N, state_dim) posterior states from a real batch.
        Returns dict of stacked tensors over (H+1, N): states, plus (H, N)
        actions/log_probs/entropies and (H+1, N) predicted reward/continue.
        """
        cfg = self.cfg
        h, z = (start_states[:, :cfg.deter_dim].contiguous(),
                start_states[:, cfg.deter_dim:].contiguous())
        states = [start_states]
        actions, log_probs, entropies = [], [], []

        state = start_states
        for _ in range(horizon):
            phase_idx = self.decode_phase(state)
            masks = self.predict_masks(state)
            action, lp, ent = actor.sample(state, masks, phase_idx)
            h = self.rssm.step_deter(h, z, action)
            _, z = self.rssm.prior(h)
            state = torch.cat([h, z], dim=-1)
            states.append(state)
            actions.append(action)
            log_probs.append(lp)
            entropies.append(ent)

        states = torch.stack(states)                 # (H+1, N, S)
        rewards = self.twohot.decode(self.reward_head(states))     # (H+1, N)
        conts = torch.sigmoid(self.continue_head(states)).squeeze(-1)
        flips = torch.sigmoid(self.flip_head(states)).squeeze(-1)  # P(perspective flipped)
        return {
            "states": states,
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies),
            "rewards": rewards,
            "continues": conts,
            "flips": flips,
        }
