#!/usr/bin/env python3
"""
Interpretability Analysis for Da Vinci Code RL Agent.

Does the self-play PPO agent actually learn logical deduction?
This script answers that question by:
1. Extracting attention weights from the 4-layer Transformer encoder
2. Analyzing whether attention patterns correspond to known reasoning strategies
3. Running probing classifiers to see if hidden representations encode belief states
4. Testing specific deductive scenarios with controlled inputs

Usage: cd training && python analyze_model.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.model import DaVinciCodePolicy, ObservationEncoder
from src.env import DaVinciCodeEnv
from src.constants import Phase, Color, CardValue, MAX_HAND_SIZE, NUM_VALUES

# ========== CONFIG ==========
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GAME_SAMPLES = 500
SEED_START = 42

# ========== ATTENTION EXTRACTION ==========

class AttentionExtractor:
    """
    Attention weight extractor for nn.TransformerEncoder.
    
    PyTorch's TransformerEncoderLayer._sa_block hardcodes need_weights=False,
    so we must override _sa_block on each layer to capture attention weights.
    """
    
    def __init__(self, model: DaVinciCodePolicy):
        self.model = model
        self.attention_weights: List[Optional[torch.Tensor]] = [None] * 4
    
    def register_hooks(self):
        """Override _sa_block on each TransformerEncoderLayer to capture attention."""
        self.attention_weights = [None] * 4
        
        # Disable PyTorch's fused fast-path which bypasses _sa_block entirely
        torch.backends.mha.set_fastpath_enabled(False)
        
        encoder = self.model.encoder.transformer
        extractor = self
        
        for i, layer in enumerate(encoder.layers):
            def make_patched_sa_block(the_layer, idx):
                def patched_sa_block(x, attn_mask, key_padding_mask, is_causal):
                    attn_output, attn_weights = the_layer.self_attn(
                        x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        is_causal=is_causal,
                        need_weights=True,
                        average_attn_weights=False
                    )
                    extractor.attention_weights[idx] = attn_weights.detach().cpu()
                    return the_layer.dropout1(attn_output)
                return patched_sa_block
            
            layer._sa_block = make_patched_sa_block(layer, i)
    
    def get_attention(self, obs: Dict[str, torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """Run encoder forward pass and return attention weights for all layers."""
        self.attention_weights = [None] * 4
        with torch.no_grad():
            self.model.encoder(obs)
        return self.attention_weights


# ========== HELPERS ==========

def load_model() -> DaVinciCodePolicy:
    """Load trained model from checkpoint."""
    model = DaVinciCodePolicy(hidden_dim=512).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys (OK if loading old checkpoint): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()
    
    timesteps = checkpoint.get("timesteps", "unknown")
    print(f"Model loaded: {CHECKPOINT_PATH} (timesteps: {timesteps})")
    return model


def make_obs_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert numpy obs to batched tensor on device."""
    return {k: torch.from_numpy(v).unsqueeze(0).to(DEVICE) for k, v in obs.items()}


def make_mask_tensor(mask: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert numpy action mask to batched bool tensor on device."""
    return {k: torch.from_numpy(v).unsqueeze(0).bool().to(DEVICE) for k, v in mask.items()}


# ========== EXPERIMENT 1: ATTENTION PATTERN ANALYSIS ==========

def analyze_attention_patterns(model: DaVinciCodePolicy, num_games: int = NUM_GAME_SAMPLES):
    """
    Play many games, collect attention weights at GUESS phase,
    and analyze whether attention patterns match known deductive reasoning.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    n_samples = 0
    
    layer_stats = {i: {
        'my_to_opp': [], 'my_to_my': [], 'opp_to_my': [], 'opp_to_opp': [],
        'phase_to_opp': [], 'phase_to_my': [],
        'same_color': [], 'diff_color': [],
        'to_hidden': [], 'to_revealed': [],
        'opp_to_opp_adj': [], 'opp_to_opp_nonadj': [],
    } for i in range(4)}
    
    for game_idx in range(num_games):
        env = DaVinciCodeEnv(seed=SEED_START + game_idx)
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_t = make_obs_tensor(obs)
            mask = env.get_action_mask()
            mask_t = make_mask_tensor(mask)
            
            phase = obs["phase"]
            
            if phase[1] == 1:  # GUESS phase
                attn_layers = extractor.get_attention(obs_t)
                
                if all(a is not None for a in attn_layers):
                    n_samples += 1
                    
                    my_hand = obs["my_hand"]
                    opp_hand = obs["opponent_hand"]
                    
                    my_active = [i for i in range(13) if my_hand[i, 0] >= 0]
                    opp_active = [i for i in range(13) if opp_hand[i, 0] >= 0]
                    opp_hidden = [i for i in opp_active if opp_hand[i, 1] == int(CardValue.HIDDEN)]
                    opp_revealed = [i for i in opp_active if opp_hand[i, 1] != int(CardValue.HIDDEN)]
                    
                    my_colors = {i: int(my_hand[i, 0]) for i in my_active}
                    opp_colors = {i: int(opp_hand[i, 0]) for i in opp_active}
                    
                    for layer_idx, attn in enumerate(attn_layers):
                        a = attn[0].mean(dim=0).numpy()  # (28, 28)
                        
                        if my_active and opp_active:
                            opp_idx = [j + 13 for j in opp_active]
                            
                            m2o = a[np.ix_(my_active, opp_idx)].mean()
                            layer_stats[layer_idx]['my_to_opp'].append(float(m2o))
                            
                            if len(my_active) > 1:
                                m2m = a[np.ix_(my_active, my_active)].mean()
                                layer_stats[layer_idx]['my_to_my'].append(float(m2m))
                            
                            o2m = a[np.ix_(opp_idx, my_active)].mean()
                            layer_stats[layer_idx]['opp_to_my'].append(float(o2m))
                            
                            if len(opp_active) > 1:
                                o2o = a[np.ix_(opp_idx, opp_idx)].mean()
                                layer_stats[layer_idx]['opp_to_opp'].append(float(o2o))
                            
                            p2o = a[26, opp_idx].mean()
                            p2m = a[26, my_active].mean()
                            layer_stats[layer_idx]['phase_to_opp'].append(float(p2o))
                            layer_stats[layer_idx]['phase_to_my'].append(float(p2m))
                        
                        if my_active and opp_active:
                            sc_vals = []
                            dc_vals = []
                            for mi in my_active:
                                for oi in opp_active:
                                    val = float(a[mi, oi + 13])
                                    if my_colors.get(mi) == opp_colors.get(oi):
                                        sc_vals.append(val)
                                    else:
                                        dc_vals.append(val)
                            if sc_vals:
                                layer_stats[layer_idx]['same_color'].append(np.mean(sc_vals))
                            if dc_vals:
                                layer_stats[layer_idx]['diff_color'].append(np.mean(dc_vals))
                        
                        if my_active:
                            if opp_hidden:
                                h_idx = [j + 13 for j in opp_hidden]
                                th = a[np.ix_(my_active, h_idx)].mean()
                                layer_stats[layer_idx]['to_hidden'].append(float(th))
                            if opp_revealed:
                                r_idx = [j + 13 for j in opp_revealed]
                                tr = a[np.ix_(my_active, r_idx)].mean()
                                layer_stats[layer_idx]['to_revealed'].append(float(tr))
                        
                        if len(opp_active) >= 3:
                            adj_vals = []
                            nonadj_vals = []
                            for k_i, oi in enumerate(opp_active):
                                for k_j, oj in enumerate(opp_active):
                                    if k_i == k_j:
                                        continue
                                    val = float(a[oi + 13, oj + 13])
                                    if abs(k_i - k_j) == 1:
                                        adj_vals.append(val)
                                    else:
                                        nonadj_vals.append(val)
                            if adj_vals:
                                layer_stats[layer_idx]['opp_to_opp_adj'].append(np.mean(adj_vals))
                            if nonadj_vals:
                                layer_stats[layer_idx]['opp_to_opp_nonadj'].append(np.mean(nonadj_vals))
            
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t, mask_t, deterministic=True)
            
            obs, _render_obs, reward, done, truncated, info, result = env.step(action[0])
        
        if (game_idx + 1) % 100 == 0:
            print(f"  Games processed: {game_idx + 1}/{num_games}, GUESS samples: {n_samples}")
    
    print(f"\nTotal GUESS-phase samples collected: {n_samples}")
    
    print("\n" + "-" * 60)
    print("ATTENTION PATTERN RESULTS (averaged across heads)")
    print("-" * 60)
    
    results = {}
    
    for layer in range(4):
        stats = layer_stats[layer]
        print(f"\n--- Layer {layer} ---")
        
        layer_results = {}
        
        def report(name, key, compare_key=None, signal_label="", threshold=1.10):
            if stats.get(key):
                v = np.mean(stats[key])
                layer_results[key] = float(v)
                line = f"  {name:35s} {v:.6f}"
                if compare_key and stats.get(compare_key):
                    v2 = np.mean(stats[compare_key])
                    ratio = v / v2 if v2 > 1e-10 else float('inf')
                    layer_results[f'{key}_vs_{compare_key}'] = float(ratio)
                    sig = f'*** {signal_label} ***' if ratio > threshold else '(no clear signal)'
                    line += f"  ratio={ratio:.4f} {sig}"
                print(line)
        
        report("My->Opp attention:", 'my_to_opp')
        report("My->My attention:", 'my_to_my')
        report("Opp->My attention:", 'opp_to_my')
        report("Opp->Opp attention:", 'opp_to_opp')
        report("Phase->Opp attention:", 'phase_to_opp')
        report("Phase->My attention:", 'phase_to_my')
        print()
        report("Same-color attn (My->Opp):", 'same_color', 'diff_color', 'ELIMINATION', 1.10)
        report("Diff-color attn (My->Opp):", 'diff_color')
        print()
        report("Attn to hidden opp:", 'to_hidden', 'to_revealed', 'STRATEGIC FOCUS', 1.10)
        report("Attn to revealed opp:", 'to_revealed')
        print()
        report("Opp adj attention:", 'opp_to_opp_adj', 'opp_to_opp_nonadj', 'ORDERING', 1.10)
        report("Opp non-adj attention:", 'opp_to_opp_nonadj')
        
        results[f'layer_{layer}'] = layer_results
    
    print("\n--- Cross-Layer Summary ---")
    comparisons = [
        ('same_color', 'diff_color', 'Color Elimination'),
        ('to_hidden', 'to_revealed', 'Strategic Focus'),
        ('opp_to_opp_adj', 'opp_to_opp_nonadj', 'Ordering Reasoning')
    ]
    for metric, comp, label in comparisons:
        vals = []
        for layer in range(4):
            if layer_stats[layer].get(metric) and layer_stats[layer].get(comp):
                ratio = np.mean(layer_stats[layer][metric]) / max(np.mean(layer_stats[layer][comp]), 1e-10)
                vals.append((layer, ratio))
        if vals:
            best_layer, best_ratio = max(vals, key=lambda x: x[1])
            all_ratios = ", ".join(f"L{l}={r:.3f}" for l, r in vals)
            sig = "YES" if best_ratio > 1.10 else "NO"
            print(f"  {label:25s}: best={best_ratio:.4f} (layer {best_layer}) [{all_ratios}] Signal: {sig}")
    
    return results


# ========== EXPERIMENT 2: PROBING ==========

def probing_experiment(model: DaVinciCodePolicy, num_games: int = NUM_GAME_SAMPLES):
    """
    Train a linear probe on the model's internal representations
    to predict the actual hidden value of opponent's cards.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LINEAR PROBING -- Belief State Encoding")
    print("=" * 70)
    print("Question: Do hidden representations encode opponent's hidden card values?")
    print("Method: Linear probe on opponent token representations -> predict true value\n")
    
    representations = []
    true_values = []
    constraint_baselines = []
    
    for game_idx in range(num_games):
        env = DaVinciCodeEnv(seed=SEED_START + game_idx)
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_t = make_obs_tensor(obs)
            mask = env.get_action_mask()
            mask_t = make_mask_tensor(mask)
            phase = obs["phase"]
            
            if phase[1] == 1:  # GUESS phase
                with torch.no_grad():
                    features, constraint_per_pos, opp_per_pos = model.encoder(obs_t)
                
                current_player = env._current_player
                opponent = 1 - current_player
                opp_hand_real = env.players[opponent]._hand
                constraint = obs["constraint_matrix"]
                
                for i in range(len(opp_hand_real)):
                    card = opp_hand_real[i]
                    if card is not None and not card.is_revealed and i < 13:
                        rep = opp_per_pos[0, i].cpu().numpy()
                        representations.append(rep)
                        true_values.append(int(card.value))
                        
                        c_row = constraint[i]
                        n_impossible = int(c_row.sum())
                        n_possible = NUM_VALUES - n_impossible
                        constraint_baselines.append(max(n_possible, 1))
            
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t, mask_t, deterministic=True)
            obs, _ro, reward, done, truncated, info, result = env.step(action[0])
        
        if (game_idx + 1) % 100 == 0:
            print(f"  Games: {game_idx + 1}/{num_games}, samples: {len(representations)}")
    
    print(f"\nTotal probing samples: {len(representations)}")
    
    if len(representations) < 200:
        print("Not enough samples for probing. Skipping.")
        return {}
    
    X = np.array(representations)
    y = np.array(true_values)
    baselines = np.array(constraint_baselines)
    
    n = len(X)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = y[perm[:split]], y[perm[split:]]
    baselines_test = baselines[perm[split:]]
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    probe = LogisticRegression(max_iter=2000, multi_class='multinomial', C=1.0, solver='lbfgs')
    probe.fit(X_train, y_train)
    
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    classes = probe.classes_
    top3_correct = 0
    for i in range(len(y_test)):
        sorted_classes = classes[np.argsort(-y_proba[i])]
        if y_test[i] in sorted_classes[:3]:
            top3_correct += 1
    top3_acc = top3_correct / len(y_test)
    
    random_acc = 1.0 / NUM_VALUES
    constraint_baseline_acc = np.mean(1.0 / baselines_test)
    unique, counts = np.unique(y_test, return_counts=True)
    majority_acc = counts.max() / len(y_test)
    
    print(f"\n{'='*50}")
    print(f"PROBING RESULTS")
    print(f"{'='*50}")
    print(f"  Train samples:              {len(X_train)}")
    print(f"  Test samples:               {len(X_test)}")
    print(f"  Linear probe accuracy:      {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Top-3 accuracy:             {top3_acc:.4f} ({top3_acc*100:.1f}%)")
    print(f"  Random baseline (1/13):     {random_acc:.4f} ({random_acc*100:.1f}%)")
    print(f"  Majority class baseline:    {majority_acc:.4f} ({majority_acc*100:.1f}%)")
    print(f"  Constraint baseline (avg):  {constraint_baseline_acc:.4f} ({constraint_baseline_acc*100:.1f}%)")
    print(f"  Probe / Random:             {acc/random_acc:.2f}x")
    print(f"  Probe / Constraint:         {acc/constraint_baseline_acc:.2f}x")
    
    if acc > constraint_baseline_acc * 1.3:
        verdict = "MODEL ENCODES BELIEF STATE BEYOND CONSTRAINT MATRIX"
    elif acc > random_acc * 1.5:
        verdict = "MODEL ENCODES SOME BELIEF STATE (may reflect constraints)"
    else:
        verdict = "NO SIGNIFICANT BELIEF STATE ENCODING DETECTED"
    print(f"\n  >>> {verdict} <<<")
    
    print(f"\n  Per-value accuracy:")
    for v in sorted(unique):
        mask_v = y_test == v
        if mask_v.sum() > 5:
            v_acc = accuracy_score(y_test[mask_v], y_pred[mask_v])
            label = "Joker" if v == 12 else str(v)
            print(f"    Value {label:>5}: {v_acc:.3f} (n={mask_v.sum()})")
    
    # Also try probing with global features + position
    print("\n--- Probing with GLOBAL features (512-dim + position) ---")
    global_reps = []
    global_vals = []
    
    for game_idx in range(min(num_games, 200)):
        env = DaVinciCodeEnv(seed=SEED_START + 5000 + game_idx)
        obs, info = env.reset()
        done = False
        
        while not done:
            obs_t = make_obs_tensor(obs)
            mask = env.get_action_mask()
            mask_t = make_mask_tensor(mask)
            phase = obs["phase"]
            
            if phase[1] == 1:
                with torch.no_grad():
                    features, _, _ = model.encoder(obs_t)
                
                current_player = env._current_player
                opponent = 1 - current_player
                opp_hand_real = env.players[opponent]._hand
                
                for i in range(len(opp_hand_real)):
                    card = opp_hand_real[i]
                    if card is not None and not card.is_revealed and i < 13:
                        feat = features[0].cpu().numpy()
                        combined = np.concatenate([feat, [i]])
                        global_reps.append(combined)
                        global_vals.append(int(card.value))
            
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t, mask_t, deterministic=True)
            obs, _ro, reward, done, truncated, info, result = env.step(action[0])
    
    if len(global_reps) >= 200:
        X_g = np.array(global_reps)
        y_g = np.array(global_vals)
        n_g = len(X_g)
        perm_g = rng.permutation(n_g)
        split_g = int(0.8 * n_g)
        
        probe_g = LogisticRegression(max_iter=2000, multi_class='multinomial', C=1.0, solver='lbfgs')
        probe_g.fit(X_g[perm_g[:split_g]], y_g[perm_g[:split_g]])
        acc_g = accuracy_score(y_g[perm_g[split_g:]], probe_g.predict(X_g[perm_g[split_g:]]))
        print(f"  Global feature probe acc:   {acc_g:.4f} ({acc_g*100:.1f}%)")
    
    return {
        'probe_accuracy': float(acc),
        'top3_accuracy': float(top3_acc),
        'random_baseline': float(random_acc),
        'majority_baseline': float(majority_acc),
        'constraint_baseline': float(constraint_baseline_acc),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'verdict': verdict
    }


# ========== EXPERIMENT 3: CONTROLLED SCENARIOS ==========

def controlled_scenario_test(model: DaVinciCodePolicy):
    """Test specific hand-crafted scenarios where the correct deduction is known."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CONTROLLED DEDUCTION SCENARIOS")
    print("=" * 70)
    print("Test whether the model makes logically correct deductions.\n")
    
    results = {}
    
    def build_obs(my_cards, opp_cards, opp_revealed, constraint=None, remaining_deck=(5, 5)):
        my_hand = np.full((13, 2), [int(Color.NONE), int(CardValue.NONE)], dtype=np.int8)
        for i, (c, v) in enumerate(my_cards):
            my_hand[i] = [c, v]
        
        opp_hand = np.full((13, 2), [int(Color.NONE), int(CardValue.NONE)], dtype=np.int8)
        for i, (c, v) in enumerate(opp_cards):
            if opp_revealed[i]:
                opp_hand[i] = [c, v]
            else:
                opp_hand[i] = [c, int(CardValue.HIDDEN)]
        
        phase = np.array([0, 1, 0], dtype=np.int8)
        remaining = np.array(remaining_deck, dtype=np.int8)
        
        if constraint is None:
            constraint = np.zeros((13, 13), dtype=np.int8)
        
        return {
            "phase": phase,
            "my_hand": my_hand,
            "opponent_hand": opp_hand,
            "remaining_deck": remaining,
            "constraint_matrix": constraint
        }
    
    def get_model_guess(obs):
        obs_t = make_obs_tensor(obs)
        
        opp_hand = obs["opponent_hand"]
        position_mask = np.zeros(13, dtype=bool)
        for i in range(13):
            if opp_hand[i, 0] >= 0 and opp_hand[i, 1] == int(CardValue.HIDDEN):
                position_mask[i] = True
        
        color_mask = np.array([True, True], dtype=bool)
        value_mask = np.ones((13, 13), dtype=bool)
        decision_mask = np.array([True, True], dtype=bool)
        
        mask_t = {
            "color": torch.from_numpy(color_mask).unsqueeze(0).bool().to(DEVICE),
            "position": torch.from_numpy(position_mask).unsqueeze(0).bool().to(DEVICE),
            "value": torch.from_numpy(value_mask).unsqueeze(0).bool().to(DEVICE),
            "decision": torch.from_numpy(decision_mask).unsqueeze(0).bool().to(DEVICE),
        }
        
        with torch.no_grad():
            action, log_probs, value = model.get_action(obs_t, mask_t, deterministic=True)
        
        pos = int(action[0, 1])
        val = int(action[0, 2])
        return pos, val
    
    def get_value_distribution(obs, position):
        obs_t = make_obs_tensor(obs)
        
        with torch.no_grad():
            features, constraint_per_pos, opp_per_pos = model.encoder(obs_t)
            
            pos_t = torch.tensor([position], dtype=torch.long, device=DEVICE)
            pos_embed = model.action_heads.position_embedding(pos_t)
            batch_indices = torch.arange(1, device=DEVICE)
            pos_constraint = constraint_per_pos[batch_indices, position]
            value_input = torch.cat([features, pos_embed, pos_constraint], dim=-1)
            value_logits = model.action_heads.value_head(value_input)
            value_probs = torch.softmax(value_logits, dim=-1)[0].cpu().numpy()
        
        return value_probs
    
    # SCENARIO 1: Elimination by own hand
    print("--- Scenario 1: Elimination by own hand ---")
    print("I have B0-B10. Opponent has 1 hidden black card. -> B11 or Joker.\n")
    
    my = [(int(Color.BLACK), i) for i in range(11)]
    opp = [(int(Color.BLACK), 5)]
    opp_rev = [False]
    
    obs = build_obs(my, opp, opp_rev, remaining_deck=(1, 6))
    pos, val = get_model_guess(obs)
    vprobs = get_value_distribution(obs, 0)
    
    correct_vals = {11, 12}
    is_correct = val in correct_vals
    prob_correct = float(vprobs[11] + vprobs[12])
    top_vals = sorted(enumerate(vprobs), key=lambda x: -x[1])[:5]
    
    print(f"  Model guess: pos={pos}, value={val}")
    print(f"  Correct: 11 or 12(Joker). P(correct): {prob_correct:.4f}")
    print(f"  Top-5: {[(i, f'{p:.3f}') for i, p in top_vals]}")
    print(f"  -> {'PASS' if is_correct else 'FAIL'}\n")
    results['s1_elimination'] = {'correct': bool(is_correct), 'prob_correct': prob_correct, 'predicted': int(val)}
    
    # SCENARIO 2: Ordering constraint from revealed neighbors
    print("--- Scenario 2: Ordering constraint ---")
    print("Opp: [B3(rev), B?(hid), B7(rev)]. Hidden must be 4,5,6 or Joker.\n")
    
    my = [(int(Color.WHITE), 0), (int(Color.WHITE), 1)]
    opp = [(int(Color.BLACK), 3), (int(Color.BLACK), 5), (int(Color.BLACK), 7)]
    opp_rev = [True, False, True]
    
    obs = build_obs(my, opp, opp_rev, remaining_deck=(5, 5))
    pos, val = get_model_guess(obs)
    vprobs = get_value_distribution(obs, 1)
    
    valid_range = {4, 5, 6, 12}
    prob_in_range = float(sum(vprobs[v] for v in valid_range))
    is_in_range = val in valid_range
    top_vals = sorted(enumerate(vprobs), key=lambda x: -x[1])[:5]
    
    print(f"  Model guess: pos={pos}, value={val}")
    print(f"  Valid: 4,5,6,12. P(valid): {prob_in_range:.4f}")
    print(f"  Top-5: {[(i, f'{p:.3f}') for i, p in top_vals]}")
    print(f"  -> pos={'OK' if pos == 1 else 'WRONG'}, value={'PASS' if is_in_range else 'FAIL'}\n")
    results['s2_ordering'] = {'correct': bool(is_in_range), 'prob_in_range': prob_in_range, 'predicted': int(val)}
    
    # SCENARIO 3: Cross-elimination
    print("--- Scenario 3: Cross-elimination ---")
    print("I have B4,B5. Opp: [B3(rev), B?(hid), B8(rev)]. Hidden != 4,5.\n")
    
    my = [(int(Color.BLACK), 4), (int(Color.BLACK), 5)]
    opp = [(int(Color.BLACK), 3), (int(Color.BLACK), 6), (int(Color.BLACK), 8)]
    opp_rev = [True, False, True]
    
    obs = build_obs(my, opp, opp_rev, remaining_deck=(4, 6))
    pos, val = get_model_guess(obs)
    vprobs = get_value_distribution(obs, 1)
    
    valid = {6, 7, 12}
    prob_valid = float(sum(vprobs[v] for v in valid))
    prob_elim = float(vprobs[4] + vprobs[5])
    is_correct = val in valid
    top_vals = sorted(enumerate(vprobs), key=lambda x: -x[1])[:5]
    
    print(f"  Model guess: pos={pos}, value={val}")
    print(f"  Valid: 6,7,12. P(valid): {prob_valid:.4f}")
    print(f"  P(eliminated 4+5): {prob_elim:.4f} (should be ~0)")
    print(f"  Top-5: {[(i, f'{p:.3f}') for i, p in top_vals]}")
    print(f"  -> {'PASS' if is_correct else 'FAIL'}\n")
    results['s3_cross_elim'] = {'correct': bool(is_correct), 'prob_valid': prob_valid, 'prob_eliminated': prob_elim}
    
    # SCENARIO 4: Constraint matrix from failed guess
    print("--- Scenario 4: Constraint matrix (failed guess) ---")
    print("Opp has 1 hidden black card. Constraint: NOT value 3.\n")
    
    my = [(int(Color.WHITE), 0), (int(Color.WHITE), 1)]
    opp = [(int(Color.BLACK), 5)]
    opp_rev = [False]
    constraint = np.zeros((13, 13), dtype=np.int8)
    constraint[0, 3] = 1
    
    obs = build_obs(my, opp, opp_rev, constraint=constraint, remaining_deck=(6, 5))
    pos, val = get_model_guess(obs)
    vprobs = get_value_distribution(obs, 0)
    
    prob_3 = float(vprobs[3])
    print(f"  Model guess: pos={pos}, value={val}")
    print(f"  P(value=3): {prob_3:.4f} (should be ~0)")
    print(f"  -> {'PASS' if val != 3 else 'FAIL'}\n")
    results['s4_constraint'] = {'correct': bool(val != 3), 'prob_eliminated': prob_3}
    
    # SCENARIO 5: Fully determined via constraint
    print("--- Scenario 5: Fully determined via constraint ---")
    print("Constraint eliminates all except value 7.\n")
    
    my = [(int(Color.WHITE), 0), (int(Color.WHITE), 1)]
    opp = [(int(Color.BLACK), 7)]
    opp_rev = [False]
    constraint = np.zeros((13, 13), dtype=np.int8)
    for v in range(13):
        if v != 7:
            constraint[0, v] = 1
    
    obs = build_obs(my, opp, opp_rev, constraint=constraint, remaining_deck=(5, 5))
    pos, val = get_model_guess(obs)
    vprobs = get_value_distribution(obs, 0)
    
    prob_7 = float(vprobs[7])
    print(f"  Model guess: pos={pos}, value={val}")
    print(f"  P(value=7): {prob_7:.4f} (should be ~1.0)")
    print(f"  -> {'PASS' if val == 7 else 'FAIL'}\n")
    results['s5_determined'] = {'correct': bool(val == 7), 'prob_correct': prob_7}
    
    # SCENARIO 6: Multiple hidden with varying info
    print("--- Scenario 6: Multiple hidden with revealed neighbors ---")
    print("Opp: [B0(rev), B?(hid), B?(hid), B11(rev)]")
    print("Pos 1 and 2 both hidden. Testing position choice.\n")
    
    my = [(int(Color.WHITE), 5)]
    opp = [
        (int(Color.BLACK), 0),
        (int(Color.BLACK), 3),
        (int(Color.BLACK), 8),
        (int(Color.BLACK), 11),
    ]
    opp_rev = [True, False, False, True]
    
    obs = build_obs(my, opp, opp_rev, remaining_deck=(4, 5))
    pos, val = get_model_guess(obs)
    vprobs_1 = get_value_distribution(obs, 1)
    vprobs_2 = get_value_distribution(obs, 2)
    
    print(f"  Model targets: pos={pos}, value={val}")
    top3_1 = sorted(enumerate(vprobs_1), key=lambda x:-x[1])[:3]
    top3_2 = sorted(enumerate(vprobs_2), key=lambda x:-x[1])[:3]
    print(f"  Pos 1 top-3: {[(i, f'{p:.3f}') for i, p in top3_1]}")
    print(f"  Pos 2 top-3: {[(i, f'{p:.3f}') for i, p in top3_2]}")
    results['s6_targeting'] = {'predicted_pos': int(pos), 'predicted_val': int(val)}
    
    # Summary
    print("\n" + "=" * 50)
    print("SCENARIO SUMMARY")
    print("=" * 50)
    pass_keys = [k for k in results if isinstance(results[k], dict) and 'correct' in results[k]]
    n_pass = sum(1 for k in pass_keys if results[k]['correct'])
    n_total = len(pass_keys)
    for k in sorted(pass_keys):
        status = "PASS" if results[k]['correct'] else "FAIL"
        print(f"  {k:25s}: {status}")
    print(f"\n  Total: {n_pass}/{n_total} passed")
    
    return results


# ========== EXPERIMENT 4: GAME STATISTICS ==========

def measure_game_stats(model: DaVinciCodePolicy, num_games: int = 1000):
    """Self-play statistics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: SELF-PLAY GAME STATISTICS")
    print("=" * 70)
    
    wins = {0: 0, 1: 0}
    game_lengths = []
    
    for game_idx in range(num_games):
        env = DaVinciCodeEnv(seed=SEED_START + 10000 + game_idx)
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 300:
            obs_t = make_obs_tensor(obs)
            mask = env.get_action_mask()
            mask_t = make_mask_tensor(mask)
            
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t, mask_t, deterministic=True)
            
            obs, _ro, reward, done, truncated, info, result = env.step(action[0])
            steps += 1
        
        game_lengths.append(steps)
        if env._winner is not None:
            wins[env._winner] += 1
        
        if (game_idx + 1) % 200 == 0:
            print(f"  Games: {game_idx + 1}/{num_games}")
    
    print(f"\n  Games played: {num_games}")
    print(f"  Player 0 wins: {wins[0]} ({wins[0]/num_games*100:.1f}%)")
    print(f"  Player 1 wins: {wins[1]} ({wins[1]/num_games*100:.1f}%)")
    draws = num_games - wins[0] - wins[1]
    if draws:
        print(f"  Draws/incomplete: {draws}")
    print(f"  Avg game length: {np.mean(game_lengths):.1f} steps (std={np.std(game_lengths):.1f})")
    
    return {
        'wins': {str(k): v for k, v in wins.items()},
        'avg_length': float(np.mean(game_lengths)),
        'std_length': float(np.std(game_lengths)),
    }


# ========== EXPERIMENT 5: ATTENTION FLOW ==========

def attention_flow_analysis(model: DaVinciCodePolicy):
    """Track how attention patterns change as game progresses."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: ATTENTION FLOW THROUGH GAME PROGRESSION")
    print("=" * 70)
    
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    early_my_to_opp = {i: [] for i in range(4)}
    late_my_to_opp = {i: [] for i in range(4)}
    early_hidden_ratio = {i: [] for i in range(4)}
    late_hidden_ratio = {i: [] for i in range(4)}
    
    for game_idx in range(100):
        env = DaVinciCodeEnv(seed=SEED_START + 20000 + game_idx)
        obs, info = env.reset()
        done = False
        guess_samples = []
        
        while not done:
            obs_t = make_obs_tensor(obs)
            mask = env.get_action_mask()
            mask_t = make_mask_tensor(mask)
            phase = obs["phase"]
            
            if phase[1] == 1:
                attn_layers = extractor.get_attention(obs_t)
                opp_hand = obs["opponent_hand"]
                my_hand = obs["my_hand"]
                
                my_active = [i for i in range(13) if my_hand[i, 0] >= 0]
                opp_active = [i for i in range(13) if opp_hand[i, 0] >= 0]
                opp_hidden = [i for i in opp_active if opp_hand[i, 1] == int(CardValue.HIDDEN)]
                opp_revealed = [i for i in opp_active if opp_hand[i, 1] != int(CardValue.HIDDEN)]
                
                if all(a is not None for a in attn_layers) and my_active and opp_active:
                    sample = {'m2o': {}, 'hr': {}, 'n_rev': len(opp_revealed)}
                    for li, attn in enumerate(attn_layers):
                        a = attn[0].mean(dim=0).numpy()
                        opp_idx = [j + 13 for j in opp_active]
                        m2o = float(a[np.ix_(my_active, opp_idx)].mean())
                        sample['m2o'][li] = m2o
                        
                        if opp_hidden and opp_revealed:
                            h_idx = [j + 13 for j in opp_hidden]
                            r_idx = [j + 13 for j in opp_revealed]
                            hv = float(a[np.ix_(my_active, h_idx)].mean())
                            rv = float(a[np.ix_(my_active, r_idx)].mean())
                            sample['hr'][li] = hv / (rv + 1e-10)
                    
                    guess_samples.append(sample)
            
            with torch.no_grad():
                action, _, _ = model.get_action(obs_t, mask_t, deterministic=True)
            obs, _ro, reward, done, truncated, info, result = env.step(action[0])
        
        if len(guess_samples) >= 2:
            half = len(guess_samples) // 2
            for s in guess_samples[:half]:
                for li in range(4):
                    early_my_to_opp[li].append(s['m2o'].get(li, 0))
                    if li in s['hr']:
                        early_hidden_ratio[li].append(s['hr'][li])
            for s in guess_samples[half:]:
                for li in range(4):
                    late_my_to_opp[li].append(s['m2o'].get(li, 0))
                    if li in s['hr']:
                        late_hidden_ratio[li].append(s['hr'][li])
    
    print("\nMy->Opp attention: early vs late game")
    print(f"{'Layer':>6} {'Early':>10} {'Late':>10} {'Change':>10}")
    print("-" * 40)
    for li in range(4):
        e = np.mean(early_my_to_opp[li]) if early_my_to_opp[li] else 0
        l = np.mean(late_my_to_opp[li]) if late_my_to_opp[li] else 0
        change = (l - e) / (e + 1e-10) * 100
        print(f"{li:>6} {e:>10.6f} {l:>10.6f} {change:>+9.1f}%")
    
    print("\nHidden/Revealed attention ratio: early vs late game")
    print(f"{'Layer':>6} {'Early':>10} {'Late':>10} {'Change':>10}")
    print("-" * 40)
    for li in range(4):
        e = np.mean(early_hidden_ratio[li]) if early_hidden_ratio[li] else 0
        l = np.mean(late_hidden_ratio[li]) if late_hidden_ratio[li] else 0
        change = (l - e) / (e + 1e-10) * 100
        print(f"{li:>6} {e:>10.4f} {l:>10.4f} {change:>+9.1f}%")
    
    return {'computed': True}


# ========== MAIN ==========

def main():
    print("=" * 70)
    print("DA VINCI CODE RL AGENT -- INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    model = load_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    all_results = {}
    
    # Exp 4: Stats (quick)
    all_results['game_stats'] = measure_game_stats(model, num_games=1000)
    
    # Exp 3: Controlled scenarios
    all_results['scenarios'] = controlled_scenario_test(model)
    
    # Exp 1: Attention patterns
    all_results['attention'] = analyze_attention_patterns(model, num_games=NUM_GAME_SAMPLES)
    
    # Exp 5: Attention flow
    all_results['attention_flow'] = attention_flow_analysis(model)
    
    # Exp 2: Probing
    try:
        all_results['probing'] = probing_experiment(model, num_games=NUM_GAME_SAMPLES)
    except ImportError as e:
        print(f"\nsklearn not available ({e}). Install: pip install scikit-learn")
        all_results['probing'] = {'error': str(e)}
    except Exception as e:
        print(f"\nProbing failed: {e}")
        import traceback; traceback.print_exc()
        all_results['probing'] = {'error': str(e)}
    
    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_path = "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")
    
    # FINAL VERDICT
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS -- HONEST ASSESSMENT")
    print("=" * 70)
    
    scenarios = all_results.get('scenarios', {})
    pass_keys = [k for k in scenarios if isinstance(scenarios[k], dict) and 'correct' in scenarios[k]]
    n_pass = sum(1 for k in pass_keys if scenarios[k]['correct'])
    n_total = len(pass_keys)
    print(f"\nControlled Scenarios: {n_pass}/{n_total} passed")
    
    probing = all_results.get('probing', {})
    if 'probe_accuracy' in probing:
        print(f"Probing Accuracy: {probing['probe_accuracy']:.1%} (random: {probing['random_baseline']:.1%}, constraint: {probing['constraint_baseline']:.1%})")
        print(f"Verdict: {probing.get('verdict', 'N/A')}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
