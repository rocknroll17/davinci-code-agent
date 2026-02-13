"""Da Vinci Code Gymnasium Environment."""

from typing import Any, Optional, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.constants import (
    Phase, Color, MAX_HAND_SIZE, NUM_VALUES,
    INITIAL_HAND_SIZE_2P, REWARD_WIN, REWARD_LOSE, REWARD_GUESS_SUCCESS,
    REWARD_JOKER_SUCCESS, REWARD_GUESS_FAIL, REWARD_STREAK_BONUS_MULTIPLIER,
    REWARD_STREAK_BREAK, REWARD_INVALID_ACTION, REWARD_STOP_DECISION,
    REWARD_STOP_WITH_DETERMINED
)
from src.deck import Deck
from src.hand import Hand
from src.cards.card import Card
from src.player import Player
from src.result.guess_result import GuessResult
from src.result.draw_result import DrawResult
from src.result.result import Result
from src.result.streak_result import StreakResult
from src.phase import PhaseCycle

VIEWER = 0


class DaVinciCodeEnv(gym.Env):
    """
    Gymnasium Environment for Da Vinci Code board game.
    
    This environment supports adversarial self-play training where
    a single policy network controls both players alternately.
    
    Observation Space:
        Dict with keys:
        - phase: One-hot vector (3,) indicating current phase
        - my_hand: Player's hand (13, 2) with [color, value]
        - opponent_hand: Opponent's hand (13, 2) with hidden values
        - remaining_deck: [black_count, white_count]
        - constraint_matrix: (13, 13) binary matrix of failed guesses
    
    Action Space:
        MultiDiscrete([2, 13, 13, 2]):
        - color: 0=BLACK, 1=WHITE (used in DRAW phase)
        - position: 0-12 (used in GUESS phase)
        - value: 0-12 (used in GUESS phase)
        - decision: 0=STOP, 1=CONTINUE (used in DECISION phase)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "DaVinciCode-v0"}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        viewer: Optional[int] = VIEWER,
    ) -> None:
        """
        Initialize the Da Vinci Code environment.
        
        Args:
            render_mode: Rendering mode ("human" or "ansi")
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.render_mode = render_mode
        self._seed = seed
        # viewer: if set to 0 or 1, render() will always show that player's perspective
        # This does NOT change the observation returned by _get_observation (keeps agent behavior).
        self.viewer: Optional[int] = viewer
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "phase": spaces.Box(
                low=0, high=1, shape=(3,), dtype=np.int8
            ),
            "my_hand": spaces.Box(
                low=-2, high=12, shape=(MAX_HAND_SIZE, 2), dtype=np.int8
            ),
            "opponent_hand": spaces.Box(
                low=-2, high=12, shape=(MAX_HAND_SIZE, 2), dtype=np.int8
            ),
            "remaining_deck": spaces.Box(
                low=0, high=13, shape=(2,), dtype=np.int8
            ),
            "constraint_matrix": spaces.Box(
                low=0, high=1, shape=(MAX_HAND_SIZE, NUM_VALUES), dtype=np.int8
            )
        })
        
        # Define action space: [color, position, value, decision]
        self.action_space = spaces.MultiDiscrete([2, 13, 13, 2])
        
        # Initialize game state variables
        self._deck: Deck = Deck(seed)
        self._current_player: int = 0
        self.players: list[Player] = [Player(0), Player(1)]
        self._phase: PhaseCycle = PhaseCycle()
        self._streak: int = 0
        self._last_drawn_card: Optional[Card] = None
        self._last_drawn_position: int = -1
        self._done: bool = False
        self._winner: Optional[int] = None

        # Track last action for rendering
        self._last_action: Optional[np.ndarray] = None
        self._last_reward: float = 0.0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        import logging
        logger = logging.getLogger()
        logger.info(f"Game Start")
        # Reset random seed
        if seed is not None:
            self._seed = seed
        
        # Reset deck
        self._deck.reset(self._seed)

        for player in self.players:
            player.reset()
        
        # Deal initial cards (4 cards each for 2 players)
        self._deal_initial_cards()
        
        # Reset game state
        self._current_player = 0
        self._phase = PhaseCycle()
        self._streak = 0
        self._last_drawn_card = None
        self._last_drawn_position = -1
        self._done = False
        self._winner = None
        self._last_action = None
        self._last_reward = 0.0
        
        return self._get_observation(), self._get_info()
    
    def render_info(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Get current observation and info without changing state.
        
        Returns:
            Tuple of (observation, info)
        """
        return self._get_render_observation(), self._get_render_info()
    
    def _deal_initial_cards(self) -> None:
        """Deal initial cards to both players."""
        for player in range(2):
            cards = self._deck.initial_draw(INITIAL_HAND_SIZE_2P)
            self.players[player]._hand.add_initial_cards(cards)
            self.players[player].update_initial_constraint(INITIAL_HAND_SIZE_2P)
        import logging
        logger = logging.getLogger()
        logger.info(f"Player 0 starts turn.")

    def step(
        self,
        action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, bool, bool, dict[str, Any], object]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array [color, position, value, decision]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._last_action = action
        
        if self._done:
            # Standard Gym returns 7 values (observation, render_observation, reward, terminated, truncated, info, result)
            return self._get_observation(), self._get_render_observation(), 0.0, True, False, self._get_info(), None
        
        # Initialize reward for current step
        step_reward = 0.0
        
        # Dispatch to appropriate phase handler
        if self._phase == Phase.DRAW:
            result = self._handle_draw_phase(action)
        elif self._phase == Phase.GUESS:
            result = self._handle_guess_phase(action)
        elif self._phase == Phase.DECISION:
            result = self._handle_decision_phase(action)

        step_reward = result.reward if result is not None else 0.0 # Get reward from the result object
        
        self._last_reward = step_reward # Update for logging/rendering
        
        # Check for game termination
        terminated = self._done
        truncated = False # Assuming no truncation in this environment

        # Ensure info contains necessary details, potentially from result object
        info = self._get_info()
        # Add result object details to info if needed elsewhere, e.g. winner
        if result:
            info['step_result_type'] = type(result).__name__
            info['step_result_reward'] = result.reward
            # Add other relevant details from result to info if necessary for trainer/logging
            if hasattr(result, 'player_id'):
                info['player_id'] = result.player_id
            if hasattr(result, 'winner') and result.winner is not None:
                info['winner'] = result.winner
        
        # Return standard 7-tuple
        return self._get_observation(), self._get_render_observation(), step_reward, terminated, truncated, info, result
    
    def _get_observation(self):
        """
        Get current observation from current player's perspective.
        
        Returns:
            Observation dictionary
        """
        phase_onehot = np.zeros(3, dtype=np.int8)
        phase_onehot[self._phase.value] = 1
        current_observation = self.players[self._current_player]._get_observation(is_mine=True)
        opponent_observation = self.players[1 - self._current_player]._get_observation(is_mine=False)
        remaining_deck = np.array(self._deck.get_remaining(), dtype=np.int8)
        observation = {
            "phase": phase_onehot,
            **current_observation,
            **opponent_observation,
            "remaining_deck": remaining_deck
        }
        return observation
    
    def _get_render_observation(self):
        """
        Get observation for rendering from viewer's perspective.
        
        Returns:
            Observation dictionary
        """
        viewer = self.viewer if self.viewer is not None else self._current_player
        phase_onehot = np.zeros(3, dtype=np.int8)
        phase_onehot[self._phase.value] = 1
        current_observation = self.players[viewer]._get_observation(is_mine=True)
        opponent_observation = self.players[1 - viewer]._get_observation(is_mine=False)
        remaining_deck = np.array(self._deck.get_remaining(), dtype=np.int8)
        observation = {
            "phase": phase_onehot,
            **current_observation,
            **opponent_observation,
            "remaining_deck": remaining_deck
        }
        return observation


    
    def _handle_draw_phase(self, action: np.ndarray) -> DrawResult:
        """
        Handle DRAW phase action.
        
        Args:
            action: Action array (only color component used)
            
        Returns:
            Reward value
        """
        color = int(action[0])
        if self._deck.is_empty():
            # No cards left, skip draw and go to guess
            self._phase.draw_done()
            return DrawResult(self._current_player, 0.0, None, None, is_invalid=False)
        # If one color is empty, force the other color
        elif self._deck.is_one_empty():
            if self._deck.black_count > 0:
                color = Color.BLACK
            else:
                color = Color.WHITE
        else:
            # Both colors available, use player's choice
            color = Color.BLACK if color == 0 else Color.WHITE
        
        # Draw card
        card = self._draw_card(color)
        if card is None:
            return DrawResult(self._current_player, REWARD_INVALID_ACTION, None, None, is_invalid=True)
        
        # Add to current player's hand
        position = self.players[self._current_player]._hand.add_card(card)
        # file = open(f"logs/card_log_{self._current_player}.txt", "a")
        # file.write(f"{self._hands[self._current_player].to_string()}\n")
        # file.close()
        self._last_drawn_card = card
        self._last_drawn_position = position
        opponent = 1 - self._current_player
        self.players[opponent].update_constraint(position)
        
        # Transition to GUESS phase
        self._phase.draw_done()
        
        return DrawResult(self._current_player, 0.0, card, position, is_invalid=False)
    
    def _draw_card(self, color: int) -> Optional[Card]:
        """
        Draw a card from the deck.
        
        Args:
            color: Color of card to draw
            
        Returns:
            Drawn card, or None if unavailable
        """
        return self._deck.draw(color)
    
    def _handle_guess_phase(self, action: np.ndarray) -> GuessResult:
        """
        Handle GUESS phase action.
        
        Args:
            action: Action array (position and value components used)
            
        Returns:
            Reward value
        """
        import logging
        logger = logging.getLogger()
        position = int(action[1])
        value = int(action[2])
        
        opponent = 1 - self._current_player
        opponent_hand = self.players[opponent]._hand
        
        # Validate position
        if position >= opponent_hand.size:
            return GuessResult(self._current_player, REWARD_INVALID_ACTION, position, value, False, is_invalid=True)
        
        target_card = opponent_hand.get_card(position)
        if target_card is None or target_card.is_revealed:
            return GuessResult(self._current_player, REWARD_INVALID_ACTION, position, value, False, is_invalid=True)
        
        # Check guess result
        if target_card.value == value:
            # Correct guess!
            reward = self._handle_guess_success(position, target_card)
        else:
            # Wrong guess
            reward = self._handle_guess_failure(position, value)

        result = GuessResult(self._current_player, reward, position, value, target_card.value == value, is_invalid=False)
        logger.info(result)

        if target_card.value != value:
            logger.info(f"{target_card}, {value}")
            self._end_turn()
        
        return result
    
    def _handle_guess_success(
        self,
        position: int,
        target_card: Card
    ) -> float:
        """
        Handle successful guess.
        
        Args:
            position: Position of guessed card
            target_card: Card that was correctly guessed
            opponent_hand: Opponent's hand
            
        Returns:
            Reward value
        """
        opponent = 1 - self._current_player
        
        self.players[opponent]._hand.reveal_card(position)
        self.players[self._current_player].guess_success(position)
        # Transition to DECISION phase
        self._phase.guess_correct()

        # Calculate reward
        if target_card.is_joker:
            reward = REWARD_JOKER_SUCCESS
        else:
            reward = REWARD_GUESS_SUCCESS
        
        # Streak bonus
        self._streak += 1
        reward += REWARD_STREAK_BONUS_MULTIPLIER * self._streak
        
        # Check win condition
        if self.players[opponent]._hand.all_revealed():
            self._done = True
            self._winner = self._current_player
            reward += REWARD_WIN
            return reward
        
        return reward
    
    def _handle_guess_failure(self, position: int, guessed_value: int) -> float:
        """
        Handle failed guess.
        
        Args:
            position: Position that was guessed
            guessed_value: Value that was incorrectly guessed
            
        Returns:
            Reward value
        """
        reward = REWARD_GUESS_FAIL

        opponent = 1 - self._current_player
        
        # Streak break penalty
        if self._streak > 0:
            reward += REWARD_STREAK_BREAK
        self._streak = 0
        index = self.players[self._current_player].guess_fail(position, guessed_value)
        self.players[opponent]._update_constraint_revealed(index)
        self._phase.guess_wrong()
        
        return reward
    
    def _handle_decision_phase(self, action: np.ndarray) -> StreakResult:
        """
        Handle DECISION phase action.
        
        Args:
            action: Action array (decision component used)
            
        Returns:
            Reward value
        """
        decision = int(action[3])
        import logging
        logger = logging.getLogger()

        if decision == 0:  # STOP
            # 확정된 미공개 카드가 있는데 맞추지 않고 stop하면 페널티
            from src.utils.game_logic import find_determined_cards
            determined = find_determined_cards(
                self.players[self._current_player]._hand,
                self.players[1 - self._current_player]._hand
            )
            penalty = len(determined) * REWARD_STOP_WITH_DETERMINED
            if determined:
                logger.info(f"Player {self._current_player} stopped with {len(determined)} determined card(s): {determined}")
            reward = StreakResult(self._current_player, REWARD_STOP_DECISION + penalty, False, is_invalid=False)
            self._phase.end_streak()
        else:  # CONTINUE
            reward = StreakResult(self._current_player, 0.0, True, is_invalid=False)
            self._phase.continue_streak()

        logger.info(reward)

        if decision == 0:
            self._end_turn()
        
        return reward
    
    def _end_turn(self) -> None:
        """End current player's turn and switch to opponent."""
        import logging
        logger = logging.getLogger()
        # Check if current player lost (all cards revealed)
        if self.players[self._current_player]._hand.all_revealed():
            self._done = True
            self._winner = 1 - self._current_player
            logger.info(f"Player {1 - self._current_player} Win.")
            return
        
        # Switch player
        logger.info(f"Player {self._current_player} ends turn.")

        self._streak = 0
        self._last_drawn_card = None
        self._last_drawn_position = -1

        self.players[self._current_player].end_turn()
        self._current_player = 1 - self._current_player
        logger.info(f"Player {self._current_player} starts turn.")
    
    def _get_info(self) -> dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            Info dictionary
        """
        viewer = self.viewer if self.viewer is not None else self._current_player

        return {
            "current_player": self._current_player,
            "viewer": viewer,
            "phase": self._phase.name,
            "streak": self._streak,
            "done": self._done,
            "winner": self._winner,
            "my_hand_size": self.players[self._current_player]._hand.size,
            "opponent_hand_size": self.players[1 - self._current_player]._hand.size,
            "viewer_hand_size": self.players[viewer]._hand.size,
            "deck_remaining": self._deck.total_count
        }
    
    def _get_render_info(self) -> dict[str, Any]:
        """
        Get info for rendering from viewer's perspective.
        
        Returns:
            Info dictionary
        """
        viewer = self.viewer if self.viewer is not None else self._current_player

        return {
            "current_player": self._current_player,
            "viewer": viewer,
            "phase": self._phase.name,
            "streak": self._streak,
            "done": self._done,
            "winner": self._winner,
            "my_hand_size": self.players[viewer]._hand.size,
            "opponent_hand_size": self.players[1 - viewer]._hand.size,
            "deck_remaining": self._deck.total_count
        }
    
    def get_action_mask(self) -> dict[str, np.ndarray]:
        """
        Get action masks for current state.
        
        Returns:
            Dictionary with masks for each action component
        """
        opponent = 1 - self._current_player
        opponent_hand = self.players[opponent]._hand
        
        # Color mask (for DRAW phase)
        color_mask = np.array([
            self._deck.has_color(Color.BLACK),
            self._deck.has_color(Color.WHITE)
        ], dtype=bool)
        
        # Position mask (for GUESS phase)
        position_mask = np.zeros(MAX_HAND_SIZE, dtype=bool)
        for i in range(opponent_hand.size):
            card = opponent_hand.get_card(i)
            if card is not None and not card.is_revealed:
                position_mask[i] = True
        
        # Per-position value mask (13 x 13): considers target position's color
        # Each value 0-11 has exactly 1 BLACK copy and 1 WHITE copy
        # Joker (12) has 1 BLACK copy and 1 WHITE copy
        # If the target position is BLACK, any value whose BLACK copy is
        # already confirmed (in my hand or opponent revealed) is impossible.
        current_player = self.players[self._current_player]
        
        # Track confirmed cards per color
        black_confirmed = np.zeros(NUM_VALUES, dtype=bool)
        white_confirmed = np.zeros(NUM_VALUES, dtype=bool)
        
        for card in current_player._hand:
            if card is not None:
                if card.color == Color.BLACK:
                    black_confirmed[card.value] = True
                else:
                    white_confirmed[card.value] = True
        
        for card in opponent_hand:
            if card is not None and card.is_revealed:
                if card.color == Color.BLACK:
                    black_confirmed[card.value] = True
                else:
                    white_confirmed[card.value] = True
        
        # Build per-position value mask
        value_mask = np.ones((MAX_HAND_SIZE, NUM_VALUES), dtype=bool)
        for i in range(opponent_hand.size):
            card = opponent_hand.get_card(i)
            if card is not None and not card.is_revealed:
                if card.color == Color.BLACK:
                    value_mask[i] = ~black_confirmed
                else:
                    value_mask[i] = ~white_confirmed
                # Safety: ensure at least one value is valid per position
                if not value_mask[i].any():
                    value_mask[i] = np.ones(NUM_VALUES, dtype=bool)
        
        # Decision mask
        decision_mask = np.array([True, True], dtype=bool)  # STOP, CONTINUE
        
        return {
            "color": color_mask,
            "position": position_mask,
            "value": value_mask,
            "decision": decision_mask
        }
    
    def render(self) -> Optional[str]:
        """
        Render the current game state.
        
        Returns:
            String representation if render_mode is "ansi"
        """
        # If no viewer is fixed and render_mode is not set to a printable mode, do nothing.
        if self.viewer is None and self.render_mode not in ["human", "ansi"]:
            return None
        
        lines = []
        lines.append("=" * 50)
        lines.append(f"Da Vinci Code - Current Turn: Player {self._current_player}")
        lines.append(f"Phase: {self._phase.name}")
        lines.append(f"Streak: {self._streak}")
        lines.append("-" * 50)

        # Determine viewer for rendering: fixed viewer overrides current player
        viewer = self.viewer if self.viewer is not None else self._current_player

        # Show hands (viewer sees their own hand fully, opponent partially)
        for i in range(2):
            player_label = "YOU" if i == viewer else "OPP"
            hand = self.players[i]._hand
            if i == viewer:
                cards_str = ", ".join(repr(c) for c in hand)
            else:
                cards_str = ", ".join(
                    repr(c) if getattr(c, "is_revealed", False) else f"[{getattr(c, 'color', '?')}?]"
                    for c in hand
                )
            lines.append(f"{player_label}: {cards_str}")
        
        lines.append("-" * 50)
        lines.append(f"Deck: B={self._deck.black_count}, W={self._deck.white_count}")
        
        if self._last_action is not None:
            lines.append(f"Last Action: {self._last_action}")
            lines.append(f"Last Reward: {self._last_reward:.2f}")
        
        if self._done:
            lines.append("=" * 50)
            if self._winner is not None:
                lines.append(f"GAME OVER - Player {self._winner} WINS!")
            else:
                lines.append("GAME OVER - DRAW")
        
        lines.append("=" * 50)
        
        output = "\n".join(lines)
        
        # Always print if viewer is fixed (for easy inspection), otherwise respect render_mode
        if self.viewer is not None or self.render_mode == "human":
            print(output)
        
        return output
    
    def close(self) -> None:
        """Clean up resources."""
        pass
 