#!/usr/bin/env python3
"""
Da Vinci Code - Watch Model Play

모델이 게임을 플레이하는 것을 관찰합니다.
한 스텝씩 진행하며 상태를 확인할 수 있습니다.

Usage:
    python play.py                    # 기본 체크포인트 로드
    python play.py --checkpoint path  # 특정 체크포인트 로드
    python play.py --auto             # 자동 진행 (Enter 불필요)
    python play.py --delay 1.0        # 자동 진행 시 딜레이 (초)
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich import box

from src.env import DaVinciCodeEnv
from src.model import DaVinciCodePolicy, obs_to_tensor, action_mask_to_tensor
from src.constants import Phase, Color, CardValue, MAX_HAND_SIZE


console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch Da Vinci Code model play")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--auto", "-a", action="store_true",
                        help="Auto-advance without waiting for input")
    parser.add_argument("--delay", "-d", type=float, default=0.8,
                        help="Delay between steps in auto mode (seconds)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="Number of episodes to play")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (argmax instead of sampling)")
    parser.add_argument("--joker", action="store_true",
                        help="Stop only when model guesses joker (auto-continue for other guesses)")
    return parser.parse_args()


class GameViewer:
    """고정된 TUI 화면으로 게임 상태를 표시하는 뷰어."""
    
    def __init__(self):
        self.console = Console()
        self.live = None
        
        # 게임 상태
        self.player0_hand = None  # (13, 2) - [color, value]
        self.player1_hand = None
        self.player0_revealed = None  # (13,) - bool array
        self.player1_revealed = None
        self.phase = "DRAW"
        self.current_player = 0
        self.streak = 0
        self.deck_black = 12
        self.deck_white = 12
        
        # 에피소드/스텝 정보
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        
        # 액션 & 모델 정보 (플레이어별)
        self.player0_action_text = ""
        self.player0_result_text = ""
        self.player0_result_style = "yellow"
        self.player1_action_text = ""
        self.player1_result_text = ""
        self.player1_result_style = "yellow"
        self.available_actions = ""
        self.model_probs = ""
        self.value_estimate = 0.0
        
        # 메시지
        self.status_message = "Press Enter to start"
        self.winner = None
    
    def _render_card(self, color: int, value: int, revealed: bool) -> Text:
        """
        카드 하나를 렌더링.
        카드 색상은 공개 여부와 무관하게 동일하게 표시.
        """
        if color == Color.NONE or value == CardValue.NONE:
            return Text("    ")
        
        # 값 문자열
        if value == CardValue.JOKER:
            val_str = " - "
        elif value == CardValue.HIDDEN:
            val_str = " ? "
        else:
            val_str = f"{value:2d} "
        
        # 스타일 결정 (정수값으로 비교: BLACK=0, WHITE=1)
        if color == 0:  # BLACK
            style = "bold white on black"
        else:  # WHITE (1)
            style = "bold black on white"
        
        return Text(val_str, style=style)
    
    def _render_hand(self, hand: np.ndarray, revealed: np.ndarray, 
                     title: str, is_current: bool) -> Panel:
        """플레이어 패 렌더링."""
        if hand is None:
            return Panel(Text("(loading...)", style="dim"), title=title)
        
        pos_row = Text()
        card_row = Text()
        status_row = Text()
        
        has_cards = False
        for i in range(MAX_HAND_SIZE):
            color = int(hand[i, 0])
            value = int(hand[i, 1])
            
            if color == Color.NONE or value == CardValue.NONE:
                continue
            
            has_cards = True
            is_revealed = revealed[i] if revealed is not None else False
            
            pos_row.append(f" {i:2d} ", style="dim cyan")
            card_row.append(self._render_card(color, value, is_revealed))
            card_row.append(" ")
            
            if is_revealed:
                status_row.append(" ●  ", style="green")  # 공개됨 = 채워진 원
            else:
                status_row.append(" ○  ", style="red")    # 숨김 = 빈 원
        
        if not has_cards:
            content = Text("(no cards)", style="dim italic")
        else:
            content = Text()
            content.append(pos_row)
            content.append("\n")
            content.append(card_row)
            content.append("\n")
            content.append(status_row)
            content.append(Text("  (●=revealed, ○=hidden)", style="dim"))
        
        border_style = "bold green" if is_current else "dim white"
        return Panel(content, title=title, border_style=border_style)
    
    def _render_game_info(self) -> Panel:
        """게임 정보 패널."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=14)
        table.add_column(style="yellow", width=16)
        
        player_str = "[bold green]Player 0 ◀[/]" if self.current_player == 0 else "[bold red]Player 1 ◀[/]"
        
        table.add_row("Episode", f"{self.episode}")
        table.add_row("Step", f"{self.step}")
        table.add_row("Phase", f"[bold magenta]{self.phase}[/]")
        table.add_row("Current Turn", player_str)
        table.add_row("Streak", f"{self.streak}")
        table.add_row("─" * 14, "─" * 16)
        table.add_row("Last Reward", f"[{'green' if self.last_reward >= 0 else 'red'}]{self.last_reward:+.3f}[/]")
        table.add_row("Total Reward", f"[bold]{self.total_reward:+.2f}[/]")
        table.add_row("─" * 14, "─" * 16)
        table.add_row("Deck (B/W)", f"[white on black] {self.deck_black:2d} [/] [black on white] {self.deck_white:2d} [/]")
        
        return Panel(table, title="📊 Game Info", border_style="blue")
    
    def _render_model_info(self) -> Panel:
        """모델 정보 패널."""
        content = Text()
        content.append("Available Actions:\n", style="bold cyan")
        content.append(self.available_actions + "\n\n", style="yellow")
        content.append("Model Log Probs:\n", style="bold cyan")
        content.append(self.model_probs + "\n\n", style="dim")
        content.append("Value Estimate: ", style="bold cyan")
        content.append(f"{self.value_estimate:.4f}", style="magenta")
        
        return Panel(content, title="🧠 Model", border_style="cyan")
    
    def _render_player_action(self, player: int) -> Panel:
        """플레이어별 액션 패널."""
        is_current = (self.current_player == player)
        
        if player == 0:
            action_text = self.player0_action_text
            result_text = self.player0_result_text
            result_style = self.player0_result_style
            title = "🎮 P0 Action"
        else:
            action_text = self.player1_action_text
            result_text = self.player1_result_text
            result_style = self.player1_result_style
            title = "👤 P1 Action"
        
        if not action_text:
            # 액션이 없으면 대기 중 표시
            content = Text("(waiting)", style="dim italic")
            border_style = "dim white"
        else:
            content = Text()
            content.append(action_text, style="bold yellow")
            if result_text:
                content.append("\n")
                content.append("Result: ", style="bold")
                content.append(result_text, style=result_style)
            # 현재 턴이면 강조, 아니면 dim
            border_style = "bold yellow" if is_current else "yellow"
        
        return Panel(content, title=title, border_style=border_style)
    
    def _render_status(self) -> Panel:
        """상태 메시지 패널."""
        if self.winner is not None:
            if self.winner == 0:
                msg = "🏆 Player 0 WINS! 🏆"
                style = "bold green"
            else:
                msg = "💀 Player 1 WINS 💀"
                style = "bold red"
        else:
            msg = self.status_message
            style = "dim"
        
        return Panel(Text(msg, justify="center", style=style), border_style="white")
    
    def render(self) -> Layout:
        """전체 레이아웃 렌더링."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="p0_section", size=6),
            Layout(name="p1_section", size=6),
            Layout(name="middle", size=10),
            Layout(name="status", size=3)
        )
        
        header_text = Text("🎴 Da Vinci Code - Model Play Viewer 🎴", 
                          style="bold white", justify="center")
        layout["header"].update(Panel(header_text, style="on blue"))
        
        # Player 0: 패 + 액션
        layout["p0_section"].split_row(
            Layout(self._render_hand(
                self.player0_hand, self.player0_revealed,
                "🎮 Player 0", self.current_player == 0
            ), name="p0_hand", ratio=3),
            Layout(self._render_player_action(0), name="p0_action", ratio=1)
        )
        
        # Player 1: 패 + 액션
        layout["p1_section"].split_row(
            Layout(self._render_hand(
                self.player1_hand, self.player1_revealed,
                "👤 Player 1", self.current_player == 1
            ), name="p1_hand", ratio=3),
            Layout(self._render_player_action(1), name="p1_action", ratio=1)
        )
        
        # 중간: 게임 정보 + 모델 정보
        layout["middle"].split_row(
            Layout(self._render_game_info(), name="info", ratio=1),
            Layout(self._render_model_info(), name="model", ratio=2)
        )
        
        layout["status"].update(self._render_status())
        
        return layout
    
    def start(self):
        """Live 디스플레이 시작."""
        self.live = Live(self.render(), console=self.console, 
                        refresh_per_second=10, screen=True)
        self.live.start()
    
    def stop(self):
        """Live 디스플레이 종료."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def update(self):
        """화면 업데이트."""
        if self.live:
            self.live.update(self.render())
    
    def update_game_state(self, env: DaVinciCodeEnv, obs: dict):
        """환경에서 게임 상태 업데이트."""
        phase_idx = int(np.argmax(obs["phase"]))
        self.phase = Phase(phase_idx).name
        
        # 턴이 바뀌었으면 새 플레이어의 액션 초기화
        new_player = env._current_player
        if new_player != self.current_player:
            if new_player == 0:
                self.player0_action_text = ""
                self.player0_result_text = ""
            else:
                self.player1_action_text = ""
                self.player1_result_text = ""
        
        self.current_player = new_player
        self.streak = env._streak
        self.deck_black = env._deck.black_count
        self.deck_white = env._deck.white_count
        
        # 플레이어 0 손패 (Hand는 list[Card] 상속)
        p0_hand = env.players[0]._hand
        p0_cards = list(p0_hand)
        self.player0_hand = np.full((MAX_HAND_SIZE, 2), [Color.NONE, CardValue.NONE], dtype=np.int8)
        self.player0_revealed = np.zeros(MAX_HAND_SIZE, dtype=bool)
        for i, card in enumerate(p0_cards):
            if i >= MAX_HAND_SIZE:
                break
            self.player0_hand[i] = [card.color.value, card.value]
            self.player0_revealed[i] = card.is_revealed
        
        # 플레이어 1 손패
        p1_hand = env.players[1]._hand
        p1_cards = list(p1_hand)
        self.player1_hand = np.full((MAX_HAND_SIZE, 2), [Color.NONE, CardValue.NONE], dtype=np.int8)
        self.player1_revealed = np.zeros(MAX_HAND_SIZE, dtype=bool)
        for i, card in enumerate(p1_cards):
            if i >= MAX_HAND_SIZE:
                break
            self.player1_hand[i] = [card.color.value, card.value]
            self.player1_revealed[i] = card.is_revealed
        
        self.update()
    
    def update_action_mask(self, action_mask: dict, phase_name: str):
        """가능한 액션 업데이트."""
        lines = []
        
        if phase_name == "DRAW":
            colors = []
            if action_mask["color"][0]:
                colors.append("BLACK")
            if action_mask["color"][1]:
                colors.append("WHITE")
            lines.append(f"Colors: {', '.join(colors)}")
        
        elif phase_name == "GUESS":
            positions = [str(i) for i, v in enumerate(action_mask["position"]) if v]
            values = [str(i) for i, v in enumerate(action_mask["value"]) if v]
            lines.append(f"Positions: {', '.join(positions[:8])}{'...' if len(positions) > 8 else ''}")
            lines.append(f"Values: {', '.join(values)}")
        
        elif phase_name == "DECISION":
            decisions = []
            if action_mask["decision"][0]:
                decisions.append("STOP")
            if action_mask["decision"][1]:
                decisions.append("CONTINUE")
            lines.append(f"Options: {', '.join(decisions)}")
        
        self.available_actions = "\n".join(lines)
        self.update()
    
    def update_model_output(self, log_probs: dict, value: float, value_probs=None, phase_name=""):
        """모델 출력 업데이트."""
        prob_strs = []
        for key, log_prob in log_probs.items():
            lp = log_prob.cpu().numpy() if hasattr(log_prob, 'cpu') else log_prob
            prob_strs.append(f"{key}: {float(lp):.3f}")
        
        # GUESS phase에서 value 확률 분포 추가 (특히 조커)
        if value_probs is not None and phase_name == "GUESS":
            top_3 = np.argsort(value_probs)[-3:][::-1]
            val_str = " | ".join([f"v{i}:{value_probs[i]:.3f}" for i in top_3])
            joker_prob = value_probs[12] if len(value_probs) > 12 else 0.0
            val_str += f" | JOKER(12):{joker_prob:.4f}"
            prob_strs.append(f"value_dist: {val_str}")
        
        self.model_probs = " | ".join(prob_strs)
        self.value_estimate = float(value.item()) if hasattr(value, 'item') else float(value)
        self.update()
    
    def update_action_result(self, action: np.ndarray, phase_name: str, 
                            result, reward: float, player: int):
        """액션 결과 업데이트 (플레이어별)."""
        # 액션 텍스트 생성
        if phase_name == "DRAW":
            color = "BLACK" if action[0] == 0 else "WHITE"
            action_text = f"Draw {color} card"
        elif phase_name == "GUESS":
            action_text = f"Guess Pos={action[1]}, Val={action[2]}"
        elif phase_name == "DECISION":
            decision = "CONTINUE" if action[3] == 1 else "STOP"
            action_text = f"Decision: {decision}"
        else:
            action_text = str(action)
        
        # 결과 스타일 결정
        result_text = ""
        result_style = "yellow"
        if result:
            result_text = str(result)
            result_upper = result_text.upper()
            if "SUCCESS" in result_upper or "WIN" in result_upper:
                result_style = "bold green"
            elif "FAIL" in result_upper or "LOSE" in result_upper:
                result_style = "bold red"
        
        # 플레이어별로 저장
        if player == 0:
            self.player0_action_text = action_text
            self.player0_result_text = result_text
            self.player0_result_style = result_style
        else:
            self.player1_action_text = action_text
            self.player1_result_text = result_text
            self.player1_result_style = result_style
        
        self.last_reward = reward
        self.total_reward += reward
        self.update()
    
    def clear_actions(self):
        """새 스텝 시작 전 액션 초기화."""
        # 현재 플레이어가 아닌 쪽 액션만 클리어 (또는 전부)
        pass  # 필요시 구현


def play_episode(policy: DaVinciCodePolicy, env: DaVinciCodeEnv,
                 viewer: GameViewer, episode: int, args) -> dict:
    """한 에피소드 플레이."""
    device = next(policy.parameters()).device
    
    obs, info = env.reset()
    viewer.episode = episode
    viewer.step = 0
    viewer.total_reward = 0.0
    viewer.last_reward = 0.0
    viewer.winner = None
    # 플레이어별 액션 초기화
    viewer.player0_action_text = ""
    viewer.player0_result_text = ""
    viewer.player1_action_text = ""
    viewer.player1_result_text = ""
    
    done = False
    joker_mode = args.joker  # 조커만 멈추는 모드
    auto_mode = args.auto    # 자동 진행 모드 (동적 변경 가능)
    enter_count = 0  # 조커 발견 후 Enter 카운트
    
    while not done:
        viewer.step += 1
        viewer.update_game_state(env, obs)
        
        # 현재 플레이어 저장 (step 전)
        current_player = env._current_player
        
        phase_idx = int(np.argmax(obs["phase"]))
        phase_name = Phase(phase_idx).name
        
        action_mask = env.get_action_mask()
        viewer.update_action_mask(action_mask, phase_name)
        
        obs_tensor = obs_to_tensor(obs, device)
        action_mask_tensor = action_mask_to_tensor(action_mask, device)
        
        with torch.no_grad():
            action, log_probs, value = policy.get_action(
                obs_tensor, action_mask_tensor, deterministic=args.deterministic
            )
            
            # GUESS phase에서 value 확률 분포 계산
            value_probs = None
            if phase_name == "GUESS":
                features, constraint_per_pos = policy.encoder(obs_tensor)
                selected_position = torch.tensor([action[0][1]], dtype=torch.long, device=device)
                pos_embed = policy.action_heads.position_embedding(selected_position)
                batch_indices = torch.arange(1, device=device)
                pos_constraint = constraint_per_pos[batch_indices, selected_position]
                value_input = torch.cat([features, pos_embed, pos_constraint], dim=-1)
                value_logits = policy.action_heads.value_head(value_input)
                
                # Apply mask
                if action_mask_tensor is not None and "value" in action_mask_tensor:
                    mask = action_mask_tensor["value"].to(device)
                    value_logits = value_logits.masked_fill(~mask, -1e9)
                
                value_probs = torch.softmax(value_logits, dim=-1)[0].cpu().numpy()
        
        action = action[0]
        viewer.update_model_output(log_probs, value, value_probs, phase_name)
        
        # Check if this is a joker guess
        is_joker_guess = (phase_name == "GUESS" and action[2] == 12)
        
        # Determine if we should pause
        should_pause = not auto_mode or (joker_mode and is_joker_guess)
        
        if should_pause:
            if is_joker_guess:
                viewer.status_message = f"🃏 JOKER GUESS! [Enter]={enter_count+1}/2 steps, [c]=hunt now, [q]=quit"
            else:
                viewer.status_message = "Press Enter to execute action (q=quit, s=skip)"
            viewer.update()
            
            try:
                user_input = input().strip().lower()
                if user_input == 'q':
                    return {"quit": True}
                elif user_input == 's':
                    return {"skip": True}
                elif user_input == 'c' and is_joker_guess:
                    # 'c' 입력: 즉시 조커 헌팅 재개
                    joker_mode = True
                    auto_mode = True
                    enter_count = 0
                elif is_joker_guess and user_input == '':
                    # Enter 입력: 카운트 증가
                    enter_count += 1
                    if enter_count >= 2:
                        # 2번 Enter 후 자동으로 조커 헌팅 재개
                        joker_mode = True
                        auto_mode = True
                        enter_count = 0
                    else:
                        # 아직 2번 안됨, 한 단계씩
                        joker_mode = False
                        auto_mode = False
            except (KeyboardInterrupt, EOFError):
                return {"quit": True}
        else:
            viewer.status_message = "Auto-advancing..."
            viewer.update()
            time.sleep(args.delay)
        
        obs, _, reward, terminated, truncated, info, result = env.step(action)
        done = terminated or truncated
        
        # 현재 플레이어의 액션 결과 업데이트
        viewer.update_action_result(action, phase_name, result, reward, current_player)
        viewer.update_game_state(env, obs)
        
        # 조커 추측 후 결과 확인을 위해 한번 더 pause
        if is_joker_guess and should_pause:
            viewer.status_message = f"🃏 Result shown. [Enter]={enter_count}/2 (auto hunt after 2), [c]=hunt now, [q]=quit"
            viewer.update()
            
            if not auto_mode:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        return {"quit": True}
                    elif user_input == 'c':
                        # 즉시 조커 헌팅 재개
                        joker_mode = True
                        auto_mode = True
                        enter_count = 0
                    elif user_input == '':
                        # Enter 입력: 카운트 증가
                        enter_count += 1
                        if enter_count >= 2:
                            joker_mode = True
                            auto_mode = True
                            enter_count = 0
                except (KeyboardInterrupt, EOFError):
                    return {"quit": True}
            else:
                time.sleep(args.delay * 2)
        
        if not done and not is_joker_guess:
            viewer.status_message = "Action executed. Press Enter for next step"
            viewer.update()
            
            if auto_mode:
                time.sleep(args.delay / 2)
            else:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        return {"quit": True}
                    elif user_input == 's':
                        return {"skip": True}
                except (KeyboardInterrupt, EOFError):
                    return {"quit": True}
    
    viewer.winner = info.get("winner", None)
    viewer.status_message = "Episode complete! Press Enter to continue"
    viewer.update()
    
    if not args.auto:
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        time.sleep(args.delay * 2)
    
    return {
        "total_reward": viewer.total_reward,
        "steps": viewer.step,
        "winner": viewer.winner
    }


def main():
    args = parse_args()
    
    console.print(Panel(
        f"[bold]Da Vinci Code - Model Play Viewer[/]\n\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Auto mode: {args.auto} (delay: {args.delay}s)\n"
        f"Deterministic: {args.deterministic}\n"
        f"Episodes: {args.episodes}",
        title="🎮 Settings",
        border_style="blue"
    ))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Using device: {device}[/]")
    
    policy = DaVinciCodePolicy().to(device)
    policy.eval()
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        timesteps = checkpoint.get("timesteps", 0)
        console.print(f"[green]✓ Loaded checkpoint (trained for {timesteps:,} timesteps)[/]")
    else:
        console.print(f"[yellow]⚠ Checkpoint not found: {args.checkpoint}[/]")
        console.print("[yellow]Using randomly initialized model[/]")
    
    env = DaVinciCodeEnv()
    
    console.print("\n[bold]Starting in 2 seconds...[/]")
    console.print("[dim]Controls: Enter=next step, q=quit, s=skip episode[/]")
    time.sleep(2)
    
    viewer = GameViewer()
    viewer.start()
    
    try:
        stats = []
        for ep in range(1, args.episodes + 1):
            result = play_episode(policy, env, viewer, ep, args)
            
            if result.get("quit"):
                break
            
            if not result.get("skip"):
                stats.append(result)
    finally:
        viewer.stop()
    
    if stats:
        console.print("\n" + "=" * 50)
        total_rewards = [s.get("total_reward", 0) for s in stats]
        wins = sum(1 for s in stats if s.get("winner") == 0)
        
        summary_table = Table(title="📊 Session Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Episodes Played", str(len(stats)))
        summary_table.add_row("Player 0 Wins", f"{wins}/{len(stats)}")
        summary_table.add_row("Mean Reward", f"{np.mean(total_rewards):.2f}")
        
        console.print(summary_table)
    
    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
