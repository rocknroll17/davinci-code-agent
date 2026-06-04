"""
Da Vinci Code Game Visualizer using Rich Live.
학습 중 게임 상태를 실시간으로 시각화합니다.
"""

import time
from typing import List, Optional

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.constants import MAX_HAND_SIZE, CardValue, Color
from src.result.result import Result


class DaVinciVisualizer:
    """Rich 기반 다빈치 코드 게임 시각화."""
    
    def __init__(self):
        self.console = Console()
        self.live: Optional[Live] = None
        
        # 게임 상태
        self.my_hand = np.full((MAX_HAND_SIZE, 2), [Color.NONE, CardValue.NONE], dtype=np.int8)
        self.opponent_hand = np.full((MAX_HAND_SIZE, 2), [Color.NONE, CardValue.NONE], dtype=np.int8)
        self.phase = "DRAW"
        self.current_player = 0
        self.streak = 0
        self.deck_black = 12
        self.deck_white = 12
        
        # 학습 상태
        self.episode = 0
        self.timesteps = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.mean_reward = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        
        # 액션 & 로그
        self.last_action = "Waiting..."
        self.logs: List[str] = []
        self.max_logs = 6
    
    def _render_card(self, color: int, value: int) -> Text:
        """카드 하나를 Text로 렌더링."""
        if color == Color.NONE or value == CardValue.NONE:
            return Text("   ", style="on #90EE90")
        
        # 배경색
        if color == Color.BLACK:
            style = "bold white on black"
        else:
            style = "bold black on white"
        
        # 값
        if value == CardValue.HIDDEN:
            val_str = " ? "
        elif value == CardValue.JOKER:
            val_str = " - "
        else:
            val_str = f"{value:2d} "
        
        return Text(val_str, style=style)
    
    def _render_hand(self, hand: np.ndarray, title: str, is_opponent: bool = False) -> Panel:
        """플레이어 패 렌더링."""
        pos_row = Text()
        card_row = Text()
        
        has_cards = False
        for i in range(MAX_HAND_SIZE):
            color = int(hand[i, 0])
            value = int(hand[i, 1])
            
            if color == Color.NONE or value == CardValue.NONE:
                continue
            
            has_cards = True
            pos_row.append(f"{i:3d} ", style="dim")
            card_row.append(self._render_card(color, value))
            card_row.append(" ")
        
        if not has_cards:
            content = Text("(empty)", style="dim italic")
        else:
            content = Text()
            content.append(pos_row)
            content.append("\n")
            content.append(card_row)
        
        border_style = "red" if is_opponent else "green"
        return Panel(content, title=title, border_style=border_style, style="on #90EE90")
    
    def _render_stats(self) -> Panel:
        """통계 패널 렌더링."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold cyan", width=12)
        table.add_column(width=12)
        
        player_str = "[bold green]YOU[/]" if self.current_player == 0 else "[bold red]OPP[/]"
        
        table.add_row("Phase", f"[yellow]{self.phase}[/]")
        table.add_row("Player", player_str)
        table.add_row("Streak", str(self.streak))
        table.add_row("Deck", f"[white on black] B:{self.deck_black} [/] [black on white] W:{self.deck_white} [/]")
        table.add_row("─" * 12, "─" * 12)
        table.add_row("Episode", f"[bold]{self.episode:,}[/]")
        table.add_row("Timesteps", f"[bold]{self.timesteps:,}[/]")
        table.add_row("Reward", f"[{'green' if self.reward >= 0 else 'red'}]{self.reward:+.3f}[/]")
        table.add_row("Total", f"[bold]{self.total_reward:+.2f}[/]")
        table.add_row("Mean Ep R", f"[cyan]{self.mean_reward:.3f}[/]")
        
        return Panel(table, title="📊 Stats", border_style="blue", style="on #90EE90")
    
    def _render_action(self) -> Panel:
        """액션 패널 렌더링."""
        return Panel(
            Text(self.last_action, style="bold yellow"),
            title="🎯 Action",
            border_style="yellow",
            style="on #90EE90"
        )
    
    def _render_logs(self) -> Panel:
        """로그 패널 렌더링."""
        log_text = "\n".join(self.logs[-self.max_logs:]) if self.logs else "(no logs)"
        return Panel(log_text, title="📝 Log", border_style="magenta", style="on #90EE90")
    
    def _render_training(self) -> Panel:
        """학습 손실 패널."""
        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", width=12)
        table.add_column(width=10)
        table.add_row("Policy Loss", f"{self.policy_loss:.4f}")
        table.add_row("Value Loss", f"{self.value_loss:.4f}")
        return Panel(table, title="🧠 Training", border_style="cyan", style="on #90EE90")
    
    def render(self) -> Layout:
        """전체 레이아웃 렌더링."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=self.max_logs + 3)
        )
        
        # Header
        header_text = Text("🎴 Da Vinci Code Self-Play Training 🎴", style="bold white on #228B22", justify="center")
        layout["header"].update(Panel(header_text, style="on #90EE90"))
        
        # Main
        layout["main"].split_row(
            Layout(name="game", ratio=3),
            Layout(name="info", ratio=1)
        )
        
        layout["game"].split_column(
            Layout(self._render_hand(self.opponent_hand, "👤 Opponent (hidden)", is_opponent=True), name="opp"),
            Layout(self._render_hand(self.my_hand, "🎮 Your Hand"), name="you"),
            Layout(self._render_action(), name="action", size=3)
        )
        
        layout["info"].split_column(
            Layout(self._render_stats(), name="stats"),
            Layout(self._render_training(), name="training", size=5)
        )
        
        # Footer
        layout["footer"].update(self._render_logs())
        
        return layout
    
    def start(self):
        """Live 디스플레이 시작."""
        self.live = Live(self.render(), console=self.console, refresh_per_second=10, screen=True)
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
    
    def update_game_state(
        self,
        my_hand: np.ndarray,
        opponent_hand: np.ndarray,
        phase: str,
        current_player: int,
        streak: int,
        deck_black: int,
        deck_white: int,
        episode: int,
        timesteps: int,
        reward: float,
        total_reward: float,
        action: Optional[np.ndarray] = None,
        result: Optional['Result'] = None
    ):
        """게임 상태 업데이트."""
        self.my_hand = my_hand.copy()
        self.opponent_hand = opponent_hand.copy()
        self.phase = phase
        self.current_player = current_player
        self.streak = streak
        self.deck_black = deck_black
        self.deck_white = deck_white
        self.episode = episode
        self.timesteps = timesteps
        self.reward = reward
        self.total_reward = total_reward
        self.result = result
        
        if action is not None:
            self._set_action_text(current_player, phase, action, result)
        
        self.update()
    
    def _set_action_text(self, current_player: int, phase: str, action: np.ndarray, result: Optional['Result'] = None):
        """액션 텍스트 설정."""
        self.last_action = str(result) if result else "Problem in Result"
    
    def update_training_stats(self, mean_reward: float, policy_loss: float, value_loss: float):
        """학습 통계 업데이트."""
        self.mean_reward = mean_reward
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.update()
    
    def add_log(self, message: str):
        """로그 추가."""
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > 100:
            self.logs = self.logs[-50:]
        self.update()


# 전역 시각화 인스턴스
_visualizer: Optional[DaVinciVisualizer] = None


def get_visualizer() -> Optional[DaVinciVisualizer]:
    """현재 시각화 인스턴스 반환."""
    return _visualizer


def set_visualizer(viz: Optional[DaVinciVisualizer]):
    """시각화 인스턴스 설정."""
    global _visualizer
    _visualizer = viz


def create_visualizer() -> DaVinciVisualizer:
    """시각화 인스턴스 생성 및 설정."""
    viz = DaVinciVisualizer()
    set_visualizer(viz)
    return viz
