"""Game logic utilities for Da Vinci Code."""

from typing import List, Tuple

from src.hand import Hand
from src.constants import Color


def find_determined_cards(my_hand: 'Hand', opponent_hand: 'Hand') -> List[Tuple[int, int]]:
    """
    상대방의 미공개 카드 중에서 값이 확정된(정답이 하나뿐인) 카드를 찾는다.
    
    모든 가능한 값 할당을 CSP(Constraint Satisfaction Problem) 방식으로
    탐색하여, 모든 유효한 할당에서 동일한 값이 나오는 위치를 확정으로 판단한다.
    
    조커를 포함한 모든 카드 타입에 대해 완전한 계산을 수행한다.
    
    핸드 정렬 규칙:
    - 비조커 카드: (value, color) 오름차순. 같은 값이면 BLACK(0) < WHITE(1)
    - 조커: 어디든 배치 가능 (정렬 제약 없음)
    - 각 (color, value) 조합은 게임 전체에서 1장만 존재
    
    Args:
        my_hand: 내 핸드 (모든 카드 값을 알고 있음)
        opponent_hand: 상대 핸드 (공개된 카드의 값만 알고 있음, 색상은 항상 보임)
    
    Returns:
        확정된 미공개 카드의 리스트: [(position, determined_value), ...]
        확정된 카드가 없으면 빈 리스트
    """
    # 1. 이미 사용된 카드 세트 (color, value) 수집
    used_cards = set()
    for card in my_hand:
        used_cards.add((int(card.color), int(card.value)))
    for card in opponent_hand:
        if card.is_revealed:
            used_cards.add((int(card.color), int(card.value)))
    
    # 2. 미공개 카드 위치/색상 수집
    hidden_indices = []
    hidden_colors = {}
    for i, card in enumerate(opponent_hand):
        if not card.is_revealed:
            hidden_indices.append(i)
            hidden_colors[i] = int(card.color)
    
    if not hidden_indices:
        return []
    
    # 3. 각 색별 사용 가능한 값 계산
    # 각 색에는 0~11(숫자) + 12(조커) = 13장이 각 1장씩 존재
    available_values = {0: set(), 1: set()}  # BLACK=0, WHITE=1
    for color in [0, 1]:
        for v in range(13):  # 0-12
            if (color, v) not in used_cards:
                available_values[color].add(v)
    
    # 4. 핸드 정보 구성 (정렬 순서 체크용)
    hand_size = len(opponent_hand)
    revealed_info = {}  # pos -> (value, color, is_joker)
    for i, card in enumerate(opponent_hand):
        if card.is_revealed:
            revealed_info[i] = (int(card.value), int(card.color), card.is_joker)
    
    # 5. CSP 백트래킹으로 모든 유효 할당 탐색
    # reference_assignment: 첫 번째 유효 할당을 저장
    # still_determined: 아직 확정 후보인 위치 집합 (모든 유효 할당에서 같은 값이면 확정)
    reference_assignment = None
    still_determined = set(hidden_indices)
    
    def get_sort_key_at(pos, assignment):
        """
        pos 위치의 정렬 키를 반환.
        조커(값 12)나 공개된 조커면 None 반환 (정렬에서 제외).
        미할당 미공개 카드도 None 반환.
        """
        if pos in revealed_info:
            val, col, is_jk = revealed_info[pos]
            if is_jk:
                return None
            return (val, col)
        elif pos in assignment:
            val = assignment[pos]
            if val == 12:  # joker
                return None
            return (val, hidden_colors[pos])
        return None
    
    def is_order_valid_at(assignment, newly_assigned_pos):
        """
        새로 할당된 위치의 좌우 이웃과 정렬 순서가 유효한지 체크.
        조커(값 12)가 할당되면 어디든 가능하므로 True 반환.
        
        좌우에서 가장 가까운 비조커 확정 카드를 찾아 비교한다.
        """
        val = assignment[newly_assigned_pos]
        if val == 12:  # joker는 어디든 가능
            return True
        
        new_key = (val, hidden_colors[newly_assigned_pos])
        
        # 왼쪽에서 가장 가까운 비조커 확정 카드
        for pos in range(newly_assigned_pos - 1, -1, -1):
            key = get_sort_key_at(pos, assignment)
            if key is not None:
                if new_key <= key:
                    return False
                break
        
        # 오른쪽에서 가장 가까운 비조커 확정 카드
        for pos in range(newly_assigned_pos + 1, hand_size):
            key = get_sort_key_at(pos, assignment)
            if key is not None:
                if new_key >= key:
                    return False
                break
        
        return True
    
    def backtrack(idx, assignment, remaining):
        """
        백트래킹으로 미공개 카드에 가능한 값을 할당.
        
        hidden_indices 순서(왼쪽→오른쪽)로 값을 할당하며,
        정렬 제약과 유일성 제약을 만족하는 모든 유효 할당을 탐색.
        
        Early termination: 모든 위치가 미확정이면 즉시 종료.
        """
        nonlocal reference_assignment, still_determined
        
        # 확정 후보가 없으면 더 탐색할 필요 없음
        if not still_determined:
            return
        
        if idx == len(hidden_indices):
            # 모든 미공개 카드에 값이 할당됨 → 유효한 할당
            if reference_assignment is None:
                reference_assignment = dict(assignment)
            else:
                for pos in list(still_determined):
                    if assignment[pos] != reference_assignment[pos]:
                        still_determined.discard(pos)
            return
        
        pos = hidden_indices[idx]
        color = hidden_colors[pos]
        
        for val in sorted(remaining[color]):
            assignment[pos] = val
            remaining[color].remove(val)
            
            if is_order_valid_at(assignment, pos):
                backtrack(idx + 1, assignment, remaining)
                
                # Early exit: 확정 후보 없으면 종료
                if not still_determined:
                    remaining[color].add(val)
                    del assignment[pos]
                    return
            
            remaining[color].add(val)
            del assignment[pos]
    
    # Deep copy remaining values
    remaining = {c: set(v) for c, v in available_values.items()}
    backtrack(0, {}, remaining)
    
    if reference_assignment is None:
        return []
    
    # 6. 확정된 위치들 반환 (위치 순서대로)
    result = []
    for pos in sorted(still_determined):
        result.append((pos, reference_assignment[pos]))
    
    return result
