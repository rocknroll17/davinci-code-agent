# validate_hand.py
from src.constants import CardValue, Color

FILE_PATH = ["logs/card_log_0.txt", "logs/card_log_1.txt"]

def parse_card(s: str):
    color, value_str = s.strip().split()
    if value_str == "-":
        return (Color.NONE, CardValue.JOKER)  # 조커는 CardValue.JOKER로 반환
    # 숫자 값 변환
    try:
        value = int(value_str)
    except ValueError:
        print(f"{color}, {value_str}")
        raise ValueError(f"잘못된 카드 값: {value_str}")
    return (Color[color.upper()], CardValue(value))  # 항상 CardValue 타입


def validate_hand(hand_line: str, line_num: int):
    """
    hand_line: "White 0, Black 4, White 5, ..."
    """
    cards_str = hand_line.strip().split(",")
    cards = [parse_card(c) for c in cards_str]

    prev_card = None
    for idx, card in enumerate(cards):
        color, value = card
        if prev_card is None:
            prev_card = card
            continue

        prev_color, prev_value = prev_card

        # 조커는 어디든 가능
        if prev_color == Color.NONE or color == Color.NONE:
            prev_card = card
            continue

        # 값 기준 오름차순 체크
        if value < prev_value:
            print(f"[Line {line_num}] 오류: 카드 {idx} ({color.name} {value}) 값이 이전 카드보다 작음")
        
        # 값 같으면 Black < White
        elif value == prev_value and prev_color.value > color.value:
            print(f"[Line {line_num}] 오류: 카드 {idx} ({color.name} {value}) 색상 순서 오류")

        prev_card = card

def validate_file(file_path: str):
    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            if line.strip():  # 빈 줄 무시
                validate_hand(line, i)

if __name__ == "__main__":
    for file_path in FILE_PATH:
        validate_file(file_path)
    print("검증 완료")
