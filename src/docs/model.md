# 모델 입·출력 설계 명세
모델 입력(INPUT) / 출력(OUTPUT) 구조 정의

## 1. INPUT (Observation Space)
- **Type:** `gym.spaces.Dict`
- **설명:** 데이터를 하나의 긴 벡터로 합치지 않고, 의미별로 키(Key)를 나누어 관리함. 이를 통해 구현 시 데이터 접근이 직관적이며 디버깅이 용이함.

### Structure (Keys)

```python
observation_space = spaces.Dict({
    "phase":             spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
    "my_hand":           spaces.Box(low=-2, high=12, shape=(13, 2), dtype=np.int8),
    "opponent_hand":     spaces.Box(low=-2, high=12, shape=(13, 2), dtype=np.int8),
    "remaining_deck":    spaces.Box(low=0, high=13, shape=(2,), dtype=np.int8),
    "constraint_matrix": spaces.Box(low=0, high=1, shape=(13, 13), dtype=np.int8)
})
```

---

### Phase

- 형태: One-hot Vector (3차원)
- 역할: 현재 활성화 가능한 Action Head를 결정하는 Gating 신호

| Vector | Phase | 설명 |
|------|------|------|
| [1, 0, 0] | DRAW | 타일 뽑기 단계 |
| [0, 1, 0] | GUESS | 상대 패 추리 단계 |
| [0, 0, 1] | DECISION | 계속 / 중단 선택 단계 |

---

### Opposite Deck (상대 패)

- 형태: 최대 13 슬롯
- 슬롯 구조: [color, value]
- 특징: 부분 관측 (Hidden 정보 포함)

opposite_deck = [
  [color, value],
  [color, value],
  ...
  (총 13 슬롯)
]

#### Color Encoding

| 값 | 의미 |
|---|---|
| 0 | BLACK |
| 1 | WHITE |
| -1 | NONE (카드 없음, 아직 카드를 받지 않음) |

#### Value Encoding

| 값 | 의미 |
|---|---|
| 0 ~ 11 | 숫자 카드 |
| -1 | HIDDEN (비공개 카드) |
| -2 | NONE (카드 없음, 아직 카드를 받지 않음) |

- HIDDEN: 카드가 존재하지만 값이 공개되지 않음(색상은 있음, 값은 비공개)
- NONE: 해당 슬롯에 카드가 존재하지 않음(색상, 값 모두 없음)

---

### My Deck (내 패)

- 구조: Opposite Deck과 동일 (13 × [color, value])
- 차이점:
  - 모든 카드가 공개됨(NONE은 존재 가능)
  - Color = -1 (NONE)은 존재 가능
  - Value = -1 (HIDDEN)은 존재하지 않음

### Remaining Deck (남은 덱 정보, 조커 포함)
remaining_deck = [
  black_count,
  white_count
]

### Contraint Matrix (제약 행렬)
다만 비공개 상대 패에 대한 추리 결과만 반영함.
게임 자체 규칙에 의한 제약(오름차순, 검정/흰색 배치 순서에 따른 제약 등)으로 발생하는 규칙에 의한 제약은 반영하지 않고 모델이 스스로 학습하도록 함.
- 형태: 13 x 13 이진 행렬
       0  1  ...  7  ... 12 (조커)
Slot0 [0, 0, ..., 1, ..., 0]  <-- "Slot 0은 7이 아니다" (마킹됨)
Slot1 [0, 0, ..., 0, ..., 0]


---


## 2. OUTPUT

OUTPUT = [
  color,
  position,
  value,
  decision
]

모든 출력은 Multi-Head Action Space의 일부이며  
현재 Phase에 따라 의미가 활성화된다.

---

### Output Head 정의

| Head | 의미 | 사용 Phase | 용도 |
|----|----|----|----|
| color | DRAW_COLOR (BLACK / WHITE) | DRAW | 뽑을 카드 색상 선택 |
| position | GUESS_POSITION (0 ~ 12) | GUESS | 상대 패 추리 위치 선택 |
| value | GUESS_VALUE (0 ~ 12) | GUESS | 상대 패 추리 값 선택 |
| decision | STOP / CONTINUE | DECISION | 한번 더 진행할지 결정 |

---

## 3. Phase별 행동 제약 처리 원칙

### DECISION Phase 제약

- 실패(오답) 직후에는:
  - DECISION Phase 자체가 호출되지 않음
  - CONTINUE 선택지는 모델 입력/출력 관점에서 존재하지 않음
- 즉, CONTINUE 불가 상황은:
  - 모델이 판단하는 문제가 아니라
  - **게임 엔진이 모델 호출 여부로 제어**

모델은 항상 “선택 가능한 상황”에서만 호출되며,  
불가능한 선택을 억지로 학습하지 않는다.

## 4. Action Mask & Rule Handling 원칙

- 게임 엔진은:
  - 가능한 행동만 모델에게 요청
  - 불가능한 Phase는 모델을 호출하지 않음
- 모델은:
  - Action Space 내부에서 불가능한 행동을
    - Action Mask
    - 또는 입력 상태(NONE, HIDDEN 등)
    를 통해 학습적으로 회피하도록 설계됨

즉,
- 규칙 위반 방지는 엔진 1차 책임
- 행동 선택의 최적화는 모델 책임

다만 불가능한 행동 선택 시 패널티 보상을 주고 다시 시도하게 함으로써, 모델의 학습 과정에서 규칙 준수를 강화할 수 있다.

---

## 5. Reward

| 항목 (Event) | 보상 값 (Value) |
|---|---|
| 게임 승리 (Win) | +10.0 |
| 게임 패배 (Lose) | -10.0 |
| 정답 추리 성공 (Guess Success) | +1.0 |
| 조커 추리 성공 (Joker Guess Success) | +2.0 |
| 오답 추리 실패 (Guess Fail) | -1.0 |
| 연속 정답 보너스 (Streak Bonus) | +(0.5 × Streak) |
| 연속 시도 중 실패 (Streak Break) | -0.5 (추가) |
| 불가능한 행동 (Invalid Action) | -5.0 |
| 턴 넘기기 (Stop Decision) | +0.1 |

---

## 6. Learning Signal Design
### 학습 단위 및 역전파 시점
모델은 게임 한 판(episode) 단위로 rollout을 수행한다.
정책 업데이트를 위한 역전파는
각 행동 시점의 누적 보상(return)을 기준으로 수행된다.
즉, 특정 행동의 역전파는
해당 행동 직후가 아니라 이후 결과를 포함한 보상에 의해 결정된다.


## 설계 요약

- Phase는 단순한 상태 정보가 아니라,  
  **현재 단계에서 의미 있는 Output만 활성화하기 위한 제어 신호(Gating Signal)**로 사용된다.

- 각 Phase마다 필요한 Action Head와 필요 없는 Action Head가 명확히 구분된다.
  - DRAW 단계: Color 출력만 의미 있음
  - GUESS 단계: Position, Value 출력만 의미 있음
  - DECISION 단계: Decision 출력만 의미 있음

- 모델 구조를 Phase-Gated 형태로 설계하여,
  **현재 Phase와 무관한 Output Head는 구조적으로 비활성화**되도록 한다.
  - 비활성 Head는 Logit을 0으로 만들거나
  - Action Mask를 통해 선택 불가능하게 처리한다.

- 이를 통해:
  - 불필요한 행동 탐색을 제거하고
  - Phase 간 Gradient 간섭을 방지하며
  - 학습 안정성과 효율을 동시에 확보한다.

- Hidden(-1)과 Empty(-2)를 명확히 구분한 입력 표현과  
  Phase 기반 Output 활성화 설계는  
  **추리 중심 게임을 Self-Play 강화학습으로 학습하기에 적합한 구조**를 만든다.

