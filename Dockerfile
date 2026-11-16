# Da Vinci Code — Self-Play RL Training
# README 기준 Python 3.10
FROM python:3.10-slim

WORKDIR /app

# 의존성 먼저 복사 → 레이어 캐시 활용.
# torch>=2.9.0 기본 휠 = CUDA 빌드. 학습은 `docker run --gpus all`로 GPU 사용.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 체크포인트/로그는 호스트에 보존하도록 볼륨으로 마운트 권장
#   docker run -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/logs:/app/logs ...
VOLUME ["/app/checkpoints", "/app/logs"]

# 컨테이너 환경엔 TTY가 없으므로 Rich 시각화 끄고 학습
CMD ["python", "main.py", "--no-viz"]
