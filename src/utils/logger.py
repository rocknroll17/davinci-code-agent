import logging
from logging.handlers import RotatingFileHandler
import os

LOG_FILE = "Game_log.log"
LOG_LEVEL = logging.INFO
ENABLE_FILE_LOG = False

logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)

# 기본: 아무 것도 남기지 않음
logger.handlers.clear()
logger.addHandler(logging.NullHandler())

# 파일 로깅은 명시적 환경변수로만 허용
if ENABLE_FILE_LOG:
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        ch = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5)
        ch.setLevel(LOG_LEVEL)
        formatter = logging.Formatter('%(asctime)s- %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)