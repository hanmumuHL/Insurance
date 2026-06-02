"""统一日志 — loguru 封装"""
import sys
from pathlib import Path
from loguru import logger as _logger

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 移除默认 handler
_logger.remove()

# 控制台输出
_logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - {message}",
)

# 文件输出 (按天轮转)
_logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
)

logger = _logger
