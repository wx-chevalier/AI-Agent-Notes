import sys
from loguru import logger

def setup_logger():
    """
    配置日志
    """
    # 移除默认处理器
    logger.remove()
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    return logger