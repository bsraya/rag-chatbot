from loguru import logger
import sys
from pathlib import Path

class Logger:
    _initialized = False

    @classmethod
    def setup(cls):
        if cls._initialized:
            return
        cls._initialized = True

        logger.remove()

        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>PID:{process.id}</magenta> | "
            "<cyan>{name}:{function}:{line}</cyan> - "
            "<level>{message}</level>"
        )

        log_file_path = Path("app.log")
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file_path,
            rotation="12:00",
            retention="3 days",
            compression="zip",
        )

        logger.add(
            sys.stderr,
            level="DEBUG",
            backtrace=True,
            diagnose=True,
            colorize=True,
            enqueue=True,
            serialize=False,
            format=console_format,
        )

        logger.debug("Loguru logger intialized")

loguru_logger = Logger()