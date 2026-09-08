"""
SONAR-GUARD Structured Logger
==============================
Central logging configuration. Import get_logger() in every module.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Console formatter with ANSI colour codes for readability."""
    COLORS = {
        logging.DEBUG:    "\033[36m",    # Cyan
        logging.INFO:     "\033[32m",    # Green
        logging.WARNING:  "\033[33m",    # Yellow
        logging.ERROR:    "\033[31m",    # Red
        logging.CRITICAL: "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Module-level root configuration (called once at import time)
# ---------------------------------------------------------------------------
_configured = False


def _configure_root(log_level: str = "INFO", log_dir: Path = None) -> None:
    """
    Configure root logger with:
      - Console handler (coloured)
      - File handler (plain text, rotating daily)
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("sonarguard")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # --- Console ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColorFormatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)

    # --- File (if log_dir provided) ---
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sonarguard_{timestamp}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)
        root.info("Log file: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'sonarguard' namespace.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        Logger: Configured logger instance.
    """
    # Initialise with defaults on first call from any module
    _configure_root()
    # Strip the project prefix to keep names short in logs
    short_name = name.replace("src.", "").replace("__main__", "app")
    return logging.getLogger(f"sonarguard.{short_name}")


def setup_logging(log_level: str = "INFO", log_dir: Path = None) -> None:
    """
    Explicit initialisation — call once in app.py or CLI entry points
    to set log level and enable file logging.

    Args:
        log_level: One of DEBUG / INFO / WARNING / ERROR.
        log_dir:   Directory to write log files. If None, file logging disabled.
    """
    global _configured
    _configured = False   # Reset so _configure_root runs again with new params
    _configure_root(log_level=log_level, log_dir=log_dir)

