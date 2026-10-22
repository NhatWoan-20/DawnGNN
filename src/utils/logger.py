"""Logging utilities for DawnGNN."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dawngnn",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a named logger with console (and optional file) output.

    Args:
        name: Logger name (used to namespace log messages).
        level: Logging level (default: INFO).
        log_file: Optional path to a log file.
        format_string: Optional custom format string.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when called multiple times
    logger.handlers.clear()

    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "dawngnn") -> logging.Logger:
    """Return an existing logger, or create a default one if none is configured."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
