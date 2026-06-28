"""Auto Video Fixer - Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance with rich output."""
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
    elif not logger.handlers:
        logger.setLevel(logging.INFO)

    if not logger.handlers:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
