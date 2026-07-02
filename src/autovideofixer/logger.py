"""Auto Video Fixer - Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# All module loggers are named "autovideofixer.<module>" (see get_logger() call
# sites). Handlers are attached ONLY here, once, so setup_logging()'s console/file
# configuration is the single source of truth -- see setup_logging()'s docstring
# for why this matters.
_ROOT_NAME = "autovideofixer"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a module logger that propagates up to the "autovideofixer" logger.

    Does NOT attach handlers or set propagate=False on `name` itself -- only
    setup_logging() does that, on the shared "autovideofixer" root logger. A
    per-module logger with its own handler and propagate=False (the previous
    design) can never be reached by a handler attached to a *different*
    logger name, which is why --log-file previously did nothing: it attached
    a FileHandler to a few hardcoded names ("autovideofixer.ai", etc.) that
    nothing actually logs through, while every real logger
    ("autovideofixer.pipeline", "autovideofixer.stages.upscale", ...) had
    already isolated itself from anything above it in the hierarchy.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    file_level: str | None = None,
) -> None:
    """Configure the single shared "autovideofixer" logger.

    Every module logger (get_logger("autovideofixer.<x>")) has no handlers of
    its own and propagates up to this one, so attaching handlers here is
    sufficient to reach all of them -- unlike the previous per-module-handler
    design, this actually gets log records to a file when log_file is given.

    Args:
        level: Console log level (DEBUG/INFO/WARNING/ERROR).
        log_file: If given, also log to this file.
        file_level: Log level for the file handler. Defaults to `level` if
            not given, so `--log-file` without `--file-log-level` behaves as
            users would expect (same verbosity as the console).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(numeric_level)
    root.propagate = False

    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    if log_file:
        file_numeric_level = getattr(logging, (file_level or level).upper(), numeric_level)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        file_handler.setLevel(file_numeric_level)
        root.addHandler(file_handler)
        # The root logger's effective level gates what reaches handlers at all,
        # so if the file wants more verbosity than the console, the logger
        # itself must be set to the more verbose of the two.
        root.setLevel(min(numeric_level, file_numeric_level))
