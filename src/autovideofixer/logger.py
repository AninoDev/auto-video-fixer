"""Auto Video Fixer - Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from autovideofixer.config import sanitize_console_text

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


_PLAIN_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _SanitizingConsoleFormatter(logging.Formatter):
    """Console-only formatter that neutralizes raw control/escape bytes.

    Attached ONLY to the console handler (RichHandler) -- file handlers keep
    the raw, unsanitized text, since a log file isn't a tty and the original
    bytes may matter for debugging. This runs *after* the normal
    ``%``-style message formatting (so ``%(message)s`` interpolation and any
    exception traceback text are both covered) but *before* RichHandler's
    own markup rendering -- Rich's ``[style]...[/style]`` bbcode-like markup
    tags are plain ASCII brackets/letters, not raw ANSI/C1 bytes, so
    sanitizing here can't damage deliberate Rich markup the code itself
    emits (e.g. via ``console.print(f"[red]...[/red]")`` or ``click.style``);
    it only strips control bytes that arrived embedded in the *data*
    (filenames, ffmpeg stderr excerpts, exception text) being logged.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return sanitize_console_text(formatted)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    file_level: str | None = None,
    auto_log_file: str | None = None,
) -> None:
    """Configure the single shared "autovideofixer" logger.

    Every module logger (get_logger("autovideofixer.<x>")) has no handlers of
    its own and propagates up to this one, so attaching handlers here is
    sufficient to reach all of them -- unlike the previous per-module-handler
    design, this actually gets log records to a file when log_file is given.

    Args:
        level: Console log level (DEBUG/INFO/WARNING/ERROR).
        log_file: If given, also log to this ADDITIONAL explicit file (e.g.
            CLI --log-file), at `file_level`/`level`.
        file_level: Log level for the `log_file` handler. Defaults to `level`
            if not given, so `--log-file` without `--file-log-level` behaves
            as users would expect (same verbosity as the console).
        auto_log_file: Path to the automatic always-on per-run log file (see
            config.get_log_dir()). Unlike `log_file`, this is always attached
            at DEBUG regardless of the console level, with a plain (no Rich
            markup) formatter, so an uploaded log is maximally useful
            independent of what the user had the console set to.
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
    console_handler.setFormatter(_SanitizingConsoleFormatter("%(message)s"))
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    effective_root_level = numeric_level

    if auto_log_file:
        auto_handler = logging.FileHandler(auto_log_file)
        auto_handler.setFormatter(logging.Formatter(_PLAIN_FILE_FORMAT))
        auto_handler.setLevel(logging.DEBUG)
        root.addHandler(auto_handler)
        effective_root_level = min(effective_root_level, logging.DEBUG)

    if log_file:
        file_numeric_level = getattr(logging, (file_level or level).upper(), numeric_level)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_PLAIN_FILE_FORMAT))
        file_handler.setLevel(file_numeric_level)
        root.addHandler(file_handler)
        effective_root_level = min(effective_root_level, file_numeric_level)

    # The root logger's effective level gates what reaches handlers at all,
    # so if any attached file handler wants more verbosity than the console,
    # the logger itself must be set to the most verbose of all of them.
    root.setLevel(effective_root_level)
