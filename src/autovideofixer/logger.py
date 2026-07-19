"""Auto Video Fixer - Logging utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from autovideofixer.config import VALID_LOG_TYPES, sanitize_console_text
from autovideofixer.logclean import get_pii_cleaner

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


class _CleaningFileFormatter(logging.Formatter):
    """File formatter for the "clean" log_type variant (REQUIREMENTS.md §
    6.7): runs the normal plain ``_PLAIN_FILE_FORMAT`` formatting, then
    substitutes every KNOWN value the current run has registered with
    ``logclean.get_pii_cleaner()`` (input/output paths, directories, VLM/LLM
    endpoints, embedded titles) for its placeholder.

    Reads the singleton at format() time (emit time), not __init__ time --
    registration only has to happen before the LOG CALL that contains a given
    value, not before setup_logging() itself. See AGENTS.md's "Log types"
    note for the known v1 gaps in registration ordering (e.g. the group
    callback's "Invocation:" line).
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return get_pii_cleaner().clean(formatted)


def _insert_log_suffix(path: str, suffix: str) -> str:
    """Insert `suffix` into `path`'s filename for "both" log_type mode.

    Before the extension when one exists (``run.log`` -> ``run-clean.log``);
    appended to the end for an extensionless name (``run`` -> ``run-clean``).
    An empty suffix returns `path` unchanged.
    """
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    file_level: str | None = None,
    auto_log_file: str | None = None,
    log_type: str = "raw",
    log_suffix_raw: str = "",
    log_suffix_clean: str = "-clean",
) -> list[str]:
    """Configure the single shared "autovideofixer" logger.

    Every module logger (get_logger("autovideofixer.<x>")) has no handlers of
    its own and propagates up to this one, so attaching handlers here is
    sufficient to reach all of them -- unlike the previous per-module-handler
    design, this actually gets log records to a file when log_file is given.

    Args:
        level: Console log level (DEBUG/INFO/WARNING/ERROR).
        log_file: If given, also log to this ADDITIONAL explicit file (e.g.
            a direct caller that isn't the `avf` CLI group callback), at
            `file_level`/`level`. Kept ALWAYS raw regardless of `log_type`
            (backward compatibility for direct callers that predate § 6.7 --
            the CLI itself never passes this, see cli.py's `main()`).
        file_level: Log level for the `log_file` handler. Defaults to `level`
            if not given, so `--log-file` without `--file-log-level` behaves
            as users would expect (same verbosity as the console).
        auto_log_file: Path to the automatic always-on per-run log file (see
            config.get_log_dir()). Unlike `log_file`, this is always attached
            at DEBUG regardless of the console level, with a plain (no Rich
            markup) formatter, so an uploaded log is maximally useful
            independent of what the user had the console set to. `log_type`
            below governs what variant(s) of THIS file get written.
        log_type: "raw" (default; today's behavior, unredacted) | "clean"
            (PII-substituted via logclean.PIICleaner) | "both" (two files,
            suffixed per log_suffix_raw/log_suffix_clean) | "none" (no file
            handler for `auto_log_file` at all). Only applies to
            `auto_log_file` -- the console is always raw, and the separate
            `log_file` param (see above) is always raw too.
        log_suffix_raw: "both" mode only -- suffix for the raw file's name.
        log_suffix_clean: "both" mode only -- suffix for the clean file's name.

    Returns:
        The list of file paths actually opened for `auto_log_file` (0, 1, or
        2 entries depending on `log_type`), PLUS `log_file` if given (in that
        order) -- callers (e.g. cli.py) use this to report the real path(s)
        instead of assuming `auto_log_file` unconditionally.

    Raises:
        ValueError: `log_type` isn't one of the four valid values, or "both"
            mode's raw/clean suffixes would produce two IDENTICAL paths (a
            silent overwrite of one file by the other is never acceptable --
            this is a config error the caller should report and exit on).
    """
    if log_type not in VALID_LOG_TYPES:
        raise ValueError(f"Invalid log_type: {log_type!r} (must be one of {VALID_LOG_TYPES!r})")

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
    opened_files: list[str] = []

    if auto_log_file and log_type != "none":
        if log_type == "both":
            raw_path = _insert_log_suffix(auto_log_file, log_suffix_raw)
            clean_path = _insert_log_suffix(auto_log_file, log_suffix_clean)
            if raw_path == clean_path:
                raise ValueError(
                    "general.log_type=both requires distinct log_suffix_raw/"
                    f"log_suffix_clean -- both currently resolve to the same path "
                    f"({raw_path!r}); refusing to silently overwrite one file with "
                    "the other"
                )
            variants = [(raw_path, False), (clean_path, True)]
        elif log_type == "clean":
            variants = [(auto_log_file, True)]
        else:  # "raw"
            variants = [(auto_log_file, False)]

        for path, clean in variants:
            handler = logging.FileHandler(path)
            handler.setFormatter(
                _CleaningFileFormatter(_PLAIN_FILE_FORMAT)
                if clean
                else logging.Formatter(_PLAIN_FILE_FORMAT)
            )
            handler.setLevel(logging.DEBUG)
            root.addHandler(handler)
            opened_files.append(path)
        effective_root_level = min(effective_root_level, logging.DEBUG)

    if log_file:
        file_numeric_level = getattr(logging, (file_level or level).upper(), numeric_level)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_PLAIN_FILE_FORMAT))
        file_handler.setLevel(file_numeric_level)
        root.addHandler(file_handler)
        effective_root_level = min(effective_root_level, file_numeric_level)
        opened_files.append(log_file)

    # The root logger's effective level gates what reaches handlers at all,
    # so if any attached file handler wants more verbosity than the console,
    # the logger itself must be set to the most verbose of all of them.
    root.setLevel(effective_root_level)

    return opened_files
