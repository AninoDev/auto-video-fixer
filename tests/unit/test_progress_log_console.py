"""The live progress region must own the log handler's console while active.

Rich coordinates output ONLY between writers that share one `Console` object.
A `Progress` live region renders anything printed through its own console
cleanly above the bars, but knows nothing about writes made through a DIFFERENT
Console -- those hit the terminal mid-redraw and shred the display.

That was exactly the split here: `process`'s per-job report tables already print
through the shared CLI console handed to `ProgressReporter`, so they interleaved
correctly, but the log handler owns a separate stderr console built by
`setup_logging()`. Every INFO/DEBUG line emitted while the bars were live was
spliced into them, which is what made the console output unusable.

`ProgressReporter` now borrows the progress console for the duration of its live
region and restores the original on exit. The swap only happens when a region
actually starts (which requires a real TTY), so piped/redirected runs are
unaffected and `avf config dump`'s machine-readable stdout is never in scope.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from autovideofixer.cli.progress import ProgressReporter
from autovideofixer.logger import swap_log_console


def _attach_handler(console):
    """Attach a RichHandler on the real root logger name, as setup_logging does."""
    root = logging.getLogger("autovideofixer")
    previous = list(root.handlers)
    root.handlers[:] = [RichHandler(console=console, show_time=False, show_path=False)]
    return root, previous


def _current_handler_console():
    for handler in logging.getLogger("autovideofixer").handlers:
        if isinstance(handler, RichHandler):
            return handler.console
    return None


class TestSwapLogConsole:
    def test_swaps_and_reports_previous(self):
        original = Console()
        root, saved = _attach_handler(original)
        try:
            replacement = Console()
            returned = swap_log_console(replacement)

            assert returned is original, "did not report the console it replaced"
            assert _current_handler_console() is replacement
        finally:
            root.handlers[:] = saved

    def test_none_reports_current_without_swapping(self):
        original = Console()
        root, saved = _attach_handler(original)
        try:
            assert swap_log_console(None) is original
            assert _current_handler_console() is original
        finally:
            root.handlers[:] = saved

    def test_no_handler_is_a_noop(self):
        root = logging.getLogger("autovideofixer")
        saved = list(root.handlers)
        root.handlers[:] = []
        try:
            assert swap_log_console(Console()) is None
        finally:
            root.handlers[:] = saved


class TestReporterBorrowsConsole:
    def _reporter(self, progress_console):
        return ProgressReporter(
            progress_console, total_jobs=1, batch_enabled=True, file_enabled=True
        )

    def test_log_console_is_the_progress_console_inside_the_region(self):
        """The whole point: one Console for bars, tables AND logs while live."""
        log_console = Console()
        progress_console = Console()
        root, saved = _attach_handler(log_console)
        try:
            with self._reporter(progress_console):
                assert _current_handler_console() is progress_console, (
                    "log handler kept its own console inside the live region -- "
                    "log lines will be written straight to the terminal mid-redraw "
                    "and garble the bars"
                )
        finally:
            root.handlers[:] = saved

    def test_original_console_restored_on_exit(self):
        log_console = Console()
        progress_console = Console()
        root, saved = _attach_handler(log_console)
        try:
            with self._reporter(progress_console):
                pass
            assert _current_handler_console() is log_console
        finally:
            root.handlers[:] = saved

    def test_original_console_restored_even_when_the_run_raises(self):
        """A cancelled/failed run must not leave logging pointed at a dead region."""
        log_console = Console()
        progress_console = Console()
        root, saved = _attach_handler(log_console)
        try:
            try:
                with self._reporter(progress_console):
                    raise KeyboardInterrupt()
            except KeyboardInterrupt:
                pass
            assert _current_handler_console() is log_console
        finally:
            root.handlers[:] = saved

    def test_reporter_works_with_no_log_handler_attached(self):
        """Must not blow up when logging was never configured."""
        root = logging.getLogger("autovideofixer")
        saved = list(root.handlers)
        root.handlers[:] = []
        try:
            with self._reporter(Console()):
                pass
        finally:
            root.handlers[:] = saved
