"""Tests for Feature 6's live progress bars (cli/progress.py).

Deliberately do NOT assert live-rendered terminal output -- these test the
pure arithmetic (batch fractional-completed, clamping), the enable
resolution helper, and ProgressReporter's state transitions using a
non-terminal `Console(file=io.StringIO())` so no real TTY is needed.
"""

from __future__ import annotations

import io

from rich.console import Console

from autovideofixer.cli.progress import (
    ProgressReporter,
    batch_completed,
    clamp01,
    resolve_show_progress,
)
from autovideofixer.core.pipeline import Job


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


class TestClamp01:
    def test_within_range_unchanged(self):
        assert clamp01(0.5) == 0.5

    def test_below_zero_clamped(self):
        assert clamp01(-0.3) == 0.0

    def test_above_one_clamped(self):
        assert clamp01(1.7) == 1.0

    def test_boundaries(self):
        assert clamp01(0.0) == 0.0
        assert clamp01(1.0) == 1.0


class TestBatchCompleted:
    def test_zero_jobs_done_partial_progress(self):
        assert batch_completed(0, 5, 0.4) == 0.4

    def test_some_jobs_done_plus_current_fraction(self):
        assert batch_completed(2, 5, 0.25) == 2.25

    def test_clamps_current_progress_out_of_range(self):
        assert batch_completed(1, 5, 1.7) == 2.0
        assert batch_completed(1, 5, -0.5) == 1.0

    def test_clamps_to_total_when_over(self):
        # jobs_done alone already at/over total -- must not exceed total_jobs.
        assert batch_completed(5, 5, 0.9) == 5.0

    def test_clamps_to_zero_floor(self):
        assert batch_completed(0, 5, 0.0) == 0.0


class TestResolveShowProgress:
    def test_both_enabled_terminal(self):
        assert resolve_show_progress(True, True, True) == (True, True)

    def test_both_enabled_non_terminal(self):
        assert resolve_show_progress(True, True, False) == (False, False)

    def test_only_batch_enabled_terminal(self):
        assert resolve_show_progress(True, False, True) == (True, False)

    def test_only_file_enabled_terminal(self):
        assert resolve_show_progress(False, True, True) == (False, True)

    def test_both_disabled_terminal(self):
        assert resolve_show_progress(False, False, True) == (False, False)

    def test_non_terminal_overrides_enabled_flags(self):
        assert resolve_show_progress(True, True, False) == (False, False)

    def test_console_is_terminal_true_but_isatty_false_yields_no_bars(self):
        # Regression for the redirected-output leak: Rich's console.is_terminal
        # can report True even when stdout isn't a real POSIX terminal (some
        # redirected/piped setups). The caller (cli.py::process) must combine
        # it with the real fd-level sys.stdout.isatty() signal via `and`
        # before calling resolve_show_progress() -- simulate that combination
        # here and confirm both bars stay off regardless of the enable flags.
        console_is_terminal = True
        stdout_isatty = False
        combined = console_is_terminal and stdout_isatty
        assert resolve_show_progress(True, True, combined) == (False, False)

    def test_both_true_yields_bars_per_enable_flags(self):
        console_is_terminal = True
        stdout_isatty = True
        combined = console_is_terminal and stdout_isatty
        assert resolve_show_progress(True, False, combined) == (True, False)


class TestProgressReporterStateTransitions:
    def test_on_progress_updates_jobs_done_zero_batch_fraction(self):
        job = Job(input_path="/tmp/a.mp4")
        reporter = ProgressReporter(_console(), total_jobs=3, batch_enabled=True, file_enabled=True)
        with reporter:
            reporter.on_progress(job, 0.5, "encoding")
            assert reporter._progress.tasks[reporter._batch_task].completed == 0.5
            assert reporter._progress.tasks[reporter._file_task].completed == 0.5

    def test_on_complete_increments_jobs_done_and_resets_file_task(self):
        job = Job(input_path="/tmp/a.mp4")
        reporter = ProgressReporter(_console(), total_jobs=3, batch_enabled=True, file_enabled=True)
        with reporter:
            reporter.on_progress(job, 0.8, "encoding")
            assert reporter.jobs_done == 0
            reporter.on_complete(job, result=None)
            assert reporter.jobs_done == 1
            assert reporter._progress.tasks[reporter._file_task].completed == 0.0
            assert reporter._progress.tasks[reporter._batch_task].completed == 1.0

    def test_job_switch_resets_file_task(self):
        job_a = Job(input_path="/tmp/a.mp4")
        job_b = Job(input_path="/tmp/b.mp4")
        reporter = ProgressReporter(_console(), total_jobs=2, batch_enabled=True, file_enabled=True)
        with reporter:
            reporter.on_progress(job_a, 0.9, "encoding")
            assert reporter._progress.tasks[reporter._file_task].completed == 0.9
            # Switching to a new job (without an intervening on_complete, e.g.
            # concurrent jobs interleaving) must reset the file bar to that
            # job's own fresh progress, not carry job_a's 0.9 forward.
            reporter.on_progress(job_b, 0.1, "stabilize")
            assert reporter._progress.tasks[reporter._file_task].completed == 0.1

    def test_batch_bar_advances_smoothly_across_multiple_jobs(self):
        job_a = Job(input_path="/tmp/a.mp4")
        job_b = Job(input_path="/tmp/b.mp4")
        reporter = ProgressReporter(
            _console(), total_jobs=2, batch_enabled=True, file_enabled=False
        )
        with reporter:
            reporter.on_progress(job_a, 0.5, "x")
            assert reporter._progress.tasks[reporter._batch_task].completed == 0.5
            reporter.on_complete(job_a, result=None)
            assert reporter._progress.tasks[reporter._batch_task].completed == 1.0
            reporter.on_progress(job_b, 0.3, "y")
            assert reporter._progress.tasks[reporter._batch_task].completed == 1.3

    def test_disabled_bar_has_no_task(self):
        job = Job(input_path="/tmp/a.mp4")
        reporter = ProgressReporter(
            _console(), total_jobs=1, batch_enabled=True, file_enabled=False
        )
        with reporter:
            assert reporter._file_task is None
            assert reporter._batch_task is not None
            # Should not raise even with no file task.
            reporter.on_progress(job, 0.5, "x")
            reporter.on_complete(job, result=None)

    def test_both_disabled_no_tasks_created(self):
        reporter = ProgressReporter(
            _console(), total_jobs=1, batch_enabled=False, file_enabled=False
        )
        assert not reporter.active
        with reporter:
            assert reporter._batch_task is None
            assert reporter._file_task is None
