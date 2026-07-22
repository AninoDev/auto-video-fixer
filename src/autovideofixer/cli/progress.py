"""Feature 6: live nested progress bars for `avf process`.

Two independent `rich.progress` bars sharing one `Progress` live region:

- BATCH bar: total = number of jobs, advancing FRACTIONALLY as
  `jobs_done + current_job_progress` so it moves smoothly within a video,
  not just in whole-video steps.
- PER-FILE bar: total = 1.0, reset at the start of each job, tracking that
  job's own 0..1 progress (as reported by `Pipeline.execute_job`'s
  `progress_callback`).

Both are independently enabled/disabled (`reporting.progress_batch` /
`reporting.progress_file`, overridable via `--progress-batch/--no-...` and
`--progress-file/--no-...`) and only ever shown when BOTH Rich's
`console.is_terminal` AND the real fd-level `sys.stdout.isatty()` are true --
the caller (`cli.py::process`) combines the two before calling
`resolve_show_progress()`, since `console.is_terminal` alone can report True
even when stdout isn't actually a POSIX terminal (e.g. some redirected/piped
setups), which would otherwise leak bar-redraw control codes into captured
output.

Concurrency note (v1 limitation): when `general.max_concurrent_jobs` > 1,
`execute_job` for several jobs can be in flight at once and their
`progress_callback` invocations interleave on whichever thread happens to
call in. This module does not attempt to render one file bar per concurrent
job -- the single file bar simply reflects whichever job most recently
reported progress, and switching to a different job's update resets the bar
to that job's own progress (see `on_progress`'s job-switch handling). The
batch bar is unaffected: it always reflects true completed-job count plus
the most recently reported in-flight fraction. Multiple file bars for
concurrent runs is left for a future iteration.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn

if TYPE_CHECKING:
    from rich.console import Console

    from autovideofixer.core.pipeline import Job, JobResult


def clamp01(value: float) -> float:
    """Clamp a fractional progress value to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def batch_completed(jobs_done: int, total_jobs: int, current_progress: float) -> float:
    """Fractional batch-bar completed value.

    `jobs_done` fully-finished jobs plus the current job's own 0..1
    progress, clamped to [0, total_jobs] so a stray progress value can't
    push the bar past the end (or below the start).
    """
    raw = jobs_done + clamp01(current_progress)
    if raw < 0.0:
        return 0.0
    if raw > total_jobs:
        return float(total_jobs)
    return raw


def resolve_show_progress(
    batch_enabled: bool, file_enabled: bool, is_real_terminal: bool
) -> tuple[bool, bool]:
    """Resolve the two config/CLI-level enable flags against TTY attachment.

    Returns (batch_shown, file_shown). Each bar is only ever shown when its
    own flag is truthy AND `is_real_terminal` is True -- piped/redirected/
    non-TTY output silently gets no bars at all, same as today's behavior
    with the feature absent.

    `is_real_terminal` must already be the caller's fully-combined "is this
    actually an interactive terminal" decision -- Rich's `console.is_terminal`
    alone can return True even when stdout isn't a real POSIX terminal (e.g.
    some redirected/piped setups still report as a terminal to Rich), so the
    caller is expected to AND it with the real fd-level `sys.stdout.isatty()`
    signal before calling this function. This function itself stays a pure,
    trivially-testable boolean gate -- it doesn't know or care how its single
    boolean input was derived.
    """
    if not is_real_terminal:
        return False, False
    return bool(batch_enabled), bool(file_enabled)


class ProgressReporter:
    """Owns the live `rich.progress.Progress` region for `avf process`.

    Construct with the shared CLI `console` so per-job Rich report tables
    (printed via `console.print` during the run) render *above* the live
    bars instead of corrupting them -- `rich.progress.Progress` supports
    interleaved `console.print` calls while its live region is active.
    """

    def __init__(
        self,
        console: Console,
        total_jobs: int,
        batch_enabled: bool,
        file_enabled: bool,
    ) -> None:
        self.console = console
        self.total_jobs = max(total_jobs, 0)
        self.batch_enabled = batch_enabled
        self.file_enabled = file_enabled

        self.jobs_done = 0
        self._current_job_key: str | None = None

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        )
        self._batch_task: TaskID | None = None
        self._file_task: TaskID | None = None

    @property
    def active(self) -> bool:
        return self.batch_enabled or self.file_enabled

    def __enter__(self) -> ProgressReporter:
        self._progress.start()
        if self.batch_enabled:
            self._batch_task = self._progress.add_task(
                self._batch_description(0.0), total=max(self.total_jobs, 1)
            )
        if self.file_enabled:
            self._file_task = self._progress.add_task("Waiting for first job...", total=1.0)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._progress.stop()

    def _batch_description(self, current_progress: float) -> str:
        return f"Batch {self.jobs_done}/{self.total_jobs}"

    def _file_description(self, job: Job, message: str) -> str:
        name = os.path.basename(job.input_path)
        return f"{name} -- {message}" if message else name

    def on_progress(self, job: Job, progress: float, message: str) -> None:
        """Called from `Pipeline.execute_job`'s `progress_callback` on every
        per-stage progress update (see pipeline.py's `progress_cb` closure) --
        i.e. as often as the currently-running stage itself reports, not just
        once per stage."""
        job_key = job.input_path
        if job_key != self._current_job_key:
            self._current_job_key = job_key
            if self._file_task is not None:
                self._progress.reset(self._file_task, total=1.0)

        clamped = clamp01(progress)

        if self._file_task is not None:
            self._progress.update(
                self._file_task,
                completed=clamped,
                description=self._file_description(job, message),
            )

        if self._batch_task is not None:
            completed = batch_completed(self.jobs_done, self.total_jobs, clamped)
            self._progress.update(
                self._batch_task,
                completed=completed,
                description=self._batch_description(clamped),
            )

    def on_complete(self, job: Job, result: JobResult) -> None:
        """Called from the CLI's job-completion callback right after a job
        finishes (success, failure, or skip)."""
        self.jobs_done += 1
        self._current_job_key = None

        if self._batch_task is not None:
            completed = batch_completed(self.jobs_done, self.total_jobs, 0.0)
            self._progress.update(
                self._batch_task,
                completed=completed,
                description=self._batch_description(0.0),
            )

        if self._file_task is not None:
            self._progress.reset(self._file_task, total=1.0)
            self._progress.update(self._file_task, completed=0.0, description="Waiting...")
