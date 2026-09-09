"""Regression tests for run reporting on a CANCELLED run.

Cancelling an in-progress `avf process` run (Ctrl-C) used to report zeros for
every job result -- an empty Summary ("Total: 0") and an empty
"Job outcomes aggregate: {}" log line -- even though jobs had already finished
and been individually reported.

Root cause: the `process` command populated its `results` list solely from
`Pipeline.execute_all()`'s RETURN VALUE. A KeyboardInterrupt propagating out
of that call means the assignment never executes, so the `finally` reporting
tail ran against a list that was still empty. `execute_all()` invokes the
per-job `callback` for every job it completes, so the fix is to accumulate
results in that callback instead.

These tests drive the real `process` command through Click's CliRunner with
`Pipeline.execute_all` patched to complete some jobs and then raise, which is
exactly the shape of a real cancellation.
"""

from __future__ import annotations

import re

from click.testing import CliRunner

from autovideofixer.cli.cli import main
from autovideofixer.core.pipeline import JobResult

# Rich emits SGR colour codes even under CliRunner, so "Total: 1" appears as
# "Total: \x1b[1;36m1\x1b[0m". Strip them before substring-asserting.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    """Output with ANSI escape sequences removed."""
    return _ANSI_RE.sub("", text)


def _completed_result(path: str) -> JobResult:
    """A JobResult that reports as a successfully completed job."""
    return JobResult(input_path=path, output_path=path, success=True)


class TestCancelledRunReporting:
    """A cancelled run must still report the jobs that actually finished."""

    def setup_method(self):
        self.runner = CliRunner()

    def _run_with_interrupt(self, monkeypatch, video_path, completed_before_cancel):
        """Invoke `process`, completing N jobs then raising KeyboardInterrupt.

        Mirrors execute_all()'s real contract: it fires `callback(job, result)`
        for each finished job before control is lost.
        """
        from autovideofixer.core.pipeline import Pipeline

        def fake_execute_all(self, callback=None, progress_callback=None):
            for job in self._jobs[:completed_before_cancel]:
                if callback:
                    callback(job, _completed_result(job.input_path))
            raise KeyboardInterrupt()

        monkeypatch.setattr(Pipeline, "execute_all", fake_execute_all)

        return self.runner.invoke(
            main,
            ["process", str(video_path), "--no-progress-batch", "--no-progress-file"],
            catch_exceptions=True,
        )

    def test_cancelled_run_reports_completed_jobs(self, monkeypatch, tmp_video_file):
        """The finished job must appear in the summary, not be reported as 0."""
        result = self._run_with_interrupt(monkeypatch, tmp_video_file, 1)
        out = _plain(result.output)

        # The reporting tail runs in a `finally`, so it still prints on cancel.
        assert "Summary:" in out, f"no summary printed; output={out!r}"
        assert "Total: 1" in out, (
            "cancelled run reported the wrong job total -- the regression is "
            f"back (expected 'Total: 1'); output={out!r}"
        )
        assert "Success: 1" in out
        assert "Total: 0" not in out

    def test_cancelled_run_before_any_job_reports_zero(self, monkeypatch, tmp_video_file):
        """Cancelling before any job finishes legitimately reports zero.

        Guards the fix from overshooting into counting jobs that never ran.
        """
        result = self._run_with_interrupt(monkeypatch, tmp_video_file, 0)
        out = _plain(result.output)

        assert "Summary:" in out
        assert "Total: 0" in out

    def test_results_not_double_counted_on_normal_completion(self, monkeypatch, tmp_video_file):
        """A normal (uncancelled) run must count each job exactly once.

        The fix accumulates in the callback AND execute_all still returns a
        list; if the return value were also appended, every job would be
        counted twice.
        """
        from autovideofixer.core.pipeline import Pipeline

        def fake_execute_all(self, callback=None, progress_callback=None):
            results = []
            for job in self._jobs:
                r = _completed_result(job.input_path)
                results.append(r)
                if callback:
                    callback(job, r)
            return results

        monkeypatch.setattr(Pipeline, "execute_all", fake_execute_all)

        result = self.runner.invoke(
            main,
            ["process", str(tmp_video_file), "--no-progress-batch", "--no-progress-file"],
            catch_exceptions=True,
        )

        out = _plain(result.output)
        assert "Total: 1" in out, (
            f"job counted more than once (double-accumulation); output={out!r}"
        )
