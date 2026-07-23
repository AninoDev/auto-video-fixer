"""Tests for content-dedup planning in `process`
(_plan_resolved_inputs / _build_content_groups / _content_signature /
_fanout_copy): content-identical inputs collapse onto a single
representative job, and other destinations for the same content become
fan-out copy targets executed after the representative finishes -- rather
than reprocessing the same bytes once per destination.

Also covers `Pipeline.resolve_output_path`, which `_plan_resolved_inputs`
uses to resolve each entry's actual output up front and which must return
the exact string `Pipeline.add_job` stores as `Job.output_path`.
"""

import io
import os
import re

import pytest
from rich.console import Console

from autovideofixer.cli.cli import (
    _content_signature,
    _fanout_copy,
    _plan_resolved_inputs,
)
from autovideofixer.config import Config
from autovideofixer.core.pipeline import Job, JobResult, Pipeline

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _capturing_console() -> Console:
    return Console(file=io.StringIO(), no_color=True)


def _console_text(console: Console) -> str:
    # Collapse Rich's own line-wrapping (default console width) so a
    # substring assertion doesn't break on where a message happened to wrap.
    text = _ANSI_RE.sub("", console.file.getvalue())  # type: ignore[attr-defined]
    return re.sub(r"\s+", " ", text)


def _video(tmp_path, name, content=b"fake video content"):
    f = tmp_path / name
    f.write_bytes(content)
    return str(f)


def _pipeline(tmp_path) -> Pipeline:
    return Pipeline(Config(tmp_path / "nonexistent.yaml"))


class TestSamePathPlanning:
    def test_same_path_twice_no_override_one_representative_no_fanout(self, tmp_path):
        path = _video(tmp_path, "a.mp4")
        console = _capturing_console()
        resolved = [(path, None), (path, None)]

        to_process, fanout = _plan_resolved_inputs(resolved, _pipeline(tmp_path), console)

        assert to_process == [(path, None)]
        assert fanout == {}

    def test_same_path_twice_different_outputs_one_representative_fanout(self, tmp_path):
        path = _video(tmp_path, "a.mp4")
        out1 = str(tmp_path / "out1.mp4")
        out2 = str(tmp_path / "out2.mp4")
        console = _capturing_console()
        resolved = [(path, out1), (path, out2)]

        to_process, fanout = _plan_resolved_inputs(resolved, _pipeline(tmp_path), console)

        assert to_process == [(path, out1)]
        assert fanout == {out1: [out2]}
        assert "additional destination" in _console_text(console)


class TestContentPlanning:
    def test_identical_content_default_output_fanout_to_others_default(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"identical bytes here")
        b = _video(tmp_path, "b.mp4", content=b"identical bytes here")
        console = _capturing_console()
        resolved = [(a, None), (b, None)]
        pipeline = _pipeline(tmp_path)

        to_process, fanout = _plan_resolved_inputs(resolved, pipeline, console)

        assert to_process == [(a, None)]
        rep_output = pipeline.resolve_output_path(a, None)
        extra_output = pipeline.resolve_output_path(b, None)
        assert rep_output != extra_output
        assert fanout == {rep_output: [extra_output]}

    def test_identical_content_same_explicit_output_one_representative_no_fanout(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"identical bytes here")
        b = _video(tmp_path, "b.mp4", content=b"identical bytes here")
        out = str(tmp_path / "shared_out.mp4")
        console = _capturing_console()
        resolved = [(a, out), (b, out)]

        to_process, fanout = _plan_resolved_inputs(resolved, _pipeline(tmp_path), console)

        assert to_process == [(a, out)]
        assert fanout == {}
        assert "Skipping duplicate input" in _console_text(console)

    def test_different_content_same_size_two_representatives_no_fanout(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"AAAAAAAAAA")
        b = _video(tmp_path, "b.mp4", content=b"BBBBBBBBBB")
        out = str(tmp_path / "shared_out.mp4")
        console = _capturing_console()
        resolved = [(a, out), (b, out)]

        to_process, fanout = _plan_resolved_inputs(resolved, _pipeline(tmp_path), console)

        # Same explicit output, but different content -- not a content
        # group, so both remain distinct jobs (both writing `out`, which is
        # a separate/pre-existing concern the planner doesn't own).
        assert to_process == [(a, out), (b, out)]
        assert fanout == {}

    def test_distinct_files_all_representatives_no_fanout(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"one")
        b = _video(tmp_path, "b.mp4", content=b"two")
        c = _video(tmp_path, "c.mp4", content=b"three-longer")
        console = _capturing_console()
        resolved = [(a, None), (b, None), (c, None)]

        to_process, fanout = _plan_resolved_inputs(resolved, _pipeline(tmp_path), console)

        assert to_process == [(a, None), (b, None), (c, None)]
        assert fanout == {}


class TestContentSignature:
    def test_small_file_hashes_whole_content(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"small content")
        b = _video(tmp_path, "b.mp4", content=b"small content")
        assert _content_signature(a) == _content_signature(b)

    def test_different_content_different_signature(self, tmp_path):
        a = _video(tmp_path, "a.mp4", content=b"content one")
        b = _video(tmp_path, "b.mp4", content=b"content two")
        assert _content_signature(a) != _content_signature(b)

    def test_large_file_samples_head_middle_tail(self, tmp_path):
        chunk = 1024 * 1024
        size = chunk * 4
        common = bytearray(size)
        common[0:10] = b"headhead12"
        common[size // 2 : size // 2 + 10] = b"middlemidd"
        common[size - 10 :] = b"tailtail12"

        a_content = bytes(common)
        b_content = bytearray(common)
        b_content[chunk + 5000] = (b_content[chunk + 5000] + 1) % 256

        a = tmp_path / "a.mp4"
        b = tmp_path / "b.mp4"
        a.write_bytes(a_content)
        b.write_bytes(bytes(b_content))

        assert _content_signature(str(a)) == _content_signature(str(b))


class TestResolveOutputPathMatchesAddJob:
    def test_default_output_matches_add_job(self, tmp_path):
        video = _video(tmp_path, "a.mp4")
        pipeline = _pipeline(tmp_path)

        expected = pipeline.resolve_output_path(video, None)
        job = pipeline.add_job(video)

        assert job.output_path == expected

    def test_override_output_matches_add_job_verbatim(self, tmp_path):
        video = _video(tmp_path, "a.mp4")
        override = str(tmp_path / "custom" / "out.mp4")
        pipeline = _pipeline(tmp_path)

        expected = pipeline.resolve_output_path(video, override)
        job = pipeline.add_job(video, output_path=override)

        assert expected == override
        assert job.output_path == expected


class TestFanoutCopy:
    def _job_and_result(self, source_path, outcome="completed"):
        job = Job(input_path=source_path, output_path=source_path)
        result = JobResult(input_path=source_path, output_path=source_path, outcome=outcome)
        return job, result

    def test_successful_copy(self, tmp_path):
        source = _video(tmp_path, "src.mp4", content=b"payload")
        extra = str(tmp_path / "extra" / "dest.mp4")
        job, result = self._job_and_result(source)
        console = _capturing_console()
        config = Config(tmp_path / "config.yaml")

        _fanout_copy(job, result, [extra], config, console)

        assert os.path.exists(extra)
        assert open(extra, "rb").read() == b"payload"
        assert "Copied output to" in _console_text(console)

    def test_overwrite_false_skips_existing_target(self, tmp_path):
        source = _video(tmp_path, "src.mp4", content=b"new payload")
        extra = tmp_path / "dest.mp4"
        extra.write_bytes(b"old payload")
        job, result = self._job_and_result(source)
        console = _capturing_console()
        config = Config(tmp_path / "config.yaml")
        config.set(False, "general", "overwrite")

        _fanout_copy(job, result, [str(extra)], config, console)

        assert extra.read_bytes() == b"old payload"
        assert "not overwriting" in _console_text(console)

    def test_missing_source_output_skips(self, tmp_path):
        missing_source = str(tmp_path / "missing.mp4")
        extra = str(tmp_path / "dest.mp4")
        job, result = self._job_and_result(missing_source)
        console = _capturing_console()
        config = Config(tmp_path / "config.yaml")

        _fanout_copy(job, result, [extra], config, console)

        assert not os.path.exists(extra)
        assert "source output not produced" in _console_text(console)

    def test_failed_outcome_skips_even_if_file_exists(self, tmp_path):
        source = _video(tmp_path, "src.mp4", content=b"payload")
        extra = str(tmp_path / "dest.mp4")
        job, result = self._job_and_result(source, outcome="failed")
        console = _capturing_console()
        config = Config(tmp_path / "config.yaml")

        _fanout_copy(job, result, [extra], config, console)

        assert not os.path.exists(extra)
        assert "source output not produced" in _console_text(console)

    def test_oserror_during_copy_reported_not_raised(self, tmp_path, monkeypatch):
        source = _video(tmp_path, "src.mp4", content=b"payload")
        extra = str(tmp_path / "dest.mp4")
        job, result = self._job_and_result(source)
        console = _capturing_console()
        config = Config(tmp_path / "config.yaml")

        import autovideofixer.cli.cli as cli_mod

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(cli_mod.shutil, "copy2", boom)

        _fanout_copy(job, result, [extra], config, console)

        assert not os.path.exists(extra)
        assert "Failed to copy" in _console_text(console)


if __name__ == "__main__":
    pytest.main([__file__])
