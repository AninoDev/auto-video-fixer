"""Tests for the AI/RIFE interpolation stage's streaming frame pipeline.

Covers the RAM-blowup fix in ``InterpolateStage._execute_ai``: the stage used
to accumulate the ENTIRE interpolated output (factor * input_frames) in a
Python list before writing it out once via ``frames_to_video()`` -- for a
long 4K clip that could balloon to tens of GB resident. It now streams every
chunk straight to a frame writer (mirroring ``UpscaleStage._execute_ai``) and
never holds more than one chunk's worth of output frames at a time.

Everything GPU/model-related is faked: no torch/CUDA/real RIFE model is
required to run these tests.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.interpolate import InterpolateStage

FRAME_SHAPE = (4, 4, 3)


def _fake_frame() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


class FakeReader:
    """Records nothing fancy -- just hands back pre-chunked batches."""

    def __init__(self, batches: list[list[Any]]):
        self._batches = list(batches)
        self.closed = False

    def next_batch(self):
        if not self._batches:
            return None
        return self._batches.pop(0)

    def frames_read(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    """Records each write_batch() call's size instead of concatenating frames
    -- this is what lets the tests assert peak-batch-size stays bounded
    instead of growing with total video length."""

    instances: list["FakeWriter"] = []

    def __init__(self, path, width, height, fps, **kwargs):
        self.path = path
        self.batch_sizes: list[int] = []
        self.closed = False
        self.close_return = True
        # Simulate the real writer creating a file on disk immediately, so
        # the stage's os.path.exists()/os.replace() partial-preservation
        # logic has something real to operate on.
        Path(path).touch()
        FakeWriter.instances.append(self)

    def write_batch(self, frames: list[Any]) -> None:
        self.batch_sizes.append(len(frames))

    def frames_written(self) -> int:
        return sum(self.batch_sizes)

    def close(self) -> bool:
        self.closed = True
        return self.close_return


class FakeInterpolator:
    """Deterministic factor*(len-1)+1 output, mirroring RIFEInterpolator's
    real contract (the last spec detail: for N input frames, interpolate_video
    returns factor*(N-1)+1 frames -- N-1 gaps, each filled with factor-1 new
    frames, plus the N original frames)."""

    def __init__(self, force_empty: bool = False, **kwargs):
        self.force_empty = force_empty
        self.loaded = False
        self.unloaded = False

    def load_model(self) -> bool:
        self.loaded = True
        return True

    def unload(self) -> None:
        self.unloaded = True

    def interpolate_video(self, frames, factor: int = 2, progress_callback=None):
        if self.force_empty:
            return []
        n = len(frames)
        return [_fake_frame() for _ in range(factor * (n - 1) + 1)]


def _fake_probe_info(frame_count: int, resolution=(64, 64), has_audio: bool = False):
    p = MagicMock()
    p.frame_count = frame_count
    p.resolution = resolution
    p.has_audio = has_audio
    p.duration = frame_count / 30.0
    p.framerate = 30.0
    return p


def _install_common_mocks(
    monkeypatch,
    *,
    reader_batches: list[list[Any]],
    interpolator: FakeInterpolator,
    probe_info,
    mux_returncode: int = 0,
):
    monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: True)
    monkeypatch.setattr(
        "autovideofixer.ai.model_cache.ensure_model_available", lambda name: (True, "")
    )
    monkeypatch.setattr(
        "autovideofixer.ai.wrappers.interpolate.RIFEInterpolator",
        lambda **kwargs: interpolator,
    )
    monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
    monkeypatch.setattr(
        "autovideofixer.ai.frame_pipe.get_frame_reader",
        lambda *a, **kw: FakeReader(reader_batches),
    )
    monkeypatch.setattr("autovideofixer.ai.frame_pipe.get_frame_writer", FakeWriter)

    def fake_run_ffmpeg(args, progress_callback=None, timeout=None):
        if mux_returncode == 0:
            # Last arg is the mux output path -- create it so the stage's
            # os.path.exists(output_path) success check passes.
            Path(args[-1]).touch()
        return MagicMock(returncode=mux_returncode, stderr="simulated mux failure")

    monkeypatch.setattr("autovideofixer.core.stages.interpolate.run_ffmpeg", fake_run_ffmpeg)


@pytest.fixture(autouse=True)
def _reset_fake_writer_instances():
    FakeWriter.instances.clear()
    yield
    FakeWriter.instances.clear()


def _stage(tmp_path) -> InterpolateStage:
    return InterpolateStage(Config(tmp_path / "nonexistent.yaml"))


class TestStreamingNoAccumulation:
    def test_streams_in_multiple_batches_bounded_peak_size(self, tmp_path, monkeypatch):
        """59 -> 3 chunks of (25, 25, 9) at chunk_size=25. Regression guard:
        the old code called frames_to_video()/frames_to_video-equivalent ONCE
        with the full factor*total_frames list; here we assert write_batch()
        is called once per chunk (not once overall) and that no single batch
        approaches the full output size."""
        factor = 2
        total_frames = 59
        batches = [[_fake_frame() for _ in range(25)] for _ in range(2)] + [
            [_fake_frame() for _ in range(9)]
        ]
        interpolator = FakeInterpolator()
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        result = stage._execute_ai(
            input_path,
            output_path,
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.COMPLETED
        assert len(FakeWriter.instances) == 1
        writer = FakeWriter.instances[0]

        # One write_batch() call per input chunk -> streamed, not buffered.
        assert len(writer.batch_sizes) == 3

        total_expected = factor * (total_frames - 1) + 1
        assert sum(writer.batch_sizes) == total_expected

        # Peak single-batch size stays roughly chunk_size*factor (~50), never
        # anywhere near total_frames*factor (~118) -- this is the actual
        # regression guard against re-introducing full accumulation.
        assert max(writer.batch_sizes) <= 25 * factor + 1
        assert max(writer.batch_sizes) < total_expected

    def test_carry_frame_dedup_exact_total_no_gap_no_dup(self, tmp_path, monkeypatch):
        """Total output frames across chunk boundaries must equal exactly
        factor*(total-1)+1 -- no off-by-one from the carry-frame prepend/drop
        logic, in either direction."""
        factor = 3
        total_frames = 77
        chunk_size = 25
        # Build reader batches summing to total_frames, chunk_size each
        # (matching the stage's hardcoded chunk_size).
        remaining = total_frames
        batches = []
        while remaining > 0:
            n = min(chunk_size, remaining)
            batches.append([_fake_frame() for _ in range(n)])
            remaining -= n

        interpolator = FakeInterpolator()
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        result = stage._execute_ai(
            input_path,
            output_path,
            None,
            time.time(),
            target_fps=90.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.COMPLETED
        expected_total = factor * (total_frames - 1) + 1
        assert result.metadata["frames_out"] == expected_total
        writer = FakeWriter.instances[0]
        assert sum(writer.batch_sizes) == expected_total

    def test_frames_out_metadata_matches_frames_written(self, tmp_path, monkeypatch):
        total_frames = 40
        batches = [[_fake_frame() for _ in range(25)], [_fake_frame() for _ in range(15)]]
        interpolator = FakeInterpolator()
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        result = stage._execute_ai(
            input_path,
            output_path,
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.COMPLETED
        writer = FakeWriter.instances[0]
        assert result.metadata["frames_out"] == writer.frames_written()
        assert result.metadata["frames_in"] == total_frames


class TestMkvTempPath:
    def test_temp_path_used_is_mkv(self, tmp_path, monkeypatch):
        total_frames = 10
        batches = [[_fake_frame() for _ in range(10)]]
        interpolator = FakeInterpolator()
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        result = stage._execute_ai(
            input_path,
            output_path,
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.COMPLETED
        writer = FakeWriter.instances[0]
        assert writer.path.endswith(".mkv")


class TestPartialOutputPreservation:
    def test_graceful_mux_failure_renames_partial_and_logs(self, tmp_path, monkeypatch, caplog):
        total_frames = 10
        batches = [[_fake_frame() for _ in range(10)]]
        interpolator = FakeInterpolator()
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
            mux_returncode=1,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        with caplog.at_level(logging.WARNING, logger="autovideofixer.stages.interpolate"):
            result = stage._execute_ai(
                input_path,
                output_path,
                None,
                time.time(),
                target_fps=60.0,
                current_fps=30.0,
            )

        assert result.status == StageStatus.FAILED
        assert "mux" in result.error.lower()

        expected_partial = str(tmp_path / "out_interp_partial.mkv")
        assert Path(expected_partial).exists()
        writer = FakeWriter.instances[0]
        assert not Path(writer.path).exists()  # renamed away, not left/deleted
        assert any("partial interpolated output preserved" in r.message for r in caplog.records)
        assert any(expected_partial in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestEmptyInterpolatorOutput:
    def test_empty_output_fails_with_expected_message(self, tmp_path, monkeypatch):
        total_frames = 10
        batches = [[_fake_frame() for _ in range(10)]]
        interpolator = FakeInterpolator(force_empty=True)
        probe_info = _fake_probe_info(total_frames)
        _install_common_mocks(
            monkeypatch,
            reader_batches=batches,
            interpolator=interpolator,
            probe_info=probe_info,
        )

        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")
        Path(input_path).touch()

        stage = _stage(tmp_path)
        result = stage._execute_ai(
            input_path,
            output_path,
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.FAILED
        assert result.error == "No frames produced by interpolator"
        # No writer should even have been constructed -- interpolate_video()
        # never produced a non-empty chunk to infer output dimensions from.
        assert len(FakeWriter.instances) == 0
