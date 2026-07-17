"""Tests for parallel-chunked traditional (minterpolate) frame interpolation.

`_resolve_chunk_count` is pure logic (mocked probe, no ffmpeg) -- plain unit
tests. The actual chunked-vs-serial comparison needs real ffmpeg and is
marked integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.interpolate import InterpolateStage


def _stage(tmp_path) -> InterpolateStage:
    return InterpolateStage(Config(tmp_path / "nonexistent.yaml"))


def _fake_probe(duration: float, frame_count: int, framerate: float = 30.0):
    p = MagicMock()
    p.duration = duration
    p.frame_count = frame_count
    p.framerate = framerate
    p.has_audio = False
    return p


class TestResolveChunkCount:
    def test_explicit_serial_override(self, tmp_path):
        stage = _stage(tmp_path)
        with patch(
            "autovideofixer.core.stages.interpolate.probe",
            return_value=_fake_probe(60.0, 1800),
        ):
            assert (
                stage._resolve_chunk_count(
                    "x.mp4", chunks_cfg=1, min_chunk_dur=5.0, current_fps=30.0
                )
                == 1
            )

    def test_short_clip_stays_serial(self, tmp_path):
        stage = _stage(tmp_path)
        with patch(
            "autovideofixer.core.stages.interpolate.probe",
            return_value=_fake_probe(3.0, 90),
        ):
            n = stage._resolve_chunk_count(
                "x.mp4", chunks_cfg=0, min_chunk_dur=5.0, current_fps=30.0
            )
        assert n == 1

    def test_long_clip_auto_chunks(self, tmp_path):
        stage = _stage(tmp_path)
        with (
            patch(
                "autovideofixer.core.stages.interpolate.probe",
                return_value=_fake_probe(120.0, 3600),
            ),
            patch("os.cpu_count", return_value=8),
        ):
            n = stage._resolve_chunk_count(
                "x.mp4", chunks_cfg=0, min_chunk_dur=5.0, current_fps=30.0
            )
        assert n > 1
        assert n <= 8

    def test_explicit_chunk_count_respected(self, tmp_path):
        stage = _stage(tmp_path)
        with patch(
            "autovideofixer.core.stages.interpolate.probe",
            return_value=_fake_probe(120.0, 3600),
        ):
            n = stage._resolve_chunk_count(
                "x.mp4", chunks_cfg=4, min_chunk_dur=5.0, current_fps=30.0
            )
        assert n == 4

    def test_too_few_frames_for_requested_chunks_falls_back_serial(self, tmp_path):
        stage = _stage(tmp_path)
        with patch(
            "autovideofixer.core.stages.interpolate.probe",
            # 20s clip, min_chunk_dur=5 allows 4 chunks by duration, but only
            # 6 total frames -- not enough for 4 chunks (need >= 2 each).
            return_value=_fake_probe(20.0, 6),
        ):
            n = stage._resolve_chunk_count(
                "x.mp4", chunks_cfg=4, min_chunk_dur=5.0, current_fps=30.0
            )
        assert n == 1

    def test_probe_failure_falls_back_serial(self, tmp_path):
        stage = _stage(tmp_path)
        with patch(
            "autovideofixer.core.stages.interpolate.probe", side_effect=RuntimeError("boom")
        ):
            n = stage._resolve_chunk_count(
                "x.mp4", chunks_cfg=4, min_chunk_dur=5.0, current_fps=30.0
            )
        assert n == 1


class TestParallelChunksOverrideForwarded:
    """Fix 1 (the OOM-incident regression test): execute() must forward an
    explicit parallel_chunks= kwarg through to _execute_traditional(), not
    silently swallow it into **kwargs. Before the fix, scene mode's
    per-scene chunk budget (_scene_worker_budget) was lost and
    _execute_traditional fell back to stages.interpolate.parallel_chunks
    (0 = auto = min(cpu, 8)) PER SCENE -- 6 scenes x up to 8 chunks each
    peaked at ~12 concurrent 4K minterpolate ffmpeg processes and OOM-killed
    a 56 GiB container.
    """

    def test_execute_forwards_parallel_chunks_override(self, tmp_path):
        stage = _stage(tmp_path)
        video_info = {"framerate": 30.0, "resolution": (320, 240), "duration": 120.0}
        # duration/frame_count long enough that auto-chunking (parallel_chunks=0,
        # the config default) would pick more than 1 chunk.
        probe_result = _fake_probe(120.0, 3600, framerate=30.0)

        fake_ffmpeg_result = MagicMock(returncode=0, stderr="")
        with (
            patch("autovideofixer.core.ffmpeg_utils.get_video_info", return_value=video_info),
            patch("autovideofixer.core.stages.interpolate.probe", return_value=probe_result),
            patch(
                "autovideofixer.core.stages.interpolate.run_ffmpeg",
                return_value=fake_ffmpeg_result,
            ),
            patch("os.cpu_count", return_value=8),
            patch(
                "autovideofixer.core.stages.interpolate.InterpolateStage."
                "_execute_traditional_parallel"
            ) as mock_parallel,
        ):
            result = stage.execute(
                "in.mp4",
                str(tmp_path / "out.mp4"),
                target_fps=60.0,
                method="traditional",
                parallel_chunks=1,
            )

        # The parallel executor must never be constructed -- parallel_chunks=1
        # forces the single-pass path regardless of what auto-chunking would
        # have picked for this duration/cpu count.
        mock_parallel.assert_not_called()
        assert result.status.value == "completed"
        assert result.metadata["parallel_chunks"] == 1

    def test_scenes_propagates_override_into_resolved_chunk_count(self, tmp_path):
        """End-to-end at the scenes.py level: interpolate_scene_clip's
        parallel_chunks_override must reach InterpolateStage.execute()'s
        resolved chunk count."""
        from autovideofixer.config import Config as _Config
        from autovideofixer.core import scenes as scenes_mod

        config = _Config(tmp_path / "nonexistent.yaml")
        config.set(0, "stages", "interpolate", "parallel_chunks")  # auto

        captured: dict[str, Any] = {}

        class _FakeStage:
            def __init__(self, _config):
                pass

            def should_run(self, _input_info):
                return True, None

            def resolve_ai_method(self, explicit_method, auto_default):
                return (explicit_method or auto_default), "auto default"

            def execute(self, _input_path, _output_path, input_info=None, **kwargs):
                captured.update(kwargs)
                from autovideofixer.core.stages.base import StageResult, StageStatus

                return StageResult(
                    status=StageStatus.COMPLETED,
                    metadata={"parallel_chunks": kwargs.get("parallel_chunks")},
                )

        fake_probe_result = _fake_probe(30.0, 900, framerate=30.0)
        with (
            patch.object(scenes_mod, "InterpolateStage", _FakeStage),
            patch.object(scenes_mod, "probe", return_value=fake_probe_result),
        ):
            ok, ran = scenes_mod.interpolate_scene_clip(
                "clip.mp4",
                str(tmp_path / "out.mp4"),
                config,
                target_fps=60.0,
                parallel_chunks_override=3,
            )

        assert ok is True
        assert ran is True
        assert captured["parallel_chunks"] == 3
        assert captured["method"] == "traditional"


class TestParallelVsSerialInterpolation:
    @pytest.mark.integration
    def test_frame_counts_close_and_no_crash(self, tmp_path):
        """Serial and parallel-chunked interpolation of the same clip should
        produce very close (not necessarily bit-identical -- see
        _execute_traditional_parallel's docstring on minterpolate's per-chunk
        duration-based retiming) total frame counts, and both must complete
        successfully."""
        import subprocess

        from autovideofixer.core.ffmpeg_utils import probe

        video = tmp_path / "moving.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30",
                "-t",
                "4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            capture_output=True,
            check=True,
        )

        stage = _stage(tmp_path)
        serial_out = tmp_path / "serial.mp4"
        parallel_out = tmp_path / "parallel.mp4"

        r1 = stage._execute_traditional_single(
            str(video), str(serial_out), None, 0.0, target=60.0, factor=2
        )
        assert r1.status.value == "completed"

        r2 = stage._execute_traditional_parallel(
            str(video),
            str(parallel_out),
            None,
            0.0,
            target=60.0,
            factor=2,
            n_chunks=2,
            current_fps=30.0,
        )
        assert r2.status.value == "completed"

        serial_frames = probe(str(serial_out)).frame_count
        parallel_frames = probe(str(parallel_out)).frame_count
        assert serial_frames > 0
        assert parallel_frames > 0
        # Not bit-exact (see docstring), but should be within a small
        # tolerance -- not off by a large fraction (which would indicate a
        # real dropped/duplicated segment rather than per-chunk rounding).
        assert abs(serial_frames - parallel_frames) <= max(4, int(serial_frames * 0.05))

    @pytest.mark.integration
    def test_parallel_dispatches_via_execute_traditional(self, tmp_path):
        """The public _execute_traditional entry point picks the parallel path
        when parallel_chunks resolves to >1 and reports it in metadata."""
        import subprocess

        video = tmp_path / "moving.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=30",
                "-t",
                "4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            capture_output=True,
            check=True,
        )
        config = Config(tmp_path / "nonexistent.yaml")
        # The 4s test clip is shorter than the default 5.0s
        # min_chunk_duration_sec, which would otherwise force the serial path
        # regardless of the parallel_chunks override -- lower it so a 4s clip
        # is still eligible for 2 chunks.
        config.set(1.0, "stages", "interpolate", "min_chunk_duration_sec")
        stage = InterpolateStage(config)
        out = tmp_path / "out.mp4"
        result = stage._execute_traditional(
            str(video),
            str(out),
            None,
            0.0,
            target_fps=60.0,
            current_fps=30.0,
            parallel_chunks=2,
        )
        assert result.status.value == "completed"
        assert result.metadata["parallel_chunks"] == 2
