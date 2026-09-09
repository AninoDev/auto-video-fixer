"""Unit tests for RetimeStage (REQUIREMENTS.md § 12.2).

``analyze_cadence()`` and ``run_ffmpeg()``/``probe()`` are mocked throughout
-- no real ffmpeg/GPU needed. Covers should_run()'s gating, SKIP (never
FAIL) on an already-honest input / no video stream / a failed mpdecimate
pass, and the COMPLETED metadata shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autovideofixer.config import Config
from autovideofixer.core.cadence import CadenceAnalysis
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.retime import RetimeStage


def _make_stage(tmp_path, enabled: bool = True) -> RetimeStage:
    config = Config(tmp_path / "nonexistent.yaml")
    config.set(enabled, "stages", "retime", "enabled")
    return RetimeStage(config)


def _padded_analysis(**overrides) -> CadenceAnalysis:
    defaults = dict(
        encoded_fps=60.0,
        total_frames=120,
        unique_frames=48,
        duplicate_ratio=0.6,
        unique_timestamps=[i * (2.0 / 48) for i in range(48)],
        detected_fps=24.0,
        nominal_fps=24.0,
        is_padded=True,
        is_regular=True,
        grid_rate=60.0,
    )
    defaults.update(overrides)
    return CadenceAnalysis(**defaults)


def _honest_analysis(**overrides) -> CadenceAnalysis:
    defaults = dict(
        encoded_fps=30.0,
        total_frames=60,
        unique_frames=59,
        duplicate_ratio=0.0167,
        unique_timestamps=[i * (2.0 / 59) for i in range(59)],
        detected_fps=29.5,
        nominal_fps=29.5,
        is_padded=False,
        is_regular=True,
        grid_rate=30.0,
    )
    defaults.update(overrides)
    return CadenceAnalysis(**defaults)


class TestShouldRun:
    def test_disabled_skips(self, tmp_path):
        stage = _make_stage(tmp_path, enabled=False)
        should_run, reason = stage.should_run({"has_video": True})
        assert should_run is False
        assert reason == "Stage disabled"

    def test_no_video_stream_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({"has_video": False})
        assert should_run is False
        assert reason == "No video stream"

    def test_has_video_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({"has_video": True})
        assert should_run is True
        assert reason is None

    def test_missing_has_video_key_defers_to_execute(self, tmp_path):
        # No positive assertion either way -- should_run() must not block
        # the stage outright just because input_info doesn't mention it.
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({})
        assert should_run is True


class TestExecuteNoOutputPath:
    def test_missing_output_path_fails(self, tmp_path):
        stage = _make_stage(tmp_path)
        result = stage.execute("in.mp4", output_path=None)
        assert result.status == StageStatus.FAILED


class TestExecuteNoVideoStream:
    def test_skips_without_calling_analyze_cadence(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch("autovideofixer.core.stages.retime.analyze_cadence") as mock_analyze:
            result = stage.execute(
                "in.mp4", output_path=str(tmp_path / "out.mkv"), input_info={"has_video": False}
            )
        assert result.status == StageStatus.SKIPPED
        assert result.skipped_reason == "No video stream"
        mock_analyze.assert_not_called()


class TestExecuteNotPadded:
    def test_skips_when_not_padded(self, tmp_path):
        stage = _make_stage(tmp_path)
        with (
            patch(
                "autovideofixer.core.stages.retime.analyze_cadence",
                return_value=_honest_analysis(),
            ),
            patch("autovideofixer.core.stages.retime.run_ffmpeg") as mock_run_ffmpeg,
        ):
            result = stage.execute("in.mp4", output_path=str(tmp_path / "out.mkv"))

        assert result.status == StageStatus.SKIPPED
        assert "already honest" in result.skipped_reason
        assert result.metadata["method"] == "mpdecimate"
        mock_run_ffmpeg.assert_not_called()


class TestExecutePadded:
    def test_completed_with_frames_removed_metadata(self, tmp_path):
        stage = _make_stage(tmp_path)
        analysis = _padded_analysis()
        fake_result = MagicMock(returncode=0, stderr="frame=   48 fps=30 ...\n")
        fake_probe = MagicMock()
        fake_probe.frame_count = 120

        with (
            patch("autovideofixer.core.stages.retime.analyze_cadence", return_value=analysis),
            patch(
                "autovideofixer.core.stages.retime.run_ffmpeg", return_value=fake_result
            ) as mock_run_ffmpeg,
            patch("autovideofixer.core.stages.retime.probe", return_value=fake_probe),
        ):
            output_path = str(tmp_path / "out.mkv")
            result = stage.execute("in.mp4", output_path=output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.output_path == output_path
        assert result.metadata["method"] == "mpdecimate"
        assert result.metadata["frames_before"] == 120
        assert result.metadata["frames_after"] == 48
        assert result.metadata["frames_removed"] == 72
        assert result.metadata["nominal_fps"] == 24.0

        # -fps_mode vfr is load-bearing (§ 12.2) -- verify it's actually
        # passed to the real mpdecimate pass.
        args = mock_run_ffmpeg.call_args.args[0]
        assert "-fps_mode" in args
        assert args[args.index("-fps_mode") + 1] == "vfr"

    def test_ffmpeg_failure_degrades_to_skip_not_fail(self, tmp_path):
        stage = _make_stage(tmp_path)
        analysis = _padded_analysis()
        fake_result = MagicMock(returncode=1, stderr="some ffmpeg error")

        with (
            patch("autovideofixer.core.stages.retime.analyze_cadence", return_value=analysis),
            patch("autovideofixer.core.stages.retime.run_ffmpeg", return_value=fake_result),
        ):
            result = stage.execute("in.mp4", output_path=str(tmp_path / "out.mkv"))

        assert result.status == StageStatus.SKIPPED
        assert "retime pass failed" in result.skipped_reason
