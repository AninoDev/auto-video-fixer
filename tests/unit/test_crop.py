"""Tests for the auto-crop stage (core/stages/crop.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus, get_stage
from autovideofixer.core.stages.crop import (
    CropFrame,
    CropStage,
    _aggregate_runs,
    _compute_max_outliers,
    _cropdetect_filter,
    _detect_crop,
    _detect_crop_full,
    _parse_crop_frames,
    _round_up_to_multiple,
    aggregate_crop_windows,
)

_CROPDETECT_STDERR = (
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:639 y1:60 y2:419 w:640 h:360 x:0 y:60 "
    "pts:1 t:0.04 crop=640:360:0:60\n"
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:639 y1:60 y2:419 w:640 h:360 x:0 y:60 "
    "pts:25 t:1.00 crop=640:360:0:60\n"
)

_CROPDETECT_RESET1_STDERR = (
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:1919 y1:140 y2:939 w:1920 h:800 x:0 y:140 "
    "pts:0 t:0.00 crop=1920:800:0:140\n"
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:1919 y1:140 y2:939 w:1920 h:800 x:0 y:140 "
    "pts:6 t:0.20 crop=1920:800:0:140\n"
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:1919 y1:140 y2:939 w:1920 h:800 x:0 y:140 "
    "pts:12 t:0.40 crop=1920:800:0:140\n"
)


def _make_frame(t: float, w: int, h: int, x: int, y: int) -> CropFrame:
    return CropFrame(t=t, w=w, h=h, x=x, y=y)


def _config(tmp_path, **crop_overrides) -> Config:
    config = Config(tmp_path / "nonexistent.yaml")
    for key, value in crop_overrides.items():
        config.set(value, "stages", "crop", key)
    return config


class TestCropConfigDefaults:
    def test_defaults_present(self, tmp_path):
        config = _config(tmp_path)
        crop_cfg = config.get("stages", "crop")
        assert crop_cfg == {
            "enabled": False,
            "limit": 24,
            "round": 2,
            "min_crop_px": 8,
            "analyze_duration_sec": 0,
            "max_outlier_ratio": 0.2,
            "transition_max_run_sec": 2.0,
            "transition_window_sec": 4.0,
            "transition_tolerance_px": 16,
            "vlm_check": False,
            "vlm_policy": "warn",
        }


class TestCropStageRegistration:
    def test_registered_under_crop(self):
        assert get_stage("crop") is CropStage

    def test_priority_between_stabilize_and_deblock(self):
        from autovideofixer.core.stages.deblock import DeblockStage
        from autovideofixer.core.stages.stabilize import StabilizeStage

        assert StabilizeStage.priority < CropStage.priority < DeblockStage.priority

    def test_pipeline_order_places_crop_after_stabilize_before_denoise(self, tmp_path):
        # deblock now runs BEFORE stabilize (see config.py's
        # pipeline.default_order / CHANGELOG for the rationale) -- crop still
        # sits right after stabilize.
        from autovideofixer.core.pipeline import Pipeline

        pipeline = Pipeline(_config(tmp_path))
        ordered = pipeline.optimize_stage_order(
            ["encode", "deblock", "crop", "detect", "stabilize"]
        )
        assert ordered == ["detect", "deblock", "stabilize", "crop", "encode"]

    def test_not_enabled_by_default_in_auto_determine(self, tmp_path):
        from autovideofixer.core.pipeline import Job, Pipeline

        pipeline = Pipeline(_config(tmp_path))
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        job = Job(input_path=str(video), output_path=str(tmp_path / "out.mp4"))
        with patch(
            "autovideofixer.core.pipeline.get_video_info",
            return_value={"resolution": (640, 480), "framerate": 30, "is_hdr": False},
        ):
            stages = pipeline.auto_determine_stages(job)
        assert "crop" not in stages

    def test_enabled_via_config_included_in_auto_determine(self, tmp_path):
        from autovideofixer.core.pipeline import Job, Pipeline

        pipeline = Pipeline(_config(tmp_path, enabled=True))
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        job = Job(input_path=str(video), output_path=str(tmp_path / "out.mp4"))
        with patch(
            "autovideofixer.core.pipeline.get_video_info",
            return_value={"resolution": (640, 480), "framerate": 30, "is_hdr": False},
        ):
            stages = pipeline.auto_determine_stages(job)
        assert "crop" in stages


class TestDetectCropParsing:
    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_takes_last_crop_line(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=_CROPDETECT_STDERR)
        result = _detect_crop("in.mp4", limit=24, round_=2, analyze_duration_sec=0)
        assert result == (640, 360, 0, 60)

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_no_crop_lines_returns_none(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="nothing useful here\n")
        assert _detect_crop("in.mp4", limit=24, round_=2, analyze_duration_sec=0) is None

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_analyze_duration_sec_adds_dash_t(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=_CROPDETECT_STDERR)
        _detect_crop("in.mp4", limit=24, round_=2, analyze_duration_sec=15)
        args = mock_run_ffmpeg.call_args[0][0]
        assert "-t" in args
        assert args[args.index("-t") + 1] == "15"


class TestCropShouldRun:
    def test_disabled(self, tmp_path):
        stage = CropStage(_config(tmp_path, enabled=False))
        should, reason = stage.should_run({"filepath": "x.mp4", "resolution": (640, 480)})
        assert should is False
        assert reason == "Stage disabled"

    def test_missing_filepath_passes_through(self, tmp_path):
        stage = CropStage(_config(tmp_path, enabled=True))
        should, reason = stage.should_run({"filepath": "", "resolution": (640, 480)})
        assert should is True
        assert reason is None

    def test_missing_resolution_passes_through(self, tmp_path):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        stage = CropStage(_config(tmp_path, enabled=True))
        should, reason = stage.should_run({"filepath": str(video), "resolution": (0, 0)})
        assert should is True
        assert reason is None

    @patch("autovideofixer.core.stages.crop._detect_crop")
    def test_quick_sample_finds_nothing_meaningful_skips(self, mock_detect, tmp_path):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        mock_detect.return_value = (640, 478, 0, 1)  # 2px total savings
        stage = CropStage(_config(tmp_path, enabled=True, min_crop_px=8))
        should, reason = stage.should_run({"filepath": str(video), "resolution": (640, 480)})
        assert should is False
        assert "Quick cropdetect sample" in reason

    @patch("autovideofixer.core.stages.crop._detect_crop")
    def test_quick_sample_finds_meaningful_crop_proceeds(self, mock_detect, tmp_path):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        mock_detect.return_value = (640, 360, 0, 60)
        stage = CropStage(_config(tmp_path, enabled=True, min_crop_px=8))
        should, reason = stage.should_run({"filepath": str(video), "resolution": (640, 480)})
        assert should is True
        assert reason is None

    @patch("autovideofixer.core.stages.crop._detect_crop")
    def test_inconclusive_quick_sample_passes_through(self, mock_detect, tmp_path):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        mock_detect.return_value = None
        stage = CropStage(_config(tmp_path, enabled=True))
        should, reason = stage.should_run({"filepath": str(video), "resolution": (640, 480)})
        assert should is True


class TestCropExecute:
    def test_no_output_path_fails(self, tmp_path):
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", None)
        assert result.status == StageStatus.FAILED
        assert "output_path" in result.error

    @patch("autovideofixer.core.stages.crop.probe")
    def test_probe_failure_fails(self, mock_probe, tmp_path):
        mock_probe.side_effect = RuntimeError("boom")
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.FAILED
        assert "probe failed" in result.error

    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_no_cropdetect_result_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = None
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "cropdetect produced no result" in result.skipped_reason

    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_savings_below_threshold_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 478, 0, 1)
        stage = CropStage(_config(tmp_path, enabled=True, min_crop_px=8))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "no meaningful border" in result.skipped_reason
        assert result.metadata["detected_crop"] == "640:478:0:1"

    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_invalid_crop_window_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (999, 999, 0, 0)  # larger than the input
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "invalid crop window" in result.skipped_reason

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_successful_crop_completes(self, mock_probe, mock_detect, mock_run_ffmpeg, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 360, 0, 60)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
        stage = CropStage(_config(tmp_path, enabled=True, min_crop_px=8))
        out = str(tmp_path / "out.mp4")
        result = stage.execute("in.mp4", out)
        assert result.status == StageStatus.COMPLETED
        assert result.output_path == out
        assert result.metadata["detected_crop"] == "640:360:0:60"
        assert result.metadata["cropped_resolution"] == "640x360"
        args = mock_run_ffmpeg.call_args[0][0]
        assert "crop=640:360:0:60" in args
        assert "-c:a" in args and "copy" in args

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_ffmpeg_encode_failure(self, mock_probe, mock_detect, mock_run_ffmpeg, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 360, 0, 60)
        mock_run_ffmpeg.return_value = MagicMock(returncode=1, stderr="ffmpeg exploded")
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.FAILED
        assert "ffmpeg exploded" in result.error

    @patch.object(CropStage, "_run_vlm_check")
    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_vlm_flag_with_skip_policy_skips_crop(
        self, mock_probe, mock_detect, mock_run_ffmpeg, mock_vlm_check, tmp_path
    ):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 360, 0, 60)
        mock_vlm_check.return_value = {
            "checked": True,
            "content_outside": True,
            "reason": "logo watermark in the letterbox area",
            "failed": False,
        }
        stage = CropStage(_config(tmp_path, enabled=True, vlm_check=True, vlm_policy="skip"))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "VLM flagged" in result.skipped_reason
        mock_run_ffmpeg.assert_not_called()

    @patch.object(CropStage, "_run_vlm_check")
    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_vlm_flag_with_warn_policy_still_crops(
        self, mock_probe, mock_detect, mock_run_ffmpeg, mock_vlm_check, tmp_path
    ):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 360, 0, 60)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
        mock_vlm_check.return_value = {
            "checked": True,
            "content_outside": True,
            "reason": "logo watermark in the letterbox area",
            "failed": False,
        }
        stage = CropStage(_config(tmp_path, enabled=True, vlm_check=True, vlm_policy="warn"))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["vlm_check"]["content_outside"] is True
        mock_run_ffmpeg.assert_called_once()

    @patch.object(CropStage, "_run_vlm_check")
    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop_full")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_vlm_no_content_outside_crops_normally(
        self, mock_probe, mock_detect, mock_run_ffmpeg, mock_vlm_check, tmp_path
    ):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 360, 0, 60)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
        mock_vlm_check.return_value = {
            "checked": True,
            "content_outside": False,
            "reason": "only black letterbox bars",
            "failed": False,
        }
        stage = CropStage(_config(tmp_path, enabled=True, vlm_check=True, vlm_policy="skip"))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.COMPLETED
        mock_run_ffmpeg.assert_called_once()

    @patch("autovideofixer.core.analysis.run_crop_vlm_check")
    def test_vlm_check_skipped_when_analysis_vlm_disabled(self, mock_check, tmp_path):
        stage = CropStage(_config(tmp_path, enabled=True, vlm_check=True))
        probe_info = MagicMock(duration=10.0)
        result = stage._run_vlm_check("in.mp4", probe_info, 640, 360, 0, 60)
        assert result["checked"] is False
        assert result["content_outside"] is False
        assert "analysis.vlm.enabled" in result["reason"]


class TestComputeMaxOutliers:
    def test_zero_ratio_disables(self):
        assert _compute_max_outliers(0.0, 1920, 1080) == 0

    def test_ratio_uses_min_dimension(self):
        # min(1920, 1080) == 1080; round(0.2 * 1080) == 216.
        assert _compute_max_outliers(0.2, 1920, 1080) == 216
        # Portrait input: min(608, 1080) == 608; round(0.2 * 608) == 122.
        assert _compute_max_outliers(0.2, 608, 1080) == 122

    def test_ratio_clamped_to_valid_range(self):
        # Negative and > 0.5 ratios are clamped defensively rather than
        # producing a nonsensical/negative max_outliers.
        assert _compute_max_outliers(-1.0, 1920, 1080) == 0
        assert _compute_max_outliers(10.0, 1920, 1080) == _compute_max_outliers(0.5, 1920, 1080)


class TestCropdetectFilterString:
    def test_max_outliers_zero_omitted(self):
        filt = _cropdetect_filter(limit=24, round_=2, reset=1, max_outliers=0)
        assert filt == "cropdetect=limit=24:round=2:reset=1"
        assert "max_outliers" not in filt

    def test_max_outliers_appended_when_positive(self):
        filt = _cropdetect_filter(limit=24, round_=2, reset=1, max_outliers=216)
        assert filt == "cropdetect=limit=24:round=2:reset=1:max_outliers=216"

    def test_reset_value_reflected(self):
        assert "reset=0" in _cropdetect_filter(limit=24, round_=2, reset=0)


class TestRoundUpToMultiple:
    def test_already_aligned_unchanged(self):
        assert _round_up_to_multiple(800, 2) == 800

    def test_rounds_up_never_down(self):
        assert _round_up_to_multiple(801, 2) == 802

    def test_multiple_of_one_is_noop(self):
        assert _round_up_to_multiple(801, 1) == 801
        assert _round_up_to_multiple(801, 0) == 801


class TestAggregateCropWindows:
    """aggregate_crop_windows() is a pure function -- test it hard."""

    def _agg(self, frames, tolerance_px=16, transition_max_run_sec=2.0, transition_window_sec=4.0):
        return aggregate_crop_windows(
            frames,
            tolerance_px=tolerance_px,
            transition_max_run_sec=transition_max_run_sec,
            transition_window_sec=transition_window_sec,
        )

    def test_uniform_windows_returns_that_window(self):
        frames = [_make_frame(t, 1920, 800, 0, 140) for t in (0.0, 1.0, 2.0, 3.0)]
        assert self._agg(frames) == (1920, 800, 0, 140)

    def test_isolated_full_frame_burst_excluded_stable_window_returned(self):
        # THE transition case: a 0.5s full-frame burst mid-video, surrounded
        # by an otherwise-stable letterboxed window.
        frames = (
            [_make_frame(t, 1920, 800, 0, 140) for t in (0.0, 1.0, 2.0)]
            + [_make_frame(t, 1920, 1080, 0, 0) for t in (2.5, 2.7)]  # 0.2s burst
            + [_make_frame(t, 1920, 800, 0, 140) for t in (3.5, 4.5, 5.5)]
        )
        assert self._agg(frames) == (1920, 800, 0, 140)

    def test_recurring_full_frame_window_kept_as_union(self):
        # Same burst window, but it ALSO occurs for a long stretch elsewhere
        # -- not a transition, must be kept (union of both).
        frames = (
            [_make_frame(t, 1920, 800, 0, 140) for t in (0.0, 1.0, 2.0)]
            + [_make_frame(t, 1920, 1080, 0, 0) for t in (2.5, 2.7)]  # 0.2s burst
            + [_make_frame(t, 1920, 800, 0, 140) for t in (3.5, 4.5)]
            # Long recurrence of the "burst" window far away in time --
            # still within transition_window_sec of nothing near it, but the
            # recurrence itself lasts >2s so it's real content, and it also
            # makes the original burst non-isolated relative to it if close
            # enough; here we place it right after the stable stretch to
            # keep it within transition_window_sec of the burst too.
            + [_make_frame(t, 1920, 1080, 0, 0) for t in (5.0, 6.0, 7.0, 8.0, 15.0)]
        )
        result = self._agg(frames)
        # Union must include the full-frame window's extent.
        assert result == (1920, 1080, 0, 0)

    def test_moving_logo_two_stable_windows_returns_union(self):
        # Two stable windows alternating in different spots over time (a
        # slowly moving overlay) -- union of both, not either alone.
        left = [_make_frame(t, 1900, 1080, 0, 0) for t in range(0, 30)]
        right = [_make_frame(t, 1900, 1080, 20, 0) for t in range(30, 60)]
        result = self._agg(left + right, transition_max_run_sec=2.0, transition_window_sec=4.0)
        # union: x spans 0..1920 (min x=0, max right = 20+1900=1920)
        assert result == (1920, 1080, 0, 0)

    def test_deviant_run_longer_than_max_run_sec_kept_even_if_isolated(self):
        frames = [_make_frame(t, 1920, 800, 0, 140) for t in (0.0, 1.0)] + [
            _make_frame(t, 1920, 1080, 0, 0) for t in (10.0, 11.0, 12.0, 13.0)
        ]  # 3s run, isolated, but longer than transition_max_run_sec=2.0
        result = self._agg(frames, transition_max_run_sec=2.0, transition_window_sec=4.0)
        assert result == (1920, 1080, 0, 0)

    def test_similar_within_tolerance_does_not_fragment_run(self):
        frames = [
            _make_frame(0.0, 1920, 800, 0, 140),
            _make_frame(1.0, 1918, 802, 1, 139),  # within tolerance_px=16 of anchor
            _make_frame(2.0, 1920, 800, 0, 140),
        ]
        result = self._agg(frames, tolerance_px=16)
        # A single run -> its union (small jitter within tolerance).
        assert result is not None
        w, h, x, y = result
        assert (w, h, x, y) != (0, 0, 0, 0)

    def test_empty_list_returns_none(self):
        assert self._agg([]) is None

    def test_all_transition_pathological_falls_back_to_union(self):
        # Every run isolated + short: nothing survives transition exclusion
        # -- must fall back to the union of everything rather than None.
        frames = [
            _make_frame(0.0, 1920, 1080, 0, 0),
            _make_frame(0.2, 1920, 800, 0, 140),
            _make_frame(10.0, 1920, 600, 0, 240),
        ]
        result = self._agg(
            frames, tolerance_px=1, transition_max_run_sec=0.5, transition_window_sec=0.1
        )
        assert result == (1920, 1080, 0, 0)


class TestAggregateRunsDiagnostics:
    """_aggregate_runs() (the internal helper backing both
    aggregate_crop_windows() and _detect_crop_full()'s INFO logging) exposes
    the runs/excluded_runs breakdown that aggregate_crop_windows() itself
    intentionally doesn't return."""

    def test_excluded_runs_reported(self):
        frames = (
            [_make_frame(t, 1920, 800, 0, 140) for t in (0.0, 1.0, 2.0)]
            + [_make_frame(t, 1920, 1080, 0, 0) for t in (2.5, 2.7)]
            + [_make_frame(t, 1920, 800, 0, 140) for t in (3.5, 4.5, 5.5)]
        )
        agg = _aggregate_runs(
            frames, tolerance_px=16, transition_max_run_sec=2.0, transition_window_sec=4.0
        )
        assert agg.window == (1920, 800, 0, 140)
        assert len(agg.excluded_runs) == 1
        assert agg.excluded_runs[0].start_t == 2.5
        assert agg.fallback_triggered is False

    def test_fallback_flag_set_when_all_transitions(self):
        frames = [
            _make_frame(0.0, 1920, 1080, 0, 0),
            _make_frame(0.2, 1920, 800, 0, 140),
            _make_frame(10.0, 1920, 600, 0, 240),
        ]
        agg = _aggregate_runs(
            frames, tolerance_px=1, transition_max_run_sec=0.5, transition_window_sec=0.1
        )
        assert agg.fallback_triggered is True
        assert agg.excluded_runs == []


class TestParseCropFrames:
    def test_parses_all_frames_with_timestamps(self):
        frames = _parse_crop_frames(_CROPDETECT_RESET1_STDERR)
        assert len(frames) == 3
        assert frames[0] == CropFrame(t=0.0, w=1920, h=800, x=0, y=140)
        assert frames[-1] == CropFrame(t=0.4, w=1920, h=800, x=0, y=140)

    def test_no_matching_lines_returns_empty_list(self):
        assert _parse_crop_frames("nothing useful here\n") == []

    def test_ignores_lines_missing_timestamp_or_crop(self):
        stderr = "some unrelated ffmpeg log line\ncrop=640:360:0:60 but no t field\n"
        assert _parse_crop_frames(stderr) == []


class TestDetectCropFull:
    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_aggregates_reset1_stderr_into_single_window(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=_CROPDETECT_RESET1_STDERR)
        result = _detect_crop_full(
            "in.mp4",
            limit=24,
            round_=2,
            analyze_duration_sec=0,
            max_outlier_ratio=0.2,
            transition_max_run_sec=2.0,
            transition_window_sec=4.0,
            transition_tolerance_px=16,
            orig_width=1920,
            orig_height=1080,
            timeout=60,
        )
        assert result == (1920, 800, 0, 140)

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_uses_reset1_and_max_outliers_in_filter(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=_CROPDETECT_RESET1_STDERR)
        _detect_crop_full(
            "in.mp4",
            limit=24,
            round_=2,
            analyze_duration_sec=0,
            max_outlier_ratio=0.2,
            transition_max_run_sec=2.0,
            transition_window_sec=4.0,
            transition_tolerance_px=16,
            orig_width=1920,
            orig_height=1080,
            timeout=60,
        )
        args = mock_run_ffmpeg.call_args[0][0]
        vf = args[args.index("-vf") + 1]
        assert "reset=1" in vf
        assert "max_outliers=216" in vf

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_no_frames_returns_none(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="nothing useful\n")
        result = _detect_crop_full(
            "in.mp4",
            limit=24,
            round_=2,
            analyze_duration_sec=0,
            max_outlier_ratio=0.2,
            transition_max_run_sec=2.0,
            transition_window_sec=4.0,
            transition_tolerance_px=16,
            orig_width=1920,
            orig_height=1080,
            timeout=60,
        )
        assert result is None

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_analyze_duration_sec_adds_dash_t(self, mock_run_ffmpeg):
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=_CROPDETECT_RESET1_STDERR)
        _detect_crop_full(
            "in.mp4",
            limit=24,
            round_=2,
            analyze_duration_sec=15,
            max_outlier_ratio=0.2,
            transition_max_run_sec=2.0,
            transition_window_sec=4.0,
            transition_tolerance_px=16,
            orig_width=1920,
            orig_height=1080,
            timeout=60,
        )
        args = mock_run_ffmpeg.call_args[0][0]
        assert "-t" in args
        assert args[args.index("-t") + 1] == "15"

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    def test_union_rounded_up_never_down(self, mock_run_ffmpeg):
        # Two frames whose union produces an odd width (799) -- must round UP
        # to 800 (round_=2), never down to 798.
        stderr = (
            "[Parsed_cropdetect_0 @ 0x1] pts:0 t:0.00 crop=1920:800:0:140\n"
            "[Parsed_cropdetect_0 @ 0x1] pts:1 t:0.20 crop=1919:799:1:141\n"
        )
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr=stderr)
        w, h, x, y = _detect_crop_full(
            "in.mp4",
            limit=24,
            round_=2,
            analyze_duration_sec=0,
            max_outlier_ratio=0.0,
            transition_max_run_sec=2.0,
            transition_window_sec=4.0,
            transition_tolerance_px=16,
            orig_width=1920,
            orig_height=1080,
            timeout=60,
        )
        assert w % 2 == 0
        assert h % 2 == 0
        # min_x=0, min_y=140; max_right=max(1920, 1919+1)=1920; max_bottom=max(940,799+141=940)=940
        assert w >= 1920
        assert h >= 800
