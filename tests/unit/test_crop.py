"""Tests for the auto-crop stage (core/stages/crop.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus, get_stage
from autovideofixer.core.stages.crop import CropStage, _detect_crop

_CROPDETECT_STDERR = (
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:639 y1:60 y2:419 w:640 h:360 x:0 y:60 "
    "pts:1 t:0.04 crop=640:360:0:60\n"
    "[Parsed_cropdetect_0 @ 0x1] x1:0 x2:639 y1:60 y2:419 w:640 h:360 x:0 y:60 "
    "pts:25 t:1.00 crop=640:360:0:60\n"
)


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

    @patch("autovideofixer.core.stages.crop._detect_crop")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_no_cropdetect_result_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = None
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "cropdetect produced no result" in result.skipped_reason

    @patch("autovideofixer.core.stages.crop._detect_crop")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_savings_below_threshold_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (640, 478, 0, 1)
        stage = CropStage(_config(tmp_path, enabled=True, min_crop_px=8))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "no meaningful border" in result.skipped_reason
        assert result.metadata["detected_crop"] == "640:478:0:1"

    @patch("autovideofixer.core.stages.crop._detect_crop")
    @patch("autovideofixer.core.stages.crop.probe")
    def test_invalid_crop_window_skips(self, mock_probe, mock_detect, tmp_path):
        mock_probe.return_value = MagicMock(resolution=(640, 480), has_audio=True, duration=10.0)
        mock_detect.return_value = (999, 999, 0, 0)  # larger than the input
        stage = CropStage(_config(tmp_path, enabled=True))
        result = stage.execute("in.mp4", str(tmp_path / "out.mp4"))
        assert result.status == StageStatus.SKIPPED
        assert "invalid crop window" in result.skipped_reason

    @patch("autovideofixer.core.stages.crop.run_ffmpeg")
    @patch("autovideofixer.core.stages.crop._detect_crop")
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
    @patch("autovideofixer.core.stages.crop._detect_crop")
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
    @patch("autovideofixer.core.stages.crop._detect_crop")
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
    @patch("autovideofixer.core.stages.crop._detect_crop")
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
    @patch("autovideofixer.core.stages.crop._detect_crop")
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
        mock_check.assert_not_called()
