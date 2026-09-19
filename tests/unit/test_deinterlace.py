"""Unit tests for DeinterlaceStage (REQUIREMENTS.md § 20).

``_run_idet_analysis()``/``run_ffmpeg()`` are mocked throughout -- no real
ffmpeg/GPU needed. The pure idet-parsing/threshold helpers
(``_parse_idet_output``/``_decide_deinterlace``) and the filter-string
builder (``_build_deinterlace_filter``) are tested directly, with no mocking
at all, per REQUIREMENTS.md § 20.2's "pure, unit-testable helper" mandate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.deinterlace import (
    DeinterlaceStage,
    IdetCounts,
    _build_deinterlace_filter,
    _decide_deinterlace,
    _fps_multiplier,
    _parse_idet_output,
)

# --------------------------------------------------------------------------
# _parse_idet_output -- pure, no ffmpeg
# --------------------------------------------------------------------------


class TestParseIdetOutput:
    def test_parses_multi_frame_line(self):
        text = (
            "[Parsed_idet_0 @ 0x1] Repeated Fields: Neither: 50 Top: 0 Bottom: 0\n"
            "[Parsed_idet_0 @ 0x1] Single frame detection: TFF: 38 BFF: 4 "
            "Progressive: 8 Undetermined: 0\n"
            "[Parsed_idet_0 @ 0x1] Multi frame detection: TFF: 41 BFF: 1 "
            "Progressive: 8 Undetermined: 0\n"
        )
        counts = _parse_idet_output(text)
        assert counts == IdetCounts(tff=41, bff=1, progressive=8, undetermined=0)

    def test_takes_last_multi_frame_occurrence(self):
        # Two idet instances in one run (shouldn't normally happen for this
        # stage's single-idet-pass analysis, but the parser must be robust
        # to it rather than silently picking the wrong one).
        text = (
            "Multi frame detection: TFF: 1 BFF: 1 Progressive: 1 Undetermined: 1\n"
            "Multi frame detection: TFF: 50 BFF: 0 Progressive: 0 Undetermined: 0\n"
        )
        counts = _parse_idet_output(text)
        assert counts == IdetCounts(tff=50, bff=0, progressive=0, undetermined=0)

    def test_empty_text_returns_none(self):
        assert _parse_idet_output("") is None
        assert _parse_idet_output(None) is None

    def test_malformed_text_returns_none(self):
        assert _parse_idet_output("this is not idet output at all") is None


# --------------------------------------------------------------------------
# _decide_deinterlace -- pure, no ffmpeg
# --------------------------------------------------------------------------


class TestDecideDeinterlace:
    def test_clearly_interlaced_tff(self):
        counts = IdetCounts(tff=50, bff=0, progressive=0, undetermined=0)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is True
        assert decision.field_order == "tff"
        assert decision.ratio == pytest.approx(1.0)

    def test_clearly_interlaced_bff(self):
        counts = IdetCounts(tff=0, bff=50, progressive=0, undetermined=0)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is True
        assert decision.field_order == "bff"

    def test_clearly_progressive_skips(self):
        counts = IdetCounts(tff=0, bff=0, progressive=50, undetermined=0)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is False
        assert decision.field_order is None
        assert decision.ratio == pytest.approx(0.0)
        assert "0/50" in decision.reason

    def test_all_undetermined_skips_and_excludes_from_denominator(self):
        counts = IdetCounts(tff=0, bff=0, progressive=0, undetermined=50)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is False
        assert decision.classified == 0
        assert decision.ratio is None
        assert "Undetermined: 50" in decision.reason

    def test_undetermined_excluded_from_denominator_not_zero_ratio(self):
        # 40 interlaced out of 50 CLASSIFIED (50 more Undetermined, excluded)
        # -> ratio 0.8, well above 0.7 -- must NOT be diluted to 40/100=0.4
        # by counting Undetermined in the denominator.
        counts = IdetCounts(tff=40, bff=0, progressive=10, undetermined=50)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.classified == 50
        assert decision.ratio == pytest.approx(0.8)
        assert decision.interlaced is True

    def test_ratio_just_above_threshold_deinterlaces(self):
        # 71/100 = 0.71 > 0.7
        counts = IdetCounts(tff=71, bff=0, progressive=29, undetermined=0)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is True
        assert decision.ratio == pytest.approx(0.71)

    def test_ratio_just_below_threshold_skips(self):
        # 70/100 = 0.70, not STRICTLY greater than 0.7 -- must skip ("exceed").
        counts = IdetCounts(tff=70, bff=0, progressive=30, undetermined=0)
        decision = _decide_deinterlace(counts, min_interlaced_ratio=0.7)
        assert decision.interlaced is False

    def test_malformed_none_counts_skips_without_raising(self):
        decision = _decide_deinterlace(None, min_interlaced_ratio=0.7)
        assert decision.interlaced is False
        assert decision.field_order is None
        assert "could not be parsed" in decision.reason


# --------------------------------------------------------------------------
# _build_deinterlace_filter -- pure, no ffmpeg
# --------------------------------------------------------------------------


class TestBuildDeinterlaceFilter:
    @pytest.mark.parametrize("filter_name", ["bwdif", "yadif", "w3fdif", "estdif"])
    @pytest.mark.parametrize("field_order", ["tff", "bff"])
    def test_frame_mode(self, filter_name, field_order):
        vf = _build_deinterlace_filter("frame", filter_name, field_order)
        assert vf == f"{filter_name}=mode=0:parity={field_order}"

    @pytest.mark.parametrize("filter_name", ["bwdif", "yadif", "w3fdif", "estdif"])
    @pytest.mark.parametrize("field_order", ["tff", "bff"])
    def test_field_mode(self, filter_name, field_order):
        vf = _build_deinterlace_filter("field", filter_name, field_order)
        assert vf == f"{filter_name}=mode=1:parity={field_order}"

    @pytest.mark.parametrize("filter_name", ["bwdif", "yadif", "w3fdif", "estdif", "bogus"])
    def test_bob_mode_is_filter_independent(self, filter_name):
        # bob is built from a separate setfield+separatefields+scale chain --
        # the `filter` config value is ignored entirely (even a bogus one).
        vf = _build_deinterlace_filter("bob", filter_name, "tff")
        assert vf == "setfield=mode=tff,separatefields,scale=iw:ih*2:flags=neighbor,setsar=1"

        vf_bff = _build_deinterlace_filter("bob", filter_name, "bff")
        assert vf_bff == "setfield=mode=bff,separatefields,scale=iw:ih*2:flags=neighbor,setsar=1"

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported stages.deinterlace.mode"):
            _build_deinterlace_filter("half-speed", "bwdif", "tff")

    def test_unsupported_filter_raises_for_frame_field_modes(self):
        with pytest.raises(ValueError, match="Unsupported stages.deinterlace.filter"):
            _build_deinterlace_filter("field", "nnedi3", "tff")

    def test_unresolved_field_order_raises(self):
        with pytest.raises(ValueError, match="resolved field_order"):
            _build_deinterlace_filter("field", "bwdif", "auto")


class TestFpsMultiplier:
    def test_frame_mode_unchanged(self):
        assert _fps_multiplier("frame") == 1.0

    def test_field_mode_doubles(self):
        assert _fps_multiplier("field") == 2.0

    def test_bob_mode_doubles(self):
        assert _fps_multiplier("bob") == 2.0


# --------------------------------------------------------------------------
# Stage-level tests (should_run/execute) -- ffmpeg mocked
# --------------------------------------------------------------------------


def _make_stage(tmp_path, **stage_overrides) -> DeinterlaceStage:
    config = Config(tmp_path / "nonexistent.yaml")
    config.set(True, "stages", "deinterlace", "enabled")
    for key, value in stage_overrides.items():
        config.set(value, "stages", "deinterlace", key)
    return DeinterlaceStage(config)


class TestShouldRun:
    def test_disabled_by_default_skips(self, tmp_path):
        # REQUIREMENTS.md § 20.2: enabled defaults False.
        config = Config(tmp_path / "nonexistent.yaml")
        stage = DeinterlaceStage(config)
        should_run, reason = stage.should_run({"has_video": True})
        assert should_run is False
        assert reason == "Stage disabled"

    def test_explicitly_disabled_skips(self, tmp_path):
        stage = _make_stage(tmp_path, enabled=False)
        should_run, reason = stage.should_run({"has_video": True})
        assert should_run is False

    def test_no_video_stream_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({"has_video": False})
        assert should_run is False
        assert reason == "No video stream"

    def test_enabled_with_video_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({"has_video": True})
        assert should_run is True


class TestExecuteNoOutputPath:
    def test_missing_output_path_fails(self, tmp_path):
        stage = _make_stage(tmp_path)
        result = stage.execute("in.mp4", output_path=None)
        assert result.status == StageStatus.FAILED


class TestExecuteExplicitFieldOrder:
    def test_tff_skips_idet_detection(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="field")
        fake_result = MagicMock(returncode=0, stderr="")
        with (
            patch("autovideofixer.core.stages.deinterlace._run_idet_analysis") as mock_idet,
            patch(
                "autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result
            ) as mock_run_ffmpeg,
        ):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        mock_idet.assert_not_called()
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["field_order"] == "tff"
        assert result.metadata["field_order_source"] == "explicit config"
        args = mock_run_ffmpeg.call_args.kwargs.get("args") or mock_run_ffmpeg.call_args.args[0]
        vf = args[args.index("-vf") + 1]
        assert vf == "bwdif=mode=1:parity=tff"

    def test_bff_skips_idet_detection(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="bff", mode="frame")
        fake_result = MagicMock(returncode=0, stderr="")
        with (
            patch("autovideofixer.core.stages.deinterlace._run_idet_analysis") as mock_idet,
            patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result),
        ):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        mock_idet.assert_not_called()
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["field_order"] == "bff"


class TestExecuteAutoDetection:
    def test_below_threshold_skips_with_counts(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="auto", min_interlaced_ratio=0.7)
        idet_stderr = "Multi frame detection: TFF: 0 BFF: 0 Progressive: 50 Undetermined: 0\n"
        with (
            patch(
                "autovideofixer.core.stages.deinterlace._run_idet_analysis",
                return_value=idet_stderr,
            ),
            patch("autovideofixer.core.stages.deinterlace.run_ffmpeg") as mock_run_ffmpeg,
        ):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.status == StageStatus.SKIPPED
        assert "0/50" in result.skipped_reason
        assert result.metadata["idet_counts"] == {
            "tff": 0,
            "bff": 0,
            "progressive": 50,
            "undetermined": 0,
        }
        mock_run_ffmpeg.assert_not_called()

    def test_above_threshold_deinterlaces_and_reports_field_order(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="auto", mode="field", min_interlaced_ratio=0.7)
        idet_stderr = "Multi frame detection: TFF: 50 BFF: 0 Progressive: 0 Undetermined: 0\n"
        fake_result = MagicMock(returncode=0, stderr="")
        with (
            patch(
                "autovideofixer.core.stages.deinterlace._run_idet_analysis",
                return_value=idet_stderr,
            ),
            patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result),
        ):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["field_order"] == "tff"
        assert result.metadata["field_order_source"] == "idet"
        assert result.metadata["idet_ratio"] == pytest.approx(1.0)

    def test_detection_error_fails_open_skips(self, tmp_path):
        """A detection failure (ffmpeg missing/timeout/crash) must SKIP, never FAIL."""
        stage = _make_stage(tmp_path, field_order="auto")
        with (
            patch(
                "autovideofixer.core.stages.deinterlace._run_idet_analysis",
                return_value=None,
            ),
            patch("autovideofixer.core.stages.deinterlace.run_ffmpeg") as mock_run_ffmpeg,
        ):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.status == StageStatus.SKIPPED
        mock_run_ffmpeg.assert_not_called()


class TestFpsOutPropagation:
    """REQUIREMENTS.md § 12.5: field/bob mode doubles the frame count and must
    report fps_out so the pipeline updates input_info["true_framerate"]."""

    def test_field_mode_doubles_fps_out(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="field")
        fake_result = MagicMock(returncode=0, stderr="")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["fps_in"] == 25.0
        assert result.metadata["fps_out"] == 50.0

    def test_bob_mode_doubles_fps_out(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="bff", mode="bob")
        fake_result = MagicMock(returncode=0, stderr="")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 30.0},
            )
        assert result.metadata["fps_out"] == 60.0

    def test_frame_mode_reports_unchanged_rate(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="frame")
        fake_result = MagicMock(returncode=0, stderr="")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.metadata["fps_out"] == 25.0

    def test_prefers_true_framerate_over_probed_framerate(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="field")
        fake_result = MagicMock(returncode=0, stderr="")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 60.0, "true_framerate": 25.0},
            )
        assert result.metadata["fps_in"] == 25.0
        assert result.metadata["fps_out"] == 50.0

    def test_missing_framerate_omits_fps_out(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="field")
        fake_result = MagicMock(returncode=0, stderr="")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute("in.mp4", output_path=str(tmp_path / "out.mkv"), input_info={})
        assert "fps_out" not in result.metadata


class TestExecuteFfmpegFailure:
    def test_ffmpeg_failure_fails_stage(self, tmp_path):
        stage = _make_stage(tmp_path, field_order="tff", mode="field")
        fake_result = MagicMock(returncode=1, stderr="boom")
        with patch("autovideofixer.core.stages.deinterlace.run_ffmpeg", return_value=fake_result):
            result = stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mkv"),
                input_info={"framerate": 25.0},
            )
        assert result.status == StageStatus.FAILED
