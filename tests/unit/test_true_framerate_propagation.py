"""Regression tests for REQUIREMENTS.md § 12.3 -- true content cadence
propagation.

The whole payoff of the true-cadence-recovery feature (§ 12) is that
InterpolateStage plans against the RECOVERED true framerate
(``input_info["true_framerate"]``, published by the retime stage) instead of
the encoded/probed one. Before this fix, a 24-in-60 input targeting 60fps
read as 60->60 ("already at target") and silently skipped -- the exact bug
this feature exists to close. These tests pin that behavior down at both
``should_run()`` and ``execute()``.

Also covers the companion regression: the raw-pipe DECODE reader used by
StabilizeStage's manual decode/transform pipe must carry ``-fps_mode
passthrough`` in its output args, or ffmpeg silently re-expands a VFR
intermediate back to CFR by duplicating frames (verified empirically: 48
real frames -> 120 output frames -- REQUIREMENTS.md § 12.4b/12.6).
"""

from __future__ import annotations

from unittest.mock import patch

from autovideofixer.config import Config
from autovideofixer.core.stages.interpolate import InterpolateStage
from autovideofixer.core.stages.stabilize import StabilizeStage


def _interpolate_stage(tmp_path) -> InterpolateStage:
    config = Config(tmp_path / "nonexistent.yaml")
    config.set(60.0, "quality", "quality_target", "target_framerate")
    return InterpolateStage(config)


class TestShouldRunPrefersTrueFramerate:
    def test_true_framerate_below_target_runs_despite_encoded_framerate_matching(self, tmp_path):
        """framerate=60 (encoded), true_framerate=24 (recovered), target=60
        -- must plan real interpolation (24->60), not skip as already-at-
        target."""
        stage = _interpolate_stage(tmp_path)
        should_run, reason = stage.should_run({"framerate": 60, "true_framerate": 24})
        assert should_run is True
        assert reason is None

    def test_plain_framerate_at_target_still_skips_without_true_framerate(self, tmp_path):
        """Unchanged pre-existing behavior when there's no recovered cadence
        at all -- proves the fix is additive, not a behavior change for
        inputs retime never touched."""
        stage = _interpolate_stage(tmp_path)
        should_run, reason = stage.should_run({"framerate": 60})
        assert should_run is False
        assert reason == "Already at or above target framerate"

    def test_true_framerate_zero_falls_back_to_framerate(self, tmp_path):
        # A falsy true_framerate (0, absent, None) must not silently plan
        # against a bogus 0fps "current" rate.
        stage = _interpolate_stage(tmp_path)
        should_run, reason = stage.should_run({"framerate": 60, "true_framerate": 0})
        assert should_run is False
        assert reason == "Already at or above target framerate"


class TestExecutePrefersTrueFramerate:
    def test_execute_plans_against_true_framerate_not_a_fresh_probe(self, tmp_path):
        """execute() must NOT re-probe for fps when input_info already
        carries true_framerate -- re-probing a VFR intermediate downstream
        of retime is unreliable (avg_frame_rate lies, see AGENTS.md's
        gotcha). Verified here by making a fresh probe return an obviously
        wrong value (30fps) that must NOT be what current_fps ends up as.
        """
        stage = _interpolate_stage(tmp_path)
        captured: dict = {}

        def _fake_traditional(self_, input_path, output_path, progress_callback, start, **kwargs):
            captured.update(kwargs)
            from autovideofixer.core.stages.base import StageResult, StageStatus

            return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        with (
            patch(
                "autovideofixer.core.ffmpeg_utils.get_video_info",
                return_value={"framerate": 30.0},
            ) as mock_probe,
            patch.object(InterpolateStage, "_execute_traditional", _fake_traditional, create=True),
        ):
            stage.execute(
                "in.mkv",
                output_path=str(tmp_path / "out.mkv"),
                target_fps=60.0,
                input_info={"framerate": 60.0, "true_framerate": 24.0},
            )

        # The fresh-probe fallback must never even be called -- true_framerate
        # from input_info was already usable.
        mock_probe.assert_not_called()
        assert captured["current_fps"] == 24.0

    def test_execute_falls_back_to_probe_when_input_info_missing(self, tmp_path):
        """Back-compat: direct/test invocations that don't pass input_info
        at all must still work exactly as before (fresh probe fallback)."""
        stage = _interpolate_stage(tmp_path)
        captured: dict = {}

        def _fake_traditional(self_, input_path, output_path, progress_callback, start, **kwargs):
            captured.update(kwargs)
            from autovideofixer.core.stages.base import StageResult, StageStatus

            return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        with (
            patch(
                "autovideofixer.core.ffmpeg_utils.get_video_info",
                return_value={"framerate": 30.0},
            ),
            patch.object(InterpolateStage, "_execute_traditional", _fake_traditional, create=True),
        ):
            stage.execute("in.mp4", output_path=str(tmp_path / "out.mp4"), target_fps=60.0)

        assert captured["current_fps"] == 30.0


class TestStabilizeRawPipeReaderPassthrough:
    """Regression: the raw-pipe DECODE reader must carry -fps_mode
    passthrough, or ffmpeg silently re-expands a VFR input back to CFR by
    duplicating frames -- negating retime's entire saving at full
    decode+transform cost (§ 12.4b, empirically verified 48 -> 120 frames)."""

    def test_decode_args_contain_fps_mode_passthrough(self):
        args = StabilizeStage._build_decode_args("ffmpeg", "in.mkv", "yuv420p", 1920, 1080)
        assert "-fps_mode" in args
        assert args[args.index("-fps_mode") + 1] == "passthrough"

    def test_fps_mode_appears_after_the_input_spec(self):
        # -fps_mode is an OUTPUT option -- placing it before -i corrupts
        # input parsing (spurious "Duplicate element"/EBML errors on
        # Matroska). Verify ordering, not just presence.
        args = StabilizeStage._build_decode_args("ffmpeg", "in.mkv", "yuv420p", 1920, 1080)
        assert args.index("-fps_mode") > args.index("-i")
