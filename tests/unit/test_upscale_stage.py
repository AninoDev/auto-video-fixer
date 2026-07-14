"""Unit tests for UpscaleStage's orientation-aware target-resolution handling.

Covers a real-world bug: a portrait input (1080x1920) already at its
orientation-rotated target (preset target_resolution=[1920, 1080], which
rotates to 1080x1920 for a portrait input per AGENTS.md's "Upscaling &
Aspect Ratio" section) was NOT recognized as "already at target" because:

1. `should_run()` compared the raw (unrotated) target against the input
   without ever rotating it for orientation.
2. `execute()`'s AI-vs-traditional method selection made the same mistake.
3. `_execute_ai()`'s per-pass scale-factor computation, when it computed a
   pass's own needed scale as <=1 (already at/past target), substituted a
   hardcoded 2x scale (`sf = 2 ** round(log2(sf)) if sf > 1 else 2`) instead
   of doing nothing -- this is the exact mechanism that produced the
   reported 2160x3840 output (1080x1920 input, RealESRGAN_x2plus at 2x, no
   downstream correction) from an input already at target.

All three are fixed; these tests cover should_run()'s gate, execute()'s
method selection, and _execute_ai()'s defensive skip/resize path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.upscale import UpscaleStage


def _make_stage(tmp_path) -> UpscaleStage:
    config = Config(tmp_path / "nonexistent.yaml")
    return UpscaleStage(config)


class TestShouldRunOrientationAware:
    """should_run() must rotate the target bounding box to the input's orientation."""

    def test_portrait_at_target_skips(self, tmp_path):
        """Exact repro of the reported bug: 1080x1920 portrait input against
        a [1920, 1080] (landscape) preset target must SKIP, not run AI."""
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (1080, 1920), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Already at target resolution"

    def test_portrait_below_target_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (540, 960), "target_resolution": [1920, 1080]}
        )
        assert should_run is True
        assert reason is None

    def test_landscape_at_target_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (1920, 1080), "target_resolution": [1920, 1080]}
        )
        assert should_run is False

    def test_landscape_below_target_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (960, 540), "target_resolution": [1920, 1080]}
        )
        assert should_run is True

    def test_square_input_at_target_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        # Square target bound uses the shorter preset edge (1080) for both dims.
        should_run, reason = stage.should_run(
            {"resolution": (1080, 1080), "target_resolution": [1920, 1080]}
        )
        assert should_run is False

    def test_square_input_below_target_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (540, 540), "target_resolution": [1920, 1080]}
        )
        assert should_run is True

    def test_few_px_under_target_after_crop_skips(self, tmp_path):
        """A crop stage shaving a few px off before upscale runs must not
        trigger a full AI pass -- within the skip-scale threshold."""
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (1072, 1908), "target_resolution": [1920, 1080]}
        )
        assert should_run is False

    def test_disabled_stage_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._stage_config = dict(stage._stage_config)
        stage._stage_config["enabled"] = False
        should_run, reason = stage.should_run(
            {"resolution": (540, 960), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Stage disabled"


class TestMethodSelectionOrientationAware:
    """execute()'s ai-vs-traditional pick must also be orientation-aware."""

    def test_portrait_at_target_picks_traditional_not_ai(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._input_info = {"resolution": (1080, 1920)}
        with (
            patch.object(stage, "_execute_traditional") as trad,
            patch.object(stage, "_execute_ai") as ai,
        ):
            trad.return_value = MagicMock(status=StageStatus.COMPLETED)
            stage.execute("in.mp4", "out.mp4", target_width=1920, target_height=1080)
        ai.assert_not_called()
        trad.assert_called_once()


class TestExecuteAiSkipThreshold:
    """_execute_ai() must not run a wasted AI pass when already at target."""

    def test_exact_match_copies_instead_of_ai_pass(self, tmp_path):
        stage = _make_stage(tmp_path)
        with (
            patch.object(stage, "_get_input_resolution", return_value=(1080, 1920)),
            patch.object(stage, "_run_single_ai_pass") as ai_pass,
            patch.object(stage, "_copy_stream") as copy_stream,
        ):
            copy_stream.return_value = MagicMock(status=StageStatus.COMPLETED)
            result = stage._execute_ai(
                "in.mp4",
                "out.mp4",
                None,
                0.0,
                target_width=1920,
                target_height=1080,
            )
        ai_pass.assert_not_called()
        copy_stream.assert_called_once()
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "copy"

    def test_few_px_under_target_resizes_instead_of_ai_pass(self, tmp_path):
        """1072x1908 vs. a [1920, 1080] target (portrait-rotated to
        1080x1920) is a ~1.0075x scale -- must resize/copy, NOT run a 2x AI
        pass (the exact production bug: this used to produce a 2160x3840
        output via a forced RealESRGAN_x2plus pass)."""
        stage = _make_stage(tmp_path)
        with (
            patch.object(stage, "_get_input_resolution", return_value=(1072, 1908)),
            patch.object(stage, "_run_single_ai_pass") as ai_pass,
            patch.object(stage, "_resize_to_exact") as resize,
        ):
            resize.return_value = MagicMock(status=StageStatus.COMPLETED)
            result = stage._execute_ai(
                "in.mp4",
                "out.mp4",
                None,
                0.0,
                target_width=1920,
                target_height=1080,
            )
        ai_pass.assert_not_called()
        resize.assert_called_once()
        # Never a forced 2x AI pass -- resize call must target near the
        # input's own size (aspect-preserving fit within the rotated
        # 1080x1920 bound), NOT input*2 (2144x3816, what the bug produced).
        called_args = resize.call_args[0]
        assert called_args[2] < 1100
        assert called_args[3] < 1950
        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "resize"

    def test_needs_upscale_runs_ai_pass(self, tmp_path):
        """Sanity check: a genuine below-target input still goes through AI."""
        stage = _make_stage(tmp_path)
        with (
            patch.object(stage, "_get_input_resolution", return_value=(540, 960)),
            patch.object(stage, "_run_single_ai_pass") as ai_pass,
        ):
            ai_pass.return_value = MagicMock(
                status=StageStatus.COMPLETED, metadata={"method": "ai", "scale": 2.0}
            )
            stage._execute_ai(
                "in.mp4",
                "out.mp4",
                None,
                0.0,
                target_width=1920,
                target_height=1080,
            )
        ai_pass.assert_called_once()


class TestEffectiveTargetBounds:
    """Shared orientation-rotation helper used by should_run/execute/_calculate_target_dims."""

    def test_portrait_rotates_bounds(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._effective_target_bounds(1080, 1920, 1920, 1080) == (1080, 1920)

    def test_landscape_keeps_bounds(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._effective_target_bounds(1920, 1080, 1920, 1080) == (1920, 1080)

    def test_square_uses_shorter_edge(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._effective_target_bounds(500, 500, 1920, 1080) == (1080, 1080)

    def test_keep_aspect_ratio_false_returns_unrotated(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._keep_aspect_ratio = False
        assert stage._effective_target_bounds(1080, 1920, 1920, 1080) == (1920, 1080)


class TestCalculateTargetDimensions:
    def test_portrait_scales_to_rotated_bound(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._calculate_target_dimensions(540, 960, 1920, 1080) == (1080, 1920)

    def test_already_at_target_is_noop(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._calculate_target_dimensions(1080, 1920, 1920, 1080) == (1080, 1920)
