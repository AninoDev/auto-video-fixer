"""Unit tests for DownscaleStage (REQUIREMENTS.md § 7).

Covers should_run()'s gating (enabled/target/already-at-or-below/tolerance,
orientation-aware) and _target_dimensions()'s fitted-dims computation.
"""

from __future__ import annotations

from autovideofixer.config import Config
from autovideofixer.core.stages.downscale import DownscaleStage


def _make_stage(tmp_path, enabled: bool = True) -> DownscaleStage:
    config = Config(tmp_path / "nonexistent.yaml")
    config.set(enabled, "stages", "downscale", "enabled")
    return DownscaleStage(config)


class TestShouldRun:
    def test_disabled_skips(self, tmp_path):
        stage = _make_stage(tmp_path, enabled=False)
        should_run, reason = stage.should_run(
            {"resolution": (3840, 2160), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Stage disabled"

    def test_no_target_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run({"resolution": (3840, 2160)})
        assert should_run is False
        assert reason == "No target resolution specified"

    def test_4k_input_runs(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (3840, 2160), "target_resolution": [1920, 1080]}
        )
        assert should_run is True
        assert reason is None

    def test_already_at_or_below_target_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (1280, 720), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Input already at or below target resolution"

    def test_exact_target_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (1920, 1080), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Input already at or below target resolution"

    def test_marginally_larger_within_tolerance_skips(self, tmp_path):
        stage = _make_stage(tmp_path)
        # 1960x1102 is ~2% larger than 1920x1080 -- within SKIP_SCALE_THRESHOLD (1.05).
        should_run, reason = stage.should_run(
            {"resolution": (1960, 1102), "target_resolution": [1920, 1080]}
        )
        assert should_run is False
        assert reason == "Within downscale tolerance"

    def test_portrait_input_landscape_target_downscales(self, tmp_path):
        stage = _make_stage(tmp_path)
        should_run, reason = stage.should_run(
            {"resolution": (2160, 3840), "target_resolution": [1920, 1080]}
        )
        assert should_run is True
        assert reason is None


class TestTargetDimensions:
    def test_4k_to_1080p_preserve_aspect(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._target_dimensions(3840, 2160, 1920, 1080) == (1920, 1080)

    def test_portrait_orientation_aware(self, tmp_path):
        stage = _make_stage(tmp_path)
        w, h = stage._target_dimensions(2160, 3840, 1920, 1080)
        assert (w, h) == (1080, 1920)

    def test_snap_limiting_mode(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(True, "stages", "downscale", "enabled")
        config.set("snap_limiting", "quality", "quality_target", "resolution_fit_mode")
        stage = DownscaleStage(config)
        w, h = stage._target_dimensions(3841, 2160, 1920, 1080)
        assert w == 1920
        assert h % 2 == 0
