"""Tests for processing stages."""

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.base import (
    BaseStage,
    StageResult,
    StageStatus,
    create_stage,
    list_stages,
    register_stage,
)


class TestStageBase:
    """Test base stage functionality."""

    def test_stage_result_success(self):
        """Test StageResult with success status."""
        result = StageResult(
            status=StageStatus.COMPLETED,
            output_path="/tmp/output.mp4",
            duration_sec=10.5,
        )
        assert result.success is True
        assert result.output_path == "/tmp/output.mp4"
        assert result.duration_sec == 10.5

    def test_stage_result_failed(self):
        """Test StageResult with failure status."""
        result = StageResult(
            status=StageStatus.FAILED,
            error="Test error",
            duration_sec=5.0,
        )
        assert result.success is False
        assert result.error == "Test error"

    def test_stage_result_skipped(self):
        """Test StageResult with skipped status."""
        result = StageResult(
            status=StageStatus.SKIPPED,
            skipped_reason="Not applicable",
        )
        assert result.success is True

    def test_register_stage(self):
        """Test stage registration decorator."""

        @register_stage
        class TestStage(BaseStage):
            name = "test_stage"
            display_name = "Test Stage"
            category = "enhancement"

            def execute(self, input_path, output_path, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        stages = list_stages()
        assert "test_stage" in stages
        assert stages["test_stage"] is TestStage

    def test_create_stage(self, tmp_path):
        """Test creating a stage instance."""
        config = Config(tmp_path / "nonexistent.yaml")

        # Register a test stage
        @register_stage
        class CreateTestStage(BaseStage):
            name = "create_test"
            display_name = "Create Test"
            category = "enhancement"

            def execute(self, input_path, output_path, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED)

        stage = create_stage("create_test", config)
        assert stage is not None
        assert isinstance(stage, CreateTestStage)

    def test_create_nonexistent_stage(self, tmp_path):
        """Test creating a stage that doesn't exist."""
        config = Config(tmp_path / "nonexistent.yaml")
        stage = create_stage("nonexistent_stage", config)
        assert stage is None

    def test_stage_should_run_default(self, tmp_path):
        """Test default should_run behavior."""
        config = Config(tmp_path / "nonexistent.yaml")

        @register_stage
        class DefaultShouldRunStage(BaseStage):
            name = "default_should_run"
            display_name = "Default Should Run"
            category = "enhancement"

            def execute(self, input_path, output_path, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED)

        stage = create_stage("default_should_run", config)
        should_run, reason = stage.should_run({})
        assert should_run is True
        assert reason is None

    def test_stage_is_enabled(self, tmp_path):
        """Test stage enabled/disabled status."""
        config = Config(tmp_path / "nonexistent.yaml")

        @register_stage
        class EnabledStage(BaseStage):
            name = "enabled_test"
            display_name = "Enabled Test"
            category = "enhancement"

            def execute(self, input_path, output_path, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED)

        stage = create_stage("enabled_test", config)
        assert stage.is_enabled() is True

        # Disable the stage
        config.set(False, "stages", "enabled_test", "enabled")
        stage2 = create_stage("enabled_test", config)
        assert stage2.is_enabled() is False

    def test_stage_estimates_complexity(self, tmp_path):
        """Test complexity estimation."""
        config = Config(tmp_path / "nonexistent.yaml")

        @register_stage
        class ComplexityStage(BaseStage):
            name = "complexity_test"
            display_name = "Complexity Test"
            category = "enhancement"

            def execute(self, input_path, output_path, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED)

        stage = create_stage("complexity_test", config)

        # Low resolution, short duration
        info = {"resolution": (1920, 1080), "duration": 60}
        complexity = stage.estimate_complexity(info)
        assert complexity >= 1.0

        # High resolution, long duration
        info = {"resolution": (3840, 2160), "duration": 3600}
        complexity = stage.estimate_complexity(info)
        assert complexity > 1.0


class TestStageTimeoutResolution:
    """BaseStage.stage_timeout(): stages.<name>.timeout -> pipeline.stage_timeout
    -> None (unlimited). See AGENTS.md's timeout section / config.py's
    resolve_timeout()."""

    def _register(self):
        @register_stage
        class TimeoutTestStage(BaseStage):
            name = "timeout_test"
            display_name = "Timeout Test"
            category = "enhancement"

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                return StageResult(status=StageStatus.COMPLETED)

    def _config(self, tmp_path) -> Config:
        self._register()
        return Config(tmp_path / "nonexistent.yaml")

    def test_default_is_none_unlimited(self, tmp_path):
        """Neither stages.timeout_test.timeout nor pipeline.stage_timeout is
        set -- both DEFAULTS are null, so the resolved timeout is None."""
        config = self._config(tmp_path)
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() is None

    def test_global_pipeline_stage_timeout_applies(self, tmp_path):
        config = self._config(tmp_path)
        config.set(900, "pipeline", "stage_timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() == 900

    def test_per_stage_timeout_wins_over_global(self, tmp_path):
        config = self._config(tmp_path)
        config.set(900, "pipeline", "stage_timeout")
        config.set(120, "stages", "timeout_test", "timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() == 120

    def test_per_stage_timeout_used_without_global_set(self, tmp_path):
        config = self._config(tmp_path)
        config.set(45, "stages", "timeout_test", "timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() == 45

    def test_global_null_explicit_means_unlimited(self, tmp_path):
        config = self._config(tmp_path)
        config.set(None, "pipeline", "stage_timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() is None

    def test_global_zero_means_unlimited(self, tmp_path):
        config = self._config(tmp_path)
        config.set(0, "pipeline", "stage_timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() is None

    def test_per_stage_zero_means_unlimited_even_with_global_set(self, tmp_path):
        config = self._config(tmp_path)
        config.set(900, "pipeline", "stage_timeout")
        config.set(0, "stages", "timeout_test", "timeout")
        stage = create_stage("timeout_test", config)
        assert stage.stage_timeout() is None

    def test_negative_global_raises(self, tmp_path):
        config = self._config(tmp_path)
        config.set(-5, "pipeline", "stage_timeout")
        stage = create_stage("timeout_test", config)
        with pytest.raises(ValueError):
            stage.stage_timeout()

    def test_negative_per_stage_raises(self, tmp_path):
        config = self._config(tmp_path)
        config.set(-1, "stages", "timeout_test", "timeout")
        stage = create_stage("timeout_test", config)
        with pytest.raises(ValueError):
            stage.stage_timeout()

    def test_garbage_value_raises(self, tmp_path):
        config = self._config(tmp_path)
        config.set("not-a-number", "stages", "timeout_test", "timeout")
        stage = create_stage("timeout_test", config)
        with pytest.raises(ValueError):
            stage.stage_timeout()

    def test_per_occurrence_override_via_base_stage_overrides_param(self, tmp_path):
        """The per-occurrence pipeline.default_order ``config: {timeout: ...}``
        mechanism deep-merges onto stages.<name> via BaseStage.__init__'s
        ``overrides`` param -- verify stage_timeout() picks that up too (the
        end-to-end StageOrderEntry version of this lives in
        test_pipeline.py::TestResolveStageOrder)."""
        config = self._config(tmp_path)
        overridden = create_stage("timeout_test", config, overrides={"timeout": 30})
        assert overridden.stage_timeout() == 30


class TestEncodeStageUsesResolvedTimeout:
    """At least one real stage's main ffmpeg pass must actually use
    stage_timeout() -- not just define it. EncodeStage is the simplest
    single-run_ffmpeg-call stage to verify this against."""

    def test_encode_main_pass_passes_resolved_timeout(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from autovideofixer.core.stages.encode import EncodeStage

        config = Config(tmp_path / "nonexistent.yaml")
        config.set(123, "pipeline", "stage_timeout")
        stage = EncodeStage(config)

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run_ffmpeg:
            mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
            stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mp4"),
                hwaccel="none",
            )

        assert mock_run_ffmpeg.call_args.kwargs["timeout"] == 123

    def test_encode_main_pass_passes_none_when_unlimited(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from autovideofixer.core.stages.encode import EncodeStage

        config = Config(tmp_path / "nonexistent.yaml")
        stage = EncodeStage(config)

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run_ffmpeg:
            mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
            stage.execute(
                "in.mp4",
                output_path=str(tmp_path / "out.mp4"),
                hwaccel="none",
            )

        assert mock_run_ffmpeg.call_args.kwargs["timeout"] is None


class TestStageRegistry:
    """Test stage registry functionality."""

    def test_list_all_stages(self):
        """Test listing all registered stages."""
        stages = list_stages()
        assert isinstance(stages, dict)
        assert len(stages) > 0

        # Check that core stages are registered
        core_stages = ["detect", "stabilize", "denoise_video", "upscale", "encode"]
        for stage_name in core_stages:
            assert stage_name in stages, f"Stage {stage_name} not registered"

    def test_stage_properties(self):
        """Test stage properties and metadata."""
        stages = list_stages()

        for name, stage_cls in stages.items():
            assert hasattr(stage_cls, "name")
            assert hasattr(stage_cls, "display_name")
            assert hasattr(stage_cls, "category")
            assert stage_cls.name == name

    def test_stage_categories(self):
        """Test that stages have proper categories."""
        stages = list_stages()
        valid_categories = {"analysis", "enhancement", "encoding", "output"}

        for name, stage_cls in stages.items():
            assert stage_cls.category in valid_categories, (
                f"Stage {name} has invalid category: {stage_cls.category}"
            )
