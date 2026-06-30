"""Tests for processing stages."""

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

    def test_create_stage(self):
        """Test creating a stage instance."""
        config = Config()

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

    def test_create_nonexistent_stage(self):
        """Test creating a stage that doesn't exist."""
        config = Config()
        stage = create_stage("nonexistent_stage", config)
        assert stage is None

    def test_stage_should_run_default(self):
        """Test default should_run behavior."""
        config = Config()

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

    def test_stage_is_enabled(self):
        """Test stage enabled/disabled status."""
        config = Config()

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

    def test_stage_estimates_complexity(self):
        """Test complexity estimation."""
        config = Config()

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
