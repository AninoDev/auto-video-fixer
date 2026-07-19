"""Tests for the AI-fallback policy.

Covers: general.ai_fallback / stages.<name>.ai_fallback resolution,
BaseStage._ai_fallback_or_fail() behavior, and end-to-end stage-level
fallback (disabled -> FAILED with reason; enabled -> traditional path used
+ WARNING logged), plus the --ai-fallback/--no-ai-fallback CLI flag.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from autovideofixer.cli.cli import main
from autovideofixer.config import Config
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class _DummyAIStage(BaseStage):
    """Minimal concrete stage for exercising BaseStage's fallback helpers
    in isolation, without needing torch/ffmpeg/a real AI-capable stage."""

    name = "dummy_ai_stage"
    display_name = "Dummy AI Stage"
    category = "enhancement"

    def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
        raise NotImplementedError


class TestIsAiFallbackEnabled:
    def test_default_enabled(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        stage = _DummyAIStage(config)
        assert stage.is_ai_fallback_enabled() is True

    def test_global_disabled(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = _DummyAIStage(config)
        assert stage.is_ai_fallback_enabled() is False

    def test_per_stage_overrides_global_to_true(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        config.set(True, "stages", "dummy_ai_stage", "ai_fallback")
        stage = _DummyAIStage(config)
        assert stage.is_ai_fallback_enabled() is True

    def test_per_stage_overrides_global_to_false(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(True, "general", "ai_fallback")
        config.set(False, "stages", "dummy_ai_stage", "ai_fallback")
        stage = _DummyAIStage(config)
        assert stage.is_ai_fallback_enabled() is False

    def test_per_stage_null_inherits_global(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        config.set(None, "stages", "dummy_ai_stage", "ai_fallback")
        stage = _DummyAIStage(config)
        assert stage.is_ai_fallback_enabled() is False


class TestAiFallbackOrFail:
    def test_disabled_returns_failed_with_reason(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = _DummyAIStage(config)
        called = []

        def traditional():
            called.append(True)
            return StageResult(status=StageStatus.COMPLETED, output_path="/x")

        result = stage._ai_fallback_or_fail("torch missing", time.time(), traditional)

        assert result.status == StageStatus.FAILED
        assert "torch missing" in result.error
        assert "dummy_ai_stage" in result.error
        assert "ai_fallback" in result.error
        assert called == []  # traditional() must never run when fallback is disabled

    def test_enabled_calls_traditional_and_warns(self, tmp_path, caplog):
        config = Config(tmp_path / "c.yaml")
        stage = _DummyAIStage(config)

        def traditional():
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path="/x",
                metadata={"method": "traditional"},
            )

        with caplog.at_level(logging.WARNING, logger="autovideofixer.stages.dummy_ai_stage"):
            result = stage._ai_fallback_or_fail("torch missing", time.time(), traditional)

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "traditional"
        assert any("torch missing" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_enabled_marks_fallback_provenance(self, tmp_path):
        """REQUIREMENTS.md § 6.4: the ONE seam every AI->traditional fallback
        flows through must mark ai_fallback_used/ai_fallback_reason on the
        returned result so reporting can tell "fell back" from "chose
        traditional outright" (core/reporting.py's classify_stage())."""
        config = Config(tmp_path / "c.yaml")
        stage = _DummyAIStage(config)

        def traditional():
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path="/x",
                metadata={"method": "traditional"},
            )

        result = stage._ai_fallback_or_fail("torch missing", time.time(), traditional)

        assert result.metadata["ai_fallback_used"] is True
        assert result.metadata["ai_fallback_reason"] == "torch missing"

    def test_disabled_does_not_mark_fallback_provenance(self, tmp_path):
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = _DummyAIStage(config)

        def traditional():
            return StageResult(status=StageStatus.COMPLETED, output_path="/x")

        result = stage._ai_fallback_or_fail("torch missing", time.time(), traditional)

        assert "ai_fallback_used" not in result.metadata


class TestStageLevelFallback:
    """End-to-end through a real AI-capable stage, with torch availability
    mocked so no actual model/GPU is needed."""

    def test_deblock_fallback_disabled_fails(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.deblock import DeblockStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = DeblockStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), None, time.time()
        )

        assert result.status == StageStatus.FAILED
        assert "PyTorch not installed" in result.error
        assert "ai_fallback" in result.error

    @patch("autovideofixer.core.stages.deblock.run_ffmpeg")
    def test_deblock_fallback_enabled_uses_traditional(
        self, mock_run_ffmpeg, tmp_path, monkeypatch, caplog
    ):
        from autovideofixer.core.stages.deblock import DeblockStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")

        config = Config(tmp_path / "c.yaml")  # ai_fallback defaults to True
        stage = DeblockStage(config)

        with caplog.at_level(logging.WARNING, logger="autovideofixer.stages.deblock"):
            result = stage._execute_ai(
                str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), None, time.time()
            )

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "traditional"
        assert any("PyTorch not installed" in r.message for r in caplog.records)
        mock_run_ffmpeg.assert_called_once()

        from autovideofixer.core.reporting import classify_stage

        assert classify_stage(result) == "ran-traditional-fallback"

    def test_denoise_video_fallback_disabled_fails(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        config = Config(tmp_path / "c.yaml")
        config.set(False, "stages", "denoise_video", "ai_fallback")
        stage = DenoiseVideoStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), None, time.time()
        )

        assert result.status == StageStatus.FAILED
        assert "PyTorch not installed" in result.error

    @patch("autovideofixer.core.stages.denoise_video.run_ffmpeg")
    def test_denoise_video_fallback_enabled_uses_traditional(
        self, mock_run_ffmpeg, tmp_path, monkeypatch
    ):
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")

        config = Config(tmp_path / "c.yaml")
        stage = DenoiseVideoStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), None, time.time()
        )

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "traditional"

    def test_interpolate_fallback_disabled_fails(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.interpolate import InterpolateStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = InterpolateStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"),
            str(tmp_path / "out.mp4"),
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.FAILED
        assert "PyTorch not installed" in result.error

    @patch("autovideofixer.core.stages.interpolate.run_ffmpeg")
    def test_interpolate_fallback_enabled_uses_traditional(
        self, mock_run_ffmpeg, tmp_path, monkeypatch
    ):
        from autovideofixer.core.stages.interpolate import InterpolateStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")

        config = Config(tmp_path / "c.yaml")
        stage = InterpolateStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"),
            str(tmp_path / "out.mp4"),
            None,
            time.time(),
            target_fps=60.0,
            current_fps=30.0,
        )

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "traditional"

    def test_upscale_fallback_disabled_fails(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.upscale import UpscaleStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        # Upscale needs input dimensions before it will even attempt an AI pass.
        monkeypatch.setattr(UpscaleStage, "_get_input_resolution", lambda self, path: (640, 360))
        config = Config(tmp_path / "c.yaml")
        config.set(False, "general", "ai_fallback")
        stage = UpscaleStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"),
            str(tmp_path / "out.mp4"),
            None,
            time.time(),
            target_width=1280,
            target_height=720,
        )

        assert result.status == StageStatus.FAILED
        assert "PyTorch not installed" in result.error

    @patch("autovideofixer.core.stages.upscale.run_ffmpeg")
    def test_upscale_fallback_enabled_uses_traditional(
        self, mock_run_ffmpeg, tmp_path, monkeypatch
    ):
        from autovideofixer.core.stages.upscale import UpscaleStage

        monkeypatch.setattr("autovideofixer.ai.torch_utils.is_torch_available", lambda: False)
        monkeypatch.setattr(UpscaleStage, "_get_input_resolution", lambda self, path: (640, 360))
        mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")

        config = Config(tmp_path / "c.yaml")
        stage = UpscaleStage(config)

        result = stage._execute_ai(
            str(tmp_path / "in.mp4"),
            str(tmp_path / "out.mp4"),
            None,
            time.time(),
            target_width=1280,
            target_height=720,
        )

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["method"] == "traditional"
        mock_run_ffmpeg.assert_called_once()


class TestAiFallbackCli:
    """--ai-fallback/--no-ai-fallback CLI flag wiring."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_flag_disables_fallback_in_config(self, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        with patch("autovideofixer.cli.cli.Pipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.add_files.return_value = []
            mock_pipeline_cls.return_value = mock_pipeline

            result = self.runner.invoke(
                main, ["process", str(test_file), "--no-ai-fallback", "--dry-run"]
            )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_flag_beats_config_default(self, tmp_path, monkeypatch):
        """--no-ai-fallback must override general.ai_fallback even when the
        loaded config file has it enabled.

        CLI flags are folded into a "cli-flags" cascade layer via
        Config.apply_layer() (not per-key Config.set() calls -- see
        AGENTS.md's "Config cascade" section), so this spies on apply_layer
        and inspects the effective config after every layer has been applied,
        rather than a single .set() call for this specific key.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text("general:\n  ai_fallback: true\n")
        monkeypatch.setenv("AVF_CONFIG", str(config_path))

        captured = {}
        real_apply_layer = Config.apply_layer

        def spy_apply_layer(self, layer, source_label):
            real_apply_layer(self, layer, source_label)
            if source_label == "cli-flags":
                captured["value"] = self.get("general", "ai_fallback")

        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        with patch.object(Config, "apply_layer", spy_apply_layer):
            result = self.runner.invoke(
                main, ["process", str(test_file), "--no-ai-fallback", "--dry-run"]
            )

        assert result.exit_code == 0
        assert captured.get("value") is False
