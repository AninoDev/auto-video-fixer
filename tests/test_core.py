"""Tests for Auto Video Fixer core functionality."""

import os
import tempfile

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import is_video_file, scan_directory
from autovideofixer.core.pipeline import Pipeline
from autovideofixer.core.presets import get_preset, list_presets


class TestConfig:
    """Test configuration system."""

    def test_default_config(self):
        config = Config()
        assert config.get("general", "max_concurrent_jobs") == 1
        assert config.get("quality", "vmaf_model") == "vmaf_v0.6.1"

    def test_set_value(self, tmp_path):
        config = Config(tmp_path / "test_config.yaml")
        config.set(2, "general", "max_concurrent_jobs")
        assert config.get("general", "max_concurrent_jobs") == 2


class TestPresets:
    """Test preset system."""

    def test_list_presets(self):
        presets = list_presets()
        assert len(presets) > 0
        assert "4k60" in presets
        assert "max_quality" in presets

    def test_get_preset(self):
        preset = get_preset("4k60")
        assert preset is not None
        assert preset.target_resolution == (3840, 2160)
        assert preset.target_framerate == 60.0

    def test_to_config(self):
        preset = get_preset("4k60")
        config = preset.to_config()
        assert "quality" in config
        assert "stages" in config


class TestAnalysis:
    """Test video analysis utilities."""

    def test_is_video_file(self, tmp_path):
        # Create test files
        video = tmp_path / "test.mp4"
        video.write_text("fake")
        image = tmp_path / "test.jpg"
        image.write_text("fake")

        assert is_video_file(str(video))
        assert not is_video_file(str(image))

    def test_scan_directory(self, tmp_path):
        # Create test files
        (tmp_path / "video1.mp4").write_text("")
        (tmp_path / "video2.mkv").write_text("")
        (tmp_path / "image.jpg").write_text("")

        videos = scan_directory(str(tmp_path))
        assert len(videos) == 2

    def test_scan_empty_directory(self, tmp_path):
        videos = scan_directory(str(tmp_path))
        assert len(videos) == 0


class TestPipeline:
    """Test pipeline engine."""

    def test_create_pipeline(self):
        config = Config()
        pipeline = Pipeline(config)
        assert len(pipeline.jobs) == 0

    def test_add_job(self, tmp_path):
        config = Config()
        pipeline = Pipeline(config)

        # Create a dummy file
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        job = pipeline.add_job(str(test_file))
        assert len(pipeline.jobs) == 1
        assert job.input_path == str(test_file)

    def test_auto_determine_stages(self, tmp_video_file):
        config = Config()
        # Set quality target
        config.set([3840, 2160], "quality", "quality_target", "target_resolution")
        config.set(60.0, "quality", "quality_target", "target_framerate")

        pipeline = Pipeline(config)

        job = pipeline.add_job(tmp_video_file)
        stages = pipeline.auto_determine_stages(job)

        # Should include upscale and interpolate based on config
        assert "upscale" in stages
        assert "interpolate" in stages
        assert "encode" in stages

    def test_optimize_stage_order(self):
        config = Config()
        pipeline = Pipeline(config)

        stages = ["encode", "upscale", "stabilize", "denoise_video"]
        ordered = pipeline.optimize_stage_order(stages)

        # Stabilize should come before upscale
        assert ordered.index("stabilize") < ordered.index("upscale")
        # Encode should be last
        assert ordered[-1] == "encode"


class TestQuality:
    """Test quality estimation (basic structure tests)."""

    def test_quality_result_creation(self):
        from autovideofixer.core.quality import QualityResult, QualityMode

        result = QualityResult(
            vmaf_score=95.0,
            psnr=40.0,
            ssim=0.95,
            mode=QualityMode.TARGET,
            target=90.0,
        )

        assert result.score == 95.0
        assert result.meets_target()

    def test_quality_not_met(self):
        from autovideofixer.core.quality import QualityResult, QualityMode

        result = QualityResult(
            vmaf_score=85.0,
            mode=QualityMode.TARGET,
            target=90.0,
        )

        assert not result.meets_target()
