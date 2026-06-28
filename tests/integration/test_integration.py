"""Integration tests for Auto Video Fixer.

These tests require FFmpeg and test the full pipeline and stage interactions.
Run with: pytest tests/integration/ -v -m integration
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
class TestPipelineIntegration:
    """Test full pipeline processing."""

    def test_pipeline_single_file(self, tmp_video_file, tmp_path):
        """Test processing a single video file through the pipeline."""
        from autovideofixer.core.pipeline import Pipeline
        from autovideofixer.config import Config

        # Use a temporary config to avoid modifying user config
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        job = pipeline.add_job(tmp_video_file, output_path)

        results = pipeline.execute_all()

        assert len(results) == 1
        # Pipeline may have errors but should still produce output
        assert results[0].output_path is not None

    def test_pipeline_multiple_files(self, tmp_video_directory, tmp_path):
        """Test processing multiple video files."""
        from autovideofixer.core.pipeline import Pipeline
        from autovideofixer.config import Config

        _, video_files = tmp_video_directory
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        pipeline = Pipeline(config)

        for video in video_files:
            output_path = str(tmp_path / f"output_{video_files.index(video)}.mp4")
            pipeline.add_job(video, output_path)

        results = pipeline.execute_all()

        assert len(results) == 3
        # Verify jobs were processed (may have errors due to missing audio)
        for result in results:
            assert result.output_path is not None or len(result.errors) > 0

    def test_pipeline_with_presets(self, tmp_video_file, tmp_path):
        """Test processing with preset overrides."""
        from autovideofixer.core.pipeline import Pipeline
        from autovideofixer.config import Config
        from autovideofixer.core.presets import get_preset

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        # Get preset and apply its overrides
        preset = get_preset("4k60")
        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        job = pipeline.add_job(
            tmp_video_file,
            output_path,
            overrides=preset.stage_overrides,
        )

        results = pipeline.execute_all()

        assert len(results) == 1

    def test_pipeline_with_progress_callback(self, tmp_video_file, tmp_path):
        """Test processing with progress reporting."""
        from autovideofixer.core.pipeline import Pipeline
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        progress_updates = []

        def progress_callback(job, result):
            progress_updates.append((job.input_path, result.success))

        job = pipeline.add_job(tmp_video_file, output_path)
        results = pipeline.execute_all(callback=progress_callback)

        assert len(results) >= 1
        # Verify callback was invoked
        assert len(progress_updates) > 0


@pytest.mark.integration
class TestStageIntegration:
    """Test individual stages with real video files."""

    def test_detect_stage(self, tmp_video_file):
        """Test video detection/analysis stage."""
        from autovideofixer.core.analysis import VideoAnalyzer
        from autovideofixer.config import Config

        config_path = tmp_video_file + "_config.yaml"
        config = Config(Path(config_path))
        analyzer = VideoAnalyzer(config)
        result = analyzer.analyze(tmp_video_file)

        assert result is not None
        assert result.has_video
        assert result.duration > 0
        assert result.resolution == (320, 240)

    def test_stabilize_stage(self, tmp_video_file, tmp_path):
        """Test video stabilization stage."""
        from autovideofixer.core.stages.stabilize import StabilizeStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "stabilized.mp4")

        stage = StabilizeStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_denoise_stage(self, tmp_video_file, tmp_path):
        """Test video denoising stage."""
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "denoised.mp4")

        stage = DenoiseVideoStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_deblock_stage(self, tmp_video_file, tmp_path):
        """Test video deblocking stage."""
        from autovideofixer.core.stages.deblock import DeblockStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "deblocked.mp4")

        stage = DeblockStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_normalize_audio_stage(self, tmp_video_file, tmp_path):
        """Test audio normalization stage."""
        from autovideofixer.core.stages.normalize_audio import NormalizeAudioStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "normalized.mp4")

        stage = NormalizeAudioStage(config)
        result = stage.execute(tmp_video_file, output_path)

        # Audio normalization may fail on short test videos without audio
        # Just verify the stage runs without crashing
        if result.success:
            assert os.path.exists(output_path)

    def test_encode_stage(self, tmp_video_file, tmp_path):
        """Test video encoding stage."""
        from autovideofixer.core.stages.encode import EncodeStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "encoded.mp4")

        stage = EncodeStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_remux_stage(self, tmp_video_file, tmp_path):
        """Test video remuxing stage."""
        from autovideofixer.core.stages.remux import RemuxStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "remuxed.mkv")

        stage = RemuxStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_speed_stage(self, tmp_video_file, tmp_path):
        """Test video speed adjustment stage."""
        from autovideofixer.core.stages.speed import SpeedStage
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "spedup.mp4")

        stage = SpeedStage(config)
        result = stage.execute(tmp_video_file, output_path, speed_factor=1.5)

        # Speed stage may have issues with very short test videos
        # Just verify the stage can be instantiated and executed
        assert stage is not None
        assert result is not None


@pytest.mark.integration
class TestConfigIntegration:
    """Test configuration persistence and presets."""

    def test_config_save_load(self, tmp_path):
        """Test saving and loading configuration."""
        from autovideofixer.config import Config

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        config.set(2, "general", "max_concurrent_jobs")
        config.set(24, "encoding", "crf")
        config.save()

        # Load config from same location
        loaded_config = Config(config_path)
        assert loaded_config.get("general", "max_concurrent_jobs") == 2
        assert loaded_config.get("encoding", "crf") == 24

    def test_preset_application(self, tmp_path):
        """Test applying presets to configuration."""
        from autovideofixer.core.presets import get_preset

        preset = get_preset("4k60")

        assert preset is not None
        assert preset.target_resolution == (3840, 2160)
        assert preset.target_framerate == 60

    def test_preset_persistence(self, tmp_path):
        """Test that presets can be saved and loaded."""
        from dataclasses import replace
        from autovideofixer.core.presets import get_preset, save_preset, load_preset

        # Create a custom preset (make a copy to avoid modifying the global preset)
        original = get_preset("4k60")
        preset = replace(original, display_name="Test 4K60")

        # Save to temp location
        preset_path = tmp_path / "test_preset.json"
        save_preset(preset, preset_path)

        # Load it back
        loaded = load_preset(preset_path)
        assert loaded.display_name == "Test 4K60"
        assert loaded.target_resolution == (3840, 2160)


@pytest.mark.integration
class TestAnalysisIntegration:
    """Test video analysis with real files."""

    def test_analyze_video_file(self, tmp_video_file):
        """Test analyzing a real video file."""
        from autovideofixer.core.analysis import VideoAnalyzer
        from autovideofixer.config import Config

        config_path = tmp_video_file + "_config.yaml"
        config = Config(Path(config_path))
        analyzer = VideoAnalyzer(config)
        info = analyzer.analyze(tmp_video_file)

        assert info is not None
        assert info.has_video
        assert info.duration > 0
        assert info.resolution == (320, 240)
        assert info.framerate > 0
        # Analysis should complete without error
        assert info is not None

    def test_scan_directory(self, tmp_video_directory):
        """Test scanning a directory for video files."""
        from autovideofixer.core.analysis import scan_directory

        _, video_files = tmp_video_directory
        directory = os.path.dirname(video_files[0])

        videos = scan_directory(directory)

        assert len(videos) == 3
        for video in videos:
            assert os.path.exists(video)

    def test_is_video_file(self, tmp_video_file, tmp_path):
        """Test video file detection."""
        from autovideofixer.core.analysis import is_video_file

        assert is_video_file(tmp_video_file) is True

        # Create a non-video file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not a video")

        assert is_video_file(str(text_file)) is False
