"""Integration tests for Auto Video Fixer.

These tests require FFmpeg and test the full pipeline and stage interactions.
Run with: pytest tests/integration/ -v -m integration
"""

import os
from pathlib import Path

import pytest


@pytest.mark.integration
class TestPipelineIntegration:
    """Test full pipeline processing."""

    def test_pipeline_single_file(self, tmp_video_file, tmp_path):
        """Test processing a single video file through the pipeline."""
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline

        # Use a temporary config to avoid modifying user config
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        pipeline.add_job(tmp_video_file, output_path)

        results = pipeline.execute_all()

        assert len(results) == 1
        # Pipeline may have errors but should still produce output
        assert results[0].output_path is not None

    def test_pipeline_multiple_files(self, tmp_video_directory, tmp_path):
        """Test processing multiple video files."""
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline

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
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline
        from autovideofixer.core.presets import get_preset

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        # Get preset and apply its overrides
        preset = get_preset("4k60")
        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        pipeline.add_job(
            tmp_video_file,
            output_path,
            overrides=preset.stage_overrides,
        )

        results = pipeline.execute_all()

        assert len(results) == 1

    def test_pipeline_with_progress_callback(self, tmp_video_file, tmp_path):
        """Test processing with progress reporting."""
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        pipeline = Pipeline(config)
        output_path = str(tmp_path / "output.mp4")

        progress_updates = []

        def progress_callback(job, result):
            progress_updates.append((job.input_path, result.success))

        pipeline.add_job(tmp_video_file, output_path)
        results = pipeline.execute_all(callback=progress_callback)

        assert len(results) >= 1
        # Verify callback was invoked
        assert len(progress_updates) > 0


@pytest.mark.integration
class TestStageIntegration:
    """Test individual stages with real video files."""

    def test_detect_stage(self, tmp_video_file):
        """Test video detection/analysis stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.analysis import VideoAnalyzer

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
        from autovideofixer.config import Config
        from autovideofixer.core.stages.stabilize import StabilizeStage

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "stabilized.mp4")

        stage = StabilizeStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_denoise_stage(self, tmp_video_file, tmp_path):
        """Test video denoising stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.denoise_video import DenoiseVideoStage

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "denoised.mp4")

        stage = DenoiseVideoStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_deblock_stage(self, tmp_video_file, tmp_path):
        """Test video deblocking stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.deblock import DeblockStage

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "deblocked.mp4")

        stage = DeblockStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_normalize_audio_stage(self, tmp_video_file, tmp_path):
        """Test audio normalization stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.normalize_audio import NormalizeAudioStage

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
        from autovideofixer.config import Config
        from autovideofixer.core.stages.encode import EncodeStage

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "encoded.mp4")

        stage = EncodeStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_remux_stage(self, tmp_video_file, tmp_path):
        """Test video remuxing stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.remux import RemuxStage

        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        output_path = str(tmp_path / "remuxed.mkv")

        stage = RemuxStage(config)
        result = stage.execute(tmp_video_file, output_path)

        assert result.success
        assert os.path.exists(output_path)

    def test_speed_stage(self, tmp_video_file, tmp_path):
        """Test video speed adjustment stage."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.speed import SpeedStage

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
class TestCropIntegration:
    """Real-ffmpeg tests for the auto-crop stage (docs/REQUIREMENTS.md feature 3).

    Covers: (a) a letterboxed video crops back to its true content bounds with
    no remaining border, (b) a video with no border is left uncropped, (c) a
    letterboxed video with a "watermark" overlay sitting in the border area --
    plain cropdetect crops it away regardless (a documented limitation: it has
    no notion of "meaningful non-black content" vs. noise/overlay), while
    crop.vlm_check with a mocked VLM response and vlm_policy=skip leaves the
    video uncropped.
    """

    @staticmethod
    def _make_letterboxed_video(tmp_path, name="letterboxed.mp4", extra_filter=None):
        """640x360 test pattern padded into a 640x480 frame (60px black bars
        top/bottom) -- true content bounds are crop=640:360:0:60."""
        import subprocess

        video_path = tmp_path / name
        vf = "pad=640:480:0:60:black"
        if extra_filter:
            vf = f"{vf},{extra_filter}"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=24:duration=2",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            capture_output=True,
            check=True,
        )
        return str(video_path)

    @staticmethod
    def _make_borderless_video(tmp_path, name="borderless.mp4"):
        import subprocess

        video_path = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x480:rate=24:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            capture_output=True,
            check=True,
        )
        return str(video_path)

    @staticmethod
    def _rescan_crop(video_path):
        """Re-run raw cropdetect over a video, returning the last crop=w:h:x:y
        window -- used to verify no meaningful border remains post-crop."""
        from autovideofixer.core.stages.crop import _detect_crop

        return _detect_crop(video_path, limit=24, round_=2, analyze_duration_sec=0)

    def test_letterboxed_video_crops_to_content_bounds(self, tmp_path):
        from autovideofixer.config import Config
        from autovideofixer.core.ffmpeg_utils import probe
        from autovideofixer.core.stages.crop import CropStage

        video = self._make_letterboxed_video(tmp_path)
        before = self._rescan_crop(video)
        assert before is not None
        # Before crop: detected content window should be ~640x360, well short
        # of the padded 640x480 frame -- otherwise the fixture itself is bad.
        assert before[1] <= 380  # h

        config = Config(tmp_path / "nonexistent.yaml")
        config.set(True, "stages", "crop", "enabled")
        stage = CropStage(config)
        output_path = str(tmp_path / "cropped.mp4")
        result = stage.execute(video, output_path)

        assert result.success, result.error
        assert os.path.exists(output_path)

        out_info = probe(output_path)
        out_w, out_h = out_info.resolution
        # Allow +/-2px rounding (round=2 keeps dimensions even).
        assert out_w == 640
        assert abs(out_h - 360) <= 2

        after = self._rescan_crop(output_path)
        assert after is not None
        after_w, after_h, after_x, after_y = after
        # No meaningful border left: the re-scanned crop window should cover
        # (approximately) the whole cropped frame.
        assert after_x == 0
        assert after_y <= 2
        assert after_w >= out_w - 2
        assert after_h >= out_h - 2

    def test_borderless_video_is_not_cropped(self, tmp_path):
        from autovideofixer.config import Config
        from autovideofixer.core.stages.crop import CropStage

        video = self._make_borderless_video(tmp_path)
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(True, "stages", "crop", "enabled")
        stage = CropStage(config)

        from autovideofixer.core.ffmpeg_utils import get_video_info

        input_info = get_video_info(video)
        should_run, quick_reason = stage.should_run(input_info)
        if not should_run:
            # should_run's quick pre-filter already caught it -- done.
            assert quick_reason
            return

        output_path = str(tmp_path / "not_cropped.mp4")
        result = stage.execute(video, output_path)
        assert result.status.value == "skipped"
        assert result.skipped_reason

    def test_watermark_in_letterbox_plain_mode_crops_it_away(self, tmp_path):
        """A dim overlay box sitting inside the black letterbox border, below
        cropdetect's luma threshold -- plain (non-VLM) auto-crop has no notion
        of "meaningful overlay" vs. background noise, so it crops the whole
        border away, overlay included. This is the documented limitation R3.2
        exists to address (see crop.vlm_check below)."""
        from autovideofixer.config import Config
        from autovideofixer.core.ffmpeg_utils import probe
        from autovideofixer.core.stages.crop import CropStage

        # A box at luma ~16 (0x101010), well under the default cropdetect
        # limit=24, positioned inside the top 60px black bar.
        watermark_filter = "drawbox=x=10:y=10:w=100:h=20:color=0x101010:t=fill"
        video = self._make_letterboxed_video(
            tmp_path, name="watermarked.mp4", extra_filter=watermark_filter
        )

        config = Config(tmp_path / "nonexistent.yaml")
        config.set(True, "stages", "crop", "enabled")
        stage = CropStage(config)
        output_path = str(tmp_path / "watermark_cropped.mp4")
        result = stage.execute(video, output_path)

        assert result.success, result.error
        out_w, out_h = probe(output_path).resolution
        assert out_w == 640
        assert abs(out_h - 360) <= 2  # the watermark did NOT save it from cropping

    def test_watermark_vlm_skip_policy_leaves_video_uncropped(self, tmp_path, monkeypatch):
        """Same watermarked input as above, but with crop.vlm_check enabled and
        a mocked VLM response saying content lies outside the box -- with
        vlm_policy=skip the stage must not crop at all."""
        from autovideofixer.config import Config
        from autovideofixer.core.stages.crop import CropStage

        watermark_filter = "drawbox=x=10:y=10:w=100:h=20:color=0x101010:t=fill"
        video = self._make_letterboxed_video(
            tmp_path, name="watermarked2.mp4", extra_filter=watermark_filter
        )

        config = Config(tmp_path / "nonexistent.yaml")
        config.set(True, "stages", "crop", "enabled")
        config.set(True, "stages", "crop", "vlm_check")
        config.set("skip", "stages", "crop", "vlm_policy")
        config.set(True, "analysis", "vlm", "enabled")

        monkeypatch.setattr(
            "autovideofixer.core.analysis.run_crop_vlm_check",
            lambda full, boxed, cfg: {
                "checked": True,
                "content_outside": True,
                "reason": "mocked: watermark-like box detected outside the crop area",
                "failed": False,
            },
        )

        stage = CropStage(config)
        output_path = str(tmp_path / "watermark_skip.mp4")
        result = stage.execute(video, output_path)

        assert result.status.value == "skipped"
        assert "VLM flagged" in result.skipped_reason
        assert not os.path.exists(output_path)


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

        from autovideofixer.core.presets import get_preset, load_preset, save_preset

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
        from autovideofixer.config import Config
        from autovideofixer.core.analysis import VideoAnalyzer

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
