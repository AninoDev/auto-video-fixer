"""Tests for pipeline engine."""

import os
import tempfile

import pytest

from autovideofixer.config import Config
from autovideofixer.core.pipeline import JobResult, Pipeline, PipelineStatus
from autovideofixer.core.stages.base import StageResult, StageStatus


class TestPipeline:
    """Test pipeline orchestration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.pipeline = Pipeline(self.config)

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        assert self.pipeline is not None
        assert len(self.pipeline.jobs) == 0
        assert self.pipeline.running is False

    def test_add_job(self, tmp_path):
        """Test adding a job to the pipeline."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video content")

        job = self.pipeline.add_job(str(test_file))

        assert len(self.pipeline.jobs) == 1
        assert job.input_path == str(test_file)
        assert job.status == PipelineStatus.IDLE
        assert job.is_queued is True

    def test_add_job_with_custom_output(self, tmp_path):
        """Test adding a job with custom output path."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video content")
        output_file = tmp_path / "output.mp4"

        job = self.pipeline.add_job(str(test_file), str(output_file))

        assert job.output_path == str(output_file)

    def test_add_job_nonexistent_file(self):
        """Test adding a job with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.pipeline.add_job("/nonexistent/file.mp4")

    def test_add_files(self, tmp_path):
        """Test adding multiple files."""
        files = []
        for i in range(3):
            f = tmp_path / f"test_{i}.mp4"
            f.write_text(f"fake video {i}")
            files.append(str(f))

        jobs = self.pipeline.add_files(files)

        assert len(jobs) == 3
        assert len(self.pipeline.jobs) == 3

    def test_add_files_from_directory(self, tmp_path):
        """Test adding files from a directory."""
        # Create video files
        for i in range(3):
            (tmp_path / f"video_{i}.mp4").write_text(f"video {i}")

        # Create non-video file
        (tmp_path / "readme.txt").write_text("not a video")

        jobs = self.pipeline.add_files([str(tmp_path)])

        assert len(jobs) == 3

    def test_clear_queue(self):
        """Test clearing the job queue."""
        # Add some jobs first
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            test_file = f.name

        try:
            self.pipeline.add_job(test_file)
            self.pipeline.add_job(test_file)
            assert len(self.pipeline.jobs) == 2

            self.pipeline.clear_queue()
            assert len(self.pipeline.jobs) == 0
        finally:
            os.unlink(test_file)

    def test_cancel(self):
        """Test canceling the pipeline."""
        self.pipeline.cancel()
        assert self.pipeline._cancel_requested is True

    def test_auto_determine_stages(self, tmp_path):
        """Test automatic stage determination."""
        import subprocess

        test_file = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(test_file),
            ],
            capture_output=True,
            check=True,
        )

        job = self.pipeline.add_job(str(test_file))

        # Configure quality targets
        self.config.set([3840, 2160], "quality", "quality_target", "target_resolution")
        self.config.set(60.0, "quality", "quality_target", "target_framerate")

        stages = self.pipeline.auto_determine_stages(job)

        assert isinstance(stages, list)
        assert "detect" in stages
        assert "encode" in stages

    def test_optimize_stage_order(self):
        """Test stage ordering optimization."""
        stages = ["encode", "upscale", "stabilize", "denoise_video"]
        ordered = self.pipeline.optimize_stage_order(stages)

        # Check ordering constraints
        assert ordered.index("stabilize") < ordered.index("upscale"), (
            "Stabilize should come before upscale"
        )
        assert ordered.index("denoise_video") < ordered.index("upscale"), (
            "Denoise should come before upscale"
        )
        assert ordered[-1] == "encode", "Encode should be last"

    def test_job_result_creation(self):
        """Test creating a job result."""
        result = JobResult(
            input_path="/input.mp4",
            output_path="/output.mp4",
            success=True,
        )

        assert result.success is True
        assert result.all_stages_passed is True
        assert result.output_size == 0  # File doesn't exist in test

    def test_job_result_with_failures(self):
        """Test job result with stage failures."""
        result = JobResult(
            input_path="/input.mp4",
            stage_results={
                "detect": StageResult(status=StageStatus.COMPLETED),
                "encode": StageResult(status=StageStatus.FAILED, error="Test error"),
            },
            errors=["encode: Test error"],
            success=False,
        )

        assert result.success is False
        assert result.all_stages_passed is False


class TestPipelineExecution:
    """Test pipeline execution (requires FFmpeg)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.pipeline = Pipeline(self.config)

    @pytest.mark.integration
    def test_execute_single_job(self, tmp_path):
        """Test executing a single job (integration test)."""
        # Create a simple test video using FFmpeg
        test_file = tmp_path / "test.mp4"
        output_file = tmp_path / "output.mp4"

        # Use ffmpeg to create a simple test video
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(test_file),
            ],
            capture_output=True,
            check=True,
        )

        job = self.pipeline.add_job(str(test_file), str(output_file))

        # Run with minimal stages
        job.stages = ["detect", "encode"]
        result = self.pipeline.execute_job(job)

        assert result is not None
        assert result.success is True
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    @pytest.mark.integration
    def test_execute_multiple_jobs(self, tmp_path):
        """Test executing multiple jobs (integration test)."""
        files = []
        for i in range(2):
            test_file = tmp_path / f"test_{i}.mp4"
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=1:size=320x240:rate=24",
                    "-c:v",
                    "libx264",
                    str(test_file),
                ],
                capture_output=True,
                check=True,
            )
            files.append(str(test_file))

        for i, f in enumerate(files):
            job = self.pipeline.add_job(f, f.replace(".mp4", "_out.mp4"))
            job.stages = ["detect", "stabilize", "deblock", "denoise_video", "encode"]

        results = self.pipeline.execute_all()

        assert len(results) == 2
        for result in results:
            assert result.success is True
