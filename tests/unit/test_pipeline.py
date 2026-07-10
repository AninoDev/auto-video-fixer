"""Tests for pipeline engine."""

import os
import tempfile
from pathlib import Path

import pytest

from autovideofixer.config import Config
from autovideofixer.core.pipeline import JobResult, Pipeline, PipelineStatus
from autovideofixer.core.stages.base import StageResult, StageStatus


class TestPipeline:
    """Test pipeline orchestration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
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
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
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


class TestSkippedTerminalStagePromotion:
    """A completed intermediate stage followed by a skipped terminal stage must
    have its temp output promoted to job.output_path, not deleted out from under
    the JobResult by the unconditional temp cleanup (regression test)."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def test_skipped_terminal_stage_promotes_temp_to_output(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.base import BaseStage

        class ProduceStage(BaseStage):
            name = "fake_produce"
            display_name = "Fake Produce"
            description = "test-only stage that writes real output"
            category = "test"
            priority = 10
            produces_output = True

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"produced content")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        class SkipStage(BaseStage):
            name = "fake_skip"
            display_name = "Fake Skip"
            description = "test-only stage that always skips"
            category = "test"
            priority = 20
            produces_output = True

            def should_run(self, input_info):
                return False, "not needed for this input"

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                raise AssertionError("SkipStage.execute should never be called")

        stage_registry = {"fake_produce": ProduceStage, "fake_skip": SkipStage}

        def fake_get_stage(name):
            return stage_registry.get(name)

        def fake_create_stage(name, config):
            cls = stage_registry.get(name)
            return cls(config) if cls else None

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_video_info",
            lambda path: {"resolution": (320, 240), "framerate": 30.0, "duration": 1.0},
        )

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "source.mp4"
        input_file.write_bytes(b"fake source video")

        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_produce", "fake_skip"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert result.output_path == job.output_path
        assert os.path.exists(job.output_path)
        with open(job.output_path, "rb") as f:
            assert f.read() == b"produced content"

        # No .avf_* intermediate temp files left behind anywhere near the input.
        for f in os.listdir(input_dir):
            assert not f.startswith(".avf_"), f"orphaned temp file: {f}"


class TestExplicitStageBypassesDisabledFlag:
    """--stage's help text says it "replaces the preset/auto-determined list", so a
    stage named explicitly must run even if stages.<name>.enabled is False --
    should_run() must not silently skip it. The auto-determined/preset path (no
    explicit job.stages) must keep honoring the enabled flag as before."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def _register_fake_stage(self, monkeypatch):
        from autovideofixer.core.stages.base import BaseStage

        class FakeToggleStage(BaseStage):
            name = "fake_toggle"
            display_name = "Fake Toggle"
            description = "test-only stage relying on the default should_run/is_enabled"
            category = "test"
            priority = 10
            produces_output = True

            # should_run is intentionally NOT overridden here -- it uses
            # BaseStage.should_run(), which is exactly what gates on is_enabled().

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"ran")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        def fake_get_stage(name):
            return FakeToggleStage if name == "fake_toggle" else None

        def fake_create_stage(name, config):
            return FakeToggleStage(config) if name == "fake_toggle" else None

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_video_info",
            lambda path: {"resolution": (320, 240), "framerate": 30.0, "duration": 1.0},
        )

    def test_explicit_stage_list_bypasses_disabled_flag(self, tmp_path, monkeypatch):
        self._register_fake_stage(monkeypatch)
        self.config.set(False, "stages", "fake_toggle", "enabled")

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_toggle"]  # explicit --stage-style request

        result = self.pipeline.execute_job(job)

        assert "fake_toggle" not in result.skipped
        assert result.stage_results["fake_toggle"].status == StageStatus.COMPLETED
        assert result.success is True

    def test_auto_determined_path_still_honors_disabled_flag(self, tmp_path, monkeypatch):
        self._register_fake_stage(monkeypatch)
        self.config.set(False, "stages", "fake_toggle", "enabled")
        # Simulate the preset/auto-determined path choosing this stage -- job.stages
        # itself is left empty so execute_job takes the auto-determination branch.
        monkeypatch.setattr(self.pipeline, "auto_determine_stages", lambda job: ["fake_toggle"])

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))

        result = self.pipeline.execute_job(job)

        assert "fake_toggle" in result.skipped
        assert result.stage_results["fake_toggle"].status == StageStatus.SKIPPED
        assert (
            result.stage_results["fake_toggle"].skipped_reason == "Stage disabled in configuration"
        )
