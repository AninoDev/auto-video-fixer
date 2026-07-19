"""Tests for pipeline engine."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_auto_determine_stages_includes_upscale_when_already_at_target(
        self, tmp_path, monkeypatch
    ):
        """auto_determine_stages() must NOT geometry-gate "upscale"'s plan
        membership against the up-front probe -- a later stage (crop) can
        shrink the frame after this plan is built, and this up-front check
        is also orientation-blind. So even an input that's already AT (or
        above) target_resolution per this raw, un-rotated comparison must
        still get "upscale" included in the plan; the in-loop, orientation-
        aware, freshly-reprobed should_run() is the actual authority on
        whether it runs (regression test for the "upscale absent from the
        plan entirely" half of the post-crop upscale bug)."""
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_video_info",
            lambda path: {
                "resolution": (1920, 1080),
                "framerate": 30.0,
                "is_hdr": False,
            },
        )
        test_file = tmp_path / "in.mp4"
        test_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(test_file))
        self.config.set([1920, 1080], "quality", "quality_target", "target_resolution")

        stages = self.pipeline.auto_determine_stages(job)

        assert "upscale" in stages

    def test_auto_determine_stages_excludes_upscale_without_target_resolution(
        self, tmp_path, monkeypatch
    ):
        """Behavior to preserve: with no target_resolution configured at
        all, "upscale" must still stay OUT of the auto-determined plan."""
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_video_info",
            lambda path: {
                "resolution": (320, 240),
                "framerate": 30.0,
                "is_hdr": False,
            },
        )
        test_file = tmp_path / "in.mp4"
        test_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(test_file))
        # No quality.quality_target.target_resolution set -- default is None.

        stages = self.pipeline.auto_determine_stages(job)

        assert "upscale" not in stages

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

        def fake_create_stage(name, config, overrides=None):
            cls = stage_registry.get(name)
            return cls(config, overrides) if cls else None

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

        def fake_create_stage(name, config, overrides=None):
            return FakeToggleStage(config, overrides) if name == "fake_toggle" else None

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


class TestResolveStageOrder:
    """pipeline.default_order now actually drives order/omission/repetition
    (see AGENTS.md's "Pipeline Behavior" -- config.py's DEFAULTS stays plain
    strings; the mapping-entry forms below are opt-in). These tests exercise
    Pipeline.resolve_stage_order() directly -- no FFmpeg or stage registry
    involved, since the ordering logic never touches either."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def test_deblock_before_stabilize_default(self):
        """New DEFAULTS order: deblock moves before stabilize."""
        entries = self.pipeline.resolve_stage_order(["detect", "deblock", "stabilize", "encode"])
        labels = [e.label for e in entries]
        assert labels.index("deblock") < labels.index("stabilize")

    def test_encode_stays_last(self):
        entries = self.pipeline.resolve_stage_order(
            ["encode", "upscale", "stabilize", "denoise_video"]
        )
        assert entries[-1].name == "encode"

    def test_order_from_config_respected(self):
        """A custom pipeline.default_order is honored, not just the hardcoded fallback."""
        self.config.set(["upscale", "deblock", "detect", "encode"], "pipeline", "default_order")
        entries = self.pipeline.resolve_stage_order(["detect", "deblock", "upscale", "encode"])
        assert [e.name for e in entries] == ["upscale", "deblock", "detect", "encode"]

    def test_stage_not_mentioned_in_order_list_is_appended_but_stays_before_encode(self):
        """Plain-string compat: a requested stage entirely absent from
        default_order is appended at the end, but "encode" must remain last."""
        self.config.set(["detect", "encode"], "pipeline", "default_order")
        entries = self.pipeline.resolve_stage_order(["detect", "speed", "encode"])
        labels = [e.label for e in entries]
        assert labels == ["detect", "speed", "encode"]

    def test_enabled_false_hard_drops_even_when_requested(self):
        """The ONLY way to hard-drop a stage that's otherwise requested/enabled
        -- omitting it from default_order entirely still gets it appended
        (see the compat test above)."""
        self.config.set(
            ["detect", {"stage": "deblock", "enabled": False}, "encode"],
            "pipeline",
            "default_order",
        )
        entries = self.pipeline.resolve_stage_order(["detect", "deblock", "encode"])
        assert [e.name for e in entries] == ["detect", "encode"]

    def test_enabled_true_forces_run_even_when_not_requested(self):
        self.config.set(
            ["detect", {"stage": "deblock", "enabled": True}, "encode"],
            "pipeline",
            "default_order",
        )
        entries = self.pipeline.resolve_stage_order(["detect", "encode"])  # deblock NOT requested
        assert [e.name for e in entries] == ["detect", "deblock", "encode"]
        deblock_entry = next(e for e in entries if e.name == "deblock")
        assert deblock_entry.forced is True

    def test_enabled_null_defers_to_global_gating(self):
        self.config.set(
            ["detect", {"stage": "deblock", "enabled": None}, "encode"],
            "pipeline",
            "default_order",
        )
        # Not requested -> does not run.
        entries = self.pipeline.resolve_stage_order(["detect", "encode"])
        assert [e.name for e in entries] == ["detect", "encode"]
        # Requested -> runs.
        entries = self.pipeline.resolve_stage_order(["detect", "deblock", "encode"])
        assert [e.name for e in entries] == ["detect", "deblock", "encode"]

    def test_repetition_produces_occurrence_qualified_labels(self):
        self.config.set(
            ["detect", "deblock", {"stage": "deblock"}, "encode"],
            "pipeline",
            "default_order",
        )
        entries = self.pipeline.resolve_stage_order(["detect", "deblock", "encode"])
        assert [e.label for e in entries] == ["detect", "deblock", "deblock#2", "encode"]
        assert [e.name for e in entries] == ["detect", "deblock", "deblock", "encode"]

    def test_single_occurrence_label_has_no_suffix(self):
        """The common case (no repeats) uses the plain stage name everywhere,
        no "#1" churn."""
        entries = self.pipeline.resolve_stage_order(["detect", "deblock"])
        labels = [e.label for e in entries]
        assert "deblock" in labels
        assert "deblock#1" not in labels

    def test_per_occurrence_config_overrides_carried_on_entry(self):
        self.config.set(
            [
                {"stage": "deblock", "config": {"strength": "low"}},
                {"stage": "deblock", "config": {"strength": "high"}},
            ],
            "pipeline",
            "default_order",
        )
        entries = self.pipeline.resolve_stage_order(["deblock"])
        assert entries[0].overrides == {"strength": "low"}
        assert entries[1].overrides == {"strength": "high"}
        assert entries[0].label == "deblock"
        assert entries[1].label == "deblock#2"

    def test_malformed_entry_mapping_without_stage_key_errors(self):
        self.config.set([{"enabled": True}], "pipeline", "default_order")
        with pytest.raises(ValueError):
            self.pipeline.resolve_stage_order(["detect"])

    def test_malformed_entry_non_str_non_mapping_errors(self):
        self.config.set([123], "pipeline", "default_order")
        with pytest.raises(ValueError):
            self.pipeline.resolve_stage_order(["detect"])

    def test_malformed_entry_bad_enabled_type_errors(self):
        self.config.set([{"stage": "deblock", "enabled": "yes"}], "pipeline", "default_order")
        with pytest.raises(ValueError):
            self.pipeline.resolve_stage_order(["deblock"])

    def test_malformed_entry_bad_config_type_errors(self):
        self.config.set(
            [{"stage": "deblock", "config": "not-a-mapping"}], "pipeline", "default_order"
        )
        with pytest.raises(ValueError):
            self.pipeline.resolve_stage_order(["deblock"])

    def test_optimize_stage_order_still_returns_plain_labels(self):
        """optimize_stage_order() stays the backward-compatible flattened view."""
        ordered = self.pipeline.optimize_stage_order(
            ["encode", "upscale", "stabilize", "denoise_video"]
        )
        assert isinstance(ordered, list)
        assert all(isinstance(s, str) for s in ordered)
        assert ordered[-1] == "encode"

    def test_per_occurrence_timeout_override_reaches_run_ffmpeg(self, tmp_path):
        """pipeline.default_order's per-occurrence ``config: {timeout: ...}``
        (StageOrderEntry.overrides) is the ONLY new machinery the timeout
        feature needs at the occurrence level -- BaseStage.__init__ already
        deep-merges it onto stages.<name> for that instance. This proves the
        override actually reaches a real stage's resolved timeout AND the
        ffmpeg call it makes, end-to-end through resolve_stage_order() +
        create_stage(), not just BaseStage.stage_timeout() in isolation.
        """
        from autovideofixer.core.stages.base import create_stage

        self.config.set(
            [{"stage": "hdr", "config": {"timeout": 1200}}],
            "pipeline",
            "default_order",
        )
        entries = self.pipeline.resolve_stage_order(["hdr"])
        assert entries[0].overrides == {"timeout": 1200}

        stage = create_stage(entries[0].name, self.config, entries[0].overrides)
        assert stage.stage_timeout() == 1200

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run_ffmpeg:
            mock_run_ffmpeg.return_value = MagicMock(returncode=0, stderr="")
            stage.execute("in.mp4", output_path=str(tmp_path / "out.mp4"))

        assert mock_run_ffmpeg.call_args.kwargs["timeout"] == 1200


class TestOccurrenceAwareExecution:
    """Full execute_job() runs exercising per-occurrence semantics: forced
    enabled/disabled bypassing stages.<name>.enabled, repetition with
    occurrence-unique temp filenames, and per-occurrence config overrides
    reaching the stage instance/kwargs. Uses fake in-process stages (no
    FFmpeg needed), following the pattern of TestExplicitStageBypassesDisabledFlag
    above."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def _register_fakes(self, monkeypatch):
        from autovideofixer.core.stages.base import BaseStage

        calls: list[dict] = []

        class RecordingStage(BaseStage):
            name = "fake_record"
            display_name = "Fake Record"
            description = "test-only stage recording its own effective config per call"
            category = "test"
            priority = 10
            produces_output = True

            def __init__(self, config, overrides=None):
                super().__init__(config, overrides)
                self._marker = self._stage_config.get("marker", "default")

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                calls.append(
                    {
                        "marker": self._marker,
                        "kwargs": dict(kwargs),
                        "output_path": output_path,
                    }
                )
                with open(output_path, "wb") as f:
                    f.write(self._marker.encode())
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        class TerminalStage(BaseStage):
            name = "fake_final"
            display_name = "Fake Final"
            description = "test-only terminal stage"
            category = "test"
            priority = 90
            produces_output = True

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"final")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        registry = {"fake_record": RecordingStage, "fake_final": TerminalStage}

        def fake_get_stage(name):
            return registry.get(name)

        def fake_create_stage(name, config, overrides=None):
            cls = registry.get(name)
            return cls(config, overrides) if cls else None

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_video_info",
            lambda path: {"resolution": (320, 240), "framerate": 30.0, "duration": 1.0},
        )
        return calls

    def test_repetition_runs_twice_with_occurrence_unique_temp_names(self, tmp_path, monkeypatch):
        calls = self._register_fakes(monkeypatch)
        self.config.set(["fake_record", "fake_record", "fake_final"], "pipeline", "default_order")

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_record", "fake_final"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert len(calls) == 2
        assert "fake_record" in result.stage_results
        assert "fake_record#2" in result.stage_results
        # The second occurrence's temp output must be a distinct path from
        # the first's (occurrence-qualified temp filename).
        first_out = calls[0]["output_path"]
        second_out = calls[1]["output_path"]
        assert first_out != second_out
        assert "fake_record#2" in os.path.basename(second_out)

    def test_per_occurrence_config_merges_over_stages_and_job_overrides(
        self, tmp_path, monkeypatch
    ):
        calls = self._register_fakes(monkeypatch)
        self.config.set({"marker": "base"}, "stages", "fake_record")
        self.config.set(
            [
                {"stage": "fake_record", "config": {"marker": "occurrence-1"}},
                {"stage": "fake_record", "config": {"marker": "occurrence-2"}},
                "fake_final",
            ],
            "pipeline",
            "default_order",
        )

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_record", "fake_final"]
        # job.stage_overrides should be superseded by each occurrence's own config.
        job.stage_overrides["fake_record"] = {"marker": "job-level"}

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert [c["marker"] for c in calls] == ["occurrence-1", "occurrence-2"]

    def test_enabled_true_bypasses_stages_enabled_false_for_that_occurrence(
        self, tmp_path, monkeypatch
    ):
        calls = self._register_fakes(monkeypatch)
        self.config.set(False, "stages", "fake_record", "enabled")
        self.config.set(
            [{"stage": "fake_record", "enabled": True}, "fake_final"],
            "pipeline",
            "default_order",
        )

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        # fake_record deliberately NOT in job.stages -- only reachable via
        # default_order's forced enabled: true.
        job.stages = ["fake_final"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert len(calls) == 1
        assert "fake_record" not in result.skipped
        assert result.stage_results["fake_record"].status == StageStatus.COMPLETED

    def test_max_stages_counts_occurrences(self, tmp_path, monkeypatch):
        self._register_fakes(monkeypatch)
        self.config.set(2, "pipeline", "max_stages")
        self.config.set(["fake_record", "fake_record", "fake_final"], "pipeline", "default_order")

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_record", "fake_final"]

        result = self.pipeline.execute_job(job)

        assert result.success is False
        assert any("max_stages" in e for e in result.errors)

    def test_quality_gate_honors_quality_timeout_config(self, tmp_path, monkeypatch):
        """The quality gate (execute_job()'s post-run estimate_ssim_psnr()
        call) must resolve and pass quality.timeout, not silently run
        unbounded/on some other stage's timeout."""
        self._register_fakes(monkeypatch)
        self.config.set("min", "quality", "quality_target", "mode")
        self.config.set(250, "quality", "timeout")
        self.config.set(["fake_record", "fake_final"], "pipeline", "default_order")

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_record", "fake_final"]

        with patch("autovideofixer.core.quality.estimate_ssim_psnr") as mock_estimate:
            from autovideofixer.core.quality import QualityResult

            mock_estimate.return_value = QualityResult(measurement_failed=False, score_override=99)
            result = self.pipeline.execute_job(job)

        assert result.success is True
        assert mock_estimate.call_args.kwargs["timeout"] == 250


class TestPerStageReprobe:
    """execute_job() must re-probe input_info after any stage that produces a
    NEW output file, so later stages' should_run()/execute() see the current
    geometry instead of a stale snapshot from before the job started
    (regression coverage for the post-crop upscale bug -- see CHANGELOG and
    AGENTS.md's "Pipeline Behavior")."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def _register_fakes(self, monkeypatch, probe_by_path):
        """probe_by_path: dict[path -> info dict] (or an Exception instance to
        raise), keyed by the exact path get_video_info() is called with, plus
        a "default" fallback used for the job's original input path."""
        from autovideofixer.core.stages.base import BaseStage

        should_run_calls: list[dict] = []

        class StageA(BaseStage):
            name = "fake_a"
            display_name = "Fake A"
            description = "test-only stage producing a new output file"
            category = "test"
            priority = 10
            produces_output = True

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"a-output")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        class StageASkip(BaseStage):
            """Variant of A that never produces output (should_run False)."""

            name = "fake_a"
            display_name = "Fake A Skip"
            description = "test-only stage that always skips"
            category = "test"
            priority = 10
            produces_output = True

            def should_run(self, input_info):
                return False, "nothing to do"

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                raise AssertionError("StageASkip.execute should never be called")

        class StageB(BaseStage):
            name = "fake_b"
            display_name = "Fake B"
            description = "test-only stage recording the input_info it receives"
            category = "test"
            priority = 20
            produces_output = True

            def should_run(self, input_info):
                should_run_calls.append(dict(input_info))
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"b-output")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        registry = {"fake_a": StageA, "fake_b": StageB}

        def fake_get_stage(name):
            return registry.get(name)

        def fake_create_stage(name, config, overrides=None):
            cls = registry.get(name)
            return cls(config, overrides) if cls else None

        _MISSING = object()

        def fake_get_video_info(path):
            info = probe_by_path.get(path, _MISSING)
            if info is _MISSING:
                info = probe_by_path.get("default", _MISSING)
            if isinstance(info, Exception):
                raise info
            assert info is not _MISSING, f"no mocked probe result for {path!r}"
            return dict(info)

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)

        return should_run_calls, registry

    def test_stage_b_sees_refreshed_resolution_from_stage_a_output(self, tmp_path, monkeypatch):
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")

        # StageA's own generated temp output path isn't known ahead of time,
        # so key the fake probe on "any path that isn't the original input" --
        # a-output geometry differs sharply from the original so a stale read
        # is unambiguous.
        original_info = {"resolution": (1920, 1080), "framerate": 30.0, "duration": 1.0}
        refreshed_info = {"resolution": (608, 1080), "framerate": 30.0, "duration": 1.0}

        # A custom mapping object whose .get() returns the ORIGINAL geometry
        # only for the exact original input path, and the REFRESHED geometry
        # for anything else (StageA's own generated temp output path isn't
        # known ahead of time, so this is keyed by exclusion rather than by
        # exact path).
        class _ProbeMap(dict):
            def get(self, key, default=None):
                if key == str(input_file):
                    return original_info
                if key in self:
                    return dict.get(self, key)
                return refreshed_info

        should_run_calls, _ = self._register_fakes(monkeypatch, _ProbeMap())
        self.config.set(["fake_a", "fake_b"], "pipeline", "default_order")

        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_a", "fake_b"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert len(should_run_calls) == 1
        assert should_run_calls[0]["resolution"] == (608, 1080)

    def test_injected_target_format_key_survives_reprobe(self, tmp_path, monkeypatch):
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")

        class _ProbeMap(dict):
            def get(self, key, default=None):
                return {"resolution": (320, 240), "framerate": 30.0, "duration": 1.0}

        should_run_calls, _ = self._register_fakes(monkeypatch, _ProbeMap())
        self.config.set(["fake_a", "fake_b"], "pipeline", "default_order")
        self.config.set("mkv", "general", "target_format")

        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_a", "fake_b"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        assert should_run_calls[0].get("target_format") == "mkv"

    def test_skipped_stage_produces_no_reprobe_call(self, tmp_path, monkeypatch):
        from autovideofixer.core.stages.base import BaseStage

        probe_calls: list[str] = []

        class SkipA(BaseStage):
            name = "fake_a"
            display_name = "Fake A Skip"
            description = "test-only stage that always skips (no output)"
            category = "test"
            priority = 10
            produces_output = True

            def should_run(self, input_info):
                return False, "nothing to do"

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                raise AssertionError("SkipA.execute should never be called")

        class RecordB(BaseStage):
            name = "fake_b"
            display_name = "Fake B"
            description = "test-only terminal stage"
            category = "test"
            priority = 20
            produces_output = True

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"b-output")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        registry = {"fake_a": SkipA, "fake_b": RecordB}

        def fake_get_stage(name):
            return registry.get(name)

        def fake_create_stage(name, config, overrides=None):
            cls = registry.get(name)
            return cls(config, overrides) if cls else None

        def fake_get_video_info(path):
            probe_calls.append(path)
            return {"resolution": (320, 240), "framerate": 30.0, "duration": 1.0}

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)

        self.config.set(["fake_a", "fake_b"], "pipeline", "default_order")

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_a", "fake_b"]

        result = self.pipeline.execute_job(job)

        assert result.success is True
        # Exactly two probes: the initial up-front probe of the job's
        # original input, and the re-probe after fake_b (which DOES produce
        # a new output, as the terminal stage). fake_a was skipped (no
        # output produced), so it must not have triggered a re-probe of its
        # own -- if it had, there would be three calls instead of two, with
        # the extra one repeating input_file back-to-back.
        assert probe_calls == [str(input_file), job.output_path]

    def test_reprobe_failure_logs_warning_and_keeps_previous_input_info(
        self, tmp_path, monkeypatch, caplog
    ):
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")

        original_info = {"resolution": (1920, 1080), "framerate": 30.0, "duration": 1.0}

        class _ProbeMap(dict):
            def get(self, key, default=None):
                if key == str(input_file):
                    return original_info
                raise RuntimeError("ffprobe exploded")

        should_run_calls, _ = self._register_fakes(monkeypatch, _ProbeMap())
        self.config.set(["fake_a", "fake_b"], "pipeline", "default_order")

        job = self.pipeline.add_job(str(input_file))
        job.stages = ["fake_a", "fake_b"]

        import logging

        with caplog.at_level(logging.WARNING):
            result = self.pipeline.execute_job(job)

        assert result.success is True
        # Re-probe failed -> stage B still gets the previous (original)
        # input_info, and the job continues rather than failing.
        assert should_run_calls[0]["resolution"] == (1920, 1080)
        assert any("re-probe" in r.getMessage().lower() for r in caplog.records)

    def test_upscale_should_run_true_with_refreshed_info_false_with_stale(self):
        """End-to-end regression at the should_run() level for the user's
        reported case: a 1920x1080 input with pillarboxed 9:16 content is
        cropped down to 608x1080. The REAL UpscaleStage.should_run() must
        return True once given the refreshed (post-crop) resolution, and
        (demonstrating the fix matters) False if it were still given the
        stale pre-crop resolution."""
        from autovideofixer.core.stages.upscale import UpscaleStage

        config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        config.set([1920, 1080], "quality", "quality_target", "target_resolution")
        stage = UpscaleStage(config)

        stale_info = {"resolution": (1920, 1080), "framerate": 30.0}
        refreshed_info = {"resolution": (608, 1080), "framerate": 30.0}

        stale_should_run, stale_reason = stage.should_run(stale_info)
        assert stale_should_run is False
        assert stale_reason == "Already at target resolution"

        fresh_should_run, _ = stage.should_run(refreshed_info)
        assert fresh_should_run is True

    def test_crop_then_upscale_end_to_end_plan_and_reprobe(self, tmp_path, monkeypatch):
        """Full execute_job() regression for BOTH halves of the post-crop
        upscale bug together: the input's ORIGINAL resolution already equals
        quality_target.target_resolution (so the old plan-time geometry gate
        in auto_determine_stages() would have excluded "upscale" from
        job.stages entirely), and a real "crop" occurrence shrinks the frame
        before "upscale" runs (so even if "upscale" WERE planned, a stale
        should_run() would misjudge it). job.stages is left empty so
        auto_determine_stages() actually decides plan membership -- this is
        NOT calling auto_determine_stages() directly, it's the real
        execute_job() path a preset/no-`--stage` run takes.
        """
        from autovideofixer.core.stages.base import BaseStage

        should_run_calls: list[dict] = []

        class FakeCrop(BaseStage):
            name = "crop"
            display_name = "Fake Crop"
            description = "test-only stage simulating a real geometry-shrinking crop"
            category = "test"
            priority = 12
            produces_output = True

            def should_run(self, input_info):
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"cropped-output")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        class FakeUpscale(BaseStage):
            name = "upscale"
            display_name = "Fake Upscale"
            description = "test-only stage recording the input_info should_run() receives"
            category = "test"
            priority = 30
            produces_output = True

            def should_run(self, input_info):
                should_run_calls.append(dict(input_info))
                return True, None

            def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"upscaled-output")
                return StageResult(status=StageStatus.COMPLETED, output_path=output_path)

        registry = {"crop": FakeCrop, "upscale": FakeUpscale}

        def fake_get_stage(name):
            return registry.get(name)

        def fake_create_stage(name, config, overrides=None):
            cls = registry.get(name)
            return cls(config, overrides) if cls else None

        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"fake input")

        original_info = {
            "resolution": (1920, 1080),
            "framerate": 30.0,
            "duration": 1.0,
            "is_hdr": False,
        }
        # Any path other than the original input (i.e. crop's temp output,
        # or upscale's own output) is the post-crop, pillarboxed-content
        # geometry from the user's real repro.
        refreshed_info = {
            "resolution": (608, 1080),
            "framerate": 30.0,
            "duration": 1.0,
            "is_hdr": False,
        }

        def fake_get_video_info(path):
            if path == str(input_file):
                return dict(original_info)
            return dict(refreshed_info)

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)

        # Real default_order already places "crop" before "upscale"; only
        # unrecognized stage names (everything else auto_determine_stages()
        # would normally add) get logged as "Unknown stage" and skipped --
        # harmless for this test, which only cares about crop/upscale.
        self.config.set(True, "stages", "crop", "enabled")
        self.config.set([1920, 1080], "quality", "quality_target", "target_resolution")

        job = self.pipeline.add_job(str(input_file))
        # job.stages left empty -> execute_job() calls auto_determine_stages()
        # for real, exercising the plan-membership fix.

        result = self.pipeline.execute_job(job)

        assert result.success is True
        # Plan-membership fix: "upscale" was actually planned and ran, even
        # though the ORIGINAL (pre-crop) resolution already equalled target.
        assert "upscale" in job.stages
        assert "upscale" in result.stage_results
        assert result.stage_results["upscale"].status == StageStatus.COMPLETED
        # Re-probe fix: should_run() received the POST-CROP geometry, not
        # the original 1920x1080.
        assert len(should_run_calls) == 1
        assert should_run_calls[0]["resolution"] == (608, 1080)


_BARE_PROBE_INFO = {
    "resolution": (0, 0),
    "framerate": 0.0,
    "duration": 1.0,
    "video_codec": "",
    "audio_codecs": [],
    "format": "",
    "probe_stderr": "",
}


class _FakeEncodeStage:
    """Registered under "encode" in these tests -- always runs and writes
    real bytes to output_path, so a decision-path "proceed" outcome is
    directly observable (the file at job.output_path actually changes)."""

    name = "encode"
    display_name = "Fake Encode"
    description = "test-only stage that writes real output"
    category = "test"
    priority = 100
    produces_output = True
    supports_gpu = False

    def __init__(self, config, overrides=None):
        self.config = config
        self._force_enabled = False

    def should_run(self, input_info):
        return True, None

    def execute(self, input_path, output_path=None, progress_callback=None, **kwargs):
        from autovideofixer.core.stages.base import StageResult, StageStatus

        with open(output_path, "wb") as f:
            f.write(b"new content")
        return StageResult(status=StageStatus.COMPLETED, output_path=output_path)


class TestExistingOutputDecisionPath:
    """REQUIREMENTS.md § 6.1/6.2: the existing-output decision path in
    execute_job() -- SKIPPED (not FAILED) by default, optional spec-check
    against effective targets, and rename-or-overwrite reprocessing of a
    verified mismatch."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def _register_fakes(self, monkeypatch, probe_by_path):
        def fake_get_stage(name):
            return _FakeEncodeStage if name == "encode" else None

        def fake_create_stage(name, config, overrides=None):
            return _FakeEncodeStage(config, overrides) if name == "encode" else None

        def fake_get_video_info(path):
            if path in probe_by_path:
                info = probe_by_path[path]
                if isinstance(info, Exception):
                    raise info
                return dict(info)
            return dict(_BARE_PROBE_INFO)

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)

    def _make_job(self, tmp_path, write_existing_output=True):
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"original input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["encode"]
        if write_existing_output:
            Path(job.output_path).write_bytes(b"pre-existing output")
        return job

    def test_existing_output_default_config_is_skipped(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(monkeypatch, {})

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.skip_reason == "output-exists"
        assert result.success is False
        assert result.stage_results == {}
        assert result.total_duration == 0.0
        assert job.status == PipelineStatus.SKIPPED
        assert result.decision_log
        # No stage ran -- the pre-existing content is untouched.
        assert Path(job.output_path).read_bytes() == b"pre-existing output"

    def test_existing_output_fail_mode_is_old_failed_behavior(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(monkeypatch, {})
        self.config.set("fail", "general", "existing_output")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert result.success is False
        assert result.skip_reason is None
        assert any("Output already exists" in e for e in result.errors)
        assert job.status == PipelineStatus.FAILED

    def test_check_disabled_skips_without_probing_existing_output(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        probed_paths: list[str] = []

        def fake_get_video_info(path):
            probed_paths.append(path)
            return dict(_BARE_PROBE_INFO)

        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.get_stage",
            lambda name: _FakeEncodeStage if name == "encode" else None,
        )
        monkeypatch.setattr(
            "autovideofixer.core.pipeline.create_stage",
            lambda name, config, overrides=None: (
                _FakeEncodeStage(config, overrides) if name == "encode" else None
            ),
        )
        self.config.set(False, "general", "check_existing_target")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.skip_reason == "output-exists"
        # Only the input was probed -- the existing output was never opened.
        assert job.output_path not in probed_paths

    def test_check_on_matching_existing_output_is_skipped(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {
                    **_BARE_PROBE_INFO,
                    "format": "mov,mp4,m4a",
                    "video_codec": "h264",
                },
            },
        )
        self.config.set("mp4", "general", "target_format")
        self.config.set("libx264", "encoding", "video_codec")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.skip_reason == "output-exists"
        assert any("match" in line for line in result.decision_log)

    def test_check_on_mismatched_reprocess_off_is_skipped_with_reasons(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {
                    **_BARE_PROBE_INFO,
                    "video_codec": "h264",
                },
            },
        )
        self.config.set("libx265", "encoding", "video_codec")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.skip_reason == "output-exists-mismatched"
        assert any("vcodec" in line for line in result.decision_log)
        assert Path(job.output_path).read_bytes() == b"pre-existing output"

    def test_mismatch_reprocess_rename_default_suffix_and_collision(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"},
            },
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        # Pre-occupy "_mismatched-1" so the rename must roll over to "-2".
        stem, ext = os.path.splitext(job.output_path)
        collision_path = f"{stem}_mismatched-1{ext}"
        Path(collision_path).write_bytes(b"already taken")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        renamed_path = f"{stem}_mismatched-2{ext}"
        assert os.path.exists(renamed_path)
        assert Path(renamed_path).read_bytes() == b"pre-existing output"
        assert Path(job.output_path).read_bytes() == b"new content"
        assert any("renamed" in line for line in result.decision_log)

    def test_mismatch_reprocess_rename_custom_suffix(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"},
            },
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        self.config.set("_old-", "general", "mismatched_rename_suffix")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        stem, ext = os.path.splitext(job.output_path)
        assert os.path.exists(f"{stem}_old-1{ext}")

    def test_mismatch_reprocess_rename_cap_exhausted_fails(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"},
            },
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        self.config.set(1, "general", "mismatched_max_renames")
        stem, ext = os.path.splitext(job.output_path)
        Path(f"{stem}_mismatched-1{ext}").write_bytes(b"already taken")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert job.status == PipelineStatus.FAILED
        # The original mismatched file is untouched -- renaming never happened.
        assert Path(job.output_path).read_bytes() == b"pre-existing output"

    def test_mismatch_reprocess_rename_cap_zero_fails(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"},
            },
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        self.config.set(0, "general", "mismatched_max_renames")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert Path(job.output_path).read_bytes() == b"pre-existing output"

    def test_mismatch_reprocess_overwrite_runs_job(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {
                job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"},
            },
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        self.config.set("overwrite", "general", "existing_mismatched")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        assert Path(job.output_path).read_bytes() == b"new content"

    def test_probe_failure_fails_job_with_stderr_surfaced(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(
            monkeypatch,
            {job.input_path: RuntimeError(f"ffprobe failed for {job.input_path}: bad atom size")},
        )

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert any("bad atom size" in e for e in result.errors)
        assert job.status == PipelineStatus.FAILED

    def test_probe_failure_skip_invalid_inputs_is_skipped(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(
            monkeypatch,
            {job.input_path: RuntimeError(f"ffprobe failed for {job.input_path}: bad atom size")},
        )
        self.config.set(True, "general", "skip_invalid_inputs")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.skip_reason == "invalid-input"
        assert job.status == PipelineStatus.SKIPPED

    def test_probe_warning_surfaced_but_job_continues_by_default(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(
            monkeypatch,
            {
                job.input_path: {
                    **_BARE_PROBE_INFO,
                    "probe_stderr": "Non-monotonous DTS",
                },
            },
        )

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"

    def test_probe_warning_fails_job_when_strict_mode_enabled(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(
            monkeypatch,
            {
                job.input_path: {
                    **_BARE_PROBE_INFO,
                    "probe_stderr": "Non-monotonous DTS",
                },
            },
        )
        self.config.set(True, "general", "fail_on_probe_warnings")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert any("Non-monotonous DTS" in e for e in result.errors)


class TestReportingFields:
    """REQUIREMENTS.md § 6.4/6.5: job_wall_ms/processing_ms populated on every
    JobResult return path, reprocessed_mismatch threaded from the § 6.2
    decision path, and scene_stats stored when scene mode runs."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.pipeline = Pipeline(self.config)

    def _register_fakes(self, monkeypatch, probe_by_path):
        def fake_get_stage(name):
            return _FakeEncodeStage if name in ("encode", "stabilize") else None

        def fake_create_stage(name, config, overrides=None):
            return _FakeEncodeStage(config, overrides) if name in ("encode", "stabilize") else None

        def fake_get_video_info(path):
            if path in probe_by_path:
                info = probe_by_path[path]
                if isinstance(info, Exception):
                    raise info
                return dict(info)
            return dict(_BARE_PROBE_INFO)

        monkeypatch.setattr("autovideofixer.core.pipeline.get_stage", fake_get_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.create_stage", fake_create_stage)
        monkeypatch.setattr("autovideofixer.core.pipeline.get_video_info", fake_get_video_info)

    def _make_job(self, tmp_path, write_existing_output=True):
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"original input")
        job = self.pipeline.add_job(str(input_file))
        job.stages = ["encode"]
        if write_existing_output:
            Path(job.output_path).write_bytes(b"pre-existing output")
        return job

    def test_completed_job_has_positive_timing_fields(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(monkeypatch, {})

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        assert result.job_wall_ms > 0
        assert result.processing_ms > 0
        # processing is a subset of the whole job's turn.
        assert result.processing_ms <= result.job_wall_ms

    def test_failed_probe_job_has_wall_time_but_zero_processing(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(
            monkeypatch,
            {job.input_path: RuntimeError(f"ffprobe failed for {job.input_path}: corrupt")},
        )

        result = self.pipeline.execute_job(job)

        assert result.outcome == "failed"
        assert result.job_wall_ms > 0
        assert result.processing_ms == 0.0

    def test_skipped_existing_output_has_wall_time_but_zero_processing(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(monkeypatch, {})

        result = self.pipeline.execute_job(job)

        assert result.outcome == "skipped"
        assert result.job_wall_ms > 0
        assert result.processing_ms == 0.0

    def test_reprocessed_mismatch_true_on_rename_reprocess(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"}},
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        assert result.reprocessed_mismatch is True

    def test_reprocessed_mismatch_true_on_overwrite_reprocess(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path)
        self._register_fakes(
            monkeypatch,
            {job.output_path: {**_BARE_PROBE_INFO, "video_codec": "h264"}},
        )
        self.config.set("libx265", "encoding", "video_codec")
        self.config.set(True, "general", "reprocess_mismatched")
        self.config.set("overwrite", "general", "existing_mismatched")

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        assert result.reprocessed_mismatch is True

    def test_reprocessed_mismatch_false_on_ordinary_completed_job(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(monkeypatch, {})

        result = self.pipeline.execute_job(job)

        assert result.outcome == "completed"
        assert result.reprocessed_mismatch is False

    def test_scene_stats_none_when_scene_mode_off(self, tmp_path, monkeypatch):
        job = self._make_job(tmp_path, write_existing_output=False)
        self._register_fakes(monkeypatch, {})

        result = self.pipeline.execute_job(job)

        assert result.scene_stats is None

    def test_scene_stats_stored_when_scene_mode_runs(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        (tmp_path / "in.mp4").write_bytes(b"original input")
        job = self.pipeline.add_job(str(tmp_path / "in.mp4"))
        job.stages = ["stabilize", "encode"]
        self._register_fakes(monkeypatch, {})
        self.config.set(True, "scenes", "enabled")

        scene_output = tmp_path / "scene_out.mp4"
        scene_output.write_bytes(b"scene mode output")
        fake_scene_result = SimpleNamespace(
            output_path=str(scene_output),
            total_scenes=5,
            kept_scenes=4,
            dropped_scenes=[{"index": 2, "start_time": 1.0, "end_time": 2.0, "reason": "dupe"}],
            stabilize_tiers={},
            interpolated_scenes=[],
        )
        monkeypatch.setattr(
            "autovideofixer.core.scenes.run_scene_pipeline", lambda *a, **k: fake_scene_result
        )

        result = self.pipeline.execute_job(job)

        assert result.scene_stats == {
            "total": 5,
            "kept": 4,
            "dropped": 1,
            "dropped_detail": [{"index": 2, "start_time": 1.0, "end_time": 2.0, "reason": "dupe"}],
        }


class TestExitCodeCountsFailedOnly:
    """CLI exit-code seam (cli.py's _count_failed()): only outcome=="failed"
    jobs should make a run exit non-zero -- SKIPPED/completed never do."""

    def test_all_skipped_or_completed_counts_zero_failed(self):
        from autovideofixer.cli.cli import _count_failed

        results = [
            JobResult(input_path="/a.mp4", success=True),
            JobResult(input_path="/b.mp4", outcome="skipped", skip_reason="output-exists"),
        ]
        assert _count_failed(results) == 0

    def test_one_real_failure_counts_one(self):
        from autovideofixer.cli.cli import _count_failed

        results = [
            JobResult(input_path="/a.mp4", success=True),
            JobResult(input_path="/b.mp4", success=False, errors=["boom"]),
            JobResult(input_path="/c.mp4", outcome="skipped", skip_reason="output-exists"),
        ]
        assert _count_failed(results) == 1
