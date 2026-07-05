"""Auto Video Fixer - Smart processing pipeline.

The pipeline orchestrates processing stages in optimal order,
handles GPU resource management, quality estimation, and
intelligent stage selection based on input/output requirements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from autovideofixer.config import Config
from autovideofixer.core.ffmpeg_utils import (
    generate_temp_path,
    get_video_info,
)
from autovideofixer.core.stages.base import (
    StageResult,
    StageStatus,
    create_stage,
    get_stage,
)


class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A single video processing job."""

    input_path: str
    output_path: str | None = None
    stages: list[str] = field(default_factory=list)  # Stage names to run
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)  # Per-stage config
    priority: int = 0
    status: PipelineStatus = PipelineStatus.IDLE
    progress: float = 0.0
    current_stage: str | None = None
    result: JobResult | None = None

    @property
    def is_queued(self) -> bool:
        return self.status == PipelineStatus.IDLE


@dataclass
class JobResult:
    """Result of a completed job."""

    input_path: str
    output_path: str | None = None
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    total_duration: float = 0.0
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    input_info: dict[str, Any] = field(default_factory=dict)
    output_info: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    quality_meets_target: bool | None = None  # None = not checked (quality_target.mode="none")
    quality_score: float | None = None

    @property
    def all_stages_passed(self) -> bool:
        for sr in self.stage_results.values():
            if sr.status == StageStatus.FAILED:
                return False
        return True

    @property
    def output_size(self) -> int:
        if self.output_path and os.path.exists(self.output_path):
            return os.path.getsize(self.output_path)
        return 0

    @property
    def input_size(self) -> int:
        if os.path.exists(self.input_path):
            return os.path.getsize(self.input_path)
        return 0

    @property
    def size_ratio(self) -> float:
        if self.input_size == 0:
            return 0.0
        return self.output_size / self.input_size


class Pipeline:
    """Main processing pipeline orchestrator.

    Manages stage ordering, resource allocation, quality estimation,
    and job execution.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._jobs: list[Job] = []
        self._running = False
        self._cancel_requested = False
        self._max_concurrent = self.config.get("general", "max_concurrent_jobs", default=1)
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            from autovideofixer.logger import get_logger

            self._logger = get_logger("autovideofixer.pipeline")
        return self._logger

    @property
    def jobs(self) -> list[Job]:
        return list(self._jobs)

    @property
    def running(self) -> bool:
        return self._running

    def add_job(
        self,
        input_path: str,
        output_path: str | None = None,
        stage_names: list[str] | None = None,
        overrides: dict[str, dict] | None = None,
        priority: int = 0,
    ) -> Job:
        """Add a processing job to the queue.

        If stage_names is None, the pipeline will determine optimal stages
        based on input analysis and user settings.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_dir = self.config.get("general", "output_dir", default=None)
            base_name = os.path.basename(input_path)
            stem, ext = os.path.splitext(base_name)
            output_filename = f"{stem}_enhanced{ext}"
            if output_dir:
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = os.path.join(os.path.dirname(input_path), output_filename)

        job = Job(
            input_path=input_path,
            output_path=output_path,
            stages=stage_names or [],
            stage_overrides=overrides or {},
            priority=priority,
        )
        self._jobs.append(job)
        return job

    def add_files(self, paths: list[str]) -> list[Job]:
        """Add multiple files/directories to the job queue."""
        from autovideofixer.core.analysis import is_video_file

        jobs = []
        for p in paths:
            if os.path.isdir(p):
                for entry in sorted(os.listdir(p)):
                    full = os.path.join(p, entry)
                    if is_video_file(full):
                        jobs.append(self.add_job(full))
            elif is_video_file(p):
                jobs.append(self.add_job(p))
        return jobs

    def auto_determine_stages(self, job: Job) -> list[str]:
        """Determine optimal processing stages for a job based on input and settings.

        Uses quality targets, input analysis, and preset preferences.
        """
        input_info = get_video_info(job.input_path)
        stages = []

        # 1. Analysis stage (always first)
        stages.append("detect")

        # 2. Enhancement stages based on input properties
        resolution = input_info.get("resolution", (0, 0))
        framerate = input_info.get("framerate", 0)
        is_hdr = input_info.get("is_hdr", False)

        # Check quality target from config
        quality_target = self.config.get("quality", "quality_target", default={})
        target_resolution = quality_target.get("target_resolution")
        target_framerate = quality_target.get("target_framerate")

        # Upscale if target resolution is higher
        if target_resolution:
            target_w, target_h = target_resolution
            if resolution[0] < target_w or resolution[1] < target_h:
                stages.append("upscale")

        # Frame interpolation if target framerate is higher
        if target_framerate and framerate > 0:
            if target_framerate > framerate:
                stages.append("interpolate")

        # HDR conversion
        if is_hdr and self.config.get("stages", "hdr_to_sdr", "enabled", default=False):
            stages.append("hdr")

        # Always-on enhancement stages
        for stage_name in [
            "stabilize",
            "denoise_video",
            "deblock",
            "normalize_volume",
            "normalize_audio",
        ]:
            if self.config.get("stages", stage_name, "enabled", default=True):
                stages.append(stage_name)

        # Speed adjustment
        speed_config = self.config.get("stages", "speed", default={})
        if speed_config.get("enabled", False):
            stages.append("speed")

        # Final encoding
        stages.append("encode")

        # Apply user overrides
        if job.stages:
            stages = job.stages

        return stages

    def _apply_config_encoding_overrides(self, job: Job) -> None:
        """Merge config["encoding"] (from a preset or user config) into
        job.stage_overrides["encode"], without clobbering any override the
        job already set explicitly.

        Preset.to_config() writes video_codec/audio_codec/crf/preset under
        "encoding", but EncodeStage.execute() reads codec/audio_codec/crf/preset
        kwargs -- video_codec is remapped to codec here.
        """
        encoding_cfg = self.config.get("encoding", default={})
        if not encoding_cfg:
            return
        encode_overrides = job.stage_overrides.setdefault("encode", {})
        key_map = {
            "video_codec": "codec",
            "audio_codec": "audio_codec",
            "crf": "crf",
            "preset": "preset",
        }
        for cfg_key, stage_kwarg in key_map.items():
            if cfg_key in encoding_cfg and stage_kwarg not in encode_overrides:
                encode_overrides[stage_kwarg] = encoding_cfg[cfg_key]

    def optimize_stage_order(self, stages: list[str]) -> list[str]:
        """Reorder stages for optimal quality and performance.

        Rules:
        1. Analysis/detection first
        2. Stabilization before enhancement (reduce noise from motion)
        3. Denoising before upscaling (don't upscale noise)
        4. Deblocking before denoising (remove compression artifacts first)
        5. Upscaling before interpolation (higher res frames interpolate better)
        6. Normalization near the end
        7. Encoding last
        """
        # Default optimal order
        default_order = [
            "detect",
            "stabilize",
            "deblock",
            "denoise_video",
            "upscale",
            "interpolate",
            "normalize_volume",
            "normalize_audio",
            "speed",
            "hdr",
            "encode",
        ]

        # Filter to only requested stages, preserving order
        ordered = [s for s in default_order if s in stages]

        # Add any remaining stages not in default_order
        for s in stages:
            if s not in ordered:
                ordered.append(s)

        return ordered

    def execute_job(
        self,
        job: Job,
        progress_callback: Callable[[Job, float, str], None] | None = None,
    ) -> JobResult:
        """Execute a single job through all determined stages.

        progress_callback, if given, is invoked as (job, overall_progress, message)
        on every per-stage progress update -- lets callers (e.g. the GUI) show live
        progress instead of only a start/finish transition.

        Returns JobResult with details of each stage outcome.
        """
        self._cancel_requested = False

        # Determine stages if not specified
        if not job.stages:
            job.stages = self.auto_determine_stages(job)

        # Optimize order, then drop anything that isn't a registered stage up front so
        # progress accounting and cleanup only ever deal with stages that actually run.
        requested_names = self.optimize_stage_order(job.stages)
        stage_names: list[str] = []
        skipped: list[str] = []
        for name in requested_names:
            if get_stage(name) is None:
                self.logger.warning(f"Unknown stage: {name}")
                skipped.append(name)
            else:
                stage_names.append(name)

        max_stages = self.config.get("pipeline", "max_stages", default=None)
        if max_stages is not None and len(stage_names) > max_stages:
            # Not truncated: the always-last "encode" stage must not be dropped, and
            # naive slicing would drop it whenever a full default pipeline (11 stages)
            # exceeds the default max_stages=10. Surface it as an explicit failure
            # instead of silently either truncating output-producing stages or
            # ignoring the configured cap outright.
            msg = (
                f"Requested {len(stage_names)} stages exceeds pipeline.max_stages="
                f"{max_stages}: {stage_names}"
            )
            self.logger.error(msg)
            job_result = JobResult(
                input_path=job.input_path,
                output_path=None,
                errors=[msg],
                success=False,
            )
            job.status = PipelineStatus.FAILED
            job.result = job_result
            job.progress = 1.0
            return job_result

        self.logger.info(f"Processing {os.path.basename(job.input_path)}: stages={stage_names}")

        input_info = get_video_info(job.input_path)
        job.input_info = input_info

        # Surface preset/config-level "encoding" and "general.target_format" settings
        # (set via Preset.to_config()) into the per-stage machinery. Config values are
        # a floor: an explicit job.stage_overrides entry always wins over them.
        self._apply_config_encoding_overrides(job)
        target_format = self.config.get("general", "target_format", default=None)
        if target_format:
            input_info["target_format"] = target_format
            job.stage_overrides.setdefault("remux", {}).setdefault("target_format", target_format)
        current_path = job.input_path
        stage_results: dict[str, StageResult] = {}
        errors: list[str] = []
        start_time = __import__("time").time()

        overwrite = self.config.get("general", "overwrite", default=False)
        if (
            not overwrite
            and job.output_path
            and os.path.exists(job.output_path)
            and os.path.abspath(job.output_path) != os.path.abspath(job.input_path)
        ):
            msg = f"Output already exists and general.overwrite is False: {job.output_path}"
            self.logger.error(msg)
            job_result = JobResult(
                input_path=job.input_path,
                output_path=None,
                errors=[msg],
                input_info=input_info,
                success=False,
            )
            job.status = PipelineStatus.FAILED
            job.result = job_result
            job.progress = 1.0
            return job_result

        # Create output directory
        output_dir = os.path.dirname(job.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        temp_dir = self.config.get("general", "temp_dir", default=None)

        try:
            for i, stage_name in enumerate(stage_names):
                if self._cancel_requested:
                    job.status = PipelineStatus.CANCELLED
                    break

                stage = create_stage(stage_name, self.config)
                if stage is None:
                    # Already filtered above; defensive only.
                    skipped.append(stage_name)
                    continue

                # Check if stage should run
                should_run, reason = stage.should_run(input_info)
                if not should_run:
                    self.logger.info(f"Skipping {stage_name}: {reason}")
                    skipped.append(stage_name)
                    stage_results[stage_name] = StageResult(
                        status=StageStatus.SKIPPED,
                        skipped_reason=reason,
                    )
                    continue

                # Apply per-stage overrides
                overrides = job.stage_overrides.get(stage_name, {})

                # Determine output path for this stage
                is_last = i == len(stage_names) - 1
                if is_last:
                    stage_output = job.output_path
                else:
                    stage_output = generate_temp_path(
                        os.path.dirname(job.input_path),
                        job.input_path,
                        suffix=f"_{stage_name}",
                        temp_dir=temp_dir,
                    )

                job.current_stage = stage_name
                self.logger.info(f"Running stage: {stage_name}")

                # Execute stage with progress
                def progress_cb(prog, msg, _i=i, _n=len(stage_names)):
                    job.progress = (_i + prog) / _n
                    if progress_callback:
                        progress_callback(job, job.progress, msg)

                try:
                    result = stage.execute(
                        current_path,
                        stage_output,
                        progress_callback=progress_cb,
                        input_info=input_info,
                        **overrides,
                    )
                except Exception as e:
                    self.logger.exception(f"Stage {stage_name} raised an unexpected exception")
                    result = StageResult(status=StageStatus.FAILED, error=str(e))

                stage_results[stage_name] = result

                if result.status == StageStatus.FAILED:
                    errors.append(f"{stage_name}: {result.error}")
                    self.logger.error(f"Stage {stage_name} failed: {result.error}")
                    if self.config.get("pipeline", "skip_stage_on_error", default=True):
                        # Continue with next stage using original input
                        self.logger.info(f"Continuing pipeline after {stage_name} failure")
                    else:
                        self.logger.error(f"Stopping pipeline: {stage_name} failed")
                        break
                else:
                    if result.output_path:
                        current_path = result.output_path
                    elif result.status == StageStatus.COMPLETED and stage.produces_output:
                        self.logger.warning(
                            f"Stage {stage_name} reported COMPLETED with no output_path; "
                            "treating as failed"
                        )
                        result.status = StageStatus.FAILED
                        result.error = result.error or "Stage completed without an output_path"
                        errors.append(f"{stage_name}: {result.error}")
        finally:
            total_time = __import__("time").time() - start_time
            # Clean up temp files regardless of how the loop above exited.
            if not self._cancel_requested:
                self._cleanup_temp_files(stage_names, stage_results, job)

        # Resolve the job's output path: the terminal stage's output, or if that stage
        # was skipped/failed, fall back to the last successfully-produced path so a
        # "success" JobResult never carries a None output_path.
        final_output_path: str | None = None
        if stage_names:
            final_result = stage_results.get(stage_names[-1])
            final_stage = create_stage(stage_names[-1], self.config)
            if final_result is not None and final_result.output_path:
                final_output_path = final_result.output_path
            elif final_result is not None and (
                final_result.status == StageStatus.SKIPPED
                or (
                    final_result.status == StageStatus.COMPLETED
                    and final_stage is not None
                    and not final_stage.produces_output
                )
            ):
                final_output_path = current_path

        # Build result
        job_result = JobResult(
            input_path=job.input_path,
            output_path=final_output_path,
            stage_results=stage_results,
            total_duration=total_time,
            errors=errors,
            skipped=skipped,
            input_info=input_info,
            success=len(errors) == 0,
        )

        # Quality gate: quality_target.mode/target were previously accepted from
        # presets/config but nothing ever ran estimate_ssim_psnr()/meets_target()
        # against them -- size_reduction's advertised "acceptable quality loss"
        # bound had no effect. This is best-effort and non-fatal (an expensive
        # extra ffmpeg pass failing shouldn't fail an otherwise-successful job);
        # it only records the result for callers/reports to act on.
        quality_target = self.config.get("quality", "quality_target", default={})
        quality_mode = quality_target.get("mode", "none")
        if job_result.success and quality_mode != "none" and final_output_path:
            try:
                from autovideofixer.core.quality import estimate_ssim_psnr

                target = quality_target.get("target")
                quality_result = estimate_ssim_psnr(
                    job.input_path, final_output_path, target=target
                )
                job_result.quality_score = quality_result.score
                job_result.quality_meets_target = quality_result.meets_target()
                if not job_result.quality_meets_target:
                    self.logger.warning(
                        f"Output quality {quality_result.score:.1f} below target "
                        f"{target} for {job.input_path}"
                    )
            except Exception as e:
                self.logger.warning(f"Quality check failed for {job.input_path}: {e}")

        job.status = (
            PipelineStatus.CANCELLED
            if self._cancel_requested
            else (PipelineStatus.COMPLETED if job_result.success else PipelineStatus.FAILED)
        )
        job.result = job_result
        job.progress = 1.0

        return job_result

    def execute_all(
        self,
        callback: Callable[[Job, JobResult], None] | None = None,
        progress_callback: Callable[[Job, float, str], None] | None = None,
    ) -> list[JobResult]:
        """Execute all jobs in the queue.

        Runs strictly sequentially when general.max_concurrent_jobs <= 1 (the
        default). When set higher, runs up to that many jobs concurrently via a
        thread pool -- safe because each job reads/writes distinct files and
        Config is read-mostly during processing. Note this does not attempt to
        serialize GPU-bound AI stages against each other; users who raise
        max_concurrent_jobs on a GPU-constrained machine may see contention/OOM
        across concurrently-running AI stages, same as running multiple avf
        processes by hand would.

        progress_callback, if given, is forwarded to execute_job() for live
        per-stage progress reporting (e.g. for a GUI progress bar).
        """
        self._running = True
        results: list[JobResult] = []

        # Sort by priority (higher priority first)
        sorted_jobs = sorted(self._jobs, key=lambda j: -j.priority)
        max_workers = max(1, self._max_concurrent)

        def _run(job: Job) -> JobResult:
            try:
                return self.execute_job(job, progress_callback=progress_callback)
            except Exception as e:
                self.logger.exception(f"Job failed: {job.input_path}")
                return JobResult(input_path=job.input_path, errors=[str(e)], success=False)

        if max_workers <= 1:
            for job in sorted_jobs:
                if not self._running:
                    break
                result = _run(job)
                results.append(result)
                if callback:
                    callback(job, result)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {}
                for job in sorted_jobs:
                    if not self._running:
                        break
                    future_to_job[executor.submit(_run, job)] = job

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    result = future.result()
                    results.append(result)
                    if callback:
                        callback(job, result)

        self._running = False
        return results

    def cancel(self) -> None:
        """Cancel all running jobs."""
        self._running = False
        self._cancel_requested = True

    def clear_queue(self) -> None:
        """Clear all jobs from the queue."""
        self._jobs.clear()

    def _cleanup_temp_files(
        self,
        stage_names: list[str],
        stage_results: dict[str, StageResult],
        job: Job,
    ) -> None:
        """Remove intermediate temp files, keep only final output.

        Runs for both COMPLETED and FAILED stages -- a failed stage can still have
        written a partial/temp output file before erroring, and with the default
        skip_stage_on_error=True those would otherwise be orphaned permanently next
        to the user's source video.
        """
        for name, result in stage_results.items():
            if result.status in (StageStatus.COMPLETED, StageStatus.FAILED) and result.output_path:
                if result.output_path != job.output_path:
                    if os.path.exists(result.output_path):
                        try:
                            os.remove(result.output_path)
                        except OSError:
                            pass

    def generate_report(self, job_result: JobResult) -> str:
        """Generate a human-readable processing report."""
        lines = [
            f"=== Processing Report: {os.path.basename(job_result.input_path)} ===",
            f"Input:  {job_result.input_path}",
            f"Output: {job_result.output_path or 'N/A'}",
            f"Duration: {job_result.total_duration:.1f}s",
            f"Success: {job_result.success}",
            "",
            "--- Stages ---",
        ]
        for name, result in job_result.stage_results.items():
            status_icon = {
                StageStatus.COMPLETED: "[green]OK[/green]",
                StageStatus.SKIPPED: "[yellow]SKIP[/yellow]",
                StageStatus.FAILED: "[red]FAIL[/red]",
            }.get(result.status, str(result.status))
            lines.append(f"  {status_icon} {name}: {result.duration_sec:.1f}s")

        if job_result.skipped:
            lines.append(f"\nSkipped: {', '.join(job_result.skipped)}")
        if job_result.errors:
            lines.append(f"\nErrors: {'; '.join(job_result.errors)}")

        return "\n".join(lines)
