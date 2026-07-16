"""Auto Video Fixer - Smart processing pipeline.

The pipeline orchestrates processing stages in optimal order,
handles GPU resource management, quality estimation, and
intelligent stage selection based on input/output requirements.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from autovideofixer.config import Config, deep_merge, redact_secrets
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

# Fallback when config `pipeline.default_order` is missing/empty -- mirrors
# Config.DEFAULTS["pipeline"]["default_order"] exactly (see config.py for the
# deblock-before-stabilize rationale). Kept as a plain list[str] since
# DEFAULTS itself never uses mapping entries.
DEFAULT_STAGE_ORDER: list[str] = [
    "detect",
    "deblock",
    "stabilize",
    "crop",
    "denoise_video",
    "upscale",
    "interpolate",
    "normalize_volume",
    "normalize_audio",
    "speed",
    "hdr",
    "encode",
]


@dataclass
class StageOrderEntry:
    """One resolved, runnable occurrence from ``pipeline.default_order``.

    label: unique key for this occurrence within the job -- the plain stage
        name for the (common-case) single occurrence, or ``"<name>#2"``,
        ``"<name>#3"``, ... for a stage repeated via ``default_order``. Used
        everywhere uniqueness matters: ``JobResult.stage_results``,
        generated temp filenames, progress/logging, ``job.current_stage``.
    name: the actual registered stage name (e.g. ``"deblock"`` for both
        ``"deblock"`` and ``"deblock#2"``) -- what's passed to
        ``get_stage()``/``create_stage()``.
    forced: tristate mirroring the order entry's ``enabled:`` field. ``True``
        forces this occurrence to run regardless of ``stages.<name>.enabled``,
        preset ``enable_stages``, or auto-determination membership (mirrors
        the existing ``explicit_stage_request``/``_force_enabled``
        mechanism -- internal ``should_run()`` sanity/dependency gates still
        apply). ``False`` means this occurrence never runs. ``None`` defers
        to whether ``name`` is in the job's requested/auto-determined stage
        set, exactly like a plain string entry always has.
    overrides: per-occurrence config overrides (an order entry's ``config:``
        mapping), deep-merged over ``stages.<name>`` <- ``job.stage_overrides[name]``
        for this occurrence only.
    """

    label: str
    name: str
    forced: bool | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


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
            output_container = self.config.get("general", "output_container", default=None)
            if output_container:
                ext = (
                    output_container if output_container.startswith(".") else f".{output_container}"
                )
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

        # Auto-crop -- opt-in, off by default (see Config.DEFAULTS["stages"]["crop"]
        # and docs/REQUIREMENTS.md feature 3). optimize_stage_order() places it
        # right after "stabilize" regardless of insertion order here.
        crop_config = self.config.get("stages", "crop", default={})
        if crop_config.get("enabled", False):
            stages.append("crop")

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

    def _resolve_order_entries(self) -> list[tuple[str, bool | None, dict[str, Any]]]:
        """Parse ``pipeline.default_order`` into ``(name, forced, overrides)`` tuples.

        Falls back to ``DEFAULT_STAGE_ORDER`` (plain names, no forcing/overrides)
        when the config key is missing or empty. Each entry is either a plain
        stage name string, or a mapping ``{stage: str, enabled?: bool|null,
        config?: dict}``.

        Raises:
            ValueError: a mapping entry has no (or a non-string) ``stage`` key,
                an ``enabled`` value that isn't true/false/null, a non-mapping
                ``config`` value, or an entry that's neither a string nor a
                mapping. Malformed entries are a hard configuration error
                (caught at order-resolution time, not silently skipped) since
                they indicate a typo'd/invalid config rather than an
                unregistered-but-otherwise-valid stage name (see the
                "Unknown stage" warning-and-skip path in ``execute_job``,
                which is a different, non-fatal case).
        """
        raw = self.config.get("pipeline", "default_order", default=None)
        if not raw:
            raw = DEFAULT_STAGE_ORDER

        parsed: list[tuple[str, bool | None, dict[str, Any]]] = []
        for entry in raw:
            if isinstance(entry, str):
                parsed.append((entry, None, {}))
                continue
            if isinstance(entry, dict):
                name = entry.get("stage")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        "Malformed pipeline.default_order entry: mapping is missing a "
                        f"string 'stage' key: {entry!r}"
                    )
                forced = entry.get("enabled", None)
                if forced is not None and not isinstance(forced, bool):
                    raise ValueError(
                        f"Malformed pipeline.default_order entry for stage {name!r}: "
                        f"'enabled' must be true, false, or null/omitted -- got {forced!r}"
                    )
                overrides = entry.get("config", {}) or {}
                if not isinstance(overrides, dict):
                    raise ValueError(
                        f"Malformed pipeline.default_order entry for stage {name!r}: "
                        f"'config' must be a mapping -- got {type(overrides).__name__}"
                    )
                parsed.append((name, forced, overrides))
                continue
            raise ValueError(
                "Malformed pipeline.default_order entry: expected a string or a "
                f"mapping with a 'stage' key -- got {type(entry).__name__}: {entry!r}"
            )
        return parsed

    def resolve_stage_order(self, requested: list[str]) -> list[StageOrderEntry]:
        """Resolve ``pipeline.default_order`` into an ordered list of occurrences.

        ``requested`` is the job's requested/auto-determined stage name set
        (``job.stages`` after ``auto_determine_stages()``/explicit ``--stage``).
        Each ``default_order`` entry independently decides whether it runs:

        - Plain string / mapping with ``enabled: null`` (or omitted): runs iff
          its stage name is in ``requested`` -- exactly today's gating.
        - ``enabled: true``: always runs, regardless of ``requested``
          membership, ``stages.<name>.enabled``, or preset ``enable_stages``
          (``should_run()``'s own internal dependency/sanity gates still
          apply -- this only bypasses the config ``enabled`` flag, mirroring
          the existing ``explicit_stage_request``/``_force_enabled``
          mechanism for ``--stage``).
        - ``enabled: false``: never runs (the only way to hard-drop a stage
          that's otherwise in ``requested`` -- omitting it from
          ``default_order`` entirely does NOT drop it, see below).

        A stage name repeated in ``default_order`` (and passing its own gate)
        produces multiple occurrences, each independently resolved and each
        chaining off the previous occurrence's output like any other stage.

        Stages in ``requested`` but not mentioned anywhere in
        ``default_order`` (as a plain string OR inside a mapping's ``stage``
        key, regardless of that mapping's ``enabled`` value) are appended at
        the end, preserving their relative order in ``requested`` -- this is
        what keeps plain ``--stage`` usage working when a stage isn't in the
        configured order list. ``encode`` is always forced back to the last
        position afterward (existing invariant), in case a remaining-append
        would otherwise have landed something after it.

        Raises:
            ValueError: propagated from ``_resolve_order_entries()`` on a
                malformed ``default_order`` entry.
        """
        entries = self._resolve_order_entries()

        mentioned: set[str] = set()
        occurrence_counts: dict[str, int] = {}
        resolved: list[StageOrderEntry] = []

        def _add_occurrence(name: str, forced: bool | None, overrides: dict[str, Any]) -> None:
            occurrence_counts[name] = occurrence_counts.get(name, 0) + 1
            idx = occurrence_counts[name]
            label = name if idx == 1 else f"{name}#{idx}"
            resolved.append(
                StageOrderEntry(label=label, name=name, forced=forced, overrides=overrides)
            )

        for name, forced, overrides in entries:
            mentioned.add(name)
            if forced is True:
                will_run = True
            elif forced is False:
                will_run = False
            else:
                will_run = name in requested
            if will_run:
                _add_occurrence(name, forced, overrides)

        # Plain-string compat: a requested stage never mentioned in
        # default_order (as a string or inside a mapping's `stage` key) is
        # appended at the end -- a stage that IS mentioned (even with
        # enabled: false) is never re-appended here; enabled: false is the
        # only way to hard-drop it (see AGENTS.md's Pipeline Behavior note).
        for name in requested:
            if name not in mentioned:
                _add_occurrence(name, None, {})

        # Invariant: "encode" must remain last, even if the remaining-stage
        # append above landed something after it (only possible when a
        # requested stage outside default_order sorts after "encode" in
        # `requested`).
        encode_positions = [i for i, e in enumerate(resolved) if e.name == "encode"]
        if encode_positions and encode_positions[-1] != len(resolved) - 1:
            idx = encode_positions[-1]
            resolved.append(resolved.pop(idx))

        return resolved

    def optimize_stage_order(self, stages: list[str]) -> list[str]:
        """Reorder ``stages`` per ``pipeline.default_order`` (config-driven).

        Backward-compatible flattened view: returns occurrence labels in
        execution order (e.g. plain ``"deblock"``, or ``"deblock"``/
        ``"deblock#2"`` if ``default_order`` repeats it and both occurrences'
        gates pass) -- see ``resolve_stage_order()`` for the full per-occurrence
        records (forced tristate, per-occurrence config overrides), which is
        what ``execute_job()`` actually uses.

        Rules (informational -- ``pipeline.default_order``, which
        ``resolve_stage_order()`` reads, is the actual source of truth):
        1. Analysis/detection first
        2. Deblocking before stabilization (compression artifacts come from
           the source video; deblocking a not-yet-warped frame keeps the
           deblock model's input accurate, and gives the stabilizer cleaner
           detail to track)
        3. Auto-crop right after stabilization: stabilize's zoom-out correction
           can itself add a black border, so cropping after it removes both
           the original letterboxing/pillarboxing AND any residual
           stabilization border in a single pass -- and running it before
           denoise/upscale/interpolate/encode means none of those
           (especially the AI-capable ones) waste compute on pixels that are
           about to be cropped away.
        4. Denoising before upscaling (don't upscale noise)
        5. Upscaling before interpolation (higher res frames interpolate better)
        6. Normalization near the end
        7. Encoding last
        """
        return [entry.label for entry in self.resolve_stage_order(stages)]

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

        # A caller-provided (e.g. CLI --stage) job.stages means the user is
        # explicitly naming which stages to run, replacing the preset/auto-determined
        # list -- captured before auto-determination fills job.stages in, so this
        # stays False for the preset/default path. Threaded onto each stage instance
        # below so should_run() bypasses that stage's `enabled: false` config instead
        # of silently skipping a stage the user explicitly asked for.
        explicit_stage_request = bool(job.stages)

        # Determine stages if not specified
        if not job.stages:
            job.stages = self.auto_determine_stages(job)

        # Resolve pipeline.default_order into occurrence records (order,
        # omission, repetition, per-occurrence enabled/config overrides --
        # see resolve_stage_order()), then drop anything that isn't a
        # registered stage up front so progress accounting and cleanup only
        # ever deal with stages that actually run.
        try:
            requested_entries = self.resolve_stage_order(job.stages)
        except ValueError as e:
            msg = f"Invalid pipeline.default_order: {e}"
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

        stage_entries: list[StageOrderEntry] = []
        skipped: list[str] = []
        for entry in requested_entries:
            if get_stage(entry.name) is None:
                self.logger.warning(f"Unknown stage: {entry.name}")
                skipped.append(entry.label)
            else:
                stage_entries.append(entry)

        max_stages = self.config.get("pipeline", "max_stages", default=None)
        if max_stages is not None and len(stage_entries) > max_stages:
            # Not truncated: the always-last "encode" stage must not be dropped, and
            # naive slicing would drop it whenever a full default pipeline (12 stages)
            # exceeds the default max_stages=10. Surface it as an explicit failure
            # instead of silently either truncating output-producing stages or
            # ignoring the configured cap outright. Counts occurrences, so a
            # default_order-repeated stage counts once per repetition.
            stage_labels = [e.label for e in stage_entries]
            msg = (
                f"Requested {len(stage_entries)} stage occurrences exceeds "
                f"pipeline.max_stages={max_stages}: {stage_labels}"
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

        stage_names: list[str] = [e.label for e in stage_entries]
        self.logger.info(f"Processing {os.path.basename(job.input_path)}: stages={stage_names}")
        if self.logger.isEnabledFor(logging.DEBUG):  # avoid building the dump otherwise
            for entry in stage_entries:
                stage_cfg = self.config.get("stages", entry.name, default={})
                job_overrides = job.stage_overrides.get(entry.name, {})
                effective = copy.deepcopy(stage_cfg)
                if job_overrides:
                    deep_merge(effective, job_overrides)
                if entry.overrides:
                    deep_merge(effective, entry.overrides)
                self.logger.debug(
                    "Stage '%s' effective config: %s", entry.label, redact_secrets(effective)
                )

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

        # Scene mode (opt-in, config `scenes.enabled` / CLI `--scene-mode`): when
        # on and this job's stage list includes stabilize and/or interpolate,
        # run those two stages per-scene (never interpolating across a cut, and
        # tiering stabilize strength per scene) and concatenate the result BEFORE
        # the remaining whole-video stages run. See core/scenes.py. Zero effect
        # on the stage list/behavior below when scenes.enabled is False (the
        # default) -- this block is a strict no-op in that case.
        scene_mode_temp_path: str | None = None
        _run_stabilize = any(e.name == "stabilize" for e in stage_entries)
        _run_interpolate = any(e.name == "interpolate" for e in stage_entries)
        if self.config.get("scenes", "enabled", default=False) and (
            _run_stabilize or _run_interpolate
        ):
            try:
                from autovideofixer.core.scenes import run_scene_pipeline

                quality_target = self.config.get("quality", "quality_target", default={})
                scene_result = run_scene_pipeline(
                    job.input_path,
                    self.config,
                    run_stabilize=_run_stabilize,
                    run_interpolate=_run_interpolate,
                    target_fps=quality_target.get("target_framerate"),
                )
            except Exception:
                self.logger.warning(
                    "Scene mode preprocessing raised an unexpected exception for %s; "
                    "falling back to whole-video processing",
                    job.input_path,
                    exc_info=True,
                )
                scene_result = None

            if scene_result is not None:
                self.logger.info(
                    "Scene mode: %d/%d scene(s) kept (%d dropped), stabilize tiers=%s, "
                    "interpolated scene(s)=%s",
                    scene_result.kept_scenes,
                    scene_result.total_scenes,
                    len(scene_result.dropped_scenes),
                    scene_result.stabilize_tiers,
                    scene_result.interpolated_scenes,
                )
                for dropped in scene_result.dropped_scenes:
                    self.logger.warning(
                        "Scene mode: dropped scene #%s (t=%.1f-%.1fs): %s",
                        dropped["index"],
                        dropped["start_time"],
                        dropped["end_time"],
                        dropped.get("reason", "no reason given"),
                    )
                current_path = scene_result.output_path
                scene_mode_temp_path = scene_result.output_path
                stage_entries = [
                    e for e in stage_entries if e.name not in ("stabilize", "interpolate")
                ]
                stage_names = [e.label for e in stage_entries]
                # Stages after this point (e.g. upscale's should_run) read
                # input_info for resolution/framerate/duration -- reprobe the
                # reassembled file so they see its actual (possibly
                # interpolated-to-a-higher-fps, possibly shorter after drops)
                # properties instead of the original input's.
                input_info = get_video_info(current_path)

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

        # Every path generate_temp_path() hands out during this job, tracked
        # independently of StageResult so a FAILED stage (which typically returns
        # no output_path) doesn't leave its partial temp file orphaned next to the
        # user's source video, and so cleanup still runs on cancel.
        generated_temp_paths: list[str] = []
        if scene_mode_temp_path is not None:
            # Reuses the same generated-temp-path cleanup as any other
            # intermediate: kept alive as long as it's current_path (the input
            # to the next stage), removed once superseded, and never removed if
            # it ends up being promoted straight to job.output_path (e.g. a
            # scene-mode-only run with stabilize/interpolate as the only
            # requested stages).
            generated_temp_paths.append(scene_mode_temp_path)

        # Whether the surviving-intermediate fallback below (final_output_path =
        # current_path) got moved onto job.output_path. Guards the outer finally's
        # cleanup-safety-net from deleting a temp file that was already promoted
        # (moved, not copied) to the user-visible output path.
        promoted_output = False

        try:
            try:
                for i, entry in enumerate(stage_entries):
                    stage_name = entry.label
                    if self._cancel_requested:
                        job.status = PipelineStatus.CANCELLED
                        break

                    # Per-occurrence config cascade: stages.<name> (baked into
                    # create_stage()'s cascaded lookup) <- job.stage_overrides[name]
                    # <- this occurrence's own `config:` overrides (highest
                    # precedence) -- merged once and threaded into BOTH the
                    # stage instance's _stage_config (so __init__-cached fields
                    # like ai_model/tile_size see it) and execute()'s **kwargs
                    # (so explicit execute() params like deblock's `strength`
                    # see it too).
                    job_overrides = job.stage_overrides.get(entry.name, {})
                    effective_overrides: dict[str, Any] = dict(job_overrides)
                    if entry.overrides:
                        deep_merge(effective_overrides, entry.overrides)

                    stage = create_stage(entry.name, self.config, overrides=effective_overrides)
                    if stage is None:
                        # Already filtered above; defensive only.
                        skipped.append(stage_name)
                        continue
                    if explicit_stage_request or entry.forced is True:
                        stage._force_enabled = True

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

                    # Determine output path for this stage
                    is_last = i == len(stage_entries) - 1
                    if is_last:
                        stage_output = job.output_path
                    else:
                        stage_output = generate_temp_path(
                            os.path.dirname(job.input_path),
                            job.input_path,
                            suffix=f"_{stage_name}",
                            temp_dir=temp_dir,
                        )
                        generated_temp_paths.append(stage_output)

                    job.current_stage = stage_name
                    self.logger.info(f"Running stage: {stage_name}")

                    # Execute stage with progress
                    def progress_cb(prog, msg, _i=i, _n=len(stage_entries)):
                        job.progress = (_i + prog) / _n
                        if progress_callback:
                            progress_callback(job, job.progress, msg)

                    try:
                        result = stage.execute(
                            current_path,
                            stage_output,
                            progress_callback=progress_cb,
                            input_info=input_info,
                            **effective_overrides,
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
                # Clean up temp files regardless of how the loop above exited, including
                # on cancel/exception -- a FAILED stage typically returns no output_path,
                # so relying solely on StageResult.output_path (as the StageResult-based
                # cleanup below does) leaves its partial temp file orphaned.
                #
                # current_path is deliberately kept alive here (not passed for removal):
                # if the terminal stage was skipped or produced no output, final-output
                # resolution below falls back to current_path as the job's result, and
                # it needs to survive long enough to be promoted to job.output_path.
                self._cleanup_temp_files(stage_names, stage_results, job, keep={current_path})
                self._cleanup_generated_temp_paths(generated_temp_paths, job, keep={current_path})

            # Resolve the job's output path: the terminal stage's output, or if that stage
            # was skipped/failed, fall back to the last successfully-produced path so a
            # "success" JobResult never carries a None output_path.
            final_output_path: str | None = None
            if not stage_names and scene_mode_temp_path is not None:
                # Scene mode ran and there are no further stages (e.g. this job's
                # only requested stages were stabilize/interpolate) -- current_path
                # is scene mode's own output and IS the job's result.
                final_output_path = current_path
            elif stage_names:
                final_result = stage_results.get(stage_names[-1])
                final_stage = create_stage(stage_entries[-1].name, self.config)
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

            # The fallback above can resolve to a surviving intermediate temp file
            # (e.g. terminal stage skipped after an earlier stage completed). That
            # temp file was kept alive through cleanup above specifically so it can
            # be promoted here to the user-visible job.output_path -- otherwise the
            # JobResult would report a path that either doesn't exist (it would have
            # been deleted as a temp file) or never reaches the location the caller
            # asked for.
            if (
                final_output_path is not None
                and final_output_path in generated_temp_paths
                and final_output_path != job.output_path
            ):
                self._promote_temp_to_output(final_output_path, job.output_path)
                final_output_path = job.output_path
                promoted_output = True
        finally:
            # Safety net: if we exited the block above (return or exception) without
            # promoting current_path, and it's a temp file we deliberately kept alive
            # through the cleanup above, it's now a genuine orphan -- remove it. Runs
            # even if an exception propagated out of the stage loop or the promotion
            # step itself.
            if not promoted_output and current_path in generated_temp_paths:
                self._cleanup_generated_temp_paths([current_path], job)

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
                if quality_result.measurement_failed:
                    # The ffmpeg comparison itself failed (e.g. couldn't determine
                    # dimensions, nonzero exit, unparseable output) -- leave
                    # quality_score/quality_meets_target as "not checked" (None)
                    # rather than reporting a fake 0.0 score that reads as a
                    # genuine failing measurement.
                    reason = quality_result.details.get("error", "unknown error")
                    self.logger.warning(
                        f"Quality check could not be measured for {job.input_path}: {reason}"
                    )
                else:
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
        keep: set[str] | None = None,
    ) -> None:
        """Remove intermediate temp files, keep only final output.

        Runs for both COMPLETED and FAILED stages -- a failed stage can still have
        written a partial/temp output file before erroring, and with the default
        skip_stage_on_error=True those would otherwise be orphaned permanently next
        to the user's source video.

        ``keep`` paths (e.g. a surviving intermediate that final-output resolution
        may still promote to job.output_path) are left alone.
        """
        keep = keep or set()
        for name, result in stage_results.items():
            if result.status in (StageStatus.COMPLETED, StageStatus.FAILED) and result.output_path:
                if result.output_path != job.output_path and result.output_path not in keep:
                    if os.path.exists(result.output_path):
                        try:
                            os.remove(result.output_path)
                        except OSError:
                            pass

    def _cleanup_generated_temp_paths(
        self, paths: list[str], job: Job, keep: set[str] | None = None
    ) -> None:
        """Remove every path generate_temp_path() handed out this job.

        Complements ``_cleanup_temp_files`` (which only knows about paths recorded
        in a COMPLETED/FAILED StageResult): a stage that fails before returning a
        result, or a job cancelled mid-stage, can still have written a partial temp
        file to a path we generated. Never removes the job's final output_path.

        ``keep`` paths (e.g. current_path, which final-output resolution may still
        promote to job.output_path) are left alone.
        """
        keep = keep or set()
        for path in paths:
            if path == job.output_path or path in keep:
                continue
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _promote_temp_to_output(self, temp_path: str, output_path: str) -> None:
        """Move a surviving intermediate temp file onto the job's final output path.

        Used when the terminal stage was skipped or produced no output but an
        earlier stage's temp file is the best available result: without this, the
        unconditional temp cleanup elsewhere would delete the only copy of that
        output before it ever reached the user-visible output path.
        """
        try:
            os.replace(temp_path, output_path)
        except OSError:
            # os.replace can fail across filesystems (EXDEV); shutil.move falls
            # back to copy+delete in that case.
            shutil.move(temp_path, output_path)

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
