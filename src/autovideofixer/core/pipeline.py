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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from autovideofixer.config import (
    Config,
    deep_merge,
    redact_secrets,
    validate_output_handling_config,
)
from autovideofixer.core.ffmpeg_utils import (
    generate_temp_path,
    get_video_info,
)
from autovideofixer.core.output_check import check_output_spec, effective_output_targets
from autovideofixer.core.reporting import classify_stage, format_media_info_lines
from autovideofixer.core.stages.base import (
    StageResult,
    StageStatus,
    create_stage,
    get_stage,
)

# Fallback when config `pipeline.default_order` is missing/empty -- mirrors
# Config.DEFAULTS["pipeline"]["default_order"] exactly (see config.py for the
# crop-first / deblock-before-stabilize rationale). Kept as a plain
# list[str] since DEFAULTS itself never uses mapping entries.
DEFAULT_STAGE_ORDER: list[str] = [
    "detect",
    "crop",
    "downscale",
    "deblock",
    "stabilize",
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


@dataclass
class _ExistingOutputDecision:
    """Return value of ``Pipeline._decide_existing_output()``.

    ``terminal`` is a terminal JobResult (SKIPPED/FAILED) the caller must
    return immediately, or ``None`` when the job should proceed normally.
    ``reprocessed_mismatch`` is only meaningful when ``terminal is None``: it
    threads the § 6.2 "a verified mismatch was reprocessed" fact out to
    ``execute_job()``'s eventual completed-job JobResult (see
    ``JobResult.reprocessed_mismatch``) -- this small dataclass exists
    instead of an out-param specifically so that fact survives past this
    method's return without execute_job() having to re-derive it.
    """

    terminal: "JobResult | None"
    reprocessed_mismatch: bool = False


class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # New first-class outcome (REQUIREMENTS.md § 6.1): an existing output with
    # overwrite disabled is no longer classified as FAILED -- see
    # JobResult.outcome/skip_reason and Pipeline._decide_existing_output().
    SKIPPED = "skipped"


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
    # --- REQUIREMENTS.md § 6.1/6.2/6.3 outcome model ---
    # Canonical tri-state outcome; ``success`` is kept in sync for backward
    # compat (``success == (outcome == "completed")``, enforced in
    # __post_init__ below). Leave unset ("") at construction time to have it
    # derived from ``success`` -- existing call sites that only ever set
    # ``success`` (e.g. the main stage-loop result, execute_all()'s generic
    # exception wrapper) don't need updating.
    outcome: str = ""  # "completed" | "failed" | "skipped"
    # Machine-readable sub-reason, populated only when outcome == "skipped":
    # "output-exists" | "output-exists-mismatched" | "invalid-input".
    skip_reason: str | None = None
    # Human-readable record of the § 6.1/6.2 decision trail for this job (what
    # was measured vs. targeted, which flags drove the outcome, rename
    # actions) -- populated even when the decision was trivial ("no
    # conflicting existing output"). This is what the future § 6.6 JSON report
    # serializes verbatim; kept on every JobResult (not just SKIPPED ones) so
    # that reporting work has a uniform field to read.
    decision_log: list[str] = field(default_factory=list)
    # --- REQUIREMENTS.md § 6.4/6.5 reporting fields (core/reporting.py reads
    # these; populated on EVERY JobResult return path, including every § 6.1-
    # 6.3 early terminal) ---
    # Scene mode (scenes.enabled) stats for this job, or None when scene mode
    # didn't run: {"total": int, "kept": int, "dropped": int,
    # "dropped_detail": [...]} (detail = the existing per-scene dropped dicts).
    scene_stats: dict[str, Any] | None = None
    # Wall-clock ms from the moment this job's turn started (INCLUDING the §
    # 6.1/6.2 decision phase and input probing) to result finalization.
    # Canonical ms number for reporting; `total_duration` (seconds) above is
    # kept working as-is for existing callers -- it covers only the
    # stage-pipeline portion, same scope as `processing_ms` below (just
    # seconds instead of ms, and 0.0 for early-terminal jobs that never got
    # that far -- see the `total_time` default in execute_job()).
    job_wall_ms: float = 0.0
    # Wall-clock ms for the stage-pipeline portion only (scene-mode
    # preprocessing counts as processing; the § 6.1/6.2 exists/probe decision
    # phase does not). Always 0.0 for a SKIPPED or early-FAILED job -- no
    # stage ever ran.
    processing_ms: float = 0.0
    # True when the § 6.2 path found an existing output that mismatched
    # effective targets and reprocessed it (rename-or-overwrite branches) --
    # this job otherwise completed normally; it's just noted as reprocessed
    # per § 6.4's "reprocessed-mismatch jobs are ordinary completed jobs
    # noted as reprocessed" requirement.
    reprocessed_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.outcome:
            self.success = self.outcome == "completed"
        else:
            self.outcome = "completed" if self.success else "failed"

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
        validate_output_handling_config(self.config)
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

        # REQUIREMENTS.md § 6.7: register this job's real input/output paths
        # with the PII-clean-log singleton -- covers GUI/programmatic use
        # (the CLI's `process` command also registers resolved input files
        # directly; registration is idempotent, so the redundancy is
        # harmless and keeps both call sites simple).
        from autovideofixer.logclean import get_pii_cleaner

        cleaner = get_pii_cleaner()
        cleaner.register_input(input_path)
        cleaner.register_output(output_path)
        # Also register the parent directories (role-based placeholders) --
        # without this, a GUI/programmatic run (which never goes through the
        # CLI's directory registration) would substitute the filename but
        # leak the real directory path around it in a clean log.
        # Both the verbatim dirname (what clean() composes full-path
        # substitutions from) and the absolute one (what other log lines may
        # render) -- register_directory() is idempotent, dupes are free.
        for d in {os.path.dirname(input_path), os.path.dirname(os.path.abspath(input_path))}:
            if d:
                cleaner.register_directory(d, "input")
        if output_path:
            for d in {
                os.path.dirname(output_path),
                os.path.dirname(os.path.abspath(output_path)),
            }:
                if d:
                    cleaner.register_directory(d, "output")

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

    def auto_determine_stages(
        self, job: Job, input_info: dict[str, Any] | None = None
    ) -> list[str]:
        """Determine optimal processing stages for a job based on input and settings.

        Uses quality targets, input analysis, and preset preferences.

        ``input_info``: an already-probed info dict, when the caller (e.g.
        ``execute_job()``) has one on hand -- avoids a redundant re-probe of
        the same file. Probes ``job.input_path`` itself (raises ``RuntimeError``
        on an unreadable input -- see ``core/ffmpeg_utils.probe()`` and
        ``docs/REQUIREMENTS.md`` § 6.3) when omitted, e.g. direct callers/tests.
        """
        if input_info is None:
            input_info = get_video_info(job.input_path)
        stages = []

        # 1. Analysis stage (always first)
        stages.append("detect")

        # 2. Enhancement stages based on input properties. Note: resolution is
        # deliberately NOT read here for the upscale membership decision --
        # see the comment above `if target_resolution:` below for why a
        # plan-time geometry check would be wrong (crop can shrink the frame
        # after this plan is built).
        framerate = input_info.get("framerate", 0)
        is_hdr = input_info.get("is_hdr", False)

        # Check quality target from config
        quality_target = self.config.get("quality", "quality_target", default={})
        target_resolution = quality_target.get("target_resolution")
        target_framerate = quality_target.get("target_framerate")

        # Upscale: include it in the plan whenever a target resolution is
        # configured at all -- do NOT gate membership on comparing this
        # up-front probe's resolution against target_resolution. Two reasons
        # a plan-time geometry check here is wrong even though it looks like
        # an obvious "skip if already big enough" optimization:
        # 1. A later stage (crop) can shrink the frame AFTER this plan is
        #    built (e.g. a 1920x1080 input with pillarboxed 9:16 content
        #    crops down to 608x1080) -- a plan-time check against the
        #    ORIGINAL resolution would exclude "upscale" from job.stages
        #    entirely, and the in-loop should_run() (which now sees
        #    freshly-reprobed post-crop geometry, per the per-stage-reprobe
        #    fix above) never even gets a chance to run, since a stage
        #    dropped from the plan is never instantiated at all.
        # 2. This up-front resolution/target_resolution comparison is also
        #    orientation-blind (raw target, not rotated to the input's
        #    orientation) -- UpscaleStage.should_run()/_effective_target_bounds()
        #    is the single authoritative, orientation-aware implementation of
        #    "is this already at target"; duplicating a cruder version of
        #    that logic here to decide membership can only produce a
        #    stricter (over-excluding) answer than should_run() itself.
        # should_run() (called every time the loop reaches this occurrence,
        # against current -- possibly post-crop -- geometry) remains the
        # authoritative run/skip decision; a real "already at target, no crop
        # involved" input just gets a should_run()-level SKIPPED instead of
        # never being planned, which is harmless and the same outcome either
        # way for that case.
        if target_resolution:
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

        # Downscale -- opt-in, off by default (see Config.DEFAULTS["stages"]["downscale"]
        # and docs/REQUIREMENTS.md § 7). pipeline.default_order places it right after
        # "crop" regardless of insertion order here.
        downscale_config = self.config.get("stages", "downscale", default={})
        if downscale_config.get("enabled", False):
            stages.append("downscale")

        # Final encoding
        stages.append("encode")

        # Apply user overrides
        if job.stages:
            stages = job.stages

        return stages

    # Keys the pipeline itself layers onto a freshly-probed input_info dict, on
    # top of whatever get_video_info()/probe().to_info_dict() returns (see
    # execute_job()'s "target_format" injection, from general.target_format).
    # Every re-probe (per-stage and the scene-mode one) must carry these
    # forward onto the new probe dict, since a fresh probe never sets them
    # itself. Extend this tuple, not the re-probe call sites, if a future
    # change injects another key.
    _INJECTED_INPUT_INFO_KEYS: tuple[str, ...] = ("target_format",)

    def _reprobe_input_info(
        self, path: str, previous_info: dict[str, Any], context: str
    ) -> dict[str, Any]:
        """Re-probe ``path`` and carry forward any pipeline-injected keys from
        ``previous_info``. Fails open: a probe error is logged at WARNING and
        ``previous_info`` is returned unchanged, matching the codebase's
        fail-open convention for auxiliary info (this must never fail the job).
        """
        try:
            fresh_info = get_video_info(path)
        except Exception as e:
            self.logger.warning(
                "Failed to re-probe %s (%s); keeping previous input_info: %s",
                path,
                context,
                e,
            )
            return previous_info
        for key in self._INJECTED_INPUT_INFO_KEYS:
            if key in previous_info:
                fresh_info[key] = previous_info[key]
        return fresh_info

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
        2. Auto-crop right after detection, before every other enhancement
           stage: stabilize's zoom is now a real percentile-based
           "borderless" zoom, so it no longer leaves a black border for a
           later crop to clean up -- cropping first instead means every
           downstream stage (especially the AI-capable ones) sizes off the
           already-cropped content and never wastes compute on pixels that
           would just be cropped away.
        3. Downscale right after crop, before the heavier deblock/denoise/
           upscale/interpolate stages, for the same reason.
        4. Deblocking before stabilization (compression artifacts come from
           the source video; deblocking a not-yet-warped frame keeps the
           deblock model's input accurate, and gives the stabilizer cleaner
           detail to track) -- deblock's default model now also doubles as
           a denoise pass, which is why denoise_video defaults to disabled.
        5. Denoising (if enabled) before upscaling (don't upscale noise)
        6. Upscaling before interpolation (higher res frames interpolate better)
        7. Normalization near the end
        8. Encoding last
        """
        return [entry.label for entry in self.resolve_stage_order(stages)]

    def _finish_terminal(self, job: Job, result: JobResult, job_start: float) -> JobResult:
        """Common bookkeeping for an early (pre-stage-loop) terminal JobResult:
        set job.status/result/progress, stamp § 6.5 timing fields, and return
        it. Shared by every § 6.1/6.2/6.3 early-return path below.

        Every caller of this helper is a terminal that never reached the
        stage loop, so ``processing_ms`` is always 0 here; ``job_wall_ms`` is
        the real wall-clock elapsed since the job's turn started
        (``job_start``, a ``time.monotonic()`` timestamp) -- includes input
        probing and the § 6.1/6.2 decision phase, per § 6.5.
        """
        result.job_wall_ms = (time.monotonic() - job_start) * 1000.0
        result.processing_ms = 0.0
        job.status = (
            PipelineStatus.SKIPPED if result.outcome == "skipped" else PipelineStatus.FAILED
        )
        job.result = result
        job.progress = 1.0
        return result

    def _terminal_probe_failure_result(
        self, job: Job, error: Exception, job_start: float
    ) -> JobResult:
        """§ 6.3: an input ffprobe that can't analyze the file at all (raised
        RuntimeError, see core/ffmpeg_utils.probe()). FAILS the job by default
        (ffprobe stderr -- surfaced via ``error``'s message -- in the log and
        JobResult.errors); general.skip_invalid_inputs makes it SKIPPED
        instead (sub-reason "invalid-input") so a batch with known-bad
        members still completes. Used both when auto_determine_stages()'s own
        probe fails and when execute_job()'s explicit probe fails, so the
        policy is uniform regardless of which one hits the bad file first.
        """
        skip_invalid = self.config.get("general", "skip_invalid_inputs", default=False)
        msg = f"Input probe failed for {job.input_path}: {error}"
        if skip_invalid:
            self.logger.warning(msg)
            result = JobResult(
                input_path=job.input_path,
                output_path=None,
                outcome="skipped",
                skip_reason="invalid-input",
                decision_log=[
                    f"input probe failed ({error}); general.skip_invalid_inputs=true -> SKIPPED"
                ],
            )
        else:
            self.logger.error(msg)
            result = JobResult(
                input_path=job.input_path,
                output_path=None,
                errors=[msg],
                outcome="failed",
                decision_log=[
                    f"input probe failed ({error}); general.skip_invalid_inputs=false -> FAILED"
                ],
            )
        return self._finish_terminal(job, result, job_start)

    def _handle_probe_warning(
        self, job: Job, input_info: dict[str, Any], probe_stderr: str, job_start: float
    ) -> JobResult | None:
        """§ 6.3: a *successful* probe with non-empty ffprobe stderr. Always
        surfaced as a WARNING (job continues); general.fail_on_probe_warnings
        promotes it to a hard FAILED instead. Returns the terminal JobResult
        in the strict-mode case, else None (caller should proceed)."""
        fail_on_warnings = self.config.get("general", "fail_on_probe_warnings", default=False)
        if not fail_on_warnings:
            self.logger.warning(
                "ffprobe reported warnings for %s (job continues): %s",
                job.input_path,
                probe_stderr,
            )
            return None
        msg = f"ffprobe reported warnings for {job.input_path}: {probe_stderr}"
        self.logger.error(msg)
        result = JobResult(
            input_path=job.input_path,
            output_path=None,
            errors=[msg],
            input_info=input_info,
            outcome="failed",
            decision_log=[
                f"input probe succeeded with warnings ({probe_stderr}); "
                "general.fail_on_probe_warnings=true -> FAILED"
            ],
        )
        return self._finish_terminal(job, result, job_start)

    def _rename_mismatched_output(
        self, output_path: str, suffix: str, max_renames: int | None
    ) -> str | None:
        """§ 6.2 rename-mismatched-output helper: renames ``output_path`` to
        ``<stem><suffix><N><ext>``, N starting at 1, first unused name wins.
        Returns the new path on success. Returns None when renaming is
        disabled (``max_renames == 0``) or exhausted (every N up to
        ``max_renames`` is already taken) -- caller treats that as FAILED.
        ``max_renames=None`` means unlimited.
        """
        if max_renames is not None and max_renames <= 0:
            return None
        stem, ext = os.path.splitext(output_path)
        n = 1
        while max_renames is None or n <= max_renames:
            candidate = f"{stem}{suffix}{n}{ext}"
            if not os.path.exists(candidate):
                os.rename(output_path, candidate)
                # REQUIREMENTS.md § 6.7: the renamed file is a real output
                # path this run created -- register it so a clean log can
                # still show a placeholder for it, same as any other output.
                from autovideofixer.logclean import get_pii_cleaner

                get_pii_cleaner().register_output(candidate)
                return candidate
            n += 1
        return None

    def _decide_existing_output(
        self,
        job: Job,
        input_info: dict[str, Any],
        stage_entries: list[StageOrderEntry],
        decision_log: list[str],
        job_start: float,
    ) -> _ExistingOutputDecision:
        """REQUIREMENTS.md § 6.1/6.2: classify an existing ``job.output_path``
        as SKIPPED/FAILED, or clear the way for normal reprocessing.

        Appends every decision made (including the trivial "nothing to
        decide" case) to ``decision_log`` in place, so it can be attached to
        whichever JobResult eventually gets built -- a terminal one returned
        directly from here, or the normal end-of-pipeline one built later in
        execute_job() when ``.terminal`` is None.

        Returns an ``_ExistingOutputDecision`` whose ``.terminal`` is a
        terminal JobResult (SKIPPED or FAILED) the caller must return
        immediately without running any stage, or None when the job should
        proceed normally (no conflicting existing output, or a mismatch
        that's being reprocessed -- on a successful rename, the old file is
        already moved out of the way by the time this returns);
        ``.reprocessed_mismatch`` is True iff proceeding normally is because a
        verified mismatch is being reprocessed (§ 6.4's "reprocessed" note).
        """
        overwrite = self.config.get("general", "overwrite", default=False)
        if (
            overwrite
            or not job.output_path
            or not os.path.exists(job.output_path)
            or os.path.abspath(job.output_path) == os.path.abspath(job.input_path)
        ):
            decision_log.append("no conflicting existing output; proceeding")
            return _ExistingOutputDecision(terminal=None)

        existing_output = self.config.get("general", "existing_output", default="skip")
        decision_log.append(f"output already exists at {job.output_path} (general.overwrite=false)")

        if existing_output == "fail":
            # Old behavior, exactly: FAILED + ERROR log.
            msg = f"Output already exists and general.overwrite is False: {job.output_path}"
            self.logger.error(msg)
            decision_log.append("general.existing_output=fail -> FAILED (legacy behavior)")
            return _ExistingOutputDecision(
                terminal=self._finish_terminal(
                    job,
                    JobResult(
                        input_path=job.input_path,
                        output_path=None,
                        errors=[msg],
                        input_info=input_info,
                        outcome="failed",
                        decision_log=list(decision_log),
                    ),
                    job_start,
                )
            )

        check_existing = self.config.get("general", "check_existing_target", default=True)
        if not check_existing:
            self.logger.info(
                "Output already exists at %s; skipping (general.check_existing_target=false, "
                "not verified against targets)",
                job.output_path,
            )
            decision_log.append(
                "general.check_existing_target=false -> SKIPPED (not verified against targets)"
            )
            return _ExistingOutputDecision(
                terminal=self._finish_terminal(
                    job,
                    JobResult(
                        input_path=job.input_path,
                        output_path=job.output_path,
                        input_info=input_info,
                        outcome="skipped",
                        skip_reason="output-exists",
                        decision_log=list(decision_log),
                    ),
                    job_start,
                )
            )

        # Spec-check the existing output against the job's effective targets.
        targets = effective_output_targets(self.config, job, stage_entries=stage_entries)
        try:
            existing_info = get_video_info(job.output_path)
        except RuntimeError as e:
            existing_info = None
            decision_log.append(f"existing output unreadable while spec-checking: {e}")
        matches, mismatch_reasons = check_output_spec(existing_info, targets)

        if matches:
            self.logger.info(
                "Output already exists at %s and matches effective targets; skipping",
                job.output_path,
            )
            decision_log.append("existing output verified against targets: match -> SKIPPED")
            return _ExistingOutputDecision(
                terminal=self._finish_terminal(
                    job,
                    JobResult(
                        input_path=job.input_path,
                        output_path=job.output_path,
                        input_info=input_info,
                        outcome="skipped",
                        skip_reason="output-exists",
                        decision_log=list(decision_log),
                    ),
                    job_start,
                )
            )

        reason_str = "; ".join(mismatch_reasons)
        reprocess = self.config.get("general", "reprocess_mismatched", default=False)
        if not reprocess:
            self.logger.warning(
                "Output already exists at %s but MISMATCHED effective targets (%s); "
                "general.reprocess_mismatched is false -- skipping without reprocessing",
                job.output_path,
                reason_str,
            )
            decision_log.append(
                f"existing output mismatched ({reason_str}); "
                "general.reprocess_mismatched=false -> SKIPPED"
            )
            return _ExistingOutputDecision(
                terminal=self._finish_terminal(
                    job,
                    JobResult(
                        input_path=job.input_path,
                        output_path=job.output_path,
                        input_info=input_info,
                        outcome="skipped",
                        skip_reason="output-exists-mismatched",
                        decision_log=list(decision_log),
                    ),
                    job_start,
                )
            )

        existing_mismatched = self.config.get("general", "existing_mismatched", default="rename")
        if existing_mismatched == "overwrite":
            self.logger.warning(
                "Output already exists at %s but mismatched effective targets (%s); "
                "reprocessing (general.existing_mismatched=overwrite, old file will be replaced)",
                job.output_path,
                reason_str,
            )
            decision_log.append(
                f"existing output mismatched ({reason_str}); reprocessing "
                "(general.existing_mismatched=overwrite; old file will be replaced)"
            )
            # Proceed normally -- the stage loop writes over job.output_path.
            return _ExistingOutputDecision(terminal=None, reprocessed_mismatch=True)

        # rename mode (default): never overwrite -- move the old mismatched
        # file aside first, then proceed with a normal reprocessing run.
        suffix = self.config.get("general", "mismatched_rename_suffix", default="_mismatched-")
        max_renames = self.config.get("general", "mismatched_max_renames", default=None)
        new_path = self._rename_mismatched_output(job.output_path, suffix, max_renames)
        if new_path is None:
            msg = (
                f"Output already exists at {job.output_path} but mismatched effective "
                f"targets ({reason_str}); renaming is disabled or exhausted "
                f"(general.mismatched_max_renames={max_renames!r}) -- failing rather than "
                "silently overwriting or skipping"
            )
            self.logger.error(msg)
            decision_log.append(
                f"existing output mismatched ({reason_str}); rename disabled/exhausted "
                f"(general.mismatched_max_renames={max_renames!r}) -> FAILED"
            )
            return _ExistingOutputDecision(
                terminal=self._finish_terminal(
                    job,
                    JobResult(
                        input_path=job.input_path,
                        output_path=None,
                        errors=[msg],
                        input_info=input_info,
                        outcome="failed",
                        decision_log=list(decision_log),
                    ),
                    job_start,
                )
            )

        self.logger.warning(
            "Output already exists at %s but mismatched effective targets (%s); renamed to "
            "%s; reprocessing",
            job.output_path,
            reason_str,
            new_path,
        )
        decision_log.append(
            f"existing output mismatched ({reason_str}); renamed "
            f"{job.output_path} -> {new_path}; reprocessing"
        )
        return _ExistingOutputDecision(terminal=None, reprocessed_mismatch=True)

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

        # § 6.5 timing: the moment this job's turn starts -- job_wall_ms
        # (populated on EVERY return path below, including every early
        # terminal) is measured from here. monotonic() so a system clock
        # adjustment mid-run can't produce a negative/nonsensical duration.
        job_start = time.monotonic()

        # A caller-provided (e.g. CLI --stage) job.stages means the user is
        # explicitly naming which stages to run, replacing the preset/auto-determined
        # list -- captured before auto-determination fills job.stages in, so this
        # stays False for the preset/default path. Threaded onto each stage instance
        # below so should_run() bypasses that stage's `enabled: false` config instead
        # of silently skipping a stage the user explicitly asked for.
        explicit_stage_request = bool(job.stages)

        # Determine stages if not specified. auto_determine_stages() probes
        # job.input_path itself (see its docstring) -- § 6.3 probe policy
        # applies here too (an unreadable input must not crash the whole
        # batch, see execute_all()'s per-job wrapper / general.skip_invalid_inputs).
        if not job.stages:
            try:
                job.stages = self.auto_determine_stages(job)
            except RuntimeError as e:
                return self._terminal_probe_failure_result(job, e, job_start)

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
            return self._finish_terminal(job, job_result, job_start)

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
            return self._finish_terminal(job, job_result, job_start)

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

        # § 6.3 input-probe policy: an unreadable input FAILS the job by
        # default (ffprobe stderr surfaced), or -- with
        # general.skip_invalid_inputs -- is SKIPPED instead so a batch with
        # known-bad members still completes. Applied uniformly regardless of
        # whether job.stages was explicit or auto-determined above.
        try:
            input_info = get_video_info(job.input_path)
        except RuntimeError as e:
            return self._terminal_probe_failure_result(job, e, job_start)
        job.input_info = input_info

        # REQUIREMENTS.md § 6.7: register the embedded video title (if any)
        # as a KNOWN value the PII cleaner can substitute -- registered here,
        # right after the probe that produced it, so it's available before
        # any later log line could echo it verbatim. Does not itself log the
        # title anywhere new.
        from autovideofixer.logclean import get_pii_cleaner

        get_pii_cleaner().register_title(input_info.get("title"))

        # § 6.5 media info: log the input's resolution/framerate/duration/
        # filesize/bitrate/codecs at job start, before any decision/stage runs.
        self.logger.info(
            "Input media info for %s: %s",
            os.path.basename(job.input_path),
            "; ".join(format_media_info_lines(input_info, None, job.input_path, None)),
        )

        # A *successful* probe can still have non-empty ffprobe stderr (see
        # core/ffmpeg_utils.probe()'s "-v error") -- always surfaced as a
        # WARNING; general.fail_on_probe_warnings promotes it to a hard
        # failure instead.
        probe_stderr = str(input_info.get("probe_stderr") or "").strip()
        if probe_stderr:
            terminal = self._handle_probe_warning(job, input_info, probe_stderr, job_start)
            if terminal is not None:
                return terminal

        # § 6.1/6.2 decision path: existing-output SKIPPED/FAILED/reprocess
        # classification. Placed here -- right after the input probe, before
        # any of the (potentially expensive) work below -- so a to-be-skipped
        # job never pays for scene-mode preprocessing or a stage run. Needs
        # stage_entries (resolved just above) to know whether "interpolate"
        # would actually run, for the framerate spec-check target.
        decision_log: list[str] = []
        decision = self._decide_existing_output(
            job, input_info, stage_entries, decision_log, job_start
        )
        if decision.terminal is not None:
            return decision.terminal
        reprocessed_mismatch = decision.reprocessed_mismatch

        # § 6.5 timing: the stage-pipeline portion starts here -- everything
        # above (input probing, the § 6.1/6.2 decision phase) is excluded
        # from `processing_ms` on purpose (it's included in `job_wall_ms`
        # instead); scene-mode preprocessing below DOES count as processing.
        processing_start = time.monotonic()

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
        scene_stats: dict[str, Any] | None = None

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
                # REQUIREMENTS.md § 6.4: scene-mode stats per video, stored on
                # JobResult for reporting (console/log/JSON) and future GUI use.
                scene_stats = {
                    "total": scene_result.total_scenes,
                    "kept": scene_result.kept_scenes,
                    "dropped": len(scene_result.dropped_scenes),
                    "dropped_detail": list(scene_result.dropped_scenes),
                }
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
                # properties instead of the original input's. Redundant with
                # (but harmless ahead of) the general per-stage re-probe in
                # the loop below, which would otherwise re-probe this same
                # file again on the very next stage's transition; doing it
                # here up front avoids that back-to-back double-probe.
                input_info = self._reprobe_input_info(
                    current_path, input_info, context="after scene mode"
                )

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

                    # REQUIREMENTS.md § 6.5: surface each stage's wall-clock
                    # duration at INFO right after it completes/fails (skips
                    # are logged above with 0 duration -- should_run() never
                    # ran the stage). classify_stage() (core/reporting.py)
                    # gives the same ran-ai/ran-traditional(-fallback)/failed
                    # bucket § 6.4's per-job table uses, so a `--verbose`-free
                    # run's log already has the provenance info, not just timing.
                    self.logger.info(
                        "Stage '%s' finished in %.2fs (status=%s, classification=%s)",
                        stage_name,
                        result.duration_sec,
                        result.status.value,
                        classify_stage(result),
                    )

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
                            if result.output_path != current_path:
                                # This stage produced a NEW file -- re-probe so any
                                # later stage's should_run()/execute() sees the
                                # actual current geometry/framerate/etc. instead of
                                # the ORIGINAL input's stale probe (e.g. crop
                                # shrinking 1080x1080 letterboxed content down to
                                # 1080x608 must be visible to upscale's
                                # should_run(), not just to its own execute()).
                                # Skipped when output_path == current_path (a
                                # passthrough result reusing the existing file) --
                                # nothing changed, so a re-probe would be wasted.
                                input_info = self._reprobe_input_info(
                                    result.output_path,
                                    input_info,
                                    context=f"after stage '{stage_name}'",
                                )
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

        # § 6.5 media info: `input_info` was already kept fresh throughout the
        # stage loop above (re-probed after every stage that produced a new
        # file -- see the per-stage re-probe comment near
        # `_reprobe_input_info`), so by this point it already describes
        # `current_path`/`final_output_path`'s actual specs (a promotion to
        # job.output_path is just a rename, not a content change). Reuse it
        # rather than spending a second ffprobe process on the same file.
        output_info: dict[str, Any] = dict(input_info) if final_output_path else {}

        # Build result
        job_result = JobResult(
            input_path=job.input_path,
            output_path=final_output_path,
            stage_results=stage_results,
            total_duration=total_time,
            errors=errors,
            skipped=skipped,
            input_info=input_info,
            output_info=output_info,
            success=len(errors) == 0,
            decision_log=decision_log,
            scene_stats=scene_stats,
            reprocessed_mismatch=reprocessed_mismatch,
            job_wall_ms=(time.monotonic() - job_start) * 1000.0,
            processing_ms=(time.monotonic() - processing_start) * 1000.0,
        )

        media_info_lines = format_media_info_lines(
            input_info, output_info, job.input_path, final_output_path
        )
        self.logger.info(
            "Media info (input -> output) for %s: %s",
            os.path.basename(job.input_path),
            "; ".join(media_info_lines),
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
                from autovideofixer.config import resolve_timeout
                from autovideofixer.core.quality import estimate_ssim_psnr

                target = quality_target.get("target")
                quality_timeout = resolve_timeout(
                    self.config.get("quality", "timeout", default=None), "quality.timeout"
                )
                quality_result = estimate_ssim_psnr(
                    job.input_path, final_output_path, target=target, timeout=quality_timeout
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
