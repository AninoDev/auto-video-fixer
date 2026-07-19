"""Auto Video Fixer - Base stage definition and registry."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from autovideofixer.config import Config, deep_merge, resolve_timeout


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of a processing stage."""

    status: StageStatus
    output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_sec: float = 0.0
    skipped_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)


class BaseStage(ABC):
    """Abstract base class for all processing stages.

    Each stage processes an input video file and produces an output.
    Stages are modular and can be independently implemented, extended,
    or swapped via the registry.
    """

    # Override in subclasses
    name: str = "base"
    display_name: str = "Base Stage"
    description: str = ""
    requires_input: bool = True
    produces_output: bool = True
    supports_gpu: bool = False
    supports_hardware_encoding: bool = False
    priority: int = 50  # Lower = runs earlier
    category: str = "processing"  # analysis, enhancement, encoding, output

    # Capability flags
    can_parallelize: bool = False
    needs_intermediate: bool = False  # Must run on intermediate file, not source

    def __init__(self, config: Config, overrides: dict[str, Any] | None = None):
        """Create a stage instance.

        Args:
            config: The job's Config.
            overrides: Optional per-occurrence config overrides (see
                ``Pipeline.resolve_stage_order`` / ``pipeline.default_order``
                entries' ``config:`` key), deep-merged onto the cascaded
                ``stages.<name>`` dict for THIS instance only -- never
                mutates ``config``'s own data. When given, `self._stage_config`
                is a deep copy of the cascaded config with `overrides` merged
                on top (occurrence overrides win); when omitted, it's the
                same live-shared dict Config.get() returns, matching prior
                behavior exactly (no copy, no perf/behavior change).
        """
        self.config = config
        stage_config = config.get("stages", self.name, default={})
        if overrides:
            stage_config = copy.deepcopy(stage_config)
            deep_merge(stage_config, overrides)
        self._stage_config = stage_config
        self._logger = None
        # Set by Pipeline.execute_job() on stage instances belonging to a job whose
        # stage list was explicitly requested (e.g. CLI --stage), as opposed to
        # auto-determined/preset-derived. --stage's help text says it "replaces the
        # preset/auto-determined list", so a stage the user explicitly named must
        # run even if stages.<name>.enabled is False in config -- otherwise
        # should_run() silently skips it as "Stage disabled in configuration",
        # contradicting the CLI's documented behavior. Auto-determined/preset stage
        # lists are unaffected: they never set this, so is_enabled() keeps honoring
        # the config flag exactly as before.
        self._force_enabled = False

    @property
    def logger(self):
        if self._logger is None:
            from autovideofixer.logger import get_logger

            self._logger = get_logger(f"autovideofixer.stages.{self.name}")
        return self._logger

    def is_enabled(self) -> bool:
        """Check if this stage is enabled in config.

        Always True when the stage was explicitly requested by name (see
        ``_force_enabled``), regardless of the ``enabled`` config flag.
        """
        if self._force_enabled:
            return True
        return self._stage_config.get("enabled", True)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        """Determine if this stage should run given input metadata.

        Returns (should_run: bool, reason: str | None).
        Override in subclasses for intelligent skipping.
        """
        if not self.is_enabled():
            return False, "Stage disabled in configuration"
        return True, None

    @abstractmethod
    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        input_info: dict | None = None,
        **kwargs: Any,
    ) -> StageResult:
        """Execute the processing stage on the input file.

        Args:
            input_path: Path to input file
            output_path: Desired output path (may be overridden)
            progress_callback: Called with (progress 0-1, status_message)
            **kwargs: Additional stage-specific parameters

        Returns:
            StageResult with outcome
        """
        ...

    def estimate_complexity(self, input_info: dict[str, Any]) -> float:
        """Estimate processing complexity (higher = longer). Used for scheduling."""
        resolution = input_info.get("resolution", (1920, 1080))
        duration = input_info.get("duration", 0)
        base = resolution[0] * resolution[1] * duration / (1920 * 1080 * 60)
        return max(1.0, base)

    def get_ffmpeg_args(self, input_path: str, output_path: str, **kwargs) -> list[str]:
        """Build FFmpeg command-line arguments. Default no-ops (pass-through)."""
        return [input_path, "-y", output_path]

    def get_ffmpeg_filter_complex(self, **kwargs) -> str | None:
        """Build FFmpeg filter complex string. Default no filters."""
        return None

    def _report_progress(
        self,
        progress: float,
        message: str,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        if callback:
            callback(min(1.0, max(0.0, progress)), message)

    def stage_timeout(self) -> float | None:
        """Resolve this stage's ffmpeg timeout (seconds) for its MAIN
        processing/mux pass(es).

        Resolution order: ``stages.<name>.timeout`` (this stage's cascaded
        config -- including any per-occurrence ``pipeline.default_order``
        ``config:`` override, since ``self._stage_config`` already has that
        merged in by ``__init__``) -> ``pipeline.stage_timeout`` (global
        default) -> ``None`` (unlimited).

        Both levels share the same null-unlimited semantics: ``None``
        (absent/explicit ``null``) or ``0`` mean "no timeout"; a positive
        number is seconds; anything else (negative, non-numeric) raises
        ``ValueError`` here -- i.e. at the point a stage actually resolves
        its effective timeout ("at use time"), not eagerly at config-load
        time. See ``resolve_timeout()`` in ``config.py``.

        Callers pass the result straight through to
        ``core.ffmpeg_utils.run_ffmpeg(..., timeout=...)``, which accepts
        ``None`` to mean "wait forever" (``subprocess.Popen.wait(timeout=None)``
        blocks indefinitely).

        NOT for short, genuinely-bounded helper calls within a stage (probes,
        single-frame extraction, quick detection samples) -- those keep their
        own small fixed timeouts regardless of this resolution; only a
        stage's whole-video main processing/mux pass(es) should use this.
        """
        per_stage = self._stage_config.get("timeout", None)
        if per_stage is not None:
            return resolve_timeout(per_stage, f"stages.{self.name}.timeout")
        global_timeout = self.config.get("pipeline", "stage_timeout", default=None)
        return resolve_timeout(global_timeout, "pipeline.stage_timeout")

    def is_ai_fallback_enabled(self) -> bool:
        """Whether this stage may silently fall back to its traditional
        FFmpeg implementation when the AI path can't run.

        Resolution order: stages.<name>.ai_fallback (True/False) if set,
        else general.ai_fallback (default True). Both are populated from
        Config.DEFAULTS, so `.get()` with a default here is just defensive.
        """
        per_stage = self._stage_config.get("ai_fallback", None)
        if per_stage is not None:
            return bool(per_stage)
        return bool(self.config.get("general", "ai_fallback", default=True))

    def resolve_ai_method(self, explicit_method: str | None, auto_default: str) -> tuple[str, str]:
        """Resolve an AI-capable stage's method ("ai" or "traditional").

        Mirrors the ``ai_fallback`` resolution convention (see
        ``is_ai_fallback_enabled`` above) so all four AI-capable stages
        (upscale, deblock, denoise_video, interpolate) pick their method the
        same way. Precedence, highest first:

          1. ``explicit_method`` -- an explicit ``method=`` kwarg passed by a
             caller (e.g. scene mode, tests) wins outright, no matter what
             config says.
          2. ``stages.<name>.use_ai`` (True/False) if set (not null).
          3. ``general.use_ai`` (True/False, set by CLI ``--ai``/``--no-ai``)
             if set -- per this same convention, the global CLI flag does
             NOT override an explicit per-stage config value (step 2 already
             won if it applied).
          4. ``auto_default`` -- the stage's own hardcoded default (e.g.
             upscale/deblock default to "ai", denoise_video/interpolate
             default to "traditional" -- both deliberate, not oversights).

        Returns ``(method, source)`` -- ``source`` is a short human-readable
        string describing which of the above resolved it, used by
        ``_log_ai_method_choice`` for the INFO method-selection log line.
        """
        if explicit_method is not None:
            return explicit_method, "explicit method= argument"
        per_stage = self._stage_config.get("use_ai", None)
        if per_stage is True:
            return "ai", f"stages.{self.name}.use_ai: true"
        if per_stage is False:
            return "traditional", f"stages.{self.name}.use_ai: false"
        general_use_ai = self.config.get("general", "use_ai", default=None)
        if general_use_ai is True:
            return "ai", "--ai"
        if general_use_ai is False:
            return "traditional", "--no-ai"
        return auto_default, "auto default"

    def _log_ai_method_choice(
        self,
        method: str,
        source: str,
        ai_desc: str,
        traditional_desc: str,
        ai_hint: str,
    ) -> None:
        """Log, once per execution, which method an AI-capable stage chose
        and why -- at INFO, so it shows up in a normal ``--verbose`` run
        without needing DEBUG. The user was confused for months about which
        path a run actually took; a grep for "using traditional"/"using AI"
        must always explain it.

        Args:
            method: "ai" or "traditional" (the resolve_ai_method() result).
            source: The resolve_ai_method() source string.
            ai_desc: Stage-specific description used when method == "ai",
                e.g. "AI interpolation (RIFE 'rife_v4.6', backend torch)".
            traditional_desc: Stage-specific description used when
                method == "traditional", e.g. "traditional minterpolate".
            ai_hint: Short AI-technology name/version used only in the
                auto-default-traditional opt-in hint, e.g.
                "AI/RIFE model 'rife_v4.6'".
        """
        if method == "ai":
            self.logger.info("%s: using %s (selected by %s)", self.name, ai_desc, source)
            return
        if source == "auto default":
            self.logger.info(
                "%s: using %s (auto default; %s is configured but not selected -- "
                "set stages.%s.use_ai: true or pass --ai to use it)",
                self.name,
                traditional_desc,
                ai_hint,
                self.name,
            )
        else:
            self.logger.info("%s: using %s (%s)", self.name, traditional_desc, source)

    def _ai_fallback_or_fail(
        self,
        reason: str,
        start: float,
        traditional: Callable[[], "StageResult"],
    ) -> "StageResult":
        """Decide whether to fall back to the traditional method, or fail.

        Call this at every point an AI-capable stage's AI path can't
        proceed: torch missing, model load failure, an inference exception,
        or CUDA OOM after tiling retries are exhausted. NOT for a
        legitimate mid-retry step (e.g. OOM-triggered tiling retry itself
        is still the AI path, not a fallback decision) or for a genuine
        processing failure unrelated to AI availability (e.g. a mux/ffmpeg
        error after AI frames were already produced) -- those should keep
        failing outright regardless of ai_fallback.

        Args:
            reason: Human-readable cause, included in both the WARNING log
                (fallback enabled) and the failure error message (disabled).
            start: The stage's `execute()` start time (time.time()), used to
                compute duration_sec on the disabled-fallback failure path.
            traditional: Zero-arg callable that runs the stage's traditional
                implementation and returns its StageResult.
        """
        if self.is_ai_fallback_enabled():
            self.logger.warning(
                "Stage '%s': AI method unavailable (%s); falling back to traditional method",
                self.name,
                reason,
            )
            result = traditional()
            # REQUIREMENTS.md § 6.4: mark provenance HERE, the one seam every
            # AI->traditional fallback flows through, so reporting can tell
            # "chose traditional" from "AI failed, fell back to traditional"
            # without each stage's traditional() implementation needing to
            # know it was called as a fallback.
            result.metadata["ai_fallback_used"] = True
            result.metadata["ai_fallback_reason"] = reason
            return result
        self.logger.error(
            "Stage '%s': AI method unavailable (%s); ai_fallback is disabled, failing stage",
            self.name,
            reason,
        )
        return StageResult(
            status=StageStatus.FAILED,
            error=(
                f"AI processing unavailable for stage '{self.name}': {reason} "
                "(ai_fallback is disabled for this run -- enable general.ai_fallback or "
                f"stages.{self.name}.ai_fallback, or pass --ai-fallback, to allow falling "
                "back to the traditional method instead)"
            ),
            duration_sec=time.time() - start,
        )


# Stage Registry - maps stage names to classes
_STAGE_REGISTRY: dict[str, type[BaseStage]] = {}


def register_stage(cls: type[BaseStage]) -> type[BaseStage]:
    """Decorator to register a stage class."""
    _STAGE_REGISTRY[cls.name] = cls
    return cls


def get_stage(name: str) -> type[BaseStage] | None:
    """Look up a registered stage by name."""
    return _STAGE_REGISTRY.get(name)


def list_stages() -> dict[str, type[BaseStage]]:
    """Return all registered stages."""
    return dict(_STAGE_REGISTRY)


def create_stage(
    name: str, config: Config, overrides: dict[str, Any] | None = None
) -> BaseStage | None:
    """Instantiate a registered stage.

    ``overrides``, when given, is deep-merged onto the stage's cascaded
    ``stages.<name>`` config for this instance only (see
    ``BaseStage.__init__``) -- used for per-occurrence config from a
    ``pipeline.default_order`` mapping entry's ``config:`` key.
    """
    cls = get_stage(name)
    if cls is None:
        return None
    return cls(config, overrides)
