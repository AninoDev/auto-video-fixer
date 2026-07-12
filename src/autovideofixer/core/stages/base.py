"""Auto Video Fixer - Base stage definition and registry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from autovideofixer.config import Config


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

    def __init__(self, config: Config):
        self.config = config
        self._stage_config = config.get("stages", self.name, default={})
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
            return traditional()
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


def create_stage(name: str, config: Config) -> BaseStage | None:
    """Instantiate a registered stage."""
    cls = get_stage(name)
    if cls is None:
        return None
    return cls(config)
