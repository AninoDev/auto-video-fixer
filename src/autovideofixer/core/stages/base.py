"""Auto Video Fixer - Base stage definition and registry."""

from __future__ import annotations

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

    @property
    def logger(self):
        if self._logger is None:
            from autovideofixer.logger import get_logger

            self._logger = get_logger(f"autovideofixer.stages.{self.name}")
        return self._logger

    def is_enabled(self) -> bool:
        """Check if this stage is enabled in config."""
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

    def cleanup(self, path: str | None = None) -> None:
        """Clean up temporary files created by this stage."""
        import os

        p = path or self._tmp_path
        if p and os.path.exists(p):
            os.remove(p)

    def _report_progress(
        self,
        progress: float,
        message: str,
        callback: Callable[[float, str], None] | None,
    ) -> None:
        if callback:
            callback(min(1.0, max(0.0, progress)), message)


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
