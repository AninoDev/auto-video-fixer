"""Auto Video Fixer - Remuxing stage (container change without re-encoding)."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class RemuxStage(BaseStage):
    """Change container format without re-encoding streams.

    Fast operation that just repackages existing streams into a new container.
    """

    name = "remux"
    display_name = "Remux"
    description = "Change container format without re-encoding"
    category = "output"
    priority = 80
    supports_hardware_encoding = False
    can_parallelize = True

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        current_format = input_info.get("format", "")
        target_format = input_info.get("target_format")
        if not target_format:
            return False, "No target format specified"
        if current_format.lower() == target_format.lower():
            return False, "Already in target format"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_format: str = "mp4",
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Remuxing...", progress_callback)

        try:
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            args = [
                "-i",
                input_path,
                "-c",
                "copy",  # Copy all streams
                "-map",
                "0",  # Map all streams
                "-y",
                output_path or input_path,
            ]

            def cb(p, m):
                self._report_progress(0.5 + p * 0.5, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=300)

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Remuxing failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Remux complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path or input_path,
                metadata={"format": target_format},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
