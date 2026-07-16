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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)

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
            import os

            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            # See normalize_audio.py: don't read and -y-truncate the same path
            # simultaneously when no explicit output_path is given. The temp
            # filename's extension must match target_format -- ffmpeg infers the
            # output muxer from it, so a hardcoded ".mp4" here would silently mux
            # into the wrong container whenever target_format is anything else
            # (e.g. mkv/webm), even though the file gets renamed back correctly.
            in_place = output_path is None
            dest = f"{input_path}.remux_tmp.{target_format}" if in_place else output_path

            args = [
                "-i",
                input_path,
                "-c",
                "copy",  # Copy all streams
                "-map",
                "0",  # Map all streams
                "-y",
                dest,
            ]

            def cb(p, m):
                self._report_progress(0.5 + p * 0.5, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

            if result.returncode != 0:
                if in_place and os.path.exists(dest):
                    os.unlink(dest)
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Remuxing failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            if in_place:
                os.replace(dest, input_path)
                dest = input_path

            self._report_progress(1.0, "Remux complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=dest,
                metadata={"format": target_format},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
