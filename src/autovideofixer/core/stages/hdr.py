"""Auto Video Fixer - HDR to SDR conversion stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class HDRStage(BaseStage):
    """Convert HDR content to SDR or enhance SDR content.

    Uses FFmpeg's color conversion filters for HDR->SDR mapping.
    """

    name = "hdr"
    display_name = "HDR Conversion"
    description = "Convert between HDR and SDR"
    category = "enhancement"
    priority = 60
    supports_gpu = False

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if not input_info.get("is_hdr"):
            return False, "Content is not HDR"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        method: str = "bt2020",
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Converting HDR to SDR...", progress_callback)

        try:
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            # Build HDR to SDR conversion filter chain
            # Uses PQ transfer function -> linear -> SDR transfer function
            filter_chain = (
                "tonemap=tonemap=hable:desat=0.1:"
                "format=yuv420p:matrix=bt2020:primaries=bt2020:transfer=bt709"
            )

            args = [
                "-i",
                input_path,
                "-vf",
                filter_chain,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "copy",
                "-y",
                output_path or input_path,
            ]

            def cb(p, m):
                self._report_progress(0.3 + p * 0.7, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=600)

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"HDR conversion failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "HDR conversion complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path or input_path,
                metadata={"method": method},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
