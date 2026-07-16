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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)

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

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to hdr stage",
                duration_sec=time.time() - start,
            )

        try:
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            # HDR (PQ/HLG, bt2020) -> SDR (bt709) requires converting to linear
            # light before tonemapping, then converting back to the target
            # transfer/matrix/primaries. tonemap's own options are only
            # `tonemap`, `param`, `desat`, `peak` -- color-space tags must be
            # applied via zscale before/after, not passed to tonemap itself.
            filter_chain = (
                "zscale=t=linear:npl=100,"
                "format=gbrpf32le,"
                "zscale=p=bt709,"
                "tonemap=tonemap=hable:desat=0,"
                "zscale=t=bt709:m=bt709:r=tv,"
                "format=yuv420p"
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
                output_path,
            ]

            def cb(p, m):
                self._report_progress(0.3 + p * 0.7, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"HDR conversion failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "HDR conversion complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={"method": method},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
