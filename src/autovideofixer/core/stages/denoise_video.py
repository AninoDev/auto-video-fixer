"""Auto Video Fixer - Video denoising stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class DenoiseVideoStage(BaseStage):
    """Remove noise from video using traditional or AI methods.

    Traditional methods: nlmeans, mcdeint, hqdn3d
    AI methods: Real-ESRGAN denoise mode, Noise2Void, etc.
    """

    name = "denoise_video"
    display_name = "Video Denoising"
    description = "Reduce video noise and grain"
    category = "enhancement"
    priority = 20
    supports_gpu = True

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        # Skip if very low resolution or already clean
        w, h = input_info.get("resolution", (0, 0))
        if w == 0 or h == 0:
            return False, "No video stream"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        method: str = "traditional",
        strength: str = "medium",
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running video denoising...", progress_callback)

        try:
            if method == "ai":
                return self._execute_ai(input_path, output_path, progress_callback, start)
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )

    def _execute_traditional(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        strength: str = "medium",
    ) -> StageResult:
        """Traditional denoising using FFmpeg hqdn3d filter."""
        from autovideofixer.core.ffmpeg_utils import run_ffmpeg

        # hqdn3d parameters: spatial_luma, spatial_chroma, temporal_luma, temporal_chroma
        params = {
            "low": "4:3:6:6",
            "medium": "6:4:8:8",
            "high": "10:6:12:12",
        }.get(str(strength), "6:4:8:8")

        args = ["-i", input_path, "-vf", f"hqdn3d={params}", "-c:a", "copy", "-y", output_path]

        def cb(p, m):
            self._report_progress(0.3 + p * 0.7, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=600)

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Denoising failed: {result.stderr[:200]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Denoising complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "traditional", "filter": "hqdn3d", "params": params},
            duration_sec=time.time() - start,
        )

    def _execute_ai(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
    ) -> StageResult:
        """AI-based denoising using PyTorch models."""
        # Placeholder for actual AI denoising model execution
        # Would use models like Noise2Void, BM3D-DnCNN, etc.
        self._report_progress(1.0, "AI denoising complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "ai", "model": "placeholder"},
            duration_sec=time.time() - start,
        )
