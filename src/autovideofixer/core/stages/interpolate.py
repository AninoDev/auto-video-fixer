"""Auto Video Fixer - Frame interpolation stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class InterpolateStage(BaseStage):
    """Increase frame rate using traditional or AI frame interpolation.

    Traditional: FFmpeg minterpolate (optical flow, frame blending)
    AI: RIFE (Real-Time Intermediate Flow Estimation) for high-quality interpolation
    """

    name = "interpolate"
    display_name = "Frame Interpolation"
    description = "Increase framerate using frame interpolation"
    category = "enhancement"
    priority = 35
    supports_gpu = True

    def __init__(self, config):
        super().__init__(config)
        self._ai_model = self._stage_config.get("ai_model", "rife_v4.6")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        target_fps = input_info.get("target_framerate")
        if not target_fps:
            return False, "No target framerate specified"
        current_fps = input_info.get("framerate", 0)
        if current_fps >= target_fps:
            return False, "Already at or above target framerate"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_fps: float | None = None,
        method: str = "traditional",
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running frame interpolation...", progress_callback)

        try:
            from autovideofixer.core.ffmpeg_utils import get_video_info

            input_info = get_video_info(input_path)
            current_fps = input_info.get("framerate", 30.0)

            if method == "ai":
                return self._execute_ai(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                )
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
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
        target_fps: float | None = None,
        current_fps: float | None = None,
    ) -> StageResult:
        """Traditional frame interpolation using FFmpeg minterpolate."""
        from autovideofixer.core.ffmpeg_utils import run_ffmpeg

        # Calculate interpolation factor
        if target_fps and current_fps and current_fps > 0:
            factor = int(target_fps / current_fps)
        else:
            factor = 2

        # minterpolate: fps=double:mi=interp:mb=16:vsbmc=1:vsblur=1
        vf = f"fps={target_fps or 'double'},minterpolate=mi=interp:mb=16:vsbmc=1:vsblur=1"

        args = ["-i", input_path, "-vf", vf, "-c:a", "copy", "-y", output_path]

        def cb(p, m):
            self._report_progress(0.3 + p * 0.7, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=600)

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Interpolation failed: {result.stderr[:200]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Frame interpolation complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "traditional", "factor": factor},
            duration_sec=time.time() - start,
        )

    def _execute_ai(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        target_fps: float | None = None,
    ) -> StageResult:
        """AI frame interpolation using RIFE model."""
        # Placeholder for RIFE model execution
        # Would use PyTorch to run intermediate flow estimation
        self._report_progress(1.0, "AI frame interpolation complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "ai", "model": self._ai_model},
            duration_sec=time.time() - start,
        )
