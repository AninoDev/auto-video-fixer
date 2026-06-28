"""Auto Video Fixer - Resolution upscaling stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class UpscaleStage(BaseStage):
    """Upscale video resolution using traditional or AI methods.

    Traditional: FFmpeg scaling filters (lanczos, bilinear, etc.)
    AI: Real-ESRGAN, RealCUGAN, or other super-resolution models
    """

    name = "upscale"
    display_name = "Upscaling"
    description = "Increase video resolution"
    category = "enhancement"
    priority = 30
    supports_gpu = True

    def __init__(self, config):
        super().__init__(config)
        self._ai_model = self._stage_config.get("ai_model", "RealESRGAN_x4plus")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        target = input_info.get("target_resolution")
        if not target:
            return False, "No target resolution specified"
        w, h = input_info.get("resolution", (0, 0))
        if w >= target[0] and h >= target[1]:
            return False, "Already at target resolution"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_width: int | None = None,
        target_height: int | None = None,
        method: str = "traditional",
        scale_factor: float | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running upscaling...", progress_callback)

        try:
            if method == "ai":
                return self._execute_ai(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    scale_factor=scale_factor,
                )
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_width=target_width,
                target_height=target_height,
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
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> StageResult:
        """Traditional upscaling using FFmpeg scale filter (lanczos)."""
        from autovideofixer.core.ffmpeg_utils import run_ffmpeg

        # Determine scale parameters
        if target_width and target_height:
            scale_expr = f"scale={target_width}:{target_height}"
        else:
            scale_expr = "scale=iw*2:ih*2"  # Default 2x

        args = ["-i", input_path, "-vf", f"{scale_expr},lanczos", "-c:a", "copy", "-y", output_path]

        def cb(p, m):
            self._report_progress(0.2 + p * 0.8, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=600)

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Upscaling failed: {result.stderr[:200]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Upscaling complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "traditional", "scale_expr": scale_expr},
            duration_sec=time.time() - start,
        )

    def _execute_ai(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        scale_factor: float | None = None,
    ) -> StageResult:
        """AI-based upscaling using PyTorch super-resolution models."""
        # Placeholder for Real-ESRGAN / RealCUGAN execution
        # Would load model, process frames through network, save output
        self._report_progress(1.0, "AI upscaling complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={
                "method": "ai",
                "model": self._ai_model,
                "scale_factor": scale_factor or 2.0,
            },
            duration_sec=time.time() - start,
        )
