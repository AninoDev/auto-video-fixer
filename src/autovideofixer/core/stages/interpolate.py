"""Auto Video Fixer - Frame interpolation stage."""

from __future__ import annotations

import os
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import run_ffmpeg
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
                    current_fps=current_fps,
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
        current_fps: float | None = None,
        **kwargs,
    ) -> StageResult:
        """AI frame interpolation using RIFE model.

        Falls back to traditional FFmpeg method if PyTorch or model
        files are not available.
        """
        # Calculate interpolation factor
        if target_fps and current_fps and current_fps > 0:
            factor = int(target_fps / current_fps)
        else:
            factor = 2

        if factor <= 1:
            factor = 2

        try:
            from autovideofixer.ai.frame_processor import FrameProcessor
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator
        except ImportError:
            self.logger.warning("PyTorch not available, falling back to traditional interpolation")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        if not is_torch_available():
            self.logger.warning("PyTorch not installed, falling back to traditional interpolation")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            success, msg = ensure_model_available(self._ai_model)
            if not success:
                self.logger.warning(f"Model not available ({msg}), falling back")
                return self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                )

        except Exception as e:
            self.logger.warning(f"Model check failed ({e}), falling back to traditional")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        # Load and run RIFE
        interpolator = RIFEInterpolator(model_name=self._ai_model)

        if not interpolator.load_model():
            self.logger.warning("Failed to load RIFE model, falling back")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        try:
            proc = FrameProcessor()
            frames = proc.extract_frames(input_path)

            if not frames:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames extracted from input video",
                    duration_sec=time.time() - start,
                )

            def cb(current, total, msg):
                self._report_progress(0.1 + (current / total) * 0.9, msg, progress_callback)

            interpolated = interpolator.interpolate_video(
                frames, factor=factor, progress_callback=cb
            )
            proc.close()

            if not interpolated:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames produced by interpolator",
                    duration_sec=time.time() - start,
                )

            # Write interpolated frames to temp file, then use FFmpeg to finalize
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_interp_{os.path.basename(input_path)}",
            )
            fps = self._get_input_fps(input_path)
            proc2 = FrameProcessor()
            if not proc2.frames_to_video(interpolated, temp_path, fps=fps):
                proc2.close()
                return StageResult(
                    status=StageStatus.FAILED,
                    error="Failed to write interpolated frames to temp file",
                    duration_sec=time.time() - start,
                )
            proc2.close()

            # Copy audio from original to interpolated temp file
            run_ffmpeg(
                [
                    "-i",
                    input_path,
                    "-i",
                    temp_path,
                    "-map",
                    "0:a:0",
                    "-map",
                    "1:v:0",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-c:a",
                    "copy",
                    "-y",
                    output_path,
                ],
                timeout=600,
            )

            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            self._report_progress(1.0, "AI frame interpolation complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "ai",
                    "model": self._ai_model,
                    "factor": factor,
                    "frames_in": len(frames),
                    "frames_out": len(interpolated),
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            self.logger.error(f"RIFE processing failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                error=f"AI interpolation failed: {e}",
                duration_sec=time.time() - start,
            )
        finally:
            interpolator.unload()

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
