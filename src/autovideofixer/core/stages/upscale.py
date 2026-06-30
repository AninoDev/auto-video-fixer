"""Auto Video Fixer - Resolution upscaling stage."""

from __future__ import annotations

import os
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class UpscaleStage(BaseStage):
    """Upscale video resolution using traditional or AI methods.

    Default: AI upscaling via Real-ESRGAN for perceptual super-resolution.
    Traditional: FFmpeg scaling filters used primarily for downscaling.
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
        self._tt_mode = self._stage_config.get("tta_mode", 0)
        self._scale_factor = self._stage_config.get("scale_factor", 4)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        target = input_info.get("target_resolution")
        if not target:
            quality_target = self.config.get("quality", "quality_target", default={})
            target = quality_target.get("target_resolution")
        if not target:
            return False, "No target resolution specified"
        w, h = input_info.get("resolution", (0, 0))
        if w >= target[0] and h >= target[1]:
            return False, "Already at target resolution"
        self._input_info = input_info
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_width: int | None = None,
        target_height: int | None = None,
        method: str | None = None,
        scale_factor: float | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running upscaling...", progress_callback)

        if target_width is None or target_height is None:
            quality_target = self.config.get("quality", "quality_target", default={})
            tr = quality_target.get("target_resolution")
            if tr:
                target_width = target_width or tr[0]
                target_height = target_height or tr[1]

        if method is None:
            info = getattr(self, "_input_info", {})
            w, h = info.get("resolution", (0, 0))
            if target_width and w < target_width:
                method = "ai"
            else:
                method = "traditional"

        try:
            if method == "ai":
                return self._execute_ai(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    scale_factor=scale_factor,
                    target_width=target_width,
                    target_height=target_height,
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
        # Determine scale parameters (use lanczos scaling algorithm)
        if target_width and target_height:
            scale_expr = f"scale={target_width}:{target_height}:flags=lanczos"
        else:
            scale_expr = "scale=iw*2:ih*2:flags=lanczos"  # Default 2x

        vf_filter = f"{scale_expr},format=yuv420p"
        args = ["-i", input_path, "-vf", vf_filter, "-c:a", "copy", "-y", output_path]

        def cb(p, m):
            self._report_progress(0.2 + p * 0.8, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=600)

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Upscaling failed: {result.stderr[:2000]}",
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
        target_width: int | None = None,
        target_height: int | None = None,
        **kwargs,
    ) -> StageResult:
        """AI-based upscaling using Real-ESRGAN via PyTorch.

        Falls back to traditional FFmpeg method if PyTorch or model
        files are not available.

        Strategy: Chain multiple AI passes (each up to 4x) until reaching
        the target resolution. This preserves more detail than single-pass
        traditional scaling.
        """
        import math
        import tempfile

        # Real-ESRGAN max scale factor per pass
        max_ai_scale = 4

        # Calculate how many AI passes we need
        if target_width and target_height:
            input_w, input_h = self._get_input_resolution(input_path) if input_path else (0, 0)
            if input_w > 0 and input_h > 0:
                # Calculate total scale needed
                scale_w = target_width / input_w
                scale_h = target_height / input_h
                total_scale = max(scale_w, scale_h)

                # Calculate number of passes needed (each pass scales by up to max_ai_scale)
                num_passes = math.ceil(math.log(total_scale) / math.log(max_ai_scale)) if total_scale > 1 else 1
            else:
                num_passes = 1
        else:
            num_passes = 1

        # Run AI upscaling in passes
        current_input = input_path
        intermediate_files = []  # Track intermediate files for cleanup

        for pass_num in range(num_passes):
            is_last_pass = pass_num == num_passes - 1

            # Calculate scale for this pass
            if scale_factor is not None:
                sf = min(scale_factor, max_ai_scale)
            else:
                if is_last_pass and input_w > 0 and input_h > 0:
                    # Calculate exact scale needed for final pass
                    current_info = self._get_input_resolution(current_input)
                    if current_info and current_info[0] > 0:
                        scale_w = target_width / current_info[0]
                        scale_h = target_height / current_info[1]
                        sf = max(scale_w, scale_h)
                        # Round to nearest power of 2 for AI
                        sf = 2 ** round(math.log2(sf)) if sf > 1 else 2
                    else:
                        sf = max_ai_scale
                else:
                    sf = max_ai_scale

            # Create temp path for intermediate output
            if is_last_pass:
                pass_output = output_path
            else:
                # Use a consistent temp directory for intermediate files
                pass_output = os.path.join(
                    tempfile.gettempdir(),
                    f".avf_ai_pass{pass_num + 1}_{os.path.splitext(os.path.basename(input_path))[0]}.mp4",
                )
                intermediate_files.append(pass_output)

            # Run single AI pass
            result = self._run_single_ai_pass(
                current_input,
                pass_output,
                progress_callback,
                start,
                sf,
            )

            if result.status != StageStatus.COMPLETED:
                # Clean up intermediate files on failure
                for f in intermediate_files:
                    if os.path.exists(f):
                        os.unlink(f)
                return result

            current_input = pass_output

        # Clean up intermediate files
        for f in intermediate_files:
            if os.path.exists(f):
                os.unlink(f)

        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "ai", "model": self._ai_model, "passes": num_passes},
            duration_sec=time.time() - start,
        )

    def _run_single_ai_pass(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        scale_factor: float,
    ) -> StageResult:
        """Run a single AI upscaling pass."""
        try:
            from autovideofixer.ai.frame_processor import FrameProcessor
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            return StageResult(
                status=StageStatus.FAILED,
                error="PyTorch not available",
                duration_sec=time.time() - start,
            )

        if not is_torch_available():
            return StageResult(
                status=StageStatus.FAILED,
                error="PyTorch not installed",
                duration_sec=time.time() - start,
            )

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            success, msg = ensure_model_available(self._ai_model)
            if not success:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Model not available: {msg}",
                    duration_sec=time.time() - start,
                )

            upscaler = RealESRGANUpscaler(
                scale=int(scale_factor),
                model_name=self._ai_model,
                tta_mode=self._tt_mode,
            )

            if not upscaler.load_model():
                return StageResult(
                    status=StageStatus.FAILED,
                    error="Failed to load model",
                    duration_sec=time.time() - start,
                )

            proc = FrameProcessor()
            frames = proc.extract_frames(input_path)

            if not frames:
                proc.close()
                upscaler.unload()
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames extracted",
                    duration_sec=time.time() - start,
                )

            def cb(current, total, msg):
                self._report_progress(0.1 + (current / total) * 0.9, msg, progress_callback)

            upscaled = upscaler.upscale_video(frames, progress_callback=cb)
            proc.close()

            if not upscaled:
                upscaler.unload()
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames produced",
                    duration_sec=time.time() - start,
                )

            # Write upscaled frames to temp file first
            temp_path = os.path.join(
                os.path.dirname(output_path) or ".",
                f".avf_upscaled_{os.path.basename(output_path)}",
            )
            fps = self._get_input_fps(input_path)
            proc2 = FrameProcessor()
            if not proc2.frames_to_video(upscaled, temp_path, fps=fps):
                proc2.close()
                upscaler.unload()
                return StageResult(
                    status=StageStatus.FAILED,
                    error="Failed to write frames",
                    duration_sec=time.time() - start,
                )
            proc2.close()

            # Copy audio from input to output, using temp file as video source
            run_ffmpeg(
                [
                    "-i", input_path,
                    "-i", temp_path,
                    "-map", "0:a:0",
                    "-map", "1:v:0",
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-c:a", "copy",
                    "-y", output_path,
                ],
                timeout=600,
            )

            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            upscaler.unload()

            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={"method": "ai", "scale": scale_factor},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"AI upscaling failed: {e}",
                duration_sec=time.time() - start,
            )

        try:
            from autovideofixer.ai.frame_processor import FrameProcessor
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            self.logger.warning("PyTorch not available, falling back to traditional upscaling")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_width=int(
                    (input_path and self._get_input_resolution(input_path)[0] * sf)
                    if input_path
                    else None
                ),
                target_height=int(
                    (self._get_input_resolution(input_path)[1] * sf) if input_path else None
                ),
            )

        if not is_torch_available():
            self.logger.warning("PyTorch not installed, falling back to traditional upscaling")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_width=int(self._get_input_resolution(input_path)[0] * sf)
                if input_path
                else None,
                target_height=int(self._get_input_resolution(input_path)[1] * sf)
                if input_path
                else None,
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
                    target_width=int(self._get_input_resolution(input_path)[0] * sf)
                    if input_path
                    else None,
                    target_height=int(self._get_input_resolution(input_path)[1] * sf)
                    if input_path
                    else None,
                )

        except Exception as e:
            self.logger.warning(f"Model check failed ({e}), falling back to traditional")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_width=int(self._get_input_resolution(input_path)[0] * sf)
                if input_path
                else None,
                target_height=int(self._get_input_resolution(input_path)[1] * sf)
                if input_path
                else None,
            )

        try:
            # Load and run Real-ESRGAN
            upscaler = RealESRGANUpscaler(
                scale=int(sf),
                model_name=self._ai_model,
                tta_mode=self._tt_mode,
            )

            if not upscaler.load_model():
                raise RuntimeError("Failed to load Real-ESRGAN model")

            proc = FrameProcessor()
            frames = proc.extract_frames(input_path)

            if not frames:
                proc.close()
                upscaler.unload()
                raise RuntimeError("No frames extracted from input video")

            def cb(current, total, msg):
                self._report_progress(0.1 + (current / total) * 0.9, msg, progress_callback)

            upscaled = upscaler.upscale_video(frames, progress_callback=cb)
            proc.close()

            if not upscaled:
                upscaler.unload()
                raise RuntimeError("No frames produced by upscaler")

            # Write upscaled frames to temp file, then use FFmpeg to finalize
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_upscaled_{os.path.basename(input_path)}",
            )
            fps = self._get_input_fps(input_path)
            proc2 = FrameProcessor()
            if not proc2.frames_to_video(upscaled, temp_path, fps=fps):
                proc2.close()
                upscaler.unload()
                raise RuntimeError("Failed to write upscaled frames to temp file")
            proc2.close()

            # Copy audio from original to upscaled temp file
            # Note: We keep the AI upscaled resolution (may exceed target)
            # The encode stage will handle final resolution adjustment
            run_ffmpeg(
                [
                    "-i", input_path,
                    "-i", temp_path,
                    "-map", "0:a:0",
                    "-map", "1:v:0",
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-c:a", "copy",
                    "-y", output_path,
                ],
                timeout=600,
            )

            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            upscaler.unload()

            self._report_progress(1.0, "AI upscaling complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "ai",
                    "model": self._ai_model,
                    "scale_factor": sf,
                    "frames_processed": len(upscaled),
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            self.logger.warning(f"AI upscaling failed ({e}), falling back to traditional")
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_width=target_width,
                target_height=target_height,
            )

    def _get_input_resolution(self, path: str) -> tuple[int, int]:
        """Get input video resolution."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.resolution
        except Exception:
            return (1920, 1080)

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
