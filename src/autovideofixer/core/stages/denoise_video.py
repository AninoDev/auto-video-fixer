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
        **kwargs,
    ) -> StageResult:
        """AI-based denoising using Real-ESRGAN in denoise mode.

        Falls back to traditional FFmpeg method if PyTorch or model
        files are not available.
        """
        try:
            from autovideofixer.ai.frame_processor import FrameProcessor
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            self.logger.warning("PyTorch not available, falling back to traditional denoising")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength="medium"
            )

        if not is_torch_available():
            self.logger.warning("PyTorch not installed, falling back to traditional denoising")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength="medium"
            )

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            success, msg = ensure_model_available(
                self._stage_config.get("ai_model", "RealESRGAN_x4plus")
            )
            if not success:
                self.logger.warning(f"Model not available ({msg}), falling back")
                return self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                )

        except Exception as e:
            self.logger.warning(f"Model check failed ({e}), falling back to traditional")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength="medium"
            )

        # Use Real-ESRGAN at scale=1 (denoise mode - no upscaling)
        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name=self._stage_config.get("ai_model", "RealESRGAN_x4plus"),
            tta_mode=self._stage_config.get("tta_mode", 0),
        )

        if not upscaler.load_model():
            self.logger.warning("Failed to load model for denoising, falling back")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength="medium"
            )

        try:
            import os as _os

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

            denoised = upscaler.upscale_video(frames, progress_callback=cb)
            proc.close()

            if not denoised:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames produced by denoiser",
                    duration_sec=time.time() - start,
                )

            # Write denoised frames to temp file, then use FFmpeg to finalize
            temp_path = _os.path.join(
                _os.path.dirname(input_path) or ".",
                f".avf_denoise_{_os.path.basename(input_path)}",
            )
            from autovideofixer.core.ffmpeg_utils import probe as _probe

            try:
                fps = _probe(input_path).framerate or 30.0
            except Exception:
                fps = 30.0

            proc2 = FrameProcessor()
            if not proc2.frames_to_video(denoised, temp_path, fps=fps):
                proc2.close()
                return StageResult(
                    status=StageStatus.FAILED,
                    error="Failed to write denoised frames to temp file",
                    duration_sec=time.time() - start,
                )
            proc2.close()

            # Copy audio from original to denoised temp file
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg as _run_ffmpeg

            _run_ffmpeg(
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
            if _os.path.exists(temp_path):
                _os.unlink(temp_path)

            self._report_progress(1.0, "AI denoising complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "ai",
                    "model": self._stage_config.get("ai_model", "RealESRGAN_x4plus"),
                    "frames_processed": len(denoised),
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            self.logger.error(f"AI denoising failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                error=f"AI denoising failed: {e}",
                duration_sec=time.time() - start,
            )
        finally:
            upscaler.unload()
