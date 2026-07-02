"""Auto Video Fixer - Video denoising stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
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

        # Apply global AI override from CLI/config. Unlike DeblockStage
        # (which defaults to AI when general.use_ai is unset), this stage's
        # `method` parameter itself defaults to "traditional" - denoising
        # benefits less from Real-ESRGAN than deblocking does, and hqdn3d is
        # far cheaper with no GPU/model dependency. That default is
        # deliberate, not an oversight.
        use_ai = self.config.get("general", "use_ai", default=None)
        if use_ai is True:
            method = "ai"
        elif use_ai is False:
            method = "traditional"

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
            from autovideofixer.ai.frame_processor import FrameProcessor, StreamingVideoWriter
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
        ai_model_name = self._stage_config.get("ai_model", "RealESRGAN_x4plus")
        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name=ai_model_name,
            tta_mode=self._stage_config.get("tta_mode", 0),
            device_preference=self.config.get("gpu", "preferred_device", default="auto"),
        )

        if not upscaler.load_model():
            self.logger.warning("Failed to load model for denoising, falling back")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength="medium"
            )

        probe_info = probe(input_path)
        total_est = probe_info.frame_count or 0
        use_chunked = total_est > 1000

        try:
            import os as _os

            proc = FrameProcessor()

            try:
                fps = probe(input_path).framerate or 30.0
            except Exception:
                fps = 30.0

            temp_path = _os.path.join(
                _os.path.dirname(input_path) or ".",
                f".avf_denoise_{_os.path.basename(input_path)}",
            )

            if use_chunked:
                chunk_size = 25
                writer = StreamingVideoWriter(temp_path, fps=fps)
                processed = 0
                frames_written = 0

                def cb(current, total, msg):
                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9, msg, progress_callback
                    )

                for chunk in proc.stream_frames(
                    input_path, chunk_size=chunk_size, max_frames=total_est
                ):
                    chunk_denoised = upscaler.upscale_video(chunk, progress_callback=cb)
                    # Write each chunk's output straight to the ffmpeg pipe
                    # instead of buffering the whole video's frames in memory.
                    writer.write(chunk_denoised)
                    frames_written += len(chunk_denoised)
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9,
                        "Denoising chunk...",
                        progress_callback,
                    )
                proc.close()
                write_ok = writer.close()

                if frames_written == 0:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced by denoiser",
                        duration_sec=time.time() - start,
                    )
                if not write_ok:
                    if _os.path.exists(temp_path):
                        _os.unlink(temp_path)
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write denoised frames to temp file",
                        duration_sec=time.time() - start,
                    )
            else:
                frames = proc.extract_frames(input_path)
                proc.close()

                if not frames:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames extracted from input video",
                        duration_sec=time.time() - start,
                    )

                def cb(current, total, msg):
                    self._report_progress(0.1 + (current / total) * 0.9, msg, progress_callback)

                denoised = upscaler.upscale_video(frames, progress_callback=cb)

                if not denoised:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced by denoiser",
                        duration_sec=time.time() - start,
                    )

                proc2 = FrameProcessor()
                if not proc2.frames_to_video(denoised, temp_path, fps=fps):
                    proc2.close()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write denoised frames to temp file",
                        duration_sec=time.time() - start,
                    )
                proc2.close()
                frames_written = len(denoised)

            # Mux processed video back with the original audio (if any). The
            # input may have no audio stream at all - mapping "0:a:0"
            # unconditionally would make ffmpeg fail, and NOT checking the
            # return code here used to mean that failure (e.g. on any
            # silent/muted input) was reported as a successful COMPLETED
            # stage with the only rendered output already deleted.
            try:
                has_audio = probe(input_path).has_audio
            except Exception:
                has_audio = False

            mux_args = ["-i", input_path, "-i", temp_path]
            mux_args += ["-map", "0:a:0", "-map", "1:v:0"] if has_audio else ["-map", "1:v:0"]
            mux_args += ["-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-y", output_path]
            mux_result = run_ffmpeg(mux_args, timeout=600)

            # Clean up temp file
            if _os.path.exists(temp_path):
                _os.unlink(temp_path)

            if mux_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Failed to finalize denoised output: {mux_result.stderr[:2000]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "AI denoising complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "ai",
                    "model": self._stage_config.get("ai_model", "RealESRGAN_x4plus"),
                    "frames_processed": frames_written,
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
