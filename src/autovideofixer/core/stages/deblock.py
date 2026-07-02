"""Auto Video Fixer - Deblocking stage."""

from __future__ import annotations

import os
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class DeblockStage(BaseStage):
    """Remove compression blocking artifacts from video.

    Traditional: FFmpeg `unsharp` filter (most reliable for deblocking).
    AI: Real-ESRGAN with scale=1 (denoise+deblock via super-resolution).
    """

    name = "deblock"
    display_name = "Deblocking"
    description = "Remove compression blocking artifacts"
    category = "enhancement"
    priority = 15
    supports_gpu = True

    def __init__(self, config):
        super().__init__(config)
        self._strength = self._stage_config.get("strength", "medium")
        self._ai_model = self._stage_config.get("ai_model", "RealESRGAN_x4plus")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        codec = input_info.get("video_codec", "")
        if codec in ("prores", "dnxhd", "hqx"):
            return False, "Lossless codec - no deblocking needed"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        strength: str | None = None,
        method: str | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running deblocking...", progress_callback)

        # Apply global AI override from CLI/config
        use_ai = self.config.get("general", "use_ai", default=None)
        if method is None:
            if use_ai is True:
                method = "ai"
            elif use_ai is False:
                method = "traditional"
            else:
                # Default: AI (better quality)
                method = "ai"

        try:
            if method == "ai":
                return self._execute_ai(input_path, output_path, progress_callback, start)
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, strength or self._strength
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
        """Traditional deblocking using FFmpeg unsharp filter."""
        params = {
            "low": "5:5:0.2:5:5:0.1",
            "medium": "5:5:0.5:5:5:0.3",
            "high": "7:7:0.8:7:7:0.5",
        }.get(str(strength), "5:5:0.5:5:5:0.3")

        # msize_x/y (kernel radius) fields were previously computed into `params`
        # but discarded -- the filter hardcoded 5:5 regardless of strength, so
        # "high" strength only changed amount, not kernel size, making it barely
        # different from "low". Use all six fields now.
        lmx, lmy, lamount, cmx, cmy, camount = params.split(":")
        args = [
            "-i",
            input_path,
            "-vf",
            f"unsharp=luma_msize_x={lmx}:luma_msize_y={lmy}:luma_amount={lamount}:"
            f"chroma_msize_x={cmx}:chroma_msize_y={cmy}:chroma_amount={camount}",
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

        def cb(p, m):
            self._report_progress(0.2 + p * 0.8, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=600)

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Deblocking failed: {result.stderr[:2000]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Traditional deblocking complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "traditional", "filter": "unsharp", "params": params},
            duration_sec=time.time() - start,
        )

    def _execute_ai(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
    ) -> StageResult:
        """AI-based deblocking using Real-ESRGAN (scale=1, no spatial scaling).

        Real-ESRGAN removes compression artifacts as a side effect of
        its super-resolution training, so we use scale=1 for pure deblocking.
        """
        try:
            from autovideofixer.ai.frame_processor import FrameProcessor, StreamingVideoWriter
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            self.logger.warning("PyTorch not available, falling back to traditional deblocking")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, self._strength
            )

        if not is_torch_available():
            self.logger.warning("PyTorch not installed, falling back to traditional deblocking")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, self._strength
            )

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            success, msg = ensure_model_available(self._ai_model)
            if not success:
                self.logger.warning(f"Model not available ({msg}), falling back")
                return self._execute_traditional(
                    input_path, output_path, progress_callback, start, self._strength
                )
        except Exception as e:
            self.logger.warning(f"Model check failed ({e}), falling back to traditional")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, self._strength
            )

        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name=self._ai_model,
            tta_mode=self._stage_config.get("tta_mode", 0),
            device_preference=self.config.get("gpu", "preferred_device", default="auto"),
        )

        if not upscaler.load_model():
            self.logger.warning("Failed to load model for deblocking, falling back")
            return self._execute_traditional(
                input_path, output_path, progress_callback, start, self._strength
            )

        probe_info = probe(input_path)
        total_est = probe_info.frame_count or 0
        use_chunked = total_est > 1000

        try:
            proc = FrameProcessor()
            fps = self._get_input_fps(input_path)
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_deblock_{os.path.basename(input_path)}",
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
                    chunk_out = upscaler.upscale_video(chunk, progress_callback=cb)
                    # Write each chunk's output straight to the ffmpeg pipe
                    # instead of buffering the whole video's frames in memory.
                    writer.write(chunk_out)
                    frames_written += len(chunk_out)
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9,
                        "Deblocking chunk...",
                        progress_callback,
                    )
                proc.close()
                write_ok = writer.close()

                if frames_written == 0:
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced",
                        duration_sec=time.time() - start,
                    )
                if not write_ok:
                    upscaler.unload()
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write deblocked frames",
                        duration_sec=time.time() - start,
                    )
            else:
                frames = proc.extract_frames(input_path)
                proc.close()

                if not frames:
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames extracted",
                        duration_sec=time.time() - start,
                    )

                def cb(current, total, msg):
                    self._report_progress(0.1 + (current / total) * 0.9, msg, progress_callback)

                all_frames = upscaler.upscale_video(frames, progress_callback=cb)

                if not all_frames:
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced",
                        duration_sec=time.time() - start,
                    )

                proc2 = FrameProcessor()
                if not proc2.frames_to_video(all_frames, temp_path, fps=fps):
                    proc2.close()
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write deblocked frames",
                        duration_sec=time.time() - start,
                    )
                proc2.close()
                frames_written = len(all_frames)

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

            if os.path.exists(temp_path):
                os.unlink(temp_path)

            upscaler.unload()

            if mux_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Failed to finalize deblocked output: {mux_result.stderr[:2000]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "AI deblocking complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "ai",
                    "model": self._ai_model,
                    "frames_processed": frames_written,
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            self.logger.error(f"AI deblocking failed: {e}")
            upscaler.unload()
            return StageResult(
                status=StageStatus.FAILED,
                error=f"AI deblocking failed: {e}",
                duration_sec=time.time() - start,
            )

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
