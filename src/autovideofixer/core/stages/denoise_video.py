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
            from autovideofixer.ai.frame_pipe import get_frame_reader, get_frame_writer
            from autovideofixer.ai.frame_processor import StageTimer
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            return self._ai_fallback_or_fail(
                "PyTorch not available",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                ),
            )

        if not is_torch_available():
            return self._ai_fallback_or_fail(
                "PyTorch not installed",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                ),
            )

        # Denoising, like deblocking, runs Real-ESRGAN at scale=1 -- the
        # output is downscaled back down from whatever the checkpoint's
        # native scale is. x4plus's RRDB body runs at FULL input resolution
        # (native scale=4 means no pixel-unshuffle pre-shrink) and its
        # upsample tail produces activations at 16x the pixel count before
        # being discarded by the scale=1 downscale -- exactly what OOMs on
        # 1080p+ input. x2plus pre-shrinks the body to half resolution and
        # caps the tail at 4x pixel count instead of 16x, so prefer it here
        # unless the user explicitly configured a different model (same
        # optimization UpscaleStage/DeblockStage apply for their own
        # scale<=2 / scale=1 passes).
        configured_model = self._stage_config.get("ai_model", "RealESRGAN_x4plus")
        ai_model_name = configured_model
        if ai_model_name == "RealESRGAN_x4plus":
            ai_model_name = "RealESRGAN_x2plus"

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            success, msg = ensure_model_available(ai_model_name)
            if not success:
                if ai_model_name != configured_model:
                    ai_model_name = configured_model
                    success, msg = ensure_model_available(ai_model_name)
                if not success:
                    return self._ai_fallback_or_fail(
                        f"model not available: {msg}",
                        start,
                        lambda: self._execute_traditional(
                            input_path, output_path, progress_callback, start, strength="medium"
                        ),
                    )

        except Exception as e:
            return self._ai_fallback_or_fail(
                f"model check failed: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                ),
            )

        # Use Real-ESRGAN at scale=1 (denoise mode - no upscaling)
        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name=ai_model_name,
            tta_mode=self._stage_config.get("tta_mode", 0),
            device_preference=self.config.get("gpu", "preferred_device", default="auto"),
            tile_size=self._stage_config.get("tile_size", 0),
            batch_size=self._stage_config.get("batch_size", 1),
            tile_batch_size=self._stage_config.get("tile_batch_size", 1),
        )

        if not upscaler.load_model():
            return self._ai_fallback_or_fail(
                "failed to load model",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                ),
            )

        probe_info = probe(input_path)
        total_est = probe_info.frame_count or 0

        try:
            import os as _os

            try:
                fps = probe(input_path).framerate or 30.0
            except Exception:
                fps = 30.0
            width, height = probe_info.resolution

            # The temp file's extension must NOT be derived from the input's
            # extension: the frame writer always muxes with codec="libx264"
            # (H.264), which webm/mkv/etc. containers can't hold -- so e.g.
            # a .webm input produced a ".avf_denoise_test.webm" temp target
            # that ffmpeg then failed to write into. ".mp4" always matches
            # the actual codec being written, regardless of input container.
            temp_path = _os.path.join(
                _os.path.dirname(input_path) or ".",
                f".avf_denoise_{_os.path.splitext(_os.path.basename(input_path))[0]}.mp4",
            )

            try:
                # All videos (regardless of frame count) go through the
                # chunked streaming path -- see DeblockStage._execute_ai for
                # why the old frame-count-only `use_chunked` threshold (and
                # its full-buffer extract_frames()/frames_to_video() route)
                # was removed.
                chunk_size = 25
                temp_crf = self._stage_config.get("temp_crf", 16)
                read_ahead = self._stage_config.get("read_ahead", 2)
                write_queue_depth = self._stage_config.get("write_queue_depth", 4)
                # Reader/writer transport (ai/frame_pipe.py): Rust
                # (avf_framepipe) when available, else a Python fallback
                # wrapping frame_processor.py's machinery -- see
                # UpscaleStage._run_single_ai_pass for why this overlaps CPU
                # decode/write with GPU inference instead of serializing
                # read -> infer -> write per chunk.
                reader = get_frame_reader(
                    input_path,
                    width,
                    height,
                    chunk_size=chunk_size,
                    read_ahead=read_ahead,
                )
                writer = get_frame_writer(
                    temp_path,
                    width,
                    height,
                    fps,
                    crf=temp_crf,
                    preset="medium",
                    write_queue=write_queue_depth,
                )
                processed = 0
                frames_written = 0
                timer = StageTimer("denoise_video")

                def cb(current, total, msg):
                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9, msg, progress_callback
                    )

                while True:
                    t0 = time.time()
                    chunk = reader.next_batch()
                    timer.record("decode_wait", time.time() - t0)
                    if chunk is None:
                        break

                    chunk_denoised = upscaler.upscale_video(
                        chunk, progress_callback=cb, timer=timer
                    )
                    # Write each chunk's output straight to the ffmpeg pipe
                    # instead of buffering the whole video's frames in memory.
                    t0 = time.time()
                    writer.write_batch(chunk_denoised)
                    timer.record("write_wait", time.time() - t0)
                    timer.end_chunk(len(chunk_denoised))

                    frames_written += len(chunk_denoised)
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9,
                        "Denoising chunk...",
                        progress_callback,
                    )
                reader.close()
                write_ok = writer.close()
                timer.summary()

                if frames_written == 0:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced by denoiser",
                        duration_sec=time.time() - start,
                    )
                if not write_ok:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write denoised frames to temp file",
                        duration_sec=time.time() - start,
                    )

                # Mux processed video back with the original audio (if any).
                # The input may have no audio stream at all - mapping
                # "0:a:0" unconditionally would make ffmpeg fail, and NOT
                # checking the return code here used to mean that failure
                # (e.g. on any silent/muted input) was reported as a
                # successful COMPLETED stage with the only rendered output
                # already deleted.
                try:
                    has_audio = probe(input_path).has_audio
                except Exception:
                    has_audio = False

                # -c:v copy: the temp file was already encoded once (at
                # temp_crf/"medium" above) -- re-encoding it again here at a
                # DIFFERENT crf (this used to be "-crf 18") was a second lossy
                # generation plus a wasted full x264 pass. Stream-copying the
                # already-encoded video track makes this mux bit-identical to
                # the temp file's video stream.
                mux_args = ["-i", input_path, "-i", temp_path]
                mux_args += ["-map", "0:a:0", "-map", "1:v:0"] if has_audio else ["-map", "1:v:0"]
                mux_args += ["-c:v", "copy", "-c:a", "copy", "-y", output_path]
                mux_result = run_ffmpeg(mux_args, timeout=600)

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
                        "model": ai_model_name,
                        "frames_processed": frames_written,
                    },
                    duration_sec=time.time() - start,
                )
            finally:
                # Always remove the internal temp file, on both the success
                # and failure paths -- previously several early-return
                # failure paths skipped cleanup entirely, orphaning a
                # partial temp file next to the input video.
                if _os.path.exists(temp_path):
                    _os.unlink(temp_path)

        except Exception as e:
            self.logger.error(f"AI denoising failed: {e}")
            return self._ai_fallback_or_fail(
                f"inference exception: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, strength="medium"
                ),
            )
        finally:
            upscaler.unload()
