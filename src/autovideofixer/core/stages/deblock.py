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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._strength = self._stage_config.get("strength", "medium")
        self._ai_model = self._stage_config.get("ai_model", "realesr-general-wdn-x4v3")

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

        # Resolve "ai" vs "traditional" per the shared precedence (explicit
        # method= kwarg > stages.deblock.use_ai > general.use_ai > the
        # stage's own auto default) -- see BaseStage.resolve_ai_method's
        # docstring. deblock's auto default stays "ai" (better quality),
        # a deliberate choice, not an oversight.
        method, source = self.resolve_ai_method(method, "ai")
        self._log_ai_method_choice(
            method,
            source,
            ai_desc=f"AI deblocking (Real-ESRGAN '{self._ai_model}')",
            traditional_desc="traditional unsharp filter",
            ai_hint=f"AI Real-ESRGAN model '{self._ai_model}'",
        )

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

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

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
            from autovideofixer.ai.frame_pipe import get_frame_reader, get_frame_writer
            from autovideofixer.ai.frame_processor import StageTimer
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            return self._ai_fallback_or_fail(
                "PyTorch not available",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, self._strength
                ),
            )

        if not is_torch_available():
            return self._ai_fallback_or_fail(
                "PyTorch not installed",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, self._strength
                ),
            )

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            # Deblocking runs Real-ESRGAN at scale=1 (no spatial upscaling): the
            # output is downscaled back from whatever the checkpoint's native
            # scale is. The default ai_model is now the compact SRVGG
            # realesr-general-wdn-x4v3 (see DEFAULTS["stages"]["deblock"] in
            # config.py) -- an order of magnitude fewer params than the RRDB
            # models, so this swap does NOT fire for the default. It only
            # matters when a user explicitly sets ai_model back to
            # "RealESRGAN_x4plus" (e.g. the max_quality preset): x4plus's RRDB
            # body runs at FULL input resolution (its native scale=4 means no
            # pixel-unshuffle pre-shrink -- see RRDBNet docstring in
            # ai/wrappers/upscale.py), and its upsample tail then produces
            # activations at 4x width/height (16x the pixel count) before
            # being thrown away by the scale=1 downscale. That tail is
            # exactly what OOMs on 1080p+ input. x2plus pre-shrinks the body
            # to half resolution AND caps the tail at 2x/4x pixel count
            # instead of 4x/16x, so prefer it whenever the user has
            # explicitly opted into x4plus -- same optimization UpscaleStage
            # already applies for its own scale<=2 passes.
            deblock_model = self._ai_model
            if deblock_model == "RealESRGAN_x4plus":
                deblock_model = "RealESRGAN_x2plus"

            success, msg = ensure_model_available(deblock_model)
            if not success:
                if deblock_model != self._ai_model:
                    deblock_model = self._ai_model
                    success, msg = ensure_model_available(deblock_model)
                if not success:
                    return self._ai_fallback_or_fail(
                        f"model not available: {msg}",
                        start,
                        lambda: self._execute_traditional(
                            input_path, output_path, progress_callback, start, self._strength
                        ),
                    )
        except Exception as e:
            return self._ai_fallback_or_fail(
                f"model check failed: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, self._strength
                ),
            )

        upscaler = RealESRGANUpscaler(
            scale=1,
            model_name=deblock_model,
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
                    input_path, output_path, progress_callback, start, self._strength
                ),
            )

        probe_info = probe(input_path)
        total_est = probe_info.frame_count or 0

        try:
            fps = self._get_input_fps(input_path)
            width, height = probe_info.resolution
            # The temp file's extension must NOT be derived from the input's
            # extension: the frame writer always muxes with codec="libx264"
            # (H.264), which webm/mkv/etc. containers can't hold -- so e.g.
            # a .webm input produced a ".avf_deblock_test.webm" temp target
            # that ffmpeg then failed to write into. ".mp4" always matches
            # the actual codec being written, regardless of input container.
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_deblock_{os.path.splitext(os.path.basename(input_path))[0]}.mp4",
            )

            try:
                # All videos (regardless of frame count) go through the
                # chunked streaming path -- there is no full-buffer
                # extract_frames()/frames_to_video() route anymore. A
                # resolution-blind frame-count threshold (the old
                # `total_est > 1000` check) meant a short but large-resolution
                # (e.g. 4K) clip could still materialize its ENTIRE frame set
                # in RAM (a 33s 4K clip is ~25GB uncompressed) with zero
                # decode/inference/write overlap; streaming has no measurable
                # downside for short clips either.
                chunk_size = 25
                temp_crf = self._stage_config.get("temp_crf", 16)
                read_ahead = self._stage_config.get("read_ahead", 2)
                write_queue_depth = self._stage_config.get("write_queue_depth", 4)
                # Reader/writer transport (ai/frame_pipe.py): Rust
                # (avf_framepipe) when available, else a Python fallback
                # wrapping frame_processor.py's machinery -- either way,
                # decode/write happen on background threads/processes so
                # they overlap with GPU inference instead of serializing
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
                timer = StageTimer("deblock")

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

                    chunk_out = upscaler.upscale_video(chunk, progress_callback=cb, timer=timer)
                    # Write each chunk's output straight to the ffmpeg pipe
                    # instead of buffering the whole video's frames in memory.
                    t0 = time.time()
                    writer.write_batch(chunk_out)
                    timer.record("write_wait", time.time() - t0)
                    timer.end_chunk(len(chunk_out))

                    frames_written += len(chunk_out)
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / total_est) * 0.9,
                        "Deblocking chunk...",
                        progress_callback,
                    )
                reader.close()
                write_ok = writer.close()
                timer.summary()

                if frames_written == 0:
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="No frames produced",
                        duration_sec=time.time() - start,
                    )
                if not write_ok:
                    upscaler.unload()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write deblocked frames",
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
                # generation plus a wasted full x264 pass over the whole video
                # for no quality benefit. Stream-copying the already-encoded
                # video track makes this mux bit-identical to the temp file's
                # video stream.
                mux_args = ["-i", input_path, "-i", temp_path]
                mux_args += ["-map", "0:a:0", "-map", "1:v:0"] if has_audio else ["-map", "1:v:0"]
                mux_args += ["-c:v", "copy", "-c:a", "copy", "-y", output_path]
                mux_result = run_ffmpeg(mux_args, timeout=self.stage_timeout())

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
                        "model": deblock_model,
                        "frames_processed": frames_written,
                    },
                    duration_sec=time.time() - start,
                )
            finally:
                # Always remove the internal temp file, on both the success
                # and failure paths -- previously several early-return
                # failure paths skipped cleanup entirely, orphaning a
                # partial temp file next to the input video.
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            self.logger.error(f"AI deblocking failed: {e}")
            upscaler.unload()
            return self._ai_fallback_or_fail(
                f"inference exception: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path, output_path, progress_callback, start, self._strength
                ),
            )

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
