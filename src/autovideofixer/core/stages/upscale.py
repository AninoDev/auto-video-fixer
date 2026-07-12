"""Auto Video Fixer - Resolution upscaling stage."""

from __future__ import annotations

import os
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
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
        self._keep_aspect_ratio = self.config.get(
            "quality", "quality_target", "keep_aspect_ratio", default=True
        )

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

        # Apply global AI override from CLI/config
        use_ai = self.config.get("general", "use_ai", default=None)
        if use_ai is True:
            method = "ai"
        elif use_ai is False:
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
        # Get input dimensions
        input_w, input_h = self._get_input_resolution(input_path)

        # Calculate target dimensions (preserving aspect ratio if configured)
        if target_width and target_height:
            final_w, final_h = self._calculate_target_dimensions(
                input_w, input_h, target_width, target_height
            )
            scale_expr = f"scale={final_w}:{final_h}:flags=lanczos"
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

        # Get input dimensions
        input_w, input_h = self._get_input_resolution(input_path) if input_path else (0, 0)

        # Calculate target dimensions (preserving aspect ratio if configured)
        if target_width and target_height and input_w > 0 and input_h > 0:
            final_target_w, final_target_h = self._calculate_target_dimensions(
                input_w, input_h, target_width, target_height
            )
        else:
            final_target_w, final_target_h = target_width, target_height

        # Calculate how many AI passes we need
        if final_target_w and final_target_h and input_w > 0 and input_h > 0:
            # Calculate total scale needed (based on longest side)
            scale_w = final_target_w / input_w
            scale_h = final_target_h / input_h
            total_scale = max(scale_w, scale_h)

            # Calculate number of passes needed (each pass scales by up to max_ai_scale)
            num_passes = (
                math.ceil(math.log(total_scale) / math.log(max_ai_scale)) if total_scale > 1 else 1
            )
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
                        current_w, current_h = current_info
                        scale_w = final_target_w / current_w
                        scale_h = final_target_h / current_h
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
                # Unique per invocation (uuid), not just input stem + pass number --
                # two jobs with same-named inputs running concurrently (or a second
                # run before the first job's cleanup completes) would otherwise
                # read/write the same shared-tempdir path and corrupt each other's
                # intermediate frames.
                import uuid

                input_stem = os.path.splitext(os.path.basename(input_path))[0]
                pass_output = os.path.join(
                    tempfile.gettempdir(),
                    f".avf_ai_pass{pass_num + 1}_{input_stem}_{uuid.uuid4().hex[:8]}.mp4",
                )
                intermediate_files.append(pass_output)

            # Run single AI pass. fallback_ctx lets _run_single_ai_pass fall back
            # to the traditional method for the WHOLE job (original input ->
            # final output_path/target dims) rather than just this one pass, if
            # the AI path can't run and ai_fallback is enabled -- chaining a
            # partial traditional result into subsequent AI passes wouldn't make
            # sense once the AI path has already proven unavailable.
            result = self._run_single_ai_pass(
                current_input,
                pass_output,
                progress_callback,
                start,
                sf,
                fallback_ctx={
                    "input_path": input_path,
                    "output_path": output_path,
                    "target_width": final_target_w,
                    "target_height": final_target_h,
                },
            )

            if result.status != StageStatus.COMPLETED:
                # Clean up intermediate files on failure
                for f in intermediate_files:
                    if os.path.exists(f):
                        os.unlink(f)
                return result

            if result.metadata.get("method") == "traditional":
                # AI became unavailable mid-run and ai_fallback allowed falling
                # back; the traditional path already wrote the final output
                # directly to output_path (bypassing pass chaining), so the
                # remaining planned AI passes must not run.
                for f in intermediate_files:
                    if os.path.exists(f):
                        os.unlink(f)
                return result

            current_input = pass_output

        # Clean up intermediate files
        for f in intermediate_files:
            if os.path.exists(f):
                os.unlink(f)

        # Each AI pass's scale factor is rounded to a power of 2 (models only support
        # discrete scales), so the chained result can land on a different resolution
        # than final_target_w/h. Correct it with one cheap ffmpeg resize rather than
        # silently shipping an off-spec resolution.
        if final_target_w and final_target_h:
            actual = self._get_input_resolution(output_path)
            if (
                actual
                and actual[0] > 0
                and (actual[0], actual[1])
                != (
                    final_target_w,
                    final_target_h,
                )
            ):
                fix_result = self._resize_to_exact(
                    output_path, final_target_w, final_target_h, start
                )
                if fix_result.status != StageStatus.COMPLETED:
                    return fix_result

        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "ai", "model": self._ai_model, "passes": num_passes},
            duration_sec=time.time() - start,
        )

    def _resize_to_exact(
        self, video_path: str, target_w: int, target_h: int, start: float
    ) -> StageResult:
        """Resize `video_path` in place to exactly target_w x target_h."""
        tmp_path = f"{video_path}.resize_tmp.mp4"
        args = [
            "-i",
            video_path,
            "-vf",
            f"scale={target_w}:{target_h}",
            "-c:a",
            "copy",
            "-y",
            tmp_path,
        ]
        result = run_ffmpeg(args, timeout=600)
        if result.returncode != 0:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Final resize to {target_w}x{target_h} failed: {result.stderr[:300]}",
                duration_sec=time.time() - start,
            )
        os.replace(tmp_path, video_path)
        return StageResult(status=StageStatus.COMPLETED, output_path=video_path)

    def _run_single_ai_pass(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        scale_factor: float,
        fallback_ctx: dict | None = None,
    ) -> StageResult:
        """Run a single AI upscaling pass.

        `fallback_ctx` (input_path/output_path/target_width/target_height for
        the *whole* job, not just this pass), if given, is used to fall back
        to a single traditional pass over the original input when the AI path
        can't run and ai_fallback is enabled (see BaseStage._ai_fallback_or_fail).
        Without it (fallback_ctx=None), AI-unavailability always fails outright
        -- used by callers that don't have a well-defined traditional
        equivalent to fall back to.
        """

        def _traditional_fallback() -> StageResult:
            ctx = fallback_ctx or {}
            return self._execute_traditional(
                ctx.get("input_path", input_path),
                ctx.get("output_path", output_path),
                progress_callback,
                start,
                target_width=ctx.get("target_width"),
                target_height=ctx.get("target_height"),
            )

        try:
            from autovideofixer.ai.frame_processor import (
                AsyncVideoWriter,
                FrameProcessor,
                StreamingVideoWriter,
            )
            from autovideofixer.ai.torch_utils import is_torch_available
            from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler
        except ImportError:
            if fallback_ctx is None:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="PyTorch not available",
                    duration_sec=time.time() - start,
                )
            return self._ai_fallback_or_fail("PyTorch not available", start, _traditional_fallback)

        if not is_torch_available():
            if fallback_ctx is None:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="PyTorch not installed",
                    duration_sec=time.time() - start,
                )
            return self._ai_fallback_or_fail("PyTorch not installed", start, _traditional_fallback)

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            # RealESRGAN_x4plus's forward pass always computes a native 4x
            # result internally (23 RRDB blocks -- the dominant cost of a
            # forward pass, measured at ~90% of total time -- run on the
            # full input resolution regardless of the final requested
            # scale), so a scale<=2 pass through x4plus wastes the bulk of
            # its compute on detail that then gets discarded by the
            # post-hoc downscale in RealESRGANUpscaler.upscale(). Real-ESRGAN
            # x2plus is architecturally a genuine 2x model (pixel-unshuffle
            # preprocessing shrinks the RRDB body's own feature map by 4x,
            # not just the output), so prefer it whenever this pass only
            # needs <=2x and the user hasn't explicitly configured a
            # different model. Measured on this project's target GPU: ~4-5x
            # faster per frame than running x4plus at native 4x and
            # discarding half the resolution.
            pass_model = self._ai_model
            if pass_model == "RealESRGAN_x4plus" and scale_factor <= 2:
                pass_model = "RealESRGAN_x2plus"

            backend = self._stage_config.get("backend", "torch")

            # The pre-flight availability check below queries the torch
            # MODEL_REGISTRY (.pth files); for the ncnn backend the wrapper's
            # own load_model() resolves/downloads the .param/.bin pair via
            # ensure_ncnn_model_available(), and a load failure already routes
            # through _ai_fallback_or_fail() -- so skip the wrong-registry
            # check entirely rather than gating ncnn on cached torch weights.
            success, msg = (True, "") if backend == "ncnn" else ensure_model_available(pass_model)
            if not success:
                # Fall back to the originally configured model if the
                # auto-selected lighter one isn't available (e.g. offline
                # and only x4plus was ever cached) rather than failing the
                # whole stage outright.
                if pass_model != self._ai_model:
                    pass_model = self._ai_model
                    success, msg = ensure_model_available(pass_model)
                if not success:
                    if fallback_ctx is None:
                        return StageResult(
                            status=StageStatus.FAILED,
                            error=f"Model not available: {msg}",
                            duration_sec=time.time() - start,
                        )
                    return self._ai_fallback_or_fail(
                        f"model not available: {msg}", start, _traditional_fallback
                    )

            upscaler = RealESRGANUpscaler(
                scale=scale_factor,
                model_name=pass_model,
                tta_mode=self._tt_mode,
                device_preference=self.config.get("gpu", "preferred_device", default="auto"),
                tile_size=self._stage_config.get("tile_size", 0),
                backend=backend,
                vulkan_device=self.config.get("gpu", "vulkan_device", default=0),
            )

            if not upscaler.load_model():
                if fallback_ctx is None:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to load model",
                        duration_sec=time.time() - start,
                    )
                return self._ai_fallback_or_fail(
                    "failed to load model", start, _traditional_fallback
                )

            proc = FrameProcessor()
            # Stream frames in chunks to avoid loading the entire video into memory.
            # For short clips (<= 1000 frames) we load all at once for simplicity.
            fps = self._get_input_fps(input_path)
            probe_info = probe(input_path)
            total_est = int(probe_info.frame_count) if probe_info.frame_count else 0
            use_chunked = total_est > 1000

            # The temp file's extension must NOT be derived from output_path's
            # extension: frames_to_video()/StreamingVideoWriter always mux
            # with codec="libx264" (H.264), which webm/mkv/etc. containers
            # can't hold -- so e.g. a .webm output target produced a
            # ".avf_upscaled_test_enhanced.webm" temp file that ffmpeg then
            # failed to write into. ".mp4" always matches the actual codec
            # being written, regardless of the final output's container.
            temp_path = os.path.join(
                os.path.dirname(output_path) or ".",
                f".avf_upscaled_{os.path.splitext(os.path.basename(output_path))[0]}.mp4",
            )

            try:
                if use_chunked:
                    chunk_size = 25
                    # AsyncVideoWriter hands the ffmpeg pipe write off to a
                    # background thread so it doesn't block the next chunk's
                    # GPU inference; stream_frames_prefetched decodes the next
                    # chunk on a background thread while the current one is
                    # being upscaled. Together these overlap CPU decode/write
                    # with GPU compute instead of serializing all three per
                    # chunk (read -> infer -> write -> read -> ...).
                    writer = AsyncVideoWriter(StreamingVideoWriter(temp_path, fps=fps))
                    total_chunks = 0
                    processed_frames = 0
                    frames_written = 0

                    def cb(current, total, msg):
                        self._report_progress(
                            0.1 + (processed_frames / (total_est * self._scale_factor)) * 0.9,
                            msg,
                            progress_callback,
                        )

                    for chunk in proc.stream_frames_prefetched(
                        input_path, chunk_size=chunk_size, max_frames=total_est
                    ):
                        total_chunks += 1
                        chunk_upscaled = upscaler.upscale_video(chunk, progress_callback=cb)
                        # Write each chunk's output straight to the ffmpeg pipe
                        # instead of buffering the whole video's frames in memory.
                        writer.write(chunk_upscaled)
                        frames_written += len(chunk_upscaled)
                        processed_frames += len(chunk)

                        self._report_progress(
                            0.1 + (processed_frames / (total_est * self._scale_factor)) * 0.9,
                            f"Processing chunk {total_chunks}...",
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
                        return StageResult(
                            status=StageStatus.FAILED,
                            error="Failed to write upscaled frames",
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

                    all_upscaled = upscaler.upscale_video(frames, progress_callback=cb)

                    if not all_upscaled:
                        upscaler.unload()
                        return StageResult(
                            status=StageStatus.FAILED,
                            error="No frames produced",
                            duration_sec=time.time() - start,
                        )

                    proc2 = FrameProcessor()
                    if not proc2.frames_to_video(all_upscaled, temp_path, fps=fps):
                        proc2.close()
                        upscaler.unload()
                        return StageResult(
                            status=StageStatus.FAILED,
                            error="Failed to write frames",
                            duration_sec=time.time() - start,
                        )
                    proc2.close()

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

                mux_args = ["-i", input_path, "-i", temp_path]
                mux_args += ["-map", "0:a:0", "-map", "1:v:0"] if has_audio else ["-map", "1:v:0"]
                mux_args += ["-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-y", output_path]
                mux_result = run_ffmpeg(mux_args, timeout=600)

                upscaler.unload()

                if mux_result.returncode != 0:
                    return StageResult(
                        status=StageStatus.FAILED,
                        error=f"Failed to finalize upscaled output: {mux_result.stderr[:2000]}",
                        duration_sec=time.time() - start,
                    )

                return StageResult(
                    status=StageStatus.COMPLETED,
                    output_path=output_path,
                    metadata={"method": "ai", "scale": scale_factor},
                    duration_sec=time.time() - start,
                )
            finally:
                # Always remove the internal temp file, on both the success
                # and failure paths -- previously several early-return
                # failure paths (e.g. frames_to_video() failing) skipped
                # cleanup entirely, orphaning a partial temp file next to the
                # job's output directory.
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            if fallback_ctx is None:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"AI upscaling failed: {e}",
                    duration_sec=time.time() - start,
                )
            return self._ai_fallback_or_fail(
                f"inference exception: {e}", start, _traditional_fallback
            )

    def _get_input_resolution(self, path: str) -> tuple[int, int]:
        """Get input video resolution."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.resolution
        except Exception:
            return (1920, 1080)

    def _calculate_target_dimensions(
        self,
        input_width: int,
        input_height: int,
        target_width: int | None,
        target_height: int | None,
    ) -> tuple[int, int]:
        """Calculate target dimensions preserving aspect ratio if configured.

        If keep_aspect_ratio is True, rotates the preset bounding box to match
        the input's orientation (portrait presets swap w/h, landscape keeps it)
        so total pixel count stays consistent regardless of aspect ratio.
        If False, scales to the exact target dimensions.

        Returns:
            (width, height) tuple with even values (required by H.264).
        """
        if target_width is None or target_height is None:
            return (input_width, input_height)

        if not self._keep_aspect_ratio:
            return self._round_to_even(target_width, target_height)

        # Rotate preset bounding box to match input orientation
        if input_height > input_width:  # Portrait input
            bound_w, bound_h = target_height, target_width
        elif input_width > input_height:  # Landscape input
            bound_w, bound_h = target_width, target_height
        else:  # Square input
            bound_w = bound_h = min(target_width, target_height)

        # Scale to fit within the (possibly rotated) bounding box
        scale_w = bound_w / input_width if input_width > 0 else 1.0
        scale_h = bound_h / input_height if input_height > 0 else 1.0
        scale_factor = min(scale_w, scale_h)

        new_width = int(input_width * scale_factor)
        new_height = int(input_height * scale_factor)

        return self._round_to_even(new_width, new_height)

    @staticmethod
    def _round_to_even(width: int, height: int) -> tuple[int, int]:
        """Round dimensions to even values (required by H.264/YUV420p)."""
        return (width if width % 2 == 0 else width + 1, height if height % 2 == 0 else height + 1)

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
