"""Auto Video Fixer - Frame interpolation stage."""

from __future__ import annotations

import os
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._ai_model = self._stage_config.get("ai_model", "rife_v4.6")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        # input_info is the probed INPUT file's own properties (from
        # get_video_info), not the preset's target -- it never contains
        # "target_framerate" unless something explicitly injects it. Fall back
        # to config.quality.quality_target.target_framerate, matching
        # UpscaleStage.should_run()'s equivalent fallback for target_resolution.
        # Without this the stage always reported "No target framerate
        # specified" and silently skipped, even when a preset like 1080p60
        # requested 60fps.
        target_fps = input_info.get("target_framerate")
        if not target_fps:
            quality_target = self.config.get("quality", "quality_target", default={})
            target_fps = quality_target.get("target_framerate")
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

        if target_fps is None:
            quality_target = self.config.get("quality", "quality_target", default={})
            target_fps = quality_target.get("target_framerate")

        # Apply global AI override from CLI/config
        use_ai = self.config.get("general", "use_ai", default=None)
        if use_ai is True:
            method = "ai"
        elif use_ai is False:
            method = "traditional"

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

    @staticmethod
    def _minterpolate_filter(target_fps: float) -> str:
        # Real minterpolate option names (mi/mb/vsblur used by a previous
        # implementation don't exist in ffmpeg and fail with "Option not
        # found" on every invocation). minterpolate takes its own fps=
        # sub-option, so the target fps must not be applied via a separate
        # leading fps= filter -- that would just duplicate/drop frames
        # before motion-compensated interpolation ever sees original timing.
        return (
            f"minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bilat:vsbmc=1:"
            f"mb_size=16:fps={target_fps}"
        )

    def _execute_traditional(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        target_fps: float | None = None,
        current_fps: float | None = None,
        parallel_chunks: int | None = None,
    ) -> StageResult:
        """Traditional frame interpolation using FFmpeg minterpolate.

        Dispatches to a parallel-chunked implementation
        (``_execute_traditional_parallel``) when ``stages.interpolate.parallel_chunks``
        (or the ``parallel_chunks`` kwarg override, used by scene-mode to bound the
        shared worker budget across concurrently-processed scenes -- see
        ``core/scenes.py``) resolves to more than 1 chunk and the input is long
        enough to be worth splitting; otherwise runs the single-process path
        that existed before chunking was added.

        minterpolate is single-threaded per ffmpeg process and can be very slow
        on long/high-resolution clips -- chunking splits the input into N
        independent time ranges (by *frame index*, not wall-clock seeking, for
        exact reproducibility) with a 1-frame overlap at each boundary, runs
        minterpolate on each chunk as its own ffmpeg process in parallel, drops
        the duplicated boundary frame from every chunk after the first, and
        concatenates the results. This is the traditional/minterpolate path
        ONLY -- the AI/RIFE path stays serial (GPU-bound; concurrent GPU jobs
        contend rather than parallelize).
        """
        if target_fps and current_fps and current_fps > 0:
            factor = int(target_fps / current_fps)
        else:
            factor = 2
        target = target_fps or (current_fps * factor if current_fps else 60)

        chunks_cfg = (
            parallel_chunks
            if parallel_chunks is not None
            else self._stage_config.get("parallel_chunks", 0)
        )
        min_chunk_dur = self._stage_config.get("min_chunk_duration_sec", 5.0)

        n_workers = self._resolve_chunk_count(input_path, chunks_cfg, min_chunk_dur, current_fps)
        if n_workers <= 1:
            return self._execute_traditional_single(
                input_path, output_path, progress_callback, start, target=target, factor=factor
            )
        return self._execute_traditional_parallel(
            input_path,
            output_path,
            progress_callback,
            start,
            target=target,
            factor=factor,
            n_chunks=n_workers,
            current_fps=current_fps,
        )

    def _resolve_chunk_count(
        self,
        input_path: str,
        chunks_cfg: int,
        min_chunk_dur: float,
        current_fps: float | None,
    ) -> int:
        """Resolve the effective chunk count for parallel traditional interpolation.

        1 (or any input too short to split into 2+ chunks of at least
        min_chunk_duration_sec) means "don't chunk, run serially" -- preserves
        the pre-chunking behavior exactly.
        """
        import os as _os

        if chunks_cfg == 1:
            return 1
        try:
            probe_info = probe(input_path)
        except Exception:
            return 1
        duration = probe_info.duration
        if duration <= 0:
            return 1

        n = chunks_cfg if chunks_cfg and chunks_cfg > 0 else min(_os.cpu_count() or 4, 8)
        n = max(1, min(n, int(duration // max(min_chunk_dur, 0.1))))
        if n <= 1:
            return 1
        # Need at least 2 source frames per chunk for minterpolate to have any
        # pair to interpolate within a chunk.
        total_frames = probe_info.frame_count or int(duration * (current_fps or 30.0))
        if total_frames < n * 2:
            return 1
        return n

    def _execute_traditional_single(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        target: float,
        factor: int,
    ) -> StageResult:
        vf = self._minterpolate_filter(target)
        args = ["-i", input_path, "-vf", vf, "-c:a", "copy", "-y", output_path]

        def cb(p, m):
            self._report_progress(0.3 + p * 0.7, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

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
            metadata={"method": "traditional", "factor": factor, "parallel_chunks": 1},
            duration_sec=time.time() - start,
        )

    def _execute_traditional_parallel(
        self,
        input_path: str,
        output_path: str,
        progress_callback,
        start: float,
        target: float,
        factor: int,
        n_chunks: int,
        current_fps: float | None,
    ) -> StageResult:
        """Chunked/parallel minterpolate: see _execute_traditional's docstring."""
        import shutil as _shutil
        import tempfile as _tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed

        probe_info = probe(input_path)
        total_frames = probe_info.frame_count
        source_fps = current_fps or probe_info.framerate or 30.0
        if total_frames < n_chunks * 2:
            return self._execute_traditional_single(
                input_path, output_path, progress_callback, start, target=target, factor=factor
            )

        # Frame-index chunk boundaries (NOT wall-clock -ss seeking) so cuts are
        # exact/reproducible regardless of container timestamp quirks. k[i] is
        # inclusive on both sides: chunk i covers source frames [k[i], k[i+1]],
        # i.e. consecutive chunks share exactly one source frame (the overlap).
        k = [round(i * (total_frames - 1) / n_chunks) for i in range(n_chunks + 1)]

        tmp_dir = _tempfile.mkdtemp(prefix="avf_interp_chunks_")
        chunk_paths: list[str] = [
            os.path.join(tmp_dir, f"chunk_{i:03d}.mp4") for i in range(n_chunks)
        ]
        errors: list[str] = []
        completed_count = 0
        # Half a target-fps frame period, used to nudge trim boundaries so they
        # land cleanly between adjacent output frames instead of exactly on a
        # frame's PTS (where float rounding could go either way).
        eps = 1.0 / (2.0 * target)

        def _run_chunk(i: int) -> tuple[int, bool, str]:
            k_start, k_end = k[i], k[i + 1]
            # Deliberately do NOT reset PTS to 0 before minterpolate (select
            # alone preserves each frame's true source timestamp): minterpolate's
            # fps= retiming is duration/PTS-based, not a fixed N-in-M-out
            # multiply, so feeding it the frames' real absolute timestamps makes
            # its output timeline for this chunk land on the same 1/target-fps
            # grid a single whole-file pass would have produced for this same
            # time range -- letting chunks be trimmed and concatenated on that
            # shared grid instead of accumulating independent per-chunk
            # zero-based rounding error at every boundary.
            select_expr = f"between(n\\,{k_start}\\,{k_end})"
            interp_filter = self._minterpolate_filter(target)
            vf = f"select='{select_expr}',{interp_filter}"
            t_start, t_end = k_start / source_fps, k_end / source_fps
            if i > 0:
                # Exclude the shared boundary frame (owned by the previous
                # chunk's output) by starting just past its output timestamp.
                vf += f",trim=start={t_start + eps:.6f}:end={t_end + eps:.6f},setpts=PTS-STARTPTS"
            else:
                vf += f",trim=end={t_end + eps:.6f},setpts=PTS-STARTPTS"
            args = [
                "-i",
                input_path,
                "-vf",
                vf,
                "-an",
                "-y",
                chunk_paths[i],
            ]
            result = run_ffmpeg(args, timeout=self.stage_timeout())
            return i, result.returncode == 0, result.stderr[:300]

        self._report_progress(
            0.1, f"Interpolating {n_chunks} chunk(s) in parallel...", progress_callback
        )
        try:
            with ThreadPoolExecutor(max_workers=n_chunks) as executor:
                futures = {executor.submit(_run_chunk, i): i for i in range(n_chunks)}
                for future in as_completed(futures):
                    i, ok, err = future.result()
                    completed_count += 1
                    if not ok:
                        errors.append(f"chunk {i}: {err}")
                    self._report_progress(
                        0.1 + 0.7 * (completed_count / n_chunks),
                        f"Interpolated chunk {completed_count}/{n_chunks}",
                        progress_callback,
                    )

            if errors:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Parallel interpolation failed on {len(errors)} chunk(s): "
                    f"{'; '.join(errors)}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(0.85, "Concatenating interpolated chunks...", progress_callback)
            list_path = os.path.join(tmp_dir, "concat_list.txt")
            with open(list_path, "w") as f:
                for p in chunk_paths:
                    # ffmpeg concat demuxer format requires escaped single quotes;
                    # chunk paths are ours (uuid-free but simple), still escape for safety.
                    f.write(f"file '{p.replace(chr(39), chr(92) + chr(39))}'\n")

            concat_args = [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c:v",
                "copy",
                "-an",
                "-y",
                os.path.join(tmp_dir, "concat_video.mp4"),
            ]
            concat_result = run_ffmpeg(concat_args, timeout=self.stage_timeout())
            if concat_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Chunk concat failed: {concat_result.stderr[:300]}",
                    duration_sec=time.time() - start,
                )

            concat_video = os.path.join(tmp_dir, "concat_video.mp4")

            # Re-attach original audio (untouched -- interpolation doesn't change
            # the video's time span, so the original audio track still lines up).
            has_audio = probe_info.has_audio
            mux_args = ["-i", concat_video, "-i", input_path]
            if has_audio:
                mux_args += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy"]
            else:
                mux_args += ["-map", "0:v:0", "-c:v", "copy", "-an"]
            mux_args += ["-y", output_path]
            mux_result = run_ffmpeg(mux_args, timeout=self.stage_timeout())
            if mux_result.returncode != 0 or not os.path.exists(output_path):
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Final mux failed: {mux_result.stderr[:300]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Frame interpolation complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "method": "traditional",
                    "factor": factor,
                    "parallel_chunks": n_chunks,
                },
                duration_sec=time.time() - start,
            )
        finally:
            _shutil.rmtree(tmp_dir, ignore_errors=True)

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
            return self._ai_fallback_or_fail(
                "PyTorch not available",
                start,
                lambda: self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                ),
            )

        if not is_torch_available():
            return self._ai_fallback_or_fail(
                "PyTorch not installed",
                start,
                lambda: self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                ),
            )

        backend = self._stage_config.get("backend", "torch")

        try:
            from autovideofixer.ai.model_cache import ensure_model_available

            # ncnn backend resolves its own .param/.bin models inside
            # load_model(); the torch registry pre-flight would check the
            # wrong registry (see the same pattern in upscale.py).
            success, msg = (
                (True, "") if backend == "ncnn" else ensure_model_available(self._ai_model)
            )
            if not success:
                return self._ai_fallback_or_fail(
                    f"model not available: {msg}",
                    start,
                    lambda: self._execute_traditional(
                        input_path,
                        output_path,
                        progress_callback,
                        start,
                        target_fps=target_fps,
                        current_fps=current_fps,
                    ),
                )

        except Exception as e:
            return self._ai_fallback_or_fail(
                f"model check failed: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                ),
            )

        # Validate resolution - RIFE model expects input frames around 256x256 or larger
        probe_info = probe(input_path)
        input_w, input_h = probe_info.resolution
        if input_w < 64 or input_h < 64:
            self.logger.warning(
                f"Input resolution {input_w}x{input_h} too small for AI interpolation "
                "(RIFE expects >= 64px)"
            )
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        # Load and run RIFE
        interpolator = RIFEInterpolator(
            model_name=self._ai_model,
            device_preference=self.config.get("gpu", "preferred_device", default="auto"),
            backend=backend,
            vulkan_device=self.config.get("gpu", "vulkan_device", default=0),
        )

        if not interpolator.load_model():
            return self._ai_fallback_or_fail(
                "failed to load RIFE model",
                start,
                lambda: self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                ),
            )

        try:
            proc = FrameProcessor()
            # Stream frames in chunks to avoid loading entire video into memory.
            use_chunked = probe_info.frame_count and probe_info.frame_count > 1000

            if use_chunked:
                chunk_size = 25
                all_interpolated: list[Any] = []
                total_original_frames = probe_info.frame_count
                processed = 0
                carry_frame: Any = None

                def cb(current, total, msg):
                    self._report_progress(
                        0.1 + (processed / total_original_frames) * 0.9,
                        msg,
                        progress_callback,
                    )

                for chunk in proc.stream_frames(
                    input_path, chunk_size=chunk_size, max_frames=total_original_frames
                ):
                    # Carry the previous chunk's last frame into this chunk
                    # so a real interpolated frame is generated across the
                    # chunk boundary instead of a hard stutter every
                    # chunk_size source frames.
                    extended = [carry_frame, *chunk] if carry_frame is not None else chunk
                    chunk_interp = interpolator.interpolate_video(
                        extended, factor=factor, progress_callback=cb
                    )
                    if carry_frame is not None:
                        # interpolate_video's first output element is always
                        # the carried frame itself, already emitted as the
                        # previous chunk's final output element -- drop the
                        # duplicate.
                        chunk_interp = chunk_interp[1:]
                    all_interpolated.extend(chunk_interp)
                    carry_frame = chunk[-1]
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / total_original_frames) * 0.9,
                        "Interpolating chunk...",
                        progress_callback,
                    )
                proc.close()
                interpolated = all_interpolated
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

                interpolated = interpolator.interpolate_video(
                    frames, factor=factor, progress_callback=cb
                )

            if not interpolated:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames produced by interpolator",
                    duration_sec=time.time() - start,
                )

            # Write interpolated frames to temp file, then use FFmpeg to finalize.
            # Interpolation adds `factor`x more frames covering the SAME time
            # span as the original clip, so the output must be written at
            # `factor`x the original fps (i.e. the actual achieved framerate)
            # to preserve duration. Using the original input's fps here (a
            # pre-existing bug) kept the frame rate unchanged and instead
            # stretched the clip's duration by `factor`x.
            #
            # The temp file's extension must NOT be derived from the input's
            # extension: frames_to_video() always muxes with codec="libx264"
            # (H.264), which webm/mkv/etc. containers can't hold -- so e.g. a
            # .webm input produced a ".avf_interp_test.webm" temp target that
            # ffmpeg then failed to write into. ".mp4" always matches the
            # actual codec being written, regardless of input container.
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_interp_{os.path.splitext(os.path.basename(input_path))[0]}.mp4",
            )
            try:
                source_fps = current_fps if current_fps else self._get_input_fps(input_path)
                fps = source_fps * factor
                proc2 = FrameProcessor()
                temp_crf = self._stage_config.get("temp_crf", 16)
                if not proc2.frames_to_video(
                    interpolated, temp_path, fps=fps, crf=temp_crf, preset="medium"
                ):
                    proc2.close()
                    return StageResult(
                        status=StageStatus.FAILED,
                        error="Failed to write interpolated frames to temp file",
                        duration_sec=time.time() - start,
                    )
                proc2.close()

                # Mux the interpolated video back with the original audio (if
                # any). Hardcoding "-map 0:a:0" on an audio-less input makes
                # ffmpeg fail with no output file written; the return code was
                # previously never checked, so that failure was silently
                # reported as a completed job after deleting the only rendered
                # content. Branch on whether an audio stream actually exists.
                #
                # -c:v copy: the temp file above was already encoded once (at
                # temp_crf/"medium") -- re-encoding it AGAIN here at a
                # different crf (this used to be "-crf 18") was a second
                # lossy generation plus a wasted full x264 pass over the
                # whole video. Stream-copying the already-encoded video track
                # makes this mux bit-identical to the temp file's video
                # stream.
                has_audio = probe_info.has_audio
                mux_args = ["-i", input_path, "-i", temp_path]
                if has_audio:
                    mux_args += ["-map", "0:a:0", "-map", "1:v:0"]
                else:
                    mux_args += ["-map", "1:v:0"]
                mux_args += ["-c:v", "copy", "-c:a", "copy", "-y", output_path]

                mux_result = run_ffmpeg(mux_args, timeout=self.stage_timeout())

                if mux_result.returncode != 0 or not os.path.exists(output_path):
                    return StageResult(
                        status=StageStatus.FAILED,
                        error=f"Failed to mux interpolated output: {mux_result.stderr[:300]}",
                        duration_sec=time.time() - start,
                    )

                self._report_progress(1.0, "AI frame interpolation complete", progress_callback)
                orig_count = probe_info.frame_count or 0
                return StageResult(
                    status=StageStatus.COMPLETED,
                    output_path=output_path,
                    metadata={
                        "method": "ai",
                        "model": self._ai_model,
                        "factor": factor,
                        "frames_in": orig_count,
                        "frames_out": len(interpolated),
                    },
                    duration_sec=time.time() - start,
                )
            finally:
                # Always remove the internal temp file, on both the success
                # and failure paths -- previously this only ran after a
                # successful frames_to_video() + before the mux-result check,
                # so any failure before that point (e.g. a codec/container
                # mismatch) orphaned a partial temp file next to the user's
                # source video.
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            self.logger.error(f"RIFE processing failed: {e}")
            return self._ai_fallback_or_fail(
                f"inference exception: {e}",
                start,
                lambda: self._execute_traditional(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                ),
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
