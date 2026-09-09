"""Auto Video Fixer - Frame interpolation stage."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

# Epsilon for float fps comparisons throughout this module (target/intermediate
# fps come from probes and preset math, never exact binary fractions).
_FPS_EPS = 0.01


@dataclass(frozen=True)
class InterpPlan:
    """Result of planning how the AI/RIFE path should reach ``target_fps``.

    See ``_plan_ai_interpolation`` for the rules that produce this.
    """

    rife_factor: int
    run_rife: bool
    run_minterpolate_finish: bool
    intermediate_fps: float


def _plan_ai_interpolation(
    current_fps: float, target_fps: float, hybrid_enabled: bool
) -> InterpPlan:
    """Plan the "RIFE under, then minterpolate up" strategy for the AI path.

    RIFE (see ai/wrappers/interpolate.py's ``interpolate_video``) is
    timestep-conditioned and supports arbitrary INTEGER factors, but never a
    fractional one -- it can only exactly reach ``current_fps * N`` for an
    integer N. minterpolate (``_minterpolate_filter``), by contrast, retimes
    to ANY target fps exactly via its own ``fps=`` sub-option and true
    motion-compensated interpolation (``mi_mode=mci``), so it's used here as
    a "finish pass" for the fractional remainder RIFE can't reach on its own.

    Rules (largest integer RIFE factor that stays AT OR BELOW target, i.e.
    floor(target/current)):
      - rife_factor = floor(target_fps / current_fps).
      - If rife_factor >= 2: run RIFE at that factor. If the resulting
        intermediate fps (current*rife_factor) is still short of target (by
        more than _FPS_EPS), also run a minterpolate finish pass -- but only
        when hybrid_enabled (the finish pass is itself a hybrid-only
        behavior). If hybrid is disabled and a finish would otherwise be
        needed, there is nothing else to do: RIFE's own integer-factor output
        is the final answer (matches legacy: no finish, output landed at
        current*rife_factor, not exactly target).
      - If rife_factor <= 1: no integer RIFE factor helps without overshooting
        (e.g. 50->60, 24->30, 60->75 all floor to 1). Two behaviors:
          - hybrid_enabled: skip RIFE entirely, reach target via minterpolate
            ALONE (the caller delegates to _execute_traditional).
          - hybrid disabled: LEGACY back-compat -- force rife_factor=2 and run
            RIFE anyway (the old overshoot behavior, e.g. 50->60 -> 100fps
            output), no finish pass.
    """
    if current_fps <= 0 or target_fps <= current_fps:
        # Shouldn't happen -- should_run() gates this -- but stay defined.
        return InterpPlan(
            rife_factor=2, run_rife=True, run_minterpolate_finish=False, intermediate_fps=0.0
        )

    rife_factor = int(target_fps / current_fps)  # floor

    if rife_factor >= 2:
        intermediate_fps = current_fps * rife_factor
        needs_finish = intermediate_fps < target_fps - _FPS_EPS
        run_finish = needs_finish and hybrid_enabled
        return InterpPlan(
            rife_factor=rife_factor,
            run_rife=True,
            run_minterpolate_finish=run_finish,
            intermediate_fps=intermediate_fps,
        )

    # rife_factor <= 1
    if hybrid_enabled:
        return InterpPlan(
            rife_factor=0,
            run_rife=False,
            run_minterpolate_finish=True,
            intermediate_fps=current_fps,
        )
    # Legacy back-compat: force factor 2, overshoot, no finish.
    return InterpPlan(
        rife_factor=2,
        run_rife=True,
        run_minterpolate_finish=False,
        intermediate_fps=current_fps * 2,
    )


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
        self._hybrid = self._stage_config.get("hybrid_ai_minterpolate", True)

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
        # REQUIREMENTS.md § 12.3: prefer the recovered true content cadence
        # (published by the retime stage) over the encoded/probed framerate
        # -- on a 24-in-60 input targeting 60fps, "framerate" alone reads as
        # 60->60 (already at target, skip) when the real content is 24fps
        # and genuinely needs interpolating.
        current_fps = input_info.get("true_framerate") or input_info.get("framerate", 0)
        if current_fps >= target_fps:
            return False, "Already at or above target framerate"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_fps: float | None = None,
        method: str | None = None,
        parallel_chunks: int | None = None,
        input_info: dict[str, Any] | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Running frame interpolation...", progress_callback)

        if target_fps is None:
            quality_target = self.config.get("quality", "quality_target", default={})
            target_fps = quality_target.get("target_framerate")

        # Resolve "ai" vs "traditional" per the shared precedence (explicit
        # method= kwarg > stages.interpolate.use_ai > general.use_ai > the
        # deliberate hardcoded default below) -- see
        # BaseStage.resolve_ai_method's docstring. interpolate's auto default
        # stays "traditional" (minterpolate): fast with decent quality, a
        # deliberate asymmetry vs. upscale/deblock defaulting to AI.
        method, source = self.resolve_ai_method(method, "traditional")
        backend = self._stage_config.get("backend", "torch")
        self._log_ai_method_choice(
            method,
            source,
            ai_desc=f"AI interpolation (RIFE '{self._ai_model}', backend {backend})",
            traditional_desc="traditional minterpolate",
            ai_hint=f"AI/RIFE model '{self._ai_model}'",
        )

        try:
            # REQUIREMENTS.md § 12.3: prefer the recovered true content
            # cadence (input_info["true_framerate"], published by the retime
            # stage) over a fresh probe -- do NOT re-probe for fps downstream
            # of retime: avg_frame_rate is unreliable on a VFR intermediate
            # (empirically: a 48-frame 2s VFR MKV still advertised
            # avg_frame_rate=60/1). Only re-probes as a fallback when the
            # caller didn't pass input_info at all (e.g. direct/test
            # invocations), matching the pre-existing behavior exactly.
            current_fps = (input_info or {}).get("true_framerate") or (input_info or {}).get(
                "framerate"
            )
            if not current_fps:
                from autovideofixer.core.ffmpeg_utils import get_video_info

                probed_info = get_video_info(input_path)
                current_fps = probed_info.get("framerate", 30.0)

            if method == "ai":
                return self._execute_ai(
                    input_path,
                    output_path,
                    progress_callback,
                    start,
                    target_fps=target_fps,
                    current_fps=current_fps,
                    input_info=input_info,
                )
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
                parallel_chunks=parallel_chunks,
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
            # "fps_out" must be reported on EVERY completed path (AI and
            # traditional alike): the pipeline uses it to refresh
            # input_info["true_framerate"] so the encode stage's CFR path
            # pins -r to the INTERPOLATED rate. Omitting it here made a
            # retimed 24-in-60 input encode back down to 24fps, silently
            # discarding every frame minterpolate had just synthesized.
            metadata={
                "method": "traditional",
                "factor": factor,
                "parallel_chunks": 1,
                "fps_out": target,
            },
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
                    # See the single-chunk path above: "fps_out" is required
                    # on every completed path so encode's CFR pin uses the
                    # interpolated rate, not the pre-interpolation cadence.
                    "fps_out": target,
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
        input_info: dict[str, Any] | None = None,
        **kwargs,
    ) -> StageResult:
        """AI frame interpolation using RIFE model.

        Falls back to traditional FFmpeg method if PyTorch or model
        files are not available.
        """
        # Plan the "RIFE under, then minterpolate up" strategy -- see
        # _plan_ai_interpolation's docstring. Falls back to the legacy
        # forced-factor-2 behavior if fps info is missing.
        if target_fps and current_fps and current_fps > 0:
            plan = _plan_ai_interpolation(current_fps, target_fps, self._hybrid)
        else:
            plan = InterpPlan(
                rife_factor=2, run_rife=True, run_minterpolate_finish=False, intermediate_fps=0.0
            )

        if not plan.run_rife:
            # Target is reachable from current_fps via minterpolate ALONE (the
            # largest integer RIFE factor is <= 1, so RIFE could only help by
            # overshooting). This is a deliberate strategy choice, not an
            # AI-unavailable fallback -- route straight to the traditional
            # path (which already retimes to the exact target) rather than
            # through _ai_fallback_or_fail, so the resulting metadata
            # method="traditional" is accurate, not a logged "fallback".
            self.logger.info(
                f"interpolate: target {target_fps}fps is reachable from "
                f"{current_fps}fps via minterpolate alone (integer RIFE factor "
                f"{plan.rife_factor} <= 1) -- skipping RIFE, running minterpolate-only"
            )
            return self._execute_traditional(
                input_path,
                output_path,
                progress_callback,
                start,
                target_fps=target_fps,
                current_fps=current_fps,
            )

        factor = plan.rife_factor
        if plan.run_minterpolate_finish:
            self.logger.info(
                f"interpolate: RIFE factor {factor} ({current_fps}fps -> "
                f"{plan.intermediate_fps}fps) then a minterpolate finish pass to "
                f"reach exact target {target_fps}fps"
            )
        else:
            self.logger.info(
                f"interpolate: RIFE factor {factor} ({current_fps}fps -> "
                f"{plan.intermediate_fps}fps), no finish pass needed"
            )

        try:
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
            from autovideofixer.ai.frame_pipe import get_frame_reader, get_frame_writer

            # Stream every chunk straight to a frame writer instead of
            # accumulating the whole interpolated output in RAM. Interpolation
            # produces `factor`x MORE frames than it reads, so the old
            # `all_interpolated.extend(...)`-per-chunk (chunked path) / full
            # `extract_frames()` (non-chunked, <=1000-frame path) both held
            # every OUTPUT frame for the entire clip -- for a 61s 4K 30->60fps
            # scene that's ~3667 frames * ~24.9MB = ~91GB resident, enough to
            # OOM/swap-thrash the host. ALL videos now go through this single
            # streaming path regardless of frame count (the old `use_chunked`
            # / `frame_count > 1000` threshold is gone entirely) -- mirrors
            # UpscaleStage._execute_ai (upscale.py), the last other AI stage
            # to carry this same full-buffer pattern before it was fixed.
            chunk_size = 25
            read_ahead = self._stage_config.get("read_ahead", 2)
            write_queue_depth = self._stage_config.get("write_queue_depth", 4)
            in_width, in_height = probe_info.resolution
            total_original_frames = probe_info.frame_count or 0

            reader = get_frame_reader(
                input_path,
                in_width,
                in_height,
                chunk_size=chunk_size,
                read_ahead=read_ahead,
            )

            source_fps = current_fps if current_fps else self._get_input_fps(input_path)
            fps = source_fps * factor

            # The temp file is DELIBERATELY Matroska (.mkv, H.264), not .mp4:
            # Matroska is written incrementally with clusters flushed as they
            # go, so a partial file stays playable/recoverable even if the
            # process is killed mid-write (OOM, host shutdown, power loss).
            # Default MP4 only writes its moov index at clean finalize, so a
            # truncated MP4 (the old behavior here) is unplayable if the
            # process dies mid-stream. H.264-in-MKV remains stream-copyable
            # (`-c:v copy`) into the final MP4 below, so this costs nothing
            # at finalize time. (The extension is still deliberately NOT
            # derived from the input's own extension -- get_frame_writer
            # always muxes with codec="libx264", which not every input
            # container can hold; ".mkv" always matches the actual codec
            # being written.)
            temp_path = os.path.join(
                os.path.dirname(input_path) or ".",
                f".avf_interp_{os.path.splitext(os.path.basename(input_path))[0]}.mkv",
            )

            writer: Any = None
            frames_written = 0
            processed = 0
            carry_frame: Any = None
            temp_crf = self._stage_config.get("temp_crf", 16)

            def cb(current, total, msg):
                denom = total_original_frames or 1
                self._report_progress(
                    0.1 + (processed / denom) * 0.9,
                    msg,
                    progress_callback,
                )

            try:
                while True:
                    chunk = reader.next_batch()
                    if chunk is None:
                        break

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

                    if writer is None and chunk_interp:
                        out_h, out_w = chunk_interp[0].shape[:2]
                        writer = get_frame_writer(
                            temp_path,
                            out_w,
                            out_h,
                            fps,
                            crf=temp_crf,
                            preset="medium",
                            write_queue=write_queue_depth,
                        )
                    # Write straight to the ffmpeg pipe and discard --
                    # `chunk_interp` is never appended to a running list, so
                    # peak memory stays ~chunk_size*factor frames, not
                    # total_frames*factor.
                    if writer is not None:
                        writer.write_batch(chunk_interp)
                        frames_written += len(chunk_interp)

                    carry_frame = chunk[-1]
                    processed += len(chunk)

                    self._report_progress(
                        0.1 + (processed / (total_original_frames or 1)) * 0.9,
                        "Interpolating chunk...",
                        progress_callback,
                    )
                reader.close()
                write_ok = writer.close() if writer is not None else False
            except Exception:
                reader.close()
                if writer is not None:
                    writer.close()
                # A genuine in-loop failure (e.g. inference exception) is a
                # graceful in-stage failure, not a hard kill -- preserve
                # whatever was flushed so far instead of leaking/discarding
                # it, then re-raise so the outer except still routes through
                # _ai_fallback_or_fail.
                self._preserve_or_discard_partial_temp(temp_path, output_path)
                raise

            if frames_written == 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error="No frames produced by interpolator",
                    duration_sec=time.time() - start,
                )
            if not write_ok:
                self._preserve_or_discard_partial_temp(temp_path, output_path)
                return StageResult(
                    status=StageStatus.FAILED,
                    error="Failed to write interpolated frames to temp file",
                    duration_sec=time.time() - start,
                )

            # Hybrid finish pass: the RIFE temp above is at intermediate_fps
            # (an exact integer multiple of current_fps), short of the exact
            # target -- run ONE minterpolate pass over it (video only) to
            # retime to exactly target_fps before the final audio mux. This
            # is a SEPARATE temp file (also .mkv for the same crash-resilience
            # reasons as the RIFE temp) so the RIFE output survives if this
            # pass fails.
            finish_temp_path: str | None = None
            video_source_for_mux = temp_path
            if plan.run_minterpolate_finish:
                # plan.run_minterpolate_finish can only be True via
                # _plan_ai_interpolation's real branches (both require a
                # non-None target_fps/current_fps to compute) -- the
                # target_fps-missing fallback plan above always has
                # run_minterpolate_finish=False. assert (not just a comment)
                # so mypy narrows float | None -> float here.
                assert target_fps is not None
                finish_temp_path = os.path.join(
                    os.path.dirname(input_path) or ".",
                    f".avf_interp_finish_{os.path.splitext(os.path.basename(input_path))[0]}.mkv",
                )
                finish_vf = self._minterpolate_filter(target_fps)
                finish_args = [
                    "-i",
                    temp_path,
                    "-vf",
                    finish_vf,
                    "-an",
                    "-y",
                    finish_temp_path,
                ]
                self._report_progress(
                    0.92, "Minterpolate finish pass to exact target fps...", progress_callback
                )
                finish_result = run_ffmpeg(finish_args, timeout=self.stage_timeout())
                if finish_result.returncode != 0 or not os.path.exists(finish_temp_path):
                    self._preserve_or_discard_partial_temp(temp_path, output_path)
                    if os.path.exists(finish_temp_path):
                        os.unlink(finish_temp_path)
                    return StageResult(
                        status=StageStatus.FAILED,
                        error=(f"Minterpolate finish pass failed: {finish_result.stderr[:300]}"),
                        duration_sec=time.time() - start,
                    )
                video_source_for_mux = finish_temp_path

            try:
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
                # stream -- and works unchanged with an MKV/H.264 temp, since
                # `-c:v copy` only cares about the codec, not the container.
                # This mux IS the "MKV -> MP4 remux" -- no separate step
                # needed.
                # video_source_for_mux is the RIFE temp directly (no finish
                # pass needed) or the finish-pass temp (already at exactly
                # target_fps) -- either way it's already fully encoded once,
                # so this mux still stream-copies the video track.
                has_audio = probe_info.has_audio
                mux_args = ["-i", input_path, "-i", video_source_for_mux]
                if has_audio:
                    mux_args += ["-map", "0:a:0", "-map", "1:v:0"]
                else:
                    mux_args += ["-map", "1:v:0"]
                mux_args += ["-c:v", "copy", "-c:a", "copy", "-y", output_path]

                mux_result = run_ffmpeg(mux_args, timeout=self.stage_timeout())

                if mux_result.returncode != 0 or not os.path.exists(output_path):
                    # Graceful in-stage failure (not a hard kill): preserve the
                    # partial MKV for inspection instead of silently unlinking
                    # it in the `finally` below.
                    self._preserve_or_discard_partial_temp(temp_path, output_path)
                    if finish_temp_path and os.path.exists(finish_temp_path):
                        os.unlink(finish_temp_path)
                    return StageResult(
                        status=StageStatus.FAILED,
                        error=f"Failed to mux interpolated output: {mux_result.stderr[:300]}",
                        duration_sec=time.time() - start,
                    )

                self._report_progress(1.0, "AI frame interpolation complete", progress_callback)
                orig_count = probe_info.frame_count or 0
                interp_metadata: dict[str, Any] = {
                    "method": "ai",
                    "model": self._ai_model,
                    "factor": factor,
                    "rife_factor": factor,
                    "minterpolate_finish": plan.run_minterpolate_finish,
                    "fps_out": target_fps,
                    "frames_in": orig_count,
                    "frames_out": frames_written,
                }
                cadence = (input_info or {}).get("cadence")
                if cadence and not cadence.get("is_regular", True):
                    # REQUIREMENTS.md § 12.4b: the raw-pipe writer's fixed
                    # carrier rate (fps = source_fps * factor above)
                    # linearizes a genuinely irregular recovered cadence to
                    # a constant rate -- a real, bounded limitation that
                    # must be surfaced, never silent.
                    interp_metadata["timeline_linearized"] = True
                return StageResult(
                    status=StageStatus.COMPLETED,
                    output_path=output_path,
                    metadata=interp_metadata,
                    duration_sec=time.time() - start,
                )
            finally:
                # SUCCESS-path cleanup only: remove the internal temp file
                # once the final muxed output supersedes it. On a HARD kill
                # (SIGKILL/OOM/power loss) this `finally` never runs at all,
                # so the partial .mkv simply survives on disk, playable up to
                # its last flushed cluster -- that's the primary benefit of
                # the MKV temp and is intentional; do NOT add code here that
                # would preemptively delete it on some other path. Graceful
                # in-stage failures are handled above via
                # `_preserve_or_discard_partial_temp()` (rename, not delete)
                # before this block ever runs, and both of those return
                # early -- so if this line executes at all, mux succeeded.
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if finish_temp_path and os.path.exists(finish_temp_path):
                    os.unlink(finish_temp_path)

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

    def _preserve_or_discard_partial_temp(self, temp_path: str, output_path: str) -> None:
        """On a graceful in-stage failure, rename the partial MKV temp aside.

        A HARD kill (SIGKILL/OOM/power loss) never reaches this method at
        all -- the process just dies and the partial `.mkv` survives at
        `temp_path` untouched, playable up to its last flushed cluster. This
        method only runs on a graceful failure the stage itself detected
        (mux returned nonzero, an inference exception, a write failure) --
        renaming to a visible sibling of the intended output lets the user
        inspect how far interpolation got and judge whether a rerun is
        worthwhile. Falls back to leaving the temp file in place if the
        rename itself fails (e.g. cross-device, permissions).
        """
        if not os.path.exists(temp_path):
            return
        stem = os.path.splitext(output_path)[0]
        partial_path = f"{stem}_interp_partial.mkv"
        try:
            os.replace(temp_path, partial_path)
            self.logger.warning(
                f"partial interpolated output preserved for inspection: {partial_path}"
            )
        except OSError as e:
            self.logger.warning(f"could not preserve partial interpolated output: {e}")

    def _get_input_fps(self, path: str) -> float:
        """Get input video framerate."""
        from autovideofixer.core.ffmpeg_utils import probe

        try:
            info = probe(path)
            return info.framerate or 30.0
        except Exception:
            return 30.0
