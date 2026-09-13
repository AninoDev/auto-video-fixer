"""Auto Video Fixer - Speed adjustment stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.config import VALID_AUDIO_RESAMPLERS, VALID_AUDIO_SPEED_METHODS
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class SpeedStage(BaseStage):
    """Adjust video and audio speed using time stretching/compression.

    Supports both speed up and slow down with pitch preservation for audio.
    """

    name = "speed"
    display_name = "Speed Adjustment"
    description = "Adjust playback speed"
    category = "enhancement"
    priority = 50
    supports_gpu = False

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        factor = self._stage_config.get("factor", 1.0)
        if factor == 1.0:
            return False, "No speed change requested"
        return True, None

    @staticmethod
    def _build_atempo_chain(speed: float) -> str:
        """Build an atempo filter chain for an arbitrary positive speed factor.

        A single atempo filter only accepts tempo scale factors in [0.5, 100].
        Factors outside that range (e.g. slow-motion below 0.5x) are reached
        by chaining multiple atempo stages whose product equals `speed`.
        """
        if speed <= 0:
            raise ValueError(f"speed factor must be positive, got {speed}")

        stages: list[float] = []
        remaining = speed
        if remaining < 0.5:
            while remaining < 0.5:
                stages.append(0.5)
                remaining /= 0.5
        elif remaining > 100:
            while remaining > 100:
                stages.append(100.0)
                remaining /= 100.0
        stages.append(remaining)

        return ",".join(f"atempo={s}" for s in stages)

    @staticmethod
    def _build_rubberband_filter(speed: float) -> str:
        """Build a single ``rubberband`` tempo filter for an arbitrary positive speed factor.

        Unlike ``atempo``, librubberband's ``tempo`` parameter accepts any positive
        value directly -- no [0.5, 100] range restriction -- so a single filter covers
        the whole factor range with none of ``_build_atempo_chain``'s multi-stage
        artifacting at extreme slow-motion/speed-up factors.
        """
        if speed <= 0:
            raise ValueError(f"speed factor must be positive, got {speed}")
        return f"rubberband=tempo={speed}"

    @staticmethod
    def _build_asetrate_chain(
        speed: float,
        input_sample_rate: int,
        audio_sample_rate: int,
        resampler: str,
    ) -> str:
        """Build a resample-based (deliberately pitch-CHANGING) speed chain.

        ``asetrate`` relabels the stream at a new sample rate without resampling the
        underlying samples, so playback speed AND pitch both shift by `speed` together
        -- exactly undoing a phone's slow-motion capture trick, where footage is
        recorded at an elevated mic sample rate and mapped down to normal speed in the
        container; speeding it back up this way restores the mic's true original
        pitch, which a pitch-preserving method (atempo/rubberband) would leave wrong.

        The trailing ``aresample`` renormalizes the now-mislabeled sample rate to a
        concrete value so downstream encoding/muxing sees a sane rate: `audio_sample_rate`
        when configured (> 0), else back to the ORIGINAL `input_sample_rate` (never left
        at the intermediate asetrate value).
        """
        if speed <= 0:
            raise ValueError(f"speed factor must be positive, got {speed}")
        if input_sample_rate <= 0:
            raise ValueError(f"input_sample_rate must be positive, got {input_sample_rate}")
        new_rate = round(input_sample_rate * speed)
        target_rate = audio_sample_rate if audio_sample_rate > 0 else input_sample_rate
        return f"asetrate={new_rate},aresample={target_rate}:resampler={resampler}"

    def _resolve_audio_filter(self, speed: float, input_path: str) -> tuple[str, str]:
        """Resolve ``stages.speed.audio_method`` into an actual ffmpeg audio filter chain.

        Returns ``(filter_chain, method_used)`` -- `method_used` can differ from the
        configured `audio_method` when a fallback kicked in: the "rubberband" filter is
        missing from this ffmpeg build, or "asetrate"'s required input sample rate
        couldn't be determined. Both fallbacks land on "atempo" (pitch-preserving,
        always available -- no external library dependency) with a WARNING log, never
        a failed stage.

        Raises ``ValueError`` for an unrecognized `audio_method`/`resampler` -- same
        "invalid enum config value" posture as `core/output_check.py`'s `fit_mode`
        check: validated at use-time (only matters if this stage actually runs), not
        eagerly at Config/Pipeline construction.
        """
        method = self._stage_config.get("audio_method", "atempo")
        if method not in VALID_AUDIO_SPEED_METHODS:
            raise ValueError(
                f"Invalid stages.speed.audio_method: {method!r} "
                f"(must be one of {VALID_AUDIO_SPEED_METHODS!r})"
            )

        if method == "rubberband":
            from autovideofixer.core.ffmpeg_utils import has_filter

            if has_filter("rubberband", self.config):
                return self._build_rubberband_filter(speed), "rubberband"
            self.logger.warning(
                "stages.speed.audio_method=rubberband requested but this ffmpeg build "
                "lacks librubberband (rubberband filter not found); falling back to atempo"
            )
            return self._build_atempo_chain(speed), "atempo"

        if method == "asetrate":
            resampler = self._stage_config.get("resampler", "soxr")
            if resampler not in VALID_AUDIO_RESAMPLERS:
                raise ValueError(
                    f"Invalid stages.speed.resampler: {resampler!r} "
                    f"(must be one of {VALID_AUDIO_RESAMPLERS!r})"
                )

            from autovideofixer.core.ffmpeg_utils import probe

            input_sample_rate = 0
            probe_error: Exception | None = None
            try:
                info = probe(input_path, self.config)
                audio_streams = info.audio_streams
                if audio_streams:
                    input_sample_rate = audio_streams[0].sample_rate
            except Exception as e:
                probe_error = e

            if input_sample_rate <= 0:
                if probe_error is not None:
                    self.logger.warning(
                        "stages.speed.audio_method=asetrate could not probe %s for its "
                        "input audio sample rate (%s); falling back to atempo",
                        input_path,
                        probe_error,
                    )
                else:
                    self.logger.warning(
                        "stages.speed.audio_method=asetrate: could not determine %s's "
                        "input audio sample rate (no audio stream, or sample_rate=0); "
                        "falling back to atempo",
                        input_path,
                    )
                return self._build_atempo_chain(speed), "atempo"

            audio_sample_rate = int(self._stage_config.get("audio_sample_rate", 48000) or 0)
            return (
                self._build_asetrate_chain(speed, input_sample_rate, audio_sample_rate, resampler),
                "asetrate",
            )

        # "atempo" (default)
        return self._build_atempo_chain(speed), "atempo"

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        factor: float | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Adjusting speed...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to speed stage",
                duration_sec=time.time() - start,
            )

        try:
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg, timing_output_args

            speed = factor if factor is not None else self._stage_config.get("factor", 1.0)

            if speed == 1.0:
                return StageResult(
                    status=StageStatus.SKIPPED,
                    skipped_reason="No speed change needed",
                    duration_sec=0.0,
                )

            # Video speed: setpts filter
            # Audio speed: resolved per stages.speed.audio_method -- "atempo"
            # (default, chained for factors outside a single atempo's [0.5, 100]
            # range), "rubberband" (pitch-preserving, no chaining, needs
            # librubberband -- falls back to atempo if missing), or "asetrate"
            # (deliberately pitch-changing, needs the input's probed sample
            # rate -- falls back to atempo if it can't be determined). See
            # _resolve_audio_filter().
            video_filter = f"setpts={1.0 / speed}*PTS"
            audio_filter, audio_method_used = self._resolve_audio_filter(speed, input_path)
            audio_sample_rate = int(self._stage_config.get("audio_sample_rate", 48000) or 0)

            args = [
                "-i",
                input_path,
                "-vf",
                video_filter,
                "-af",
                audio_filter,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
            ]
            # 0 = leave the input's own sample rate alone (no -ar emitted).
            if audio_sample_rate > 0:
                args += ["-ar", str(audio_sample_rate)]
            args += [
                *timing_output_args(),
                "-y",
                output_path,
            ]

            def cb(p, m):
                self._report_progress(0.3 + p * 0.7, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Speed adjustment failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Speed adjustment complete", progress_callback)

            # `setpts` rescales timestamps and keeps the frame COUNT, so the
            # stream's real cadence changes even though nothing here re-times
            # frames: N frames over a duration of D/speed is an effective rate
            # of `in_fps * speed`. Reporting it as `fps_out` is what keeps
            # input_info["true_framerate"] current (see Pipeline.execute_job's
            # fps_out propagation), and that matters twice over:
            #   - `encode`'s CFR path pins `-r true_framerate`; without this a
            #     0.5x slow-motion clip stayed tagged at its PRE-speed rate, so
            #     ffmpeg duplicated frames to reach it -- the output claimed
            #     60fps while carrying only 30fps of real motion.
            #   - a second `interpolate` occurrence placed after `speed` (via a
            #     repeated pipeline.default_order entry) can only plan real
            #     slow-motion interpolation if it sees the post-speed rate.
            in_fps = 0.0
            info = kwargs.get("input_info") or {}
            try:
                in_fps = float(info.get("true_framerate") or info.get("framerate") or 0.0)
            except TypeError, ValueError:
                in_fps = 0.0

            metadata: dict[str, Any] = {
                # REQUIREMENTS.md § 6.4: speed is traditional-only (FFmpeg
                # setpts/atempo, no AI path) -- uniform provenance.
                "method": "traditional",
                "speed_factor": speed,
                # Audio filter actually used -- may differ from the configured
                # stages.speed.audio_method when a fallback kicked in (see
                # _resolve_audio_filter).
                "audio_method": audio_method_used,
            }
            if in_fps > 0:
                metadata["fps_in"] = in_fps
                metadata["fps_out"] = in_fps * speed

            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata=metadata,
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
