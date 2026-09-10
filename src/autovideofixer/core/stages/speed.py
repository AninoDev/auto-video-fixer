"""Auto Video Fixer - Speed adjustment stage."""

from __future__ import annotations

import time
from typing import Any

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
            # Audio speed: atempo filter chain (a single atempo only accepts
            # tempo scale factors in [0.5, 100]; chain multiple stages for
            # factors outside that range, e.g. slow-motion below 0.5x)
            video_filter = f"setpts={1.0 / speed}*PTS"
            audio_filter = self._build_atempo_chain(speed)

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
