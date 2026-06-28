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

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        factor = self._stage_config.get("factor", 1.0)
        if factor == 1.0:
            return False, "No speed change requested"
        return True, None

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

        try:
            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            speed = factor if factor is not None else self._stage_config.get("factor", 1.0)

            if speed == 1.0:
                return StageResult(
                    status=StageStatus.SKIPPED,
                    skipped_reason="No speed change needed",
                    duration_sec=0.0,
                )

            # Video speed: setpts filter
            # Audio speed: atempo filter (with pitch preservation via aresample)
            video_filter = f"setpts={1.0 / speed}*PTS"
            audio_filter = f"atempo={speed}"

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
                "-y",
                output_path or input_path,
            ]

            def cb(p, m):
                self._report_progress(0.3 + p * 0.7, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=600)

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Speed adjustment failed: {result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Speed adjustment complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path or input_path,
                metadata={"speed_factor": speed},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
