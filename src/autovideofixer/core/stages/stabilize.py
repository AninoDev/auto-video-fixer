"""Auto Video Fixer - Video stabilization/deshaking stage."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class StabilizeStage(BaseStage):
    """Detect and correct video shake using FFmpeg's vidstab filters.

    Automatically detects shake intensity and applies correction.
    """

    name = "stabilize"
    display_name = "Stabilization"
    description = "Detect and correct camera shake"
    category = "enhancement"
    priority = 10
    supports_hardware_encoding = False

    def __init__(self, config):
        super().__init__(config)
        self._threshold = self._stage_config.get("threshold", 2.0)
        self._smoothness = self._stage_config.get("smoothness", 10)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        threshold: float | None = None,
        smoothness: int | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        thresh = threshold if threshold is not None else self._threshold
        smooth = smoothness if smoothness is not None else self._smoothness

        trf_path = None
        try:
            trf_fd, trf_path = tempfile.mkstemp(suffix=".trf", prefix="avf_stab_")
            os.close(trf_fd)

            self._report_progress(0.1, "Detecting shake...", progress_callback)

            # Step 1: Detect motion
            detection_args = [
                "-i",
                input_path,
                "-vf",
                f"vidstabdetect=shakiness=10:accuracy=15:result={trf_path}",
                "-f",
                "null",
                "-",
            ]
            det_result = run_ffmpeg(detection_args, timeout=300)

            if det_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Shake detection failed: {det_result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            # Step 2: Apply stabilization
            self._report_progress(0.5, "Applying stabilization...", progress_callback)

            transform_args = [
                "-hide_banner",
                "-i",
                input_path,
                "-vf",
                f"vidstabtransform=smoothing={smooth}:input={trf_path}",
                "-c:a",
                "copy",
                "-y",
                output_path,
            ]
            trans_result = run_ffmpeg(transform_args, timeout=600)

            if trans_result.returncode != 0:
                # Remove empty/failed output file
                if os.path.exists(output_path):
                    os.remove(output_path)
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Stabilization failed: {trans_result.stderr[-300:]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Stabilization complete", progress_callback)

            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={"smoothness": smooth, "threshold": thresh},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
        finally:
            if trf_path and os.path.exists(trf_path):
                try:
                    os.remove(trf_path)
                except OSError:
                    pass
