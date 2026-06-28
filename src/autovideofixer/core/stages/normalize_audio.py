"""Auto Video Fixer - Audio normalization and volume normalization."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class NormalizeAudioStage(BaseStage):
    """Normalize audio volume to target level (EBU R128 / LUFS).

    Uses FFmpeg's loudnorm filter for two-pass normalization.
    """

    name = "normalize_audio"
    display_name = "Audio Normalization"
    description = "Normalize audio volume to target loudness"
    category = "enhancement"
    priority = 40
    supports_gpu = False

    def __init__(self, config):
        super().__init__(config)
        self._target_db = self._stage_config.get("target_db", -23.0)
        self._true_peak = self._stage_config.get("true_peak_db", -2.0)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if not input_info.get("has_audio"):
            return False, "No audio stream"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_db: float | None = None,
        true_peak: float | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Normalizing audio...", progress_callback)

        try:
            target = target_db if target_db is not None else self._target_db
            peak = true_peak if true_peak is not None else self._true_peak

            from autovideofixer.core.ffmpeg_utils import run_ffmpeg

            # Two-pass loudnorm: first pass measures, second pass applies
            args = [
                "-i",
                input_path,
                "-af",
                f"loudnorm=I={target}:TP={peak}:print_format=json",
                "-vn",
                "-sn",
                "-dn",
                "-f",
                "null",
                "-",
            ]
            measure_result = run_ffmpeg(args, timeout=300)

            if measure_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Loudness measurement failed: {measure_result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            measured_i = target
            measured_tp = peak
            measured_lra = 7.0
            measured_thresh = -34.0
            offset = 0.0
            try:
                measured_data = _extract_loudnorm_json(measure_result.stderr)
                measured_i = measured_data.get("input_i", target)
                measured_tp = measured_data.get("input_tp", peak)
                measured_lra = measured_data.get("input_lra", 7.0)
                measured_thresh = measured_data.get("input_thresh", -34.0)
                offset = measured_data.get("target_offset", 0.0)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(
                    f"Failed to parse loudnorm first pass JSON: {e}, using target values"
                )

            # Apply normalization
            self._report_progress(0.5, "Applying normalization...", progress_callback)

            args = [
                "-i",
                input_path,
                "-af",
                f"loudnorm=I={target}:TP={peak}:measured_I={measured_i}:measured_TP={measured_tp}:measured_LRA={measured_lra}:measured_thresh={measured_thresh}:offset={offset}:linear=true:print_format=summary",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-y",
                output_path or input_path,
            ]

            def cb(p, m):
                self._report_progress(0.5 + p * 0.5, m, progress_callback)

            norm_result = run_ffmpeg(args, progress_callback=cb, timeout=600)

            if norm_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Normalization failed: {norm_result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Audio normalization complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path or input_path,
                metadata={"target_db": target, "true_peak": peak},
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )


class NormalizeVolumeStage(BaseStage):
    """Normalize video volume (alias for audio normalization with different config)."""

    name = "normalize_volume"
    display_name = "Volume Normalization"
    description = "Normalize video volume"
    category = "enhancement"
    priority = 40
    supports_gpu = False

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if not input_info.get("has_audio"):
            return False, "No audio stream"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        **kwargs,
    ) -> StageResult:
        # Reuse audio normalization logic
        normalizer = NormalizeAudioStage(self.config)
        return normalizer.execute(input_path, output_path, progress_callback, **kwargs)


def _extract_loudnorm_json(output: str) -> dict[str, Any]:
    """Extract loudnorm JSON from FFmpeg output.

    The loudnorm filter with print_format=json outputs a JSON object to stdout.
    This function extracts it from the output, handling cases where other
    FFmpeg output may be mixed in.
    """
    match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', output, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise
