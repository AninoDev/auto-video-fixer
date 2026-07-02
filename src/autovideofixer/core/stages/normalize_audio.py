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

    NOTE: this performs the exact same operation as NormalizeVolumeStage below
    (same loudnorm filter, same algorithm) -- the two names/config sections exist
    for backward-compatible --stage/config addressing, not because they do
    different things. Built-in presets only enable "normalize_volume" to avoid
    running loudnorm twice on the same audio; enable "normalize_audio" instead
    (or in addition) only if you specifically want to address it by that name,
    e.g. via `--stage normalize_audio` with different target_db/true_peak kwargs.
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

            import os

            # ffmpeg reading and -y-truncating the same path (when no explicit
            # output_path is given) races reading input against writing output on
            # the same file. Write to a distinct temp path and only replace
            # input_path with it after a successful encode.
            in_place = output_path is None
            dest = f"{input_path}.normalize_tmp.mp4" if in_place else output_path

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
                dest,
            ]

            def cb(p, m):
                self._report_progress(0.5 + p * 0.5, m, progress_callback)

            norm_result = run_ffmpeg(args, progress_callback=cb, timeout=600)

            if norm_result.returncode != 0:
                if in_place and os.path.exists(dest):
                    os.unlink(dest)
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Normalization failed: {norm_result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            if in_place:
                os.replace(dest, input_path)
                dest = input_path

            self._report_progress(1.0, "Audio normalization complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=dest,
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
    """Normalize audio volume to target level -- identical operation to
    NormalizeAudioStage above (same loudnorm filter), just addressed under a
    separate stage name/config section (stages.normalize_volume in config.py's
    DEFAULTS) for backward compatibility. This is the one enabled by default in
    built-in presets; see NormalizeAudioStage's docstring for why both exist.
    """

    name = "normalize_volume"
    display_name = "Volume Normalization"
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
        # Reuse NormalizeAudioStage's implementation, but resolve target_db/
        # true_peak from THIS stage's own config section (stages.normalize_volume)
        # first -- constructing a bare NormalizeAudioStage(self.config) here would
        # read stages.normalize_audio instead (a different, unrelated config
        # section that isn't even in DEFAULTS), silently ignoring whatever the
        # user configured under stages.normalize_volume.
        normalizer = NormalizeAudioStage(self.config)
        return normalizer.execute(
            input_path,
            output_path,
            progress_callback,
            target_db=target_db if target_db is not None else self._target_db,
            true_peak=true_peak if true_peak is not None else self._true_peak,
            **kwargs,
        )


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
