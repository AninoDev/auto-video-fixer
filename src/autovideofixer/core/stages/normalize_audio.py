"""Auto Video Fixer - Audio normalization and volume normalization."""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

# Default silence-skip threshold in LUFS. Inputs with no real audio track get a
# silent stereo track added earlier in the pipeline, so by the time normalization
# runs there IS an audio stream -- just silent, or near-silent if the source added
# dithering noise. Two-pass loudnorm measures measured_I = -inf on pure digital
# silence, and the second (apply) pass blows up feeding that back in as
# measured_I=-inf (infinite gain). -80.0 dBFS is chosen because it sits just above
# what 2-3 LSBs of 16-bit dither can produce: 20*log10(3/32768) ~= -80.8 dBFS, so
# genuine (if extremely quiet) 16-bit-sourced audio stays above the threshold while
# dither-only "silence" and true digital silence both fall at/below it.
DEFAULT_SILENCE_THRESHOLD_DB = -80.0


def _resolve_measured_i(raw: Any) -> float:
    """Robustly parse loudnorm's measured ``input_i`` into a float LUFS value.

    loudnorm's JSON pass emits ``"-inf"`` (as a *string*) for pure digital
    silence -- ``json.loads()`` decodes that into a Python ``str``, not a
    float, so a naive ``float(input_i)`` is needed for the normal numeric
    case, plus explicit handling for ``"-inf"``/``"inf"``/``"nan"``/anything
    else that isn't a finite number. All of those collapse to a single
    silence sentinel (``float("-inf")``) rather than raising -- this must
    never crash the stage, only classify the input as silent.
    """
    try:
        value = float(raw)
    except TypeError, ValueError:
        return float("-inf")
    if math.isnan(value) or math.isinf(value):
        return float("-inf")
    return value


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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._target_db = self._stage_config.get("target_db", -23.0)
        self._true_peak = self._stage_config.get("true_peak_db", -2.0)
        self._silence_threshold_db = self._stage_config.get(
            "silence_threshold_db", DEFAULT_SILENCE_THRESHOLD_DB
        )

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if not input_info.get("has_audio"):
            return False, "No audio stream"
        return True, None

    def _silence_skip_result(
        self,
        *,
        measured_i_raw: Any,
        threshold_db: float,
        stage_label: str,
        input_path: str,
        output_path: str | None,
        start: float,
    ) -> StageResult | None:
        """Build a pass-through result if the measured loudness indicates
        silence/near-silence, else return None (proceed with normalization).

        Shared by NormalizeAudioStage and NormalizeVolumeStage -- both run
        the identical loudnorm algorithm (see module docstring), so this is
        implemented once; each caller passes its OWN cascaded threshold and
        stage name/label, so a per-occurrence config override on either
        stage section is honored automatically without this method needing
        to know which stage section it came from.

        Mirrors the mid-execute skip convention used by UpscaleStage's
        "already at target resolution" gate and StabilizeStage's
        "no stabilization needed" path: returns StageStatus.COMPLETED (not
        FAILED) with output_path pointing at an unmodified copy of the
        input and skipped_reason set, so the pipeline treats this as
        success and downstream stages receive the input passed through
        unchanged.
        """
        measured_i = _resolve_measured_i(measured_i_raw)
        basename = os.path.basename(input_path)

        if math.isinf(measured_i):
            self.logger.info(
                f"{basename}: input audio is silent (measured I = -inf LUFS); "
                f"skipping {stage_label} -- nothing to normalize"
            )
            skipped_reason = "Silent input (measured I = -inf LUFS)"
        elif measured_i <= threshold_db:
            self.logger.info(
                f"{basename}: no significant audio (measured I = {measured_i:.1f} LUFS "
                f"<= threshold {threshold_db:.1f} dB); skipping {stage_label}"
            )
            skipped_reason = (
                f"Near-silent input (measured I = {measured_i:.1f} LUFS "
                f"<= threshold {threshold_db:.1f} dB)"
            )
        else:
            return None

        from autovideofixer.core.ffmpeg_utils import run_ffmpeg

        # Same in-place-vs-explicit-output-path handling as the normal apply
        # pass below: writing directly onto input_path while also reading it
        # races reads against writes on the same file. Written as `... if
        # output_path is None else output_path` (not the equivalent
        # `in_place`-first form) so mypy narrows the else-branch to `str`.
        in_place = output_path is None
        dest: str = f"{input_path}.normalize_tmp.mp4" if output_path is None else output_path

        copy_result = run_ffmpeg(
            ["-hide_banner", "-i", input_path, "-c", "copy", "-y", dest],
            timeout=self.stage_timeout(),
        )
        if copy_result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Silence-skip passthrough copy failed: {copy_result.stderr[:200]}",
                duration_sec=time.time() - start,
            )

        if in_place:
            os.replace(dest, input_path)
            dest = input_path

        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=dest,
            metadata={
                # REQUIREMENTS.md § 6.4: normalize_audio/normalize_volume are
                # traditional-only (FFmpeg loudnorm, no AI path) -- uniform
                # provenance.
                "method": "traditional",
                "skipped": True,
                "measured_i": measured_i,
                "threshold_db": threshold_db,
            },
            duration_sec=time.time() - start,
            skipped_reason=skipped_reason,
        )

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        target_db: float | None = None,
        true_peak: float | None = None,
        silence_threshold_db: float | None = None,
        stage_label: str | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Normalizing audio...", progress_callback)

        try:
            target = target_db if target_db is not None else self._target_db
            peak = true_peak if true_peak is not None else self._true_peak
            threshold = (
                silence_threshold_db
                if silence_threshold_db is not None
                else self._silence_threshold_db
            )
            label = stage_label if stage_label is not None else self.name

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

            # Silence/near-silence: skip normalization rather than feeding an
            # unusable (-inf or absurd) measured_I into the second loudnorm
            # pass, which produces infinite gain and blows up. See
            # _silence_skip_result()'s docstring for the convention matched.
            skip_result = self._silence_skip_result(
                measured_i_raw=measured_i,
                threshold_db=threshold,
                stage_label=label,
                input_path=input_path,
                output_path=output_path,
                start=start,
            )
            if skip_result is not None:
                self._report_progress(
                    1.0, "Skipping normalization (silent input)", progress_callback
                )
                return skip_result

            # Apply normalization
            self._report_progress(0.5, "Applying normalization...", progress_callback)

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

            norm_result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

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
                metadata={"method": "traditional", "target_db": target, "true_peak": peak},
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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._target_db = self._stage_config.get("target_db", -23.0)
        self._true_peak = self._stage_config.get("true_peak_db", -2.0)
        self._silence_threshold_db = self._stage_config.get(
            "silence_threshold_db", DEFAULT_SILENCE_THRESHOLD_DB
        )

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
        silence_threshold_db: float | None = None,
        **kwargs,
    ) -> StageResult:
        # Reuse NormalizeAudioStage's implementation, but resolve target_db/
        # true_peak/silence_threshold_db from THIS stage's own config section
        # (stages.normalize_volume) first -- constructing a bare
        # NormalizeAudioStage(self.config) here would read stages.normalize_audio
        # instead (a different, unrelated config section that isn't even in
        # DEFAULTS), silently ignoring whatever the user configured under
        # stages.normalize_volume. stage_label is passed explicitly too, so the
        # silence-skip log message names "normalize_volume" (the stage actually
        # running), not the delegate's own "normalize_audio" name.
        normalizer = NormalizeAudioStage(self.config)
        return normalizer.execute(
            input_path,
            output_path,
            progress_callback,
            target_db=target_db if target_db is not None else self._target_db,
            true_peak=true_peak if true_peak is not None else self._true_peak,
            silence_threshold_db=(
                silence_threshold_db
                if silence_threshold_db is not None
                else self._silence_threshold_db
            ),
            stage_label=self.name,
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
