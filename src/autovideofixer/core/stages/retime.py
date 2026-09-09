"""Auto Video Fixer - True source cadence recovery stage (REQUIREMENTS.md § 12.2).

Runs immediately after ``detect`` (see ``pipeline.default_order``), so every
later stage sees the reduced, honest frame set and the compute saving
compounds across the whole pipeline. A single ffmpeg pass:

    ffmpeg -i IN -vf mpdecimate=hi=<hi>:lo=<lo>:frac=<frac> -fps_mode vfr \\
           -c:v <temp codec> -crf <temp_crf> -c:a copy -y OUT.mkv

``-fps_mode vfr`` is load-bearing: it tells ffmpeg to honour the surviving
frames' own timestamps instead of conforming the output back to a constant
rate -- this IS the feature (see AGENTS.md's timestamp-safety notes and
``core/cadence.py`` for the decode-only analysis pass this stage's decision
is based on).

SKIPs (never FAILS) when ``analyze_cadence()`` reports the input isn't
padded, when there's no video stream, or when the analysis/mpdecimate pass
itself errors -- a cadence miss must never take down an otherwise fine job.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from autovideofixer.core.cadence import analyze_cadence
from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

_FRAME_COUNT_RE = re.compile(r"frame=\s*(\d+)")


class RetimeStage(BaseStage):
    """Recover true source cadence and drop duplicate/padding frames.

    Traditional/FFmpeg only -- no AI alternative, same as CropStage/
    DownscaleStage (doesn't participate in the ai_fallback mechanism).
    """

    name = "retime"
    display_name = "Cadence Recovery"
    description = "Recover the true source framerate and drop padding/duplicate frames"
    category = "enhancement"
    # Runs immediately after detect (priority=1) -- pipeline.default_order is
    # authoritative for actual execution order, this is only a fallback
    # ordering signal for callers that don't consult it.
    priority = 2
    supports_gpu = False

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if self._explicitly_no_video(input_info):
            return False, "No video stream"
        return True, None

    @staticmethod
    def _explicitly_no_video(input_info: dict[str, Any] | None) -> bool:
        """True only when ``input_info`` positively asserts no video stream
        (``has_video: False``) -- absent/unknown defers to execute()'s own
        analysis rather than blocking the stage outright."""
        if input_info is None:
            return False
        return "has_video" in input_info and not input_info.get("has_video")

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        input_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Analyzing source cadence...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to retime stage",
                duration_sec=time.time() - start,
            )

        if self._explicitly_no_video(input_info):
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="No video stream",
                duration_sec=time.time() - start,
            )

        min_duplicate_ratio = self._stage_config.get("min_duplicate_ratio", 0.05)

        # analyze_cadence() never raises -- a cadence miss (ffmpeg missing,
        # malformed/truncated decode, unreadable input) degrades to a safe
        # is_padded=False result (logged at WARNING inside analyze_cadence),
        # which the branch below treats identically to a genuinely honest
        # input: SKIPPED, never FAILED.
        analysis = analyze_cadence(input_path, self.config)

        base_metadata = {
            "method": "mpdecimate",
            "encoded_fps": analysis.encoded_fps,
            "detected_fps": analysis.detected_fps,
            "nominal_fps": analysis.nominal_fps,
            "duplicate_ratio": analysis.duplicate_ratio,
            "is_regular": analysis.is_regular,
            "grid_rate": analysis.grid_rate,
        }

        if not analysis.is_padded:
            self._report_progress(1.0, "Cadence already honest", progress_callback)
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=(
                    f"Input cadence already honest (duplicate_ratio="
                    f"{analysis.duplicate_ratio:.3f} < min_duplicate_ratio="
                    f"{min_duplicate_ratio:.3f})"
                ),
                metadata=base_metadata,
                duration_sec=time.time() - start,
            )

        hi = self._stage_config.get("hi", 768)
        lo = self._stage_config.get("lo", 320)
        frac = self._stage_config.get("frac", 0.33)
        temp_crf = self._stage_config.get("temp_crf", 16)

        vf = f"mpdecimate=hi={hi}:lo={lo}:frac={frac}"
        args = [
            "-i",
            input_path,
            "-vf",
            vf,
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(temp_crf),
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

        def cb(p: float, m: str) -> None:
            self._report_progress(0.1 + p * 0.9, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

        if result.returncode != 0:
            # A cadence miss must never take down an otherwise fine job:
            # degrade to SKIPPED (with a warning) rather than FAILED.
            self.logger.warning(
                "retime: mpdecimate pass failed for %s (%s); skipping retime for this input",
                input_path,
                result.stderr[:300],
            )
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=f"retime pass failed, degrading to skip: {result.stderr[:200]}",
                metadata=base_metadata,
                duration_sec=time.time() - start,
            )

        frames_before = 0
        try:
            frames_before = probe(input_path, self.config).frame_count
        except Exception:
            pass
        frames_after = self._parse_final_frame_count(result.stderr) or analysis.unique_frames
        frames_removed = max(0, frames_before - frames_after) if frames_before else 0
        duplicate_ratio = (
            frames_removed / frames_before if frames_before > 0 else analysis.duplicate_ratio
        )

        self._report_progress(1.0, "Cadence recovery complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={
                "method": "mpdecimate",
                "encoded_fps": analysis.encoded_fps,
                "detected_fps": analysis.detected_fps,
                "nominal_fps": analysis.nominal_fps,
                "frames_before": frames_before,
                "frames_after": frames_after,
                "frames_removed": frames_removed,
                "duplicate_ratio": duplicate_ratio,
                "is_regular": analysis.is_regular,
                "grid_rate": analysis.grid_rate,
            },
            duration_sec=time.time() - start,
        )

    @staticmethod
    def _parse_final_frame_count(stderr: str) -> int | None:
        """Parse ffmpeg's own progress ``frame=<N>`` counter, taking the LAST
        occurrence (the final tally) -- used as the post-retime frame count
        when reporting stage metadata. ``None`` on no match."""
        matches = _FRAME_COUNT_RE.findall(stderr or "")
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None
