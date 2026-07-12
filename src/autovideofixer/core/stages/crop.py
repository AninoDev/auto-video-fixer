"""Auto Video Fixer - Auto-crop stage.

Detects a video's true content bounds (stripping letterboxing/pillarboxing,
including any residual border stabilize can introduce) and crops to them.
Strictly opt-in -- see docs/REQUIREMENTS.md feature 3.

Detection is FFmpeg's ``cropdetect`` filter run with ``reset=0``, which never
resets its accumulated bounding box between frames: the crop window can only
grow (loosen) as the scan progresses, so the LAST ``crop=w:h:x:y`` line ffmpeg
reports is the union across the whole scanned range -- the furthest the real
content ever reaches toward each edge. This is deliberately NOT a per-frame
crop (which would flicker the window scene-to-scene as content moves); it's a
single crop window applied to the whole video/re-encode.

Optional VLM assist (``stages.crop.vlm_check``) helps distinguish a genuine
content edge from a watermark/logo that sits outside the true content area
(e.g. positioned relative to a letterboxed frame) and would otherwise "widen"
cropdetect's result. See ``_run_vlm_check`` and ``core.analysis.run_crop_vlm_check``.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

# Cheap pre-filter window for should_run(): a real full-scan (the authoritative
# check) happens in execute() against whatever file the pipeline actually hands
# this stage -- see should_run()'s docstring for why these two checks can
# legitimately disagree.
_QUICK_SAMPLE_SEC = 10.0


class CropStage(BaseStage):
    """Detect and remove black borders (letterbox/pillarbox) via FFmpeg cropdetect.

    Traditional only -- there is no AI alternative; this stage doesn't
    participate in the ai_fallback mechanism.
    """

    name = "crop"
    display_name = "Auto Crop"
    description = "Detect and crop black borders (letterbox/pillarbox)"
    category = "enhancement"
    # Between stabilize (10) and deblock (15): crop must run AFTER stabilize
    # (stabilization can itself add a black border via zoom-out correction,
    # and this stage should remove both the original letterboxing AND any
    # residual stabilization border in one pass) and BEFORE
    # deblock/denoise/upscale/interpolate/encode, so those more expensive
    # stages (especially AI ones) never spend compute on pixels that are
    # about to be cropped away. See Pipeline.optimize_stage_order().
    priority = 12
    supports_gpu = False

    def __init__(self, config):
        super().__init__(config)
        self._limit = self._stage_config.get("limit", 24)
        self._round = self._stage_config.get("round", 2)
        self._min_crop_px = self._stage_config.get("min_crop_px", 8)
        self._analyze_duration_sec = self._stage_config.get("analyze_duration_sec", 0)
        self._vlm_check = self._stage_config.get("vlm_check", False)
        self._vlm_policy = self._stage_config.get("vlm_policy", "warn")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        """Quick pre-filter: is this even worth attempting?

        NOT authoritative. ``input_info`` describes the job's ORIGINAL input
        file (see Pipeline.execute_job() -- it's probed once up front and not
        re-probed per stage outside of scene mode), not necessarily the actual
        file this stage's execute() will receive as ``input_path`` (e.g. after
        stabilize has already produced an intermediate). A short cropdetect
        sample here can therefore say "nothing to crop" on the pre-stabilize
        file even though stabilize goes on to add a border -- execute() always
        re-runs a full, authoritative cropdetect pass against the real file it
        gets and can independently return SKIPPED, so a false "proceed" here
        is harmless. A false "skip" here (this quick sample missing a border
        that only appears later in the video, or only after stabilize) is the
        real risk this pre-filter accepts in exchange for staying cheap --
        same class of limitation as every other stage's should_run() in this
        codebase, which also only sees this one static input_info snapshot.
        """
        if not self.is_enabled():
            return False, "Stage disabled"

        filepath = input_info.get("filepath") or ""
        width, height = input_info.get("resolution", (0, 0))
        if not filepath or not os.path.exists(filepath) or width <= 0 or height <= 0:
            # Can't do a useful quick check -- let execute() make the real call.
            return True, None

        sample_secs = _QUICK_SAMPLE_SEC
        if self._analyze_duration_sec and self._analyze_duration_sec > 0:
            sample_secs = min(sample_secs, self._analyze_duration_sec)

        detected = _detect_crop(filepath, self._limit, self._round, sample_secs)
        if detected is None:
            return True, None  # inconclusive -- let execute() decide properly

        crop_w, crop_h, _, _ = detected
        savings_w = width - crop_w
        savings_h = height - crop_h
        if savings_w < self._min_crop_px and savings_h < self._min_crop_px:
            return False, (
                f"Quick cropdetect sample ({sample_secs:.0f}s) found no meaningful "
                f"border to crop (savings {savings_w}x{savings_h}px < "
                f"min_crop_px={self._min_crop_px})"
            )
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Analyzing content bounds...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to crop stage",
                duration_sec=time.time() - start,
            )

        try:
            probe_info = probe(input_path)
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"probe failed: {e}",
                duration_sec=time.time() - start,
            )

        orig_w, orig_h = probe_info.resolution
        if orig_w <= 0 or orig_h <= 0:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="Could not determine input resolution",
                duration_sec=time.time() - start,
            )

        self._report_progress(0.1, "Running cropdetect (whole-scan union)...", progress_callback)
        detected = _detect_crop(input_path, self._limit, self._round, self._analyze_duration_sec)
        if detected is None:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="cropdetect produced no result (unreadable video or filter failure)",
                duration_sec=time.time() - start,
            )

        crop_w, crop_h, crop_x, crop_y = detected

        if crop_w <= 0 or crop_h <= 0 or crop_w > orig_w or crop_h > orig_h:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=(
                    f"cropdetect produced an invalid crop window ({crop_w}x{crop_h}) "
                    f"for a {orig_w}x{orig_h} input"
                ),
                duration_sec=time.time() - start,
            )

        savings_w = orig_w - crop_w
        savings_h = orig_h - crop_h
        if savings_w < self._min_crop_px and savings_h < self._min_crop_px:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=(
                    f"Detected crop {crop_w}:{crop_h}:{crop_x}:{crop_y} saves only "
                    f"{savings_w}x{savings_h}px, below min_crop_px={self._min_crop_px} -- "
                    "no meaningful border to remove"
                ),
                metadata={
                    "detected_crop": f"{crop_w}:{crop_h}:{crop_x}:{crop_y}",
                    "original_resolution": f"{orig_w}x{orig_h}",
                },
                duration_sec=time.time() - start,
            )

        vlm_meta: dict[str, Any] = {}
        if self._vlm_check:
            self._report_progress(0.3, "Running VLM content/watermark check...", progress_callback)
            vlm_meta = self._run_vlm_check(input_path, probe_info, crop_w, crop_h, crop_x, crop_y)
            if vlm_meta.get("content_outside"):
                reason = vlm_meta.get("reason", "")
                self.logger.warning(
                    "crop: VLM flagged meaningful content outside the proposed crop box "
                    "(%dx%d+%d+%d on %dx%d input): %s",
                    crop_w,
                    crop_h,
                    crop_x,
                    crop_y,
                    orig_w,
                    orig_h,
                    reason,
                )
                if self._vlm_policy == "skip":
                    return StageResult(
                        status=StageStatus.SKIPPED,
                        skipped_reason=(
                            "VLM flagged meaningful content (watermark/logo/text) outside "
                            f"the detected crop box; crop.vlm_policy=skip. Reason: {reason}"
                        ),
                        metadata={
                            "vlm_check": vlm_meta,
                            "detected_crop": f"{crop_w}:{crop_h}:{crop_x}:{crop_y}",
                        },
                        duration_sec=time.time() - start,
                    )
                # policy == "warn" (default): proceed with the plain cropdetect crop.

        self._report_progress(
            0.5, f"Cropping to {crop_w}x{crop_h}+{crop_x}+{crop_y}...", progress_callback
        )

        try:
            has_audio = probe_info.has_audio
        except Exception:
            has_audio = False

        args = [
            "-i",
            input_path,
            "-vf",
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
        ]
        args += ["-c:a", "copy"] if has_audio else []
        args += ["-y", output_path]

        def cb(p, m):
            self._report_progress(0.5 + p * 0.5, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=3600)
        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Crop encode failed: {result.stderr[:2000]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Auto-crop complete", progress_callback)
        metadata = {
            "detected_crop": f"{crop_w}:{crop_h}:{crop_x}:{crop_y}",
            "original_resolution": f"{orig_w}x{orig_h}",
            "cropped_resolution": f"{crop_w}x{crop_h}",
            "savings_px": f"{savings_w}x{savings_h}",
        }
        if vlm_meta:
            metadata["vlm_check"] = vlm_meta
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata=metadata,
            duration_sec=time.time() - start,
        )

    def _run_vlm_check(
        self,
        input_path: str,
        probe_info,
        crop_w: int,
        crop_h: int,
        crop_x: int,
        crop_y: int,
    ) -> dict[str, Any]:
        """Extract one frame two ways (plain, and with the proposed crop box
        drawn via ``drawbox``) and ask the VLM whether meaningful content
        (watermark/logo/text belonging to the video) lies outside the box.

        Fails open on any error: returns ``content_outside: False`` and logs a
        WARNING, so a VLM/network problem never blocks the plain cropdetect
        result from being applied.
        """
        from autovideofixer.core.analysis import run_crop_vlm_check

        vlm_config = self.config.get("analysis", "vlm", default={})
        if not vlm_config.get("enabled", False):
            return {
                "checked": False,
                "content_outside": False,
                "reason": "analysis.vlm.enabled is False",
                "failed": False,
            }

        duration = probe_info.duration or 0.0
        sample_t = max(0.0, duration / 2.0)

        tmp_dir = tempfile.mkdtemp(prefix="avf_crop_vlm_")
        full_frame = os.path.join(tmp_dir, "full.jpg")
        boxed_frame = os.path.join(tmp_dir, "boxed.jpg")
        try:
            full_args = [
                "-ss",
                f"{sample_t:.3f}",
                "-i",
                input_path,
                "-frames:v",
                "1",
                "-y",
                full_frame,
            ]
            r1 = run_ffmpeg(full_args, timeout=60)

            box_filter = (
                f"drawbox=x={crop_x}:y={crop_y}:w={crop_w}:h={crop_h}:color=red@0.8:thickness=6"
            )
            box_args = [
                "-ss",
                f"{sample_t:.3f}",
                "-i",
                input_path,
                "-vf",
                box_filter,
                "-frames:v",
                "1",
                "-y",
                boxed_frame,
            ]
            r2 = run_ffmpeg(box_args, timeout=60)

            if (
                r1.returncode != 0
                or r2.returncode != 0
                or not os.path.exists(full_frame)
                or not os.path.exists(boxed_frame)
            ):
                self.logger.warning(
                    "crop: failed to extract VLM-check frames; proceeding with plain "
                    "cropdetect result"
                )
                return {
                    "checked": False,
                    "content_outside": False,
                    "reason": "frame extraction failed",
                    "failed": True,
                }

            return run_crop_vlm_check(full_frame, boxed_frame, self.config)
        except Exception as e:
            self.logger.warning(
                "crop: VLM check raised an exception (%s); proceeding with plain cropdetect result",
                e,
            )
            return {"checked": False, "content_outside": False, "reason": str(e), "failed": True}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_crop(
    input_path: str,
    limit: int,
    round_: int,
    analyze_duration_sec: float,
) -> tuple[int, int, int, int] | None:
    """Run FFmpeg cropdetect over (a sample of) the video and return the LAST
    reported ``crop=w:h:x:y`` window.

    ``reset=0`` makes cropdetect accumulate the loosest safe crop across the
    whole scanned range instead of resetting per-frame/per-GOP -- the last
    ``crop=`` line in ffmpeg's stderr is therefore the union of the entire
    scan: the furthest the real content ever reaches toward each edge. This is
    exactly the whole-video (not per-frame) bound this stage wants, not a
    flickering per-scene crop.

    Returns None if cropdetect produced no output at all (e.g. an unreadable
    file), never on a "no border found" result -- that case still yields a
    crop window equal to (or very close to) the full frame, which the caller
    compares against ``min_crop_px``.
    """
    args = ["-i", input_path]
    if analyze_duration_sec and analyze_duration_sec > 0:
        args += ["-t", str(analyze_duration_sec)]
    args += ["-vf", f"cropdetect=limit={limit}:round={round_}:reset=0", "-f", "null", "-"]

    result = run_ffmpeg(args, timeout=1800)
    matches = _CROP_RE.findall(result.stderr or "")
    if not matches:
        return None
    w, h, x, y = matches[-1]
    return int(w), int(h), int(x), int(y)
