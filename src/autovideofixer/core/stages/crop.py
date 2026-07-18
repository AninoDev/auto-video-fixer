"""Auto Video Fixer - Auto-crop stage.

Detects a video's true content bounds (stripping letterboxing/pillarboxing,
including any residual border stabilize can introduce) and crops to them.
Strictly opt-in -- see docs/REQUIREMENTS.md feature 3.

Detection (``execute()``'s authoritative pass, via ``_detect_crop_full()``) runs
FFmpeg's ``cropdetect`` filter with ``reset=1`` (recompute per analyzed frame,
subject to cropdetect's own default ``skip=2`` frame-skip) and
``max_outliers=<N>`` (``stages.crop.max_outlier_ratio``-derived -- lets a
bright logo/overlay sitting in the letterbox area still count as border,
instead of widening the crop to "protect" it). Every per-frame
``crop=w:h:x:y`` / ``t:<seconds>`` line is parsed and fed to
``aggregate_crop_windows()``, which groups frames into runs of matching
windows, excludes isolated short-lived runs as transitions (e.g. a single
bright full-frame flash), and returns the union of the surviving runs'
windows. This replaces an earlier ``reset=0`` "union that can only grow"
design: a single full-frame transition anywhere in the scan used to
permanently widen the crop window to the full frame for the rest of the scan.
``should_run()``'s cheap ~10s pre-filter sample (``_detect_crop()``) keeps the
old single-pass ``reset=0`` behavior -- it only decides whether cropping is
worth attempting at all, so it doesn't need max_outliers/aggregation
precision.

Optional VLM assist (``stages.crop.vlm_check``) is a secondary safety check,
not the primary overlay defense (that's ``max_outlier_ratio`` above): it can
still flag content outside the detected box for review. See
``_run_vlm_check`` and ``core.analysis.run_crop_vlm_check``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

logger = logging.getLogger(__name__)

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
# Per-frame reset=1 cropdetect lines look like:
#   [Parsed_cropdetect_0 @ 0x1] x1:0 x2:639 y1:60 y2:419 w:640 h:360 x:0 y:60
#   pts:25 t:1.00 crop=640:360:0:60
# -- the t:<seconds> and crop=w:h:x:y fields we need are on the same line, in
# that order, separated by the pts field.
_CROP_FRAME_RE = re.compile(r"t:(?P<t>[\d.]+).*?crop=(?P<w>\d+):(?P<h>\d+):(?P<x>\d+):(?P<y>\d+)")

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

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._limit = self._stage_config.get("limit", 24)
        self._round = self._stage_config.get("round", 2)
        self._min_crop_px = self._stage_config.get("min_crop_px", 8)
        self._analyze_duration_sec = self._stage_config.get("analyze_duration_sec", 0)
        self._max_outlier_ratio = self._stage_config.get("max_outlier_ratio", 0.2)
        self._transition_max_run_sec = self._stage_config.get("transition_max_run_sec", 2.0)
        self._transition_window_sec = self._stage_config.get("transition_window_sec", 4.0)
        self._transition_tolerance_px = self._stage_config.get("transition_tolerance_px", 16)
        self._vlm_check = self._stage_config.get("vlm_check", False)
        self._vlm_policy = self._stage_config.get("vlm_policy", "warn")

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        """Quick pre-filter: is this even worth attempting?

        NOT authoritative. ``Pipeline.execute_job()`` re-probes ``input_info``
        (including ``filepath``) after every stage that produces a new output
        file, so by the time this runs, ``input_info`` describes the actual
        file this stage's execute() will receive as ``input_path`` (e.g. the
        stabilize-produced intermediate, border and all) -- not the job's
        original input. The remaining gap is entirely temporal, not
        positional: this quick cropdetect sample only looks at a short window
        near the start of the (now-current) file, so it can still say
        "nothing to crop" while a border/letterbox only becomes visible later
        in the video. execute() always re-runs a full, authoritative
        cropdetect pass against the real file it gets and can independently
        return SKIPPED, so a false "proceed" here is harmless. A false "skip"
        here (this quick sample missing something that only shows up later in
        the timeline) is the real risk this pre-filter accepts in exchange
        for staying cheap -- same class of limitation as every other stage's
        should_run() in this codebase, which also only sees a point-in-time
        input_info snapshot (freshened once per stage transition, not
        continuously).
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

        # Bounded helper call (sample_secs is capped at _QUICK_SAMPLE_SEC == 10s
        # of content) -- a genuinely short scan, so this keeps a small fixed
        # timeout rather than the resolved stage/global timeout that
        # execute()'s real full-scan pass below uses. Deliberately still the
        # OLD single-pass reset=0 union (no max_outliers, no per-frame
        # aggregation): this is only a cheap "worth attempting?" prefilter, not
        # the authoritative crop window, so aggregation/transition precision
        # doesn't matter here -- execute() always re-derives the real window.
        detected = _detect_crop(filepath, self._limit, self._round, sample_secs, timeout=60)
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

        self._report_progress(
            0.1,
            "Running cropdetect (per-frame + transition-resilient aggregation)...",
            progress_callback,
        )
        # Whole-video (or configured analyze_duration_sec) scan -- uses the
        # resolved stage/global timeout, not a small fixed one, since
        # analyze_duration_sec=0 means "scan the entire input".
        detected = _detect_crop_full(
            input_path,
            self._limit,
            self._round,
            self._analyze_duration_sec,
            max_outlier_ratio=self._max_outlier_ratio,
            transition_max_run_sec=self._transition_max_run_sec,
            transition_window_sec=self._transition_window_sec,
            transition_tolerance_px=self._transition_tolerance_px,
            orig_width=orig_w,
            orig_height=orig_h,
            timeout=self.stage_timeout(),
            logger_=self.logger,
        )
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

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())
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
    timeout: float | None = 1800,
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

    ``timeout``: this is a free function (no ``self``/stage config access),
    so the caller resolves the effective timeout and passes it through --
    ``CropStage.should_run()``'s bounded quick-sample call passes a small
    fixed value, ``CropStage.execute()``'s real whole-scan pass passes
    ``self.stage_timeout()``. Defaults to the historical fixed 1800 for any
    other/test caller that doesn't pass one explicitly.
    """
    args = ["-i", input_path]
    if analyze_duration_sec and analyze_duration_sec > 0:
        args += ["-t", str(analyze_duration_sec)]
    args += ["-vf", f"cropdetect=limit={limit}:round={round_}:reset=0", "-f", "null", "-"]

    result = run_ffmpeg(args, timeout=timeout)
    matches = _CROP_RE.findall(result.stderr or "")
    if not matches:
        return None
    w, h, x, y = matches[-1]
    return int(w), int(h), int(x), int(y)


def _compute_max_outliers(max_outlier_ratio: float, width: int, height: int) -> int:
    """Convert ``stages.crop.max_outlier_ratio`` into cropdetect's
    ``max_outliers`` pixel count for a given probed resolution.

    ``max_outliers`` is a single absolute pixel count applied by cropdetect to
    BOTH the row scan (bounded by height) and column scan (bounded by width),
    so basing it on ``min(width, height)`` keeps the tolerance conservative on
    whichever axis is smaller, rather than letting a ratio computed off the
    larger axis blow past a reasonable fraction of the smaller one.
    """
    ratio = max(0.0, min(0.5, max_outlier_ratio))
    if ratio <= 0:
        return 0
    return round(ratio * min(width, height))


def _cropdetect_filter(limit: int, round_: int, reset: int, max_outliers: int = 0) -> str:
    filt = f"cropdetect=limit={limit}:round={round_}:reset={reset}"
    if max_outliers > 0:
        filt += f":max_outliers={max_outliers}"
    return filt


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` UP to the nearest multiple of ``multiple`` -- never
    down, so a rounded crop window is never smaller (never cuts real content)
    than the aggregated union it was derived from. ``multiple <= 1`` is a
    no-op (nothing to align to)."""
    if multiple <= 1:
        return value
    remainder = value % multiple
    if remainder == 0:
        return value
    return value + (multiple - remainder)


@dataclass(frozen=True)
class CropFrame:
    """One per-frame cropdetect observation: timestamp + crop window."""

    t: float
    w: int
    h: int
    x: int
    y: int


def _parse_crop_frames(stderr: str) -> list[CropFrame]:
    """Parse every per-frame ``t:<seconds> ... crop=w:h:x:y`` cropdetect line
    from a ``reset=1`` run's captured stderr.

    NOTE on memory: ``reset=1`` makes cropdetect emit one line per analyzed
    frame (still subject to cropdetect's own default ``skip=2``). For very
    long inputs, ``run_ffmpeg``'s captured stderr -- and this parsed list --
    can reach tens of MB. Acceptable for now (every other stage already
    buffers full stderr), but worth knowing if a multi-hour input ever shows
    up as a memory complaint.
    """
    frames: list[CropFrame] = []
    for line in (stderr or "").splitlines():
        m = _CROP_FRAME_RE.search(line)
        if not m:
            continue
        frames.append(
            CropFrame(
                t=float(m.group("t")),
                w=int(m.group("w")),
                h=int(m.group("h")),
                x=int(m.group("x")),
                y=int(m.group("y")),
            )
        )
    return frames


def _windows_similar_edges(a: CropFrame, b: CropFrame, tolerance_px: int) -> bool:
    """Two windows are "the same" if every edge (left, top, right, bottom)
    differs by at most ``tolerance_px``."""
    return (
        abs(a.x - b.x) <= tolerance_px
        and abs(a.y - b.y) <= tolerance_px
        and abs((a.x + a.w) - (b.x + b.w)) <= tolerance_px
        and abs((a.y + a.h) - (b.y + b.h)) <= tolerance_px
    )


def _union(frames: list[CropFrame]) -> tuple[int, int, int, int]:
    """Union (max extent toward every edge) of a set of crop windows."""
    min_x = min(f.x for f in frames)
    min_y = min(f.y for f in frames)
    max_right = max(f.x + f.w for f in frames)
    max_bottom = max(f.y + f.h for f in frames)
    return max_right - min_x, max_bottom - min_y, min_x, min_y


@dataclass(frozen=True)
class _Run:
    """A maximal sequence of consecutive (by timestamp) frames whose windows
    all match the run's first ("anchor") frame's window within tolerance."""

    frames: list[CropFrame] = field(default_factory=list)

    @property
    def start_t(self) -> float:
        return self.frames[0].t

    @property
    def end_t(self) -> float:
        return self.frames[-1].t

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def anchor(self) -> CropFrame:
        return self.frames[0]


def _run_gap_sec(a: _Run, b: _Run) -> float:
    """Time gap between two runs' timestamp ranges; 0 if they overlap."""
    if a.end_t < b.start_t:
        return b.start_t - a.end_t
    if b.end_t < a.start_t:
        return a.start_t - b.end_t
    return 0.0


@dataclass(frozen=True)
class _AggregationResult:
    runs: list[_Run]
    excluded_runs: list[_Run]
    window: tuple[int, int, int, int] | None
    fallback_triggered: bool = False


def _aggregate_runs(
    frames: list[CropFrame],
    *,
    tolerance_px: int,
    transition_max_run_sec: float,
    transition_window_sec: float,
) -> _AggregationResult:
    if not frames:
        return _AggregationResult(runs=[], excluded_runs=[], window=None)

    # Sort defensively by timestamp -- ffmpeg emits per-frame lines in decode
    # order, which should already be monotonic in t, but don't assume it.
    ordered = sorted(frames, key=lambda f: f.t)

    runs: list[_Run] = []
    current: list[CropFrame] = [ordered[0]]
    for frame in ordered[1:]:
        if _windows_similar_edges(current[0], frame, tolerance_px):
            current.append(frame)
        else:
            runs.append(_Run(frames=current))
            current = [frame]
    runs.append(_Run(frames=current))

    def _is_transition(idx: int) -> bool:
        run = runs[idx]
        # Rule: a run is a transition iff it's short-lived AND no OTHER run
        # nearby (in time) has a similar window. The same window recurring
        # nearby (or a run outlasting transition_max_run_sec even in
        # isolation) means it's real content geometry, not a transient flash.
        if run.duration > transition_max_run_sec:
            return False
        for j, other in enumerate(runs):
            if j == idx:
                continue
            if _run_gap_sec(run, other) <= transition_window_sec and _windows_similar_edges(
                run.anchor, other.anchor, tolerance_px
            ):
                return False
        return True

    flags = [_is_transition(i) for i in range(len(runs))]
    surviving = [r for r, is_t in zip(runs, flags) if not is_t]
    excluded = [r for r, is_t in zip(runs, flags) if is_t]
    fallback_triggered = False

    if not surviving:
        # Pathological: every run got classified as a transition (e.g. a
        # single video with nothing but brief, non-recurring windows). Fall
        # back to the union of everything rather than returning nothing --
        # per spec, moving/drifting content should never get cropped off.
        logger.debug(
            "aggregate_crop_windows: all %d run(s) classified as transitions; "
            "falling back to union of all frames",
            len(runs),
        )
        surviving = runs
        excluded = []
        fallback_triggered = True

    window = _union([f for r in surviving for f in r.frames])
    return _AggregationResult(
        runs=runs, excluded_runs=excluded, window=window, fallback_triggered=fallback_triggered
    )


def aggregate_crop_windows(
    frames: list[CropFrame],
    *,
    tolerance_px: int,
    transition_max_run_sec: float,
    transition_window_sec: float,
) -> tuple[int, int, int, int] | None:
    """Aggregate per-frame ``reset=1`` cropdetect windows into a single
    transition-resilient crop window. Pure function, no I/O (other than an
    internal DEBUG log for the pathological all-transition fallback) --
    designed for heavy unit testing.

    Algorithm:
      1. Group consecutive (by timestamp) frames into runs whose windows all
         match the run's first window within ``tolerance_px`` (per edge
         coordinate: left/top/right/bottom).
      2. A run is a TRANSITION (excluded from the result) iff its duration
         is <= ``transition_max_run_sec`` AND no other run within
         ``transition_window_sec`` before or after it (by timestamp) has a
         similar (``tolerance_px``) window. An isolated deviant burst is a
         transition; the same window recurring nearby means it's real
         content geometry and must be kept.
      3. The result is the UNION (max extent toward every edge) of all
         windows in the surviving runs, unrounded -- callers that need
         encoder-friendly (e.g. even) dimensions should round the result UP
         (never down) to their target multiple.
      4. Edge cases: empty input -> None. All runs classified as transitions
         (pathological) -> fall back to the union of every run (logged at
         DEBUG). A single run -> its own window.

    Returns None for empty ``frames``.
    """
    return _aggregate_runs(
        frames,
        tolerance_px=tolerance_px,
        transition_max_run_sec=transition_max_run_sec,
        transition_window_sec=transition_window_sec,
    ).window


def _detect_crop_full(
    input_path: str,
    limit: int,
    round_: int,
    analyze_duration_sec: float,
    *,
    max_outlier_ratio: float,
    transition_max_run_sec: float,
    transition_window_sec: float,
    transition_tolerance_px: int,
    orig_width: int,
    orig_height: int,
    timeout: float | None = 1800,
    logger_: logging.Logger | None = None,
) -> tuple[int, int, int, int] | None:
    """Authoritative crop detection for ``CropStage.execute()``: FFmpeg
    ``cropdetect`` with ``reset=1`` (one crop window per analyzed frame,
    still subject to cropdetect's default ``skip=2``) and
    ``max_outliers=<N>`` derived from ``max_outlier_ratio`` (see
    ``_compute_max_outliers``), aggregated via ``aggregate_crop_windows``
    into a single transition-resilient crop window.

    Returns None if cropdetect produced no per-frame output at all (e.g. an
    unreadable file) -- mirrors ``_detect_crop``'s contract.
    """
    max_outliers = _compute_max_outliers(max_outlier_ratio, orig_width, orig_height)

    args = ["-i", input_path]
    if analyze_duration_sec and analyze_duration_sec > 0:
        args += ["-t", str(analyze_duration_sec)]
    args += [
        "-vf",
        _cropdetect_filter(limit, round_, reset=1, max_outliers=max_outliers),
        "-f",
        "null",
        "-",
    ]

    # See _parse_crop_frames()'s docstring for the stderr-size memory note --
    # reset=1 prints one crop= line per analyzed frame.
    result = run_ffmpeg(args, timeout=timeout)
    frames = _parse_crop_frames(result.stderr or "")
    if not frames:
        return None

    agg = _aggregate_runs(
        frames,
        tolerance_px=transition_tolerance_px,
        transition_max_run_sec=transition_max_run_sec,
        transition_window_sec=transition_window_sec,
    )
    if agg.window is None:
        return None

    w, h, x, y = agg.window
    # Each per-frame window cropdetect reports is already rounded to `round_`,
    # but the union combines left/top from one frame with right/bottom from
    # another, so the resulting w/h is not guaranteed to still be a multiple
    # of round_. Round UP (never down) so the final window never shrinks
    # below the aggregated union (never cuts real content).
    w = _round_up_to_multiple(w, round_)
    h = _round_up_to_multiple(h, round_)

    log = logger_ or logger
    excluded_ranges = ", ".join(f"{r.start_t:.2f}-{r.end_t:.2f}s" for r in agg.excluded_runs[:5])
    more = f" (+{len(agg.excluded_runs) - 5} more)" if len(agg.excluded_runs) > 5 else ""
    log.info(
        "crop: analyzed %d frame(s) into %d run(s); excluded %d run(s) as transitions%s%s; "
        "final window %d:%d:%d:%d",
        len(frames),
        len(agg.runs),
        len(agg.excluded_runs),
        f" ({excluded_ranges}{more})" if excluded_ranges else "",
        " [pathological all-transition fallback]" if agg.fallback_triggered else "",
        w,
        h,
        x,
        y,
    )

    return w, h, x, y
