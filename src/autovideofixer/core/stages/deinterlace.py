"""Auto Video Fixer - Deinterlacing stage (REQUIREMENTS.md § 20).

Runs right after ``retime``, before ``crop`` (see ``pipeline.default_order``):
every downstream stage (crop bounds, scene detection, RIFE interpolation)
benefits from seeing progressive frames, and comb artefacts actively corrupt
motion-based analysis.

**Off by default** (``stages.deinterlace.enabled: false``). Detection (via
ffmpeg's ``idet`` filter) is a heuristic that false-positives on detailed
progressive motion -- see ``_decide_deinterlace()`` and
``Config.DEFAULTS["stages"]["deinterlace"]`` for the measured numbers that
motivate this. Deinterlacing progressive content is destructive (it halves
either resolution or frame rate for no reason), so this stage must never
auto-enable itself off a probe -- the user opts in explicitly.

Three output-rate modes (``stages.deinterlace.mode``):

- ``"frame"`` -- one progressive frame per input FRAME (half the field
  rate). ``<filter>=mode=0``.
- ``"field"`` (default once enabled) -- one progressive frame per FIELD,
  full motion (doubles the frame count). ``<filter>=mode=1``. This is why
  the stage reports ``fps_out`` in its metadata (REQUIREMENTS.md § 12.5) --
  the pipeline propagates that into ``input_info["true_framerate"]``, and
  without it ``encode``'s CFR path would pin the pre-deinterlace rate.
- ``"bob"`` -- each field line-doubled to full height at the full field
  rate (also doubles frame count, same ``fps_out`` reporting as "field").
  None of the four supported filters (bwdif/yadif/w3fdif/estdif) expose a
  literal line-doubling mode -- verified against ffmpeg n9.0.1's own
  ``-h filter=<name>`` for all four, each only supports mode=0 (frame) and
  mode=1 (field) -- so "bob" is built from a separate, filter-independent
  chain (``setfield`` + ``separatefields`` + ``scale``) instead;
  ``stages.deinterlace.filter`` is ignored when ``mode`` is ``"bob"``.

Field order (``stages.deinterlace.field_order``): ``"auto"`` (default, from
``idet`` -- see ``_decide_deinterlace()``), ``"tff"``, or ``"bff"``. An
explicit ``tff``/``bff`` skips ``idet`` detection entirely and always
proceeds with that order -- getting this backwards produces juddering worse
than leaving the content interlaced, so it must be overridable outright.

Detection failure (ffmpeg missing, a malformed/truncated run, a timeout)
fails open: logged at WARNING, stage SKIPS -- never FAILS a job over a
detection problem, matching ``analyze_cadence()``'s convention
(``core/cadence.py``).

Does NOT implement REQUIREMENTS.md § 20.3 (recovering interlaced content
from an already-progressive/line-doubled conversion) -- deliberately
deferred, blocked on real affected samples.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

from autovideofixer.config import Config
from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path, run_ffmpeg, timing_output_args
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus
from autovideofixer.logger import get_logger

logger = get_logger("autovideofixer.stages.deinterlace")

_IDET_MULTI_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)

_SUPPORTED_MODES = ("frame", "field", "bob")
_SUPPORTED_FILTERS = ("bwdif", "yadif", "w3fdif", "estdif")


@dataclass(frozen=True)
class IdetCounts:
    """Parsed ``idet`` "Multi frame detection" tally.

    The multi-frame (cumulative) line is used rather than the single-frame
    line -- it's idet's own more reliable, noise-tolerant classification,
    accumulated across neighboring frames rather than judged per-frame.
    """

    tff: int
    bff: int
    progressive: int
    undetermined: int


@dataclass(frozen=True)
class DeinterlaceDecision:
    """Result of applying ``min_interlaced_ratio`` to a set of ``IdetCounts``.

    ``ratio`` and ``classified`` are ``None``/``0`` when there was nothing to
    classify (parse failure, or every frame was Undetermined) -- ``reason``
    always explains why, in a form suitable for a SKIPPED ``skipped_reason``.
    """

    interlaced: bool
    field_order: str | None
    ratio: float | None
    classified: int
    reason: str


def _parse_idet_output(text: str | None) -> IdetCounts | None:
    """Parse the LAST "Multi frame detection" line out of ``idet``'s stderr.

    Pure and unit-testable without ffmpeg. Returns ``None`` on malformed or
    empty input -- never raises.
    """
    if not text:
        return None
    matches = list(_IDET_MULTI_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    try:
        return IdetCounts(
            tff=int(m.group(1)),
            bff=int(m.group(2)),
            progressive=int(m.group(3)),
            undetermined=int(m.group(4)),
        )
    except ValueError:
        return None


def _decide_deinterlace(
    counts: IdetCounts | None, min_interlaced_ratio: float
) -> DeinterlaceDecision:
    """Decide whether to deinterlace, and with which field order, from idet counts.

    Pure and unit-testable without ffmpeg (REQUIREMENTS.md § 20.2):

    - Undetermined frames are excluded from the denominator -- a static
      progressive clip reports ``Undetermined: 50`` (nothing classified),
      which must SKIP, not be silently treated as 0% interlaced out of 50.
    - The interlaced fraction (TFF+BFF) of CLASSIFIED frames (TFF+BFF+
      Progressive) must strictly EXCEED ``min_interlaced_ratio`` --
      measured on the reference ffmpeg build (n9.0.1): genuinely interlaced
      content reports ``TFF: 50, Progressive: 0`` (ratio 1.0, clears any
      sane threshold); a progressive clip with detailed motion can still
      report a nonzero TFF count (idet's known false-positive mode) but
      well under a 0.7 threshold in practice; a static progressive clip
      reports 100% Undetermined (0 classified -- SKIP, no ratio to compute).
    - Malformed/empty idet output (``counts is None``) SKIPs identically to
      an all-Undetermined result -- never raises.
    """
    if counts is None:
        return DeinterlaceDecision(
            interlaced=False,
            field_order=None,
            ratio=None,
            classified=0,
            reason=(
                "idet output could not be parsed (malformed or empty); "
                "skipping deinterlace detection"
            ),
        )

    classified = counts.tff + counts.bff + counts.progressive
    if classified == 0:
        return DeinterlaceDecision(
            interlaced=False,
            field_order=None,
            ratio=None,
            classified=0,
            reason=(
                f"idet classified 0 frames (Undetermined: {counts.undetermined}); "
                "skipping deinterlace"
            ),
        )

    interlaced_count = counts.tff + counts.bff
    ratio = interlaced_count / classified

    if ratio > min_interlaced_ratio:
        field_order = "tff" if counts.tff >= counts.bff else "bff"
        return DeinterlaceDecision(
            interlaced=True,
            field_order=field_order,
            ratio=ratio,
            classified=classified,
            reason=(
                f"idet: {interlaced_count}/{classified} classified frames interlaced "
                f"(ratio={ratio:.3f} > min_interlaced_ratio={min_interlaced_ratio:.3f}); "
                f"field_order={field_order} (TFF={counts.tff}, BFF={counts.bff}, "
                f"Progressive={counts.progressive}, Undetermined={counts.undetermined})"
            ),
        )

    return DeinterlaceDecision(
        interlaced=False,
        field_order=None,
        ratio=ratio,
        classified=classified,
        reason=(
            f"idet: {interlaced_count}/{classified} classified frames interlaced "
            f"(ratio={ratio:.3f} <= min_interlaced_ratio={min_interlaced_ratio:.3f}); "
            f"TFF={counts.tff}, BFF={counts.bff}, Progressive={counts.progressive}, "
            f"Undetermined={counts.undetermined})"
        ),
    )


def _build_deinterlace_filter(mode: str, filter_name: str, field_order: str) -> str:
    """Build the ffmpeg ``-vf`` filter string for ``mode``/``filter_name``/``field_order``.

    Pure and unit-testable without ffmpeg. Raises ``ValueError`` (caught by
    ``execute()`` and turned into a FAILED StageResult) for an unrecognized
    ``mode``/``filter_name`` -- there is no known combination of the four
    supported filters and the "frame"/"field" modes that's unsupported
    (verified against ffmpeg n9.0.1: bwdif/yadif/w3fdif/estdif all accept
    the same mode=0/1 option identically), so the only rejection path in
    practice is an invalid enum value.
    """
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported stages.deinterlace.mode: {mode!r} "
            f"(must be one of {list(_SUPPORTED_MODES)})"
        )
    if field_order not in ("tff", "bff"):
        raise ValueError(
            f"_build_deinterlace_filter requires a resolved field_order ('tff' or 'bff'), "
            f"got {field_order!r}"
        )

    if mode == "bob":
        # Filter-independent: none of the four supported filters expose a
        # literal line-doubling ("bob") mode -- see the module docstring and
        # Config.DEFAULTS["stages"]["deinterlace"]["mode"]'s comment.
        # setfield pins the field order so separatefields splits in the
        # right temporal sequence; separatefields then halves the height
        # (one field per output frame, doubling frame count); scale doubles
        # the height back with flags=neighbor (literal line duplication, not
        # interpolation -- "line-doubled", not smoothed).
        return f"setfield=mode={field_order},separatefields,scale=iw:ih*2:flags=neighbor,setsar=1"

    if filter_name not in _SUPPORTED_FILTERS:
        raise ValueError(
            f"Unsupported stages.deinterlace.filter: {filter_name!r} "
            f"(must be one of {list(_SUPPORTED_FILTERS)})"
        )

    mode_value = 0 if mode == "frame" else 1
    return f"{filter_name}=mode={mode_value}:parity={field_order}"


def _fps_multiplier(mode: str) -> float:
    """How output frame rate relates to input frame rate for ``mode``.

    "field" and "bob" both emit one output frame per FIELD (doubling frame
    count); "frame" emits one output frame per input FRAME (unchanged).
    """
    return 2.0 if mode in ("field", "bob") else 1.0


def _run_idet_analysis(
    input_path: str, config: Config, sample_sec: float | None, timeout: float | None
) -> str | None:
    """Run a single decode-only ``idet`` pass over ``input_path`` and return its stderr.

    ``idet`` logs its "Single frame detection"/"Multi frame detection" tally
    lines to stderr at the default "info" loglevel, same convention as
    ``core/cadence.py``'s ``showinfo`` analysis pass. Returns ``None`` on any
    failure (ffmpeg missing, non-zero exit, timeout, or an unexpected
    exception) -- callers must fail open (WARNING log, SKIP), never FAIL a
    job over a detection problem.
    """
    try:
        ffmpeg = get_ffmpeg_path(config)
    except FileNotFoundError as e:
        logger.warning("deinterlace detection: %s", e)
        return None

    args = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        "idet",
    ]
    if sample_sec and sample_sec > 0:
        args += ["-t", str(sample_sec)]
    args += ["-f", "null", "-"]

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning("deinterlace detection: ffmpeg run failed for %s: %s", input_path, e)
        return None

    if result.returncode != 0:
        logger.warning(
            "deinterlace detection: ffmpeg exited %s for %s",
            result.returncode,
            input_path,
        )
        return None

    return result.stderr


class DeinterlaceStage(BaseStage):
    """Deinterlace fielded content into progressive frames (traditional/FFmpeg only).

    No AI alternative -- this stage doesn't participate in the ai_fallback
    mechanism, same as CropStage/DownscaleStage/RetimeStage.
    """

    name = "deinterlace"
    display_name = "Deinterlacing"
    description = "Convert interlaced (fielded) content into progressive frames"
    category = "enhancement"
    # Between retime (2) and crop (10ish): runs right after retime, before
    # crop, so every downstream stage sees progressive frames -- comb
    # artefacts actively corrupt motion-based analysis. pipeline.
    # default_order is authoritative for actual execution order, this is
    # only a fallback ordering signal.
    priority = 3
    supports_gpu = False

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        if self._explicitly_no_video(input_info):
            return False, "No video stream"
        return True, None

    @staticmethod
    def _explicitly_no_video(input_info: dict[str, Any] | None) -> bool:
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
        self._report_progress(0.0, "Analyzing interlacing...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to deinterlace stage",
                duration_sec=time.time() - start,
            )

        if self._explicitly_no_video(input_info):
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="No video stream",
                duration_sec=time.time() - start,
            )

        mode = self._stage_config.get("mode", "field")
        filter_name = self._stage_config.get("filter", "bwdif")
        field_order_cfg = self._stage_config.get("field_order", "auto")
        min_interlaced_ratio = self._stage_config.get("min_interlaced_ratio", 0.7)
        analysis_sample_sec = self._stage_config.get("analysis_sample_sec", 120)

        idet_counts: IdetCounts | None = None
        metadata: dict[str, Any] = {"method": "traditional", "mode": mode, "filter": filter_name}

        if field_order_cfg in ("tff", "bff"):
            resolved_field_order = field_order_cfg
            metadata["field_order_source"] = "explicit config"
        elif field_order_cfg == "auto":
            timeout = self.stage_timeout()
            stderr_text = _run_idet_analysis(input_path, self.config, analysis_sample_sec, timeout)
            idet_counts = _parse_idet_output(stderr_text)
            decision = _decide_deinterlace(idet_counts, min_interlaced_ratio)
            if idet_counts is not None:
                metadata["idet_counts"] = {
                    "tff": idet_counts.tff,
                    "bff": idet_counts.bff,
                    "progressive": idet_counts.progressive,
                    "undetermined": idet_counts.undetermined,
                }
            if not decision.interlaced:
                self._report_progress(1.0, "Not interlaced", progress_callback)
                return StageResult(
                    status=StageStatus.SKIPPED,
                    skipped_reason=decision.reason,
                    metadata=metadata,
                    duration_sec=time.time() - start,
                )
            resolved_field_order = decision.field_order or "tff"
            metadata["field_order_source"] = "idet"
            metadata["idet_ratio"] = decision.ratio
        else:
            return StageResult(
                status=StageStatus.FAILED,
                error=(
                    f"Invalid stages.deinterlace.field_order: {field_order_cfg!r} "
                    "(must be one of 'auto', 'tff', 'bff')"
                ),
                duration_sec=time.time() - start,
            )

        try:
            vf = _build_deinterlace_filter(mode, filter_name, resolved_field_order)
        except ValueError as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                metadata=metadata,
                duration_sec=time.time() - start,
            )

        metadata["field_order"] = resolved_field_order

        info = input_info or {}
        fps_in = 0.0
        try:
            fps_in = float(info.get("true_framerate") or info.get("framerate") or 0.0)
        except TypeError, ValueError:
            fps_in = 0.0

        args = [
            "-i",
            input_path,
            "-vf",
            vf,
            *timing_output_args(),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(self._stage_config.get("temp_crf", 16)),
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

        def cb(p: float, m: str) -> None:
            self._report_progress(0.2 + p * 0.8, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Deinterlacing failed: {result.stderr[:2000]}",
                metadata=metadata,
                duration_sec=time.time() - start,
            )

        if fps_in > 0:
            metadata["fps_in"] = fps_in
            metadata["fps_out"] = fps_in * _fps_multiplier(mode)

        self._report_progress(1.0, "Deinterlacing complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata=metadata,
            duration_sec=time.time() - start,
        )
