"""Auto Video Fixer - True source cadence recovery (REQUIREMENTS.md § 12.1).

``analyze_cadence()`` recovers a video's TRUE content framerate -- as opposed
to the ENCODED framerate the container claims -- by running FFmpeg's
``mpdecimate`` filter (near-duplicate frame detection) immediately followed
by ``showinfo`` in a single decode-only pass (no encode, so it's cheap
relative to any real processing stage):

    ffmpeg -i IN -map 0:v:0 -an -sn \\
           -vf mpdecimate=hi=<hi>:lo=<lo>:frac=<frac>,showinfo \\
           -f null -

``mpdecimate`` drops near-duplicate frames; ``showinfo`` immediately
downstream of it therefore logs one ``pts_time`` line (to stderr, at the
default "info" loglevel) per SURVIVING (unique) frame. Parsing those
pts_time values recovers the true content timeline directly, with no
pixel-comparison code of our own -- and ``mpdecimate``'s hi/lo/frac
thresholds are exactly the tunables needed to tolerate encoder noise between
frames that are visually "identical".

NOTE -- deviation from REQUIREMENTS.md § 12.1's sketched mechanism (which
names ``metadata=print:file=-`` instead of ``showinfo``): empirically
(ffmpeg n9.0.1), ``metadata=print`` only emits its per-frame header for a
frame that already carries at least one attached metadata KV pair (e.g. from
``signalstats``) -- ``mpdecimate`` itself never attaches any, so
``mpdecimate,metadata=print`` silently prints nothing, for every input,
always. ``showinfo`` has no such precondition and was verified end-to-end
against a real 24-in-60-padded file (120 -> 48 survivors, exactly matching
§ 12.6's documented mpdecimate reduction). Same filter-chain shape and cost,
different (working) extraction mechanism.

This module never raises: a cadence miss (ffmpeg not found, a malformed/
truncated run, a bogus/unreadable input) must never take down an otherwise
fine job. Every failure path logs a WARNING and returns a ``CadenceAnalysis``
with ``is_padded=False`` (i.e. "assume honest, do nothing") so callers can
treat cadence recovery purely as an optimization, never a correctness
dependency.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

from autovideofixer.config import Config, resolve_timeout
from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path, probe
from autovideofixer.logger import get_logger

logger = get_logger("autovideofixer.cadence")

# Standard frame rates nominal_fps snaps detected_fps onto, when within
# stages.retime.snap_tolerance -- see REQUIREMENTS.md § 12.1.
_STANDARD_RATES: tuple[float, ...] = (
    23.976,
    24.0,
    25.0,
    29.97,
    30.0,
    48.0,
    50.0,
    59.94,
    60.0,
    100.0,
    120.0,
)

_PTS_TIME_RE = re.compile(r"pts_time:([0-9.]+)")


@dataclass(frozen=True)
class CadenceAnalysis:
    """Recovered true-content-cadence report for one input (§ 12.1).

    ``encoded_fps``: what the container claims (today's ``probe().framerate``).
    ``total_frames``/``unique_frames``: frames within the analysed window,
        before/after mpdecimate; ``duplicate_ratio = 1 - unique/total``.
    ``unique_timestamps``: the recovered timeline (surviving frames' own
        ``pts_time``, in seconds).
    ``detected_fps``: ``unique_frames / analysed_duration``.
    ``nominal_fps``: ``detected_fps`` snapped to the nearest standard rate
        when within ``snap_tolerance``, else ``detected_fps`` unchanged.
    ``is_padded``: ``duplicate_ratio >= min_duplicate_ratio``.
    ``is_regular``: coefficient of variation of the inter-frame gaps is under
        ``regularity_tolerance`` -- computed on the RAW recovered gaps, so
        grid quantization (§ 12.6) can make a uniform-cadence source read as
        mildly irregular; ``grid_rate`` disambiguates that case.
    ``grid_rate``: the rate whose period divides all recovered gaps (i.e. the
        original encoded grid), when the timeline is a subset of a uniform
        grid -- true for any input that was itself CFR-encoded. ``None`` when
        no such grid was found.
    """

    encoded_fps: float
    total_frames: int
    unique_frames: int
    duplicate_ratio: float
    unique_timestamps: list[float] = field(default_factory=list)
    detected_fps: float = 0.0
    nominal_fps: float = 0.0
    is_padded: bool = False
    is_regular: bool = True
    grid_rate: float | None = None


def _empty_analysis(encoded_fps: float = 0.0) -> CadenceAnalysis:
    """A safe "assume honest, do nothing" result for any failure path."""
    return CadenceAnalysis(
        encoded_fps=encoded_fps,
        total_frames=0,
        unique_frames=0,
        duplicate_ratio=0.0,
        unique_timestamps=[],
        detected_fps=encoded_fps,
        nominal_fps=encoded_fps,
        is_padded=False,
        is_regular=True,
        grid_rate=None,
    )


def _snap_to_standard_rate(detected_fps: float, tolerance: float) -> float:
    """Snap ``detected_fps`` to the nearest ``_STANDARD_RATES`` entry when
    within ``tolerance`` (relative), else return it unchanged."""
    if detected_fps <= 0:
        return detected_fps
    best: float | None = None
    best_err = tolerance
    for rate in _STANDARD_RATES:
        err = abs(detected_fps - rate) / rate
        if err <= best_err:
            best = rate
            best_err = err
    return best if best is not None else detected_fps


def _coefficient_of_variation(gaps: list[float]) -> float:
    """Population coefficient of variation (std/mean) of ``gaps``. 0.0 for
    fewer than 2 gaps or a zero mean (not enough data to call it irregular)."""
    if len(gaps) < 2:
        return 0.0
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return 0.0
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return float(variance**0.5) / mean


def _compute_grid_rate(
    timestamps: list[float], encoded_fps: float, tolerance: float
) -> float | None:
    """Find the rate whose period divides all recovered gaps.

    Uses the container's own ``encoded_fps`` as the candidate grid (the
    overwhelmingly common case: a padded input's duplicate frames were
    inserted onto its own encoded grid) -- if every gap between surviving
    frames is within ``tolerance`` of an integer multiple of that grid's
    period, the grid is confirmed and ``encoded_fps`` is returned; else
    ``None`` (no uniform grid found).
    """
    if len(timestamps) < 2 or encoded_fps <= 0:
        return None
    period = 1.0 / encoded_fps
    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        n = gap / period
        nearest = round(n)
        if nearest <= 0:
            return None
        if abs(n - nearest) > tolerance:
            return None
    return encoded_fps


def _parse_pts_times(text: str, instance: int | None = None) -> list[float]:
    """Parse every ``pts_time:<float>`` occurrence out of ffmpeg's ``showinfo``
    stderr, in order. Malformed/empty text yields ``[]``, never raises.

    ``instance`` selects ONE ``showinfo`` filter instance by its ffmpeg log
    prefix (``[Parsed_showinfo_<n> @ ..]``). The analysis chain runs showinfo
    twice -- instance 0 before ``mpdecimate`` (every decoded frame) and
    instance 2 after it (only the survivors) -- so the two counts come from a
    single decode pass and must be attributed to the right filter. ``None``
    matches any instance (used by the fail-open path and by callers that only
    care about survivors).

    A single showinfo record can wrap across more than one log line, so lines
    are matched on the prefix and the pts_time is taken from the same line --
    counting prefix occurrences alone double-counts (observed 251 log lines
    for 123 frames).
    """
    times: list[float] = []
    if not text:
        return times
    if instance is None:
        for match in _PTS_TIME_RE.finditer(text):
            try:
                times.append(float(match.group(1)))
            except ValueError:
                continue
        return times

    prefix = f"Parsed_showinfo_{instance} "
    for line in text.splitlines():
        if prefix not in line:
            continue
        # Distinct name from the finditer loop above: that binds a non-optional
        # Match, search() returns Match | None, and reusing the name makes mypy
        # infer the narrower type for both.
        line_match = _PTS_TIME_RE.search(line)
        if not line_match:
            continue
        try:
            times.append(float(line_match.group(1)))
        except ValueError:
            continue
    return times


def analyze_cadence(path: str, config: Config, sample_sec: float | None = None) -> CadenceAnalysis:
    """Recover the true content cadence of the video at ``path``.

    ``sample_sec``: analyse only the first N seconds (cheaper decision --
    the retime STAGE's own mpdecimate pass always runs over the whole
    input regardless of this). ``None`` reads
    ``stages.retime.analysis_sample_sec`` (default 120); ``0`` analyses the
    whole file.

    Never raises -- any failure (ffmpeg missing, probe failure, no video
    stream, a malformed/truncated ffmpeg run, a timeout) is logged at
    WARNING and a safe not-padded ``CadenceAnalysis`` is returned instead,
    so a cadence miss can never fail a job.
    """
    retime_cfg = config.get("stages", "retime", default={}) or {}
    hi = retime_cfg.get("hi", 768)
    lo = retime_cfg.get("lo", 320)
    frac = retime_cfg.get("frac", 0.33)
    min_duplicate_ratio = retime_cfg.get("min_duplicate_ratio", 0.05)
    snap_tolerance = retime_cfg.get("snap_tolerance", 0.02)
    regularity_tolerance = retime_cfg.get("regularity_tolerance", 0.15)
    if sample_sec is None:
        sample_sec = retime_cfg.get("analysis_sample_sec", 120)

    try:
        probe_result = probe(path, config)
    except Exception as e:
        logger.warning("cadence analysis: probe failed for %s: %s", path, e)
        return _empty_analysis()

    if not probe_result.has_video:
        return _empty_analysis()

    encoded_fps = probe_result.framerate
    duration = probe_result.duration

    try:
        ffmpeg = get_ffmpeg_path(config)
    except FileNotFoundError as e:
        logger.warning("cadence analysis: %s", e)
        return _empty_analysis(encoded_fps)

    # NOTE -- deviation from the metadata=print mechanism sketched in
    # REQUIREMENTS.md § 12.1: empirically (ffmpeg n9.0.1), `metadata=print`
    # only emits its "frame:N pts:P pts_time:T" header for a frame that
    # already carries at least one attached metadata KV pair (e.g. from a
    # filter like signalstats) -- mpdecimate itself never attaches any, so
    # `mpdecimate,metadata=print:file=-` silently prints NOTHING for every
    # frame, on every input, always hitting this module's fail-open "no
    # frames parsed" path. `showinfo` has no such precondition: it logs a
    # `pts_time:<float>` line (to stderr, at the default "info" loglevel) for
    # EVERY frame it receives, unconditionally -- verified end-to-end against
    # a real 24-in-60-padded file (120 -> 48 survivors, exactly matching
    # § 12.6's documented mpdecimate reduction). Same filter-chain shape and
    # cost (still one decode-only pass, no encode) as the spec, different
    # (working) mechanism for extracting it.
    # A showinfo BEFORE mpdecimate as well as after, so the total (decoded)
    # frame count is MEASURED rather than derived. Deriving it as
    # `duration * encoded_fps` uses avg_frame_rate, which § 12.3 documents as
    # unreliable for VFR: a genuine 123-frame VFR capture with ZERO duplicate
    # frames advertises 60fps over 3s and so "measured" 180 total frames,
    # inventing a 0.317 duplicate_ratio and sending a file that needed nothing
    # through a full needless re-encode. Measuring both sides gives 123/123 =
    # 0.000 and the stage correctly SKIPs. ffmpeg labels each filter instance
    # in its log prefix (`[Parsed_showinfo_0 @ ..]` pre, `[Parsed_showinfo_2 @
    # ..]` post -- mpdecimate is instance 1), which is what makes the two
    # streams separable from one pass. Verified: native VFR 123/123 (0.000),
    # VFR-transcoded-to-CFR 180/123 (0.317), 24-in-60 120/48 (0.600).
    vf = f"showinfo,mpdecimate=hi={hi}:lo={lo}:frac={frac},showinfo"
    args = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        vf,
    ]
    if sample_sec and sample_sec > 0:
        args += ["-t", str(sample_sec)]
    args += ["-f", "null", "-"]

    timeout = resolve_timeout(
        retime_cfg.get("timeout") or config.get("pipeline", "stage_timeout", default=None),
        "stages.retime.timeout",
    )

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning("cadence analysis: ffmpeg run failed for %s: %s", path, e)
        return _empty_analysis(encoded_fps)

    if result.returncode != 0:
        logger.warning(
            "cadence analysis: ffmpeg exited %s for %s -- treating as not padded",
            result.returncode,
            path,
        )
        return _empty_analysis(encoded_fps)

    # showinfo logs to stderr at the default "info" loglevel (not stdout --
    # see the NOTE above the filter-chain construction).
    # Instance 2 = post-mpdecimate survivors (the recovered timeline);
    # instance 0 = every decoded frame, giving a MEASURED total. Fall back to
    # matching any instance if the prefixes are absent (e.g. a future ffmpeg
    # log-format change), so a format drift degrades to the old behaviour
    # rather than reporting zero frames.
    unique_timestamps = _parse_pts_times(result.stderr, instance=2)
    decoded_timestamps = _parse_pts_times(result.stderr, instance=0)
    if not unique_timestamps:
        unique_timestamps = _parse_pts_times(result.stderr)
        decoded_timestamps = []
    unique_frames = len(unique_timestamps)
    if unique_frames == 0:
        logger.warning(
            "cadence analysis: no frames parsed from mpdecimate output for %s "
            "-- treating as not padded",
            path,
        )
        return _empty_analysis(encoded_fps)

    analysed_duration = duration
    if sample_sec and sample_sec > 0:
        analysed_duration = min(duration, sample_sec) if duration > 0 else sample_sec

    if decoded_timestamps:
        # MEASURED total (preferred). See the filter-chain comment: deriving
        # this from avg_frame_rate invents duplicates on genuine VFR input.
        total_frames = max(unique_frames, len(decoded_timestamps))
    elif analysed_duration > 0 and encoded_fps > 0:
        total_frames = max(unique_frames, round(analysed_duration * encoded_fps))
    else:
        total_frames = unique_frames

    duplicate_ratio = 1.0 - (unique_frames / total_frames) if total_frames > 0 else 0.0
    detected_fps = unique_frames / analysed_duration if analysed_duration > 0 else 0.0
    nominal_fps = _snap_to_standard_rate(detected_fps, snap_tolerance)
    is_padded = duplicate_ratio >= min_duplicate_ratio

    gaps = [
        unique_timestamps[i + 1] - unique_timestamps[i] for i in range(len(unique_timestamps) - 1)
    ]
    is_regular = _coefficient_of_variation(gaps) < regularity_tolerance
    grid_rate = _compute_grid_rate(unique_timestamps, encoded_fps, regularity_tolerance)

    return CadenceAnalysis(
        encoded_fps=encoded_fps,
        total_frames=total_frames,
        unique_frames=unique_frames,
        duplicate_ratio=duplicate_ratio,
        unique_timestamps=unique_timestamps,
        detected_fps=detected_fps,
        nominal_fps=nominal_fps,
        is_padded=is_padded,
        is_regular=is_regular,
        grid_rate=grid_rate,
    )


def probe_frame_timestamps(
    path: str, config: Config, *, timeout: float | None = None
) -> list[float] | None:
    """Read the ascending presentation timestamps of ``path``'s video stream
    (REQUIREMENTS.md § 16.2 -- the per-gap adaptive AI interpolation
    timeline probe).

    Lives here (rather than ``core/ffmpeg_utils.py``) because it reuses this
    module's ``showinfo``-parsing machinery (``_parse_pts_times()``) and
    fail-open conventions almost verbatim -- the only difference from
    ``analyze_cadence()`` is that this probe wants EVERY decoded frame's
    timestamp, not just mpdecimate's survivors, so the filter chain is a
    bare ``showinfo`` with no ``mpdecimate`` ahead of it:

        ffmpeg -i IN -map 0:v:0 -an -sn -vf showinfo -f null -

    Returns ``None`` on ANY failure -- ffmpeg/ffprobe missing, no video
    stream, a malformed/truncated run, a timeout, or zero timestamps
    parsed -- and NEVER raises. Callers (``InterpolateStage``) must fall
    back to the existing uniform-factor interpolation path on ``None``,
    logged at INFO: a timeline miss must never fail a job, same fail-open
    contract as ``analyze_cadence()``.

    This does not itself validate the returned count against any expected
    frame count -- REQUIREMENTS.md § 16.2 asks callers to validate that
    against the frame count the reader actually produces (or, cheaply, the
    same probe's ``frame_count``), since only the caller knows what to
    compare against and how strict to be about it.
    """
    try:
        probe_result = probe(path, config)
    except Exception as e:
        logger.info("timeline probe: probe failed for %s: %s", path, e)
        return None
    if not probe_result.has_video:
        return None

    try:
        ffmpeg = get_ffmpeg_path(config)
    except FileNotFoundError as e:
        logger.info("timeline probe: %s", e)
        return None

    args = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        "showinfo",
        "-f",
        "null",
        "-",
    ]

    resolved_timeout = resolve_timeout(
        timeout or config.get("pipeline", "stage_timeout", default=None),
        "stages.interpolate.timeline_probe_timeout",
    )

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=resolved_timeout,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.info("timeline probe: ffmpeg run failed for %s: %s", path, e)
        return None

    if result.returncode != 0:
        logger.info(
            "timeline probe: ffmpeg exited %s for %s -- falling back to uniform-factor "
            "interpolation",
            result.returncode,
            path,
        )
        return None

    # showinfo logs to stderr at the default "info" loglevel -- only one
    # filter instance in this chain, so any-instance parsing is unambiguous.
    timestamps = _parse_pts_times(result.stderr)
    if not timestamps:
        logger.info(
            "timeline probe: no frame timestamps parsed for %s -- falling back to "
            "uniform-factor interpolation",
            path,
        )
        return None

    return timestamps


__all__ = ["CadenceAnalysis", "analyze_cadence", "probe_frame_timestamps"]
