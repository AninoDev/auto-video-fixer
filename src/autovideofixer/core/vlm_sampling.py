"""VLM frame sampling: planner + sampler shared by whole-video and per-scene analysis.

REQUIREMENTS.md § 25. ``VideoAnalyzer.run_vlm_analysis`` (whole video) and
``run_vlm_analysis_for_scene`` (one scene) both call :func:`sample_vlm_frames` with the
time range to look at. The module is layered so the maths is testable without any video:

* **Pure planning** -- :func:`base_frame_count`, :func:`segment_centres`, :func:`tiles_for`,
  :func:`fit_scale`, :func:`plan_sampling`. Numbers in, numbers out.
* **Pure selection** -- :func:`select_frames` drives any object with a ``read(t)`` method
  (a "frame source") through the duplicate-removal and refill loop. Only needs numpy.
* **ffmpeg / cv2 I/O** -- :class:`FFmpegFrameSource` and :func:`sample_vlm_frames` (frame
  timestamps from ffprobe, accurate-seek decode, resizing, JPEG encoding, temp-dir ownership).

Why one shared sampler: the old code had two near-identical extractors that seeked by
``frame_index = t * fps`` (wrong for VFR input), could send the same frame twice and had no
notion of what the model server can digest.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

BALANCE_MODES = ("auto", "temporal", "spatial", "fixed")

# Below this long-side the picture is useless to a VLM; frames are dropped instead (§ 25.3).
ABSOLUTE_MIN_LONG_SIDE_PX = 64
# Width of the grayscale thumbnail used for duplicate / motion comparison (§ 25.2).
THUMB_WIDTH = 64
# Refill gives up after this many extraction attempts per target frame (§ 25.2).
REFILL_ATTEMPT_FACTOR = 4
# Number of tiny thumbnails the "auto" balance motion probe reads (§ 25.3).
MOTION_PROBE_FRAMES = 8
# Mean adjacent-thumbnail difference that counts as "maximum motion" (m = 1). Calibrated in
# REQUIREMENTS.md § 25.5.
MOTION_FULL_SCALE = 0.10
# Duration saturation constant: d / (d + 30 s) (§ 25.3).
DURATION_HALF_SATURATION_SEC = 30.0
# Decoded-frame bytes the shared multi-endpoint cache may hold before it drops the oldest
# images (their thumbnails stay; a later read re-decodes). Bounds the cross-check memory the
# way freeing rejected frames bounds the single-endpoint path (§ 25.5, § 27.3).
SHARED_CACHE_BUDGET_BYTES = 512 * 1024 * 1024


# --------------------------------------------------------------------------------------
# Configuration value objects
# --------------------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class SamplingConfig:
    """``analysis.vlm.sampling`` (validated)."""

    min_frames: int = 4
    max_frames: int = 8
    interval_sec: float = 10.0
    dedupe_consecutive: bool = True
    dedupe_threshold: float = 0.002
    balance: str = "auto"
    min_long_side_px: int = 336

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        min_frames: int | None = None,
        max_frames: int | None = None,
        interval_sec: float | None = None,
        where: str = "analysis.vlm.sampling",
    ) -> SamplingConfig:
        """Build from the config block, with per-run overrides (CLI flags / kwargs).

        Raises ``ValueError`` naming the offending key. If a per-run ``max_frames`` override
        is below the configured ``min_frames`` (and ``min_frames`` was not itself overridden)
        the minimum is lowered to match, so ``--max-sample-frames 2`` just works.
        """
        d = cls()
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid {where}: expected a mapping, got {type(raw).__name__}")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {where}: {', '.join(map(str, unknown))} "
                f"(valid: {', '.join(cls.__dataclass_fields__)})"
            )

        def pick(name: str, default: Any) -> Any:
            v = raw.get(name)
            return default if v is None else v

        cfg_min = pick("min_frames", d.min_frames)
        cfg_max = pick("max_frames", d.max_frames)
        cfg_interval = pick("interval_sec", d.interval_sec)
        eff_min = min_frames if min_frames is not None else cfg_min
        eff_max = max_frames if max_frames is not None else cfg_max
        eff_interval = interval_sec if interval_sec is not None else cfg_interval
        if max_frames is not None and min_frames is None and _is_int(eff_max):
            if _is_int(eff_min) and eff_max < eff_min:
                eff_min = max(1, eff_max)

        dedupe = pick("dedupe_consecutive", d.dedupe_consecutive)
        threshold = pick("dedupe_threshold", d.dedupe_threshold)
        balance = pick("balance", d.balance)
        min_side = pick("min_long_side_px", d.min_long_side_px)

        if not _is_int(eff_min) or eff_min < 1:
            raise ValueError(f"Invalid {where}.min_frames {eff_min!r}: must be an integer >= 1")
        if not _is_int(eff_max) or eff_max < 1:
            raise ValueError(f"Invalid {where}.max_frames {eff_max!r}: must be an integer >= 1")
        if eff_max < eff_min:
            raise ValueError(
                f"Invalid {where}: max_frames ({eff_max}) is smaller than min_frames ({eff_min})"
            )
        if not _is_num(eff_interval) or eff_interval <= 0:
            raise ValueError(f"Invalid {where}.interval_sec {eff_interval!r}: must be a number > 0")
        if not isinstance(dedupe, bool):
            raise ValueError(
                f"Invalid {where}.dedupe_consecutive {dedupe!r}: must be true or false"
            )
        if not _is_num(threshold) or not 0 <= threshold < 1:
            raise ValueError(
                f"Invalid {where}.dedupe_threshold {threshold!r}: must be a number in [0, 1)"
            )
        if balance not in BALANCE_MODES:
            raise ValueError(f"Invalid {where}.balance {balance!r}: must be one of {BALANCE_MODES}")
        if not _is_int(min_side) or min_side < ABSOLUTE_MIN_LONG_SIDE_PX:
            raise ValueError(
                f"Invalid {where}.min_long_side_px {min_side!r}: must be an integer "
                f">= {ABSOLUTE_MIN_LONG_SIDE_PX}"
            )
        return cls(
            min_frames=int(eff_min),
            max_frames=int(eff_max),
            interval_sec=float(eff_interval),
            dedupe_consecutive=dedupe,
            dedupe_threshold=float(threshold),
            balance=str(balance),
            min_long_side_px=int(min_side),
        )


@dataclass(frozen=True)
class VisionLimits:
    """Model image limits (``analysis.endpoints.<name>.vision``). ``None`` = no limit."""

    max_long_side_px: int | None = None
    max_megapixels: float | None = None
    tile_width: int | None = None
    tile_height: int | None = None
    tokens_per_tile: float = 1
    max_tiles_per_frame: int | None = None
    max_tiles_per_call: int | None = None
    max_frames_per_call: int | None = None
    jpeg_quality: int = 90

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> VisionLimits:
        """Tolerant constructor for an already-validated mapping (see
        ``endpoints.resolve_vision``); missing keys take the defaults."""
        raw = raw or {}
        kwargs = {k: raw[k] for k in cls.__dataclass_fields__ if raw.get(k) is not None}
        return cls(**kwargs)

    @property
    def has_tiles(self) -> bool:
        return bool(self.tile_width and self.tile_height)


# --------------------------------------------------------------------------------------
# Pure planning
# --------------------------------------------------------------------------------------


def base_frame_count(
    duration: float,
    cfg: SamplingConfig,
    max_frames_per_call: int | None = None,
    real_frames: int | None = None,
) -> int:
    """§ 25.1 steps 1-3: ``ceil(dur/interval)`` clamped to the bounds and real-frame count.

    Never exceeds ``real_frames`` (distinct frames actually in the range) and never pads.
    Hard limit order: real frames and ``max_frames_per_call`` beat ``min_frames``.
    """
    n = max(1, math.ceil(max(duration, 0.0) / cfg.interval_sec))
    n = max(cfg.min_frames, min(cfg.max_frames, n))
    if max_frames_per_call is not None:
        n = min(n, max_frames_per_call)
    if real_frames is not None:
        n = min(n, max(1, real_frames))
    return max(1, n)


def segment_centres(start: float, duration: float, n: int) -> list[float]:
    """Centres of ``n`` equal segments of ``[start, start + duration)`` (§ 25.1 step 4)."""
    if n <= 0 or duration <= 0:
        return []
    return [start + (i + 0.5) * duration / n for i in range(n)]


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Aspect-preserving size at ``scale`` (<= 1 by construction; never below 1 px)."""
    return max(1, round(width * scale)), max(1, round(height * scale))


def tiles_for(width: int, height: int, limits: VisionLimits) -> int | None:
    """``ceil(W/tw) * ceil(H/th)`` -- models pad each axis up to a tile multiple.

    ``None`` when no tile size is configured (tile maths disabled).
    """
    if not limits.has_tiles:
        return None
    assert limits.tile_width and limits.tile_height
    return math.ceil(width / limits.tile_width) * math.ceil(height / limits.tile_height)


def estimated_tokens(tiles: int | None, limits: VisionLimits) -> int | None:
    if tiles is None:
        return None
    return math.ceil(tiles * limits.tokens_per_tile)


def pixel_cap_scale(width: int, height: int, limits: VisionLimits) -> float:
    """Largest scale (<= 1: downscale only) allowed by ``max_long_side_px`` / ``max_megapixels``.

    Applied even when no tile size is configured.
    """
    scale = 1.0
    long_side = max(width, height)
    if limits.max_long_side_px:
        scale = min(scale, limits.max_long_side_px / long_side)
    if limits.max_megapixels:
        scale = min(scale, math.sqrt(limits.max_megapixels * 1e6 / (width * height)))
    return max(scale, 0.0)


def _long_side_scale(width: int, height: int, long_side_px: int) -> float:
    return min(1.0, long_side_px / max(width, height))


def fit_scale(
    width: int,
    height: int,
    limits: VisionLimits,
    max_scale: float,
    tile_budget: int | None,
) -> float | None:
    """Largest ``scale <= max_scale`` whose frame fits ``tile_budget`` tiles.

    Never goes below the absolute 64-px-long-side floor; returns ``None`` if even the floor
    does not fit (the caller then drops frames). ``tile_budget=None`` (or tiles disabled)
    means no tile constraint: ``max_scale`` is returned (clamped to the floor).
    """
    floor = min(max_scale, _long_side_scale(width, height, ABSOLUTE_MIN_LONG_SIDE_PX))
    if tile_budget is None or not limits.has_tiles:
        return max(max_scale, floor)

    def fits(s: float) -> bool:
        w, h = scaled_size(width, height, s)
        t = tiles_for(w, h, limits)
        assert t is not None
        return t <= tile_budget

    if fits(max_scale):
        return max_scale
    if not fits(floor):
        return None
    lo, hi = floor, max_scale  # fits(lo) true, fits(hi) false
    for _ in range(40):
        mid = (lo + hi) / 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def temporal_preference(balance: str, duration: float, motion: float | None) -> float | None:
    """``p`` in [0, 1] for the frames-vs-resolution trade; ``None`` for ``fixed``.

    ``auto``: mean of the normalised motion ``m`` (0 if unknown -> 0.5) and the duration
    saturation ``d / (d + 30 s)``. Long, high-motion ranges prefer more, smaller frames.
    """
    if balance == "fixed":
        return None
    if balance == "temporal":
        return 1.0
    if balance == "spatial":
        return 0.0
    m = 0.5 if motion is None else min(1.0, max(0.0, motion))
    d = max(duration, 0.0)
    sat = d / (d + DURATION_HALF_SATURATION_SEC)
    return min(1.0, max(0.0, (m + sat) / 2))


@dataclass(frozen=True)
class SamplePlan:
    """Result of :func:`plan_sampling`: how many frames, at what size."""

    target_frames: int
    width: int
    height: int
    scale: float
    tiles_per_frame: int | None
    balance: str
    temporal_pref: float | None
    n_lo: int
    n_hi: int
    notes: tuple[str, ...] = ()

    def est_total_tokens(self, limits: VisionLimits, frames: int | None = None) -> int | None:
        if self.tiles_per_frame is None:
            return None
        n = self.target_frames if frames is None else frames
        return math.ceil(n * self.tiles_per_frame * limits.tokens_per_tile)


def plan_sampling(
    src_width: int,
    src_height: int,
    duration: float,
    real_frames: int | None,
    cfg: SamplingConfig,
    limits: VisionLimits,
    motion: float | None = None,
) -> SamplePlan:
    """Pick ``(n, size)`` for one range (§ 25.1 + § 25.3). Pure.

    ``n_lo`` is the § 25.1 count; ``n_hi`` the most frames that fit ``max_tiles_per_call`` at
    ``min_long_side_px`` (capped by ``max_frames``, ``max_frames_per_call`` and the real-frame
    count). ``n = round(n_lo + p * (n_hi - n_lo))``, then the largest scale that fits.
    Priority: hard limits > ``min_frames`` > ``min_long_side_px``; floor 64 px, below which
    frames are dropped instead.
    """
    notes: list[str] = []
    n_lo = base_frame_count(duration, cfg, limits.max_frames_per_call, real_frames)
    cap_n = cfg.max_frames
    if limits.max_frames_per_call is not None:
        cap_n = min(cap_n, limits.max_frames_per_call)
    if real_frames is not None:
        cap_n = min(cap_n, max(1, real_frames))
    cap_n = max(1, cap_n)

    # Pixel caps and the per-frame tile cap always apply.
    s_cap = pixel_cap_scale(src_width, src_height, limits)
    if limits.max_tiles_per_frame is not None and limits.has_tiles:
        s_tpf = fit_scale(src_width, src_height, limits, s_cap, limits.max_tiles_per_frame)
        if s_tpf is None:
            s_tpf = min(s_cap, _long_side_scale(src_width, src_height, ABSOLUTE_MIN_LONG_SIDE_PX))
            notes.append("max_tiles_per_frame unreachable even at the 64 px floor")
        s_cap = s_tpf
    s_cap = max(s_cap, 1e-6)

    budget = limits.max_tiles_per_call
    balance = cfg.balance
    p: float | None = None
    n = n_lo
    n_hi = n_lo

    if budget is not None and limits.has_tiles:
        s_min = min(s_cap, _long_side_scale(src_width, src_height, cfg.min_long_side_px))
        w_min, h_min = scaled_size(src_width, src_height, s_min)
        tiles_min = tiles_for(w_min, h_min, limits) or 1
        n_hi = max(0, min(cap_n, budget // tiles_min))
        p = temporal_preference(balance, duration, motion)
        n_floor = min(cfg.min_frames, cap_n)
        if p is None:  # fixed: keep the § 25.1 count, shrink resolution
            n = n_lo
        elif n_hi >= n_lo:
            n = math.floor(n_lo + p * (n_hi - n_lo) + 0.5)
        else:
            # Can't fit n_lo frames even at min_long_side: min_frames outranks
            # min_long_side_px, so keep at least min_frames (shrinking below min_long_side).
            n = min(n_lo, max(n_hi, n_floor))
            notes.append("tile budget below min_long_side_px for the requested frame count")
        n = max(1, min(n, cap_n))
        scale: float | None = None
        while n >= 1:
            per_frame = budget // n
            if limits.max_tiles_per_frame is not None:
                per_frame = min(per_frame, limits.max_tiles_per_frame)
            scale = fit_scale(src_width, src_height, limits, s_cap, per_frame)
            if scale is not None:
                break
            n -= 1
            notes.append("dropped a frame: tile budget unreachable at the 64 px floor")
        if scale is None:
            n = 1
            scale = min(s_cap, _long_side_scale(src_width, src_height, ABSOLUTE_MIN_LONG_SIDE_PX))
            notes.append("max_tiles_per_call unreachable even with one 64 px frame")
    else:
        if budget is not None:
            notes.append("max_tiles_per_call ignored: no tile_width/tile_height configured")
        scale = s_cap
        n = n_lo

    w, h = scaled_size(src_width, src_height, scale)
    return SamplePlan(
        target_frames=n,
        width=w,
        height=h,
        scale=scale,
        tiles_per_frame=tiles_for(w, h, limits),
        balance=balance,
        temporal_pref=p,
        n_lo=n_lo,
        n_hi=n_hi,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# Selection: duplicate removal + refill (pure; any frame source)
# --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    """One decoded frame: ``key`` identifies the decoded frame (its timestamp), ``thumb`` is
    the 64-px-wide grayscale thumbnail, ``image`` the (already resized) frame."""

    key: Any
    ts: float
    thumb: np.ndarray
    image: Any = None


class FrameSource(Protocol):
    def read(self, t: float) -> Candidate | None:  # pragma: no cover - protocol
        ...


def thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute luma difference of two equal-size uint8 thumbnails, normalised to 0-1."""
    if a.shape != b.shape:
        return 1.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() / 255.0)


@dataclass
class Selection:
    kept: list[Candidate]
    dropped_duplicates: int
    attempts: int
    target: int
    exhausted: bool = False


def select_frames(
    source: FrameSource,
    start: float,
    end: float,
    target: int,
    *,
    dedupe: bool = True,
    threshold: float = 0.002,
    min_gap: float = 0.0,
) -> Selection:
    """Pick up to ``target`` distinct frames from ``[start, end)`` (§ 25.2).

    Initial positions are segment centres. A candidate is dropped when it is the same decoded
    frame as one already seen (not counted as a duplicate), or (with ``dedupe``) within
    ``threshold`` of an adjacent KEPT frame (counted; A, A, A, B -> A, B; A, B, A is fine).
    Then, while below ``target``, refill at the
    midpoint of the largest time gap between probed positions (range edges included),
    re-checking both neighbours. Stops at ``target``, when no gap wider than ``min_gap``
    remains (frame granularity exhausted) or after ``4 * target`` attempts.
    """
    kept: list[Candidate] = []
    seen: set[Any] = set()
    dropped = 0
    attempts = 0
    limit = REFILL_ATTEMPT_FACTOR * target
    probes: list[float] = [start, end]

    def attempt(t: float) -> None:
        nonlocal dropped, attempts
        attempts += 1
        cand = source.read(t)
        if cand is None:
            return
        if cand.key in seen:
            # The position resolved to a frame already looked at (frame granularity), not a
            # duplicate in the video, so it isn't counted in ``dropped_duplicates``.
            return
        seen.add(cand.key)
        idx = 0
        while idx < len(kept) and kept[idx].ts < cand.ts:
            idx += 1
        if dedupe:
            neighbours = []
            if idx > 0:
                neighbours.append(kept[idx - 1])
            if idx < len(kept):
                neighbours.append(kept[idx])
            if any(thumb_diff(cand.thumb, nb.thumb) <= threshold for nb in neighbours):
                dropped += 1
                # Free the full-size frame now: sources may cache candidates, and holding
                # every rejected 4K frame would cost ~25 MB each.
                cand.image = None
                return
        kept.insert(idx, cand)

    centres = segment_centres(start, end - start, target)
    for t in centres:
        if attempts >= limit:
            break
        probes.append(t)
        attempt(t)
    probes.sort()

    exhausted = False
    while len(kept) < target and attempts < limit:
        gaps = [(probes[i + 1] - probes[i], i) for i in range(len(probes) - 1)]
        gap, i = max(gaps, default=(0.0, 0))
        if gap <= max(min_gap, 1e-9):
            exhausted = True
            break
        mid = (probes[i] + probes[i + 1]) / 2
        probes.insert(i + 1, mid)
        attempt(mid)
    return Selection(
        kept=kept,
        dropped_duplicates=dropped,
        attempts=attempts,
        target=target,
        exhausted=exhausted,
    )


def motion_score(source: FrameSource, start: float, end: float) -> float | None:
    """Cheap motion probe for ``balance: auto``: mean adjacent difference across
    ``MOTION_PROBE_FRAMES`` evenly spaced thumbnails, scaled so ``MOTION_FULL_SCALE`` -> 1."""
    thumbs: list[np.ndarray] = []
    for t in segment_centres(start, end - start, MOTION_PROBE_FRAMES):
        cand = source.read(t)
        if cand is not None:
            thumbs.append(cand.thumb)
    if len(thumbs) < 2:
        return None
    diffs = [thumb_diff(a, b) for a, b in zip(thumbs, thumbs[1:], strict=False)]
    return min(1.0, (sum(diffs) / len(diffs)) / MOTION_FULL_SCALE)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


@dataclass
class SamplingReport:
    """What was actually sent to the VLM for one range (§ 25.4)."""

    frames: int = 0
    dropped_duplicates: int = 0
    width: int = 0
    height: int = 0
    source_width: int = 0
    source_height: int = 0
    tiles_per_frame: int | None = None
    est_tiles: int | None = None
    est_vision_tokens: int | None = None
    target_frames: int = 0
    attempts: int = 0
    balance: str = ""
    temporal_pref: float | None = None
    jpeg_quality: int = 90
    start_time: float = 0.0
    end_time: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def frame_size(self) -> str:
        return f"{self.width}x{self.height}" if self.width and self.height else ""

    def summary(self) -> str:
        text = (
            f"{self.frames} frame(s) sent (target {self.target_frames}), "
            f"{self.dropped_duplicates} duplicate(s) dropped"
        )
        if self.frame_size:
            text += f", {self.frame_size}"
        if self.est_tiles is not None:
            text += f", ~{self.est_tiles} tile(s) / ~{self.est_vision_tokens} vision token(s)"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "dropped_duplicates": self.dropped_duplicates,
            "frame_size": self.frame_size,
            "width": self.width,
            "height": self.height,
            "source_size": f"{self.source_width}x{self.source_height}",
            "tiles_per_frame": self.tiles_per_frame,
            "est_tiles": self.est_tiles,
            "est_vision_tokens": self.est_vision_tokens,
            "target_frames": self.target_frames,
            "attempts": self.attempts,
            "balance": self.balance,
            "temporal_pref": self.temporal_pref,
            "jpeg_quality": self.jpeg_quality,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "notes": list(self.notes),
        }


@dataclass
class SampledFrames:
    """JPEG paths (time order) + the report. The caller owns ``tmp_dir`` and must remove it
    after the provider has read the files."""

    paths: list[str]
    tmp_dir: str
    report: SamplingReport


# --------------------------------------------------------------------------------------
# ffmpeg / cv2 I/O
# --------------------------------------------------------------------------------------


@dataclass
class Timeline:
    """What ffprobe says about the video stream: display size and every frame's time.

    ``times`` are display-order frame timestamps in seconds *relative to the container's
    start_time* (the same clock ffmpeg's ``-ss`` uses), one per packet. Exact for CFR and VFR
    alike -- this is what lets the sampler pick "the frame on screen at t" without trusting
    ``frame_index / fps``.
    """

    width: int
    height: int
    times: list[float]
    duration: float = 0.0  # container-reported duration (0 = unknown)


def _probe_stream(filepath: str, ffprobe: str) -> tuple[int, int, float, float] | None:
    """``(display_width, display_height, start_time, duration)`` of the first video stream."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_side_data=rotation:stream_tags=rotate:format=start_time,duration",
        "-of",
        "json",
        "-i",
        filepath,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL
        )
        data = json.loads(out.stdout or "{}")
        stream = data["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError:
        return None
    rotation = 0.0
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = float(sd["rotation"])
    if not rotation:
        try:
            rotation = float((stream.get("tags") or {}).get("rotate", 0))
        except TypeError, ValueError:
            rotation = 0.0
    if int(abs(rotation)) % 180 == 90:  # ffmpeg autorotates, so the frames come out swapped
        width, height = height, width
    try:
        start = float((data.get("format") or {}).get("start_time") or 0.0)
    except TypeError, ValueError:
        start = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except TypeError, ValueError:
        duration = 0.0
    return (
        width,
        height,
        start if math.isfinite(start) else 0.0,
        duration if math.isfinite(duration) and duration > 0 else 0.0,
    )


def probe_timeline(
    filepath: str,
    ffprobe: str,
    start: float | None = None,
    end: float | None = None,
) -> Timeline | None:
    """Frame times for the video (or, with ``start``/``end`` seconds, just near that range).

    Reads packet timestamps only (no decoding). ``None`` if the file can't be probed or has no
    video packets.
    """
    info = _probe_stream(filepath, ffprobe)
    if info is None:
        return None
    width, height, start_time, duration = info
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if start is not None and end is not None:
        lo = max(0.0, start - 1.0) + start_time
        cmd += ["-read_intervals", f"{lo:.3f}%{end + 1.0 + start_time:.3f}"]
    cmd += ["-show_entries", "packet=pts_time,dts_time", "-of", "csv=p=0", "-i", filepath]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL
        )
    except OSError, subprocess.SubprocessError:
        return None
    times: list[float] = []
    for line in out.stdout.splitlines():
        fields = line.strip().split(",")
        for raw in fields[:2]:  # pts_time, falling back to dts_time
            try:
                v = float(raw)
            except ValueError:
                continue
            if math.isfinite(v):
                times.append(v - start_time)
                break
    if not times:
        return None
    times.sort()
    return Timeline(width=width, height=height, times=times, duration=duration)


_SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


class FFmpegFrameSource:
    """Frame source that decodes the frame on screen at time ``t`` with ``ffmpeg -ss``.

    Why not ``cv2.VideoCapture.set(CAP_PROP_POS_MSEC)``: OpenCV converts the time to a frame
    *number* using the average fps and returns the first frame at or after it. On variable
    frame rate input that lands on the wrong frame, seconds away in a long hold
    (REQUIREMENTS.md § 25.5). Here the frame index comes from ``Timeline`` (real timestamps),
    ffmpeg accurate-seeks to exactly that frame's timestamp, and ``showinfo`` reports the
    timestamp that was really decoded, which is checked against the plan. The frame index is
    also the decoded-frame identity (``Candidate.key``).
    """

    def __init__(
        self,
        filepath: str,
        timeline: Timeline,
        ffmpeg: str,
        lo: int = 0,
        hi: int | None = None,
        *,
        cache_budget_bytes: int | None = None,
    ):
        self.filepath = filepath
        self.timeline = timeline
        self.ffmpeg = ffmpeg
        self.lo = lo
        self.hi = len(timeline.times) - 1 if hi is None else hi
        self.target_size: tuple[int, int] | None = None
        self.decodes = 0
        self._cache: dict[tuple[int, tuple[int, int] | None], Candidate] = {}
        # Shared mode (several endpoints read through ``ScaledView``): the cached candidate
        # keeps its image until the byte budget is exceeded, then the oldest images are
        # dropped and re-decoded if wanted again. ``None`` = the single-endpoint behaviour,
        # where ``select_frames`` frees rejected candidates' images in place.
        self.cache_budget_bytes = cache_budget_bytes

    def index_at(self, t: float) -> int:
        """Index of the frame on screen at ``t`` (last frame with time <= t), within range."""
        k = bisect.bisect_right(self.timeline.times, t + 1e-6) - 1
        return min(max(k, self.lo), self.hi)

    def read(self, t: float) -> Candidate | None:
        k = self.index_at(t)
        cached = self._cache.get((k, self.target_size))
        if cached is not None and (cached.image is not None or self.cache_budget_bytes is None):
            return cached
        cand = self._decode(k)
        if cand is not None:
            self._cache[(k, self.target_size)] = cand
            self._enforce_budget(cand)
        return cand

    def _enforce_budget(self, newest: Candidate) -> None:
        if self.cache_budget_bytes is None:
            return
        total = sum(c.image.nbytes for c in self._cache.values() if c.image is not None)
        for cand in self._cache.values():
            if total <= self.cache_budget_bytes:
                break
            if cand is newest or cand.image is None:
                continue
            total -= cand.image.nbytes
            cand.image = None

    def clear(self) -> None:
        """Drop every cached frame (the shared cache is freed once all JPEGs are written)."""
        self._cache.clear()

    def _run(self, seek: float, w: int, h: int, *, fast: bool) -> tuple[bytes, float] | None:
        """One ffmpeg decode of the first frame at/after ``seek``; ``(bgr bytes, its time)``.

        ``fast`` seeks the input (``-ss`` before ``-i``). Formats without a seek index (MPEG-TS
        with long GOPs, some damaged files) can then yield nothing, so the slow path decodes
        from the start and selects the frame in the filter graph instead.
        """
        vf = "showinfo"
        if (w, h) != (self.timeline.width, self.timeline.height):
            vf += f",scale={w}:{h}:flags=area"
        if not fast:
            vf = f"select=gte(t\\,{seek:.6f})," + vf
        cmd = [self.ffmpeg, "-v", "info", "-nostdin"]
        if fast:
            cmd += ["-ss", f"{seek:.6f}"]
        cmd += ["-i", self.filepath, "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-dn"]
        cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.decodes += 1
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600, stdin=subprocess.DEVNULL)
        except OSError, subprocess.SubprocessError:
            logger.warning("ffmpeg frame extraction failed for %r at %.3fs", self.filepath, seek)
            return None
        if len(proc.stdout) != w * h * 3:
            return None
        # decode-and-check: showinfo's pts_time is relative to the -ss seek point (fast path)
        m = _SHOWINFO_PTS_RE.search(proc.stderr.decode("utf-8", "replace"))
        if not m:
            return proc.stdout, seek
        return proc.stdout, (seek if fast else 0.0) + float(m.group(1))

    def _decode(self, k: int) -> Candidate | None:
        import cv2

        times = self.timeline.times
        want = times[k]
        w, h = self.target_size or (self.timeline.width, self.timeline.height)
        seek = max(0.0, want - 0.0004)  # frames closer than 0.4 ms apart can't be told apart
        got = self._run(seek, w, h, fast=True) or self._run(seek, w, h, fast=False)
        if got is None:
            logger.debug("no frame decoded at %.3fs for %r", want, self.filepath)
            return None
        raw, actual = got
        if abs(actual - want) > 0.002:
            k2 = min(max(bisect.bisect_left(times, actual - 0.002), self.lo), self.hi)
            if abs(times[k2] - actual) > 0.002:
                k2 = k
            logger.debug("wanted frame %d (%.3fs) but decoded %.3fs", k, want, actual)
            k = k2
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tw = THUMB_WIDTH
        thumb = cv2.resize(gray, (tw, max(1, round(h * tw / w))), interpolation=cv2.INTER_AREA)
        return Candidate(key=k, ts=times[k], thumb=thumb, image=frame)


class ScaledView:
    """One endpoint's window onto a shared :class:`FFmpegFrameSource` (§ 27.3).

    The source decodes each frame once at the largest size any endpoint's plan needs; this view
    hands the selection loop a *fresh* ``Candidate`` per read with the image downscaled
    (``INTER_AREA``, never up) to this endpoint's own plan size. Fresh, so ``select_frames``
    freeing a rejected candidate's image can't blank the cached frame another endpoint still
    wants.
    """

    def __init__(self, source: FFmpegFrameSource, size: tuple[int, int]):
        self.source = source
        self.size = size

    def read(self, t: float) -> Candidate | None:
        import cv2

        cand = self.source.read(t)
        if cand is None or cand.image is None:
            return None
        image = cand.image
        w, h = self.size
        if image.shape[1] != w or image.shape[0] != h:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return Candidate(key=cand.key, ts=cand.ts, thumb=cand.thumb, image=image)


def _empty_frames(
    vision: VisionLimits, sampling: SamplingConfig, start_time: float
) -> SampledFrames:
    report = SamplingReport(
        jpeg_quality=vision.jpeg_quality, balance=sampling.balance, start_time=start_time
    )
    return SampledFrames(paths=[], tmp_dir="", report=report)


def sample_vlm_frames(
    filepath: str,
    sampling: SamplingConfig,
    vision: VisionLimits,
    start_time: float = 0.0,
    end_time: float | None = None,
    *,
    prefix: str = "avf_samples_",
    config: Any = None,
) -> SampledFrames:
    """Plan, extract, de-duplicate and JPEG-encode the VLM frames for one range.

    ``end_time=None`` means to the end of the video. Returns an empty ``SampledFrames``
    (``paths == []``) when the video can't be probed or decoded. The caller removes
    ``tmp_dir`` once the provider call is done (the files must outlive this call).
    ``config`` (optional ``Config``) only selects the ffmpeg/ffprobe binaries.
    """
    return sample_vlm_frames_multi(
        filepath, sampling, [vision], start_time, end_time, prefix=prefix, config=config
    )[0]


def sample_vlm_frames_multi(
    filepath: str,
    sampling: SamplingConfig,
    visions: Sequence[VisionLimits],
    start_time: float = 0.0,
    end_time: float | None = None,
    *,
    prefix: str = "avf_samples_",
    config: Any = None,
) -> list[SampledFrames]:
    """:func:`sample_vlm_frames` for several endpoints at once (cross-check, § 27.3).

    One probe, one frame source and one decode cache serve every entry of ``visions``; each
    gets its own plan (frame count and size from its own limits), its own selection and its own
    JPEGs (at its own quality, in its own temp dir) and report. With more than one entry the
    source decodes each frame once at the largest size any plan needs and each endpoint's
    :class:`ScaledView` downsizes it; a single entry behaves exactly like the original sampler.
    Result order matches ``visions``; an endpoint whose sampling found nothing gets an empty
    ``SampledFrames`` while the others are unaffected. The caller removes every ``tmp_dir``.
    """
    import cv2

    from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

    def empties() -> list[SampledFrames]:
        return [_empty_frames(v, sampling, start_time) for v in visions]

    if not visions:
        return []
    try:
        ffmpeg, ffprobe = get_ffmpeg_path(config), get_ffprobe_path(config)
    except FileNotFoundError:
        logger.warning("ffmpeg/ffprobe not found; cannot sample frames for the VLM")
        return empties()
    if not os.path.isfile(filepath):
        return empties()
    ranged = start_time > 0 or end_time is not None
    timeline = probe_timeline(
        filepath, ffprobe, start_time if ranged else None, end_time if ranged else None
    )
    if timeline is None:
        return empties()
    times = timeline.times
    gaps = sorted(b - a for a, b in zip(times, times[1:], strict=False) if b > a)
    typical_gap = gaps[len(gaps) // 2] if gaps else 1.0 / 30.0
    # End of the last frame: the container duration when sane (a VFR file's last frame can
    # be held for any length), else the last timestamp plus a typical frame gap.
    total_end = times[-1] + typical_gap
    if timeline.duration > times[-1]:
        total_end = timeline.duration
    start = max(0.0, start_time)
    end = total_end if end_time is None else min(end_time, total_end)
    if end - start <= 0:
        no_range = empties()
        for e in no_range:
            e.report.start_time, e.report.end_time = start, end
            e.report.source_width, e.report.source_height = timeline.width, timeline.height
        return no_range

    # Frames visible during [start, end): the one on screen at `start` (a long VFR hold may
    # begin before the range) through the last one that starts before `end`. These are the
    # "distinct real frames" of the range.
    last = len(times) - 1
    lo = min(max(bisect.bisect_right(times, start + 1e-6) - 1, 0), last)
    hi = min(max(bisect.bisect_left(times, end - 1e-6) - 1, lo), last)
    real_frames = hi - lo + 1
    shared = len(visions) > 1
    if shared:
        source = FFmpegFrameSource(
            filepath, timeline, ffmpeg, lo, hi, cache_budget_bytes=SHARED_CACHE_BUDGET_BYTES
        )
    else:
        source = FFmpegFrameSource(filepath, timeline, ffmpeg, lo, hi)

    motion: float | None = None
    if sampling.balance == "auto" and any(v.max_tiles_per_call is not None for v in visions):
        # tiny decodes: the probe only needs 64-px thumbnails
        source.target_size = (
            THUMB_WIDTH,
            max(1, round(timeline.height * THUMB_WIDTH / timeline.width)),
        )
        motion = motion_score(source, start, end)

    plans = [
        plan_sampling(
            timeline.width, timeline.height, end - start, real_frames, sampling, v, motion
        )
        for v in visions
    ]
    if shared:
        source.target_size = (max(p.width for p in plans), max(p.height for p in plans))

    out: list[SampledFrames] = []
    try:
        for vision, plan in zip(visions, plans, strict=True):
            if shared:
                frames_from: FrameSource = ScaledView(source, (plan.width, plan.height))
            else:
                source.target_size = (plan.width, plan.height)
                frames_from = source
            sel = select_frames(
                frames_from,
                start,
                end,
                plan.target_frames,
                dedupe=sampling.dedupe_consecutive,
                threshold=sampling.dedupe_threshold,
                min_gap=typical_gap / 2,
            )
            sampled = _write_frames(sel, plan, vision, sampling, prefix, start, end, timeline, cv2)
            out.append(sampled)
    except Exception:
        for done in out:
            shutil.rmtree(done.tmp_dir, ignore_errors=True)
        raise
    finally:
        source.clear()
    return out


def _write_frames(
    sel: Selection,
    plan: SamplePlan,
    vision: VisionLimits,
    sampling: SamplingConfig,
    prefix: str,
    start: float,
    end: float,
    timeline: Timeline,
    cv2: Any,
) -> SampledFrames:
    """JPEG-encode one endpoint's selected frames and fill in its report."""
    empty = _empty_frames(vision, sampling, start)
    report = empty.report
    report.start_time, report.end_time = start, end
    report.source_width, report.source_height = timeline.width, timeline.height
    if not sel.kept:
        return empty

    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    paths: list[str] = []
    try:
        for i, cand in enumerate(sel.kept):
            p = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            if cv2.imwrite(p, cand.image, [cv2.IMWRITE_JPEG_QUALITY, vision.jpeg_quality]):
                paths.append(p)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if not paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return empty

    n = len(paths)
    report.frames = n
    report.dropped_duplicates = sel.dropped_duplicates
    report.width, report.height = plan.width, plan.height
    report.tiles_per_frame = plan.tiles_per_frame
    report.est_tiles = None if plan.tiles_per_frame is None else n * plan.tiles_per_frame
    report.est_vision_tokens = plan.est_total_tokens(vision, n)
    report.target_frames = plan.target_frames
    report.attempts = sel.attempts
    report.temporal_pref = plan.temporal_pref
    report.notes = list(plan.notes)
    if n < plan.target_frames:
        report.notes.append(
            f"only {n} distinct frame(s) found for a target of {plan.target_frames}"
        )
    # The images were only needed for the JPEGs; free them before the next endpoint's pass.
    for cand in sel.kept:
        cand.image = None
    return SampledFrames(paths=paths, tmp_dir=tmp_dir, report=report)
