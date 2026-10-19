"""Reusable analysis module (docs/REQUIREMENTS.md § 28).

The stable entry point other code -- built-in stages, custom stages, scripts -- calls instead
of wiring ``VideoAnalyzer`` by hand::

    result = analyze_video(path, config, profile="thorough", scope="scenes")
    result.to_dict()          # JSON-serialisable
    result.analysis           # the underlying VideoAnalysis

``analyze_video`` applies the profile (``core/analysis_profiles.py``), validates the effective
config, runs ``VideoAnalyzer.analyze`` and wraps the outcome in an ``AnalysisResult``. With an
``AnalysisCache`` it returns the cached result when the same file (realpath, size, mtime) is
analysed again under the same effective analysis config, so several stages of one job share
one analysis. ``Pipeline.execute_job`` creates one cache per job and hands it to every stage
(``BaseStage.analysis_cache``); there is deliberately no module-level cache -- a process-wide
one would outlive the files it describes and could hand a stage another job's analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from autovideofixer.config import Config
from autovideofixer.core.analysis import VideoAnalysis, VideoAnalyzer, validate_vlm_scope
from autovideofixer.core.analysis_profiles import apply_profile
from autovideofixer.core.cross_check import CrossCheckConfig
from autovideofixer.core.vlm_outputs import OutputsConfig
from autovideofixer.core.vlm_sampling import SamplingConfig
from autovideofixer.endpoints import resolve_endpoint, resolve_role_endpoint
from autovideofixer.logger import get_logger

logger = get_logger(__name__)

# Keyword overrides ``analyze_video(overrides=...)`` forwards to ``VideoAnalyzer.analyze``
# (the per-call equivalents of the ``avf analyze`` flags).
OVERRIDE_KEYS = frozenset(
    {
        "include_vlm",
        "include_events",
        "prompt_append",
        "prompt_override",
        "system_prompt_override",
        "scene_threshold",
        "min_scene_duration",
        "max_sample_frames",
        "min_sample_frames",
        "sample_interval_sec",
        "vlm_model",
        "vlm_api_url",
        "vlm_endpoint",
    }
)

ProgressCallback = Callable[[str, str], None]


def resolve_analysis_config(config: Config, profile: str | None) -> Config:
    """``config`` with ``profile`` layered onto ``analysis`` (see ``analysis_profiles``)."""
    return apply_profile(config, profile)


def validate_vlm_setup(
    config: Config,
    *,
    vlm_flag: bool | None = None,
    vlm_endpoint: str | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
    interval_sec: float | None = None,
    scope: str | None = None,
) -> None:
    """Fail fast (``ValueError`` naming the key) on a VLM setup that would fail per file.

    Checks the scope, and -- only when the VLM will run (``vlm_flag`` / the config) -- the
    endpoint name(s), cross-check block, sampling and outputs. This is the up-front check
    ``avf analyze`` runs once (exit 2) instead of once per file.
    """
    validate_vlm_scope(
        scope if scope is not None else config.get("analysis", "vlm", "scope", default="video")
    )
    if not (
        vlm_endpoint is not None
        or vlm_flag
        or (vlm_flag is None and config.get("analysis", "vlm", "enabled", default=False))
    ):
        return
    if CrossCheckConfig.from_config(config).enabled:
        # The cross-check lists its own endpoints; --vlm-endpoint replaces the primary.
        if vlm_endpoint is not None:
            resolve_endpoint(config, vlm_endpoint)
    else:
        resolve_role_endpoint(config, "vlm", vlm_endpoint)
    SamplingConfig.from_mapping(
        config.get("analysis", "vlm", "sampling", default={}),
        min_frames=min_frames,
        max_frames=max_frames,
        interval_sec=interval_sec,
    )
    OutputsConfig.from_config(config.get("analysis", "vlm", default={}))


def analysis_to_dict(analysis: VideoAnalysis) -> dict[str, Any]:
    """Full structured result for one file (``avf analyze --json`` and ``AnalysisResult``).

    Everything the table and CSV show, plus the sampling report and every call's stats,
    untruncated. Keys introduced by § 28 (``scope``, ``profile``, per-scene ``vlm``) only
    appear when the run used them, so a default (``video``-scope) run's document is the one
    earlier releases wrote.
    """
    vlm_ran = bool(analysis.vlm_calls or analysis.vlm_sampling)
    per_scene = {r["index"]: r for r in analysis.scene_vlm}
    scenes = []
    for i, sc in enumerate(analysis.scenes):
        entry: dict[str, Any] = {
            "start_time": sc.start_time,
            "end_time": sc.end_time,
            "event_type": sc.event_type,
            "confidence": sc.confidence,
            "description": sc.description,
        }
        if analysis.scope != "video" and i in per_scene:
            rec = per_scene[i]
            entry["vlm"] = {
                k: rec.get(k)
                for k in (
                    "summary",
                    "tags",
                    "objects",
                    "media_type",
                    "rating",
                    "rating_reasoning",
                    "extra",
                    "call",
                    "sampling",
                    "cross_check",
                )
            }
        scenes.append(entry)
    doc: dict[str, Any] = {
        "filepath": analysis.filepath,
        "filename": analysis.filename,
        "duration_sec": analysis.duration,
        "resolution": list(analysis.resolution),
        "framerate_fps": analysis.framerate,
        "video_codec": analysis.video_codec,
        "has_video": analysis.has_video,
        "has_audio": analysis.has_audio,
        "hdr": analysis.is_hdr,
        "scenes_detected": analysis.total_scenes,
        "scenes": scenes,
        "vlm": {
            "summary": analysis.vlm_summary,
            "tags": list(analysis.vlm_tags),
            "objects": list(analysis.vlm_objects),
            "media_type": analysis.media_type,
            "rating": analysis.content_rating,
            "rating_reasoning": analysis.rating_reasoning,
            "extra": dict(analysis.vlm_extra),
        }
        if vlm_ran and analysis.scope != "scenes"
        else None,
        "vlm_sampling": analysis.vlm_sampling,
        "vlm_calls": list(analysis.vlm_calls),
    }
    if analysis.cross_check is not None:
        # § 27: every leg's output + call and the judge's verdict, untruncated.
        doc["cross_check"] = analysis.cross_check
    if analysis.scope != "video" or analysis.profile is not None:
        doc["scope"] = analysis.scope
        doc["profile"] = analysis.profile
    return doc


@dataclass
class AnalysisResult:
    """Outcome of ``analyze_video``: the analysis plus how it was produced."""

    path: str
    profile: str | None
    scope: str
    config_hash: str
    analysis: VideoAnalysis = field(repr=False)
    # True when this object came from the cache (no new work, no new VLM calls).
    cached: bool = False

    @property
    def scene_descriptions(self) -> list[dict[str, Any]]:
        """Per-scene VLM records (empty unless scope included ``scenes``)."""
        return self.analysis.scene_vlm

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (what ``input_info["analysis"]`` and the sidecar carry).

        Times are in the analysed file's timeline -- see ``map_time`` for the output's.
        """
        doc = analysis_to_dict(self.analysis)
        doc["profile"] = self.profile
        doc["scope"] = self.scope
        doc["config_hash"] = self.config_hash
        return doc


class AnalysisCache:
    """Thread-safe cache of ``AnalysisResult`` keyed by file identity + config hash.

    Concurrent callers asking for the same key compute it once (the others wait on a per-key
    lock and get the finished result). A failed computation is not cached.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[tuple[Any, ...], AnalysisResult] = {}
        self._key_locks: dict[tuple[Any, ...], threading.Lock] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
            self._key_locks.clear()

    def get_or_compute(
        self, key: tuple[Any, ...], compute: Callable[[], AnalysisResult]
    ) -> AnalysisResult:
        with self._lock:
            hit = self._results.get(key)
            if hit is not None:
                return _as_cached(hit)
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                hit = self._results.get(key)
            if hit is not None:
                return _as_cached(hit)
            result = compute()
            with self._lock:
                self._results[key] = result
            return result


def _as_cached(result: AnalysisResult) -> AnalysisResult:
    return AnalysisResult(
        path=result.path,
        profile=result.profile,
        scope=result.scope,
        config_hash=result.config_hash,
        analysis=result.analysis,
        cached=True,
    )


def _config_hash(
    eff_config: Config, profile: str | None, scope: str, overrides: Mapping[str, Any]
) -> str:
    payload = {
        "analysis": eff_config.get("analysis", default={}),
        "profile": profile,
        "scope": scope,
        "overrides": dict(overrides),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _file_identity(path: str) -> tuple[str, int, int] | None:
    try:
        st = os.stat(path)
        return os.path.realpath(path), st.st_size, st.st_mtime_ns
    except OSError:
        return None


def analyze_video(
    path: str,
    config: Config,
    *,
    profile: str | None = None,
    scope: str | None = None,
    cache: AnalysisCache | None = None,
    progress_callback: ProgressCallback | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Analyse ``path`` and return an ``AnalysisResult``.

    ``profile``: an analysis profile name (built-in or ``analysis.profiles``); ``scope``:
    ``video`` | ``scenes`` | ``both`` (None = the profile's / ``analysis.vlm.scope``).
    ``overrides``: ``VideoAnalyzer.analyze`` keyword overrides (``OVERRIDE_KEYS``), which win
    over the profile and the config. ``cache``: see ``AnalysisCache``; without one every call
    does the work.

    Raises ``ValueError`` for an unknown profile / invalid analysis config (before any
    expensive work) and whatever probing or scene detection raises; a VLM that is down is
    NOT an error here -- the analysis comes back without VLM data and the failed calls are in
    ``analysis.vlm_calls``.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"Unknown analysis override(s): {', '.join(sorted(unknown))}")
    eff = resolve_analysis_config(config, profile)
    eff_scope = validate_vlm_scope(
        scope if scope is not None else eff.get("analysis", "vlm", "scope", default="video")
    )
    validate_vlm_setup(
        eff,
        vlm_flag=overrides.get("include_vlm"),
        vlm_endpoint=overrides.get("vlm_endpoint"),
        min_frames=overrides.get("min_sample_frames"),
        max_frames=overrides.get("max_sample_frames"),
        interval_sec=overrides.get("sample_interval_sec"),
        scope=eff_scope,
    )
    digest = _config_hash(eff, profile, eff_scope, overrides)

    def compute() -> AnalysisResult:
        analysis = VideoAnalyzer(eff).analyze(
            path, progress_callback=progress_callback, scope=eff_scope, **overrides
        )
        analysis.profile = profile
        return AnalysisResult(
            path=path, profile=profile, scope=eff_scope, config_hash=digest, analysis=analysis
        )

    identity = _file_identity(path) if cache is not None else None
    if cache is None or identity is None:
        return compute()
    return cache.get_or_compute((*identity, digest), compute)


def map_time(seconds: float, input_info: Mapping[str, Any] | None) -> float:
    """An analysis timestamp expressed on the current output's timeline.

    ``input_info["analysis_time_scale"]`` is output-seconds per analysed-second: 1.0 when
    published by the ``analyze`` stage, multiplied by the ``time_scale`` metadata of every
    later stage that rescales time (``speed`` reports ``1 / factor``). Missing = 1.0.
    """
    scale = (input_info or {}).get("analysis_time_scale", 1.0)
    return seconds * float(scale)


def obtain_analysis(
    config: Config,
    stage_name: str,
    stage_config: Mapping[str, Any],
    input_path: str,
    input_info: Mapping[str, Any] | None,
    cache: AnalysisCache | None,
    *,
    need_scenes: bool = False,
) -> tuple[dict[str, Any], float] | None:
    """The analysis a stage should consume, as ``(analysis dict, time_scale)``; None = skip.

    Preference order (documented in docs/DEVELOPER.md "Writing a stage that uses analysis"):

    1. ``input_info["analysis"]`` published by an earlier ``analyze`` stage, with the
       ``analysis_time_scale`` accumulated since (``speed`` ...). Skipped when ``need_scenes``
       and that analysis has no per-scene results (``analyze`` ran with scope ``video``):
       enabling ``analyze`` must not silently disable a scene consumer that would otherwise
       analyse per scene itself, so it falls through to 2.
    2. ``stages.<stage>.analyze_if_missing`` (default true): run ``analyze_video`` now on
       ``input_path`` through the job's cache (profile/scope from the stage's own
       ``profile``/``scope`` keys, else ``stages.analyze.*``); the analysed file is the current
       intermediate, so the time scale is 1.0. ``need_scenes`` upgrades a ``video`` scope to
       ``scenes`` (per-scene descriptions are the point of the caller).
    3. Otherwise, or when the analysis itself fails, one WARNING line and None -- a stage
       that consumes the analysis must not fail the job over it.
    """
    info = input_info or {}
    published = info.get("analysis")
    if isinstance(published, dict):
        if not need_scenes or published.get("scope") in ("scenes", "both"):
            return published, float(info.get("analysis_time_scale", 1.0))
        logger.info(
            "%s: the published analysis has scope %r (no per-scene results); analysing per scene",
            stage_name,
            published.get("scope"),
        )
    if not stage_config.get("analyze_if_missing", True):
        logger.warning(
            "%s: no analysis available (the 'analyze' stage did not run and "
            "stages.%s.analyze_if_missing is false); skipping",
            stage_name,
            stage_name,
        )
        return None
    analyze_cfg = config.get("stages", "analyze", default={}) or {}
    profile = stage_config.get("profile") or analyze_cfg.get("profile")
    scope = stage_config.get("scope") or analyze_cfg.get("scope")
    try:
        if need_scenes and scope is None:
            eff = resolve_analysis_config(config, profile)
            if eff.get("analysis", "vlm", "scope", default="video") == "video":
                scope = "scenes"
        elif need_scenes and scope == "video":
            scope = "scenes"
        result = analyze_video(input_path, config, profile=profile, scope=scope, cache=cache)
    except Exception as e:
        logger.warning("%s: analysis failed (%s: %s); skipping", stage_name, type(e).__name__, e)
        logger.debug("analysis failure detail", exc_info=True)
        return None
    return result.to_dict(), 1.0
