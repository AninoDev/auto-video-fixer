"""Auto Video Fixer - Video analysis utilities.

Provides video file detection, content analysis via VLM (Ollama, OpenAI,
custom API), scene/event detection with highlighting, clip extraction,
and duplicate/similar video detection via perceptual hashing.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from autovideofixer.config import Config

logger = logging.getLogger(__name__)

# Signature for progress callbacks threaded through VideoAnalyzer.analyze():
# progress_callback(phase, detail) -- phase is a short stable machine-readable
# tag (e.g. "probe", "scenes_start", "scene_detection", "vlm_request", "done"),
# detail is a human-readable string for that phase (may be empty).
ProgressCallback = Callable[[str, str], None]

# Supported video extensions
VIDEO_EXTENSIONS: set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ogv",
    ".ts",
    ".vob",
    ".rm",
    ".rmvb",
    ".asf",
    ".f4v",
    ".mxf",
}

# Supported image frame extensions (for still image analysis)
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# VLM prompt for video analysis
_VLM_SYSTEM_PROMPT = (
    "You are a video analysis assistant. Analyze the provided frames from a video "
    "and provide: a brief summary of the content (1-2 sentences), relevant tags "
    "(comma-separated keywords), objects detected (comma-separated), and an "
    "appropriate content rating (G, PG, PG-13, R). Respond in valid JSON format "
    'with keys: "summary", "tags", "objects", "rating".'
)

_VLM_USER_PROMPT = (
    "Analyze these video frames and describe the content. "
    "Provide a summary, tags, detected objects, and content rating."
)


def is_video_file(path: str) -> bool:
    """Check if a file is a video based on extension and/or content."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    """Check if a file is an image based on extension."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def scan_directory(
    directory: str,
    recursive: bool = True,
) -> list[str]:
    """Scan a directory for video files."""
    videos: list[str] = []
    if not os.path.isdir(directory):
        return videos

    if recursive:
        for root, _dirs, files in os.walk(directory):
            for f in sorted(files):
                if f.startswith("."):
                    # Skip hidden files, including orphaned ".avf_*" intermediate
                    # temp files that pipeline.py writes next to the input.
                    continue
                full = os.path.join(root, f)
                if is_video_file(full):
                    videos.append(full)
    else:
        for f in sorted(os.listdir(directory)):
            if f.startswith("."):
                continue
            full = os.path.join(directory, f)
            if is_video_file(full):
                videos.append(full)

    return videos


@dataclass
class SceneEvent:
    """A detected event/scene in a video."""

    start_time: float
    end_time: float
    event_type: str = "scene_change"
    confidence: float = 0.0
    description: str | None = None
    frame_numbers: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of this scene in seconds."""
        return self.end_time - self.start_time


@dataclass
class VideoClip:
    """A clip extracted from a video based on scene boundaries."""

    source_path: str
    start_time: float
    end_time: float
    output_path: str
    scene_index: int = 0
    event_type: str = "scene_change"


@dataclass
class VideoAnalysis:
    """Complete analysis result for a video."""

    filepath: str
    filename: str
    duration: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    framerate: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    is_hdr: bool = False
    video_codec: str = ""
    total_scenes: int = 0
    scenes: list[SceneEvent] = field(default_factory=list)
    vlm_summary: str | None = None
    vlm_tags: list[str] = field(default_factory=list)
    vlm_objects: list[str] = field(default_factory=list)
    content_rating: str | None = None
    similar_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoAnalyzer:
    """Video analysis engine supporting VLM integration and event detection."""

    def __init__(self, config: Config):
        self.config = config
        self._analysis_cache: dict[str, VideoAnalysis] = {}

    def analyze(
        self,
        filepath: str,
        include_vlm: bool | None = None,
        include_events: bool | None = None,
        prompt_append: str | None = None,
        prompt_override: str | None = None,
        system_prompt_override: str | None = None,
        progress_callback: ProgressCallback | None = None,
        scene_threshold: float | None = None,
        min_scene_duration: float | None = None,
        max_sample_frames: int | None = None,
        sample_interval_sec: float | None = None,
        vlm_model: str | None = None,
        vlm_api_url: str | None = None,
    ) -> VideoAnalysis:
        """Perform full analysis on a video file.

        Args:
            filepath: Path to the video file
            include_vlm: Whether to run VLM analysis (None = use config)
            include_events: Whether to detect events (None = use config)
            scene_threshold: Scene-change sensitivity override for this call
                (None = use ``analysis.event_detection.scene_change_threshold``
                from config). See ``_detect_scene_changes`` for the metric.
            min_scene_duration: Minimum scene duration override for this call
                (None = use ``analysis.event_detection.min_scene_duration_sec``).
            prompt_append: Extra text appended to the VLM user prompt (None =
                use ``analysis.vlm.prompt_append`` from config). See
                ``run_vlm_analysis`` for full semantics.
            prompt_override: Replaces the default VLM user prompt entirely
                (None = use ``analysis.vlm.prompt_override`` from config).
            system_prompt_override: Replaces the default VLM system prompt
                entirely (None = use ``analysis.vlm.system_prompt_override``).
            progress_callback: Optional ``callback(phase, detail)`` invoked at
                phase transitions (probing, scene detection start/progress/done,
                VLM sampling/request, done) so callers (e.g. the CLI) can show
                a live status line on slow videos. See ``ProgressCallback``.
            max_sample_frames: VLM sample-frame count override (None = use
                ``analysis.vlm.max_sample_frames``). See ``run_vlm_analysis``.
            sample_interval_sec: VLM frame-sampling interval override in
                seconds (None = use ``analysis.vlm.sample_interval_sec``).
            vlm_model: VLM model name override (None = use
                ``analysis.vlm.model``).
            vlm_api_url: VLM provider URL override (None = use
                ``analysis.vlm.api_url``). See ``run_vlm_analysis`` for the
                ``allow_http`` gate this still goes through.

        Returns:
            VideoAnalysis with all detected information
        """

        def _report(phase: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(phase, detail)

        if filepath in self._analysis_cache:
            return self._analysis_cache[filepath]

        from autovideofixer.core.ffmpeg_utils import probe

        _report("probe", f"Probing {os.path.basename(filepath)}")
        info = probe(filepath)
        _report(
            "probe_done",
            f"{info.resolution[0]}x{info.resolution[1]} @ {info.framerate:.1f}fps, "
            f"{info.duration:.1f}s",
        )
        analysis = VideoAnalysis(
            filepath=filepath,
            filename=info.filename,
            duration=info.duration,
            resolution=info.resolution,
            framerate=info.framerate,
            has_video=info.has_video,
            has_audio=info.has_audio,
            is_hdr=info.is_hdr,
            video_codec=info.video_stream.codec_name if info.video_stream else "",
        )

        if (
            include_events
            if include_events is not None
            else self.config.get("analysis", "event_detection", "enabled", default=True)
        ):
            _report("scenes_start", "Detecting scenes/events")
            analysis.scenes = self.detect_events(
                filepath,
                min_duration=min_scene_duration,
                threshold=scene_threshold,
                progress_callback=progress_callback,
            )
            analysis.total_scenes = len(analysis.scenes)
            _report("scenes_done", f"{analysis.total_scenes} scene(s) detected")

        if (
            include_vlm
            if include_vlm is not None
            else self.config.get("analysis", "vlm", "enabled", default=False)
        ):
            vlm_result = self.run_vlm_analysis(
                filepath,
                prompt_append=prompt_append,
                prompt_override=prompt_override,
                system_prompt_override=system_prompt_override,
                progress_callback=progress_callback,
                max_sample_frames=max_sample_frames,
                sample_interval_sec=sample_interval_sec,
                model=vlm_model,
                api_url=vlm_api_url,
            )
            analysis.vlm_summary = vlm_result.get("summary")
            analysis.vlm_tags = vlm_result.get("tags", [])
            analysis.vlm_objects = vlm_result.get("objects", [])
            analysis.content_rating = vlm_result.get("rating")
            _report("vlm_done", "VLM analysis complete")

        _report("done", "Analysis complete")
        self._analysis_cache[filepath] = analysis
        return analysis

    def detect_events(
        self,
        filepath: str,
        min_duration: float | None = None,
        classify_events: bool = False,
        progress_callback: ProgressCallback | None = None,
        threshold: float | None = None,
    ) -> list[SceneEvent]:
        """Detect scene changes and events in a video.

        Uses frame differencing to find scene boundaries, with optional
        event-type classification via frame-property heuristics (edge
        density/brightness) — not a VLM call. See run_vlm_analysis() for
        actual VLM-based content understanding.

        Args:
            filepath: Path to the video file
            min_duration: Minimum scene duration in seconds (None = use
                ``analysis.event_detection.min_scene_duration_sec`` from config).
                Cuts closer together than this are absorbed into the following
                scene rather than emitted as their own (short) scene -- lower
                this for fast-cut content if legitimate short scenes are being
                merged away.
            classify_events: Whether to heuristically classify event types
            progress_callback: Optional ``callback(phase, detail)`` invoked
                periodically during the frame-differencing pass (phase
                "scene_detection") so a caller can show progress on long videos.
            threshold: Scene-change sensitivity (None = use
                ``analysis.event_detection.scene_change_threshold`` from
                config). See ``_detect_scene_changes`` for what the metric
                measures and what value range is useful.

        Returns:
            List of SceneEvent objects
        """
        if min_duration is None:
            min_duration = self.config.get(
                "analysis", "event_detection", "min_scene_duration_sec", default=2.0
            )

        if threshold is None:
            threshold = self.config.get(
                "analysis", "event_detection", "scene_change_threshold", default=0.15
            )

        # Logged unconditionally (not just under --verbose) since "did my
        # --scene-threshold/--min-scene-duration override actually take
        # effect" is exactly the kind of thing that's otherwise invisible.
        logger.info(
            "scene detection for %s: using threshold=%.3f min_duration=%.2fs",
            filepath,
            threshold,
            min_duration,
        )

        scenes = _detect_scene_changes(
            filepath, threshold, min_duration, progress_callback=progress_callback
        )

        if classify_events and scenes:
            scenes = _classify_events(scenes, filepath, self.config)

        return scenes

    def extract_clip(
        self,
        filepath: str,
        start_time: float,
        end_time: float,
        output_dir: str | None = None,
    ) -> VideoClip | None:
        """Extract a clip from a video file.

        Args:
            filepath: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Output directory (default: temp dir)

        Returns:
            VideoClip on success, None on failure
        """
        from autovideofixer.core.ffmpeg_utils import run_ffmpeg

        if not os.path.isfile(filepath):
            return None

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="avf_clip_")

        stem = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(output_dir, f"{stem}_clip_{start_time:.1f}_{end_time:.1f}.mp4")

        try:
            run_ffmpeg(
                [
                    "-ss",
                    str(start_time),
                    "-i",
                    filepath,
                    "-to",
                    str(end_time - start_time),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-y",
                    output_path,
                ],
                timeout=300,
            )
            return VideoClip(
                source_path=filepath,
                start_time=start_time,
                end_time=end_time,
                output_path=output_path,
            )
        except Exception:
            return None

    def extract_scenes_as_clips(
        self,
        filepath: str,
        scenes: list[SceneEvent],
        output_dir: str | None = None,
    ) -> list[VideoClip]:
        """Extract all detected scenes as separate clip files.

        Args:
            filepath: Source video path
            scenes: List of SceneEvent objects
            output_dir: Output directory

        Returns:
            List of VideoClip objects
        """
        clips: list[VideoClip] = []
        for idx, scene in enumerate(scenes):
            clip = self.extract_clip(
                filepath,
                scene.start_time,
                scene.end_time,
                output_dir=output_dir,
            )
            if clip is not None:
                clip.scene_index = idx
                clip.event_type = scene.event_type
                clips.append(clip)
        return clips

    def run_vlm_analysis(
        self,
        filepath: str,
        sample_interval_sec: float | None = None,
        prompt_append: str | None = None,
        prompt_override: str | None = None,
        system_prompt_override: str | None = None,
        progress_callback: ProgressCallback | None = None,
        max_sample_frames: int | None = None,
        model: str | None = None,
        api_url: str | None = None,
    ) -> dict[str, Any]:
        """Run VLM (Vision Language Model) analysis on video content.

        Samples frames at intervals and sends them to a configured VLM
        provider for content analysis.

        Args:
            filepath: Path to the video file
            sample_interval_sec: Seconds between frame samples (None = use
                ``analysis.vlm.sample_interval_sec`` from config, default 10.0).
                Also settable per-run via ``avf analyze --sample-interval``.
            prompt_append: Extra text appended to the (possibly overridden) user
                prompt, e.g. job-specific context. None falls back to
                ``analysis.vlm.prompt_append`` in config; empty string suppresses it.
            prompt_override: Replaces the default user prompt entirely when
                non-empty. None falls back to ``analysis.vlm.prompt_override``.
                ``prompt_append`` (if any) is still appended after an override.
            system_prompt_override: Replaces the default system prompt entirely
                when non-empty. None falls back to
                ``analysis.vlm.system_prompt_override``.
            progress_callback: Optional ``callback(phase, detail)`` invoked
                before frame sampling ("vlm_sampling") and before the request
                to the provider ("vlm_request") -- useful since the request
                itself can take several seconds with no other feedback.
            max_sample_frames: Max frames sampled and sent to the provider
                (None = use ``analysis.vlm.max_sample_frames``, default 8).
                Also settable per-run via ``avf analyze --max-sample-frames``.
            model: Model name override (None = use ``analysis.vlm.model``,
                default "llava"). Also settable per-run via
                ``avf analyze --vlm-model``.
            api_url: Provider base/endpoint URL override (None = use
                ``analysis.vlm.api_url``). Also settable per-run via
                ``avf analyze --vlm-url``. Flows through to the same
                ``allow_http``/HTTPS-required gate as the config value (see
                ``_call_custom_api``) -- there is deliberately no CLI override
                for ``allow_http`` or ``api_key``; those stay config-file-only.

        Returns:
            Dict with keys: summary, tags, objects, rating
        """

        def _report(phase: str, detail: str = "") -> None:
            if progress_callback is not None:
                progress_callback(phase, detail)

        vlm_config = self.config.get("analysis", "vlm", default={})
        provider = vlm_config.get("provider", "local")
        model = model if model is not None else vlm_config.get("model", "llava")
        api_url = api_url if api_url is not None else vlm_config.get("api_url", "")
        api_key = vlm_config.get("api_key", "")
        max_frames = (
            max_sample_frames
            if max_sample_frames is not None
            else vlm_config.get("max_sample_frames", 8)
        )
        sample_interval_sec = (
            sample_interval_sec
            if sample_interval_sec is not None
            else vlm_config.get("sample_interval_sec", 10.0)
        )

        resolved_append = (
            prompt_append if prompt_append is not None else vlm_config.get("prompt_append", "")
        ) or ""
        resolved_override = (
            prompt_override
            if prompt_override is not None
            else vlm_config.get("prompt_override", "")
        ) or ""
        resolved_system_override = (
            system_prompt_override
            if system_prompt_override is not None
            else vlm_config.get("system_prompt_override", "")
        ) or ""

        system_prompt = resolved_system_override or _VLM_SYSTEM_PROMPT
        user_prompt = resolved_override or _VLM_USER_PROMPT
        if resolved_append:
            user_prompt = f"{user_prompt}\n\n{resolved_append}"

        _report("vlm_sampling", f"Extracting up to {max_frames} sample frame(s)")
        frames = _extract_sample_frames(filepath, sample_interval_sec, max_frames=max_frames)
        if not frames:
            return {"summary": "", "tags": [], "objects": []}

        # All sample frames live in one temp dir created by _extract_sample_frames;
        # it must be cleaned up here, after the provider has read the frames into
        # base64, not inside _extract_sample_frames itself (which would delete the
        # files before they're ever read).
        tmp_dir = os.path.dirname(frames[0])
        try:
            _report(
                "vlm_request",
                f"Sending {len(frames)} frame(s) to {provider} VLM provider (model={model})",
            )
            return _dispatch_vlm(
                frames,
                provider,
                model,
                api_url,
                api_key,
                system_prompt,
                user_prompt,
                allow_http=bool(vlm_config.get("allow_http", False)),
                max_tokens=int(vlm_config.get("max_tokens", 1024)),
                context=filepath,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def run_vlm_analysis_for_scene(
        self,
        filepath: str,
        scene: "SceneEvent",
        prompt_append: str | None = None,
        prompt_override: str | None = None,
        system_prompt_override: str | None = None,
        max_sample_frames: int | None = None,
        sample_interval_sec: float | None = None,
        model: str | None = None,
        api_url: str | None = None,
    ) -> dict[str, Any]:
        """Run VLM analysis over only the frames within one scene's time range.

        Used by scene-based processing (see core/scenes.py): samples fewer frames
        for short scenes -- ``N = min(max_sample_frames, ceil(scene_duration /
        sample_interval_sec))``, floored at 1 for any non-empty scene -- rather than
        always sending ``max_sample_frames`` regardless of how short the scene is.
        Shares the same provider config/dispatch as ``run_vlm_analysis`` (VLM
        connection layer, HTTPS/allow_http gate, prompt handling); only the frame
        sampling is scene-scoped.

        Returns the same dict shape as ``run_vlm_analysis``: summary/tags/objects/
        rating (empty on any failure -- this method never raises).
        """
        import math

        vlm_config = self.config.get("analysis", "vlm", default={})
        provider = vlm_config.get("provider", "local")
        model = model if model is not None else vlm_config.get("model", "llava")
        api_url = api_url if api_url is not None else vlm_config.get("api_url", "")
        api_key = vlm_config.get("api_key", "")
        max_frames_cfg = (
            max_sample_frames
            if max_sample_frames is not None
            else vlm_config.get("max_sample_frames", 8)
        )
        interval = (
            sample_interval_sec
            if sample_interval_sec is not None
            else vlm_config.get("sample_interval_sec", 10.0)
        )

        duration = max(scene.duration, 0.0)
        if duration <= 0:
            return {"summary": "", "tags": [], "objects": [], "rating": None}
        scene_max_frames = max(1, min(max_frames_cfg, math.ceil(duration / max(interval, 0.1))))

        resolved_append = (
            prompt_append if prompt_append is not None else vlm_config.get("prompt_append", "")
        ) or ""
        resolved_override = (
            prompt_override
            if prompt_override is not None
            else vlm_config.get("prompt_override", "")
        ) or ""
        resolved_system_override = (
            system_prompt_override
            if system_prompt_override is not None
            else vlm_config.get("system_prompt_override", "")
        ) or ""
        system_prompt = resolved_system_override or _VLM_SYSTEM_PROMPT
        user_prompt = resolved_override or _VLM_USER_PROMPT
        if resolved_append:
            user_prompt = f"{user_prompt}\n\n{resolved_append}"

        frames = _extract_sample_frames_range(
            filepath, scene.start_time, scene.end_time, interval, max_frames=scene_max_frames
        )
        if not frames:
            return {"summary": "", "tags": [], "objects": [], "rating": None}

        tmp_dir = os.path.dirname(frames[0])
        try:
            return _dispatch_vlm(
                frames,
                provider,
                model,
                api_url,
                api_key,
                system_prompt,
                user_prompt,
                allow_http=bool(vlm_config.get("allow_http", False)),
                max_tokens=int(vlm_config.get("max_tokens", 1024)),
                context=f"{filepath} scene[{scene.start_time:.1f}-{scene.end_time:.1f}]",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def find_similar(
        self,
        filepath: str,
        candidates: list[str],
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find similar/duplicate videos from a list of candidates.

        Uses perceptual hashing to compare videos.

        Args:
            filepath: Reference video to compare
            candidates: List of candidate video paths
            threshold: Similarity threshold (0-1, higher = more similar)

        Returns:
            List of (filepath, similarity_score) tuples, sorted by similarity
        """
        if threshold is None:
            threshold = self.config.get(
                "analysis", "duplicate_detection", "similarity_threshold", default=0.95
            )

        ref_hash = compute_video_hash(filepath)
        results: list[tuple[str, float]] = []

        for candidate in candidates:
            if candidate == filepath:
                continue
            cand_hash = compute_video_hash(candidate)
            similarity = hash_similarity(ref_hash, cand_hash)
            if similarity >= threshold:
                results.append((candidate, similarity))

        return sorted(results, key=lambda x: -x[1])

    def find_duplicates(
        self,
        files: list[str],
        threshold: float | None = None,
    ) -> list[list[str]]:
        """Find all duplicate videos in a batch.

        Computes perceptual hashes for all files, then groups files
        that are above the similarity threshold of each other.

        Args:
            files: List of video file paths
            threshold: Similarity threshold (0-1, higher = more similar)

        Returns:
            List of groups, where each group contains paths of duplicate files.
            Only groups with 2+ members are returned.
        """
        if threshold is None:
            threshold = self.config.get(
                "analysis", "duplicate_detection", "similarity_threshold", default=0.95
            )

        # Compute hashes for all files
        hashes: dict[str, str] = {}
        for f in files:
            h = compute_video_hash(f)
            if h:
                hashes[f] = h

        # Union-find over the similarity graph so each file lands in exactly
        # one cluster, instead of a per-pair min()-keyed dict that can emit
        # overlapping groups (e.g. both [A,B,C] and [B,C]) for 3+ mutually
        # similar files.
        parent: dict[str, str] = {f: f for f in hashes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        paths = list(hashes.keys())
        for i, path_a in enumerate(paths):
            for path_b in paths[i + 1 :]:
                if hash_similarity(hashes[path_a], hashes[path_b]) >= threshold:
                    union(path_a, path_b)

        clusters: dict[str, list[str]] = {}
        for f in paths:
            clusters.setdefault(find(f), []).append(f)

        result = [sorted(members) for members in clusters.values() if len(members) >= 2]
        return sorted(result, key=lambda g: -len(g))

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._analysis_cache.clear()


# ─── VLM Providers ─────────────────────────────────────────────────


def _frames_to_base64(frame_paths: list[str]) -> list[str]:
    """Convert frame image paths to base64-encoded strings."""
    encoded: list[str] = []
    for path in frame_paths:
        try:
            with open(path, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("utf-8"))
        except OSError:
            continue
    return encoded


def _parse_vlm_response(response_text: str) -> dict[str, Any]:
    """Parse a VLM JSON response into analysis results."""
    import json

    original = response_text.strip()
    # Strip markdown code fences if present. Anchored regex (not positional
    # line-slicing) so a response truncated before the closing fence doesn't
    # get reduced to an empty string.
    text = re.sub(r"^```(?:json)?\s*\n?", "", original)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        data = json.loads(text)
        return {
            "summary": data.get("summary", ""),
            "tags": [t.strip() for t in data.get("tags", "").split(",") if t.strip()]
            if isinstance(data.get("tags"), str)
            else data.get("tags", []),
            "objects": [o.strip() for o in data.get("objects", "").split(",") if o.strip()]
            if isinstance(data.get("objects"), str)
            else data.get("objects", []),
            "rating": data.get("rating"),
        }
    except json.JSONDecodeError, AttributeError:
        # Fallback: treat the original (pre-fence-stripping) response as the
        # summary, so a truncated/malformed fenced response still degrades to
        # real content instead of the empty string left by failed stripping.
        return {"summary": original, "tags": [], "objects": [], "rating": None}


def _call_ollama(
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_b64_list: list[str],
    max_tokens: int = 1024,
) -> str:
    """Send a request to an Ollama API endpoint.

    Args:
        api_url: Ollama API base URL (e.g., http://localhost:11434)
        model: Model name
        system_prompt: System message
        user_prompt: User message text
        image_b64_list: Base64-encoded image strings

    Returns:
        Response text from the model
    """
    import json

    # Ollama's /api/chat expects images as a plain base64 list on the message
    # object itself (message["images"]), not as OpenAI-style content blocks.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": image_b64_list},
        ],
        "stream": False,
        "format": "json",
        "options": {"num_predict": max_tokens},
    }

    try:
        import urllib.request

        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "")
    except Exception:
        logger.warning("Ollama VLM request to %s failed", api_url, exc_info=True)
        return ""


def _call_openai(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_b64_list: list[str],
    max_tokens: int = 1024,
) -> str:
    """Send a request to the OpenAI Vision API.

    Args:
        api_key: OpenAI API key
        model: Model name (e.g., gpt-4o)
        system_prompt: System message
        user_prompt: User message text
        image_b64_list: Base64-encoded image strings

    Returns:
        Response text from the model
    """
    import json

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for b64 in image_b64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        logger.warning("OpenAI VLM request failed (model=%s)", model, exc_info=True)
        return ""


def _call_custom_api(
    api_key: str,
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_b64_list: list[str],
    allow_http: bool = False,
    max_tokens: int = 1024,
) -> str:
    """Send a request to a custom API endpoint.

    Uses the OpenAI-compatible chat completions format.

    Args:
        api_key: API key for authentication
        api_url: Custom API base URL
        model: Model name
        system_prompt: System message
        user_prompt: User message text
        image_b64_list: Base64-encoded image strings

    Returns:
        Response text from the API
    """
    import json
    from urllib.parse import urlparse

    # api_url is config-supplied (trusted-input trust model: it comes from the
    # user's own config.yaml or a preset they chose to apply, not from network
    # input). Still, require https except for loopback so a shared/untrusted
    # config/preset can't silently point frame uploads + credentials at an
    # arbitrary internal address over plaintext.
    # analysis.vlm.allow_http opts out for e.g. a LAN inference box that only
    # speaks plain HTTP; frames and the API key then travel unencrypted, which
    # is the user's call to make for their own network.
    parsed = urlparse(api_url)
    if (
        parsed.scheme != "https"
        and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        and not allow_http
    ):
        logger.warning(
            "Refusing non-https custom VLM API URL for non-loopback host: %s "
            "(set analysis.vlm.allow_http: true in config.yaml to permit plain "
            "HTTP, e.g. for a server on your own LAN)",
            api_url,
        )
        return ""

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for b64 in image_b64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            api_url.rstrip("/"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        logger.warning("Custom API VLM request to %s failed", api_url, exc_info=True)
        return ""


def _dispatch_vlm(
    frames: list[str],
    provider: str,
    model: str,
    api_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    allow_http: bool = False,
    context: str = "",
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Shared provider dispatch for both whole-video and per-scene VLM analysis.

    Extracted out of ``run_vlm_analysis`` so ``run_vlm_analysis_for_scene`` doesn't
    duplicate the provider routing/error handling. Never raises -- any exception
    is logged at WARNING and degrades to an empty result.
    """
    try:
        if provider in ("local", "ollama"):
            return _run_ollama_vlm(
                frames, model, api_url, system_prompt, user_prompt, max_tokens=max_tokens
            )
        elif provider == "openai":
            return _run_openai_vlm(
                frames, api_key, model, system_prompt, user_prompt, max_tokens=max_tokens
            )
        elif provider == "api":
            return _run_api_vlm(
                frames,
                api_key,
                api_url,
                model,
                system_prompt,
                user_prompt,
                allow_http=allow_http,
                max_tokens=max_tokens,
            )
        else:
            return {"summary": "", "tags": [], "objects": [], "rating": None}
    except Exception:
        logger.warning("VLM analysis failed for %r (provider=%s)", context, provider, exc_info=True)
        return {"summary": "", "tags": [], "objects": [], "rating": None}


def _run_ollama_vlm(
    frames: list[str],
    model: str,
    api_url: str,
    system_prompt: str = _VLM_SYSTEM_PROMPT,
    user_prompt: str = _VLM_USER_PROMPT,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run analysis using Ollama."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    base_url = api_url or "http://localhost:11434"
    response = _call_ollama(
        base_url,
        model,
        system_prompt,
        user_prompt,
        image_b64,
        max_tokens=max_tokens,
    )
    return _parse_vlm_response(response)


def _run_openai_vlm(
    frames: list[str],
    api_key: str,
    model: str,
    system_prompt: str = _VLM_SYSTEM_PROMPT,
    user_prompt: str = _VLM_USER_PROMPT,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run analysis using OpenAI Vision API."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    response = _call_openai(
        api_key, model, system_prompt, user_prompt, image_b64, max_tokens=max_tokens
    )
    return _parse_vlm_response(response)


def _run_api_vlm(
    frames: list[str],
    api_key: str,
    api_url: str,
    model: str,
    system_prompt: str = _VLM_SYSTEM_PROMPT,
    user_prompt: str = _VLM_USER_PROMPT,
    allow_http: bool = False,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run analysis using a custom API endpoint (OpenAI-compatible)."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    response = _call_custom_api(
        api_key,
        api_url,
        model,
        system_prompt,
        user_prompt,
        image_b64,
        allow_http=allow_http,
        max_tokens=max_tokens,
    )
    return _parse_vlm_response(response)


# ─── Scene Detection ───────────────────────────────────────────────


def _select_near_misses(
    near_misses: list[tuple[float, float]], limit: int = 10
) -> list[tuple[float, float]]:
    """Return the `limit` highest-scoring entries from a list of (time, diff_score)
    near-miss candidates, highest score first.

    Pulled out of `_detect_scene_changes` as a small pure function so the
    "which near-misses get reported" selection logic is unit-testable without
    decoding real video frames -- see TestNearMissTracking in
    tests/unit/test_scene_detection.py.
    """
    return sorted(near_misses, key=lambda item: -item[1])[:limit]


def _detect_scene_changes(
    filepath: str,
    threshold: float,
    min_duration_sec: float,
    progress_callback: ProgressCallback | None = None,
) -> list[SceneEvent]:
    """Detect scene changes using frame differencing.

    The metric: each consecutive pair of frames is converted to grayscale,
    downscaled to 320x180, and compared with ``cv2.absdiff`` (per-pixel
    absolute luma difference). ``diff_score`` is the mean of that difference
    image divided by 255, i.e. the *average fractional luma change per pixel*
    between the two frames, in [0, 1]. A hard cut between visually distinct
    shots typically scores ~0.15-1.0; static or slowly-panning content within
    a single shot typically scores well under 0.05 (measured on a synthetic
    ground-truth clip of 12 visually distinct 5s segments -- see
    tests/unit/test_scene_detection.py's TestSceneDetectionCalibration -- max
    within-segment score 0.040, min cut score 0.184). ``threshold`` (default
    0.15, ``analysis.event_detection.scene_change_threshold``) is the cutoff
    above which a frame pair is called a cut:
      - Lower (e.g. 0.05-0.10): more sensitive -- catches subtler cuts (soft
        transitions, similar-toned shots) but risks false positives from
        camera motion, flicker, or compression noise in busy real-world
        footage.
      - Higher (e.g. 0.3-0.5): only very hard, high-contrast cuts register;
        gradual/soft cuts and same-toned shot changes are missed (this was
        the previous default of 0.3, which under-detected on real footage --
        see CHANGELOG).
    ``min_duration_sec`` (default 2.0,
    ``analysis.event_detection.min_scene_duration_sec``) then absorbs any cut
    that would produce a scene shorter than this into the following scene --
    it doesn't affect whether a cut is *detected*, only whether it's allowed
    to start a new scene on its own. For content with legitimately short
    scenes (fast cuts under ~2s apart), lower this too or short scenes will
    be silently merged into their neighbor even though the underlying cuts
    were both correctly detected.
    """
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return []

    scenes: list[SceneEvent] = []
    prev_frame: Any = None
    scene_start = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    current_time = 0.0

    # "Near-miss" tracking: frame pairs that scored close to (but under) the
    # threshold, i.e. plausible cuts that a slightly lower threshold would
    # catch. Floor of threshold/4 keeps ordinary within-shot noise/motion out
    # of the list -- see the "near-miss scores" log line below, which exists
    # specifically to answer "would lowering the threshold find more cuts,
    # and where?" without having to rerun detection at several thresholds.
    near_miss_floor = threshold / 4.0
    near_misses: list[tuple[float, float]] = []  # (time, diff_score), diff_score < threshold
    # Raw diff_score of each actually-detected cut (NOT SceneEvent.confidence --
    # the final trailing SceneEvent after the loop gets a hardcoded confidence
    # of 0.5 that isn't a real score, so this list is tracked separately to
    # keep the score summary below accurate).
    cut_scores: list[float] = []

    # Report progress roughly every 5% of the video (or every 200 frames if
    # the container doesn't report a usable frame count) so a caller watching
    # a long scan doesn't see a silent hang.
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    report_interval = max(int(total_frames / 20), 1) if total_frames > 0 else 200

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prefer the container's actual presentation timestamp so scene
        # boundaries stay accurate on variable-frame-rate sources; fall back
        # to a constant-fps estimate for backends that don't report POS_MSEC.
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time = msec / 1000.0 if msec > 0 else (frame_idx / fps if fps > 0 else 0.0)

        # Convert to grayscale and resize for faster comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            diff_score = float(diff.mean()) / 255.0

            if diff_score > threshold:
                # Scene change detected. Only close out and start a new scene
                # when the current one is long enough; otherwise keep
                # accumulating so a too-short cut is absorbed into the next
                # scene instead of leaving an unaccounted time gap.
                if current_time - scene_start >= min_duration_sec:
                    scenes.append(
                        SceneEvent(
                            start_time=scene_start,
                            end_time=current_time,
                            confidence=min(diff_score, 1.0),
                        )
                    )
                    cut_scores.append(diff_score)
                    scene_start = current_time
            elif diff_score > near_miss_floor:
                near_misses.append((current_time, diff_score))

        prev_frame = gray
        frame_idx += 1

        if progress_callback is not None and frame_idx % report_interval == 0:
            if total_frames > 0:
                pct = min(100, int(frame_idx / total_frames * 100))
                progress_callback(
                    "scene_detection", f"{pct}% (frame {frame_idx}/{int(total_frames)})"
                )
            else:
                progress_callback("scene_detection", f"frame {frame_idx}")

    cap.release()

    # Final scene: subject to the same min-duration filter as every other
    # scene, except when it's the only scene detected (i.e. no cuts were
    # found at all) — a short whole video should still yield one scene.
    if scene_start < current_time:
        final_duration = current_time - scene_start
        if final_duration >= min_duration_sec or not scenes:
            scenes.append(
                SceneEvent(
                    start_time=scene_start,
                    end_time=current_time,
                    confidence=0.5,
                )
            )

    if cut_scores:
        logger.info(
            "scene detection for %s: %d cut(s) found (threshold=%.3f), "
            "cut score min=%.3f median=%.3f max=%.3f",
            filepath,
            len(cut_scores),
            threshold,
            min(cut_scores),
            statistics.median(cut_scores),
            max(cut_scores),
        )
    else:
        logger.info(
            "scene detection for %s: no cuts found above threshold %.3f", filepath, threshold
        )

    if near_misses:
        top_near_misses = _select_near_misses(near_misses)
        near_miss_str = ", ".join(f"{score:.3f} @ {t:.1f}s" for t, score in top_near_misses)
        logger.info(
            "near-miss scores for %s below threshold %.3f (above %.3f): %s",
            filepath,
            threshold,
            near_miss_floor,
            near_miss_str,
        )

    return scenes


def _classify_events(
    scenes: list[SceneEvent],
    filepath: str,
    config: Config,
) -> list[SceneEvent]:
    """Classify scene events using frame-property heuristics.

    Samples a frame from the middle of each scene and classifies it via
    simple image heuristics (edge density, brightness) — NOT a VLM call.
    This is a fast, offline pre-classification; real content-understanding
    classification is provided separately by VideoAnalyzer.run_vlm_analysis.
    """
    import cv2

    for scene in scenes:
        if scene.confidence < 0.2:
            scene.event_type = "talking_head"
            scene.description = "Low-motion scene"
            continue

        # Sample a frame from the middle of the scene
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            continue
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        midpoint_sec = (scene.start_time + scene.end_time) / 2
        target_frame = int(midpoint_sec * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            scene.event_type = "scene_change"
            continue

        # Simple heuristic classification based on frame properties
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(edges.sum()) / (gray.shape[0] * gray.shape[1] * 255)
        brightness = float(gray.mean())

        if edge_ratio > 0.15 and brightness < 100:
            scene.event_type = "action"
            scene.description = "High-motion dark scene"
        elif edge_ratio < 0.05:
            scene.event_type = "landscape"
            scene.description = "Low-detail scenic shot"
        elif brightness > 200:
            scene.event_type = "transition"
            scene.description = "Bright scene (possible fade)"
        else:
            scene.event_type = "scene_change"
            scene.description = "Standard scene"

        scene.confidence = min(scene.confidence, 1.0)

    return scenes


# ─── Frame Extraction ──────────────────────────────────────────────


def _extract_sample_frames(
    filepath: str,
    interval_sec: float,
    max_frames: int = 20,
) -> list[str]:
    """Extract evenly-spaced sample frames from a video.

    Samples frames at regular intervals to provide representative
    visual content for VLM analysis.

    Args:
        filepath: Path to the video file
        interval_sec: Seconds between samples
        max_frames: Maximum number of frames to extract

    Returns:
        List of paths to extracted frame images
    """
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * interval_sec))
    frames: list[str] = []
    frame_idx = 0

    tmp_dir = tempfile.mkdtemp(prefix="avf_samples_")

    try:
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(frame_path)

            frame_idx += frame_interval
    finally:
        cap.release()

    return frames


def _extract_sample_frames_range(
    filepath: str,
    start_time: float,
    end_time: float,
    interval_sec: float,
    max_frames: int = 8,
) -> list[str]:
    """Extract evenly-spaced sample frames from within [start_time, end_time).

    Same mechanics as ``_extract_sample_frames`` (frames written as JPEGs to a
    fresh temp dir, caller is responsible for cleanup) but scoped to a scene's
    time range instead of the whole file -- used by per-scene VLM sampling.
    Always samples at least one frame (the scene's start) if the range is
    non-empty and the video can be opened, even if ``interval_sec`` alone would
    have produced zero samples for a very short scene.
    """
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = max(end_time - start_time, 0.0)
    if duration <= 0 or max_frames <= 0:
        cap.release()
        return []

    # Evenly space `max_frames` samples across the scene (at least 1), rather
    # than stepping by interval_sec from the start -- for a short scene,
    # interval_sec-stepping could land every sample on (near) the same frame.
    n_samples = max(1, min(max_frames, int(duration / max(interval_sec, 0.1)) + 1))
    if n_samples == 1:
        sample_times = [start_time + duration / 2.0]
    else:
        step = duration / n_samples
        sample_times = [start_time + i * step for i in range(n_samples)]

    tmp_dir = tempfile.mkdtemp(prefix="avf_scene_samples_")
    frames: list[str] = []
    try:
        for i, t in enumerate(sample_times):
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(frame_path)
    finally:
        cap.release()

    return frames


# ─── Scene Content Coordinator (drop non-content scenes) ───────────


_COORDINATOR_SYSTEM_PROMPT = (
    "You are a video-editing assistant. You will be given a list of scenes from a single "
    "video, each with a per-scene visual description produced by a separate vision model. "
    "Determine the video's main subject/purpose, then identify any scenes that are NOT part "
    "of that main content -- for example a 'like and subscribe' interstitial, a channel "
    "branding/promo bumper, or an unrelated clip spliced into the middle of the video. "
    'Respond in valid JSON with keys: "main_content_summary" (string), "drop" (array of '
    'scene index integers to drop), "reasons" (object mapping each dropped index, as a '
    'string, to a short reason). If no scenes should be dropped, return an empty "drop" '
    "array. Do not drop a scene unless you are confident it is not part of the main content."
)


def _build_coordinator_prompt(scene_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "Scenes (index, time range, duration, VLM description):",
    ]
    for s in scene_summaries:
        lines.append(
            f"- index={s['index']} t={s['start_time']:.1f}s-{s['end_time']:.1f}s "
            f"duration={s['duration']:.1f}s summary={s.get('summary', '') or '(none)'} "
            f"tags={', '.join(s.get('tags', []) or [])}"
        )
    lines.append(
        '\nReturn JSON: {"main_content_summary": ..., "drop": [indices], '
        '"reasons": {"<index>": "..."}}'
    )
    return "\n".join(lines)


def _parse_coordinator_response(response_text: str, num_scenes: int) -> dict[str, Any] | None:
    """Parse the coordinator LLM's JSON response.

    Returns None (fail-open signal to the caller) on any parse failure or if the
    response isn't the expected shape -- e.g. "drop" isn't a list, or contains an
    out-of-range index. Never raises.
    """
    import json

    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return None

    if not isinstance(data, dict):
        return None

    drop = data.get("drop", [])
    if not isinstance(drop, list):
        return None

    valid_drop: list[int] = []
    for idx in drop:
        if isinstance(idx, bool) or not isinstance(idx, int):
            return None
        if idx < 0 or idx >= num_scenes:
            return None
        valid_drop.append(idx)

    reasons = data.get("reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}

    return {
        "main_content_summary": data.get("main_content_summary", ""),
        "drop": valid_drop,
        "reasons": reasons,
    }


def run_scene_coordinator(
    scene_summaries: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    """Run the coordinating text-LLM pass over all per-scene VLM summaries.

    ``scene_summaries``: list of dicts with keys index/start_time/end_time/
    duration/summary/tags (see ``_build_coordinator_prompt``).

    Config: ``analysis.llm`` (provider/model/api_key/api_url/allow_http) -- see
    ``Config.DEFAULTS``. Reuses the same provider dispatch as per-frame VLM calls
    (``_dispatch_vlm`` with an empty image list, i.e. a text-only request), so the
    same allow_http/HTTPS gate applies to a non-loopback ``api_url``.

    Failure mode (R1.7 / the scene-drop spec's "fail open, never fail closed to
    drop everything"): on ANY failure -- request error, timeout, malformed/
    unparseable response, an out-of-range or non-integer index in "drop" -- this
    returns ``{"drop": [], "reasons": {}, "failed": True, ...}``, i.e. keep every
    scene. This function never raises and never returns a "drop everything"
    result as a side effect of a parse failure.
    """
    llm_config = config.get("analysis", "llm", default={})
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "llama3")
    api_url = llm_config.get("api_url", "")
    api_key = llm_config.get("api_key", "")
    allow_http = bool(llm_config.get("allow_http", False))
    max_tokens = int(llm_config.get("max_tokens", 4096))

    if not scene_summaries:
        return {"drop": [], "reasons": {}, "main_content_summary": "", "failed": False}

    user_prompt = _build_coordinator_prompt(scene_summaries)

    # Note: this calls the raw (unparsed) provider functions directly rather than
    # _dispatch_vlm -- _dispatch_vlm runs the response through _parse_vlm_response,
    # which expects the summary/tags/objects/rating shape, not the coordinator's
    # own main_content_summary/drop/reasons shape (see _parse_coordinator_response).
    raw_text = _call_coordinator_provider(
        provider,
        model,
        api_url,
        api_key,
        _COORDINATOR_SYSTEM_PROMPT,
        user_prompt,
        allow_http,
        max_tokens=max_tokens,
    )
    if raw_text is None:
        logger.warning(
            "Scene coordinator LLM call failed or was unavailable (provider=%s); "
            "failing open -- keeping all %d scene(s)",
            provider,
            len(scene_summaries),
        )
        return {
            "drop": [],
            "reasons": {},
            "main_content_summary": "",
            "failed": True,
        }

    parsed = _parse_coordinator_response(raw_text, len(scene_summaries))
    if parsed is None:
        logger.warning(
            "Scene coordinator LLM response could not be parsed (provider=%s); "
            "failing open -- keeping all %d scene(s). Raw response: %.500r",
            provider,
            len(scene_summaries),
            raw_text,
        )
        return {
            "drop": [],
            "reasons": {},
            "main_content_summary": "",
            "failed": True,
        }

    if parsed["drop"]:
        logger.warning(
            "Scene coordinator: dropping %d of %d scene(s): %s",
            len(parsed["drop"]),
            len(scene_summaries),
            ", ".join(
                f"#{idx} (t={scene_summaries[idx]['start_time']:.1f}-"
                f"{scene_summaries[idx]['end_time']:.1f}s): "
                f"{parsed['reasons'].get(str(idx), parsed['reasons'].get(idx, 'no reason given'))}"
                for idx in parsed["drop"]
            ),
        )
    else:
        logger.info(
            "Scene coordinator: keeping all %d scene(s) (no non-content scenes flagged)",
            len(scene_summaries),
        )

    parsed["failed"] = False
    return parsed


def _call_coordinator_provider(
    provider: str,
    model: str,
    api_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    allow_http: bool,
    max_tokens: int = 4096,
) -> str | None:
    """Call the raw (unparsed) text-completion provider for the coordinator pass.

    Returns the raw response text, or None on any failure (network error,
    exception, refused non-HTTPS URL). Unlike ``_run_*_vlm``, this returns the
    provider's raw text instead of running it through ``_parse_vlm_response``
    (the coordinator has its own response shape/parser, see
    ``_parse_coordinator_response``).
    """
    try:
        if provider in ("local", "ollama"):
            base_url = api_url or "http://localhost:11434"
            text = _call_ollama(
                base_url, model, system_prompt, user_prompt, [], max_tokens=max_tokens
            )
        elif provider == "openai":
            text = _call_openai(
                api_key, model, system_prompt, user_prompt, [], max_tokens=max_tokens
            )
        elif provider == "api":
            text = _call_custom_api(
                api_key,
                api_url,
                model,
                system_prompt,
                user_prompt,
                [],
                allow_http=allow_http,
                max_tokens=max_tokens,
            )
        else:
            return None
    except Exception:
        logger.warning("Scene coordinator provider call raised", exc_info=True)
        return None
    return text or None


# ─── Auto-crop VLM assist (docs/REQUIREMENTS.md feature 3) ─────────

_CROP_VLM_SYSTEM_PROMPT = (
    "You help an automated video-cropping tool decide whether it is safe to "
    "crop away the border region of a frame. You will be shown two images of "
    "the same video frame: the plain frame, and the same frame with a red "
    "rectangle drawn around the region the tool proposes to KEEP. Everything "
    "outside the red rectangle would be cropped away and permanently lost."
)

_CROP_VLM_USER_PROMPT = (
    "Look at the region OUTSIDE the red rectangle in the second image. Does it "
    "contain meaningful content that belongs to the video -- a logo, "
    "watermark, caption, or other text/graphic -- or is it only black/empty "
    "space (letterboxing/pillarboxing)? "
    'Respond with ONLY a JSON object: {"content_outside": true|false, '
    '"reason": "brief explanation"}.'
)


def _parse_crop_vlm_response(response_text: str) -> dict[str, Any] | None:
    """Parse the crop VLM check's response.

    Tries strict JSON first (optionally fenced); falls back to a tolerant
    yes/no scan of the raw text if the response isn't valid JSON, so a model
    that ignores the "JSON only" instruction still yields a usable answer
    instead of forcing a hard failure. Returns None only when neither parse
    path can extract a clear answer -- the caller treats that as fail-open.
    """
    import json

    original = response_text.strip()
    if not original:
        return None

    text = re.sub(r"^```(?:json)?\s*\n?", "", original)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "content_outside" in data:
            return {
                "content_outside": bool(data.get("content_outside")),
                "reason": str(data.get("reason", "")),
            }
    except json.JSONDecodeError, TypeError:
        pass

    # Tolerant fallback: no valid/expected-shape JSON -- look for a clear
    # yes/no signal near the start of the raw response.
    lowered = original.lower()[:80]
    if re.search(r"\b(true|yes)\b", lowered):
        return {"content_outside": True, "reason": original[:300]}
    if re.search(r"\b(false|no)\b", lowered):
        return {"content_outside": False, "reason": original[:300]}
    return None


def run_crop_vlm_check(
    full_frame_path: str,
    boxed_frame_path: str,
    config: Config,
) -> dict[str, Any]:
    """Ask the configured VLM whether meaningful content lies outside a
    proposed crop box (see ``CropStage._run_vlm_check`` in
    ``core/stages/crop.py``).

    Uses ``analysis.vlm`` (provider/model/api_key/api_url/allow_http) --
    the same connection layer and HTTPS/allow_http gate as
    ``VideoAnalyzer.run_vlm_analysis``. This is an internal, fixed prompt
    pair (system + user) -- ``analysis.vlm.prompt_override``/
    ``prompt_append`` do NOT apply here.

    Fails open on any error (unreachable endpoint, exception, unparseable
    response): returns ``content_outside: False`` with ``failed: True`` and a
    ``reason`` explaining why, so the caller can log a WARNING and proceed
    with the plain cropdetect result rather than block on VLM availability.
    Never raises.
    """
    vlm_config = config.get("analysis", "vlm", default={})
    provider = vlm_config.get("provider", "local")
    model = vlm_config.get("model", "llava")
    api_url = vlm_config.get("api_url", "")
    api_key = vlm_config.get("api_key", "")
    allow_http = bool(vlm_config.get("allow_http", False))

    images_b64 = _frames_to_base64([full_frame_path, boxed_frame_path])
    if len(images_b64) < 2:
        return {
            "checked": False,
            "content_outside": False,
            "reason": "failed to read one or both check frames",
            "failed": True,
        }

    try:
        if provider in ("local", "ollama"):
            base_url = api_url or "http://localhost:11434"
            raw = _call_ollama(
                base_url, model, _CROP_VLM_SYSTEM_PROMPT, _CROP_VLM_USER_PROMPT, images_b64
            )
        elif provider == "openai":
            raw = _call_openai(
                api_key, model, _CROP_VLM_SYSTEM_PROMPT, _CROP_VLM_USER_PROMPT, images_b64
            )
        elif provider == "api":
            raw = _call_custom_api(
                api_key,
                api_url,
                model,
                _CROP_VLM_SYSTEM_PROMPT,
                _CROP_VLM_USER_PROMPT,
                images_b64,
                allow_http=allow_http,
            )
        else:
            raw = ""
    except Exception as e:
        logger.warning("Crop VLM check raised an exception (provider=%s)", provider, exc_info=True)
        return {"checked": False, "content_outside": False, "reason": str(e), "failed": True}

    if not raw:
        return {
            "checked": False,
            "content_outside": False,
            "reason": "VLM request failed or returned an empty response",
            "failed": True,
        }

    parsed = _parse_crop_vlm_response(raw)
    if parsed is None:
        logger.warning("Crop VLM response could not be parsed; raw=%.500r", raw)
        return {
            "checked": False,
            "content_outside": False,
            "reason": "unparseable VLM response",
            "failed": True,
        }

    parsed["checked"] = True
    parsed["failed"] = False
    return parsed


# ─── Perceptual Hashing & Duplicate Detection ──────────────────────


def compute_video_hash(filepath: str, num_frames: int = 30) -> str:
    """Compute a perceptual hash of a video for similarity comparison.

    Uses average hash (ahash) on evenly-spaced frames. Each frame is
    resized to 16x16 grayscale, hashed via the ahash algorithm, and
    all frame hashes are combined via majority voting per bit position.

    Args:
        filepath: Path to the video file
        num_frames: Number of frames to sample

    Returns:
        Binary hash string (e.g., '10110010...')
    """
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return ""

    step = max(1, total_frames // num_frames)
    hashes: list[str] = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            # A single transient decode glitch shouldn't truncate all
            # subsequent sampling — skip this frame and keep going.
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16))
        mean = gray.mean()
        h = "".join("1" if p > mean else "0" for p in gray.flatten())
        hashes.append(h)

    cap.release()

    if not hashes:
        return ""

    # Combine all frame hashes via majority vote per bit position
    hash_len = len(hashes[0])
    combined = []
    for i in range(hash_len):
        bits = [h[i] for h in hashes if i < len(h)]
        combined.append("1" if bits.count("1") > len(bits) / 2 else "0")

    return "".join(combined)


def compute_video_dhash(filepath: str, width: int = 16) -> str:
    """Compute a difference hash (dhash) of a video.

    Dhash compares adjacent pixels to detect structural patterns,
    which is more robust than ahash for videos with similar
    content but different lighting/contrast.

    Args:
        filepath: Path to the video file
        width: Hash width in pixels (default 16)

    Returns:
        Binary hash string
    """
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return ""

    step = max(1, total_frames // 10)  # Sample fewer frames for dhash
    hashes: list[str] = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, width + 1))

        # Dhash: compare each pixel to its right neighbor
        bits = []
        for row in range(gray.shape[0]):
            for col in range(gray.shape[1] - 1):
                bits.append("1" if gray[row, col] > gray[row, col + 1] else "0")
        hashes.append("".join(bits))

    cap.release()

    if not hashes:
        return ""

    hash_len = len(hashes[0])
    combined = []
    for i in range(hash_len):
        bits = [h[i] for h in hashes if i < len(h)]
        combined.append("1" if bits.count("1") > len(bits) / 2 else "0")

    return "".join(combined)


def hash_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity between two perceptual hashes (0-1).

    Uses Hamming distance: percentage of matching bits.

    Args:
        hash1: First hash string
        hash2: Second hash string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 0.0

    matching = sum(a == b for a, b in zip(hash1, hash2))
    return matching / len(hash1)
