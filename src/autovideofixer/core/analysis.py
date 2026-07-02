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
import tempfile
from dataclasses import dataclass, field
from typing import Any

from autovideofixer.config import Config

logger = logging.getLogger(__name__)

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
                full = os.path.join(root, f)
                if is_video_file(full):
                    videos.append(full)
    else:
        for f in sorted(os.listdir(directory)):
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
    ) -> VideoAnalysis:
        """Perform full analysis on a video file.

        Args:
            filepath: Path to the video file
            include_vlm: Whether to run VLM analysis (None = use config)
            include_events: Whether to detect events (None = use config)

        Returns:
            VideoAnalysis with all detected information
        """
        if filepath in self._analysis_cache:
            return self._analysis_cache[filepath]

        from autovideofixer.core.ffmpeg_utils import probe

        info = probe(filepath)
        analysis = VideoAnalysis(
            filepath=filepath,
            filename=info.filename,
            duration=info.duration,
            resolution=info.resolution,
            framerate=info.framerate,
            has_video=info.has_video,
            has_audio=info.has_audio,
            is_hdr=info.is_hdr,
        )

        if (
            include_events
            if include_events is not None
            else self.config.get("analysis", "event_detection", "enabled", default=True)
        ):
            analysis.scenes = self.detect_events(filepath)
            analysis.total_scenes = len(analysis.scenes)

        if (
            include_vlm
            if include_vlm is not None
            else self.config.get("analysis", "vlm", "enabled", default=False)
        ):
            vlm_result = self.run_vlm_analysis(filepath)
            analysis.vlm_summary = vlm_result.get("summary")
            analysis.vlm_tags = vlm_result.get("tags", [])
            analysis.vlm_objects = vlm_result.get("objects", [])
            analysis.content_rating = vlm_result.get("rating")

        self._analysis_cache[filepath] = analysis
        return analysis

    def detect_events(
        self,
        filepath: str,
        min_duration: float | None = None,
        classify_events: bool = False,
    ) -> list[SceneEvent]:
        """Detect scene changes and events in a video.

        Uses frame differencing to find scene boundaries, with optional
        event-type classification via frame-property heuristics (edge
        density/brightness) — not a VLM call. See run_vlm_analysis() for
        actual VLM-based content understanding.

        Args:
            filepath: Path to the video file
            min_duration: Minimum scene duration in seconds
            classify_events: Whether to heuristically classify event types

        Returns:
            List of SceneEvent objects
        """
        if min_duration is None:
            min_duration = self.config.get(
                "analysis", "event_detection", "min_scene_duration_sec", default=2.0
            )

        threshold = self.config.get(
            "analysis", "event_detection", "scene_change_threshold", default=0.3
        )

        scenes = _detect_scene_changes(filepath, threshold, min_duration)

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
        sample_interval_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Run VLM (Vision Language Model) analysis on video content.

        Samples frames at intervals and sends them to a configured VLM
        provider for content analysis.

        Args:
            filepath: Path to the video file
            sample_interval_sec: Seconds between frame samples

        Returns:
            Dict with keys: summary, tags, objects, rating
        """
        vlm_config = self.config.get("analysis", "vlm", default={})
        provider = vlm_config.get("provider", "local")
        model = vlm_config.get("model", "llava")
        api_url = vlm_config.get("api_url", "")
        api_key = vlm_config.get("api_key", "")
        max_frames = vlm_config.get("max_sample_frames", 8)

        frames = _extract_sample_frames(filepath, sample_interval_sec, max_frames=max_frames)
        if not frames:
            return {"summary": "", "tags": [], "objects": []}

        # All sample frames live in one temp dir created by _extract_sample_frames;
        # it must be cleaned up here, after the provider has read the frames into
        # base64, not inside _extract_sample_frames itself (which would delete the
        # files before they're ever read).
        tmp_dir = os.path.dirname(frames[0])
        try:
            if provider in ("local", "ollama"):
                return _run_ollama_vlm(frames, model, api_url)
            elif provider == "openai":
                return _run_openai_vlm(frames, api_key, model)
            elif provider == "api":
                return _run_api_vlm(frames, api_key, api_url, model)
            else:
                return {"summary": "", "tags": [], "objects": []}
        except Exception:
            logger.warning(
                "VLM analysis failed for %r (provider=%s)", filepath, provider, exc_info=True
            )
            return {"summary": "", "tags": [], "objects": []}
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
        "max_tokens": 1000,
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
    parsed = urlparse(api_url)
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        logger.warning("Refusing non-https custom VLM API URL for non-loopback host: %s", api_url)
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
        "max_tokens": 1000,
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


def _run_ollama_vlm(
    frames: list[str],
    model: str,
    api_url: str,
) -> dict[str, Any]:
    """Run analysis using Ollama."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    base_url = api_url or "http://localhost:11434"
    response = _call_ollama(
        base_url,
        model,
        _VLM_SYSTEM_PROMPT,
        _VLM_USER_PROMPT,
        image_b64,
    )
    return _parse_vlm_response(response)


def _run_openai_vlm(
    frames: list[str],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Run analysis using OpenAI Vision API."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    response = _call_openai(api_key, model, _VLM_SYSTEM_PROMPT, _VLM_USER_PROMPT, image_b64)
    return _parse_vlm_response(response)


def _run_api_vlm(
    frames: list[str],
    api_key: str,
    api_url: str,
    model: str,
) -> dict[str, Any]:
    """Run analysis using a custom API endpoint (OpenAI-compatible)."""
    image_b64 = _frames_to_base64(frames)
    if not image_b64:
        return {"summary": "", "tags": [], "objects": []}

    response = _call_custom_api(
        api_key,
        api_url,
        model,
        _VLM_SYSTEM_PROMPT,
        _VLM_USER_PROMPT,
        image_b64,
    )
    return _parse_vlm_response(response)


# ─── Scene Detection ───────────────────────────────────────────────


def _detect_scene_changes(
    filepath: str,
    threshold: float,
    min_duration_sec: float,
) -> list[SceneEvent]:
    """Detect scene changes using frame differencing.

    Compares consecutive frames at reduced resolution and identifies
    boundaries where pixel difference exceeds the threshold.
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
                    scene_start = current_time

        prev_frame = gray
        frame_idx += 1

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
