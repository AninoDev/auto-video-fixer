"""Auto Video Fixer - Video analysis utilities.

Provides video file detection, content analysis via VLM,
event detection, and duplicate/similar video detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from autovideofixer.config import Config

# Supported video extensions
VIDEO_EXTENSIONS = {
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def is_video_file(path: str) -> bool:
    """Check if a file is a video based on extension and/or content."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return True
    # Could also check magic bytes for more reliable detection
    return False


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
    videos = []
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

        # Basic probe
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

        # Scene/event detection
        if (
            include_events
            if include_events is not None
            else self.config.get("analysis", "event_detection", "enabled", default=True)
        ):
            analysis.scenes = self.detect_events(filepath)
            analysis.total_scenes = len(analysis.scenes)

        # VLM analysis
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
    ) -> list[SceneEvent]:
        """Detect scene changes and events in a video.

        Uses frame differencing to find scene boundaries.
        """
        if min_duration is None:
            min_duration = self.config.get(
                "analysis", "event_detection", "min_scene_duration_sec", default=2.0
            )

        threshold = self.config.get(
            "analysis", "event_detection", "scene_change_threshold", default=0.3
        )

        # Extract frames and compare
        scenes = _detect_scene_changes(filepath, threshold, min_duration)
        return scenes

    def run_vlm_analysis(
        self,
        filepath: str,
        sample_interval_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Run VLM (Vision Language Model) analysis on video content.

        Samples frames at intervals and sends them to a VLM for analysis.
        """
        vlm_config = self.config.get("analysis", "vlm", default={})
        provider = vlm_config.get("provider", "local")
        model = vlm_config.get("model", "llava")
        api_url = vlm_config.get("api_url", "")
        api_key = vlm_config.get("api_key", "")

        # Extract sample frames
        frames = _extract_sample_frames(filepath, sample_interval_sec)
        if not frames:
            return {"summary": "", "tags": [], "objects": []}

        if provider == "local":
            return _run_local_vlm(frames, model, api_url)
        elif provider == "api":
            return _run_api_vlm(frames, api_key, api_url, model)
        elif provider == "ollama":
            return _run_ollama_vlm(frames, model, api_url)
        elif provider == "openai":
            return _run_openai_vlm(frames, api_key, model)
        else:
            return {"summary": "", "tags": [], "objects": []}

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

        ref_hash = _compute_video_hash(filepath)
        results = []

        for candidate in candidates:
            if candidate == filepath:
                continue
            cand_hash = _compute_video_hash(candidate)
            similarity = _hash_similarity(ref_hash, cand_hash)
            if similarity >= threshold:
                results.append((candidate, similarity))

        return sorted(results, key=lambda x: -x[1])

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._analysis_cache.clear()


# ─── Internal helpers ──────────────────────────────────────────────


def _detect_scene_changes(
    filepath: str,
    threshold: float,
    min_duration_sec: float,
) -> list[SceneEvent]:
    """Detect scene changes using frame differencing."""
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return []

    scenes: list[SceneEvent] = []
    prev_frame = None
    scene_start = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps if fps > 0 else 0.0

        # Convert to grayscale and resize for faster comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            diff_score = float(diff.mean()) / 255.0

            if diff_score > threshold:
                # Scene change detected
                if current_time - scene_start >= min_duration_sec:
                    scenes.append(
                        SceneEvent(
                            start_time=scene_start,
                            end_time=current_time,
                            confidence=diff_score,
                        )
                    )
                scene_start = current_time

        prev_frame = gray
        frame_idx += 1

    cap.release()

    # Final scene
    if scene_start < current_time:
        scenes.append(
            SceneEvent(
                start_time=scene_start,
                end_time=current_time,
                confidence=0.5,
            )
        )

    return scenes


def _extract_sample_frames(
    filepath: str,
    interval_sec: float,
    max_frames: int = 20,
) -> list[str]:
    """Extract evenly-spaced sample frames from a video."""
    import tempfile

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
        while frame_idx < max_frames * frame_interval:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)

            frame_idx += frame_interval
    finally:
        cap.release()

    return frames


def _run_local_vlm(
    frames: list[str],
    model_name: str,
    api_url: str,
) -> dict[str, Any]:
    """Run analysis using a local VLM (e.g., Ollama, local server)."""
    # Placeholder for local VLM integration
    # In practice, this would use a local inference server
    return {"summary": "", "tags": [], "objects": []}


def _run_api_vlm(
    frames: list[str],
    api_key: str,
    api_url: str,
    model: str,
) -> dict[str, Any]:
    """Run analysis using an API-based VLM."""
    # Placeholder for API-based VLM (e.g., custom API endpoint)
    return {"summary": "", "tags": [], "objects": []}


def _run_ollama_vlm(
    frames: list[str],
    model: str,
    api_url: str,
) -> dict[str, Any]:
    """Run analysis using Ollama."""
    # Placeholder for Ollama integration
    return {"summary": "", "tags": [], "objects": []}


def _run_openai_vlm(
    frames: list[str],
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Run analysis using OpenAI Vision API."""
    # Placeholder for OpenAI integration
    return {"summary": "", "tags": [], "objects": []}


def _compute_video_hash(filepath: str, num_frames: int = 30) -> str:
    """Compute a perceptual hash of a video for similarity comparison.

    Extracts evenly-spaced frames, computes per-frame hashes, and combines.
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
    hashes = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale, resize, and hash
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 36))
        mean = gray.mean()
        h = "".join("1" if p > mean else "0" for p in gray.flatten())
        hashes.append(h)

    cap.release()

    # Combine all frame hashes (majority vote for each bit position)
    if not hashes:
        return ""

    combined = []
    for i in range(len(hashes[0])):
        bits = [h[i] for h in hashes if i < len(h)]
        combined.append("1" if bits.count("1") > len(bits) / 2 else "0")

    return "".join(combined)


def _hash_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity between two perceptual hashes (0-1)."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 0.0

    matching = sum(a == b for a, b in zip(hash1, hash2))
    return matching / len(hash1)
