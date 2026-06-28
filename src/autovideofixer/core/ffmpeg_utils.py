"""Auto Video Fixer - FFmpeg utilities for probe, encoding, and hardware detection."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from autovideofixer.config import Config


class HWAccel(Enum):
    AUTO = "auto"
    NONE = "none"
    CUDA = "cuda"
    VAAPI = "vaapi"
    QSV = "qsv"
    VULKAN = "vulkan"
    D3D11 = "d3d11"
    METAL = "videotoolbox"


@dataclass
class StreamInfo:
    """Information about a media stream."""

    index: int
    codec_type: str  # video, audio, subtitle
    codec_name: str = ""
    codec_long_name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    bit_rate: int = 0
    channels: int = 0
    sample_rate: int = 0
    pixel_format: str = ""
    profile: str = ""
    color_space: str = ""
    color_range: str = ""
    color_transfer: str = ""
    is_default: bool = False
    language: str = ""

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_audio(self) -> bool:
        return self.codec_type == "audio"


@dataclass
class ProbeResult:
    """Complete probe information for a media file."""

    filepath: str
    filename: str
    duration: float = 0.0
    bit_rate: int = 0
    format_name: str = ""
    format_long_name: str = ""
    streams: list[StreamInfo] = field(default_factory=list)

    @property
    def video_stream(self) -> StreamInfo | None:
        for s in self.streams:
            if s.is_video:
                return s
        return None

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.is_audio]

    @property
    def resolution(self) -> tuple[int, int]:
        vs = self.video_stream
        return (vs.width, vs.height) if vs else (0, 0)

    @property
    def framerate(self) -> float:
        vs = self.video_stream
        return vs.fps if vs else 0.0

    @property
    def has_video(self) -> bool:
        return self.video_stream is not None

    @property
    def has_audio(self) -> bool:
        return len(self.audio_streams) > 0

    @property
    def is_hdr(self) -> bool:
        vs = self.video_stream
        if not vs:
            return False
        ct = vs.color_transfer or ""
        return ct.lower() in ("smpte2084", "arib-std-b67")

    def to_info_dict(self) -> dict[str, Any]:
        """Convert to dictionary for pipeline use."""
        vs = self.video_stream
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "duration": self.duration,
            "bit_rate": self.bit_rate,
            "format": self.format_name,
            "resolution": self.resolution,
            "width": vs.width if vs else 0,
            "height": vs.height if vs else 0,
            "framerate": self.framerate,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "is_hdr": self.is_hdr,
            "video_codec": vs.codec_name if vs else "",
            "audio_codecs": [s.codec_name for s in self.audio_streams],
            "audio_count": len(self.audio_streams),
            "streams": len(self.streams),
        }


def get_ffmpeg_path(config: Config | None = None) -> str:
    """Find ffmpeg binary path."""
    if config:
        path = config.get("ffmpeg", "binary")
        if path and os.path.isfile(path):
            return path
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "ffmpeg not found. Install FFmpeg and ensure it is in PATH, "
        "or set ffmpeg.binary in configuration."
    )


def get_ffprobe_path(config: Config | None = None) -> str:
    """Find ffprobe binary path."""
    if config:
        path = config.get("ffmpeg", "binary")
        if path:
            base = os.path.dirname(path)
            candidate = os.path.join(base, "ffprobe" + (".exe" if os.name == "nt" else ""))
            if os.path.isfile(candidate):
                return candidate
    for name in ("ffprobe", "ffprobe.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("ffprobe not found.")


def probe(filepath: str, config: Config | None = None) -> ProbeResult:
    """Probe a media file and return detailed information."""
    ffprobe = get_ffprobe_path(config)
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {filepath}: {result.stderr}")

    data = json.loads(result.stdout)
    return _parse_probe_result(data, filepath)


def _parse_probe_result(data: dict, filepath: str) -> ProbeResult:
    """Parse ffprobe JSON output into ProbeResult."""
    streams = []
    for i, s in enumerate(data.get("streams", [])):
        stream = StreamInfo(
            index=i,
            codec_type=s.get("codec_type", ""),
            codec_name=s.get("codec_name", ""),
            codec_long_name=s.get("codec_long_name", ""),
            width=s.get("width", 0),
            height=s.get("height", 0),
            fps=_parse_fps(s),
            duration=_safe_float(s.get("duration", 0)),
            bit_rate=s.get("bit_rate", 0) or 0,
            channels=s.get("channels", 0),
            sample_rate=s.get("sample_rate", 0),
            pixel_format=s.get("pix_fmt", ""),
            profile=s.get("profile", ""),
            color_space=s.get("color_space", ""),
            color_range=s.get("color_range", ""),
            color_transfer=s.get("color_transfer", ""),
            is_default=s.get("disposition", {}).get("default", 0) != 0,
            language=s.get("tags", {}).get("language", ""),
        )
        streams.append(stream)

    fmt = data.get("format", {})
    return ProbeResult(
        filepath=filepath,
        filename=os.path.basename(filepath),
        duration=_safe_float(fmt.get("duration", 0)),
        bit_rate=fmt.get("bit_rate", 0) or 0,
        format_name=fmt.get("format_name", ""),
        format_long_name=fmt.get("format_long_name", ""),
        streams=streams,
    )


def _parse_fps(s: dict) -> float:
    """Parse framerate from stream info, handling both avg_frame_rate and r_frame_rate."""
    fps_str = s.get("avg_frame_rate") or s.get("r_frame_rate", "0/1")
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            num, den = float(num), float(den)
            return num / den if den else 0.0
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def detect_hardware_acceleration() -> list[str]:
    """Detect available FFmpeg hardware acceleration methods."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hwaccels"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        hwaccels = []
        for line in result.stdout.strip().split("\n")[1:]:
            line = line.strip()
            if line:
                hwaccels.append(line.lower())
        return hwaccels
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def resolve_hwaccel(preferred: str = "auto") -> str:
    """Resolve hardware acceleration preference to actual method.

    Args:
        preferred: One of 'auto', 'cuda', 'vaapi', 'qsv', 'vulkan', 'd3d11', 'videotoolbox', 'none'

    Returns:
        Resolved hwaccel string suitable for FFmpeg -hwaccel flag
    """
    if preferred == "none":
        return "none"

    available = detect_hardware_acceleration()

    if preferred == "auto":
        # Prefer order: cuda > vaapi > qsv > videotoolbox > d3d11 > vulkan
        for method in ["cuda", "vaapi", "qsv", "videotoolbox", "d3d11", "vulkan"]:
            if method in available:
                return method
        return "none"

    if preferred.lower() in available:
        return preferred.lower()

    return "none"


def build_hwaccel_args(hwaccel: str) -> list[str]:
    """Build FFmpeg hardware acceleration command-line arguments."""
    if hwaccel == "none":
        return []
    return ["-hwaccel", hwaccel]


def run_ffmpeg(
    args: list[str],
    progress_callback: callable | None = None,
    timeout: int = 3600,
    capture_stderr: bool = True,
) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with optional progress reporting.

    Args:
        args: FFmpeg command arguments
        progress_callback: Optional callback(progress: float, message: str)
        timeout: Command timeout in seconds
        capture_stderr: Whether to capture stderr for parsing

    Returns:
        CompletedProcess result
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hide_banner"] + args

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        text=True,
    )

    stderr_lines = []
    if capture_stderr and proc.stderr:
        for line in proc.stderr:
            stderr_lines.append(line)
            if progress_callback:
                _parse_ffmpeg_progress(line, progress_callback)

    stdout, _ = proc.communicate(timeout=timeout)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr="".join(stderr_lines),
    )


def _parse_ffmpeg_progress(
    line: str,
    callback: callable,
) -> None:
    """Parse FFmpeg stderr line for time-based progress estimation.

    Looks for patterns like: time=00:01:23.45
    """
    match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        # Progress requires knowing total duration; we pass current time as info
        callback(0.0, f"Processing: {h:02d}:{m:02d}:{s:05.2f}")


def estimate_duration(input_path: str) -> float:
    """Get estimated duration of a media file."""
    try:
        info = probe(input_path)
        return info.duration
    except Exception:
        return 0.0


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def generate_temp_path(
    base_dir: str,
    original_path: str,
    suffix: str = "_proc",
) -> str:
    """Generate a safe temporary file path for intermediate processing."""
    import uuid

    base = os.path.dirname(original_path) or base_dir
    ext = os.path.splitext(original_path)[1]
    return os.path.join(base, f".avf_{uuid.uuid4().hex[:8]}{suffix}{ext}")


def get_video_info(input_path: str) -> dict[str, Any]:
    """Convenience function: probe and return info dict."""
    p = probe(input_path)
    return p.to_info_dict()
