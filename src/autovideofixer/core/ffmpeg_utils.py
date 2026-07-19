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
    nb_frames: int = 0

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
    # ffprobe's stderr from the run that produced this result, even on a
    # successful (rc=0) probe -- see probe()'s "-v error" flag below. Non-empty
    # here means ffprobe emitted warnings/errors about the file despite still
    # producing usable JSON on stdout (e.g. a slightly malformed container);
    # surfaced by Pipeline's § 6.3 probe policy (general.fail_on_probe_warnings).
    stderr: str = ""
    # Embedded video title, from format.tags.title, falling back to the video
    # stream's own tags.title when the format-level tag is absent. "" when
    # neither is set. Captured (not logged anywhere new by this alone) so
    # it's a KNOWN value § 6.7's PII cleaner can register -- see
    # logclean.PIICleaner.register_title() / Pipeline.execute_job().
    title: str = ""

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

    @property
    def frame_count(self) -> int:
        vs = self.video_stream
        if vs and vs.nb_frames > 0:
            return vs.nb_frames
        if vs and vs.fps > 0:
            return int(self.duration * vs.fps)
        return 0

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
            "probe_stderr": self.stderr,
            "title": self.title,
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
    """Probe a media file and return detailed information.

    Uses ``-v error`` (not ``-v quiet``) so a failing probe's ``RuntimeError``
    carries an actually-informative ``stderr`` (ffprobe still only writes
    errors/warnings there, never routine progress chatter -- the JSON payload
    stays on stdout either way) -- see docs/REQUIREMENTS.md § 6.3. This also
    means a *successful* (rc=0) probe can still have non-empty stderr (e.g. a
    slightly malformed container ffprobe recovers from); callers that care
    read it back via ``ProbeResult.stderr``/``to_info_dict()["probe_stderr"]``.
    """
    ffprobe = get_ffprobe_path(config)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "--",
        filepath,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {filepath}: {result.stderr}")

    data = json.loads(result.stdout)
    parsed = _parse_probe_result(data, filepath)
    parsed.stderr = result.stderr
    return parsed


def _parse_probe_result(data: dict, filepath: str) -> ProbeResult:
    """Parse ffprobe JSON output into ProbeResult."""
    streams = []
    video_stream_title = ""
    for i, s in enumerate(data.get("streams", [])):
        tags = s.get("tags", {})
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
            language=tags.get("language", ""),
            nb_frames=_safe_int(s.get("nb_frames", 0)),
        )
        streams.append(stream)
        if stream.is_video and not video_stream_title:
            video_stream_title = tags.get("title", "")

    fmt = data.get("format", {})
    # REQUIREMENTS.md § 6.7: format-level title wins, falling back to the
    # video stream's own tags.title when the format-level tag is absent.
    title = fmt.get("tags", {}).get("title", "") or video_stream_title
    return ProbeResult(
        filepath=filepath,
        filename=os.path.basename(filepath),
        duration=_safe_float(fmt.get("duration", 0)),
        bit_rate=fmt.get("bit_rate", 0) or 0,
        format_name=fmt.get("format_name", ""),
        format_long_name=fmt.get("format_long_name", ""),
        streams=streams,
        title=title,
    )


def _rate_str_to_float(rate_str: str | None) -> float:
    """Convert an ffprobe "N/D" (or plain numeric) rate string to a float.

    Returns 0.0 for missing/undefined rates (including ffprobe's "0/0",
    which it emits when the rate can't be determined -- e.g. duration
    unknown), so callers can treat 0.0 uniformly as "not available".
    """
    if not rate_str:
        return 0.0
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            num, den = float(num), float(den)
            return num / den if den else 0.0
        return float(rate_str)
    except ValueError, ZeroDivisionError:
        return 0.0


def _parse_fps(s: dict) -> float:
    """Parse framerate from stream info, preferring avg_frame_rate.

    avg_frame_rate is the honest "real frame count / real duration" rate.
    r_frame_rate ("tbr") is ffmpeg's declared/theoretical rate and, for VFR
    or YouTube-origin sources, can be a multiple of the true average (e.g.
    r_frame_rate=59.94 vs avg_frame_rate=29.64 for the same file). Using
    r_frame_rate anywhere frame count is reconciled with wall-clock time
    (e.g. as the input `-r` for a headerless/raw pipe) silently halves the
    apparent duration of the real frames.

    Falls back to r_frame_rate only when avg_frame_rate is missing or
    undefined (ffprobe emits "0/0" when duration is unknown).
    """
    avg_fps = _rate_str_to_float(s.get("avg_frame_rate"))
    if avg_fps > 0:
        return avg_fps
    return _rate_str_to_float(s.get("r_frame_rate", "0/1"))


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except ValueError, TypeError:
        return 0.0


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except ValueError, TypeError:
        return 0


def detect_hardware_acceleration() -> list[str]:
    """Detect available FFmpeg hardware acceleration methods."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-nostdin", "-hwaccels"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL
        )
        hwaccels = []
        for line in result.stdout.strip().split("\n")[1:]:
            line = line.strip()
            if line:
                hwaccels.append(line.lower())
        return hwaccels
    except FileNotFoundError, subprocess.TimeoutExpired:
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
    timeout: float | None = 3600,
    capture_stderr: bool = True,
) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with optional progress reporting.

    Args:
        args: FFmpeg command arguments
        progress_callback: Optional callback(progress: float, message: str)
        timeout: Command timeout in seconds, or ``None`` for no timeout
            (waits indefinitely -- ``subprocess.Popen.wait(timeout=None)``
            blocks forever rather than raising). Most stage-level callers
            resolve this via ``BaseStage.stage_timeout()``
            (``stages.<name>.timeout`` -> ``pipeline.stage_timeout`` ->
            ``None``) so a fixed wall-clock cap doesn't kill legitimately
            long real-world inputs; short genuinely-bounded helper calls
            (probes, hwaccel detection, single-frame extraction, etc.) still
            pass a small fixed int here. The default of 3600 only applies to
            callers that don't pass ``timeout`` at all.
        capture_stderr: Whether to capture stderr for parsing

    Returns:
        CompletedProcess result
    """
    import shlex
    import threading

    from autovideofixer.logger import get_logger

    logger = get_logger("autovideofixer.ffmpeg")

    ffmpeg = get_ffmpeg_path()
    # -nostdin: every ffmpeg invocation in this codebase is non-interactive
    # (no caller ever intends to feed it keyboard commands like 'q'). Without
    # it, ffmpeg switches its controlling terminal's tty to raw mode to poll
    # for those keys and -- if killed/crashed/backgrounded before it exits
    # cleanly -- never restores it, leaving the user's shell with no echo
    # and a stray newline per command. stdin=DEVNULL below is belt-and-
    # suspenders: even if a future ffmpeg build ever changed -nostdin's
    # behavior, a subprocess with no access to the real tty on stdin can't
    # raw-mode it either way. See AGENTS.md's "every new subprocess spawn
    # must detach stdin" gotcha.
    cmd = [ffmpeg, "-hide_banner", "-nostdin"] + args

    # Always logged at DEBUG (not gated on failure) so `--log-file` with
    # `--file-log-level DEBUG` captures every command run in a job, including
    # ones from stages that ran and succeeded *before* a later stage failed --
    # essential for diagnosing a failure that was actually caused by an
    # earlier stage's output, not the stage that visibly errored.
    logger.debug("ffmpeg command: %s", shlex.join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        text=True,
    )

    # Drain stderr on a background thread so a stalled/hung ffmpeg (no more output,
    # but not exited) doesn't block forever on the blocking `for line in proc.stderr`
    # iterator before ever reaching a timeout-aware wait call below. This mirrors the
    # pattern stabilize.py already uses for its manual decode/transform pipe.
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        if capture_stderr and proc.stderr:
            for line in proc.stderr:
                stderr_lines.append(line)
                if progress_callback:
                    _parse_ffmpeg_progress(line, progress_callback)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        stderr_thread.join(timeout=5)
        raise

    stderr_thread.join(timeout=5)
    full_stderr = "".join(stderr_lines)

    if proc.returncode != 0:
        # StageResult.error truncates stderr to 200-500 chars, which is often not
        # enough to see the actual ffmpeg error (it's usually near the end, e.g.
        # "Unrecognized option" or a codec/filter error a few lines from EOF).
        # Log the full command + stderr here, once, centrally, regardless of
        # which stage called us or how much of the error message it kept.
        logger.error(
            "ffmpeg failed (exit %s): %s\n--- full stderr ---\n%s",
            proc.returncode,
            shlex.join(cmd),
            full_stderr,
        )

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="",
        stderr=full_stderr,
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
    temp_dir: str | None = None,
    ext: str = ".mkv",
) -> str:
    """Generate a safe temporary file path for intermediate processing.

    Uses ``temp_dir`` (typically sourced from config's ``general.temp_dir``) when
    provided; otherwise falls back to the input file's own directory, or ``base_dir``.

    Always defaults to a ``.mkv`` extension regardless of the input container:
    intermediate stages hardcode codecs (e.g. libx264) that aren't valid in every
    container (e.g. WebM only permits VP8/VP9+Opus/Vorbis), so re-using the input's
    extension for intermediates can make ffmpeg reject the output outright. MKV can
    hold essentially any codec, so it's a safe universal intermediate container.
    """
    import uuid

    base = temp_dir or os.path.dirname(original_path) or base_dir
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f".avf_{uuid.uuid4().hex[:8]}{suffix}{ext}")


def get_video_info(input_path: str) -> dict[str, Any]:
    """Convenience function: probe and return info dict."""
    p = probe(input_path)
    return p.to_info_dict()
