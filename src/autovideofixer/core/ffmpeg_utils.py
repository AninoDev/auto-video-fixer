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
    # Added for normalize_codecs (REQUIREMENTS.md § 22.2): carried explicitly
    # through a video re-encode so the later `hdr` stage still sees HDR.
    color_primaries: str = ""
    # ffprobe's raw bits_per_raw_sample ("10", "N/A", ...) parsed to an int,
    # 0 when absent. Secondary to pix_fmt for bit depth (see pix_fmt_bit_depth).
    bits_per_raw_sample: int = 0
    # disposition.attached_pic: cover art stored as a "video" stream. Never the
    # stream ffmpeg's default selection picks as the real video.
    is_attached_pic: bool = False
    # ffprobe's channel_layout string ("stereo", "5.1(side)", ""), informational.
    channel_layout: str = ""
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
    def selected_video_stream(self) -> StreamInfo | None:
        """The video stream ffmpeg's DEFAULT stream selection (no ``-map``)
        picks: the highest-resolution non-attached-picture video stream (first
        on ties); falls back to ``video_stream`` when only cover art exists.
        Used by stages that must reason about the exact stream a map-less
        command will carry (normalize_codecs)."""
        real = [s for s in self.streams if s.is_video and not s.is_attached_pic]
        if not real:
            return self.video_stream
        return max(real, key=lambda s: (s.width * s.height, -s.index))

    @property
    def selected_audio_stream(self) -> StreamInfo | None:
        """The audio stream ffmpeg's default selection picks: most channels,
        first on ties."""
        auds = self.audio_streams
        if not auds:
            return None
        return max(auds, key=lambda s: (s.channels, -s.index))

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
            # normalize_codecs inputs (REQUIREMENTS.md § 22.2): the stream
            # ffmpeg's default (map-less) selection carries, described by the
            # fields the stage needs. "" / 0 / [] when absent.
            "pix_fmt": (
                self.selected_video_stream.pixel_format if self.selected_video_stream else ""
            ),
            "bit_depth": pix_fmt_bit_depth(
                self.selected_video_stream.pixel_format if self.selected_video_stream else "",
                self.selected_video_stream.bits_per_raw_sample if self.selected_video_stream else 0,
            ),
            "color_primaries": (
                self.selected_video_stream.color_primaries if self.selected_video_stream else ""
            ),
            "color_trc": (
                self.selected_video_stream.color_transfer if self.selected_video_stream else ""
            ),
            "color_space": (
                self.selected_video_stream.color_space if self.selected_video_stream else ""
            ),
            "color_range": (
                self.selected_video_stream.color_range if self.selected_video_stream else ""
            ),
            "selected_video_codec": (
                self.selected_video_stream.codec_name if self.selected_video_stream else ""
            ),
            "selected_audio_codec": (
                self.selected_audio_stream.codec_name if self.selected_audio_stream else ""
            ),
            "streams": len(self.streams),
            "probe_stderr": self.stderr,
            "title": self.title,
        }


def pix_fmt_bit_depth(pix_fmt: str, bits_per_raw_sample: int = 0) -> int:
    """Bits per component implied by an ffmpeg pixel-format name.

    ``yuv420p`` -> 8, ``yuv420p10le`` -> 10, ``p010le`` -> 10, ``gbrp12le`` ->
    12, ``gray16le`` -> 16, ``nv12`` -> 8. Falls back to ``bits_per_raw_sample``
    (ffprobe's value, already int-coerced) when the name carries no depth, and
    finally to 8 for a non-empty name / 0 for an empty one (unknown). The
    probe value is only a fallback because encoders pad: a 10-bit-in-12 file
    can report either, whereas the pix_fmt is what a filter graph will see.
    """
    name = (pix_fmt or "").lower()
    if not name:
        return bits_per_raw_sample if bits_per_raw_sample > 0 else 0
    if name.startswith("p0") and name[1:4].isdigit():  # p010le, p016le
        return int(name[1:4])
    m = re.search(r"p(\d{2})(?:le|be)?$", name) or re.search(r"(\d{2})(?:le|be)$", name)
    if m:
        return min(int(m.group(1)), 16)  # rgb48le etc. report packed totals
    if bits_per_raw_sample > 8:
        return bits_per_raw_sample
    return 8


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
            # ffprobe returns these as STRINGS ("44100", "2", "128000"), so they
            # need the same _safe_int coercion the other numeric fields get --
            # the dataclass annotations say int, and callers rely on that. A raw
            # string here crashed SpeedStage's asetrate path with
            # "'<=' not supported between instances of 'str' and 'int'" on any
            # real input (unit tests passed ints directly and never caught it).
            bit_rate=_safe_int(s.get("bit_rate", 0)),
            channels=_safe_int(s.get("channels", 0)),
            sample_rate=_safe_int(s.get("sample_rate", 0)),
            pixel_format=s.get("pix_fmt", ""),
            profile=s.get("profile", ""),
            color_space=s.get("color_space", ""),
            color_range=s.get("color_range", ""),
            color_transfer=s.get("color_transfer", ""),
            color_primaries=s.get("color_primaries", ""),
            bits_per_raw_sample=_safe_int(s.get("bits_per_raw_sample", 0)),
            is_attached_pic=s.get("disposition", {}).get("attached_pic", 0) == 1,
            channel_layout=s.get("channel_layout", ""),
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


_FILTER_CACHE: dict[str, set[str]] = {}


def list_ffmpeg_filters(config: Config | None = None) -> set[str]:
    """Return the set of filter names this ffmpeg build advertises (``ffmpeg -filters``).

    Cached per resolved ffmpeg binary path -- a given ffmpeg build's filter set never
    changes mid-process, so repeated capability checks (e.g. SpeedStage's
    rubberband-availability check, see stages.speed.audio_method) don't re-spawn ffmpeg
    on every call. Returns an empty set (rather than raising) if ffmpeg can't be found
    or the ``-filters`` probe itself fails/times out -- callers treat that the same as
    "filter not available".
    """
    ffmpeg = get_ffmpeg_path(config)
    cached = _FILTER_CACHE.get(ffmpeg)
    if cached is not None:
        return cached

    names: set[str] = set()
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
        # Each filter line looks like: " T.C rubberband        A->A       Apply time-...".
        # The first token is the capability flags column, the second is the filter name.
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", parts[1]):
                names.add(parts[1])
    except OSError, subprocess.TimeoutExpired:
        pass

    _FILTER_CACHE[ffmpeg] = names
    return names


def has_filter(name: str, config: Config | None = None) -> bool:
    """Whether this ffmpeg build supports the named filter (see ``list_ffmpeg_filters``)."""
    return name in list_ffmpeg_filters(config)


def build_hwaccel_args(hwaccel: str) -> list[str]:
    """Build FFmpeg hardware acceleration command-line arguments."""
    if hwaccel == "none":
        return []
    return ["-hwaccel", hwaccel]


def timing_output_args(mode: str = "passthrough") -> list[str]:
    """Build the shared ``-fps_mode`` OUTPUT arguments (REQUIREMENTS.md § 12.4).

    ``-fps_mode`` is an OUTPUT option -- callers must place these args
    somewhere in the output section of the command (after ``-i``, before the
    output path). Emitting them before ``-i`` corrupts input parsing
    (observed as spurious ``Duplicate element``/EBML errors on Matroska).

    ``"passthrough"`` (the only mode every timestamp-safe stage in this
    codebase actually uses) tells ffmpeg to honor each frame's own PTS
    exactly rather than silently re-conforming the stream to CFR -- required
    on every stage downstream of ``retime`` so a VFR intermediate's dropped
    duplicate frames stay dropped instead of being silently reinserted (see
    ``core/stages/retime.py`` and AGENTS.md's timestamp-safety notes).
    """
    return ["-fps_mode", mode]


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

    # A bare relative filename ("clip.mp4") has dirname "", and os.makedirs("")
    # raises FileNotFoundError -- so resolve against the cwd instead of crashing.
    base = temp_dir or os.path.dirname(original_path) or base_dir or os.getcwd()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f".avf_{uuid.uuid4().hex[:8]}{suffix}{ext}")


def sibling_temp_path(path: str, tag: str) -> str:
    """A scratch path next to ``path`` that keeps ``path``'s extension.

    ``<root>.<tag><ext>`` -- e.g. ``/t/a.mkv`` -> ``/t/a.resize_tmp.mkv``. Stage
    code that writes a scratch file and then ``os.replace``s it over the real
    destination MUST use this rather than a hardcoded extension: ffmpeg picks
    the muxer from the extension, so a hardcoded ``.mp4`` scratch replaced onto a
    ``.mkv`` destination leaves MP4 bytes in a file named ``.mkv`` (and a
    stream-copied Vorbis/Opus/FLAC track can't be written to the ``.mp4``
    scratch at all). REQUIREMENTS.md § 22.3.
    """
    root, ext = os.path.splitext(path)
    return f"{root}.{tag}{ext or '.mkv'}"


def get_video_info(input_path: str) -> dict[str, Any]:
    """Convenience function: probe and return info dict."""
    p = probe(input_path)
    return p.to_info_dict()
