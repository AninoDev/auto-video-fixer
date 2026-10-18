"""Auto Video Fixer - Codec normalization stage (REQUIREMENTS.md § 22.2).

Runs immediately after ``retime`` (see ``pipeline.default_order``), before every
heavy stage. Inputs with "unusual" codecs -- VP9/AV1 video, Opus/Vorbis/FLAC/PCM
audio, i.e. a typical YouTube ``.webm`` -- used to fail later stages for no real
reason: those stages stream-copy a source-derived stream (``-c:a copy`` /
``-c copy``) into a container (or ffmpeg build) that cannot hold it (Vorbis in
MP4), killing the job. This stage makes every stream it keeps one of a small
set of universally compatible codecs so every later copy path is safe without
each stage having to reason about codecs.

It touches a stream **only if it has to**:

* video codec in ``video_codecs_allowed`` (default ``[h264]``) -> ``-c:v copy``;
  otherwise re-encode with ``video_encoder`` (``libx264``) at ``temp_crf`` (16)
  / ``preset`` (medium). 8-bit -> ``yuv420p``; >8-bit -> ``yuv420p10le`` so
  10-bit/HDR content is not crushed. Colour metadata is carried explicitly so
  the later ``hdr`` stage still detects HDR.
* audio codec in ``audio_codecs_allowed`` (default ``[aac]``) -> ``-c:a copy``;
  otherwise ``aac`` at ``audio_bitrate`` (256k), channel layout preserved.
* SKIPs (no new file, no re-encode) when nothing needs changing.

Why video is re-encoded even though a later stage usually re-encodes it again:
the temp quality is visually lossless, and the alternative (every stage knowing
which codecs its container accepts) is the failure mode this stage removes. Why
it is NOT folded into ``retime``: retime SKIPs on honest input, which would
leave the very inputs this stage exists for (honest-cadence ``.webm``) unfixed.

Stream selection: deliberately **no ``-map``**. Every other ffmpeg-pass stage
(crop, downscale, deinterlace, retime, ...) relies on ffmpeg's default
selection (best video, best audio, one subtitle), so this stage must carry
exactly that -- widening to ``-map 0`` would feed extra streams to stages that
never expected them, narrowing would drop what they would have kept. The
copy-vs-transcode decision is made for the stream the default selection picks
(``ProbeResult.selected_video_stream`` / ``selected_audio_stream``, which
mirror ffmpeg's "highest resolution, non-cover-art" / "most channels" rules).
If that mirroring is ever wrong for an exotic multi-stream file the failure mode
is benign: a stream is copied (pre-feature behaviour) or needlessly transcoded.

Timestamp safety (REQUIREMENTS.md § 12.4): the output carries
``-fps_mode passthrough`` so a VFR intermediate (retime's output) is NOT
re-conformed to CFR -- re-conforming would silently re-insert every duplicate
frame retime just dropped. Verified on a real VFR retime output.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from autovideofixer.core.ffmpeg_utils import (
    ProbeResult,
    StreamInfo,
    pix_fmt_bit_depth,
    probe,
    run_ffmpeg,
    timing_output_args,
)
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

# ffprobe reports "unknown" for unset colour fields; ffmpeg rejects that as an
# option value, so such fields are simply not passed.
_UNSET_COLOR_VALUES = {"", "unknown", "unspecified", "reserved", "n/a"}


@dataclass
class StreamPlan:
    """What to do with one kept stream."""

    kind: str  # "video" | "audio"
    action: str  # "copy" | "transcode"
    from_codec: str
    to_codec: str  # == from_codec when copying
    detail: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind}
        if self.action == "copy":
            out["action"] = f"copied ({self.from_codec})"
        else:
            out["action"] = f"transcoded ({self.from_codec}->{self.to_codec})"
        out["copied"] = self.action == "copy"
        out.update(self.detail)
        return out


class NormalizeCodecsStage(BaseStage):
    """Rewrite only the streams whose codec is outside the allowed sets."""

    name = "normalize_codecs"
    display_name = "Codec Normalization"
    description = "Transcode unusual video/audio codecs (VP9, AV1, Opus, Vorbis...) to H.264/AAC"
    category = "enhancement"
    # Right after retime (priority=2); pipeline.default_order is authoritative.
    priority = 3
    supports_gpu = False

    # -- planning ---------------------------------------------------------

    def _allowed(self, key: str, default: list[str]) -> set[str]:
        raw = self._stage_config.get(key, default)
        if isinstance(raw, str):
            raw = [raw]
        return {str(c).lower() for c in (raw or [])}

    def _plan_from_streams(
        self, video: StreamInfo | None, audio: StreamInfo | None
    ) -> list[StreamPlan]:
        """Plan for the streams the default selection would carry."""
        video_ok = self._allowed("video_codecs_allowed", ["h264"])
        audio_ok = self._allowed("audio_codecs_allowed", ["aac"])
        plans: list[StreamPlan] = []
        if video is not None:
            codec = (video.codec_name or "").lower()
            if codec in video_ok:
                plans.append(StreamPlan("video", "copy", codec, codec))
            else:
                depth = pix_fmt_bit_depth(video.pixel_format, video.bits_per_raw_sample)
                plans.append(
                    StreamPlan(
                        "video",
                        "transcode",
                        codec or "unknown",
                        str(self._stage_config.get("video_encoder", "libx264")),
                        {
                            "bit_depth": depth,
                            "pix_fmt": "yuv420p10le" if depth > 8 else "yuv420p",
                            "width": video.width,
                            "height": video.height,
                            "color": self._color_args(video),
                        },
                    )
                )
        if audio is not None:
            codec = (audio.codec_name or "").lower()
            if codec in audio_ok:
                plans.append(StreamPlan("audio", "copy", codec, codec))
            else:
                plans.append(
                    StreamPlan(
                        "audio",
                        "transcode",
                        codec or "unknown",
                        str(self._stage_config.get("audio_codec", "aac")),
                        {"channels": audio.channels, "channel_layout": audio.channel_layout},
                    )
                )
        return plans

    @staticmethod
    def _color_args(video: StreamInfo) -> list[str]:
        """Explicit colour-metadata output options for a video re-encode.

        An RGB source (VP9/FFV1 ``gbrp*``, matrix ``gbr``) is converted to YUV by
        the re-encode, so its matrix and range describe a layout the output no
        longer has: libx264 rejects ``-colorspace gbr`` outright ("Error setting
        option colorspace to value gbr", found on a real RGB VP9 WebM), and RGB's
        full range would mislabel swscale's limited-range YUV. Only primaries and
        transfer (which survive the conversion) are carried for RGB sources.
        """
        pix_fmt = (video.pixel_format or "").lower()
        is_rgb = (video.color_space or "").strip().lower() == "gbr" or pix_fmt.startswith(
            ("gbr", "rgb", "bgr", "argb", "abgr")
        )
        pairs = [
            ("-color_primaries", video.color_primaries),
            ("-color_trc", video.color_transfer),
        ]
        if not is_rgb:
            pairs += [("-colorspace", video.color_space), ("-color_range", video.color_range)]
        args: list[str] = []
        for flag, value in pairs:
            if (value or "").strip().lower() not in _UNSET_COLOR_VALUES:
                args += [flag, value]
        return args

    def _build_args(self, input_path: str, output_path: str, plans: list[StreamPlan]) -> list[str]:
        args = ["-i", input_path]
        for plan in plans:
            if plan.kind == "video":
                if plan.action == "copy":
                    args += ["-c:v", "copy"]
                    continue
                args += [
                    "-c:v",
                    plan.to_codec,
                    "-preset",
                    str(self._stage_config.get("preset", "medium")),
                    "-crf",
                    str(self._stage_config.get("temp_crf", 16)),
                    "-pix_fmt",
                    plan.detail["pix_fmt"],
                ]
                # yuv420p needs even dimensions; VP9/AV1 allow odd ones
                # (e.g. 853x480). Trim the odd pixel row/column (invisible)
                # instead of letting libx264 reject the frame.
                if plan.detail.get("width", 0) % 2 or plan.detail.get("height", 0) % 2:
                    args += ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"]
                args += plan.detail["color"]
            else:
                if plan.action == "copy":
                    args += ["-c:a", "copy"]
                else:
                    args += [
                        "-c:a",
                        plan.to_codec,
                        "-b:a",
                        str(self._stage_config.get("audio_bitrate", "256k")),
                    ]
        # Output option: honour each frame's own PTS (REQUIREMENTS.md § 12.4).
        args += [*timing_output_args(), "-y", output_path]
        return args

    # -- stage API --------------------------------------------------------

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        info = input_info or {}
        if "has_video" in info and not info.get("has_video") and not info.get("has_audio"):
            return False, "No video or audio stream"
        # Cheap pre-check from the probe the pipeline already holds. Only a
        # POSITIVE "nothing to do" short-circuits; absent/partial info defers to
        # execute()'s authoritative probe of the actual input file.
        if "selected_video_codec" in info and "selected_audio_codec" in info:
            video_ok = self._allowed("video_codecs_allowed", ["h264"])
            audio_ok = self._allowed("audio_codecs_allowed", ["aac"])
            v = str(info.get("selected_video_codec") or "").lower()
            a = str(info.get("selected_audio_codec") or "").lower()
            if (not v or v in video_ok) and (not a or a in audio_ok):
                return False, "All kept streams already use allowed codecs"
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        input_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, "Inspecting codecs...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to normalize_codecs stage",
                duration_sec=time.time() - start,
            )

        # Probe the ACTUAL input file, not just input_info: when this stage is
        # forced by name (--stage) or input_info is a hand-built dict, it may
        # not describe input_path at all, and getting the codec wrong here
        # means copying a stream the next stage cannot hold.
        try:
            pr: ProbeResult = probe(input_path, self.config)
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"normalize_codecs could not probe {input_path}: {e}",
                duration_sec=time.time() - start,
            )

        plans = self._plan_from_streams(pr.selected_video_stream, pr.selected_audio_stream)
        streams_meta = [p.describe() for p in plans]
        if not plans:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="No video or audio stream",
                duration_sec=time.time() - start,
            )
        if all(p.action == "copy" for p in plans):
            self._report_progress(1.0, "Codecs already compatible", progress_callback)
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="All kept streams already use allowed codecs",
                metadata={"streams": streams_meta},
                duration_sec=time.time() - start,
            )

        summary = ", ".join(
            f"{p.kind} {p.from_codec}->{p.to_codec}" for p in plans if p.action == "transcode"
        )
        self._report_progress(0.1, f"Normalizing codecs ({summary})...", progress_callback)

        args = self._build_args(input_path, output_path, plans)

        def cb(p: float, m: str) -> None:
            self._report_progress(0.1 + p * 0.9, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())
        if result.returncode != 0 or not os.path.exists(output_path):
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Codec normalization failed: {(result.stderr or '')[-2000:]}",
                metadata={"streams": streams_meta},
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Codec normalization complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={"method": "traditional", "streams": streams_meta},
            duration_sec=time.time() - start,
        )
