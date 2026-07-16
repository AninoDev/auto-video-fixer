"""Auto Video Fixer - Video encoding stage."""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class EncodeStage(BaseStage):
    """Encode video to target format with quality controls.

    Supports software and hardware encoding, various codecs,
    and VMAF-based quality targeting.
    """

    name = "encode"
    display_name = "Encoding"
    description = "Encode video to target format and quality"
    category = "encoding"
    priority = 90
    supports_hardware_encoding = True

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        # Encode is always the final stage
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
        hwaccel: str = "auto",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        **kwargs,
    ) -> StageResult:
        start = time.time()
        self._report_progress(0.0, f"Encoding with {codec}...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to encode stage",
                duration_sec=time.time() - start,
            )

        try:
            from autovideofixer.core.ffmpeg_utils import (
                build_hwaccel_args,
                resolve_hwaccel,
                run_ffmpeg,
            )

            actual_hwaccel = resolve_hwaccel(hwaccel)
            # We only have hardware *encoders* mapped for cuda/vaapi/qsv/videotoolbox
            # (hw_codec_map below); d3d11/vulkan have no encoder mapping, so falling
            # through to a software encoder. Requesting `-hwaccel d3d11`/`-hwaccel
            # vulkan` for decode while encoding in software feeds hw-decoded frames to
            # a sw encoder with no hwdownload/format filter -- treat these as "none"
            # for hwaccel arg-building so decode stays consistent with the encoder.
            hw_args = build_hwaccel_args(
                actual_hwaccel if actual_hwaccel not in ("d3d11", "vulkan") else "none"
            )

            # Map software codec to hardware codec for hwaccel
            hw_codec_map = {
                "libx264": {
                    "cuda": "h264_nvenc",
                    "vaapi": "h264_vaapi",
                    "qsv": "h264_qsv",
                    "videotoolbox": "h264_videotoolbox",
                },
                "libx265": {
                    "cuda": "hevc_nvenc",
                    "vaapi": "hevc_vaapi",
                    "qsv": "hevc_qsv",
                    "videotoolbox": "hevc_videotoolbox",
                },
                "libvpx-vp9": {
                    "vaapi": "vp9_vaapi",
                    "qsv": "vp9_qsv",
                },
                "libvpx": {
                    "vaapi": "vp8_vaapi",
                },
            }

            codec_for_args = codec
            if actual_hwaccel not in ("none", "d3d11", "vulkan") and codec in hw_codec_map:
                codec_for_args = hw_codec_map[codec].get(actual_hwaccel, codec)

            # Build filter chain
            vf = kwargs.get("filter_complex") or ""

            # Rate-control flags differ per encoder family: libx264/libx265/libvpx
            # accept -preset/-crf, but the hardware encoders above don't (nvenc uses
            # -cq, vaapi uses -qp with no -preset at all, qsv uses -global_quality with
            # no -preset, videotoolbox has neither and falls back to bitrate control).
            # Passing -preset/-crf unconditionally makes ffmpeg reject the command
            # outright on every hwaccel path.
            if codec_for_args.endswith("_nvenc"):
                rate_args = ["-preset", preset, "-cq", str(crf)]
            elif codec_for_args.endswith("_vaapi"):
                rate_args = ["-qp", str(crf)]
            elif codec_for_args.endswith("_qsv"):
                rate_args = ["-global_quality", str(crf)]
            elif codec_for_args.endswith("_videotoolbox"):
                rate_args = []
            else:
                rate_args = ["-preset", preset, "-crf", str(crf)]

            # Build command
            args = (
                hw_args
                + [
                    "-i",
                    input_path,
                    "-c:v",
                    codec_for_args,
                ]
                + rate_args
                + [
                    "-c:a",
                    audio_codec,
                    "-b:a",
                    audio_bitrate,
                ]
            )

            if vf:
                args.extend(["-vf", vf])

            args.extend(["-y", output_path])

            def cb(p, m):
                self._report_progress(0.2 + p * 0.8, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Encoding failed: {result.stderr[:500]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Encoding complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "codec": codec,
                    "preset": preset,
                    "crf": crf,
                    "hwaccel": actual_hwaccel,
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
