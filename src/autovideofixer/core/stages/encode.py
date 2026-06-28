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

    def __init__(self, config):
        super().__init__(config)

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

        try:
            from autovideofixer.core.ffmpeg_utils import (
                build_hwaccel_args,
                resolve_hwaccel,
                run_ffmpeg,
            )

            actual_hwaccel = resolve_hwaccel(hwaccel)
            hw_args = build_hwaccel_args(actual_hwaccel)

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
            if actual_hwaccel != "none" and codec in hw_codec_map:
                codec_for_args = hw_codec_map[codec].get(actual_hwaccel, codec)

            # Build filter chain
            vf = kwargs.get("filter_complex") or ""

            # Build command
            args = hw_args + [
                "-i",
                input_path,
                "-c:v",
                codec_for_args,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                audio_codec,
                "-b:a",
                audio_bitrate,
            ]

            if vf:
                args.extend(["-vf", vf])

            args.extend(["-y", output_path or input_path])

            def cb(p, m):
                self._report_progress(0.2 + p * 0.8, m, progress_callback)

            result = run_ffmpeg(args, progress_callback=cb, timeout=3600)

            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Encoding failed: {result.stderr[:500]}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Encoding complete", progress_callback)
            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path or input_path,
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
