"""Auto Video Fixer - Video/Audio detection stage.

Analyzes input files to determine what processing is needed.
"""

from __future__ import annotations

from typing import Any

from autovideofixer.core.ffmpeg_utils import ProbeResult, probe
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class DetectStage(BaseStage):
    """Detect and analyze video properties to determine needed processing.

    This stage runs first to gather detailed information about the input
    that subsequent stages can use for decision-making.
    """

    name = "detect"
    display_name = "Analysis & Detection"
    description = "Analyze video to determine required processing stages"
    category = "analysis"
    priority = 1
    requires_input = True
    produces_output = False
    can_parallelize = False

    def __init__(self, config):
        super().__init__(config)

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        # Always run detection first
        return True, None

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        **kwargs,
    ) -> StageResult:
        import time

        start = time.time()
        self._report_progress(0.0, "Analyzing video properties...", progress_callback)

        try:
            info = probe(input_path)
            metadata = self._build_detection_metadata(info)
            self._report_progress(1.0, "Analysis complete", progress_callback)

            return StageResult(
                status=StageStatus.COMPLETED,
                metadata=metadata,
                duration_sec=time.time() - start,
            )
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )

    def _build_detection_metadata(self, info: ProbeResult) -> dict[str, Any]:
        """Build comprehensive detection metadata from probe results."""
        vs = info.video_stream
        metadata = {
            "has_video": info.has_video,
            "has_audio": info.has_audio,
            "duration": info.duration,
            "format": info.format_name,
            "resolution": info.resolution,
            "width": vs.width if vs else 0,
            "height": vs.height if vs else 0,
            "framerate": info.framerate,
            "is_hdr": info.is_hdr,
            "video_codec": vs.codec_name if vs else "",
            "pixel_format": vs.pixel_format if vs else "",
            "color_space": vs.color_space if vs else "",
            "color_transfer": vs.color_transfer if vs else "",
            "bit_rate": info.bit_rate,
            "audio_count": len(info.audio_streams),
            "audio_codecs": [s.codec_name for s in info.audio_streams],
            "needs_processing": [],
        }

        # Auto-detect processing needs
        needs = metadata["needs_processing"]

        if info.is_hdr:
            needs.append("hdr")

        if info.framerate < 30 and info.framerate > 0:
            needs.append("interpolate")

        if info.format_name in ("matroska", "mkv"):
            needs.append("remux")

        # Check for potential shake (requires analysis)
        metadata["shake_detected"] = False  # Will be set by shake analysis

        return metadata
