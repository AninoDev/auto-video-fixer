"""Auto Video Fixer - Resolution downscaling stage (REQUIREMENTS.md § 7).

Pure-FFmpeg, no-AI stage that shrinks an OVERSIZED input down to the
configured target resolution box BEFORE the heavier stages (deblock,
denoise_video, upscale, interpolate) run -- avoids spending AI/traditional
compute on pixels a downstream encode would just discard anyway. Off by
default (``stages.downscale.enabled: false``); opt in via
``--downscale``/``stages.downscale.enabled: true``.

Complementary to ``UpscaleStage``, not redundant with it: both stages share
``core.output_check.compute_fitted_dimensions()`` for the actual dimension
math, so for a given target box:

- An OVERSIZED input: downscale shrinks it to the target box; upscale then
  sees an input already at target and skips ("Already at target resolution").
- A SMALL/at-target input: downscale's own ``should_run()`` skips it
  ("Input already at or below target resolution" / "Within downscale
  tolerance"); upscale (if enabled) runs as usual.

Never runs on an input that needs upscaling -- ``should_run()``'s
``binding_ratio >= 1.0`` check is the guard.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from autovideofixer.config import Config
from autovideofixer.core.ffmpeg_utils import probe, run_ffmpeg, timing_output_args
from autovideofixer.core.output_check import SKIP_SCALE_THRESHOLD, compute_fitted_dimensions
from autovideofixer.core.output_check import effective_target_bounds as _effective_target_bounds_fn
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class DownscaleStage(BaseStage):
    """Downscale an oversized video to a target resolution box (traditional/FFmpeg only).

    No AI alternative -- this stage doesn't participate in the ai_fallback
    mechanism, same as CropStage.
    """

    name = "downscale"
    display_name = "Downscaling"
    description = "Shrink an oversized video to the target resolution box"
    category = "enhancement"
    # Between crop (12) and deblock (15): downscale should run after crop
    # (crop can shrink the frame further, changing what "oversized" means)
    # but before the heavier deblock/denoise/upscale/interpolate stages, so
    # those never spend compute on pixels a downscale would shrink away
    # anyway. pipeline.default_order is authoritative for actual execution
    # order -- this is only a fallback ordering signal.
    priority = 13
    supports_gpu = False

    def __init__(self, config: Config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._keep_aspect_ratio = self.config.get(
            "quality", "quality_target", "keep_aspect_ratio", default=True
        )
        self._fit_mode = self.config.get(
            "quality", "quality_target", "resolution_fit_mode", default="preserve_aspect"
        )
        self._dimension_multiple = self.config.get(
            "quality", "quality_target", "dimension_multiple", default=2
        )
        self._snap_tolerance = self.config.get(
            "quality", "quality_target", "snap_tolerance", default=0.01
        )

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"

        target = input_info.get("target_resolution")
        if not target:
            quality_target = self.config.get("quality", "quality_target", default={})
            target = quality_target.get("target_resolution")
        if not target:
            return False, "No target resolution specified"

        w, h = input_info.get("resolution", (0, 0))
        if w <= 0 or h <= 0:
            # No usable resolution info -- defer to execute() rather than
            # blocking the stage outright.
            self._input_info = input_info
            return True, None

        bound_w, bound_h = self._effective_target_bounds(w, h, target[0], target[1])
        binding_ratio = max(bound_w / w, bound_h / h)

        if binding_ratio >= 1.0:
            # Input is not larger than the (orientation-aware) target box --
            # it needs upscaling (or is already at target), never downscaling.
            return False, "Input already at or below target resolution"
        if binding_ratio > 1.0 / SKIP_SCALE_THRESHOLD:
            # Only marginally larger than target (within ~5% linear) -- not
            # worth a re-encode for a negligible size reduction.
            return False, "Within downscale tolerance"

        self._input_info = input_info
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
        self._report_progress(0.0, "Running downscaling...", progress_callback)

        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to downscale stage",
                duration_sec=time.time() - start,
            )

        input_w, input_h = self._get_input_resolution(input_path)

        target_width = kwargs.get("target_width")
        target_height = kwargs.get("target_height")
        if target_width is None or target_height is None:
            quality_target = self.config.get("quality", "quality_target", default={})
            tr = quality_target.get("target_resolution")
            if tr:
                target_width = target_width or tr[0]
                target_height = target_height or tr[1]

        if not target_width or not target_height:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="No target resolution specified",
                duration_sec=time.time() - start,
            )

        final_w, final_h = self._target_dimensions(input_w, input_h, target_width, target_height)
        scale_expr = f"scale={final_w}:{final_h}:flags=lanczos"
        vf_filter = f"{scale_expr},format=yuv420p"
        args = [
            "-i",
            input_path,
            "-vf",
            vf_filter,
            *timing_output_args(),
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

        def cb(p: float, m: str) -> None:
            self._report_progress(0.2 + p * 0.8, m, progress_callback)

        result = run_ffmpeg(args, progress_callback=cb, timeout=self.stage_timeout())

        if result.returncode != 0:
            return StageResult(
                status=StageStatus.FAILED,
                error=f"Downscaling failed: {result.stderr[:2000]}",
                duration_sec=time.time() - start,
            )

        self._report_progress(1.0, "Downscaling complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata={
                "method": "traditional",
                "scale_expr": scale_expr,
                "resolution": [final_w, final_h],
            },
            duration_sec=time.time() - start,
        )

    def _target_dimensions(
        self, in_w: int, in_h: int, target_width: int, target_height: int
    ) -> tuple[int, int]:
        """Compute the fitted output dimensions for ``in_w``x``in_h`` against
        ``target_width``x``target_height``, per the configured
        keep_aspect_ratio/resolution_fit_mode/dimension_multiple/
        snap_tolerance. Extracted so it's unit-testable without ffmpeg."""
        return compute_fitted_dimensions(
            in_w,
            in_h,
            target_width,
            target_height,
            self._keep_aspect_ratio,
            self._fit_mode,
            self._dimension_multiple,
            self._snap_tolerance,
        )

    def _effective_target_bounds(
        self, input_width: int, input_height: int, target_width: int, target_height: int
    ) -> tuple[int, int]:
        """Return the target bounding box, rotated to match input orientation.

        Thin wrapper around the shared ``core.output_check.effective_target_bounds()``
        -- see ``UpscaleStage._effective_target_bounds`` for the same pattern.
        """
        return _effective_target_bounds_fn(
            input_width, input_height, target_width, target_height, self._keep_aspect_ratio
        )

    def _get_input_resolution(self, path: str) -> tuple[int, int]:
        """Get input video resolution."""
        try:
            info = probe(path)
            return info.resolution
        except Exception:
            return (1920, 1080)
