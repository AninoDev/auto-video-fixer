"""Auto Video Fixer - Video stabilization/deshaking stage."""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Any

from autovideofixer.core.ffmpeg_utils import run_ffmpeg
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class StabilizeStage(BaseStage):
    """Detect and correct video shake using FFmpeg's vidstab filters.

    Automatically detects shake intensity and applies correction only when needed.
    Also detects scene changes for segmented processing.
    """

    name = "stabilize"
    display_name = "Stabilization"
    description = "Detect and correct camera shake"
    category = "enhancement"
    priority = 10
    supports_hardware_encoding = False

    def __init__(self, config, overrides: dict[str, Any] | None = None):
        super().__init__(config, overrides)
        self._threshold = self._stage_config.get("threshold", 2.0)
        self._smoothness = self._stage_config.get("smoothness", 40)
        self._maxshift = self._stage_config.get("maxshift", 20)
        self._scene_threshold = self._stage_config.get("scene_threshold", 0.98)
        self._zoom_enabled = self._stage_config.get("zoom_enabled", True)
        self._zoom_mode = self._stage_config.get("zoom_mode", "black")
        self._zoom_threshold = self._stage_config.get("zoom_threshold", 50.0)  # pixels
        # 1.0 = today's optzoom=1 behavior (fill frame for every frame); 0.0 = no
        # zoom; in between = a static zoom sized to the zoom_coverage-quantile of
        # per-frame required-zoom estimates. See _compute_static_zoom_pct().
        self._zoom_coverage = max(
            0.0, min(1.0, float(self._stage_config.get("zoom_coverage", 1.0)))
        )
        self._sharpen_enabled = self._stage_config.get("sharpen_enabled", True)
        self._optalgo = self._stage_config.get("optalgo", "gauss")
        self._shakiness = self._stage_config.get("shakiness", 10)
        self._pipe_timeout = self._stage_config.get("pipe_timeout", 1800)

        self.logger.debug(
            f"StabilizeStage initialized: smoothness={self._smoothness}, "
            f"maxshift={self._maxshift}, shakiness={self._shakiness}, "
            f"threshold={self._threshold}"
        )

    def should_run(self, input_info: dict[str, Any]) -> tuple[bool, str | None]:
        if not self.is_enabled():
            return False, "Stage disabled"
        return True, None

    def _analyze_trf_file(self, trf_path: str, threshold: float) -> tuple[bool, float]:
        """Parse vidstabdetect output to determine shake intensity.

        The TRF file uses format:
            Frame N (List M [(LM dx dy x y w h contrast magnitude),...])

        Returns:
            (needs_stabilization, avg_magnitude)
        """
        try:
            with open(trf_path, "r") as f:
                content = f.read()

            # Extract all LM entries: (LM dx dy x y w h contrast magnitude)
            lm_pattern = r"\(LM\s+(-?\d+)\s+(-?\d+)\s+"
            matches = re.findall(lm_pattern, content)

            if not matches:
                return False, 0.0

            magnitudes = []
            for dx_str, dy_str in matches:
                dx = float(dx_str)
                dy = float(dy_str)
                magnitude = (dx**2 + dy**2) ** 0.5
                magnitudes.append(magnitude)

            avg_mag = sum(magnitudes) / len(magnitudes)

            # Count significant movements (> 2 pixels)
            significant = [m for m in magnitudes if m > 2.0]
            significant_pct = len(significant) / len(magnitudes) if magnitudes else 0.0

            # Needs stabilization if average magnitude exceeds threshold
            # or if significant percentage is high
            needs_stab = (avg_mag > threshold) or (significant_pct > 0.5)

            return needs_stab, avg_mag

        except Exception:
            return False, 0.0

    def _get_video_dimensions(self, input_path: str) -> tuple[int, int]:
        """Get video width and height from ffprobe."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0",
                    input_path,
                ],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                text=True,
            )

            if result.stdout and "," in result.stdout:
                parts = result.stdout.strip().split(",")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 1920, 1080

    def _get_video_framerate(self, input_path: str) -> float:
        """Get video framerate from ffprobe.

        Reads avg_frame_rate, not r_frame_rate. r_frame_rate is ffmpeg's
        "declared"/tbr rate, which for VFR or YouTube-origin sources can be
        a multiple of the true average rate (e.g. r_frame_rate=59.94 tbr vs.
        avg_frame_rate=29.64 for a real-world 29.64fps-average clip). This
        value is fed to the raw-pipe decode->transform handoff as an input
        `-r`, which forcibly re-times the piped (timestamp-less) raw frames.
        Using the inflated r_frame_rate there compresses N real frames into
        N/2 seconds of output -- the video plays back at ~2x speed and, once
        muxed against the original (correctly-timed) audio track, freezes on
        the last frame for the remainder of the audio. avg_frame_rate is the
        honest "real frame count / real duration" rate and is what must be
        used any time frame count is being reconciled with wall-clock time.
        Falls back to r_frame_rate only if avg_frame_rate is unavailable
        (e.g. "0/0", which ffprobe emits when duration is unknown).
        """
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate",
                    "-of",
                    "csv=p=0",
                    input_path,
                ],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                text=True,
            )

            if not result.stdout:
                return 30.0

            # csv order matches -show_entries order: avg_frame_rate,r_frame_rate.
            # Prefer avg_frame_rate; fall back to r_frame_rate only if avg is
            # missing/undefined (e.g. "0/0").
            fields = result.stdout.strip().split(",")
            for field in fields:
                if "/" not in field:
                    continue
                num_str, den_str = field.split("/")
                num, den = float(num_str), float(den_str)
                if den and num:
                    return num / den
            return 30.0
        except Exception:
            return 30.0

    def _clean_trf_outliers(self, trf_path: str) -> str:
        """Remove outlier frames from TRF file to prevent artifact bursts.

        Returns path to cleaned TRF file.
        """
        import tempfile

        try:
            with open(trf_path, "r") as f:
                content = f.read()

            # Parse TRF file into structured data
            # Format: Frame N (List M [(LM dx dy x y w h contrast magnitude),...])
            frames_data = {}
            current_frame = None

            for line in content.split("\n"):
                if line.startswith("Frame "):
                    match = re.match(r"Frame (\d+)", line)
                    if match:
                        current_frame = int(match.group(1))
                        frames_data[current_frame] = []
                elif line.startswith("(") and current_frame is not None:
                    # Parse LM entries
                    lm_matches = re.findall(r"\(LM\s+(-?\d+)\s+(-?\d+)\s+", line)
                    for dx, dy in lm_matches:
                        frames_data[current_frame].append((int(dx), int(dy)))

            if not frames_data:
                return trf_path

            # Calculate average movement magnitude for each frame
            frame_magnitudes = {}
            for frame, lms in frames_data.items():
                if lms:
                    total_mag = sum((dx**2 + dy**2) ** 0.5 for dx, dy in lms)
                    frame_magnitudes[frame] = total_mag / len(lms)

            # Calculate median magnitude
            magnitudes = list(frame_magnitudes.values())
            if not magnitudes:
                return trf_path

            from statistics import median

            median_mag = median(magnitudes)

            # Threshold: frames with magnitude > 3x median are outliers
            threshold = median_mag * 3

            # Identify outlier frames
            outlier_frames = {frame for frame, mag in frame_magnitudes.items() if mag > threshold}

            if not outlier_frames:
                return trf_path

            self.logger.info(f"Cleaned {len(outlier_frames)} outlier frames from TRF")

            # Rebuild TRF file with cleaned data
            lines = content.split("\n")
            cleaned_lines = []

            for line in lines:
                if line.startswith("Frame "):
                    match = re.match(r"Frame (\d+)", line)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num in outlier_frames:
                            # Replace this frame's LM values with zeros
                            new_line = re.sub(r"\(LM\s+-?\d+\s+-?\d+\s+", "(LM 0 0 ", line)
                            cleaned_lines.append(new_line)
                        else:
                            cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            # Write cleaned TRF to temp file
            cleaned_fd, cleaned_path = tempfile.mkstemp(suffix=".trf", prefix="avf_clean_")
            os.close(cleaned_fd)

            with open(cleaned_path, "w") as f:
                f.write("\n".join(cleaned_lines))

            return cleaned_path

        except Exception as e:
            self.logger.warning(f"TRF cleaning failed: {e}")
            return trf_path

    def _movement_extent(self, trf_path: str) -> float:
        """Return the max dx/dy excursion (pixels) across all LM entries in a TRF file.

        This is only used to *gate* whether digital zoom compensation is
        worthwhile at all (via ``zoom_threshold``) -- not to compute the zoom
        amount itself. vidstabtransform's own ``optzoom`` (see
        ``execute()``) is used for the actual zoom amount: it operates on the
        smoothed camera path it computes internally, which is what actually
        determines the visible border, whereas the raw per-block LM
        dx/dy values parsed here are frame-to-frame local-motion estimates
        that don't reflect the cumulative/smoothed excursion vidstabtransform
        will apply. Hand-deriving a zoom percentage from them (the previous
        implementation) both used the wrong signal and had an inverted sign
        (it produced values in [-20, 0], i.e. it could only zoom OUT or do
        nothing -- vidstabtransform's `zoom` option is >0 = zoom in, <0 =
        zoom out -- so it was structurally incapable of ever removing a
        border).
        """
        try:
            with open(trf_path, "r") as f:
                content = f.read()

            lm_pattern = r"\(LM\s+(-?\d+)\s+(-?\d+)\s+"
            matches = re.findall(lm_pattern, content)

            if not matches:
                return 0.0

            dx_values = [float(m[0]) for m in matches]
            dy_values = [float(m[1]) for m in matches]

            max_dx = max(dx_values) - min(dx_values)
            max_dy = max(dy_values) - min(dy_values)
            return max(max_dx, max_dy)

        except Exception as e:
            self.logger.warning(f"Movement extent calculation failed: {e}")
            return 0.0

    def _compute_static_zoom_pct(
        self, trf_path: str, video_width: int, video_height: int, coverage: float
    ) -> float:
        """Approximate the static vidstabtransform ``zoom=<pct>`` needed so that
        ``coverage`` fraction of frames end up border-free.

        LIMITATION (state this honestly, do not treat this as exact): what
        actually determines each frame's visible border is
        vidstabtransform's own internally-computed SMOOTHED camera path (a
        function of smoothing/maxshift/optalgo/interpol), which this method
        has no access to. All it can see is the raw per-block local-motion
        (LM) values in the TRF file -- frame-to-frame motion estimates, not
        the smoothed path -- which are integrated here into an approximate
        cumulative camera-path position. The quantile of these raw,
        un-smoothed excursions is a DIRECTIONALLY CORRECT approximation of
        the quantile of the true smoothed-path excursions, not an exact
        match: individual frames' actual post-smoothing border requirements
        can come out higher or lower than this estimate. This is exactly
        why zoom_coverage is a tunable dial and not a guarantee once
        coverage < 1.0 -- at coverage=1.0 the exact optzoom=1 delegation is
        used instead (no approximation involved). This also does NOT
        account for rotation's contribution to border size -- the TRF's
        rotation component, if present, is ignored here, same limitation as
        _movement_extent().

        Returns a zoom percentage suitable for vidstabtransform's
        ``zoom=`` option (>0 = zoom in), or 0.0 if the TRF can't be parsed
        or has no LM data.
        """
        try:
            with open(trf_path, "r") as f:
                content = f.read()

            # Real vidstabdetect TRF output puts a frame's "Frame N (List M
            # [...])" header and its LM entries on ONE line (as all the
            # existing TRF-derived tests in this file build their fixtures);
            # the LM search below runs on every line unconditionally
            # (not `elif`) so it still finds entries whether they're on the
            # same line as the "Frame N" header or, hypothetically, a
            # continuation line.
            frames_data: dict[int, list[tuple[float, float]]] = {}
            current_frame = None
            for line in content.split("\n"):
                match = re.match(r"Frame (\d+)", line)
                if match:
                    current_frame = int(match.group(1))
                    frames_data.setdefault(current_frame, [])
                if current_frame is not None:
                    for dx, dy in re.findall(r"\(LM\s+(-?\d+)\s+(-?\d+)\s+", line):
                        frames_data[current_frame].append((float(dx), float(dy)))

            if not frames_data:
                return 0.0

            ordered_frames = sorted(frames_data)
            avg_dx = []
            avg_dy = []
            for fr in ordered_frames:
                lms = frames_data[fr]
                if lms:
                    avg_dx.append(sum(d[0] for d in lms) / len(lms))
                    avg_dy.append(sum(d[1] for d in lms) / len(lms))
                else:
                    avg_dx.append(0.0)
                    avg_dy.append(0.0)

            if not avg_dx:
                return 0.0

            # Integrate frame-to-frame local motion into an approximate raw
            # cumulative camera-path position -- see the LIMITATION note
            # above for why this is an approximation of vidstabtransform's
            # actual path, not the path itself.
            cum_x = []
            cum_y = []
            cx = cy = 0.0
            for dx, dy in zip(avg_dx, avg_dy):
                cx += dx
                cy += dy
                cum_x.append(cx)
                cum_y.append(cy)

            # What actually determines the visible border per frame is the
            # GAP between the raw path and vidstabtransform's SMOOTHED
            # path -- not the raw path's deviation from a single global
            # center. A plain global median (an earlier version of this
            # method) treats the whole clip as one static reference point,
            # which lets the raw path's own random-walk-like drift over a
            # long clip inflate every frame's "required zoom" far past what
            # vidstabtransform actually needs (verified empirically: it
            # overshot vidstabtransform's own logged "Final zoom" for
            # optzoom=1 by ~4-5x on a synthetic test clip). A local moving
            # average over a window matching `smoothness` (the same
            # smoothing window vidstabtransform itself uses, see __init__)
            # approximates that smoothed path much more directly, so the
            # deviation used below is close to the actual quantity
            # vidstabtransform computes -- still an approximation (see the
            # LIMITATION note above), but a substantially tighter one.
            window = max(1, self._smoothness)
            n_frames = len(cum_x)

            def _local_avg(series: list[float], i: int) -> float:
                lo = max(0, i - window // 2)
                hi = min(n_frames, i + window // 2 + 1)
                return sum(series[lo:hi]) / (hi - lo)

            smoothed_x = [_local_avg(cum_x, i) for i in range(n_frames)]
            smoothed_y = [_local_avg(cum_y, i) for i in range(n_frames)]

            # For a frame shifted by `shift` px off the smoothed path along a
            # dimension of size `dim`, vidstabtransform's zoom=Z% scales the
            # frame by (1 + Z/100) around its center; the shifted edge is
            # only fully covered once (Z/100) * dim/2 >= shift, i.e. Z >=
            # 200 * shift / dim. Required zoom for a frame is the max of its
            # x/y needs (zoom is applied uniformly, not per-axis).
            required_pct = []
            for x, sx, y, sy in zip(cum_x, smoothed_x, cum_y, smoothed_y):
                shift_x = abs(x - sx)
                shift_y = abs(y - sy)
                pct_x = (200.0 * shift_x / video_width) if video_width else 0.0
                pct_y = (200.0 * shift_y / video_height) if video_height else 0.0
                required_pct.append(max(pct_x, pct_y))

            required_pct.sort()
            n = len(required_pct)
            coverage = max(0.0, min(1.0, coverage))
            idx = min(n - 1, int(round(coverage * (n - 1))))
            return max(0.0, required_pct[idx])

        except Exception as e:
            self.logger.warning(f"Static zoom computation failed: {e}")
            return 0.0

    def _detect_scenes(self, input_path: str, scene_threshold: float = 0.98) -> list[float]:
        """Detect scene changes by comparing consecutive frames.

        Returns:
            List of timestamps (in seconds) where scene changes occur.
        """
        import tempfile

        import numpy as np
        from PIL import Image

        scene_changes = []
        temp_dir = tempfile.mkdtemp(prefix="avf_scene_")

        try:
            # Extract frames at 1 fps to reduce processing. Frame i of this
            # sampled sequence corresponds to timestamp ~i seconds -- do NOT
            # index into ffprobe's full per-decoded-frame pts_time array with
            # this same counter, its length/stride has nothing to do with the
            # 1fps-sampled sequence and produces timestamps compressed into
            # the first few seconds of the video regardless of actual length.
            frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

            extract_args = ["-i", input_path, "-vf", "fps=1", "-q:v", "2", frame_pattern]

            run_ffmpeg(extract_args, timeout=120)

            # Load and compare consecutive frames
            import glob

            frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

            if len(frame_files) < 2:
                return []

            prev_frame = None

            for i, frame_file in enumerate(frame_files):
                if i == 0:
                    prev_frame = np.array(Image.open(frame_file).convert("L"))
                    continue

                curr_frame = np.array(Image.open(frame_file).convert("L"))

                # Calculate structural similarity or correlation
                if prev_frame.shape != curr_frame.shape:
                    prev_frame = curr_frame
                    continue

                # Normalize frames
                prev_norm = prev_frame.astype(float) / 255.0
                curr_norm = curr_frame.astype(float) / 255.0

                # Calculate correlation
                correlation = np.corrcoef(prev_norm.flatten(), curr_norm.flatten())[0, 1]

                # If correlation is low, it's a scene change
                if correlation < scene_threshold and i > 0:
                    scene_changes.append(float(i))

                prev_frame = curr_frame

        except Exception as e:
            self.logger.warning(f"Scene detection failed: {e}")
        finally:
            # Cleanup
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        return scene_changes

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback=None,
        threshold: float | None = None,
        smoothness: int | None = None,
        input_info: dict | None = None,
        **kwargs,
    ) -> StageResult:
        start = time.time()
        thresh = threshold if threshold is not None else self._threshold
        smooth = smoothness if smoothness is not None else self._smoothness

        trf_path = None
        try:
            trf_fd, trf_path = tempfile.mkstemp(suffix=".trf", prefix="avf_stab_")
            os.close(trf_fd)

            self._report_progress(0.1, "Detecting shake...", progress_callback)

            # Step 1: Detect motion (use ascii format for parsing)
            detection_args = [
                "-i",
                input_path,
                "-vf",
                f"vidstabdetect=shakiness={self._shakiness}:accuracy=15:result={trf_path}:fileformat=ascii",
                "-f",
                "null",
                "-",
            ]
            det_result = run_ffmpeg(detection_args, timeout=300)

            if det_result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Shake detection failed: {det_result.stderr[:200]}",
                    duration_sec=time.time() - start,
                )

            # Step 2: Analyze shake intensity
            needs_stab, avg_value = self._analyze_trf_file(trf_path, thresh)
            self.logger.debug(
                f"Shake analysis: needs_stab={needs_stab}, "
                f"avg_value={avg_value:.3f}, threshold={thresh}"
            )

            # Step 2.5: Detect scene changes
            self._report_progress(0.2, "Detecting scenes...", progress_callback)
            scene_changes = self._detect_scenes(input_path, self._scene_threshold)

            # Step 2.6: Calculate zoom if needed
            self._report_progress(0.25, "Analyzing movement extent...", progress_callback)
            video_width, video_height = self._get_video_dimensions(input_path)
            # Whether to let vidstabtransform apply its own optimal zoom
            # (see the `zoom_param` construction below for why we delegate
            # the actual zoom *amount* to vidstabtransform's optzoom rather
            # than hand-computing a percentage).
            apply_zoom = False
            if self._zoom_enabled and needs_stab:
                movement = self._movement_extent(trf_path)
                apply_zoom = movement >= self._zoom_threshold
                self.logger.debug(
                    f"Movement extent: {movement:.2f}px, threshold={self._zoom_threshold}, "
                    f"apply_zoom={apply_zoom}"
                )
            else:
                self.logger.debug(
                    f"Zoom skipped: enabled={self._zoom_enabled}, needs_stab={needs_stab}"
                )

            if not needs_stab:
                self.logger.info(
                    f"Skipping stabilization (avg shake: {avg_value:.3f} < threshold: {thresh})"
                )
                self._report_progress(
                    1.0, f"No stabilization needed (avg: {avg_value:.3f})", progress_callback
                )
                # Skip stabilization, just copy the input
                run_ffmpeg(
                    [
                        "-hide_banner",
                        "-i",
                        input_path,
                        "-c",
                        "copy",
                        "-y",
                        output_path,
                    ],
                    timeout=120,
                )
                return StageResult(
                    status=StageStatus.COMPLETED,
                    output_path=output_path,
                    metadata={
                        "smoothness": smooth,
                        "threshold": thresh,
                        "skipped": True,
                        "avg_shake": avg_value,
                        "scene_changes": scene_changes,
                        "num_scenes": len(scene_changes) + 1 if scene_changes else 1,
                    },
                    duration_sec=time.time() - start,
                    skipped_reason=f"No stabilization needed (avg shake: {avg_value:.3f})",
                )

            # Step 3: Clean TRF outliers
            self._report_progress(0.5, "Cleaning TRF outliers...", progress_callback)
            clean_trf_path = self._clean_trf_outliers(trf_path)

            # Step 4 & 5: Pipe raw video from decode to stabilization
            # This avoids writing huge raw files to disk and prevents vid.stab
            # from corrupting decoder reference frames (B-frame issue)
            # vidstab only supports yuv420p, so we convert regardless of source format
            pixel_format = "yuv420p"
            framerate = self._get_video_framerate(input_path)

            crop_mode = "black" if self._zoom_mode == "black" else "keep"
            # Delegate the actual zoom amount to vidstabtransform's built-in
            # optzoom rather than a hand-computed percentage: optzoom
            # operates on vidstabtransform's own smoothed camera path (the
            # thing that actually determines the visible border after
            # smoothing/maxshift/optalgo are applied), so it reliably
            # eliminates borders regardless of how the raw per-block LM
            # values in the TRF relate to the final transform. optzoom=1
            # ("optimal static zoom") picks a single constant zoom factor
            # sufficient to cover the worst frame in the whole clip -- no
            # borders can ever appear, at the cost of being not-as-tight as
            # a per-frame adaptive zoom. zoom=0 leaves the zoom amount to
            # optzoom to decide; explicit optzoom=0 disables zoom entirely
            # when zoom_enabled=False or movement is below zoom_threshold,
            # preserving prior "no zoom, borders acceptable" behavior for
            # zoom_enabled=False (vidstabtransform's own default is
            # optzoom=1, so it must be explicitly zeroed here).
            #
            # zoom_coverage (0.0-1.0) shapes WHAT zoom is applied once
            # apply_zoom (the movement-extent gate above) has already
            # decided zoom applies at all -- it does not change the gate
            # itself. 1.0 (default) = today's exact optzoom=1 behavior
            # (guaranteed no border on any frame). 0.0 = no zoom (all
            # borders visible, same as apply_zoom=False). In between: a
            # static zoom= percentage computed from the zoom_coverage-th
            # quantile of per-frame required-zoom estimates (see
            # _compute_static_zoom_pct's docstring for the accuracy
            # limitation -- it approximates vidstabtransform's own smoothed
            # camera path from the raw TRF, it does not read it directly).
            static_zoom_pct = None
            if not apply_zoom:
                zoom_param = ":zoom=0:optzoom=0"
            elif self._zoom_coverage >= 1.0:
                zoom_param = ":zoom=0:optzoom=1"
            elif self._zoom_coverage <= 0.0:
                zoom_param = ":zoom=0:optzoom=0"
            else:
                static_zoom_pct = self._compute_static_zoom_pct(
                    clean_trf_path, video_width, video_height, self._zoom_coverage
                )
                zoom_param = f":zoom={static_zoom_pct:.4f}:optzoom=0"
                self.logger.debug(
                    f"zoom_coverage={self._zoom_coverage} -> static zoom={static_zoom_pct:.4f}%"
                )

            # Build filter chain with optional sharpening
            stab_filter = (
                f"vidstabtransform=smoothing={smooth}:input={clean_trf_path}:"
                f"crop={crop_mode}:interpol=bilinear:maxshift={self._maxshift}:"
                f"optalgo={self._optalgo}{zoom_param}"
            )

            # Add sharpening if stabilization was auto-triggered (not user-disabled)
            auto_triggered = needs_stab and self._sharpen_enabled
            if auto_triggered:
                stab_filter = f"{stab_filter},unsharp=3:3:0.5:3:3:0.0"

            self._report_progress(0.6, "Stabilizing (pipe decode→transform)...", progress_callback)
            self.logger.info(
                f"Piping raw video: {video_width}x{video_height} @ {framerate}fps, "
                f"format={pixel_format}"
            )
            self.logger.debug(f"vidstabtransform filter: {stab_filter}")
            self.logger.debug(
                f"Config: smoothness={self._smoothness}, maxshift={self._maxshift}, "
                f"zoom_enabled={self._zoom_enabled}, zoom_mode={self._zoom_mode}, "
                f"apply_zoom={apply_zoom}"
            )

            # Use subprocess to pipe decode stdout → transform stdin
            import subprocess as sp

            from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path

            ffmpeg_bin = get_ffmpeg_path()

            # Decode process: outputs raw video (no audio) to stdout
            # NOTE: -s explicitly sets output size to match transform's -s input
            # This prevents corruption when ffprobe returns incorrect dimensions
            decode_proc = sp.Popen(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-i",
                    input_path,
                    "-vf",
                    f"format={pixel_format}",
                    "-c:v",
                    "rawvideo",
                    "-f",
                    "rawvideo",
                    "-s",
                    f"{video_width}x{video_height}",
                    "-bufsize",
                    "10M",
                    "-",
                ],
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )

            # Check if audio exists in input
            has_audio = input_info and input_info.get("has_audio", False) if input_info else False

            # Transform process: reads raw video from stdin, audio from original file
            if has_audio:
                transform_proc = sp.Popen(
                    [
                        ffmpeg_bin,
                        "-hide_banner",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        pixel_format,
                        "-s",
                        f"{video_width}x{video_height}",
                        "-r",
                        str(framerate),
                        "-i",
                        "-",
                        "-i",
                        input_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a:0",
                        "-vf",
                        stab_filter,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-c:a",
                        "copy",
                        "-y",
                        output_path,
                    ],
                    stdin=decode_proc.stdout,
                    stderr=sp.PIPE,
                )
            else:
                transform_proc = sp.Popen(
                    [
                        ffmpeg_bin,
                        "-hide_banner",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        pixel_format,
                        "-s",
                        f"{video_width}x{video_height}",
                        "-r",
                        str(framerate),
                        "-i",
                        "-",
                        "-i",
                        input_path,
                        "-map",
                        "0:v",
                        "-vf",
                        stab_filter,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "18",
                        "-an",
                        "-y",
                        output_path,
                    ],
                    stdin=decode_proc.stdout,
                    stderr=sp.PIPE,
                )

            # Close decode's stdout in parent - only transform reads it
            decode_proc.stdout.close()

            # Drain both processes' stderr concurrently via background threads
            # instead of sequential communicate(). Draining decode then
            # transform in sequence deadlocks: if transform writes enough to
            # stderr (e.g. elevated logging on a long/slow transcode) to fill
            # the OS pipe buffer before decode exits, transform blocks on its
            # own stderr write, stops reading stdin, decode blocks on its
            # stdout write, and neither process ever exits.
            import threading

            decode_stderr_chunks: list[bytes] = []
            transform_stderr_chunks: list[bytes] = []

            def _drain(pipe, sink: list[bytes]) -> None:
                try:
                    for chunk in iter(lambda: pipe.read(65536), b""):
                        sink.append(chunk)
                except OSError, ValueError:
                    pass
                finally:
                    try:
                        pipe.close()
                    except OSError:
                        pass

            decode_stderr_thread = threading.Thread(
                target=_drain, args=(decode_proc.stderr, decode_stderr_chunks), daemon=True
            )
            transform_stderr_thread = threading.Thread(
                target=_drain, args=(transform_proc.stderr, transform_stderr_chunks), daemon=True
            )
            decode_stderr_thread.start()
            transform_stderr_thread.start()

            def _kill_both() -> None:
                for proc in (decode_proc, transform_proc):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                decode_stderr_thread.join(timeout=5)
                transform_stderr_thread.join(timeout=5)

            try:
                decode_proc.wait(timeout=self._pipe_timeout)
            except sp.TimeoutExpired:
                _kill_both()
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Stabilization decode process timed out after {self._pipe_timeout}s",
                    duration_sec=time.time() - start,
                )

            try:
                transform_proc.wait(timeout=self._pipe_timeout)
            except sp.TimeoutExpired:
                _kill_both()
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Stabilization transform process timed out after {self._pipe_timeout}s",
                    duration_sec=time.time() - start,
                )

            decode_stderr_thread.join(timeout=5)
            transform_stderr_thread.join(timeout=5)
            decode_stderr = b"".join(decode_stderr_chunks)
            transform_stderr = b"".join(transform_stderr_chunks)

            decode_stderr_text = decode_stderr[:5000].decode(errors="replace")
            transform_stderr_text = transform_stderr[:5000].decode(errors="replace")

            if decode_proc.returncode != 0:
                self.logger.error(f"Decode failed: {decode_stderr_text}")
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Raw decode failed: {decode_stderr_text}",
                    duration_sec=time.time() - start,
                )

            if transform_proc.returncode != 0:
                self.logger.error(f"Transform failed: {transform_stderr_text}")
                # Remove empty/failed output file
                if os.path.exists(output_path):
                    os.remove(output_path)
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Stabilization failed: {transform_stderr_text}",
                    duration_sec=time.time() - start,
                )

            self._report_progress(1.0, "Stabilization complete", progress_callback)

            return StageResult(
                status=StageStatus.COMPLETED,
                output_path=output_path,
                metadata={
                    "smoothness": smooth,
                    "threshold": thresh,
                    "avg_shake": avg_value,
                    "scene_changes": scene_changes,
                    "num_scenes": len(scene_changes) + 1 if scene_changes else 1,
                    "pixel_format": pixel_format,
                    "apply_zoom": apply_zoom,
                    "zoom_coverage": self._zoom_coverage,
                    "static_zoom_pct": static_zoom_pct,
                },
                duration_sec=time.time() - start,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_sec=time.time() - start,
            )
        finally:
            if trf_path and os.path.exists(trf_path):
                try:
                    os.remove(trf_path)
                except OSError:
                    pass
