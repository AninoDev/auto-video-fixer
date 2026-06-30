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

    def __init__(self, config):
        super().__init__(config)
        self._threshold = self._stage_config.get("threshold", 2.0)
        self._smoothness = self._stage_config.get("smoothness", 40)
        self._maxshift = self._stage_config.get("maxshift", 20)
        self._scene_threshold = self._stage_config.get("scene_threshold", 0.98)
        self._zoom_enabled = self._stage_config.get("zoom_enabled", True)
        self._zoom_mode = self._stage_config.get("zoom_mode", "black")
        self._zoom_threshold = self._stage_config.get("zoom_threshold", 50.0)  # pixels
        self._sharpen_enabled = self._stage_config.get("sharpen_enabled", True)
        self._optalgo = self._stage_config.get("optalgo", "gauss")
        self._shakiness = self._stage_config.get("shakiness", 10)
        
        self.logger.debug(f"StabilizeStage initialized: smoothness={self._smoothness}, maxshift={self._maxshift}, shakiness={self._shakiness}, threshold={self._threshold}")

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
            max_mag = max(magnitudes)
            
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
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                input_path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, text=True)

            if result.stdout and "," in result.stdout:
                parts = result.stdout.strip().split(",")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 1920, 1080

    def _get_video_framerate(self, input_path: str) -> float:
        """Get video framerate from ffprobe."""
        try:
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                input_path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, text=True)

            if result.stdout and "/" in result.stdout:
                num, den = result.stdout.strip().split("/")
                num, den = float(num), float(den)
                return num / den if den else 30.0
            return 30.0
        except Exception:
            return 30.0

    def _clean_trf_outliers(self, trf_path: str) -> str:
        """Remove outlier frames from TRF file to prevent artifact bursts.
        
        Returns path to cleaned TRF file.
        """
        import tempfile
        
        try:
            with open(trf_path, 'r') as f:
                content = f.read()
            
            # Parse TRF file into structured data
            # Format: Frame N (List M [(LM dx dy x y w h contrast magnitude),...])
            frames_data = {}
            current_frame = None
            
            for line in content.split('\n'):
                if line.startswith('Frame '):
                    match = re.match(r"Frame (\d+)", line)
                    if match:
                        current_frame = int(match.group(1))
                        frames_data[current_frame] = []
                elif line.startswith('(') and current_frame is not None:
                    # Parse LM entries
                    lm_matches = re.findall(r'\(LM\s+(-?\d+)\s+(-?\d+)\s+', line)
                    for dx, dy in lm_matches:
                        frames_data[current_frame].append((int(dx), int(dy)))
            
            if not frames_data:
                return trf_path
            
            # Calculate average movement magnitude for each frame
            frame_magnitudes = {}
            for frame, lms in frames_data.items():
                if lms:
                    total_mag = sum((dx**2 + dy**2)**0.5 for dx, dy in lms)
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
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                if line.startswith('Frame '):
                    match = re.match(r"Frame (\d+)", line)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num in outlier_frames:
                            # Replace this frame's LM values with zeros
                            new_line = re.sub(
                                r'\(LM\s+-?\d+\s+-?\d+\s+',
                                '(LM 0 0 ',
                                line
                            )
                            cleaned_lines.append(new_line)
                        else:
                            cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
            
            # Write cleaned TRF to temp file
            cleaned_fd, cleaned_path = tempfile.mkstemp(suffix=".trf", prefix="avf_clean_")
            os.close(cleaned_fd)
            
            with open(cleaned_path, 'w') as f:
                f.write('\n'.join(cleaned_lines))
            
            return cleaned_path
            
        except Exception as e:
            self.logger.warning(f"TRF cleaning failed: {e}")
            return trf_path

    def _calculate_zoom(self, trf_path: str, video_width: int, video_height: int) -> float:
        """Calculate required zoom percentage based on movement extent in TRF file.

        Returns:
            Zoom percentage (negative = zoom out, positive = zoom in).
            Returns 0 if zoom is not needed.
        """
        try:
            with open(trf_path, "r") as f:
                content = f.read()

            # Extract all LM entries: (LM dx dy x y w h contrast magnitude)
            lm_pattern = r"\(LM\s+(-?\d+)\s+(-?\d+)\s+"
            matches = re.findall(lm_pattern, content)

            if not matches:
                return 0.0

            # Calculate movement ranges
            dx_values = [float(m[0]) for m in matches]
            dy_values = [float(m[1]) for m in matches]

            max_dx = max(dx_values) - min(dx_values)
            max_dy = max(dy_values) - min(dy_values)
            max_movement = max(max_dx, max_dy)

            # Only zoom if movement exceeds threshold
            if max_movement < self._zoom_threshold:
                return 0.0

            # Calculate zoom percentage based on movement relative to frame size
            # Aim to keep at least 80% of frame visible
            zoom_percent = -((max_movement / min(video_width, video_height)) * 100 * 0.75)
            
            # Clamp zoom to reasonable range (-20% to 0%)
            zoom_percent = max(-20.0, min(0.0, zoom_percent))

            return zoom_percent

        except Exception as e:
            self.logger.warning(f"Zoom calculation failed: {e}")
            return 0.0

    def _detect_scenes(self, input_path: str, scene_threshold: float = 0.98) -> list[float]:
        """Detect scene changes by comparing consecutive frames.

        Returns:
            List of timestamps (in seconds) where scene changes occur.
        """
        import tempfile
        from PIL import Image
        import numpy as np

        scene_changes = []
        temp_dir = tempfile.mkdtemp(prefix="avf_scene_")

        try:
            # Extract frames at 1 fps to reduce processing
            frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            extract_args = [
                "-i", input_path,
                "-vf", "fps=1",
                "-q:v", "2",
                frame_pattern
            ]
            
            run_ffmpeg(extract_args, timeout=120)

            # Load and compare consecutive frames
            import glob
            frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

            if len(frame_files) < 2:
                return []

            prev_frame = None
            prev_time = 0.0

            # Get frame timestamps from ffprobe
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "frame=pts_time",
                "-of", "csv=p=0",
                input_path
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, text=True)

            if result.stdout:
                times = [float(t) for t in result.stdout.strip().split('\n')]
            else:
                times = list(range(len(frame_files)))

            for i, frame_file in enumerate(frame_files):
                if i == 0:
                    prev_frame = np.array(Image.open(frame_file).convert('L'))
                    prev_time = times[i] if i < len(times) else 0.0
                    continue

                curr_frame = np.array(Image.open(frame_file).convert('L'))

                # Calculate structural similarity or correlation
                if prev_frame.shape != curr_frame.shape:
                    prev_frame = curr_frame
                    prev_time = times[i] if i < len(times) else 0.0
                    continue

                # Normalize frames
                prev_norm = prev_frame.astype(float) / 255.0
                curr_norm = curr_frame.astype(float) / 255.0

                # Calculate correlation
                correlation = np.corrcoef(prev_norm.flatten(), curr_norm.flatten())[0, 1]

                # If correlation is low, it's a scene change
                if correlation < scene_threshold and i > 0:
                    scene_changes.append(times[i])

                prev_frame = curr_frame
                prev_time = times[i] if i < len(times) else 0.0

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
            self.logger.debug(f"Shake analysis: needs_stab={needs_stab}, avg_value={avg_value:.3f}, threshold={thresh}")

            # Step 2.5: Detect scene changes
            self._report_progress(0.2, "Detecting scenes...", progress_callback)
            scene_changes = self._detect_scenes(input_path, self._scene_threshold)

            # Step 2.6: Calculate zoom if needed
            self._report_progress(0.25, "Analyzing movement extent...", progress_callback)
            video_width, video_height = self._get_video_dimensions(input_path)
            zoom_value = 0.0
            if self._zoom_enabled and needs_stab:
                zoom_value = self._calculate_zoom(trf_path, video_width, video_height)
                self.logger.debug(f"Zoom calculated: {zoom_value:.2f}%")
            else:
                self.logger.debug(f"Zoom skipped: enabled={self._zoom_enabled}, needs_stab={needs_stab}")

            if not needs_stab:
                self.logger.info(f"Skipping stabilization (avg shake: {avg_value:.3f} < threshold: {thresh})")
                self._report_progress(1.0, f"No stabilization needed (avg: {avg_value:.3f})", progress_callback)
                # Skip stabilization, just copy the input
                run_ffmpeg(
                    [
                        "-hide_banner",
                        "-i", input_path,
                        "-c", "copy",
                        "-y", output_path,
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
            zoom_param = f":zoom={zoom_value}:optzoom=2" if zoom_value != 0.0 else ""

            # Build filter chain with optional sharpening
            stab_filter = f"vidstabtransform=smoothing={smooth}:input={clean_trf_path}:crop={crop_mode}:interpol=bilinear:maxshift={self._maxshift}:optalgo={self._optalgo}{zoom_param}"

            # Add sharpening if stabilization was auto-triggered (not user-disabled)
            auto_triggered = needs_stab and self._sharpen_enabled
            if auto_triggered:
                stab_filter = f"{stab_filter},unsharp=3:3:0.5:3:3:0.0"

            self._report_progress(0.6, "Stabilizing (pipe decode→transform)...", progress_callback)
            self.logger.info(f"Piping raw video: {video_width}x{video_height} @ {framerate}fps, format={pixel_format}")
            self.logger.debug(f"vidstabtransform filter: {stab_filter}")
            self.logger.debug(f"Config: smoothness={self._smoothness}, maxshift={self._maxshift}, zoom_enabled={self._zoom_enabled}, zoom_mode={self._zoom_mode}, zoom_value={zoom_value}")

            # Use subprocess to pipe decode stdout → transform stdin
            import subprocess as sp
            from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path

            ffmpeg_bin = get_ffmpeg_path()
            
            # Decode process: outputs raw video (no audio) to stdout
            decode_proc = sp.Popen(
                [ffmpeg_bin, "-hide_banner", "-i", input_path,
                 "-vf", f"format={pixel_format}", "-c:v", "rawvideo",
                 "-f", "rawvideo", "-bufsize", "10M", "-"],
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )

            # Transform process: reads raw video from stdin, audio from original file
            transform_proc = sp.Popen(
                [ffmpeg_bin, "-hide_banner",
                 "-f", "rawvideo",
                 "-pix_fmt", pixel_format, "-s", f"{video_width}x{video_height}",
                 "-r", str(framerate), "-i", "-",
                 "-i", input_path,
                 "-map", "0:v", "-map", "1:a:0",
                 "-vf", stab_filter,
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-c:a", "copy",
                 "-y", output_path],
                stdin=decode_proc.stdout,
                stderr=sp.PIPE,
            )

            # Close decode's stdout in parent - only transform reads it
            decode_proc.stdout.close()

            # Wait for both to complete
            _, decode_stderr = decode_proc.communicate()
            transform_stderr = transform_proc.communicate()[1]

            if decode_proc.returncode != 0:
                self.logger.error(f"Decode failed: {decode_stderr[:500].decode() if isinstance(decode_stderr, bytes) else decode_stderr}")
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Raw decode failed: {decode_stderr[:500]}",
                    duration_sec=time.time() - start,
                )

            if transform_proc.returncode != 0:
                self.logger.error(f"Transform failed: {transform_stderr[:500].decode() if isinstance(transform_stderr, bytes) else transform_stderr}")
                # Remove empty/failed output file
                if os.path.exists(output_path):
                    os.remove(output_path)
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"Stabilization failed: {transform_stderr[:500]}",
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
