"""Auto Video Fixer - Frame extraction and conversion utilities.

Handles extracting frames from video files, converting between
numpy arrays and PyTorch tensors, and managing frame buffers.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

_logger: Any = None


def _get_logger():
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.ai.frame_processor")
    return _logger


class FrameProcessor:
    """Extracts and converts video frames for AI processing.

    Handles frame extraction from video files, color space conversion,
    and tensor creation. Frames are returned as numpy arrays that can
    be directly converted to PyTorch tensors.
    """

    def __init__(self, batch_size: int = 1, keep_open: bool = False):
        self.batch_size = max(1, batch_size)
        self.keep_open = keep_open
        self._cap = None

    def extract_frames(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
    ) -> list[Any]:
        """Extract frames from a video file as numpy arrays.

        Args:
            video_path: Path to video file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds (None = to end).
            max_frames: Maximum number of frames to extract.

        Returns:
            List of numpy arrays (H, W, 3) in BGR, uint8.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate start frame
        start_frame = int(start_sec * fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if end_sec is not None:
            max_frames = min(max_frames or total_frames, int((end_sec - start_sec) * fps))

        frames: list[Any] = []
        frame_idx = start_frame

        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            frame_idx += 1

        cap.release()
        return frames

    def extract_frame_pairs(
        self,
        video_path: str,
        interval: int = 1,
    ) -> list[tuple[Any, Any]]:
        """Extract consecutive frame pairs for interpolation.

        Args:
            video_path: Path to video file.
            interval: Frame interval between pairs (1 = consecutive).

        Returns:
            List of (prev_frame, next_frame) tuples as numpy arrays.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        pairs: list[tuple[Any, Any]] = []
        prev_frame = None

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return pairs

        prev_frame = frame
        frame_count = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0 and prev_frame is not None:
                pairs.append((prev_frame, frame))

            prev_frame = frame
            frame_count += 1

        cap.release()
        return pairs

    def frames_to_video(
        self,
        frames: list[Any],
        output_path: str,
        fps: float = 30.0,
        codec: str = "libx264",
    ) -> bool:
        """Write frames to a video file using FFmpeg.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8.
            output_path: Path to output video file.
            fps: Frames per second.
            codec: FFmpeg video codec.

        Returns:
            True if successful.
        """
        import subprocess

        if not frames:
            return False

        h, w = frames[0].shape[:2]

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-pix_fmt", "bgr24",
            "-i", "-",
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    def frames_to_temp_video(
        self,
        frames: list[Any],
        fps: float = 30.0,
        suffix: str = "_ai",
    ) -> str:
        """Write frames to a temporary video file.

        Args:
            frames: List of numpy arrays.
            fps: Frames per second.
            suffix: Suffix for the temp filename.

        Returns:
            Path to the temporary video file.
        """
        ext = ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix + ext, prefix="avf_frame_")
        os.close(fd)

        if not self.frames_to_video(frames, path, fps):
            os.unlink(path)
            raise RuntimeError("Failed to write temp video from frames")
        return path

    @staticmethod
    def bgr_to_rgb(frame: Any) -> Any:
        """Convert BGR numpy array to RGB."""
        return frame[:, :, ::-1].copy()

    @staticmethod
    def rgb_to_bgr(frame: Any) -> Any:
        """Convert RGB numpy array to BGR."""
        return frame[:, :, ::-1].copy()

    @staticmethod
    def _fourcc_from_codec(codec: str) -> str:
        """Map FFmpeg codec name to OpenCV fourcc."""
        mapping = {
            "libx264": "H264",
            "libx265": "H265",
            "mpeg4": "XVID",
            "h264": "H264",
            "h265": "H265",
            "vp8": "VP80",
            "vp9": "VP90",
        }
        return mapping.get(codec, "H264")

    def close(self) -> None:
        """Release any open resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
