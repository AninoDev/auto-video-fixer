"""Auto Video Fixer - Frame extraction and conversion utilities.

Handles extracting frames from video files, converting between
numpy arrays and PyTorch tensors, and managing frame buffers.
Supports both full-memory extraction (for small clips) and
chunked/streaming extraction (for long videos).
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
from typing import Any, Generator, Iterable, Iterator

import cv2

_logger: Any = None


def _get_logger():
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.ai.frame_processor")
    return _logger


class PrefetchIterator:
    """Runs a source iterable on a background thread with a bounded lookahead queue.

    In the chunked AI-processing loop (see core/stages/{upscale,deblock,
    denoise_video}.py), decoding the next chunk of frames (OpenCV
    ``VideoCapture.read()``) was happening synchronously on the main thread
    in between GPU inference calls -- the GPU sat idle during every decode,
    and the CPU sat idle during every inference. ``cv2.VideoCapture.read()``
    releases the GIL while it decodes (it's a C/C++ call into libavcodec),
    so running it on a separate thread lets that decode genuinely overlap
    with GPU inference happening on the main thread, rather than just being
    interleaved by the GIL.

    A bounded queue (``maxsize``) caps how many chunks of frames can be
    decoded ahead of the consumer, so a slow consumer doesn't let the
    prefetch thread buffer the entire video in memory.
    """

    _SENTINEL = object()

    def __init__(self, source: Iterable[Any], maxsize: int = 2):
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._exception: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(source,), daemon=True, name="avf-frame-prefetch"
        )
        self._thread.start()

    def _run(self, source: Iterable[Any]) -> None:
        try:
            for item in source:
                self._queue.put(item)
        except BaseException as exc:  # noqa: BLE001 - propagated to consumer thread
            self._exception = exc
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self) -> "PrefetchIterator":
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is self._SENTINEL:
            self._thread.join()
            if self._exception is not None:
                raise self._exception
            raise StopIteration
        return item


class AsyncVideoWriter:
    """Wraps a ``StreamingVideoWriter`` so ``write()`` never blocks the caller.

    The chunked AI-processing loop previously called
    ``StreamingVideoWriter.write(chunk)`` synchronously right after each
    chunk finished inference, blocking the main thread (and therefore
    delaying the *next* chunk's GPU inference) on the ffmpeg pipe write.
    Handing the chunk to a background writer thread lets the next chunk's
    inference start immediately instead of waiting for that write to land.
    """

    def __init__(self, writer: "StreamingVideoWriter"):
        self._writer = writer
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=4)
        self._exception: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="avf-frame-writer")
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            try:
                self._writer.write(chunk)
            except BaseException as exc:  # noqa: BLE001 - re-raised on close()
                self._exception = exc
                return

    def write(self, frames: list[Any]) -> None:
        if not frames:
            return
        self._queue.put(frames)

    def close(self) -> bool:
        """Wait for all queued writes to flush, then finalize the file."""
        self._queue.put(None)
        self._thread.join()
        if self._exception is not None:
            raise self._exception
        return self._writer.close()


class FrameProcessor:
    """Extracts and converts video frames for AI processing.

    Handles frame extraction from video files, color space conversion,
    and tensor creation. Frames are returned as numpy arrays that can
    be directly converted to PyTorch tensors. Supports both full-memory
    extraction (for small clips) and chunked/streaming extraction.
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
        """Extract frames from a video file as numpy arrays (full memory).

        Args:
            video_path: Path to video file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds (None = to end).
            max_frames: Maximum number of frames to extract.

        Returns:
            List of numpy arrays (H, W, 3) in BGR, uint8.
        """
        frames: list[Any] = []
        for chunk in self._stream_frames(video_path, start_sec, end_sec, max_frames):
            frames.extend(chunk)
        return frames

    def stream_frames(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
        chunk_size: int | None = None,
    ) -> Iterator[list[Any]]:
        """Yield frame chunks as a generator, avoiding loading all frames into memory.

        Args:
            video_path: Path to video file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds (None = to end).
            max_frames: Maximum number of frames to extract (for the whole video).
            chunk_size: Frames per chunk (default: 25).

        Yields:
            Each call yields a list of numpy arrays (chunk of frames).
        """
        yield from self._stream_frames(
            video_path, start_sec, end_sec, max_frames, chunk_size=chunk_size
        )

    def _stream_frames(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
        chunk_size: int | None = None,
    ) -> Generator[list[Any], None, None]:
        """Internal frame-reading generator. Always yields chunks of frames.

        Note: a function is a generator if it contains ANY `yield` in its body,
        regardless of which branch executes - a `return value` on some other
        branch would NOT hand `value` to the caller, it would just end
        iteration early. So this always yields lists; callers that want a
        flat list (extract_frames) or a fixed chunk size (stream_frames) do
        their own reshaping on top of this.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame = int(start_sec * fps)
            start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_sec is not None:
                max_frames = min(max_frames or total_frames, int((end_sec - start_sec) * fps))

            # Internal read-batch granularity. Not tied to self.batch_size
            # (that's a separate, caller-facing inference-batching knob) -
            # reusing it here previously caused extract_frames() to silently
            # return frames wrapped in singleton lists whenever batch_size
            # defaulted to 1.
            effective_chunk = chunk_size if chunk_size and chunk_size > 0 else 500

            chunk: list[Any] = []
            total_yielded = 0
            while True:
                if max_frames is not None and total_yielded >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                chunk.append(frame)
                total_yielded += 1
                if len(chunk) >= effective_chunk:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
        finally:
            cap.release()

    def stream_frames_prefetched(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
        chunk_size: int | None = None,
        lookahead: int = 2,
    ) -> Iterator[list[Any]]:
        """Like :meth:`stream_frames`, but decodes on a background thread.

        Decoding the next chunk overlaps with whatever the caller does with
        the current chunk (typically GPU inference) instead of happening
        serially in between each call. See ``PrefetchIterator`` for why this
        achieves real overlap despite the GIL.
        """
        source = self._stream_frames(
            video_path, start_sec, end_sec, max_frames, chunk_size=chunk_size
        )
        yield from PrefetchIterator(source, maxsize=max(1, lookahead))

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
        pairs: list[tuple[Any, Any]] = []
        prev_frame = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

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

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-pix_fmt",
            "bgr24",
            "-i",
            "-",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        proc = None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            return proc.returncode == 0
        except Exception:
            return False
        finally:
            # If the write loop raised (e.g. broken pipe because ffmpeg
            # already exited), the subprocess would otherwise never be
            # reaped and would leak as a zombie.
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass

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


class StreamingVideoWriter:
    """Incrementally writes frame chunks to a video file via an ffmpeg pipe.

    Unlike ``FrameProcessor.frames_to_video()``, which requires every frame
    to be buffered in memory before a single write call, this lets a caller
    push chunks as they're produced (e.g. from a chunked AI-processing loop)
    without ever holding the whole video's frames in memory at once.

    Usage:
        writer = StreamingVideoWriter(output_path, fps=30.0)
        for chunk in produce_chunks():
            writer.write(chunk)
        ok = writer.close()
    """

    def __init__(self, output_path: str, fps: float = 30.0, codec: str = "libx264"):
        self._output_path = output_path
        self._fps = fps
        self._codec = codec
        self._proc: Any = None
        self._failed = False

    def write(self, frames: list[Any]) -> None:
        """Write a chunk of frames (numpy arrays, BGR uint8) to the stream."""
        import subprocess

        if not frames or self._failed:
            return

        if self._proc is None:
            h, w = frames[0].shape[:2]
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{w}x{h}",
                "-r",
                str(self._fps),
                "-pix_fmt",
                "bgr24",
                "-i",
                "-",
                "-c:v",
                self._codec,
                "-pix_fmt",
                "yuv420p",
                self._output_path,
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        try:
            for frame in frames:
                self._proc.stdin.write(frame.tobytes())
        except Exception:
            self._failed = True
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait()
                except Exception:
                    pass
            raise

    def close(self) -> bool:
        """Finalize the video file. Returns True on success.

        Safe to call even if no frames were ever written (returns False).
        """
        if self._proc is None or self._failed:
            return False
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait()
            return self._proc.returncode == 0
        except Exception:
            return False
        finally:
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
        else:
            self.close()
