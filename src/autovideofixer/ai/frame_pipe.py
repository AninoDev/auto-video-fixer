"""Auto Video Fixer - frame I/O transport adapter.

Wraps the threaded Rust `avf_framepipe` extension (see
`rust/avf_framepipe/src/lib.rs`) behind two factory functions,
`get_frame_reader()` / `get_frame_writer()`, that hand back an object with a
stable `next_batch()`/`frames_read()`/`close()` (reader) or
`write_batch()`/`frames_written()`/`close()` (writer) surface -- regardless
of whether the Rust extension is actually available.

Same lazy-import-with-fallback contract as `avf_scenes`/`avf_hashing` (see
`core/analysis.py`): `try: import avf_framepipe` / `except ImportError`,
falling back to a thin Python wrapper around the existing
`ai/frame_processor.py` machinery (`FrameProcessor.stream_frames_prefetched`
for reading, `AsyncVideoWriter`/`StreamingVideoWriter` for writing) so call
sites never branch on backend -- they just call `get_frame_reader()`/
`get_frame_writer()` and use the returned object uniformly.

`frame_processor.py` itself is untouched by this module; it remains the
fallback implementation and is still usable directly by anything that
doesn't need transport-backend selection.

Note on fallback parity gaps (Python backend only, not user-visible via the
adapter's API surface):
- `write_queue` (backpressure depth on the writer's internal queue) is a
  real bounded-channel size in the Rust backend but a fixed constant
  (`AsyncVideoWriter`'s hardcoded `maxsize=4`) in the Python fallback --
  frame_processor.py is intentionally left unmodified, so this knob is
  accepted for signature parity but only takes effect on the Rust backend.
- `frames_written()` on the Python fallback counts frames as they're
  *enqueued* to the background writer thread, not as they're actually
  flushed to the ffmpeg pipe (the Rust backend counts the latter). Both
  converge to the same final count once `close()` returns.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path
from autovideofixer.logger import get_logger

logger = get_logger("autovideofixer.ai.frame_pipe")

# Lazily-imported PyO3/maturin Rust extension implementing threaded,
# bounded-channel ffmpeg frame I/O (docs/REQUIREMENTS.md R5.3). Not
# available means the compiled wheel wasn't built for this
# environment/platform -- fall back to the pure-Python transport below
# rather than hard-failing. See rust/avf_framepipe/src/lib.rs for the Rust
# side.
try:
    import avf_framepipe as _avf_framepipe_native

    RUST_AVAILABLE = True
    logger.debug("avf_framepipe Rust extension available; using Rust frame transport backend.")
except ImportError:
    _avf_framepipe_native = None
    RUST_AVAILABLE = False
    logger.debug(
        "avf_framepipe Rust extension not available; falling back to pure-Python "
        "frame transport (see rust/avf_framepipe/ and AGENTS.md's Setup & Commands "
        "for the build step)."
    )


class FrameReaderProtocol(Protocol):
    """Common surface both the Rust and Python-fallback readers expose."""

    def next_batch(self) -> list[Any] | None: ...

    def frames_read(self) -> int: ...

    def close(self) -> None: ...


class FrameWriterProtocol(Protocol):
    """Common surface both the Rust and Python-fallback writers expose."""

    def write_batch(self, frames: list[Any]) -> None: ...

    def frames_written(self) -> int: ...

    def close(self) -> bool: ...


class _PythonFrameReader:
    """Fallback reader: wraps `FrameProcessor.stream_frames_prefetched()`.

    Exposes the same `next_batch()`/`frames_read()`/`close()` surface as the
    Rust `avf_framepipe.FrameReader` so call sites never need to branch on
    backend.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        chunk_size: int = 25,
        read_ahead: int = 2,
        ffmpeg_path: str | None = None,
    ) -> None:
        from autovideofixer.ai.frame_processor import FrameProcessor

        # width/height aren't needed by the OpenCV-based decode path (each
        # frame already carries its own shape) -- kept as constructor
        # arguments purely for signature parity with the Rust backend.
        self._width = width
        self._height = height
        # ffmpeg_path is unused here: FrameProcessor decodes via OpenCV
        # (cv2.VideoCapture), not by shelling out to ffmpeg itself.
        self._ffmpeg_path = ffmpeg_path
        self._proc = FrameProcessor()
        self._iter = iter(
            self._proc.stream_frames_prefetched(path, chunk_size=chunk_size, lookahead=read_ahead)
        )
        self._frames_read = 0
        self._closed = False

    def next_batch(self) -> list[Any] | None:
        if self._closed:
            return None
        try:
            batch = next(self._iter)
        except StopIteration:
            return None
        self._frames_read += len(batch)
        return batch

    def frames_read(self) -> int:
        return self._frames_read

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._proc.close()


class _PythonFrameWriter:
    """Fallback writer: wraps `AsyncVideoWriter(StreamingVideoWriter(...))`.

    Exposes the same `write_batch()`/`frames_written()`/`close()` surface as
    the Rust `avf_framepipe.FrameWriter` so call sites never need to branch
    on backend.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "libx264",
        crf: int | None = None,
        preset: str | None = None,
        write_queue: int = 4,
        ffmpeg_path: str | None = None,
    ) -> None:
        from autovideofixer.ai.frame_processor import AsyncVideoWriter, StreamingVideoWriter

        self._width = width
        self._height = height
        # write_queue/ffmpeg_path aren't threaded through -- see module
        # docstring's "fallback parity gaps" note. AsyncVideoWriter's queue
        # depth is a hardcoded constant in frame_processor.py (left
        # unmodified by design), and StreamingVideoWriter always shells out
        # to the "ffmpeg" found on PATH.
        self._write_queue = write_queue
        self._ffmpeg_path = ffmpeg_path
        self._writer = AsyncVideoWriter(
            StreamingVideoWriter(path, fps=fps, codec=codec, crf=crf, preset=preset)
        )
        self._frames_written = 0
        self._closed = False
        self._last_result = False

    def write_batch(self, frames: list[Any]) -> None:
        if self._closed:
            raise RuntimeError("frame_pipe: write_batch() called after close()")
        if not frames:
            return
        self._writer.write(list(frames))
        self._frames_written += len(frames)

    def frames_written(self) -> int:
        return self._frames_written

    def close(self) -> bool:
        if self._closed:
            # Match the Rust backend's idempotent close() contract: repeat
            # calls return the same final result rather than raising.
            return self._last_result
        self._closed = True
        self._last_result = self._writer.close()
        return self._last_result


def get_frame_reader(
    path: str,
    width: int,
    height: int,
    *,
    chunk_size: int = 25,
    read_ahead: int = 2,
    ffmpeg_path: str | None = None,
) -> FrameReaderProtocol:
    """Return a frame reader for `path`, using the Rust backend when available.

    `width`/`height` are the expected decoded frame dimensions (required by
    the Rust backend to reshape raw bytes into arrays; accepted but unused
    by the Python fallback, which gets shape from OpenCV directly).
    """
    resolved_ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
    if RUST_AVAILABLE:
        logger.debug("frame_pipe: using Rust FrameReader backend for %s", path)
        return cast(
            FrameReaderProtocol,
            _avf_framepipe_native.FrameReader(
                path,
                width,
                height,
                chunk_size=chunk_size,
                read_ahead=read_ahead,
                ffmpeg_path=resolved_ffmpeg_path,
            ),
        )
    logger.debug("frame_pipe: using Python FrameReader fallback for %s", path)
    return _PythonFrameReader(
        path,
        width,
        height,
        chunk_size=chunk_size,
        read_ahead=read_ahead,
        ffmpeg_path=resolved_ffmpeg_path,
    )


def get_frame_writer(
    path: str,
    width: int,
    height: int,
    fps: float,
    *,
    codec: str = "libx264",
    crf: int | None = None,
    preset: str | None = None,
    write_queue: int = 4,
    ffmpeg_path: str | None = None,
) -> FrameWriterProtocol:
    """Return a frame writer for `path`, using the Rust backend when available."""
    resolved_ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
    if RUST_AVAILABLE:
        logger.debug("frame_pipe: using Rust FrameWriter backend for %s", path)
        return cast(
            FrameWriterProtocol,
            _avf_framepipe_native.FrameWriter(
                path,
                width,
                height,
                fps,
                codec=codec,
                crf=crf,
                preset=preset,
                write_queue=write_queue,
                ffmpeg_path=resolved_ffmpeg_path,
            ),
        )
    logger.debug("frame_pipe: using Python FrameWriter fallback for %s", path)
    return _PythonFrameWriter(
        path,
        width,
        height,
        fps,
        codec=codec,
        crf=crf,
        preset=preset,
        write_queue=write_queue,
        ffmpeg_path=resolved_ffmpeg_path,
    )
