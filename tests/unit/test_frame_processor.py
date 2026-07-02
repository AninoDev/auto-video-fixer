"""Tests for AI frame processor."""

import pytest

from autovideofixer.ai.frame_processor import FrameProcessor


class TestFrameProcessor:
    """Test frame extraction and conversion."""

    def test_processor_creation(self):
        """Test creating a FrameProcessor."""
        proc = FrameProcessor()
        assert proc.batch_size == 1
        assert proc.keep_open is False

    def test_processor_batch_size(self):
        """Test batch size configuration."""
        proc = FrameProcessor(batch_size=4)
        assert proc.batch_size == 4

    def test_processor_negative_batch_size(self):
        """Test negative batch size is clamped."""
        proc = FrameProcessor(batch_size=-1)
        assert proc.batch_size == 1

    def test_bgr_to_rgb(self):
        """Test BGR to RGB conversion."""
        import numpy as np

        frame = np.array([[[0, 0, 255]]], dtype="uint8")  # BGR blue
        rgb = FrameProcessor.bgr_to_rgb(frame)
        assert rgb[0, 0, 2] == 0  # R channel
        assert rgb[0, 0, 0] == 255  # B channel (was R)

    def test_rgb_to_bgr(self):
        """Test RGB to BGR conversion."""
        import numpy as np

        frame = np.array([[[255, 0, 0]]], dtype="uint8")  # RGB red
        bgr = FrameProcessor.rgb_to_bgr(frame)
        assert bgr[0, 0, 0] == 0  # B channel
        assert bgr[0, 0, 2] == 255  # R channel (was B)

    def test_fourcc_mapping(self):
        """Test codec to fourcc mapping."""
        assert FrameProcessor._fourcc_from_codec("libx264") == "H264"
        assert FrameProcessor._fourcc_from_codec("libx265") == "H265"
        assert FrameProcessor._fourcc_from_codec("unknown") == "H264"

    def test_extract_frames_nonexistent(self):
        """Test extracting frames from nonexistent file."""
        proc = FrameProcessor()
        with pytest.raises(RuntimeError):
            proc.extract_frames("/nonexistent/video.mp4")

    def test_extract_frame_pairs_nonexistent(self):
        """Test extracting frame pairs from nonexistent file."""
        proc = FrameProcessor()
        with pytest.raises(RuntimeError):
            proc.extract_frame_pairs("/nonexistent/video.mp4")

    def test_frames_to_video_empty(self):
        """Test writing empty frames list."""
        proc = FrameProcessor()
        result = proc.frames_to_video([], "/tmp/test_empty.mp4")
        assert result is False

    def test_context_manager(self):
        """Test FrameProcessor context manager."""
        with FrameProcessor() as proc:
            assert proc is not None

    def test_close(self):
        """Test closing the processor."""
        proc = FrameProcessor()
        proc.close()  # Should not raise
        assert proc._cap is None


class TestFrameProcessorRealVideo:
    """Regression tests against a real decoded video.

    _stream_frames() contains `yield` statements, which makes the whole
    method a generator function regardless of which branch executes - a
    `return frames` on some other branch does NOT hand `frames` back to a
    `for` loop over the call, it just ends iteration early with nothing
    yielded. Extracting via mocks alone doesn't catch this: it only shows
    up against a real cv2.VideoCapture.
    """

    def test_extract_frames_returns_flat_ndarray_list(self, tmp_video_file):
        import numpy as np

        proc = FrameProcessor()
        frames = proc.extract_frames(tmp_video_file, max_frames=5)

        assert len(frames) == 5
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3

    def test_stream_frames_chunk_sizes(self, tmp_video_file):
        proc = FrameProcessor()
        chunks = list(proc.stream_frames(tmp_video_file, max_frames=7, chunk_size=3))

        assert [len(c) for c in chunks] == [3, 3, 1]
        for chunk in chunks:
            for frame in chunk:
                assert frame.ndim == 3

    def test_extract_frames_small_video_default_batch_size(self, tmp_video_file):
        """A video shorter than any chunk size must still extract as a flat list.

        This is the exact scenario that used to silently return frames
        wrapped in singleton lists (`[[frame], [frame], ...]`) because the
        internal chunking granularity was tied to `self.batch_size`, which
        defaults to 1.
        """
        proc = FrameProcessor()  # default batch_size=1
        frames = proc.extract_frames(tmp_video_file)

        assert len(frames) > 0
        assert not isinstance(frames[0], list)
        assert frames[0].ndim == 3
