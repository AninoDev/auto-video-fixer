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
