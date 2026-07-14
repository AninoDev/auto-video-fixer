"""Tests for AI frame processor."""

import pytest

from autovideofixer.ai.frame_processor import (
    AsyncVideoWriter,
    FrameProcessor,
    PrefetchIterator,
    StageTimer,
    gpu_forward_timer,
)


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

    def test_stream_frames_prefetched_matches_stream_frames(self, tmp_video_file):
        """Prefetched decoding must yield identical chunks to the synchronous path."""
        proc = FrameProcessor()
        sync_chunks = list(proc.stream_frames(tmp_video_file, max_frames=7, chunk_size=3))

        proc2 = FrameProcessor()
        prefetched_chunks = list(
            proc2.stream_frames_prefetched(tmp_video_file, max_frames=7, chunk_size=3)
        )

        assert [len(c) for c in prefetched_chunks] == [len(c) for c in sync_chunks]
        for sync_chunk, prefetch_chunk in zip(sync_chunks, prefetched_chunks):
            for sync_frame, prefetch_frame in zip(sync_chunk, prefetch_chunk):
                assert (sync_frame == prefetch_frame).all()


class TestPrefetchIterator:
    """PrefetchIterator decodes/produces items on a background thread."""

    def test_yields_items_in_order(self):
        items = list(PrefetchIterator(iter([1, 2, 3, 4, 5]), maxsize=2))
        assert items == [1, 2, 3, 4, 5]

    def test_empty_source(self):
        assert list(PrefetchIterator(iter([]), maxsize=2)) == []

    def test_propagates_exception_from_source(self):
        def bad_source():
            yield 1
            yield 2
            raise ValueError("boom")

        it = PrefetchIterator(bad_source(), maxsize=2)
        collected = []
        with pytest.raises(ValueError, match="boom"):
            for item in it:
                collected.append(item)
        assert collected == [1, 2]


class TestAsyncVideoWriter:
    """AsyncVideoWriter defers writes to a background thread without dropping data."""

    def test_forwards_all_chunks_to_underlying_writer(self):
        from unittest.mock import MagicMock

        underlying = MagicMock()
        underlying.close.return_value = True

        writer = AsyncVideoWriter(underlying)
        writer.write([1, 2])
        writer.write([3])
        writer.write([])  # no-op, must not be forwarded
        ok = writer.close()

        assert ok is True
        assert underlying.write.call_count == 2
        underlying.write.assert_any_call([1, 2])
        underlying.write.assert_any_call([3])

    def test_close_reraises_write_exception(self):
        from unittest.mock import MagicMock

        underlying = MagicMock()
        underlying.write.side_effect = RuntimeError("pipe broke")

        writer = AsyncVideoWriter(underlying)
        writer.write([1])
        with pytest.raises(RuntimeError, match="pipe broke"):
            writer.close()


class TestStageTimer:
    """Tests for the AI-stage per-phase timing + periodic throughput helper."""

    def test_phase_records_wall_time(self):
        import time as _time

        timer = StageTimer("test_stage", interval_sec=9999, interval_chunks=9999)
        with timer.phase("decode_wait"):
            _time.sleep(0.01)
        timer.end_chunk(5)
        assert timer._totals["decode_wait"] >= 0.01
        assert timer._frames_done == 5
        assert timer._chunks_done == 1

    def test_record_accumulates_across_chunks(self):
        timer = StageTimer("test_stage", interval_sec=9999, interval_chunks=9999)
        timer.record("gpu_forward", 0.1)
        timer.end_chunk(2)
        timer.record("gpu_forward", 0.2)
        timer.end_chunk(3)
        assert timer._totals["gpu_forward"] == pytest.approx(0.3)
        assert timer._frames_done == 5

    def test_end_chunk_logs_debug_and_throughput_at_interval(self, caplog):
        logger = __import__("logging").getLogger("test.stage_timer")
        caplog.set_level("DEBUG", logger="test.stage_timer")
        # interval_chunks=1 forces an INFO throughput line on every chunk.
        timer = StageTimer("test_stage", logger=logger, interval_sec=9999, interval_chunks=1)
        timer.record("decode_wait", 0.01)
        timer.end_chunk(10)

        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert any("chunk #1" in r.getMessage() for r in debug_records)
        assert any("throughput" in r.getMessage() for r in info_records)
        assert any("frames_done=10" in r.getMessage() for r in info_records)

    def test_summary_logs_percentage_breakdown(self, caplog):
        logger = __import__("logging").getLogger("test.stage_timer.summary")
        caplog.set_level("INFO", logger="test.stage_timer.summary")
        timer = StageTimer("test_stage", logger=logger)
        timer.record("decode_wait", 1.0)
        timer.record("gpu_forward", 3.0)
        timer.end_chunk(4)
        timer.summary()

        summary_records = [
            r for r in caplog.records if r.levelname == "INFO" and "finished" in r.getMessage()
        ]
        assert len(summary_records) == 1
        msg = summary_records[0].getMessage()
        assert "frames=4" in msg
        assert "gpu_forward=3.0s (75%)" in msg
        assert "decode_wait=1.0s (25%)" in msg

    def test_summary_with_no_phase_data_does_not_crash(self):
        timer = StageTimer("test_stage")
        timer.summary()  # must not raise (division-by-zero guarded)


class TestGpuForwardTimer:
    """Tests for gpu_forward_timer's CPU/no-device wall-time fallback path.

    The CUDA-Event branch is exercised indirectly by
    TestPinnedStagingPoolCudaCorrectness / the real-model integration tests
    elsewhere -- this only covers the always-available fallback.
    """

    def test_none_device_uses_wall_time(self):
        import time as _time

        with gpu_forward_timer(None) as result:
            _time.sleep(0.01)
        assert result.elapsed_sec >= 0.01

    def test_cpu_device_uses_wall_time(self):
        import time as _time

        class _FakeDevice:
            type = "cpu"

        with gpu_forward_timer(_FakeDevice()) as result:
            _time.sleep(0.01)
        assert result.elapsed_sec >= 0.01
