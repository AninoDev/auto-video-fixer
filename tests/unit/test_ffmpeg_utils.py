"""Tests for FFmpeg utilities."""

import os
import subprocess
from pathlib import Path

import pytest

from autovideofixer.core.ffmpeg_utils import (
    get_ffmpeg_path,
    get_ffprobe_path,
    probe,
    ProbeResult,
    StreamInfo,
    detect_hardware_acceleration,
    resolve_hwaccel,
    generate_temp_path,
)


class TestFFmpegDetection:
    """Test FFmpeg binary detection."""

    def test_get_ffmpeg_path(self):
        """Test finding FFmpeg binary."""
        path = get_ffmpeg_path()
        assert path is not None
        assert os.path.isfile(path)

    def test_get_ffprobe_path(self):
        """Test finding FFprobe binary."""
        path = get_ffprobe_path()
        assert path is not None
        assert os.path.isfile(path)

    def test_ffmpeg_version(self):
        """Test FFmpeg version detection."""
        path = get_ffmpeg_path()
        result = subprocess.run([path, "-version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "ffmpeg version" in result.stdout.lower()


class TestProbe:
    """Test media file probing."""

    @pytest.mark.integration
    def test_probe_valid_video(self, tmp_path):
        """Test probing a valid video file (integration test)."""
        import subprocess
        
        test_file = tmp_path / "test.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-c:a", "aac", str(test_file)
        ], capture_output=True, check=True)
        
        result = probe(str(test_file))
        
        assert isinstance(result, ProbeResult)
        assert result.filename == "test.mp4"
        assert result.duration > 0
        assert result.has_video is True
        assert result.has_audio is True
        assert result.resolution == (320, 240)
        assert result.framerate > 0

    @pytest.mark.integration
    def test_probe_video_info(self, tmp_path):
        """Test extracting video information (integration test)."""
        import subprocess
        
        test_file = tmp_path / "test.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=640x480:rate=30",
            "-c:v", "libx264", "-c:a", "aac", str(test_file)
        ], capture_output=True, check=True)
        
        result = probe(str(test_file))
        
        assert result.resolution == (640, 480)
        assert abs(result.framerate - 30.0) < 1.0
        assert result.format_name == "mov,mp4,m4a,3gp,3g2,mj2"

    def test_probe_nonexistent_file(self):
        """Test probing a nonexistent file."""
        with pytest.raises(Exception):
            probe("/nonexistent/file.mp4")


class TestStreamInfo:
    """Test stream information."""

    def test_stream_info_properties(self):
        """Test StreamInfo properties."""
        stream = StreamInfo(
            index=0,
            codec_type="video",
            codec_name="h264",
            width=1920,
            height=1080,
            fps=30.0,
        )
        
        assert stream.is_video is True
        assert stream.is_audio is False
        assert stream.codec_name == "h264"

    def test_audio_stream_info(self):
        """Test audio stream properties."""
        stream = StreamInfo(
            index=1,
            codec_type="audio",
            codec_name="aac",
            channels=2,
            sample_rate=48000,
        )
        
        assert stream.is_audio is True
        assert stream.is_video is False


class TestHardwareAcceleration:
    """Test hardware acceleration detection."""

    def test_detect_hardware_acceleration(self):
        """Test detecting available hardware acceleration."""
        hwaccels = detect_hardware_acceleration()
        assert isinstance(hwaccels, list)
        # May be empty if no GPU available
        for accel in hwaccels:
            assert isinstance(accel, str)

    def test_resolve_hwaccel_auto(self):
        """Test auto resolution of hardware acceleration."""
        result = resolve_hwaccel("auto")
        assert isinstance(result, str)
        assert result in ["auto", "cuda", "vaapi", "qsv", "none", ...]

    def test_resolve_hwaccel_none(self):
        """Test disabling hardware acceleration."""
        result = resolve_hwaccel("none")
        assert result == "none"


class TestPathGeneration:
    """Test temporary path generation."""

    def test_generate_temp_path(self, tmp_path):
        """Test generating temporary file paths."""
        input_path = str(tmp_path / "input.mp4")
        
        temp_path = generate_temp_path(str(tmp_path), input_path, suffix="_test")
        
        assert temp_path.endswith("_test.mp4")
        assert str(tmp_path) in temp_path
        assert ".avf_" in temp_path

    def test_generate_temp_path_uniqueness(self, tmp_path):
        """Test that generated paths are unique."""
        input_path = str(tmp_path / "input.mp4")
        
        paths = set()
        for _ in range(10):
            temp_path = generate_temp_path(str(tmp_path), input_path)
            paths.add(temp_path)
        
        assert len(paths) == 10  # All paths should be unique
