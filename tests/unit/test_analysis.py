"""Tests for video analysis utilities."""

import os

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import (
    VideoAnalyzer,
    is_image_file,
    is_video_file,
    scan_directory,
)


class TestFileDetection:
    """Test file type detection."""

    def test_is_video_file_valid(self, tmp_path):
        """Test detecting valid video files."""
        video = tmp_path / "test.mp4"
        video.write_text("fake video")

        assert is_video_file(str(video)) is True

    def test_is_video_file_invalid_extension(self, tmp_path):
        """Test rejecting non-video extensions."""
        file = tmp_path / "test.txt"
        file.write_text("not a video")

        assert is_video_file(str(file)) is False

    def test_is_video_file_nonexistent(self):
        """Test handling nonexistent files."""
        assert is_video_file("/nonexistent/file.mp4") is False

    def test_is_image_file_valid(self, tmp_path):
        """Test detecting valid image files."""
        image = tmp_path / "test.jpg"
        image.write_text("fake image")

        assert is_image_file(str(image)) is True

    def test_is_image_file_invalid(self, tmp_path):
        """Test rejecting non-image files."""
        file = tmp_path / "test.mp4"
        file.write_text("not an image")

        assert is_image_file(str(file)) is False

    def test_video_extensions(self):
        """Test all supported video extensions."""
        from autovideofixer.core.analysis import VIDEO_EXTENSIONS

        expected = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
        assert expected.issubset(VIDEO_EXTENSIONS)


class TestDirectoryScanning:
    """Test directory scanning for video files."""

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning an empty directory."""
        videos = scan_directory(str(tmp_path))
        assert len(videos) == 0

    def test_scan_directory_with_videos(self, tmp_path):
        """Test scanning directory with video files."""
        # Create video files
        (tmp_path / "video1.mp4").write_text("")
        (tmp_path / "video2.mkv").write_text("")
        (tmp_path / "video3.avi").write_text("")

        # Create non-video files
        (tmp_path / "readme.txt").write_text("")
        (tmp_path / "image.jpg").write_text("")

        videos = scan_directory(str(tmp_path))

        assert len(videos) == 3
        video_names = [os.path.basename(v) for v in videos]
        assert "video1.mp4" in video_names
        assert "video2.mkv" in video_names
        assert "video3.avi" in video_names

    def test_scan_directory_recursive(self, tmp_path):
        """Test recursive directory scanning."""
        # Create subdirectory with videos
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "sub_video.mp4").write_text("")

        # Create video in root
        (tmp_path / "root_video.mp4").write_text("")

        # Scan recursively
        videos = scan_directory(str(tmp_path), recursive=True)
        assert len(videos) == 2

        # Scan non-recursively
        videos = scan_directory(str(tmp_path), recursive=False)
        assert len(videos) == 1

    def test_scan_nonexistent_directory(self):
        """Test scanning a nonexistent directory."""
        videos = scan_directory("/nonexistent/directory")
        assert len(videos) == 0


class TestVideoAnalyzer:
    """Test video analysis functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.analyzer = VideoAnalyzer(self.config)

    def test_analyzer_creation(self):
        """Test creating a video analyzer."""
        assert self.analyzer is not None
        assert self.analyzer._analysis_cache == {}

    def test_analyze_nonexistent_file(self):
        """Test analyzing a nonexistent file."""
        with pytest.raises(Exception):
            self.analyzer.analyze("/nonexistent/video.mp4")

    def test_clear_cache(self):
        """Test clearing the analysis cache."""
        self.analyzer._analysis_cache["test"] = "data"
        assert len(self.analyzer._analysis_cache) == 1

        self.analyzer.clear_cache()
        assert len(self.analyzer._analysis_cache) == 0

    @pytest.mark.integration
    def test_analyze_real_video(self, tmp_path):
        """Test analyzing a real video file (integration test)."""
        import subprocess

        test_file = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(test_file),
            ],
            capture_output=True,
            check=True,
        )

        analysis = self.analyzer.analyze(str(test_file))

        assert analysis.filename == "test.mp4"
        assert analysis.duration > 0
        assert analysis.has_video is True
        assert analysis.has_audio is False  # testsrc has no audio
