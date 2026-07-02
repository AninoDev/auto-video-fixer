"""Tests for scene detection, clip extraction, and duplicate detection."""

import os
import tempfile
from pathlib import Path

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import (
    SceneEvent,
    VideoAnalyzer,
    compute_video_dhash,
    compute_video_hash,
    hash_similarity,
    scan_directory,
)


class TestSceneEvent:
    """Test SceneEvent dataclass."""

    def test_scene_event_creation(self):
        """Test creating a SceneEvent."""
        event = SceneEvent(start_time=0.0, end_time=5.0, event_type="talking_head")
        assert event.start_time == 0.0
        assert event.end_time == 5.0
        assert event.duration == 5.0
        assert event.event_type == "talking_head"
        assert event.confidence == 0.0
        assert event.description is None

    def test_scene_event_duration(self):
        """Test scene duration calculation."""
        event = SceneEvent(start_time=10.5, end_time=15.3)
        assert event.duration == pytest.approx(4.8)

    def test_scene_event_defaults(self):
        """Test default values."""
        event = SceneEvent(start_time=0.0, end_time=1.0)
        assert event.event_type == "scene_change"
        assert event.confidence == 0.0
        assert event.description is None
        assert event.frame_numbers == []


class TestDetectSceneChanges:
    """Test scene change detection."""

    @pytest.mark.integration
    def test_detect_scenes_single_scene(self, tmp_path):
        """Test detecting scenes in a short uniform video (no cuts)."""
        import subprocess

        video = tmp_path / "uniform.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video),
            ],
            capture_output=True,
            check=True,
        )

        from autovideofixer.core.analysis import _detect_scene_changes

        scenes = _detect_scene_changes(str(video), threshold=0.3, min_duration_sec=0.5)
        # A uniform test pattern with no cuts should have 1 scene (the whole video)
        assert len(scenes) >= 1
        assert scenes[0].start_time == pytest.approx(0.0, abs=0.1)

    @pytest.mark.integration
    def test_detect_scenes_with_threshold(self, tmp_video_file):
        """Test that lower threshold detects more scenes."""
        from autovideofixer.core.analysis import _detect_scene_changes

        scenes_high = _detect_scene_changes(tmp_video_file, 0.5, 0.5)
        scenes_low = _detect_scene_changes(tmp_video_file, 0.1, 0.5)
        # Lower threshold should detect at least as many scenes
        assert len(scenes_low) >= len(scenes_high)

    def test_detect_scenes_nonexistent_file(self):
        """Test scene detection on a nonexistent file."""
        from autovideofixer.core.analysis import _detect_scene_changes

        scenes = _detect_scene_changes("/nonexistent/video.mp4", 0.3, 1.0)
        assert scenes == []


class TestVideoAnalyzerEvents:
    """Test VideoAnalyzer event detection."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.analyzer = VideoAnalyzer(self.config)

    @pytest.mark.integration
    def test_detect_events(self, tmp_video_file):
        """Test detecting events in a real video."""
        scenes = self.analyzer.detect_events(tmp_video_file)
        assert isinstance(scenes, list)
        assert len(scenes) >= 1

    @pytest.mark.integration
    def test_detect_events_with_min_duration(self, tmp_path):
        """Test that min_duration filters out short scenes."""
        import subprocess

        from autovideofixer.core.analysis import _detect_scene_changes

        video = tmp_path / "long.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video),
            ],
            capture_output=True,
            check=True,
        )

        scenes_long_min = _detect_scene_changes(str(video), 0.1, 3.0)
        for s in scenes_long_min:
            assert s.duration >= 3.0, f"Scene duration {s.duration} < 3.0"

    def test_clear_cache(self):
        """Test clearing analysis cache."""
        self.analyzer._analysis_cache["key"] = "value"
        assert len(self.analyzer._analysis_cache) == 1
        self.analyzer.clear_cache()
        assert len(self.analyzer._analysis_cache) == 0


class TestPerceptualHashing:
    """Test perceptual hashing functions."""

    def test_hash_nonexistent_video(self):
        """Test hashing a nonexistent file."""
        h = compute_video_hash("/nonexistent/video.mp4")
        assert h == ""

    def test_dhash_nonexistent_video(self):
        """Test dhash on a nonexistent file."""
        h = compute_video_dhash("/nonexistent/video.mp4")
        assert h == ""

    @pytest.mark.integration
    def test_hash_real_video(self, tmp_video_file):
        """Test computing hash on a real video."""
        h = compute_video_hash(tmp_video_file)
        assert len(h) > 0
        # Should be a binary string
        assert all(c in "01" for c in h)

    @pytest.mark.integration
    def test_dhash_real_video(self, tmp_video_file):
        """Test computing dhash on a real video."""
        h = compute_video_dhash(tmp_video_file)
        assert len(h) > 0
        assert all(c in "01" for c in h)

    @pytest.mark.integration
    def test_same_video_same_hash(self, tmp_video_file):
        """Test that the same video produces the same hash."""
        h1 = compute_video_hash(tmp_video_file)
        h2 = compute_video_hash(tmp_video_file)
        assert h1 == h2

    @pytest.mark.integration
    def test_different_videos_different_hashes(self, tmp_video_file, tmp_path):
        """Test that different videos have different hashes."""
        import subprocess

        video2 = tmp_path / "different.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "smptebars=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video2),
            ],
            capture_output=True,
            check=True,
        )

        h1 = compute_video_hash(tmp_video_file)
        h2 = compute_video_hash(str(video2))
        # Different test patterns should produce different hashes
        assert h1 != h2

    def test_hash_similarity_identical(self):
        """Test similarity of identical hashes."""
        h = "101100101010" * 4  # 64 bits
        sim = hash_similarity(h, h)
        assert sim == 1.0

    def test_hash_similarity_completely_different(self):
        """Test similarity of completely different hashes."""
        h1 = "0" * 64
        h2 = "1" * 64
        sim = hash_similarity(h1, h2)
        assert sim == 0.0

    def test_hash_similarity_partial(self):
        """Test similarity of partially matching hashes."""
        h1 = "11110000" * 8  # 64 bits, half ones
        h2 = "11111111" * 8  # 64 bits, all ones
        sim = hash_similarity(h1, h2)
        assert 0.0 < sim < 1.0
        assert sim == pytest.approx(0.5)

    def test_hash_similarity_different_lengths(self):
        """Test similarity of hashes with different lengths."""
        sim = hash_similarity("1010", "10101010")
        assert sim == 0.0

    def test_hash_similarity_empty(self):
        """Test similarity with empty hashes."""
        assert hash_similarity("", "1010") == 0.0
        assert hash_similarity("1010", "") == 0.0


class TestDuplicateDetection:
    """Test duplicate detection functionality."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.analyzer = VideoAnalyzer(self.config)

    @pytest.mark.integration
    def test_find_similar_same_video(self, tmp_video_file):
        """Test finding a video similar to itself (excluded)."""
        results = self.analyzer.find_similar(tmp_video_file, [tmp_video_file])
        # Same file is excluded from results
        assert tmp_video_file not in [r[0] for r in results]

    @pytest.mark.integration
    def test_find_similar_different_videos(self, tmp_video_file, tmp_path):
        """Test finding no similar videos for different content."""
        import subprocess

        video2 = tmp_path / "different.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "smptebars=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video2),
            ],
            capture_output=True,
            check=True,
        )

        results = self.analyzer.find_similar(tmp_video_file, [str(video2)], threshold=0.95)
        assert len(results) == 0

    @pytest.mark.integration
    def test_find_duplicates_batch(self, tmp_video_file, tmp_path):
        """Test batch duplicate detection."""
        import subprocess

        # Create a copy (should be detected as duplicate)
        video_copy = tmp_path / "copy.mp4"
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
                str(video_copy),
            ],
            capture_output=True,
            check=True,
        )

        # Create a different video
        video_diff = tmp_path / "different.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "smptebars=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video_diff),
            ],
            capture_output=True,
            check=True,
        )

        # With low threshold, testsrc copies should be grouped together
        results = self.analyzer.find_duplicates(
            [tmp_video_file, str(video_copy), str(video_diff)],
            threshold=0.5,
        )
        # Should find at least one group (the two testsrc videos)
        assert len(results) >= 1


class TestExtractClip:
    """Test clip extraction."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.analyzer = VideoAnalyzer(self.config)

    @pytest.mark.integration
    def test_extract_clip(self, tmp_video_file, tmp_path):
        """Test extracting a clip from a video."""
        clip = self.analyzer.extract_clip(tmp_video_file, 0.0, 1.0, output_dir=str(tmp_path))
        assert clip is not None
        assert clip.start_time == 0.0
        assert clip.end_time == 1.0
        assert os.path.exists(clip.output_path)

    @pytest.mark.integration
    def test_extract_clip_nonexistent(self):
        """Test extracting clip from nonexistent video."""
        clip = self.analyzer.extract_clip("/nonexistent/video.mp4", 0.0, 1.0)
        assert clip is None

    @pytest.mark.integration
    def test_extract_scenes_as_clips(self, tmp_video_file, tmp_path):
        """Test extracting all scenes as clips."""
        scenes = [
            SceneEvent(start_time=0.0, end_time=0.5, event_type="test"),
            SceneEvent(start_time=0.5, end_time=1.0, event_type="test"),
        ]
        clips = self.analyzer.extract_scenes_as_clips(
            tmp_video_file, scenes, output_dir=str(tmp_path)
        )
        assert len(clips) == 2
        for clip in clips:
            assert os.path.exists(clip.output_path)


class TestDirectoryScanning:
    """Test directory scanning for videos."""

    def test_scan_with_videos(self, tmp_path):
        """Test scanning directory with videos."""
        (tmp_path / "a.mp4").write_text("")
        (tmp_path / "b.mkv").write_text("")
        (tmp_path / "c.txt").write_text("")

        videos = scan_directory(str(tmp_path), recursive=False)
        assert len(videos) == 2

    def test_scan_recursive(self, tmp_path):
        """Test recursive scanning."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.mp4").write_text("")
        (subdir / "nested.mp4").write_text("")

        videos = scan_directory(str(tmp_path), recursive=True)
        assert len(videos) == 2
