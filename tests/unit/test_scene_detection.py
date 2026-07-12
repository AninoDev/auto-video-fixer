"""Tests for scene detection, clip extraction, and duplicate detection."""

import os
import tempfile
from pathlib import Path

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import (
    SceneEvent,
    VideoAnalyzer,
    _select_near_misses,
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

    @pytest.mark.integration
    def test_detect_scenes_reports_progress(self, tmp_video_file):
        """The frame-differencing loop periodically calls progress_callback."""
        from autovideofixer.core.analysis import _detect_scene_changes

        calls: list[tuple[str, str]] = []
        _detect_scene_changes(
            tmp_video_file,
            threshold=0.3,
            min_duration_sec=0.5,
            progress_callback=lambda phase, detail: calls.append((phase, detail)),
        )

        assert calls, "expected at least one progress_callback invocation"
        assert all(phase == "scene_detection" for phase, _ in calls)
        # Detail should reference frame progress (percentage and/or frame count).
        assert all("frame" in detail for _, detail in calls)

    def test_detect_scenes_no_progress_callback_is_a_noop(self):
        """Omitting progress_callback (the default) doesn't error on a missing file."""
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


class TestNearMissTracking:
    """Tests for near-miss score tracking/reporting -- added to answer "would
    lowering --scene-threshold find more cuts, and where?" without a rerun.

    `_select_near_misses` is the pure "which candidates get reported" selection
    logic pulled out of `_detect_scene_changes` (see its docstring); tested
    here directly with synthetic (time, score) pairs so it doesn't need real
    video decoding. The full logging behavior (near_miss_floor filtering
    during the actual frame-differencing loop) is covered by
    test_near_miss_logging_on_real_clip below (integration, real ffmpeg clip).
    """

    def test_select_near_misses_returns_highest_scores_first(self):
        near_misses = [(1.0, 0.05), (2.0, 0.12), (3.0, 0.09), (4.0, 0.14), (5.0, 0.02)]
        top = _select_near_misses(near_misses, limit=3)
        assert top == [(4.0, 0.14), (2.0, 0.12), (3.0, 0.09)]

    def test_select_near_misses_respects_default_limit_of_ten(self):
        # 20 synthetic candidates with scores straddling an implied threshold
        # (e.g. threshold=0.20, floor=0.05): scores 0.00-0.19 in 0.01 steps.
        near_misses = [(float(i), i / 100.0) for i in range(20)]
        top = _select_near_misses(near_misses)
        assert len(top) == 10
        # Highest-scoring synthetic candidates (0.19 down to 0.10) reported first.
        assert [score for _, score in top] == [round(x / 100, 2) for x in range(19, 9, -1)]

    def test_select_near_misses_empty_input(self):
        assert _select_near_misses([]) == []

    def test_select_near_misses_fewer_than_limit(self):
        near_misses = [(1.0, 0.08), (2.0, 0.11)]
        top = _select_near_misses(near_misses, limit=10)
        assert top == [(2.0, 0.11), (1.0, 0.08)]

    @pytest.mark.integration
    def test_near_miss_logging_on_real_clip(self, tmp_path, caplog):
        """With threshold set above this clip's real cut scores, the near-miss
        log line fires and lists candidates in the (threshold/4, threshold) band."""
        import logging

        ground_truth = TestSceneDetectionCalibration._build_ground_truth(tmp_path)

        from autovideofixer.core.analysis import _detect_scene_changes

        with caplog.at_level(logging.INFO, logger="autovideofixer.core.analysis"):
            # 0.3 is above every real cut score on this small clip (see
            # TestSceneDetectionCalibration.test_old_default_threshold_under_detects
            # -- 0.3 under-detects it), so real cuts should surface as near-misses.
            _detect_scene_changes(ground_truth, threshold=0.3, min_duration_sec=0.2)

        near_miss_records = [r for r in caplog.records if "near-miss scores" in r.message]
        assert near_miss_records, "expected a near-miss log line"
        message = near_miss_records[0].message
        assert "threshold 0.300" in message
        assert "@" in message  # "<score> @ <time>s" entries present

    @pytest.mark.integration
    def test_no_near_miss_log_when_scores_cleanly_separated(self, tmp_path, caplog):
        """At the shipped default threshold (0.15), this clip's real cuts (~0.17-0.30)
        all clear the threshold outright and its noise floor (~0.001) sits well below
        threshold/4 -- nothing lands in the near-miss band, so no log line fires."""
        import logging

        ground_truth = TestSceneDetectionCalibration._build_ground_truth(tmp_path)

        from autovideofixer.core.analysis import _detect_scene_changes

        with caplog.at_level(logging.INFO, logger="autovideofixer.core.analysis"):
            _detect_scene_changes(ground_truth, threshold=0.15, min_duration_sec=0.2)

        near_miss_records = [r for r in caplog.records if "near-miss scores" in r.message]
        assert not near_miss_records


class TestSceneDetectionCalibration:
    """Calibration test for the scene_change_threshold default (0.15).

    Builds a small synthetic ground-truth video by concatenating 4 visually
    distinct 1s segments (testsrc2, solid red, solid blue, smptebars) -- 3
    known hard cuts, no motion/noise within a segment. This is a smaller,
    faster version of the 12-segment/11-cut clip used to derive the default:
    on that larger clip the previous default (0.3) found only 9/12 segments
    (missed 3 real cuts scoring 0.18-0.27), while 0.15 found all 12 with zero
    false positives (max within-segment score measured: 0.040; min actual-cut
    score measured: 0.184). This test asserts the same "no missed cuts, no
    false positives within a segment" property holds at the shipped default.
    """

    @staticmethod
    def _build_ground_truth(tmp_path) -> str:
        import subprocess

        sources = [
            "testsrc2=size=64x36:rate=10:duration=1",
            "color=red:size=64x36:rate=10:duration=1",
            "color=blue:size=64x36:rate=10:duration=1",
            "smptebars=size=64x36:rate=10:duration=1",
        ]
        segments = []
        for i, src in enumerate(sources):
            seg = tmp_path / f"seg{i}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    src,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    "10",
                    str(seg),
                ],
                capture_output=True,
                check=True,
            )
            segments.append(seg)

        concat_file = tmp_path / "concat.txt"
        concat_file.write_text("".join(f"file '{seg}'\n" for seg in segments))

        ground_truth = tmp_path / "ground_truth.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(ground_truth),
            ],
            capture_output=True,
            check=True,
        )
        return str(ground_truth)

    @pytest.mark.integration
    def test_default_threshold_finds_all_known_cuts_no_false_positives(self, tmp_path):
        from autovideofixer.config import Config
        from autovideofixer.core.analysis import _detect_scene_changes

        default_threshold = Config.DEFAULTS["analysis"]["event_detection"]["scene_change_threshold"]
        assert default_threshold == pytest.approx(0.15)

        ground_truth = self._build_ground_truth(tmp_path)
        # min_duration_sec well under the 1s segment length so it can't mask
        # a missed/extra cut by merging segments together.
        scenes = _detect_scene_changes(ground_truth, default_threshold, 0.2)

        # 4 segments -> 4 scenes (3 detected cuts), each ~1s.
        assert len(scenes) == 4, f"expected 4 scenes (3 cuts), got {len(scenes)}: {scenes}"
        for scene in scenes:
            assert scene.duration == pytest.approx(1.0, abs=0.3)

    @pytest.mark.integration
    def test_old_default_threshold_under_detects(self, tmp_path):
        """Regression guard: 0.3 (the previous default) missed real cuts on this clip."""
        from autovideofixer.core.analysis import _detect_scene_changes

        ground_truth = self._build_ground_truth(tmp_path)
        scenes = _detect_scene_changes(ground_truth, 0.3, 0.2)
        assert len(scenes) < 4, "expected the old 0.3 threshold to under-detect on this clip"


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
