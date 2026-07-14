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


class TestRustPythonParity:
    """Differential test: the Rust (avf_scenes) and pure-Python scene
    detection implementations must find the same scene boundaries, per
    docs/REQUIREMENTS.md R5.1's verification bar. Skipped (not failed) if the
    avf_scenes extension isn't built in this environment -- see AGENTS.md's
    Setup & Commands for the build step.
    """

    @staticmethod
    def _require_rust():
        from autovideofixer.core.analysis import _detect_scene_changes_rs_native

        if _detect_scene_changes_rs_native is None:
            pytest.skip("avf_scenes Rust extension not built in this environment")

    @staticmethod
    def _assert_scenes_match(py_scenes, rs_scenes, time_tol=0.05, score_tol=0.01):
        assert len(py_scenes) == len(rs_scenes), (
            f"scene count mismatch: python={len(py_scenes)} rust={len(rs_scenes)}"
        )
        for p, r in zip(py_scenes, rs_scenes):
            assert p.start_time == pytest.approx(r.start_time, abs=time_tol)
            assert p.end_time == pytest.approx(r.end_time, abs=time_tol)
            assert p.confidence == pytest.approx(r.confidence, abs=score_tol)

    @pytest.mark.integration
    def test_parity_on_calibration_fixture(self, tmp_path):
        self._require_rust()
        from autovideofixer.core.analysis import (
            _detect_scene_changes_python,
            _detect_scene_changes_rust,
        )

        ground_truth = TestSceneDetectionCalibration._build_ground_truth(tmp_path)

        py_scenes, py_cuts, _ = _detect_scene_changes_python(ground_truth, 0.15, 0.2)
        rs_scenes, rs_cuts, _ = _detect_scene_changes_rust(ground_truth, 0.15, 0.2)

        self._assert_scenes_match(py_scenes, rs_scenes)
        assert len(py_cuts) == len(rs_cuts)
        for p, r in zip(py_cuts, rs_cuts):
            # Bounded float divergence from the different decode/scale paths
            # (cv2's INTER_LINEAR + BT.601 vs ffmpeg's bilinear scale +
            # format=gray) -- measured well under 0.005 on this fixture.
            assert p == pytest.approx(r, abs=0.005)

    @pytest.mark.integration
    def test_parity_on_video_with_motion(self, tmp_path):
        """A clip with continuous panning/motion (no real cuts) followed by one
        hard cut -- exercises the near-miss-scoring band, not just clean cuts."""
        import subprocess

        self._require_rust()
        from autovideofixer.core.analysis import (
            _detect_scene_changes_python,
            _detect_scene_changes_rust,
        )

        seg1 = tmp_path / "pan.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "mandelbrot=size=320x180:rate=25",
                "-t",
                "3",
                "-vf",
                "zoompan=z='min(zoom+0.002,1.3)':d=1:s=320x180",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(seg1),
            ],
            capture_output=True,
            check=True,
        )
        seg2 = tmp_path / "life.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "life=size=320x180:rate=25:mold=10",
                "-t",
                "3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(seg2),
            ],
            capture_output=True,
            check=True,
        )
        concat_file = tmp_path / "concat.txt"
        concat_file.write_text(f"file '{seg1}'\nfile '{seg2}'\n")
        motion_video = tmp_path / "motion.mp4"
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
                str(motion_video),
            ],
            capture_output=True,
            check=True,
        )

        py_scenes, _, _ = _detect_scene_changes_python(str(motion_video), 0.15, 0.2)
        rs_scenes, _, _ = _detect_scene_changes_rust(str(motion_video), 0.15, 0.2)

        self._assert_scenes_match(py_scenes, rs_scenes)


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
