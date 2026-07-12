"""Tests for scene-based processing (core/scenes.py).

Most of these are marked @pytest.mark.integration since they exercise real
ffmpeg (splitting, concat, stabilize/interpolate stage execution) -- see
tests/conftest.py's `multi_scene_video` fixture (3 hard-cut scenes, ~6s
total). Pure-logic pieces (worker budget math) are plain unit tests.
"""

from __future__ import annotations

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import SceneEvent
from autovideofixer.core.ffmpeg_utils import probe
from autovideofixer.core.scenes import (
    _scene_worker_budget,
    concat_video_clips,
    mux_video_audio,
    run_scene_pipeline,
    split_scene_audio,
    split_scene_video,
)


def _cfg(tmp_path) -> Config:
    config = Config(tmp_path / "nonexistent.yaml")
    # multi_scene_video segments are exactly 2s each; the trailing leftover
    # segment after the last detected cut can compute a hair under 2.0s
    # depending on float/frame-timing rounding, which the default
    # min_scene_duration_sec=2.0 would then silently merge away. Lower it
    # so all 3 fixture scenes are reliably detected.
    config.set(1.0, "analysis", "event_detection", "min_scene_duration_sec")
    return config


class TestSceneWorkerBudget:
    def test_single_scene(self, tmp_path):
        scene_workers, chunks = _scene_worker_budget(1, _cfg(tmp_path))
        assert scene_workers == 1
        assert chunks >= 1

    def test_multiple_scenes_splits_budget(self, tmp_path):
        scene_workers, chunks = _scene_worker_budget(4, _cfg(tmp_path))
        assert scene_workers >= 1
        assert chunks >= 1
        # Never oversubscribe: scene_workers * chunks should not wildly
        # exceed the total budget (loose bound, exact value is cpu-dependent).
        assert scene_workers * chunks <= 16


class TestSplitAndConcat:
    @pytest.mark.integration
    def test_split_scene_video(self, multi_scene_video, tmp_path):
        scene = SceneEvent(start_time=0.0, end_time=2.0)
        out = tmp_path / "scene0.mp4"
        ok = split_scene_video(multi_scene_video, scene, str(out))
        assert ok
        assert out.exists()
        info = probe(str(out))
        assert info.duration == pytest.approx(2.0, abs=0.3)
        assert not info.has_audio  # video-only extraction

    @pytest.mark.integration
    def test_split_scene_audio(self, multi_scene_video, tmp_path):
        # multi_scene_video fixture has no audio track -- verify this fails
        # gracefully (no audio stream to map) rather than hanging/crashing.
        scene = SceneEvent(start_time=0.0, end_time=2.0)
        out = tmp_path / "scene0.m4a"
        ok = split_scene_audio(multi_scene_video, scene, str(out))
        assert ok is False

    @pytest.mark.integration
    def test_concat_video_clips_single(self, multi_scene_video, tmp_path):
        out = tmp_path / "concat.mp4"
        ok = concat_video_clips([multi_scene_video], str(out), str(tmp_path))
        assert ok
        assert out.exists()

    @pytest.mark.integration
    def test_concat_video_clips_multiple(self, multi_scene_video, tmp_path):
        scene = SceneEvent(start_time=0.0, end_time=2.0)
        clip_a = tmp_path / "a.mp4"
        clip_b = tmp_path / "b.mp4"
        assert split_scene_video(multi_scene_video, scene, str(clip_a))
        assert split_scene_video(multi_scene_video, scene, str(clip_b))

        out = tmp_path / "concat.mp4"
        ok = concat_video_clips([str(clip_a), str(clip_b)], str(out), str(tmp_path))
        assert ok
        info = probe(str(out))
        assert info.duration == pytest.approx(4.0, abs=0.5)

    @pytest.mark.integration
    def test_mux_video_audio_no_audio_path_copies(self, multi_scene_video, tmp_path):
        out = tmp_path / "muxed.mp4"
        ok = mux_video_audio(multi_scene_video, None, str(out))
        assert ok
        assert out.exists()


class TestRunScenePipeline:
    @pytest.mark.integration
    def test_fewer_than_two_scenes_returns_none(self, tmp_path, multi_scene_video):
        """A video with < 2 detected scenes has nothing to split -- scene mode
        must fall back (return None) rather than doing pointless work."""
        config = _cfg(tmp_path)
        # Force a threshold so high nothing is detected as a cut.
        config.set(0.99, "analysis", "event_detection", "scene_change_threshold")
        result = run_scene_pipeline(
            multi_scene_video, config, run_stabilize=False, run_interpolate=False, target_fps=None
        )
        assert result is None

    @pytest.mark.integration
    def test_full_pipeline_no_drop(self, multi_scene_video, tmp_path):
        config = _cfg(tmp_path)
        result = run_scene_pipeline(
            multi_scene_video,
            config,
            run_stabilize=True,
            run_interpolate=False,
            target_fps=None,
        )
        assert result is not None
        assert result.total_scenes >= 2
        assert result.kept_scenes == result.total_scenes
        assert result.dropped_scenes == []

        orig_duration = probe(multi_scene_video).duration
        out_duration = probe(result.output_path).duration
        # Duration preserved within a generous tolerance (re-encoding at
        # scene boundaries can introduce small rounding, but not seconds).
        assert out_duration == pytest.approx(orig_duration, abs=1.0)

    @pytest.mark.integration
    def test_interpolate_never_crosses_a_cut(self, multi_scene_video, tmp_path):
        """Each scene is interpolated independently -- the output must have a
        uniform target framerate and a duration close to the original
        (interpolation doesn't change a scene's time span, only its frame
        count)."""
        config = _cfg(tmp_path)
        result = run_scene_pipeline(
            multi_scene_video,
            config,
            run_stabilize=False,
            run_interpolate=True,
            target_fps=48.0,
        )
        assert result is not None
        out_info = probe(result.output_path)
        assert out_info.framerate == pytest.approx(48.0, abs=1.0)
        orig_duration = probe(multi_scene_video).duration
        assert out_info.duration == pytest.approx(orig_duration, abs=1.0)

    @pytest.mark.integration
    def test_drop_non_content_fails_open_without_reachable_llm(self, multi_scene_video, tmp_path):
        """With no VLM/coordinator reachable (default local ollama at
        localhost:11434, not running in the test environment), drop mode must
        fail open: every scene kept, nothing dropped."""
        config = _cfg(tmp_path)
        config.set(True, "scenes", "drop_non_content")
        # Point at a guaranteed-unreachable port so this doesn't depend on
        # whether the test host happens to have something on 11434.
        config.set("http://127.0.0.1:1", "analysis", "vlm", "api_url")
        config.set("ollama", "analysis", "vlm", "provider")
        config.set("http://127.0.0.1:1", "analysis", "llm", "api_url")
        config.set("ollama", "analysis", "llm", "provider")

        result = run_scene_pipeline(
            multi_scene_video,
            config,
            run_stabilize=False,
            run_interpolate=False,
            target_fps=None,
        )
        assert result is not None
        assert result.dropped_scenes == []
        assert result.kept_scenes == result.total_scenes
