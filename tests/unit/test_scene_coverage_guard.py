"""Scene mode must never silently delete the part of a video it didn't detect.

Scene mode processes ONLY the detected scenes and concatenates them, so any
span the scene list doesn't cover is DELETED from the output. A scene list's
final `end_time` comes from the last frame the detector actually decoded, so if
decoding stops early the tail of the video just disappears -- no error, no
warning, every stage COMPLETED, exit 0.

Real incident (2026-09-19): a 44.93s input produced scenes covering only
0.000-11.208s. 33.7s -- 75% of the video -- was dropped. Every stage then
processed 673 frames instead of ~2700. With a 0.25x speed stage downstream, the
surviving 11.2s became a 45s output, which looked superficially plausible next
to a 44.93s input, which is exactly why it went unnoticed.
"""

from __future__ import annotations

from autovideofixer.core.analysis import SceneEvent
from autovideofixer.core.scenes import SCENE_COVERAGE_TOLERANCE_SEC


def _threshold(duration: float) -> float:
    """The guard's effective shortfall threshold, mirroring run_scene_pipeline."""
    return max(SCENE_COVERAGE_TOLERANCE_SEC, duration * 0.02)


def _fires(covered: float, duration: float) -> bool:
    return (duration - covered) > _threshold(duration)


class TestCoverageThreshold:
    def test_the_real_incident_trips_the_guard(self):
        """44.93s input, scenes covering 11.208s -> must be caught."""
        assert _fires(11.208333, 44.93)

    def test_normal_trailing_rounding_does_not_trip_it(self):
        """A few dropped trailing frames must not fire a warning every run."""
        assert not _fires(44.90, 44.93)

    def test_half_second_short_on_a_short_clip_does_not_trip_it(self):
        assert not _fires(9.6, 10.0)

    def test_threshold_scales_with_duration(self):
        """2% of a long input is a bigger absolute allowance than the floor."""
        assert _threshold(10.0) == SCENE_COVERAGE_TOLERANCE_SEC
        assert _threshold(600.0) == 12.0

    def test_losing_a_quarter_of_a_long_video_trips_it(self):
        assert _fires(450.0, 600.0)


class TestFinalSceneExtension:
    """The repair: extend the last scene rather than dropping the tail."""

    def _scenes(self):
        return [
            SceneEvent(start_time=0.0, end_time=4.9, confidence=0.5),
            SceneEvent(start_time=4.9, end_time=8.754166, confidence=0.5),
            SceneEvent(start_time=8.754166, end_time=11.208333, confidence=0.5),
        ]

    def test_extending_recovers_the_whole_duration(self):
        scenes = self._scenes()
        duration = 44.93

        assert _fires(scenes[-1].end_time, duration)
        scenes[-1].end_time = duration

        assert scenes[-1].end_time == duration
        assert not _fires(scenes[-1].end_time, duration)

    def test_extension_preserves_earlier_boundaries(self):
        """Only the tail is repaired; detected cuts stay where they were."""
        scenes = self._scenes()
        original = [(s.start_time, s.end_time) for s in scenes[:-1]]
        scenes[-1].end_time = 44.93

        assert [(s.start_time, s.end_time) for s in scenes[:-1]] == original

    def test_extended_scene_duration_is_consistent(self):
        scenes = self._scenes()
        scenes[-1].end_time = 44.93

        assert scenes[-1].duration == 44.93 - scenes[-1].start_time
        assert sum(s.duration for s in scenes) == 44.93
