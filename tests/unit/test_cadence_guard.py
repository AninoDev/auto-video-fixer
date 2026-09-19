"""Unit tests for core/cadence_guard.py (REQUIREMENTS.md § 19.1).

Pure unit tests -- no ffmpeg, no GPU, no video files. Covers
``resolve_cadence()``'s frame-count-vs-cadence check, including the exact
714-frames/23.8s/9.864fps shape of the real incident this guard exists to
catch (see the module docstring), and ``cadence_metadata()``.
"""

from __future__ import annotations

import pytest

from autovideofixer.core.cadence_guard import (
    DEFAULT_CADENCE_GUARD_TOLERANCE,
    CadenceResolution,
    cadence_metadata,
    resolve_cadence,
)


class TestResolveCadenceMatching:
    def test_matching_counts_no_mismatch_carrier_unchanged(self):
        # 30fps for 10s -> exactly 300 frames.
        result = resolve_cadence(actual_frames=300, input_duration=10.0, candidate_fps=30.0)
        assert result.mismatch is False
        assert result.resolved_fps == 30.0
        assert result.reason is None
        assert result.expected_frames == pytest.approx(300.0)

    def test_small_rounding_difference_within_tolerance_no_mismatch(self):
        # 299 vs expected 300 is well within the default 15% tolerance.
        result = resolve_cadence(actual_frames=299, input_duration=10.0, candidate_fps=30.0)
        assert result.mismatch is False
        assert result.resolved_fps == 30.0


class TestResolveCadenceIncidentRegression:
    """The exact shape of the real incident (REQUIREMENTS.md § 19,
    module docstring): a 23.8s clip's raw pipe re-expanded 235 frames back
    to 714, then tagged them at the 235-frame-derived cadence of 9.864fps
    -- correct only for 235 frames, giving 714 / 9.864 = 72.4s."""

    def test_3x_disagreement_flags_mismatch_and_resolves_near_30fps(self):
        result = resolve_cadence(actual_frames=714, input_duration=23.8, candidate_fps=9.864)
        assert result.mismatch is True
        assert result.reason is not None
        assert "714" in result.reason
        assert "9.864" in result.reason
        # The bad carrier must NOT be trusted.
        assert result.resolved_fps != pytest.approx(9.864, abs=0.01)
        # Self-consistent resolution: 714 / 23.8 ~= 30fps (the real cadence).
        assert result.resolved_fps == pytest.approx(714 / 23.8, rel=1e-9)
        assert result.resolved_fps == pytest.approx(30.0, abs=0.5)
        assert result.expected_frames == pytest.approx(23.8 * 9.864)
        assert result.actual_frames == 714

    def test_incident_case_does_not_reproduce_stretched_duration(self):
        """Using resolved_fps instead of the bad carrier gives back the
        original (correct) duration, not the stretched 72.4s."""
        result = resolve_cadence(actual_frames=714, input_duration=23.8, candidate_fps=9.864)
        reconstructed_duration = result.actual_frames / result.resolved_fps
        assert reconstructed_duration == pytest.approx(23.8, rel=1e-9)
        # The bad carrier's own implied duration WOULD have been ~72.4s --
        # sanity-check that this is in fact the bug being avoided.
        assert 714 / 9.864 == pytest.approx(72.4, abs=0.1)


class TestResolveCadenceUnknownOrZero:
    def test_zero_duration_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=714, input_duration=0.0, candidate_fps=9.864)
        assert result.mismatch is False
        assert result.resolved_fps == 9.864

    def test_none_duration_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=714, input_duration=None, candidate_fps=9.864)
        assert result.mismatch is False
        assert result.resolved_fps == 9.864

    def test_negative_duration_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=714, input_duration=-1.0, candidate_fps=9.864)
        assert result.mismatch is False

    def test_zero_candidate_fps_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=714, input_duration=23.8, candidate_fps=0.0)
        assert result.mismatch is False
        assert result.resolved_fps == 0.0

    def test_none_candidate_fps_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=714, input_duration=23.8, candidate_fps=None)
        assert result.mismatch is False
        assert result.resolved_fps == 0.0

    def test_zero_actual_frames_no_crash_no_false_alarm(self):
        # A genuinely empty output is a different failure mode (stages
        # already fail outright on "no frames produced") -- not this guard's
        # job.
        result = resolve_cadence(actual_frames=0, input_duration=23.8, candidate_fps=30.0)
        assert result.mismatch is False
        assert result.resolved_fps == 30.0

    def test_negative_actual_frames_no_crash_no_false_alarm(self):
        result = resolve_cadence(actual_frames=-5, input_duration=23.8, candidate_fps=30.0)
        assert result.mismatch is False


class TestResolveCadenceToleranceBoundary:
    """expected_frames = 10 * 30 = 300; default tolerance 0.15 -> the
    boundary is +/- 45 frames (255..345)."""

    def test_just_inside_tolerance_no_mismatch(self):
        # 44/300 = 0.1467 < 0.15
        result = resolve_cadence(actual_frames=344, input_duration=10.0, candidate_fps=30.0)
        assert result.mismatch is False
        assert result.resolved_fps == 30.0

    def test_just_outside_tolerance_mismatch(self):
        # 46/300 = 0.1533 > 0.15
        result = resolve_cadence(actual_frames=346, input_duration=10.0, candidate_fps=30.0)
        assert result.mismatch is True
        assert result.resolved_fps == pytest.approx(346 / 10.0)

    def test_exactly_at_tolerance_boundary_no_mismatch(self):
        # abs(ratio - 1.0) <= tolerance is inclusive.
        result = resolve_cadence(actual_frames=345, input_duration=10.0, candidate_fps=30.0)
        assert result.mismatch is False

    def test_custom_tolerance_respected(self):
        # 320/300 = 0.0667 over -- mismatch under a strict 5% tolerance...
        result = resolve_cadence(
            actual_frames=320, input_duration=10.0, candidate_fps=30.0, tolerance=0.05
        )
        assert result.mismatch is True
        # ...but not under the looser default.
        result_default = resolve_cadence(actual_frames=320, input_duration=10.0, candidate_fps=30.0)
        assert result_default.mismatch is False


class TestDefaultTolerance:
    def test_default_tolerance_constant(self):
        assert DEFAULT_CADENCE_GUARD_TOLERANCE == 0.15


class TestCadenceMetadata:
    def test_no_mismatch_returns_empty_dict(self):
        result = resolve_cadence(actual_frames=300, input_duration=10.0, candidate_fps=30.0)
        assert cadence_metadata(result) == {}

    def test_mismatch_returns_expected_fields(self):
        result = resolve_cadence(actual_frames=714, input_duration=23.8, candidate_fps=9.864)
        meta = cadence_metadata(result)
        assert meta["cadence_mismatch"] is True
        assert meta["cadence_expected_frames"] == result.expected_frames
        assert meta["cadence_actual_frames"] == 714
        assert meta["cadence_resolved_fps"] == result.resolved_fps

    def test_dataclass_defaults(self):
        result = CadenceResolution(resolved_fps=30.0, mismatch=False)
        assert result.reason is None
        assert result.expected_frames is None
        assert result.actual_frames == 0
