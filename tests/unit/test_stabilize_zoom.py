"""Unit tests for StabilizeStage's SIGNED zoom_coverage dial (REQUIREMENTS.md § 17).

`stages.stabilize.zoom_coverage=0.0` used to mean "no zoom" -- a bug, since no zoom is not no
cropping (a stabilized frame is still translated/rotated by the smoothing path, so it loses
content off one edge while showing a border on the other). The fix makes zoom_coverage a SIGNED
dial spanning zoom-OUT (preserve content) through no-zoom to zoom-IN (eliminate borders):

    0.00 -> zoom=-max(B)   (every frame's content fully preserved, borders everywhere)
    0.25 -> zoom=-p50(B)   (~50% of frames fully preserved)
    0.50 -> zoom=0         (no zoom at all -- vidstabtransform's own default, and the OLD 0.0
                            behaviour)
    0.75 -> zoom=+p50(B)   (~50% of frames border-free)
    1.00 -> zoom=+max(B)   (every frame border-free -- delegated to optzoom=1 exactly as before,
                            bit-for-bit unchanged)

These tests cover the pure q -> signed-zoom mapping (`StabilizeStage._signed_zoom_from_coverage`/
`_zoom_quantile`) given synthetic B values, so expected quantiles are exact -- the TRF-derived B
values themselves can't be unit-tested precisely (§ 17.3's accuracy caveat), only the mapping.
"""

from __future__ import annotations

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.stabilize import StabilizeStage


def _make_stage(tmp_path) -> StabilizeStage:
    config = Config(tmp_path / "nonexistent.yaml")
    return StabilizeStage(config)


# A synthetic B (per-frame required-zoom-IN) distribution with an EXACT, hand-computable
# nearest-rank median and max: sorted ascending, 21 values (odd count -> unambiguous median
# index), spaced 1.0 apart: [0.0, 1.0, 2.0, ..., 20.0]. Nearest-rank quantile at p uses
# idx = round(p * (n - 1)) = round(p * 20).
B_VALUES = [float(v) for v in range(21)]  # 0.0 .. 20.0, unsorted-order irrelevant to quantile
B_MAX = 20.0
B_P50 = 10.0  # idx = round(0.5 * 20) = 10 -> B_VALUES[10] == 10.0


class TestZoomQuantile:
    """Pure nearest-rank quantile helper."""

    def test_empty_returns_zero(self):
        assert StabilizeStage._zoom_quantile([], 0.5) == 0.0

    def test_single_value_returns_that_value_for_any_p(self):
        for p in (0.0, 0.3, 0.7, 1.0):
            assert StabilizeStage._zoom_quantile([42.0], p) == 42.0

    def test_all_zeros_returns_zero(self):
        assert StabilizeStage._zoom_quantile([0.0] * 10, 0.9) == 0.0

    def test_p_zero_is_min(self):
        assert StabilizeStage._zoom_quantile(B_VALUES, 0.0) == 0.0

    def test_p_one_is_max(self):
        assert StabilizeStage._zoom_quantile(B_VALUES, 1.0) == B_MAX

    def test_p_half_is_median(self):
        assert StabilizeStage._zoom_quantile(B_VALUES, 0.5) == B_P50

    def test_p_clamped_below_zero(self):
        assert StabilizeStage._zoom_quantile(B_VALUES, -5.0) == 0.0

    def test_p_clamped_above_one(self):
        assert StabilizeStage._zoom_quantile(B_VALUES, 5.0) == B_MAX

    def test_unsorted_input_same_as_sorted(self):
        shuffled = [10.0, 2.0, 20.0, 0.0, 7.0, 15.0]
        assert StabilizeStage._zoom_quantile(shuffled, 1.0) == StabilizeStage._zoom_quantile(
            sorted(shuffled), 1.0
        )


class TestSignedZoomFromCoverageTable:
    """The exact q -> signed-zoom mapping across the whole range, per REQUIREMENTS.md § 17's
    table, using B_VALUES where max(B)=20.0 and p50(B)=10.0 are exact by construction."""

    def test_q_0_00_is_negative_max(self):
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.0) == -B_MAX

    def test_q_0_25_is_negative_p50(self):
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.25) == -B_P50

    def test_q_0_50_is_zero(self):
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.5) == 0.0

    def test_q_0_75_is_positive_p50(self):
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.75) == B_P50

    def test_q_1_00_is_positive_max(self):
        """Note: execute() never actually calls this function at q=1.0 (it delegates to
        optzoom=1 directly -- see TestQ1DelegatesToOptzoom below); this only checks the pure
        mapping function's own value at that endpoint is consistent with the table."""
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 1.0) == B_MAX


class TestSignedZoomFromCoverageContinuity:
    def test_continuous_through_zero_at_half(self):
        below = StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.5 - 1e-9)
        at = StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.5)
        above = StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.5 + 1e-9)
        assert at == 0.0
        # Approaching from either side stays near zero -- much smaller in magnitude than
        # either true endpoint (-20.0 / +20.0).
        assert abs(below) < 1.0
        assert abs(above) < 1.0

    def test_approaches_from_below_are_non_positive(self):
        for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.499):
            assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, q) <= 0.0

    def test_approaches_from_above_are_non_negative(self):
        for q in (0.501, 0.6, 0.7, 0.8, 0.9, 1.0):
            assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, q) >= 0.0


class TestSignedZoomFromCoverageMonotonic:
    def test_monotonically_increasing_across_full_range(self):
        qs = [i / 100.0 for i in range(0, 101)]
        values = [StabilizeStage._signed_zoom_from_coverage(B_VALUES, q) for q in qs]
        for a, b in zip(values, values[1:]):
            assert b >= a


class TestSignedZoomFromCoverageClamping:
    def test_clamped_to_negative_100(self):
        huge_b = [500.0] * 5
        assert StabilizeStage._signed_zoom_from_coverage(huge_b, 0.0) == -100.0

    def test_clamped_to_positive_100(self):
        huge_b = [500.0] * 5
        assert StabilizeStage._signed_zoom_from_coverage(huge_b, 1.0) == 100.0

    def test_within_range_not_clamped(self):
        assert StabilizeStage._signed_zoom_from_coverage(B_VALUES, 0.0) == -20.0


class TestSignedZoomFromCoverageDegenerateInputs:
    """Must never raise or divide by zero."""

    def test_empty_b_values(self):
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert StabilizeStage._signed_zoom_from_coverage([], q) == 0.0

    def test_all_zero_b_values(self):
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert StabilizeStage._signed_zoom_from_coverage([0.0] * 8, q) == 0.0

    def test_single_b_value(self):
        assert StabilizeStage._signed_zoom_from_coverage([15.0], 0.0) == -15.0
        assert StabilizeStage._signed_zoom_from_coverage([15.0], 0.5) == 0.0
        assert StabilizeStage._signed_zoom_from_coverage([15.0], 1.0) == 15.0


class TestQ1DelegatesToOptzoom:
    """q=1.0 must keep delegating to optzoom=1 exactly as today, so the default stays
    bit-for-bit unchanged -- execute()'s zoom_param branch, not the estimate function."""

    def _zoom_param_for(self, stage, apply_zoom):
        if not apply_zoom:
            return ":zoom=0:optzoom=0"
        if stage._zoom_coverage >= 1.0:
            return ":zoom=0:optzoom=1"
        b_values = stage._compute_required_zoom_percentages("unused.trf", 1920, 1080)
        pct = stage._signed_zoom_from_coverage(b_values, stage._zoom_coverage)
        return f":zoom={pct:.4f}:optzoom=0"

    def test_default_coverage_is_one_and_delegates(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._zoom_coverage == 1.0
        assert self._zoom_param_for(stage, apply_zoom=True) == ":zoom=0:optzoom=1"

    def test_explicit_one_delegates(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = 1.0
        assert self._zoom_param_for(stage, apply_zoom=True) == ":zoom=0:optzoom=1"


class TestZoomEnabledFalseGate:
    """§ 17.1: zoom_enabled=False (apply_zoom=False, the movement-extent gate result) must
    still emit NO zoom at all -- never redirected to the zoom-out behaviour, regardless of
    zoom_coverage."""

    def _zoom_param_for(self, stage, apply_zoom):
        if not apply_zoom:
            return ":zoom=0:optzoom=0"
        if stage._zoom_coverage >= 1.0:
            return ":zoom=0:optzoom=1"
        b_values = stage._compute_required_zoom_percentages("unused.trf", 1920, 1080)
        pct = stage._signed_zoom_from_coverage(b_values, stage._zoom_coverage)
        return f":zoom={pct:.4f}:optzoom=0"

    @pytest.mark.parametrize("coverage", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_gate_false_always_no_zoom(self, tmp_path, coverage):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = coverage
        assert self._zoom_param_for(stage, apply_zoom=False) == ":zoom=0:optzoom=0"
