"""Unit tests for core.output_check.compute_fitted_dimensions() (REQUIREMENTS.md § 7).

Covers both fit modes (preserve_aspect / snap_limiting), keep_aspect_ratio=False,
a non-default dimension_multiple, and input validation.
"""

from __future__ import annotations

import pytest

from autovideofixer.core.output_check import compute_fitted_dimensions


class TestPreserveAspect:
    def test_landscape_exact_fit(self):
        # 3840x2160 (16:9) input against a [1920, 1080] target -> exact fit.
        assert compute_fitted_dimensions(3840, 2160, 1920, 1080) == (1920, 1080)

    def test_portrait_rotated_target(self):
        # Portrait input against a landscape preset target rotates the bounding box.
        assert compute_fitted_dimensions(1080, 1920, 1920, 1080) == (1080, 1920)

    def test_off_aspect_landscape_rounds_up_to_even(self):
        # A 4K input that's slightly off 16:9 (e.g. 3840x2158) truncates then rounds
        # UP to the next even value on both axes under preserve_aspect -- this is the
        # "1918/1078"-style short-of-clean-target result snap_limiting fixes.
        w, h = compute_fitted_dimensions(3840, 2158, 1920, 1080)
        # height-bound: scale = 1080/2158 = 0.500463...; w = int(3840*scale) = 1921 -> 1922
        assert h == 1080
        assert w % 2 == 0

    def test_square_input_uses_shorter_edge(self):
        assert compute_fitted_dimensions(1000, 1000, 1920, 1080) == (1080, 1080)

    def test_matches_upscale_stage_existing_behavior(self):
        # Reproduces UpscaleStage._calculate_target_dimensions()'s pre-refactor
        # inline scale+_round_to_even for a representative off-aspect input
        # (this is the exact repro AGENTS.md's "Upscaling & Aspect Ratio"
        # section cites: 1072x1908 vs. a [1920, 1080] preset target rotates
        # to a 1080x1920 bound, landing width a couple px short at 1078).
        input_w, input_h = 1072, 1908
        target_w, target_h = 1920, 1080
        bound_w, bound_h = 1080, 1920  # portrait input rotates the landscape target
        scale = min(bound_w / input_w, bound_h / input_h)
        expected_w = int(input_w * scale)
        expected_h = int(input_h * scale)
        expected_w = expected_w if expected_w % 2 == 0 else expected_w + 1
        expected_h = expected_h if expected_h % 2 == 0 else expected_h + 1
        assert compute_fitted_dimensions(input_w, input_h, target_w, target_h) == (
            expected_w,
            expected_h,
        )
        assert compute_fitted_dimensions(input_w, input_h, target_w, target_h) == (1078, 1920)


class TestSnapLimiting:
    """snap_limiting is "snap-to-box-when-close": the limiting axis always
    lands exactly on its target bound; the derived (non-limiting) axis is
    ALSO snapped exactly onto its bound when it would otherwise fall short
    by no more than snap_tolerance (default 1%), else it's left at its
    aspect-preserving value rounded to the nearest multiple."""

    def test_snaps_both_axes_when_within_tolerance_height_limiting(self):
        # Height is the limiting axis (scale_h=1080/812 < scale_w=1920/1440);
        # derived width is 1915.27 -- a 0.25% gap from 1920, well within the
        # default 1% snap_tolerance -- so BOTH axes land exactly on target.
        assert compute_fitted_dimensions(1440, 812, 1920, 1080, fit_mode="snap_limiting") == (
            1920,
            1080,
        )

    def test_snaps_both_axes_when_within_tolerance_width_limiting(self):
        # Width is the limiting axis (scale_w=1920/1920=1.0 < scale_h); derived
        # height is exactly 1076 -- a 0.37% gap from 1080, within tolerance --
        # so BOTH axes land exactly on target.
        assert compute_fitted_dimensions(1920, 1076, 1920, 1080, fit_mode="snap_limiting") == (
            1920,
            1080,
        )

    def test_no_snap_for_genuinely_different_aspect_ratio(self):
        # Width is limiting (lands exactly on 1920); derived height is 1053 --
        # a 2.5% gap from 1080, beyond the default 1% tolerance -- so the
        # derived axis is left at its aspect-preserving value (rounded to the
        # nearest multiple, 1052 via round-half-to-even) instead of snapping.
        assert compute_fitted_dimensions(3840, 2106, 1920, 1080, fit_mode="snap_limiting") == (
            1920,
            1052,
        )

    def test_no_snap_just_outside_tolerance(self):
        # 1440x819 against [1920, 1080]: derived width gap is ~1.10%, just
        # over the default 1% tolerance -- must NOT snap to 1920.
        w, h = compute_fitted_dimensions(1440, 819, 1920, 1080, fit_mode="snap_limiting")
        assert h == 1080  # limiting axis still exact
        assert w != 1920  # derived axis stays off-target (not snapped)

    def test_custom_snap_tolerance_widens_snapping(self):
        # The same just-outside-default-tolerance case snaps once tolerance
        # is raised past its ~1.10% gap.
        w, h = compute_fitted_dimensions(
            1440, 819, 1920, 1080, fit_mode="snap_limiting", snap_tolerance=0.02
        )
        assert (w, h) == (1920, 1080)

    def test_snap_tolerance_ignored_by_preserve_aspect(self):
        # A large snap_tolerance must not affect preserve_aspect's output.
        assert compute_fitted_dimensions(
            3840, 2106, 1920, 1080, fit_mode="preserve_aspect", snap_tolerance=1.0
        ) == compute_fitted_dimensions(
            3840, 2106, 1920, 1080, fit_mode="preserve_aspect", snap_tolerance=0.0
        )

    def test_never_returns_zero_even_for_tiny_multiple_mismatch(self):
        w, h = compute_fitted_dimensions(1, 1000000, 1920, 1080, fit_mode="snap_limiting")
        assert w >= 2
        assert h >= 2


class TestKeepAspectRatioFalse:
    def test_exact_target_both_modes_identical(self):
        for mode in ("preserve_aspect", "snap_limiting"):
            assert compute_fitted_dimensions(
                640, 480, 1921, 1079, keep_aspect_ratio=False, fit_mode=mode
            ) == (1920, 1080)


class TestNonDefaultMultiple:
    def test_multiple_of_four(self):
        w, h = compute_fitted_dimensions(3840, 2160, 1921, 1081, multiple=4)
        assert w % 4 == 0
        assert h % 4 == 0

    def test_multiple_of_four_snap(self):
        w, h = compute_fitted_dimensions(
            3841, 2160, 1920, 1080, fit_mode="snap_limiting", multiple=4
        )
        assert w % 4 == 0
        assert h % 4 == 0


class TestValidation:
    def test_bad_fit_mode_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            compute_fitted_dimensions(100, 100, 200, 200, fit_mode="bogus")

    def test_bad_multiple_raises(self):
        with pytest.raises(ValueError):
            compute_fitted_dimensions(100, 100, 200, 200, multiple=0)

    def test_negative_multiple_raises(self):
        with pytest.raises(ValueError):
            compute_fitted_dimensions(100, 100, 200, 200, multiple=-2)

    def test_snap_tolerance_above_one_raises(self):
        with pytest.raises(ValueError, match="snap_tolerance"):
            compute_fitted_dimensions(100, 100, 200, 200, snap_tolerance=1.5)

    def test_snap_tolerance_negative_raises(self):
        with pytest.raises(ValueError, match="snap_tolerance"):
            compute_fitted_dimensions(100, 100, 200, 200, snap_tolerance=-0.1)
