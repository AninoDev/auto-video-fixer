"""Unit tests for core/vlm_sampling.py (REQUIREMENTS.md § 25): the pure planner, the
duplicate-removal/refill selection (with a fake frame source) and the config/endpoint
validation. Real-media behaviour is covered in tests/integration/test_vlm_sampling_e2e.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autovideofixer.core.vlm_sampling import (
    ABSOLUTE_MIN_LONG_SIDE_PX,
    Candidate,
    SamplingConfig,
    SamplingReport,
    VisionLimits,
    base_frame_count,
    fit_scale,
    motion_score,
    pixel_cap_scale,
    plan_sampling,
    scaled_size,
    segment_centres,
    select_frames,
    temporal_preference,
    thumb_diff,
    tiles_for,
)
from autovideofixer.endpoints import resolve_vision

CFG = SamplingConfig()


# ---------------------------------------------------------------- § 25.1 frame count


class TestBaseFrameCount:
    def test_ceil_of_duration_over_interval(self):
        cfg = SamplingConfig(min_frames=1, max_frames=20, interval_sec=10)
        assert base_frame_count(25, cfg) == 3
        assert base_frame_count(30, cfg) == 3
        assert base_frame_count(30.1, cfg) == 4

    def test_clamped_up_to_min_and_down_to_max(self):
        assert base_frame_count(5, CFG) == 4  # ceil = 1 -> min_frames
        assert base_frame_count(1000, CFG) == 8  # ceil = 100 -> max_frames

    def test_at_least_one_even_for_zero_duration(self):
        assert base_frame_count(0, SamplingConfig(min_frames=1)) == 1

    def test_max_frames_per_call_beats_min_frames(self):
        assert base_frame_count(100, CFG, max_frames_per_call=2) == 2

    def test_never_exceeds_real_frames(self):
        assert base_frame_count(100, CFG, real_frames=3) == 3
        assert base_frame_count(100, CFG, real_frames=1) == 1

    def test_zero_real_frames_still_returns_one(self):
        assert base_frame_count(100, CFG, real_frames=0) == 1


def test_segment_centres_are_segment_midpoints():
    assert segment_centres(10.0, 8.0, 4) == [11.0, 13.0, 15.0, 17.0]
    assert segment_centres(0.0, 10.0, 1) == [5.0]
    assert segment_centres(0.0, 0.0, 3) == []
    assert segment_centres(0.0, 5.0, 0) == []


# ---------------------------------------------------------------- § 25.3 tile maths


class TestTiles:
    LIM = VisionLimits(tile_width=448, tile_height=448)

    def test_axes_round_up_independently(self):
        assert tiles_for(1000, 500, self.LIM) == 3 * 2
        assert tiles_for(448, 448, self.LIM) == 1
        assert tiles_for(449, 448, self.LIM) == 2
        assert tiles_for(1, 1, self.LIM) == 1

    def test_rectangular_tiles(self):
        lim = VisionLimits(tile_width=100, tile_height=50)
        assert tiles_for(250, 120, lim) == 3 * 3

    def test_disabled_without_tile_size(self):
        assert tiles_for(1920, 1080, VisionLimits()) is None


class TestPixelCaps:
    def test_long_side_cap_preserves_aspect(self):
        lim = VisionLimits(max_long_side_px=720)
        plan = plan_sampling(3840, 2160, 60, 1000, CFG, lim)
        assert (plan.width, plan.height) == (720, 405)

    def test_portrait_long_side(self):
        lim = VisionLimits(max_long_side_px=720)
        plan = plan_sampling(1080, 1920, 60, 1000, CFG, lim)
        assert (plan.width, plan.height) == (405, 720)

    def test_never_upscales(self):
        lim = VisionLimits(max_long_side_px=720)
        plan = plan_sampling(320, 180, 60, 1000, CFG, lim)
        assert (plan.width, plan.height) == (320, 180)
        assert plan.scale == 1.0

    def test_no_limits_keeps_source_size(self):
        plan = plan_sampling(1920, 1080, 60, 1000, CFG, VisionLimits())
        assert (plan.width, plan.height) == (1920, 1080)

    def test_megapixel_cap(self):
        lim = VisionLimits(max_megapixels=0.5)
        plan = plan_sampling(1920, 1080, 60, 1000, CFG, lim)
        assert plan.width * plan.height <= 0.5e6 * 1.005
        assert plan.width * plan.height > 0.45e6
        assert abs(plan.width / plan.height - 1920 / 1080) < 0.01

    def test_tightest_cap_wins(self):
        lim = VisionLimits(max_long_side_px=1000, max_megapixels=0.1)
        assert pixel_cap_scale(1920, 1080, lim) == pytest.approx(math.sqrt(1e5 / (1920 * 1080)))

    def test_max_tiles_per_frame(self):
        lim = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_frame=4)
        plan = plan_sampling(1920, 1080, 60, 1000, CFG, lim)
        assert plan.tiles_per_frame is not None and plan.tiles_per_frame <= 4
        # ... and it is the *largest* such size, not an arbitrary shrink.
        bigger = scaled_size(1920, 1080, plan.scale * 1.02)
        assert tiles_for(*bigger, lim) > 4

    def test_scaled_size_never_zero(self):
        assert scaled_size(10, 10, 0.0001) == (1, 1)


# ---------------------------------------------------------------- balance modes


class TestBalance:
    """1920x1080, 448-px tiles, 24-tile call budget, 25 s range: n_lo = 4 (min_frames),
    n_hi = 8 (max_frames; 8 frames fit at 336 px long side, 1 tile each)."""

    LIM = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_call=24)

    def _plan(self, balance, motion=None, **cfg):
        c = SamplingConfig(balance=balance, **cfg)
        return plan_sampling(1920, 1080, 25, 1000, c, self.LIM, motion)

    def test_endpoints_of_the_range_are_as_documented(self):
        plan = self._plan("spatial")
        assert (plan.n_lo, plan.n_hi) == (4, 8)

    def test_spatial_keeps_the_plain_count_and_maximises_size(self):
        plan = self._plan("spatial")
        assert plan.target_frames == 4
        assert plan.temporal_pref == 0.0
        assert plan.target_frames * plan.tiles_per_frame <= 24
        bigger = scaled_size(1920, 1080, plan.scale * 1.02)
        assert 4 * tiles_for(*bigger, self.LIM) > 24

    def test_temporal_takes_the_most_frames(self):
        plan = self._plan("temporal")
        assert plan.target_frames == 8
        assert plan.temporal_pref == 1.0
        assert plan.target_frames * plan.tiles_per_frame <= 24
        # 24 // 8 = 3 tiles per frame, but a 16:9 frame only fits as 2x1 (797x448) or 2x2
        assert plan.tiles_per_frame == 2

    def test_temporal_frames_are_smaller_than_spatial_ones(self):
        assert self._plan("temporal").width < self._plan("spatial").width

    def test_fixed_keeps_count_and_only_shrinks_resolution(self):
        plan = self._plan("fixed", max_frames=8)
        assert plan.target_frames == 4
        assert plan.temporal_pref is None
        assert plan.target_frames * plan.tiles_per_frame <= 24

    def test_auto_sits_between_the_extremes(self):
        plan = self._plan("auto", motion=1.0)
        assert 4 < plan.target_frames < 8
        assert 0 < plan.temporal_pref < 1

    def test_auto_static_short_prefers_space(self):
        plan = self._plan("auto", motion=0.0)
        assert plan.target_frames == 5  # p = 25/55/2 = 0.23 -> round(4 + 0.9)
        assert plan.target_frames < self._plan("auto", motion=1.0).target_frames

    def test_auto_long_range_prefers_time(self):
        p_short = temporal_preference("auto", 10, 0.5)
        p_long = temporal_preference("auto", 600, 0.5)
        assert p_long > p_short
        assert temporal_preference("auto", 0, 0.0) == 0.0
        assert temporal_preference("auto", 1e9, 1.0) == pytest.approx(1.0)

    def test_auto_without_motion_probe_is_neutral(self):
        assert temporal_preference("auto", 0, None) == 0.25

    def test_budget_never_exceeded_in_any_mode(self):
        for mode in ("auto", "temporal", "spatial", "fixed"):
            for motion in (0.0, 0.5, 1.0):
                plan = self._plan(mode, motion)
                assert plan.target_frames * plan.tiles_per_frame <= 24, (mode, motion)

    def test_budget_ignored_without_tile_size_with_note(self):
        lim = VisionLimits(max_tiles_per_call=10)
        plan = plan_sampling(1920, 1080, 25, 1000, CFG, lim)
        assert plan.target_frames == 4
        assert any("ignored" in n for n in plan.notes)

    def test_big_budget_changes_nothing(self):
        lim = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_call=10_000)
        plan = plan_sampling(1920, 1080, 25, 1000, SamplingConfig(balance="spatial"), lim)
        assert plan.target_frames == 4
        assert (plan.width, plan.height) == (1920, 1080)

    def test_tokens_per_tile(self):
        lim = VisionLimits(tile_width=448, tile_height=448, tokens_per_tile=256)
        plan = plan_sampling(896, 448, 25, 1000, CFG, lim)
        assert plan.tiles_per_frame == 2
        assert plan.est_total_tokens(lim, 4) == 4 * 2 * 256


class TestPriorities:
    def test_max_frames_per_call_is_a_hard_limit_over_min_frames(self):
        lim = VisionLimits(max_frames_per_call=3)
        plan = plan_sampling(1920, 1080, 100, 1000, CFG, lim)
        assert plan.target_frames == 3

    def test_max_frames_per_call_caps_temporal_growth(self):
        lim = VisionLimits(
            tile_width=448, tile_height=448, max_tiles_per_call=100, max_frames_per_call=5
        )
        plan = plan_sampling(1920, 1080, 25, 1000, SamplingConfig(balance="temporal"), lim)
        assert plan.target_frames == 5

    def test_min_frames_outranks_min_long_side(self):
        # Tiny 100-px tiles: a 336-px-long-side frame costs 4*2=8 tiles; budget 20 fits only
        # 2 of them, but min_frames=4 wins, so frames shrink below 336 px.
        lim = VisionLimits(tile_width=100, tile_height=100, max_tiles_per_call=20)
        plan = plan_sampling(1920, 1080, 100, 1000, SamplingConfig(), lim)
        assert plan.target_frames == 4
        assert ABSOLUTE_MIN_LONG_SIDE_PX <= max(plan.width, plan.height) < 336
        assert plan.target_frames * plan.tiles_per_frame <= 20

    def test_floor_64px_then_frames_are_dropped(self):
        # 64x36 costs 2x2=4 tiles of 32 px. Budget 8 -> at most 2 frames, even though
        # min_frames is 4: below 64 px frames go instead.
        lim = VisionLimits(tile_width=32, tile_height=32, max_tiles_per_call=8)
        plan = plan_sampling(1920, 1080, 100, 1000, SamplingConfig(), lim)
        assert plan.target_frames == 2
        assert max(plan.width, plan.height) == ABSOLUTE_MIN_LONG_SIDE_PX
        assert plan.target_frames * plan.tiles_per_frame <= 8
        assert plan.notes

    def test_unsatisfiable_budget_degrades_to_one_floor_frame(self):
        lim = VisionLimits(tile_width=8, tile_height=8, max_tiles_per_call=2)
        plan = plan_sampling(1920, 1080, 100, 1000, SamplingConfig(), lim)
        assert plan.target_frames == 1
        assert max(plan.width, plan.height) == ABSOLUTE_MIN_LONG_SIDE_PX
        assert any("unreachable" in n for n in plan.notes)

    def test_real_frames_cap_plan(self):
        plan = plan_sampling(640, 360, 100, 3, CFG, VisionLimits())
        assert plan.target_frames == 3

    def test_source_smaller_than_min_long_side_is_not_upscaled(self):
        lim = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_call=24)
        plan = plan_sampling(200, 100, 100, 1000, SamplingConfig(balance="temporal"), lim)
        assert (plan.width, plan.height) == (200, 100)


def test_fit_scale_returns_none_when_floor_does_not_fit():
    lim = VisionLimits(tile_width=8, tile_height=8)
    assert fit_scale(1920, 1080, lim, 1.0, 1) is None


def test_fit_scale_without_budget_is_max_scale():
    assert fit_scale(1920, 1080, VisionLimits(tile_width=8, tile_height=8), 0.5, None) == 0.5


# ---------------------------------------------------------------- config validation


class TestSamplingConfigValidation:
    def test_defaults(self):
        c = SamplingConfig.from_mapping(None)
        assert (c.min_frames, c.max_frames, c.interval_sec) == (4, 8, 10.0)
        assert c.dedupe_consecutive and c.dedupe_threshold == 0.002
        assert c.balance == "auto" and c.min_long_side_px == 336

    def test_overrides_beat_config(self):
        c = SamplingConfig.from_mapping(
            {"max_frames": 8, "interval_sec": 10}, max_frames=6, interval_sec=2.5
        )
        assert (c.max_frames, c.interval_sec) == (6, 2.5)

    def test_max_override_below_min_lowers_min(self):
        assert SamplingConfig.from_mapping({}, max_frames=2).min_frames == 2

    def test_explicit_conflicting_min_and_max_is_an_error(self):
        with pytest.raises(ValueError, match="max_frames .* smaller than min_frames"):
            SamplingConfig.from_mapping({}, min_frames=6, max_frames=3)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("min_frames", 0, "min_frames"),
            ("min_frames", 2.5, "min_frames"),
            ("max_frames", -1, "max_frames"),
            ("interval_sec", 0, "interval_sec"),
            ("interval_sec", "ten", "interval_sec"),
            ("dedupe_consecutive", "yes", "dedupe_consecutive"),
            ("dedupe_threshold", 1.5, "dedupe_threshold"),
            ("dedupe_threshold", -0.1, "dedupe_threshold"),
            ("balance", "sideways", "balance"),
            ("min_long_side_px", 10, "min_long_side_px"),
            ("min_long_side_px", True, "min_long_side_px"),
        ],
    )
    def test_invalid_values_name_the_key(self, key, value, match):
        with pytest.raises(ValueError, match=f"analysis.vlm.sampling.*{match}|{match}"):
            SamplingConfig.from_mapping({key: value})

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown key.*max_sample_frames"):
            SamplingConfig.from_mapping({"max_sample_frames": 4})

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="expected a mapping"):
            SamplingConfig.from_mapping([1, 2])  # type: ignore[arg-type]


class TestResolveVision:
    def test_defaults(self):
        v = resolve_vision(None, "analysis.endpoints.x")
        assert v["jpeg_quality"] == 90 and v["tokens_per_tile"] == 1
        assert v["max_long_side_px"] is None and v["max_tiles_per_call"] is None
        lim = VisionLimits.from_mapping(v)
        assert lim.jpeg_quality == 90 and not lim.has_tiles

    def test_full_example(self):
        v = resolve_vision(
            {
                "max_long_side_px": 720,
                "max_megapixels": 0.5,
                "tile_width": 448,
                "tile_height": 448,
                "tokens_per_tile": 0.5,
                "max_tiles_per_frame": 6,
                "max_tiles_per_call": 24,
                "max_frames_per_call": 8,
                "jpeg_quality": 75,
            },
            "analysis.endpoints.x",
        )
        assert VisionLimits.from_mapping(v).has_tiles
        assert v["tokens_per_tile"] == 0.5

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("max_long_side_px", 0),
            ("max_long_side_px", 720.5),
            ("max_long_side_px", "720"),
            ("max_megapixels", -1),
            ("max_megapixels", True),
            ("tile_width", -4),
            ("tokens_per_tile", 0),
            ("max_frames_per_call", 0),
            ("jpeg_quality", 0),
            ("jpeg_quality", 101),
            ("jpeg_quality", 80.5),
        ],
    )
    def test_invalid_values_name_endpoint_and_key(self, key, value):
        with pytest.raises(ValueError, match=rf"analysis\.endpoints\.mine\.vision\.{key}"):
            resolve_vision({key: value}, "analysis.endpoints.mine")

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="analysis.endpoints.mine.vision.*max_width"):
            resolve_vision({"max_width": 3}, "analysis.endpoints.mine")

    def test_tile_size_needs_both_axes(self):
        with pytest.raises(ValueError, match="tile_width and tile_height must be set together"):
            resolve_vision({"tile_width": 448}, "analysis.endpoints.mine")

    @pytest.mark.parametrize("key", ["max_tiles_per_frame", "max_tiles_per_call"])
    def test_tile_budgets_need_a_tile_size(self, key):
        with pytest.raises(ValueError, match=f"{key} needs tile_width"):
            resolve_vision({key: 4}, "analysis.endpoints.mine")

    def test_resolve_endpoint_surfaces_it(self):
        from autovideofixer.endpoints import resolve_endpoint

        cfg = {"analysis": {"endpoints": {"e": {"vision": {"jpeg_quality": 0}}}}}
        with pytest.raises(ValueError, match="analysis.endpoints.e.vision.jpeg_quality"):
            resolve_endpoint(cfg, "e")
        cfg = {"analysis": {"endpoints": {"e": {"vision": {"max_long_side_px": 720}}}}}
        assert resolve_endpoint(cfg, "e").vision["max_long_side_px"] == 720


# ---------------------------------------------------------------- § 25.2 selection


class FakeSource:
    """A video of ``len(labels)`` frames at ``fps``. Frame i looks like ``labels[i]``
    (constant thumbnail of brightness ``labels[i]``); ``read(t)`` returns the frame whose
    interval contains t, keyed by index, like the cv2 source does with timestamps."""

    def __init__(self, labels, fps=10.0):
        self.labels = list(labels)
        self.fps = fps
        self.reads = 0

    def read(self, t):
        self.reads += 1
        i = min(max(int(t * self.fps), 0), len(self.labels) - 1)
        thumb = np.full((4, 8), int(self.labels[i]) * 40, dtype=np.uint8)
        return Candidate(key=i, ts=i / self.fps, thumb=thumb, image=f"frame{i}")

    @property
    def duration(self):
        return len(self.labels) / self.fps


def _run(labels, target, fps=10.0, **kw):
    src = FakeSource(labels, fps)
    kw.setdefault("min_gap", 0.5 / fps)
    sel = select_frames(src, 0.0, src.duration, target, **kw)
    return src, sel


def _labels_of(sel, src):
    return [src.labels[c.key] for c in sel.kept]


class TestSelectFrames:
    def test_all_distinct_needs_no_refill(self):
        # 4 equal segments, 4 different labels
        src, sel = _run([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25, 4)
        assert _labels_of(sel, src) == [0, 1, 2, 3]
        assert sel.dropped_duplicates == 0 and sel.attempts == 4

    def test_a_a_a_b_drops_repeats_and_stops_below_target(self):
        # centres land on A, A, A, B. Only two distinct pictures exist.
        src, sel = _run([0] * 75 + [1] * 25, 4)
        assert _labels_of(sel, src) == [0, 1]
        assert sel.dropped_duplicates >= 2
        assert len(sel.kept) < sel.target

    def test_a_b_a_is_allowed(self):
        src, sel = _run([0] * 34 + [1] * 33 + [0] * 33, 3)
        assert _labels_of(sel, src) == [0, 1, 0]
        assert sel.dropped_duplicates == 0

    def test_refill_finds_the_frame_the_centres_missed(self):
        # B hides between the centres at t=3.75 (A) and 6.25: refill must probe the widest
        # gap and find it.
        labels = [0] * 50 + [1] * 10 + [2] * 10 + [3] * 30
        src, sel = _run(labels, 4)
        assert sorted(_labels_of(sel, src)) == [0, 1, 2, 3]
        assert sel.dropped_duplicates >= 1
        # kept frames are in time order
        assert [c.ts for c in sel.kept] == sorted(c.ts for c in sel.kept)

    def test_refill_rechecks_both_neighbours(self):
        # A | B | A: a refilled midpoint that equals the *right* neighbour must be dropped even
        # though it differs from the left one.
        labels = [0] * 40 + [1] * 20 + [0] * 40
        src, sel = _run(labels, 4)
        ls = _labels_of(sel, src)
        assert all(a != b for a, b in zip(ls, ls[1:], strict=False)), ls

    def test_static_range_ends_below_min_frames(self):
        src, sel = _run([0] * 100, 4)
        assert len(sel.kept) == 1
        assert sel.dropped_duplicates == sel.attempts - 1

    def test_attempt_cap_is_four_times_target(self):
        # Long static range at a high frame rate: gaps never run out, only the cap stops it.
        src, sel = _run([0] * 10_000, 4, fps=1000.0)
        assert sel.attempts == 16 == src.reads
        src, sel = _run([0] * 10_000, 2, fps=1000.0)
        assert sel.attempts == 8 == src.reads

    def test_frame_granularity_exhaustion_stops_early(self):
        # 3 frames, all identical: only 3 positions are meaningfully distinct in time.
        src, sel = _run([0, 0, 0], 4, fps=1.0)
        assert sel.exhausted
        assert sel.attempts < 16

    def test_never_returns_the_same_decoded_frame_twice(self):
        # 3 real frames, target 6: each position resolves onto one of them.
        src, sel = _run([0, 1, 2], 6, fps=1.0)
        keys = [c.key for c in sel.kept]
        assert len(keys) == len(set(keys)) <= 3
        assert sorted(_labels_of(sel, src)) == [0, 1, 2]

    def test_revisiting_a_frame_is_not_a_duplicate(self):
        # 3 distinct frames, target 6: refill positions land on frames already kept. Those
        # are sampler revisits, not repeated frames in the video, so the report says 0.
        src, sel = _run([0, 1, 2], 6, fps=1.0)
        assert src.reads > 3
        assert sel.dropped_duplicates == 0

    def test_rejected_frames_release_their_image(self):
        # Sources may cache candidates (FFmpegFrameSource does); a rejected full-size frame
        # must not stay alive in that cache.
        made = []

        class Caching(FakeSource):
            def read(self, t):
                cand = super().read(t)
                made.append(cand)
                return cand

        src = Caching([0] * 75 + [1] * 25)
        sel = select_frames(src, 0.0, src.duration, 4, min_gap=0.05)
        kept = {id(c) for c in sel.kept}
        rejected = [c for c in made if id(c) not in kept]
        assert rejected and all(c.image is None for c in rejected)
        assert all(c.image is not None for c in sel.kept)

    def test_same_frame_dedupe_holds_even_with_dedupe_off(self):
        src, sel = _run([0, 1, 2], 6, fps=1.0, dedupe=False)
        keys = [c.key for c in sel.kept]
        assert len(keys) == len(set(keys)) == 3

    def test_dedupe_off_keeps_visually_identical_distinct_frames(self):
        src, sel = _run([0] * 100, 4, dedupe=False)
        assert len(sel.kept) == 4 and sel.dropped_duplicates == 0

    def test_threshold_is_inclusive_and_configurable(self):
        # labels 0 and 1 differ by 40/255 = 0.157 in thumb_diff
        labels = [0] * 50 + [1] * 50
        _, strict = _run(labels, 2, threshold=0.002)
        _, loose = _run(labels, 2, threshold=0.2)
        assert len(strict.kept) == 2
        assert len(loose.kept) == 1

    def test_unreadable_frames_are_skipped(self):
        class Flaky(FakeSource):
            def read(self, t):
                return None if t < 2.0 else super().read(t)

        src = Flaky([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
        sel = select_frames(src, 0.0, src.duration, 4, min_gap=0.05)
        assert len(sel.kept) >= 2


def test_thumb_diff_is_normalised_mean_abs_luma():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 255, dtype=np.uint8)
    assert thumb_diff(a, a) == 0.0
    assert thumb_diff(a, b) == 1.0
    c = a.copy()
    c[0, :] = 255  # a quarter of the pixels fully different
    assert thumb_diff(a, c) == pytest.approx(0.25)
    assert thumb_diff(a, np.zeros((2, 2), dtype=np.uint8)) == 1.0


def test_motion_score_scales_and_saturates():
    class Src:
        def __init__(self, step):
            self.step = step

        def read(self, t):
            v = int(t * self.step) % 256
            return Candidate(key=t, ts=t, thumb=np.full((4, 4), v, dtype=np.uint8))

    assert motion_score(Src(0.0), 0.0, 8.0) == 0.0
    low = motion_score(Src(1.0), 0.0, 8.0)
    high = motion_score(Src(10.0), 0.0, 8.0)
    assert 0 < low < high <= 1.0
    assert motion_score(Src(100.0), 0.0, 8.0) == 1.0

    class Dead:
        def read(self, t):
            return None

    assert motion_score(Dead(), 0.0, 8.0) is None


# ---------------------------------------------------------------- reporting


class TestReport:
    def test_summary_with_and_without_tiles(self):
        r = SamplingReport(frames=6, dropped_duplicates=2, width=720, height=405, target_frames=8)
        assert r.frame_size == "720x405"
        assert "6 frame(s) sent" in r.summary() and "2 duplicate(s)" in r.summary()
        assert "token" not in r.summary()
        r.est_tiles, r.est_vision_tokens = 12, 24
        assert "~12 tile(s) / ~24 vision token(s)" in r.summary()

    def test_to_dict_has_the_reported_keys(self):
        d = SamplingReport(frames=3, width=640, height=360).to_dict()
        for key in ("frames", "dropped_duplicates", "frame_size", "est_tiles", "est_vision_tokens"):
            assert key in d
        assert d["frame_size"] == "640x360"
        assert d["est_vision_tokens"] is None

    def test_empty_report_has_no_size(self):
        assert SamplingReport().frame_size == ""
