"""Tests for per-gap adaptive AI interpolation (REQUIREMENTS.md § 16).

Covers three layers, from purest to most integrated:

1. ``resample_plan()`` (``ai/wrappers/interpolate.py``) -- the pure,
   side-effect-free timeline-resampling core. No GPU, no torch inference,
   no video file: exercised exhaustively with plain float lists.
2. ``execute_resample_plan()`` -- the streaming executor, exercised with
   fake in-memory "frames" (plain ints/strings) and a fake
   ``interpolate_fn``, so it never touches RIFE either.
3. ``_plan_ai_interpolation()``'s new ``target_approach`` parameter
   (``core/stages/interpolate.py``) -- worked examples from § 12/§ 16.
4. ``InterpolateStage._get_adaptive_timeline()`` -- the fail-open probe/
   validation wrapper, with ``probe()``/``probe_frame_timestamps()`` mocked.

Everything GPU/model/ffmpeg-related is mocked throughout; nothing here
requires torch, a real video file, or a GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autovideofixer.ai.wrappers.interpolate import (
    PlanEntry,
    execute_resample_plan,
    plan_stats,
    resample_plan,
)
from autovideofixer.config import Config
from autovideofixer.core.stages.interpolate import InterpolateStage, _plan_ai_interpolation

# ---------------------------------------------------------------------------
# resample_plan() -- pure core
# ---------------------------------------------------------------------------


class TestResamplePlanUniform:
    def test_24_to_60_grid_count_exact(self):
        n = 240  # 10s @ 24fps
        timestamps = [i / 24.0 for i in range(n)]
        plan = resample_plan(timestamps, 60.0)
        duration = timestamps[-1] - timestamps[0]
        expected_count = int(math.floor(duration * 60.0)) + 1
        assert len(plan) == expected_count

    def test_uniform_input_never_needs_gap_fallback(self):
        """A perfectly uniform 24fps timeline never exceeds the default cap
        (8) -- there should be zero "hold"/"blend" entries."""
        timestamps = [i / 24.0 for i in range(240)]
        plan = resample_plan(timestamps, 60.0)
        assert all(e.op in ("emit", "synthesize") for e in plan)

    def test_drift_free_over_long_sequence(self):
        """10k-frame uniform sequence: output count must equal the grid
        count EXACTLY, proving no accumulated per-gap rounding error."""
        n = 10_000
        timestamps = [i / 24.0 for i in range(n)]
        plan = resample_plan(timestamps, 60.0)
        duration = timestamps[-1] - timestamps[0]
        expected_count = int(math.floor(duration * 60.0)) + 1
        assert len(plan) == expected_count
        # Last entry must reference the final bracket, never overshoot it.
        assert plan[-1].k <= n - 1

    def test_timesteps_match_expected_local_positions(self):
        """24fps -> 60fps: each source gap (1/24s) contains grid points at
        known fractional offsets; spot check a few synthesize timesteps."""
        timestamps = [0.0, 1 / 24, 2 / 24]
        plan = resample_plan(timestamps, 60.0, eps=1e-6)
        synths = [e for e in plan if e.op == "synthesize"]
        for e in synths:
            assert 0.0 < e.local < 1.0


class TestResamplePlanIrregular:
    """The phone-60->YouTube case: long gaps get MORE synthesized frames
    than short gaps -- this is the entire point of the feature."""

    def test_long_gap_gets_more_synthesized_frames_than_short_gap(self):
        # A short gap (1/60s) followed by a long "stall" gap (4/60s, as if
        # a camera stall held one frame 4x), both targeting 60fps.
        timestamps = [0.0, 1 / 60, 1 / 60 + 4 / 60]
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=100)
        short_gap_synth = [e for e in plan if e.op == "synthesize" and e.k == 0]
        long_gap_synth = [e for e in plan if e.op == "synthesize" and e.k == 1]
        assert len(long_gap_synth) > len(short_gap_synth)

    def test_irregular_cadence_reaches_grid_count_exactly(self):
        """Mixed short/long gaps still hit the grid count exactly (no
        special-casing breaks the drift-free guarantee)."""
        gaps = [0.05, 0.033, 0.05, 0.2, 0.033, 0.05] * 50
        timestamps = [0.0]
        for g in gaps:
            timestamps.append(timestamps[-1] + g)
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=100)
        duration = timestamps[-1] - timestamps[0]
        expected_count = int(math.floor(duration * 60.0)) + 1
        assert len(plan) == expected_count


class TestResamplePlanSnapping:
    def test_grid_landing_on_real_frame_emits_original(self):
        """target_fps == source fps (grid lands exactly on every input
        frame): every entry must be "emit", never "synthesize"."""
        timestamps = [i / 30.0 for i in range(10)]
        plan = resample_plan(timestamps, 30.0, eps=1e-6)
        assert all(e.op == "emit" for e in plan)
        # And the referenced original frame indices must be 0..9 in order.
        assert [e.k for e in plan] == list(range(10))

    def test_near_landing_within_eps_snaps_to_original(self):
        # Grid point at 1/60 lands a hair off frame index 1 of a 30fps
        # source (which sits at 1/30 = 2/60) -- use a generous eps so a
        # near-miss still snaps instead of synthesizing.
        timestamps = [0.0, 1.0 / 30.0 + 1e-4, 2.0 / 30.0]
        plan = resample_plan(timestamps, 30.0, eps=1e-2)
        assert all(e.op == "emit" for e in plan)


class TestMaxIntermediatesPerGapCap:
    def test_cap_triggers_hold_fallback(self):
        # One huge gap (1 second) targeting 60fps needs ~59 intermediates,
        # way past the default cap of 8.
        timestamps = [0.0, 1.0]
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=8, gap_fallback="hold")
        assert not any(e.op == "synthesize" for e in plan)
        assert any(e.op == "hold" for e in plan)

    def test_cap_triggers_blend_fallback(self):
        timestamps = [0.0, 1.0]
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=8, gap_fallback="blend")
        assert not any(e.op == "synthesize" for e in plan)
        assert any(e.op == "blend" for e in plan)

    def test_below_cap_still_synthesizes(self):
        # A short gap needing only ~3 intermediates stays under the
        # default cap of 8 and must synthesize normally.
        timestamps = [0.0, 4 / 60]
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=8)
        assert any(e.op == "synthesize" for e in plan)
        assert not any(e.op in ("hold", "blend") for e in plan)

    def test_invalid_gap_fallback_raises(self):
        with pytest.raises(ValueError):
            resample_plan([0.0, 1.0], 60.0, gap_fallback="nonsense")

    def test_plan_stats_counts_capped_gaps_as_distinct(self):
        # Two independent stalled gaps -- gaps_capped must count DISTINCT
        # brackets, not total capped entries.
        timestamps = [0.0, 1.0, 1.0 + 1 / 24, 2.0 + 1 / 24]
        plan = resample_plan(timestamps, 60.0, max_intermediates_per_gap=8)
        stats = plan_stats(plan)
        assert stats["gaps_capped"] == 2


class TestTargetFpsEdgeCasesAndDegenerateInputs:
    def test_target_fps_below_source_no_crash(self):
        timestamps = [i / 60.0 for i in range(120)]
        plan = resample_plan(timestamps, 24.0)
        assert len(plan) > 0
        # Downsampling: not every source frame need appear.
        assert len(plan) < len(timestamps)

    def test_target_fps_equal_to_source(self):
        timestamps = [i / 30.0 for i in range(30)]
        plan = resample_plan(timestamps, 30.0)
        assert len(plan) > 0

    def test_empty_timeline(self):
        assert resample_plan([], 60.0) == []

    def test_single_frame_timeline(self):
        plan = resample_plan([1.23], 60.0)
        assert plan == [PlanEntry(op="emit", k=0)]

    def test_zero_target_fps_no_crash(self):
        plan = resample_plan([0.0, 1 / 24, 2 / 24], 0.0)
        assert len(plan) == 3
        assert all(e.op == "emit" for e in plan)

    def test_negative_target_fps_no_crash(self):
        plan = resample_plan([0.0, 1 / 24], -5.0)
        assert len(plan) == 2

    def test_duplicate_timestamps_no_division_by_zero(self):
        timestamps = [0.0, 0.0, 0.0, 1 / 24, 2 / 24]
        plan = resample_plan(timestamps, 60.0)
        assert len(plan) > 0

    def test_all_duplicate_timestamps_degenerate(self):
        timestamps = [1.0, 1.0, 1.0]
        plan = resample_plan(timestamps, 60.0)
        assert plan == [PlanEntry(op="emit", k=i) for i in range(3)]

    def test_non_monotonic_timestamps_no_crash(self):
        timestamps = [0.0, 0.05, 0.02, 0.09, 0.20]
        plan = resample_plan(timestamps, 60.0)
        assert len(plan) > 0
        for e in plan:
            assert 0.0 <= e.local <= 1.0

    def test_zero_length_individual_gap_amid_normal_gaps(self):
        timestamps = [0.0, 1 / 24, 1 / 24, 2 / 24, 3 / 24]
        plan = resample_plan(timestamps, 60.0)
        assert len(plan) > 0


# ---------------------------------------------------------------------------
# execute_resample_plan() -- streaming executor
# ---------------------------------------------------------------------------


class TestExecuteResamplePlan:
    def test_emit_entries_reproduce_original_frames(self):
        frames = ["f0", "f1", "f2"]
        plan = [PlanEntry(op="emit", k=0), PlanEntry(op="emit", k=1), PlanEntry(op="emit", k=2)]
        out = list(execute_resample_plan(frames, plan, lambda a, b, t: f"interp({a},{b},{t})"))
        assert out == ["f0", "f1", "f2"]

    def test_synthesize_calls_interpolate_fn_with_correct_pair_and_timestep(self):
        frames = ["f0", "f1"]
        plan = [PlanEntry(op="synthesize", k=0, local=0.25)]
        calls = []

        def fake_interp(a, b, t):
            calls.append((a, b, t))
            return f"mid({a},{b},{t})"

        out = list(execute_resample_plan(frames, plan, fake_interp))
        assert out == ["mid(f0,f1,0.25)"]
        assert calls == [("f0", "f1", 0.25)]

    def test_hold_repeats_frame_k_without_calling_interpolate_fn(self):
        frames = ["f0", "f1"]
        plan = [PlanEntry(op="hold", k=0)]
        calls = []
        out = list(execute_resample_plan(frames, plan, lambda a, b, t: calls.append(1) or "X"))
        assert out == ["f0"]
        assert calls == []

    def test_blend_crossfades_without_calling_interpolate_fn(self):
        import numpy as np

        frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
        frame_b = np.full((2, 2, 3), 200, dtype=np.uint8)
        plan = [PlanEntry(op="blend", k=0, local=0.5)]
        calls = []
        out = list(
            execute_resample_plan([frame_a, frame_b], plan, lambda a, b, t: calls.append(1) or a)
        )
        assert calls == []
        assert out[0].shape == (2, 2, 3)
        assert 90 <= int(out[0][0, 0, 0]) <= 110  # ~halfway between 0 and 200

    def test_full_plan_end_to_end_ordering(self):
        timestamps = [0.0, 1 / 24, 2 / 24]
        plan = resample_plan(timestamps, 60.0, eps=1e-6)
        frames = ["f0", "f1", "f2"]

        def fake_interp(a, b, t):
            return f"({a}->{b}@{t:.3f})"

        out = list(execute_resample_plan(frames, plan, fake_interp))
        assert len(out) == len(plan)
        # First output must be the very first original frame.
        assert out[0] == "f0"

    def test_holds_at_most_two_source_frames_via_lazy_pull(self):
        """The executor must pull from frame_source lazily -- it should not
        need to exhaust the source before producing output, proving it
        isn't buffering the whole stream up front."""
        pulled = []

        def frame_gen():
            for i in range(1000):
                pulled.append(i)
                yield f"f{i}"

        plan = [PlanEntry(op="emit", k=0)]
        out = list(execute_resample_plan(frame_gen(), plan, lambda a, b, t: a))
        assert out == ["f0"]
        # Only the frames needed to prime the 2-frame window were pulled,
        # not the whole 1000-frame source.
        assert len(pulled) <= 2

    def test_source_exhausted_early_degrades_instead_of_crashing(self):
        """A plan referencing more frames than the source actually yields
        (a stale/mismatched timeline) must not raise -- see
        REQUIREMENTS.md § 16.2's fail-open contract."""
        frames = ["f0"]
        plan = [PlanEntry(op="emit", k=0), PlanEntry(op="synthesize", k=0, local=0.5)]
        out = list(execute_resample_plan(frames, plan, lambda a, b, t: "SHOULD_NOT_BE_CALLED"))
        assert out == ["f0", "f0"]

    def test_empty_source_yields_nothing(self):
        out = list(execute_resample_plan([], [PlanEntry(op="emit", k=0)], lambda a, b, t: a))
        assert out == []


# ---------------------------------------------------------------------------
# _plan_ai_interpolation()'s target_approach (§ 16.4)
# ---------------------------------------------------------------------------


class TestTargetApproachUnderUnchanged:
    """Default target_approach="under" must be bit-for-bit identical to the
    pre-§16.4 behavior -- these mirror test_interpolate_hybrid.py's cases."""

    def test_24_to_60_default_matches_explicit_under(self):
        default_plan = _plan_ai_interpolation(24.0, 60.0, hybrid_enabled=True)
        under_plan = _plan_ai_interpolation(
            24.0, 60.0, hybrid_enabled=True, target_approach="under"
        )
        assert default_plan == under_plan
        assert under_plan.rife_factor == 2
        assert under_plan.run_minterpolate_finish is True

    def test_50_to_60_default_matches_explicit_under(self):
        default_plan = _plan_ai_interpolation(50.0, 60.0, hybrid_enabled=True)
        under_plan = _plan_ai_interpolation(
            50.0, 60.0, hybrid_enabled=True, target_approach="under"
        )
        assert default_plan == under_plan
        assert under_plan.run_rife is False


class TestTargetApproachOver:
    def test_24_to_60_over_picks_factor_3_then_finishes_down(self):
        plan = _plan_ai_interpolation(24.0, 60.0, hybrid_enabled=True, target_approach="over")
        assert plan.run_rife is True
        assert plan.rife_factor == 3
        assert plan.intermediate_fps == pytest.approx(72.0)
        assert plan.run_minterpolate_finish is True

    def test_50_to_60_over_picks_factor_2_then_finishes_down(self):
        plan = _plan_ai_interpolation(50.0, 60.0, hybrid_enabled=True, target_approach="over")
        assert plan.run_rife is True
        assert plan.rife_factor == 2
        assert plan.intermediate_fps == pytest.approx(100.0)
        assert plan.run_minterpolate_finish is True

    def test_30_to_120_over_exact_no_finish(self):
        plan = _plan_ai_interpolation(30.0, 120.0, hybrid_enabled=True, target_approach="over")
        assert plan.rife_factor == 4
        assert plan.run_minterpolate_finish is False

    def test_over_never_skips_rife(self):
        # ratio is always > 1 when target > current (should_run()'s gate),
        # so ceil() is always >= 2 -- "over" never delegates to
        # minterpolate-alone the way "under" does for e.g. 50->60.
        plan = _plan_ai_interpolation(24.0, 30.0, hybrid_enabled=True, target_approach="over")
        assert plan.run_rife is True
        assert plan.rife_factor == 2


class TestTargetApproachNearest:
    def test_50_to_60_nearest_matches_under_minterpolate_only(self):
        # floor=1 (50fps, no help), ceil=2 (100fps, error 40) -- floor is
        # nearer (error 10), and floor<2 means minterpolate alone.
        plan = _plan_ai_interpolation(50.0, 60.0, hybrid_enabled=True, target_approach="nearest")
        assert plan.run_rife is False
        assert plan.run_minterpolate_finish is True

    def test_30_to_120_nearest_exact(self):
        plan = _plan_ai_interpolation(30.0, 120.0, hybrid_enabled=True, target_approach="nearest")
        assert plan.rife_factor == 4
        assert plan.run_minterpolate_finish is False

    def test_nearest_picks_closer_of_floor_and_ceil(self):
        # current=24, target=58: floor=2 (48fps, err 10), ceil=3 (72fps, err
        # 14) -- floor (2) should win.
        plan = _plan_ai_interpolation(24.0, 58.0, hybrid_enabled=True, target_approach="nearest")
        assert plan.rife_factor == 2

    def test_nearest_prefers_ceil_when_strictly_closer(self):
        # current=24, target=70: floor=2 (48fps, err 22), ceil=3 (72fps,
        # err 2) -- ceil (3) should win.
        plan = _plan_ai_interpolation(24.0, 70.0, hybrid_enabled=True, target_approach="nearest")
        assert plan.rife_factor == 3


class TestTargetApproachHybridDisabledUnaffected:
    """target_approach must not change the legacy overshoot behavior when
    hybrid_ai_minterpolate is disabled."""

    @pytest.mark.parametrize("approach", ["under", "nearest", "over"])
    def test_50_to_60_hybrid_disabled(self, approach):
        plan = _plan_ai_interpolation(50.0, 60.0, hybrid_enabled=False, target_approach=approach)
        assert plan.run_rife is True
        assert plan.rife_factor == 2
        assert plan.run_minterpolate_finish is False
        assert plan.intermediate_fps == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# InterpolateStage._get_adaptive_timeline() -- fail-open probe/validation
# ---------------------------------------------------------------------------


class TestGetAdaptiveTimeline:
    def _stage(self, tmp_path) -> InterpolateStage:
        return InterpolateStage(Config(tmp_path / "nonexistent.yaml"))

    def test_matching_count_returns_timeline(self, tmp_path, monkeypatch):
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")
        Path(input_path).touch()

        probe_info = MagicMock(has_video=True, frame_count=3, duration=0.125)
        timestamps = [0.0, 1 / 24, 2 / 24]
        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
        monkeypatch.setattr(
            "autovideofixer.core.cadence.probe_frame_timestamps",
            lambda path, config: timestamps,
        )

        result = stage._get_adaptive_timeline(input_path)
        assert result is not None
        got_timestamps, got_probe_info = result
        assert got_timestamps == timestamps
        assert got_probe_info is probe_info

    def test_mismatched_frame_count_does_NOT_fall_back(self, tmp_path, monkeypatch):
        """A probe frame_count disagreeing with the timeline must be IGNORED.

        frame_count is derived from avg_frame_rate, which REQUIREMENTS.md
        § 12.3 documents as unreliable for VFR -- and VFR is the only input
        the adaptive path exists to serve. Measured on real files: a 48-frame
        retimed intermediate reports frame_count=119, and a 123-frame native
        VFR capture reports 180. Rejecting on that mismatch silently disabled
        the whole feature while every gate still passed. See also
        tests/unit/test_adaptive_timeline_gate.py.
        """
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")
        Path(input_path).touch()

        # A realistic VFR disagreement: 3 real frames, container claims 100.
        probe_info = MagicMock(has_video=True, frame_count=100, duration=0.125)
        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
        monkeypatch.setattr(
            "autovideofixer.core.cadence.probe_frame_timestamps",
            lambda path, config: [0.0, 1 / 24, 2 / 24],
        )

        assert stage._get_adaptive_timeline(input_path) is not None

    def test_probe_failure_falls_back_to_none(self, tmp_path, monkeypatch):
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")

        def raise_probe(path):
            raise RuntimeError("ffprobe exploded")

        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", raise_probe)
        assert stage._get_adaptive_timeline(input_path) is None

    def test_no_timestamps_parsed_falls_back_to_none(self, tmp_path, monkeypatch):
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")
        Path(input_path).touch()

        probe_info = MagicMock(has_video=True, frame_count=3)
        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
        monkeypatch.setattr(
            "autovideofixer.core.cadence.probe_frame_timestamps",
            lambda path, config: None,
        )
        assert stage._get_adaptive_timeline(input_path) is None

    def test_no_video_stream_falls_back_to_none(self, tmp_path, monkeypatch):
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")

        probe_info = MagicMock(has_video=False, frame_count=0)
        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
        assert stage._get_adaptive_timeline(input_path) is None

    def test_single_timestamp_falls_back_to_none(self, tmp_path, monkeypatch):
        stage = self._stage(tmp_path)
        input_path = str(tmp_path / "in.mp4")
        Path(input_path).touch()

        probe_info = MagicMock(has_video=True, frame_count=1)
        monkeypatch.setattr("autovideofixer.core.stages.interpolate.probe", lambda path: probe_info)
        monkeypatch.setattr(
            "autovideofixer.core.cadence.probe_frame_timestamps",
            lambda path, config: [0.0],
        )
        assert stage._get_adaptive_timeline(input_path) is None
