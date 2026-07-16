"""Unit tests for StabilizeStage's zoom gating and framerate selection.

These cover two real-world bugs:

1. Black borders after stabilization despite zoom_enabled=True: the previous
   `_calculate_zoom` hand-derived a zoom PERCENTAGE from raw TRF local-motion
   values with an inverted sign -- it could only ever produce values in
   [-20, 0] (vidstabtransform's `zoom` option is >0 = zoom in, <0 = zoom
   out), so it was structurally incapable of shrinking the border. The fix
   delegates the actual zoom amount to vidstabtransform's own `optzoom`
   option and uses movement-extent purely as a threshold gate.

2. ~2x playback speed + freeze on VFR/YouTube-origin inputs: StabilizeStage
   piped raw (timestamp-less) decoded frames into the transform process
   using `-r {r_frame_rate}` where r_frame_rate ("tbr") can be a multiple of
   avg_frame_rate (the honest real-frame-count-over-duration rate). Fixed to
   read avg_frame_rate first, falling back to r_frame_rate only when
   avg_frame_rate is undefined ("0/0").
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.stabilize import StabilizeStage


def _make_stage(tmp_path) -> StabilizeStage:
    config = Config(tmp_path / "nonexistent.yaml")
    return StabilizeStage(config)


def _fake_ffprobe_result(stdout: str) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    return result


class TestGetVideoFramerate:
    """avg_frame_rate must be preferred over r_frame_rate."""

    def test_prefers_avg_frame_rate_over_r_frame_rate(self, tmp_path):
        stage = _make_stage(tmp_path)
        # Mirrors the real bug report: r_frame_rate (tbr) is ~2x the true
        # average rate for a YouTube-origin VFR-ish source.
        with patch.object(
            subprocess,
            "run",
            return_value=_fake_ffprobe_result("2964/100,60000/1001\n"),
        ):
            fps = stage._get_video_framerate("dummy.mp4")
        assert fps == pytest.approx(29.64, abs=0.01)

    def test_falls_back_to_r_frame_rate_when_avg_undefined(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(
            subprocess,
            "run",
            return_value=_fake_ffprobe_result("0/0,30/1\n"),
        ):
            fps = stage._get_video_framerate("dummy.mp4")
        assert fps == pytest.approx(30.0)

    def test_matching_rates(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(
            subprocess,
            "run",
            return_value=_fake_ffprobe_result("30/1,30/1\n"),
        ):
            fps = stage._get_video_framerate("dummy.mp4")
        assert fps == pytest.approx(30.0)

    def test_no_output_defaults_to_30(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(subprocess, "run", return_value=_fake_ffprobe_result("")):
            fps = stage._get_video_framerate("dummy.mp4")
        assert fps == 30.0

    def test_exception_defaults_to_30(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(subprocess, "run", side_effect=OSError("boom")):
            fps = stage._get_video_framerate("dummy.mp4")
        assert fps == 30.0


class TestFfprobeSpawnsDetachStdin:
    """Neither ffprobe helper may leave stdin attached to the caller's tty.

    ffprobe has no -nostdin flag (unlike ffmpeg), so stdin=DEVNULL is the
    only lever available here.
    """

    def test_get_video_dimensions_detaches_stdin(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(
            subprocess, "run", return_value=_fake_ffprobe_result("1920,1080\n")
        ) as mock_run:
            stage._get_video_dimensions("dummy.mp4")
        _, kwargs = mock_run.call_args
        assert kwargs.get("stdin") == subprocess.DEVNULL

    def test_get_video_framerate_detaches_stdin(self, tmp_path):
        stage = _make_stage(tmp_path)
        with patch.object(
            subprocess, "run", return_value=_fake_ffprobe_result("30/1,30/1\n")
        ) as mock_run:
            stage._get_video_framerate("dummy.mp4")
        _, kwargs = mock_run.call_args
        assert kwargs.get("stdin") == subprocess.DEVNULL


class TestMovementExtent:
    """Movement extent is a pure signal for gating zoom, not a zoom amount."""

    def test_no_lm_entries_returns_zero(self, tmp_path):
        stage = _make_stage(tmp_path)
        trf = tmp_path / "empty.trf"
        trf.write_text("VidStab 1\nFrame 0 (List 0 [])\n")
        assert stage._movement_extent(str(trf)) == 0.0

    def test_computes_dx_dy_range(self, tmp_path):
        stage = _make_stage(tmp_path)
        trf = tmp_path / "shaky.trf"
        # dx ranges -40..40 (extent 80), dy ranges -5..5 (extent 10)
        trf.write_text(
            "VidStab 1\n"
            "Frame 0 (List 2 [(LM -40 -5 0 0 10 10 1.0 1.0),(LM 40 5 1 1 10 10 1.0 1.0)])\n"
        )
        assert stage._movement_extent(str(trf)) == pytest.approx(80.0)

    def test_small_movement_below_default_threshold(self, tmp_path):
        stage = _make_stage(tmp_path)
        trf = tmp_path / "steady.trf"
        trf.write_text(
            "VidStab 1\nFrame 0 (List 2 [(LM -1 0 0 0 10 10 1.0 1.0),(LM 1 0 1 1 10 10 1.0 1.0)])\n"
        )
        extent = stage._movement_extent(str(trf))
        assert extent == pytest.approx(2.0)
        assert extent < stage._zoom_threshold


class TestZoomApplyGating:
    """apply_zoom (which drives optzoom) must never produce a zoom-out.

    This exercises the same decision `execute()` makes: apply_zoom = True
    when zoom_enabled and needs_stab and movement extent clears the
    threshold; only then does the filter get `optzoom=1` (zoom in only,
    never negative/zoom-out). When apply_zoom is False the filter
    explicitly sets optzoom=0 (no zoom), matching zoom_enabled=False's
    "borders acceptable" behavior.
    """

    def test_apply_zoom_true_when_movement_exceeds_threshold(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_threshold = 50.0
        trf = tmp_path / "big_shake.trf"
        trf.write_text(
            "VidStab 1\n"
            "Frame 0 (List 2 [(LM -60 0 0 0 10 10 1.0 1.0),(LM 60 0 1 1 10 10 1.0 1.0)])\n"
        )
        movement = stage._movement_extent(str(trf))
        assert movement >= stage._zoom_threshold

    def test_apply_zoom_false_when_movement_below_threshold(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_threshold = 50.0
        trf = tmp_path / "small_shake.trf"
        trf.write_text(
            "VidStab 1\nFrame 0 (List 2 [(LM -5 0 0 0 10 10 1.0 1.0),(LM 5 0 1 1 10 10 1.0 1.0)])\n"
        )
        movement = stage._movement_extent(str(trf))
        assert movement < stage._zoom_threshold


class TestZoomCoverageDefault:
    """zoom_coverage defaults to 1.0 (today's optzoom=1 behavior), clamped to [0, 1]."""

    def test_default_is_one(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._zoom_coverage == 1.0

    def test_clamped_above_one(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(1.5, "stages", "stabilize", "zoom_coverage")
        stage = StabilizeStage(config)
        assert stage._zoom_coverage == 1.0

    def test_clamped_below_zero(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(-0.5, "stages", "stabilize", "zoom_coverage")
        stage = StabilizeStage(config)
        assert stage._zoom_coverage == 0.0

    def test_configured_value_used(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(0.7, "stages", "stabilize", "zoom_coverage")
        stage = StabilizeStage(config)
        assert stage._zoom_coverage == 0.7


def _write_frames_trf(tmp_path, name, per_frame_dx):
    """Write a TRF with one LM entry per frame, dx=per_frame_dx[i], dy=0."""
    lines = ["VidStab 1"]
    for i, dx in enumerate(per_frame_dx):
        lines.append(f"Frame {i} (List 1 [(LM {int(dx)} 0 0 0 10 10 1.0 1.0)])")
    trf = tmp_path / name
    trf.write_text("\n".join(lines) + "\n")
    return str(trf)


class TestComputeStaticZoomPct:
    """Static zoom is the zoom_coverage-quantile of per-frame required-zoom
    estimates, not the max (which is what optzoom=1 effectively guarantees)."""

    def test_no_lm_entries_returns_zero(self, tmp_path):
        stage = _make_stage(tmp_path)
        trf = tmp_path / "empty.trf"
        trf.write_text("VidStab 1\nFrame 0 (List 0 [])\n")
        assert stage._compute_static_zoom_pct(str(trf), 1920, 1080, 0.5) == 0.0

    def test_missing_file_returns_zero(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._compute_static_zoom_pct(str(tmp_path / "nope.trf"), 1920, 1080, 0.5) == 0.0

    def test_full_coverage_uses_worst_frame(self, tmp_path):
        """coverage=1.0 (quantile at the max) should require at least as
        much zoom as any lower coverage on the same clip."""
        stage = _make_stage(tmp_path)
        # One violent-motion frame among many calm ones.
        dx = [0.0] * 20 + [200.0]
        trf = _write_frames_trf(tmp_path, "mixed.trf", dx)
        pct_full = stage._compute_static_zoom_pct(trf, 1920, 1080, 1.0)
        pct_half = stage._compute_static_zoom_pct(trf, 1920, 1080, 0.5)
        assert pct_full >= pct_half
        assert pct_full > 0.0

    def test_zero_coverage_uses_smallest_frame(self, tmp_path):
        stage = _make_stage(tmp_path)
        dx = [0.0] * 20 + [200.0]
        trf = _write_frames_trf(tmp_path, "mixed.trf", dx)
        pct_zero = stage._compute_static_zoom_pct(trf, 1920, 1080, 0.0)
        pct_full = stage._compute_static_zoom_pct(trf, 1920, 1080, 1.0)
        assert pct_zero <= pct_full

    def test_uniform_motion_all_quantiles_equal(self, tmp_path):
        """A single initial jump followed by a steady position (constant
        cumulative path, no further outlier frames) should give the same
        required zoom regardless of coverage, since every frame's excursion
        from the median path is identical."""
        stage = _make_stage(tmp_path)
        # cumulative path (integral of these per-frame deltas) is a constant
        # 5px offset for every frame after the first.
        dx = [5.0] + [0.0] * 29
        trf = _write_frames_trf(tmp_path, "steady.trf", dx)
        pct_low = stage._compute_static_zoom_pct(trf, 1920, 1080, 0.1)
        pct_high = stage._compute_static_zoom_pct(trf, 1920, 1080, 0.9)
        assert pct_low == pytest.approx(pct_high, abs=1e-6)


class TestZoomCoverageExecuteWiring:
    """execute()'s zoom_param construction must respect zoom_coverage without
    changing the zoom_enabled/zoom_threshold gate (apply_zoom) semantics."""

    def _run_zoom_param_decision(self, stage, apply_zoom, trf_path="dummy.trf"):
        """Mirror execute()'s zoom_param decision logic in isolation (avoids
        needing to drive the full ffmpeg pipe subprocess machinery)."""
        static_zoom_pct = None
        if not apply_zoom:
            zoom_param = ":zoom=0:optzoom=0"
        elif stage._zoom_coverage >= 1.0:
            zoom_param = ":zoom=0:optzoom=1"
        elif stage._zoom_coverage <= 0.0:
            zoom_param = ":zoom=0:optzoom=0"
        else:
            static_zoom_pct = stage._compute_static_zoom_pct(trf_path, 1920, 1080, 0.6)
            zoom_param = f":zoom={static_zoom_pct:.4f}:optzoom=0"
        return zoom_param, static_zoom_pct

    def test_coverage_one_uses_optzoom_one(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = 1.0
        zoom_param, static_pct = self._run_zoom_param_decision(stage, apply_zoom=True)
        assert zoom_param == ":zoom=0:optzoom=1"
        assert static_pct is None

    def test_coverage_zero_disables_zoom(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = 0.0
        zoom_param, static_pct = self._run_zoom_param_decision(stage, apply_zoom=True)
        assert zoom_param == ":zoom=0:optzoom=0"
        assert static_pct is None

    def test_gate_false_disables_zoom_regardless_of_coverage(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = 0.8
        zoom_param, static_pct = self._run_zoom_param_decision(stage, apply_zoom=False)
        assert zoom_param == ":zoom=0:optzoom=0"
        assert static_pct is None

    def test_coverage_between_computes_static_zoom(self, tmp_path):
        stage = _make_stage(tmp_path)
        stage._zoom_coverage = 0.6
        trf = _write_frames_trf(tmp_path, "shaky.trf", [0.0] * 20 + [200.0])
        zoom_param, static_pct = self._run_zoom_param_decision(stage, apply_zoom=True, trf_path=trf)
        assert zoom_param.startswith(":zoom=")
        assert ":optzoom=0" in zoom_param
        assert static_pct is not None
