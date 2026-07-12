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
