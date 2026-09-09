"""The adaptive-interpolation gate must not reject VFR input (REQUIREMENTS.md
§ 16.2).

`_get_adaptive_timeline()` decides whether the § 16.1 adaptive path runs at
all. An earlier implementation validated the probed timeline's length against
`probe().frame_count` -- but that count is derived from `avg_frame_rate`, which
§ 12.3 documents as unreliable for VFR, and a VFR intermediate is the ONLY
input this path exists to serve.

Measured on real files: a 48-frame retimed intermediate reports
`frame_count=119`, and a 123-frame native VFR capture reports `180`. So the
equality check failed 100% of the time on exactly the inputs adaptive
interpolation is for, silently disabling the feature while every gate still
passed and the stage still reported COMPLETED.

These tests pin the gate to the timeline's own structure instead.
"""

from __future__ import annotations

from unittest.mock import patch

from autovideofixer.config import Config
from autovideofixer.core.stages.interpolate import InterpolateStage


class _FakeProbe:
    has_video = True

    def __init__(self, frame_count, duration):
        self.frame_count = frame_count
        self.duration = duration


def _stage():
    return InterpolateStage(Config(config_path="/nonexistent/fresh.yaml"))


def _run_gate(timestamps, frame_count, duration=2.0):
    stage = _stage()
    with (
        patch(
            "autovideofixer.core.stages.interpolate.probe",
            return_value=_FakeProbe(frame_count, duration),
        ),
        patch(
            "autovideofixer.core.cadence.probe_frame_timestamps",
            return_value=timestamps,
        ),
    ):
        return stage._get_adaptive_timeline("in.mkv")


class TestAdaptiveGateAcceptsVfr:
    def test_vfr_timeline_accepted_despite_bogus_frame_count(self):
        """The real-world regression: 48 real frames, container claims 119."""
        timestamps = [i * (1 / 24) for i in range(48)]
        result = _run_gate(timestamps, frame_count=119, duration=2.0)

        assert result is not None, (
            "adaptive path was rejected for a VFR input -- the frame_count "
            "equality check is back, which silently disables the entire feature"
        )
        assert result[0] == timestamps

    def test_native_vfr_capture_accepted(self):
        """123 real frames, container claims 180."""
        timestamps = [i * (1 / 41) for i in range(123)]
        result = _run_gate(timestamps, frame_count=180, duration=3.0)

        assert result is not None

    def test_matching_count_still_accepted(self):
        """An honest CFR file must keep working too."""
        timestamps = [i * (1 / 30) for i in range(60)]
        result = _run_gate(timestamps, frame_count=60, duration=2.0)

        assert result is not None


class TestAdaptiveGateStillRejectsBadTimelines:
    """The gate must stay meaningful -- it just uses the right signal now."""

    def test_non_monotonic_rejected(self):
        timestamps = [0.0, 0.1, 0.05, 0.2]
        assert _run_gate(timestamps, frame_count=4, duration=0.2) is None

    def test_zero_span_rejected(self):
        timestamps = [0.5, 0.5, 0.5]
        assert _run_gate(timestamps, frame_count=3, duration=1.0) is None

    def test_truncated_probe_rejected(self):
        """A timeline covering a fraction of the file is a partial probe."""
        timestamps = [i * (1 / 30) for i in range(10)]  # ~0.3s
        assert _run_gate(timestamps, frame_count=300, duration=10.0) is None

    def test_single_frame_rejected(self):
        assert _run_gate([0.0], frame_count=1, duration=0.1) is None

    def test_empty_timeline_rejected(self):
        assert _run_gate([], frame_count=0, duration=0.0) is None

    def test_missing_timeline_rejected(self):
        assert _run_gate(None, frame_count=48, duration=2.0) is None

    def test_probe_failure_fails_open(self):
        """A probe exception must return None, never propagate."""
        stage = _stage()
        with patch(
            "autovideofixer.core.stages.interpolate.probe",
            side_effect=RuntimeError("probe blew up"),
        ):
            assert stage._get_adaptive_timeline("in.mkv") is None
