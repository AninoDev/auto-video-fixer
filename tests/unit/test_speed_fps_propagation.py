"""`speed` must report its real output cadence (REQUIREMENTS.md § 12.5).

`setpts` rescales timestamps and keeps the frame COUNT, so N frames over a
duration of D/factor is an effective rate of `in_fps * factor`. The stage used
to report only `speed_factor`, so `input_info["true_framerate"]` kept the
PRE-speed rate. Two things broke as a result:

  - `encode`'s CFR path pins `-r true_framerate`, so a 0.5x slow-motion clip
    stayed tagged at 60fps while carrying only 30fps of real motion -- ffmpeg
    duplicated frames to fill the gap, and the output claimed a framerate it
    did not actually have.
  - A second `interpolate` occurrence placed after `speed` saw the stale rate
    and skipped as "already at or above target", so slow motion could never be
    re-interpolated back up to the target.

The pipeline's propagation is keyed on the PRESENCE of `fps_out`, not on a
stage name, so any future retiming stage is covered by reporting it.
"""

from __future__ import annotations

from unittest.mock import patch

from autovideofixer.config import Config
from autovideofixer.core.stages.speed import SpeedStage


class _Ok:
    returncode = 0
    stderr = ""
    stdout = ""


def _run(factor, input_info):
    config = Config(config_path="/nonexistent/fresh.yaml")
    stage = SpeedStage(config)
    with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=_Ok()):
        return stage.execute("in.mp4", "out.mp4", factor=factor, input_info=input_info)


class TestSpeedReportsRealCadence:
    def test_slow_motion_halves_the_rate(self):
        """0.5x on 60fps yields 30fps of real motion."""
        result = _run(0.5, {"framerate": 60.0})

        assert result.metadata["fps_in"] == 60.0
        assert result.metadata["fps_out"] == 30.0, (
            "speed did not report its real output cadence -- encode will pin the "
            "pre-speed rate and duplicate frames to reach it"
        )

    def test_speed_up_doubles_the_rate(self):
        result = _run(2.0, {"framerate": 30.0})

        assert result.metadata["fps_out"] == 60.0

    def test_prefers_true_framerate_over_probed_framerate(self):
        """After retime/interpolate, true_framerate is the current cadence."""
        result = _run(0.5, {"framerate": 60.0, "true_framerate": 24.0})

        assert result.metadata["fps_in"] == 24.0
        assert result.metadata["fps_out"] == 12.0

    def test_missing_framerate_omits_fps_out_rather_than_guessing(self):
        """No usable input rate -> report nothing rather than a wrong number."""
        result = _run(0.5, {})

        assert "fps_out" not in result.metadata
        assert result.metadata["speed_factor"] == 0.5

    def test_non_numeric_framerate_does_not_raise(self):
        result = _run(0.5, {"framerate": "not-a-number"})

        assert "fps_out" not in result.metadata

    def test_factor_one_still_skips(self):
        """Unchanged: a no-op speed change is SKIPPED, not COMPLETED."""
        result = _run(1.0, {"framerate": 60.0})

        assert result.status.name == "SKIPPED"


class TestPipelinePropagatesAnyFpsOut:
    """Propagation must be keyed on fps_out's presence, not on a stage name."""

    def test_speed_metadata_shape_matches_what_pipeline_reads(self):
        """Guards the contract between SpeedStage and Pipeline.execute_job.

        The pipeline updates input_info["true_framerate"] from
        result.metadata["fps_out"] for any COMPLETED stage that reports it.
        """
        result = _run(0.5, {"framerate": 60.0})

        assert result.status.name == "COMPLETED"
        assert isinstance(result.metadata.get("fps_out"), float)
