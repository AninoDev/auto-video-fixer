"""REQUIREMENTS.md § 12.5: CFR deliverables must be constant at the stream's
TRUE cadence, not at the container's stale nominal rate.

`-fps_mode cfr` on its own conforms the output to whatever rate the container
still advertises. For a retimed 24-in-60 input that is still 60, so ffmpeg
faithfully re-inserts exactly the duplicate frames `retime` just removed
(observed end-to-end: 120 frames in, 48 after retime, 120 back out). Pinning
`-r` to the recovered cadence is what makes the deliverable honest 24fps.

The counterpart risk is over-pinning: if `true_framerate` were left at the
pre-interpolation cadence, encode would pin `-r 24` on a stream interpolate
had just raised to 60fps and throw away every synthesized frame. That is why
`interpolate` must report `fps_out` on EVERY completed path so the pipeline
can refresh `true_framerate`.
"""

from __future__ import annotations

from unittest.mock import patch

from autovideofixer.config import Config
from autovideofixer.core.stages.encode import EncodeStage


def _encode_args(monkeypatch, output_timing, input_info):
    """Run EncodeStage.execute and capture the ffmpeg args it builds."""
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run_ffmpeg(args, **kwargs):
        captured["args"] = args
        return _Result()

    config = Config(config_path="/nonexistent/fresh.yaml")
    config.apply_layer({"general": {"output_timing": output_timing}}, "test")
    stage = EncodeStage(config)

    # encode.py imports run_ffmpeg lazily INSIDE execute(), so the patch must
    # target the source module, not the stage module's namespace.
    with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", side_effect=fake_run_ffmpeg):
        stage.execute(
            "in.mkv",
            "out.mp4",
            input_info=input_info,
        )
    return captured.get("args", [])


def _rate_after_r(args):
    """The value ffmpeg was told to use for -r, or None if unpinned."""
    return args[args.index("-r") + 1] if "-r" in args else None


class TestCfrPinsRecoveredRate:
    def test_cfr_pins_r_to_true_framerate(self, monkeypatch):
        """A retimed 24-in-60 input must encode at 24, not the container's 60."""
        args = _encode_args(monkeypatch, "cfr", {"framerate": 60.0, "true_framerate": 24.0})

        assert "-fps_mode" in args and "cfr" in args
        assert _rate_after_r(args) == "24.0", (
            "CFR output was not pinned to the recovered cadence -- ffmpeg will "
            f"re-pad the dropped duplicate frames back in; args={args}"
        )

    def test_cfr_uses_interpolated_rate_when_interpolation_raised_it(self, monkeypatch):
        """After interpolate, true_framerate is the INTERPOLATED rate.

        The pipeline refreshes true_framerate from interpolate's fps_out, so
        encode must pin 60 here -- pinning the pre-interpolation 24 would
        discard every synthesized frame.
        """
        args = _encode_args(monkeypatch, "cfr", {"framerate": 60.0, "true_framerate": 60.0})

        assert _rate_after_r(args) == "60.0"

    def test_cfr_without_retime_does_not_pin_rate(self, monkeypatch):
        """No recovered cadence -> unchanged pre-feature behaviour (no -r)."""
        args = _encode_args(monkeypatch, "cfr", {"framerate": 30.0})

        assert _rate_after_r(args) is None

    def test_vfr_never_pins_a_constant_rate(self, monkeypatch):
        """VFR carries real timestamps; pinning -r would destroy them."""
        args = _encode_args(monkeypatch, "vfr", {"framerate": 60.0, "true_framerate": 24.0})

        assert "vfr" in args
        assert _rate_after_r(args) is None, f"VFR output must not be rate-pinned; args={args}"

    def test_passthrough_never_pins_a_constant_rate(self, monkeypatch):
        args = _encode_args(monkeypatch, "passthrough", {"framerate": 60.0, "true_framerate": 24.0})

        assert _rate_after_r(args) is None


class TestInterpolateReportsFpsOut:
    """Every completed interpolate path must report fps_out (see module docstring)."""

    def test_traditional_single_chunk_reports_fps_out(self):
        from autovideofixer.core.stages.interpolate import InterpolateStage

        config = Config(config_path="/nonexistent/fresh.yaml")
        stage = InterpolateStage(config)

        class _Ok:
            returncode = 0
            stderr = ""
            stdout = ""

        with patch("autovideofixer.core.stages.interpolate.run_ffmpeg", return_value=_Ok()):
            result = stage._execute_traditional_single("in.mkv", "out.mkv", None, 0.0, 60.0, 2)

        assert result.metadata.get("fps_out") == 60.0, (
            "traditional interpolate did not report fps_out -- the pipeline "
            "cannot refresh true_framerate and encode will pin the stale rate"
        )
