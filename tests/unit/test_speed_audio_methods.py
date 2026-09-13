"""Tests for the speed stage's configurable audio speed methods.

`SpeedStage` supports three ``stages.speed.audio_method`` values:

  - "atempo" (default): the original chained atempo filter, pitch-preserving,
    UNCHANGED behavior from before this feature.
  - "rubberband": ffmpeg's `rubberband` filter, also pitch-preserving but a
    single filter (no chaining) -- falls back to atempo (with a WARNING) if
    this ffmpeg build lacks librubberband.
  - "asetrate": resample-based and deliberately pitch-CHANGING (restores a
    phone slow-motion clip's true mic pitch) -- needs the input's probed
    audio sample rate, falls back to atempo (with a WARNING) if that can't
    be determined.

These tests exercise the pure filter-string builders directly (no ffmpeg
needed) and the fallback-selection logic in `_resolve_audio_filter` with
`ffmpeg_utils.has_filter`/`probe` mocked -- consistent with how the existing
`test_speed_fps_propagation.py` mocks `run_ffmpeg`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.ffmpeg_utils import ProbeResult, StreamInfo
from autovideofixer.core.stages.speed import SpeedStage


def _stage(overrides=None) -> SpeedStage:
    config = Config(config_path="/nonexistent/fresh.yaml")
    return SpeedStage(config, overrides=overrides)


class _Ok:
    returncode = 0
    stderr = ""
    stdout = ""


def _run(overrides, factor=0.25, input_info=None):
    stage = _stage(overrides)
    with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=_Ok()):
        return stage.execute(
            "in.mp4", "out.mp4", factor=factor, input_info=input_info or {"framerate": 30.0}
        )


class TestAtempoChainUnchanged:
    """`_build_atempo_chain` behavior must be exactly what it was before this feature."""

    def test_quarter_speed_chains_two_stages(self):
        assert SpeedStage._build_atempo_chain(0.25) == "atempo=0.5,atempo=0.5"

    def test_half_speed_single_stage(self):
        assert SpeedStage._build_atempo_chain(0.5) == "atempo=0.5"

    def test_double_speed_single_stage(self):
        assert SpeedStage._build_atempo_chain(2.0) == "atempo=2.0"

    def test_eight_x_speed_chains_two_stages(self):
        # 8.0 is within a single atempo's [0.5, 100] range -- no chaining needed.
        assert SpeedStage._build_atempo_chain(8.0) == "atempo=8.0"

    def test_extreme_speed_up_chains(self):
        # 150 > 100, needs chaining: 100 * 1.5
        assert SpeedStage._build_atempo_chain(150.0) == "atempo=100.0,atempo=1.5"

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            SpeedStage._build_atempo_chain(0)
        with pytest.raises(ValueError):
            SpeedStage._build_atempo_chain(-1.0)


class TestRubberbandFilter:
    def test_quarter_speed(self):
        assert SpeedStage._build_rubberband_filter(0.25) == "rubberband=tempo=0.25"

    def test_no_chaining_at_extreme_factor(self):
        # Unlike atempo, rubberband needs no chaining even far outside [0.5, 100].
        assert SpeedStage._build_rubberband_filter(0.01) == "rubberband=tempo=0.01"

    def test_zero_or_negative_raises(self):
        with pytest.raises(ValueError):
            SpeedStage._build_rubberband_filter(0)


class TestAsetrateChain:
    def test_quarter_speed_44100hz_input(self):
        chain = SpeedStage._build_asetrate_chain(
            0.25, input_sample_rate=44100, audio_sample_rate=48000, resampler="soxr"
        )
        assert chain == "asetrate=11025,aresample=48000:resampler=soxr"

    def test_audio_sample_rate_zero_targets_input_rate(self):
        chain = SpeedStage._build_asetrate_chain(
            0.5, input_sample_rate=44100, audio_sample_rate=0, resampler="soxr"
        )
        assert chain == "asetrate=22050,aresample=44100:resampler=soxr"

    def test_swr_resampler(self):
        chain = SpeedStage._build_asetrate_chain(
            2.0, input_sample_rate=48000, audio_sample_rate=48000, resampler="swr"
        )
        assert chain == "asetrate=96000,aresample=48000:resampler=swr"

    def test_zero_or_negative_speed_raises(self):
        with pytest.raises(ValueError):
            SpeedStage._build_asetrate_chain(0, 44100, 48000, "soxr")

    def test_non_positive_input_sample_rate_raises(self):
        with pytest.raises(ValueError):
            SpeedStage._build_asetrate_chain(0.5, 0, 48000, "soxr")


class TestResolveAudioFilterRubberbandFallback:
    def test_rubberband_available_used_directly(self):
        stage = _stage({"audio_method": "rubberband"})
        with patch("autovideofixer.core.ffmpeg_utils.has_filter", return_value=True):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert chain == "rubberband=tempo=0.25"
        assert used == "rubberband"

    def test_rubberband_missing_falls_back_to_atempo_with_warning(self):
        stage = _stage({"audio_method": "rubberband"})
        with (
            patch("autovideofixer.core.ffmpeg_utils.has_filter", return_value=False),
            patch.object(stage, "_logger") as mock_logger,
        ):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert chain == "atempo=0.5,atempo=0.5"
        assert used == "atempo"
        mock_logger.warning.assert_called_once()


class TestResolveAudioFilterAsetrateFallback:
    def _probe_result(self, sample_rate: int) -> ProbeResult:
        stream = StreamInfo(index=0, codec_type="audio", sample_rate=sample_rate)
        return ProbeResult(filepath="in.mp4", filename="in.mp4", streams=[stream])

    def test_asetrate_with_known_sample_rate(self):
        stage = _stage({"audio_method": "asetrate"})
        with patch(
            "autovideofixer.core.ffmpeg_utils.probe", return_value=self._probe_result(44100)
        ):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert chain == "asetrate=11025,aresample=48000:resampler=soxr"
        assert used == "asetrate"

    def test_asetrate_probe_failure_falls_back_to_atempo_with_warning(self):
        stage = _stage({"audio_method": "asetrate"})
        with (
            patch(
                "autovideofixer.core.ffmpeg_utils.probe", side_effect=RuntimeError("probe failed")
            ),
            patch.object(stage, "_logger") as mock_logger,
        ):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert chain == "atempo=0.5,atempo=0.5"
        assert used == "atempo"
        mock_logger.warning.assert_called_once()

    def test_asetrate_no_audio_stream_falls_back_to_atempo_with_warning(self):
        stage = _stage({"audio_method": "asetrate"})
        no_audio = ProbeResult(filepath="in.mp4", filename="in.mp4", streams=[])
        with (
            patch("autovideofixer.core.ffmpeg_utils.probe", return_value=no_audio),
            patch.object(stage, "_logger") as mock_logger,
        ):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert chain == "atempo=0.5,atempo=0.5"
        assert used == "atempo"
        mock_logger.warning.assert_called_once()

    def test_asetrate_zero_sample_rate_falls_back_to_atempo(self):
        stage = _stage({"audio_method": "asetrate"})
        with patch("autovideofixer.core.ffmpeg_utils.probe", return_value=self._probe_result(0)):
            chain, used = stage._resolve_audio_filter(0.25, "in.mp4")
        assert used == "atempo"


class TestInvalidEnumConfig:
    def test_invalid_audio_method_fails_the_stage(self):
        result = _run({"audio_method": "bogus"})
        assert result.status.name == "FAILED"
        assert "audio_method" in result.error

    def test_invalid_resampler_fails_the_stage(self):
        result = _run({"audio_method": "asetrate", "resampler": "bogus"})
        assert result.status.name == "FAILED"
        assert "resampler" in result.error


class TestOutputSampleRateFlag:
    def _args_from(self, overrides, factor=0.25):
        stage = _stage(overrides)
        captured = {}

        def fake_run_ffmpeg(args, **kwargs):
            captured["args"] = args
            return _Ok()

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", side_effect=fake_run_ffmpeg):
            stage.execute("in.mp4", "out.mp4", factor=factor, input_info={"framerate": 30.0})
        return captured["args"]

    def test_ar_emitted_when_positive(self):
        args = self._args_from({"audio_sample_rate": 48000})
        assert "-ar" in args
        assert args[args.index("-ar") + 1] == "48000"

    def test_ar_omitted_when_zero(self):
        args = self._args_from({"audio_sample_rate": 0})
        assert "-ar" not in args

    def test_ar_uses_configured_default_of_48000(self):
        # Default audio_sample_rate (no override) is 48000, per Config.DEFAULTS.
        args = self._args_from({})
        assert "-ar" in args
        assert args[args.index("-ar") + 1] == "48000"


class TestSpeedFpsPropagationStillWorksWithAudioMethod:
    """Guard against the new audio_method metadata key breaking fps_in/fps_out."""

    def test_metadata_includes_audio_method_and_fps(self):
        result = _run({}, factor=0.5, input_info={"framerate": 60.0})
        assert result.status.name == "COMPLETED"
        assert result.metadata["audio_method"] == "atempo"
        assert result.metadata["fps_in"] == 60.0
        assert result.metadata["fps_out"] == 30.0
