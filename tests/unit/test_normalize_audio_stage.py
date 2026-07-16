"""Unit tests for the normalize_audio / normalize_volume silence-skip fix.

Inputs with no real audio track get a silent stereo track added earlier in
the pipeline, so by the time NormalizeAudioStage/NormalizeVolumeStage run
there IS an audio stream -- just silent, or near-silent with dithering.
Two-pass loudnorm's first pass measures `input_i = -inf` on pure silence,
and feeding that back into the second pass blows up (infinite gain),
failing the stage. These tests cover the fix: the stage should skip
normalization (pass the input through unchanged, StageStatus.COMPLETED)
rather than fail, matching UpscaleStage's "already at target" /
StabilizeStage's "no stabilization needed" mid-execute skip convention.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.normalize_audio import (
    DEFAULT_SILENCE_THRESHOLD_DB,
    NormalizeAudioStage,
    NormalizeVolumeStage,
    _resolve_measured_i,
)


def _make_config(tmp_path) -> Config:
    return Config(tmp_path / "nonexistent.yaml")


def _measure_result(stderr: str) -> MagicMock:
    return MagicMock(returncode=0, stderr=stderr)


def _loudnorm_json(input_i: str) -> str:
    return (
        "[Parsed_loudnorm_0 @ 0x0] \n"
        "{\n"
        f'\t"input_i" : "{input_i}",\n'
        '\t"input_tp" : "-1.00",\n'
        '\t"input_lra" : "0.00",\n'
        '\t"input_thresh" : "-30.00",\n'
        '\t"output_i" : "-23.00",\n'
        '\t"output_tp" : "-2.00",\n'
        '\t"output_lra" : "0.00",\n'
        '\t"output_thresh" : "-33.00",\n'
        '\t"normalization_type" : "dynamic",\n'
        '\t"target_offset" : "0.00"\n'
        "}\n"
    )


class TestResolveMeasuredI:
    def test_parses_finite_string(self):
        assert _resolve_measured_i("-16.5") == pytest.approx(-16.5)

    def test_negative_inf_string_is_sentinel(self):
        assert math.isinf(_resolve_measured_i("-inf"))

    def test_positive_inf_string_is_sentinel(self):
        assert math.isinf(_resolve_measured_i("inf"))

    def test_nan_string_is_sentinel(self):
        assert math.isinf(_resolve_measured_i("nan"))

    def test_unparseable_is_sentinel(self):
        assert math.isinf(_resolve_measured_i("not-a-number"))

    def test_none_is_sentinel(self):
        assert math.isinf(_resolve_measured_i(None))

    def test_float_passthrough(self):
        assert _resolve_measured_i(-23.0) == -23.0


class TestNormalizeAudioSilenceSkip:
    """measured_I == -inf (pure digital silence)."""

    def test_perfect_silence_skips_and_passes_through(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            # First call: measurement pass. Second call: passthrough copy.
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-inf")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.success is True
        assert result.output_path == output_path
        assert result.skipped_reason is not None
        assert "silent" in result.skipped_reason.lower()
        # Only measurement + passthrough copy -- no second loudnorm apply pass.
        assert mock_run.call_count == 2
        second_call_args = mock_run.call_args_list[1].args[0]
        assert "-c" in second_call_args
        assert "copy" in second_call_args
        assert "loudnorm" not in " ".join(second_call_args)

    def test_near_silence_below_default_threshold_skips(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-85.2")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert "near-silent" in result.skipped_reason.lower()
        assert mock_run.call_count == 2

    def test_above_threshold_proceeds_to_second_pass(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-79.0")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.skipped_reason is None
        assert mock_run.call_count == 2
        second_call_args = mock_run.call_args_list[1].args[0]
        assert any("loudnorm" in arg for arg in second_call_args)

    def test_unparseable_input_i_skips_without_raising(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("nan")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.error is None

    def test_threshold_override_via_stage_config(self, tmp_path):
        """A more permissive (lower) threshold should NOT skip a value that
        would have skipped under the default -80.0."""
        config = _make_config(tmp_path)
        config.set(-90.0, "stages", "normalize_audio", "silence_threshold_db")
        stage = NormalizeAudioStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-85.2")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.skipped_reason is None  # -85.2 > -90.0 threshold now
        second_call_args = mock_run.call_args_list[1].args[0]
        assert any("loudnorm" in arg for arg in second_call_args)

    def test_threshold_override_via_per_occurrence_overrides(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config, overrides={"silence_threshold_db": -70.0})
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-75.0")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        # -75.0 <= -70.0 override threshold -> skip, even though it would
        # NOT have skipped under the -80.0 default.
        assert result.status == StageStatus.COMPLETED
        assert result.skipped_reason is not None

    def test_no_audio_track_should_run_skips(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeAudioStage(config)
        should_run, reason = stage.should_run({"has_audio": False})
        assert should_run is False
        assert reason == "No audio stream"


class TestNormalizeVolumeSilenceSkip:
    """NormalizeVolumeStage delegates to NormalizeAudioStage but must use its
    OWN config section's silence_threshold_db and report its own stage name."""

    def test_perfect_silence_skips(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeVolumeStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-inf")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        assert result.status == StageStatus.COMPLETED
        assert result.skipped_reason is not None

    def test_own_config_section_threshold_used(self, tmp_path):
        """normalize_volume's own silence_threshold_db must be honored, not
        normalize_audio's (a different, unrelated config section)."""
        config = _make_config(tmp_path)
        config.set(-90.0, "stages", "normalize_volume", "silence_threshold_db")
        stage = NormalizeVolumeStage(config)
        input_path = str(tmp_path / "in.mp4")
        output_path = str(tmp_path / "out.mp4")

        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as mock_run:
            mock_run.side_effect = [
                _measure_result(_loudnorm_json("-85.2")),
                MagicMock(returncode=0, stderr=""),
            ]
            result = stage.execute(input_path, output_path)

        # -85.2 > -90.0 (normalize_volume's overridden threshold) -> proceeds
        assert result.skipped_reason is None

    def test_no_audio_track_should_run_skips(self, tmp_path):
        config = _make_config(tmp_path)
        stage = NormalizeVolumeStage(config)
        should_run, reason = stage.should_run({"has_audio": False})
        assert should_run is False
        assert reason == "No audio stream"


class TestDefaultThresholdConstant:
    def test_default_is_minus_80(self):
        assert DEFAULT_SILENCE_THRESHOLD_DB == -80.0
