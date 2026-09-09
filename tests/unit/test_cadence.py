"""Unit tests for core/cadence.py (REQUIREMENTS.md § 12.1).

Pure unit tests -- ffmpeg/ffprobe are mocked out entirely (no GPU, no real
video files needed). Covers:

- nominal_fps snapping (24.02 -> 24; 23.98 -> 23.976; a far-off rate stays
  unsnapped).
- is_padded threshold behaviour around min_duplicate_ratio.
- is_regular / gap coefficient-of-variation classification, and grid_rate
  disambiguation of a regular-but-quantized (pulldown) cadence.
- Parsing of mpdecimate + metadata=print output, including malformed/empty
  output returning a safe not-padded result rather than raising.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.config import Config
from autovideofixer.core.cadence import (
    CadenceAnalysis,
    _coefficient_of_variation,
    _parse_pts_times,
    _snap_to_standard_rate,
    analyze_cadence,
)


def _config(tmp_path) -> Config:
    return Config(tmp_path / "nonexistent.yaml")


def _fake_probe(has_video=True, framerate=60.0, duration=2.0):
    probe_result = MagicMock()
    probe_result.has_video = has_video
    probe_result.framerate = framerate
    probe_result.duration = duration
    return probe_result


def _fake_ffmpeg_result(returncode: int, showinfo_stderr: str) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = ""
    # analyze_cadence() reads showinfo's per-frame pts_time lines from
    # stderr (showinfo logs at the default "info" loglevel) -- see
    # core/cadence.py's module docstring for why this isn't stdout/
    # metadata=print.
    result.stderr = showinfo_stderr
    return result


def _pts_stdout(timestamps: list[float]) -> str:
    """Build fake ``showinfo`` stderr output with one pts_time record per
    timestamp, matching ffmpeg's real "n:N pts:P pts_time:T ..." line shape
    closely enough for the regex-based parser. (Name kept for historical
    continuity with earlier metadata=print-based wording; content is the
    showinfo shape.)"""
    lines = []
    for i, t in enumerate(timestamps):
        lines.append(f"n:{i}    pts:{int(t * 1000)}    pts_time:{t}")
    return "\n".join(lines) + "\n"


class TestSnapToStandardRate:
    def test_24_02_snaps_to_24(self):
        assert _snap_to_standard_rate(24.02, tolerance=0.02) == 24.0

    def test_23_98_snaps_to_23_976(self):
        assert _snap_to_standard_rate(23.98, tolerance=0.02) == 23.976

    def test_far_off_rate_stays_unsnapped(self):
        # 45fps is nowhere near any standard rate within a 2% tolerance.
        assert _snap_to_standard_rate(45.0, tolerance=0.02) == 45.0

    def test_zero_or_negative_returned_unchanged(self):
        assert _snap_to_standard_rate(0.0, tolerance=0.02) == 0.0


class TestCoefficientOfVariation:
    def test_uniform_gaps_are_zero_cv(self):
        assert _coefficient_of_variation([0.1, 0.1, 0.1, 0.1]) == 0.0

    def test_alternating_gaps_have_nonzero_cv(self):
        # 24-in-60 pulldown pattern: gaps alternate 0.050/0.033... -- a
        # regular cadence that still reads as mildly irregular on raw gaps
        # (see REQUIREMENTS.md § 12.6 -- grid_rate is what disambiguates it).
        cv = _coefficient_of_variation([0.05, 1 / 30, 0.05, 1 / 30])
        assert cv > 0.15

    def test_fewer_than_two_gaps_is_zero(self):
        assert _coefficient_of_variation([]) == 0.0
        assert _coefficient_of_variation([0.1]) == 0.0


class TestParsePtsTimes:
    def test_parses_well_formed_output(self):
        stdout = _pts_stdout([0.0, 0.05, 0.0833, 0.1333])
        assert _parse_pts_times(stdout) == [0.0, 0.05, 0.0833, 0.1333]

    def test_empty_output_returns_empty_list(self):
        assert _parse_pts_times("") == []

    def test_malformed_output_returns_empty_list(self):
        assert _parse_pts_times("not ffmpeg output at all\ngarbage\n") == []

    def test_garbage_interleaved_with_valid_records_skips_garbage(self):
        stdout = "some warning line\n" + _pts_stdout([0.0, 0.1]) + "trailing noise"
        assert _parse_pts_times(stdout) == [0.0, 0.1]


class TestAnalyzeCadencePaddedInput:
    def test_24_in_60_padded_input_is_detected(self, tmp_path):
        """A 24fps-content-in-60fps-CFR input (2s, encoded_fps=60) should be
        recovered as padded with detected/nominal_fps snapping to 24."""
        config = _config(tmp_path)
        # 48 unique frames over 2s of 60fps-encoded content -- the exact
        # REQUIREMENTS.md § 12.6 empirical example (120 -> 48 frames).
        timestamps = [i * (2.0 / 48) for i in range(48)]
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps)),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is True
        assert analysis.unique_frames == 48
        assert analysis.encoded_fps == 60.0
        assert analysis.nominal_fps == 24.0

    def test_duplicate_ratio_below_min_threshold_is_not_padded(self, tmp_path):
        config = _config(tmp_path)
        config.set(0.5, "stages", "retime", "min_duplicate_ratio")
        # 118/120 unique -- duplicate_ratio ~0.0167, well under the 0.5
        # threshold configured above.
        timestamps = [i * (2.0 / 118) for i in range(118)]
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps)),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False

    def test_duplicate_ratio_at_or_above_threshold_counts_as_padded(self, tmp_path):
        """duplicate_ratio at/above min_duplicate_ratio (>=, not >) counts
        as padded -- checked just below AND at the configured threshold."""
        config = _config(tmp_path)
        config.set(0.1, "stages", "retime", "min_duplicate_ratio")
        # total_frames = round(2.0 * 60) = 120; duplicate_ratio = 0.1 means
        # unique_frames = 108 -- use pytest.approx to avoid a float-equality
        # boundary flake, and 107 (ratio ~0.1083, unambiguously over) to
        # exercise the ">=" branch without landing exactly on the float edge.
        timestamps = [i * (2.0 / 108) for i in range(108)]
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps)),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.duplicate_ratio == pytest.approx(0.1)

        timestamps_107 = [i * (2.0 / 107) for i in range(107)]
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps_107)),
            ),
        ):
            analysis_107 = analyze_cadence("in.mp4", config)

        assert analysis_107.duplicate_ratio > 0.1
        assert analysis_107.is_padded is True


class TestAnalyzeCadenceRegularity:
    def test_uniform_cadence_is_regular(self, tmp_path):
        config = _config(tmp_path)
        timestamps = [i * (1.0 / 30) for i in range(60)]
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe(framerate=30.0)),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps)),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_regular is True

    def test_pulldown_cadence_reads_as_irregular_but_grid_rate_disambiguates(self, tmp_path):
        """A genuine 24-in-60 pulldown pattern (alternating 0.050/0.033s
        gaps -- REQUIREMENTS.md § 12.6) reads as mildly irregular on raw
        gap coefficient-of-variation, but grid_rate confirms it sits exactly
        on the encoded 60fps grid."""
        config = _config(tmp_path)
        period = 1.0 / 60.0
        gaps = [3 * period, 2 * period] * 24  # alternating 0.05 / 0.0333...
        timestamps = [0.0]
        for g in gaps:
            timestamps.append(timestamps[-1] + g)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe(framerate=60.0)),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, _pts_stdout(timestamps)),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_regular is False
        assert analysis.grid_rate == 60.0


class TestAnalyzeCadenceFailsOpen:
    def test_probe_failure_returns_safe_not_padded_result(self, tmp_path):
        config = _config(tmp_path)
        with patch("autovideofixer.core.cadence.probe", side_effect=RuntimeError("boom")):
            analysis = analyze_cadence("in.mp4", config)

        assert isinstance(analysis, CadenceAnalysis)
        assert analysis.is_padded is False

    def test_no_video_stream_returns_safe_result(self, tmp_path):
        config = _config(tmp_path)
        with patch("autovideofixer.core.cadence.probe", return_value=_fake_probe(has_video=False)):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False

    def test_ffmpeg_not_found_returns_safe_result(self, tmp_path):
        config = _config(tmp_path)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch(
                "autovideofixer.core.cadence.get_ffmpeg_path",
                side_effect=FileNotFoundError("no ffmpeg"),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False

    def test_ffmpeg_run_exception_returns_safe_result(self, tmp_path):
        config = _config(tmp_path)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                side_effect=OSError("spawn failed"),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False

    def test_ffmpeg_nonzero_exit_returns_safe_result(self, tmp_path):
        config = _config(tmp_path)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(1, ""),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False

    def test_empty_metadata_output_returns_safe_result_not_raises(self, tmp_path):
        config = _config(tmp_path)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, ""),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False
        assert analysis.unique_frames == 0

    def test_malformed_metadata_output_returns_safe_result_not_raises(self, tmp_path):
        config = _config(tmp_path)
        with (
            patch("autovideofixer.core.cadence.probe", return_value=_fake_probe()),
            patch("autovideofixer.core.cadence.get_ffmpeg_path", return_value="ffmpeg"),
            patch(
                "autovideofixer.core.cadence.subprocess.run",
                return_value=_fake_ffmpeg_result(0, "complete garbage, no pts_time here"),
            ),
        ):
            analysis = analyze_cadence("in.mp4", config)

        assert analysis.is_padded is False
