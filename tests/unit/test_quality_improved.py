"""Tests for improved quality estimation (SSIM/PSNR).

Regression fixtures below use the real ffmpeg n8.1.2 `psnr`/`ssim` filter
summary-line format, captured live against `local_test_videos/` fixtures:

    PSNR y:18.29 u:42.98 v:43.59 average:20.05 min:19.07 max:21.35
    SSIM Y:0.74 (5.85) U:0.98 (19.88) V:0.99 (20.50) All:0.82 (7.53)
"""

from unittest.mock import MagicMock, patch

from autovideofixer.core.ffmpeg_utils import ProbeResult, StreamInfo
from autovideofixer.core.quality import (
    QualityResult,
    _parse_ssim_psnr_stderr,
    estimate_ssim_psnr,
)


def _fake_probe_result(width: int, height: int, video_stream_index: int = 0) -> ProbeResult:
    """Build a ProbeResult whose video stream isn't necessarily streams[0], to
    mirror containers that mux an audio stream before the video stream."""
    streams = []
    for i in range(video_stream_index):
        streams.append(StreamInfo(index=i, codec_type="audio"))
    streams.append(
        StreamInfo(index=video_stream_index, codec_type="video", width=width, height=height)
    )
    return ProbeResult(filepath="dist.mp4", filename="dist.mp4", streams=streams)


class TestParseSsimPsnr:
    """Test SSIM/PSNR stderr parsing against real ffmpeg output format."""

    def test_parse_empty_stderr(self):
        result = _parse_ssim_psnr_stderr("")
        assert result == {}

    def test_parse_psnr_values(self):
        stderr = (
            "[Parsed_psnr_2 @ 0x1] PSNR y:18.296554 u:42.987233 v:43.593307 "
            "average:20.050578 min:19.070202 max:21.354430\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert result["psnr"] == 20.050578
        assert result["psnr_y"] == 18.296554
        assert result["psnr_min"] == 19.070202
        assert result["psnr_max"] == 21.354430

    def test_parse_ssim_values(self):
        stderr = (
            "[Parsed_ssim_3 @ 0x1] SSIM Y:0.740100 (5.851931) U:0.989723 (19.881337) "
            "V:0.991102 (20.507235) All:0.823537 (7.533471)\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert result["ssim"] == 0.823537
        assert result["ssim_y"] == 0.740100
        assert 0 < result["ssim"] <= 1

    def test_parse_combined(self):
        stderr = (
            "[Parsed_psnr_2 @ 0x1] PSNR y:18.296554 u:42.987233 v:43.593307 "
            "average:20.050578 min:19.070202 max:21.354430\n"
            "[Parsed_ssim_3 @ 0x1] SSIM Y:0.740100 (5.851931) U:0.989723 (19.881337) "
            "V:0.991102 (20.507235) All:0.823537 (7.533471)\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert "psnr" in result
        assert "ssim" in result

    def test_parse_infinite_values(self):
        """Identical frames make ffmpeg report 'inf' rather than a number."""
        stderr = (
            "[Parsed_psnr_2 @ 0x1] PSNR y:inf u:inf v:inf average:inf min:inf max:inf\n"
            "[Parsed_ssim_3 @ 0x1] SSIM Y:1.000000 (inf) U:1.000000 (inf) "
            "V:1.000000 (inf) All:1.000000 (inf)\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert result["psnr"] == float("inf")
        assert result["ssim"] == 1.0

    def test_parse_no_matching_lines(self):
        stderr = "some random output without metrics\n"
        result = _parse_ssim_psnr_stderr(stderr)
        assert result == {}


class TestEstimateSsimPsnr:
    """Test SSIM/PSNR quality estimation."""

    def test_estimate_with_bad_files(self, tmp_path):
        """Nonexistent input files should fail cleanly, not raise."""
        result = estimate_ssim_psnr(
            str(tmp_path / "nonexistent1.mp4"),
            str(tmp_path / "nonexistent2.mp4"),
        )
        assert isinstance(result, QualityResult)
        assert "error" in result.details

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_estimate_mock_ffmpeg(self, mock_run):
        """Test estimation with mocked FFmpeg, using the real output format."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = (
            "[Parsed_psnr_2 @ 0x1] PSNR y:44.0 u:45.0 v:45.0 average:45.0 min:44.0 max:46.0\n"
            "[Parsed_ssim_3 @ 0x1] SSIM Y:0.975 (16.0) U:0.98 (17.0) V:0.98 (17.0) "
            "All:0.980 (17.0)\n"
        )
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert result.psnr == 45.0
        assert result.ssim == 0.980
        # This path doesn't compute VMAF - it must not be populated from PSNR.
        assert result.vmaf_score == 0.0

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_timeout_defaults_to_none_unlimited(self, mock_run):
        """quality.timeout's DEFAULTS value is null (unlimited) -- a caller
        that doesn't pass timeout explicitly must not silently fall back to
        run_ffmpeg's own historical fixed default."""
        mock_result = MagicMock(returncode=0, stderr="")
        mock_run.return_value = mock_result

        estimate_ssim_psnr("ref.mp4", "dist.mp4")

        assert mock_run.call_args.kwargs["timeout"] is None

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_explicit_timeout_reaches_run_ffmpeg(self, mock_run):
        mock_result = MagicMock(returncode=0, stderr="")
        mock_run.return_value = mock_result

        estimate_ssim_psnr("ref.mp4", "dist.mp4", timeout=250)

        assert mock_run.call_args.kwargs["timeout"] == 250

    @patch("autovideofixer.core.quality.run_ffmpeg", side_effect=RuntimeError("ffmpeg failed"))
    def test_estimate_ffmpeg_error(self, mock_run):
        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert isinstance(result, QualityResult)
        assert "error" in result.details

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_estimate_ffmpeg_nonzero_exit(self, mock_run):
        """A non-zero ffmpeg exit must not be treated as a valid (zeroed) result."""
        mock_result = MagicMock()
        mock_result.returncode = 234
        mock_result.stderr = "Cannot find an unused video input stream\n"
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert "error" in result.details

    @patch("autovideofixer.core.ffmpeg_utils.probe")
    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_mismatched_resolution_scales_reference_to_output(self, mock_run, mock_probe):
        """Regression test: comparing an upscaled output against its original
        reference must scale the reference to the output's resolution in the
        filter graph, or ffmpeg rejects the comparison with a dimension mismatch
        (surfacing as a fake 0.0 score via the pre-fix code path)."""
        mock_probe.return_value = _fake_probe_result(3840, 2160)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = (
            "[Parsed_psnr_2 @ 0x1] PSNR y:44.0 u:45.0 v:45.0 average:45.0 min:44.0 max:46.0\n"
            "[Parsed_ssim_3 @ 0x1] SSIM Y:0.975 (16.0) U:0.98 (17.0) V:0.98 (17.0) "
            "All:0.980 (17.0)\n"
        )
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref_1080p.mp4", "dist_4k.mp4")

        assert result.measurement_failed is False
        assert result.psnr == 45.0
        cmd = mock_run.call_args[0][0]
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        assert "scale=3840:2160" in filter_complex
        # Reference (input 0) is what gets scaled, not the distorted output.
        assert filter_complex.startswith("[0:v]scale=3840:2160")

    @patch("autovideofixer.core.ffmpeg_utils.probe")
    def test_uses_video_stream_not_streams_index_zero(self, mock_probe):
        """A container where an audio stream is muxed before the video stream
        must still pick up the video stream's dimensions, not silently fall back
        to the no-scale filter (streams[0] would be audio: width/height 0)."""
        mock_probe.return_value = _fake_probe_result(1280, 720, video_stream_index=1)

        with patch("autovideofixer.core.quality.run_ffmpeg") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = (
                "[Parsed_psnr_2 @ 0x1] PSNR y:40.0 u:41.0 v:41.0 "
                "average:41.0 min:40.0 max:42.0\n"
                "[Parsed_ssim_3 @ 0x1] SSIM Y:0.9 (10.0) U:0.9 (10.0) V:0.9 (10.0) "
                "All:0.900 (10.0)\n"
            )
            mock_run.return_value = mock_result

            estimate_ssim_psnr("ref.mp4", "dist.mkv")

            cmd = mock_run.call_args[0][0]
            filter_complex = cmd[cmd.index("-filter_complex") + 1]
            assert "scale=1280:720" in filter_complex

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_measurement_failed_flag_set_on_nonzero_exit(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Width and height of input videos must be same\n"
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4", target=95.0)

        assert result.measurement_failed is True
        # A failed measurement must not read as a genuine 0.0 quality score.
        assert result.score == 0.0
        assert result.meets_target() is False

    @patch("autovideofixer.core.quality.run_ffmpeg", side_effect=RuntimeError("boom"))
    def test_measurement_failed_flag_set_on_exception(self, mock_run):
        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert result.measurement_failed is True

    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_measurement_failed_flag_set_on_unparseable_output(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no metrics here\n"
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert result.measurement_failed is True

    def test_measurement_succeeded_flag_false_by_default(self):
        assert QualityResult().measurement_failed is False


class TestQualityResult:
    """Test QualityResult with SSIM/PSNR data."""

    def test_result_with_ssim_psnr(self):
        result = QualityResult(
            vmaf_score=0.0,
            psnr=42.5,
            ssim=0.975,
            ms_ssim=0.970,
            details={"psnr": 42.5, "ssim": 0.975, "psnr_min": 40.1, "ssim_min": 0.960},
        )
        assert result.psnr == 42.5
        assert result.ssim == 0.975
        assert result.ms_ssim == 0.970
