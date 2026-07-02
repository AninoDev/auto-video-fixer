"""Tests for improved quality estimation (SSIM/PSNR).

Regression fixtures below use the real ffmpeg n8.1.2 `psnr`/`ssim` filter
summary-line format, captured live against `local_test_videos/` fixtures:

    PSNR y:18.29 u:42.98 v:43.59 average:20.05 min:19.07 max:21.35
    SSIM Y:0.74 (5.85) U:0.98 (19.88) V:0.99 (20.50) All:0.82 (7.53)
"""

from unittest.mock import MagicMock, patch

from autovideofixer.core.quality import (
    QualityResult,
    _parse_ssim_psnr_stderr,
    estimate_ssim_psnr,
)


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
