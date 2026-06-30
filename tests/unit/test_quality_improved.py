"""Tests for improved quality estimation (SSIM/PSNR)."""

from unittest.mock import MagicMock, patch

from autovideofixer.core.quality import (
    QualityResult,
    _parse_ssim_psnr_stderr,
    estimate_ssim_psnr,
)


class TestParseSsimPsnr:
    """Test SSIM/PSNR stderr parsing."""

    def test_parse_empty_stderr(self):
        """Test parsing empty stderr."""
        result = _parse_ssim_psnr_stderr("")
        assert result == {}

    def test_parse_psnr_values(self):
        """Test parsing PSNR values from FFmpeg output."""
        stderr = (
            "frame=   18  psnr_y: 45.234 psnr_u: 44.123 psnr_v: 43.567\n"
            "frame=   36  psnr_y: 44.891 psnr_u: 43.789 psnr_v: 43.234\n"
            "PSNR average: 44.563  PSNR minimum: 43.234  PSNR maximum: 45.234\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert "psnr" in result
        assert result["psnr"] > 0

    def test_parse_ssim_values(self):
        """Test parsing SSIM values from FFmpeg output."""
        stderr = (
            "frame=   18  ssim_y: 0.9823 ssim_u: 0.9712 ssim_v: 0.9678\n"
            "frame=   36  ssim_y: 0.9801 ssim_u: 0.9701 ssim_v: 0.9667\n"
            "SSIM average: 0.9750  SSIM maximum: 0.9823  SSIM minimum: 0.9667\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert "ssim" in result
        assert 0 < result["ssim"] <= 1

    def test_parse_combined(self):
        """Test parsing combined PSNR and SSIM."""
        stderr = (
            "frame=   18  psnr_y: 42.5 ssim_y: 0.975\nPSNR average: 42.5  SSIM average: 0.975\n"
        )
        result = _parse_ssim_psnr_stderr(stderr)
        assert "psnr" in result
        assert "ssim" in result

    def test_parse_no_matching_lines(self):
        """Test parsing stderr with no PSNR/SSIM lines."""
        stderr = "some random output without metrics\n"
        result = _parse_ssim_psnr_stderr(stderr)
        assert result == {}


class TestEstimateSsimPsnr:
    """Test SSIM/PSNR quality estimation."""

    def test_estimate_with_bad_files(self, tmp_path):
        """Test estimation with nonexistent files."""
        result = estimate_ssim_psnr(
            str(tmp_path / "nonexistent1.mp4"),
            str(tmp_path / "nonexistent2.mp4"),
        )
        assert isinstance(result, QualityResult)

    @patch("autovideofixer.core.quality.get_ffmpeg_path")
    @patch("autovideofixer.core.quality.run_ffmpeg")
    def test_estimate_mock_ffmpeg(self, mock_run, mock_path):
        """Test estimation with mocked FFmpeg."""
        mock_path.return_value = "/usr/bin/ffmpeg"

        # Mock successful FFmpeg output
        mock_result = MagicMock()
        mock_result.stderr = (
            "frame=   18  psnr_y: 45.0 ssim_y: 0.980\nPSNR average: 45.0  SSIM average: 0.980\n"
        )
        mock_run.return_value = mock_result

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert result.psnr > 0
        assert 0 < result.ssim <= 1

    @patch("autovideofixer.core.quality.get_ffmpeg_path")
    @patch("autovideofixer.core.quality.run_ffmpeg", side_effect=RuntimeError("ffmpeg failed"))
    def test_estimate_ffmpeg_error(self, mock_run, mock_path):
        """Test estimation when FFmpeg fails."""
        mock_path.return_value = "/usr/bin/ffmpeg"

        result = estimate_ssim_psnr("ref.mp4", "dist.mp4")
        assert isinstance(result, QualityResult)
        assert "error" in result.details


class TestQualityResult:
    """Test QualityResult with SSIM/PSNR data."""

    def test_result_with_ssim_psnr(self):
        """Test creating a result with SSIM and PSNR."""
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
