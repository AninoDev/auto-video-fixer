"""Tests for quality estimation."""

import os
from pathlib import Path

import pytest

from autovideofixer.core.quality import (
    QualityResult,
    QualityMode,
    estimate_quality_vmaf,
    estimate_quality_loss,
)


class TestQualityResult:
    """Test quality result data structure."""

    def test_quality_result_creation(self):
        """Test creating a quality result."""
        result = QualityResult(
            vmaf_score=95.0,
            psnr=40.0,
            ssim=0.95,
            mode=QualityMode.TARGET,
            target=90.0,
        )
        
        assert result.vmaf_score == 95.0
        assert result.psnr == 40.0
        assert result.ssim == 0.95
        assert result.mode == QualityMode.TARGET

    def test_quality_result_score_none_mode(self):
        """Test score calculation with NONE mode."""
        result = QualityResult(
            vmaf_score=85.0,
            mode=QualityMode.NONE,
        )
        
        assert result.score == 85.0

    def test_quality_result_score_target_mode(self):
        """Test score calculation with TARGET mode."""
        result = QualityResult(
            vmaf_score=92.0,
            mode=QualityMode.TARGET,
            target=90.0,
        )
        
        assert result.score == 92.0
        assert result.meets_target() is True

    def test_quality_result_not_meeting_target(self):
        """Test target meeting when below threshold."""
        result = QualityResult(
            vmaf_score=85.0,
            mode=QualityMode.TARGET,
            target=90.0,
        )
        
        assert result.meets_target() is False

    def test_quality_result_min_mode(self):
        """Test score calculation with MIN mode."""
        result = QualityResult(
            vmaf_score=90.0,
            details={"vmaf_min": 85.0, "vmaf": 90.0, "vmaf_max": 95.0},
            mode=QualityMode.MIN,
        )
        
        assert result.score == 85.0

    def test_quality_result_avg_mode(self):
        """Test score calculation with AVG mode."""
        result = QualityResult(
            vmaf_score=90.0,
            details={"vmaf": 90.0},
            mode=QualityMode.AVG,
        )
        
        assert result.score == 90.0

    def test_quality_result_max_mode(self):
        """Test score calculation with MAX mode."""
        result = QualityResult(
            vmaf_score=90.0,
            details={"vmaf_max": 95.0},
            mode=QualityMode.MAX,
        )
        
        assert result.score == 95.0


class TestQualityEstimation:
    """Test quality estimation functions."""

    def test_estimate_quality_loss_acceptable(self):
        """Test quality loss estimation when acceptable."""
        acceptable, reason = estimate_quality_loss(
            original_size=1000000,
            new_size=800000,
            quality=QualityResult(vmaf_score=95.0, mode=QualityMode.NONE),
            max_loss_pct=10.0,
        )
        
        assert acceptable is True

    def test_quality_loss_unacceptable(self):
        """Test quality loss estimation when unacceptable."""
        acceptable, reason = estimate_quality_loss(
            original_size=1000000,
            new_size=500000,
            quality=QualityResult(vmaf_score=70.0, mode=QualityMode.TARGET, target=90.0),
            max_loss_pct=10.0,
        )
        
        assert acceptable is False
        assert "exceeds" in reason

    def test_quality_loss_no_quality_check(self):
        """Test quality loss without quality metrics."""
        acceptable, reason = estimate_quality_loss(
            original_size=1000000,
            new_size=800000,
            quality=None,
            max_loss_pct=10.0,
        )
        
        assert acceptable is True

    def test_quality_loss_zero_original_size(self):
        """Test quality loss with zero original size."""
        acceptable, reason = estimate_quality_loss(
            original_size=0,
            new_size=500000,
        )
        
        assert acceptable is False
        assert "zero" in reason

    @pytest.mark.integration
    def test_estimate_vmaf_quality(self, tmp_path):
        """Test VMAF quality estimation (integration test, requires FFmpeg)."""
        import subprocess
        
        # Create two test videos with different quality
        ref_file = tmp_path / "ref.mp4"
        dist_file = tmp_path / "dist.mp4"
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24",
            "-c:v", "libx264", "-crf", "10", str(ref_file)
        ], capture_output=True, check=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24",
            "-c:v", "libx264", "-crf", "28", str(dist_file)
        ], capture_output=True, check=True)
        
        result = estimate_quality_vmaf(str(ref_file), str(dist_file))
        
        # VMAF should return a score between 0 and 100
        assert 0 <= result.vmaf_score <= 100
