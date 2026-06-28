"""Auto Video Fixer - Video quality estimation using VMAF and other metrics."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum

from autovideofixer.core.ffmpeg_utils import get_ffmpeg_path, run_ffmpeg


class QualityMode(Enum):
    NONE = "none"
    MIN = "min"
    AVG = "avg"
    MAX = "max"
    TARGET = "target"


@dataclass
class QualityResult:
    """Result of quality estimation."""

    vmaf_score: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    ms_ssim: float = 0.0
    mode: QualityMode = QualityMode.NONE
    target: float = 0.0
    acceptable: bool = True
    details: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    @property
    def score(self) -> float:
        """Return the score based on the quality mode."""
        if self.mode == QualityMode.NONE:
            return self.vmaf_score
        elif self.mode == QualityMode.MIN:
            return self.details.get("vmaf_min", self.vmaf_score)
        elif self.mode == QualityMode.AVG:
            return self.details.get("vmaf", self.vmaf_score)
        elif self.mode == QualityMode.MAX:
            return self.details.get("vmaf_max", self.vmaf_score)
        elif self.mode == QualityMode.TARGET:
            return self.vmaf_score
        return self.vmaf_score

    def meets_target(self) -> bool:
        """Check if quality meets the configured target."""
        if self.mode == QualityMode.NONE:
            return True
        return self.score >= self.target


def estimate_quality_vmaf(
    reference: str,
    distorted: str,
    model: str = "vmaf_v0.6.1",
    features: str = "psnr,ssim,ms_ssim,fast",
) -> QualityResult:
    """Estimate quality between reference and distorted video using VMAF.

    VMAF (Video Multi-Method Assessment Fusion) is a perceptual video quality
    metric developed by Netflix. It combines multiple metrics into a single
    score from 0-100 (higher is better).

    Args:
        reference: Path to original/high-quality reference video
        distorted: Path to processed/encoded video
        model: VMAF model version
        features: Additional metrics to compute

    Returns:
        QualityResult with scores
    """
    ffmpeg = get_ffmpeg_path()

    fd, json_path = tempfile.mkstemp(suffix=".json", prefix="avf_vmaf_")
    os.close(fd)

    try:
        feature_flags = ":".join(f"{f.strip()}=1" for f in features.split(","))
        filter_complex = f"[0:v][1:v]vmaf=model={model}:{feature_flags}:result={json_path}"

        cmd = [
            ffmpeg,
            "-i",
            reference,
            "-i",
            distorted,
            "-filter_complex",
            filter_complex,
            "-f",
            "null",
            "-",
        ]

        run_ffmpeg(cmd, timeout=600)
        scores = _parse_vmaf_json(json_path)

        if scores is None:
            return QualityResult(
                vmaf_score=0.0,
                psnr=0.0,
                ssim=0.0,
                ms_ssim=0.0,
                details={"error": "VMAF computation failed"},
            )

        return QualityResult(
            vmaf_score=scores.get("vmaf", 0.0),
            psnr=scores.get("psnr_average", 0.0),
            ssim=scores.get("ssim_mean", 0.0),
            ms_ssim=scores.get("ms_ssim_mean", 0.0),
            details=scores,
        )

    except Exception as e:
        return QualityResult(
            vmaf_score=0.0,
            details={"error": str(e)},
        )
    finally:
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except OSError:
                pass


def _parse_vmaf_json(path: str) -> dict[str, float] | None:
    """Parse VMAF JSON output file."""
    import json

    try:
        with open(path) as f:
            data = json.load(f)

        # VMAF can output either a single score or a list of frame scores
        if isinstance(data, dict):
            return {
                "vmaf": data.get("mean", data.get("vmaf", 0.0)),
                "vmaf_min": data.get("min", data.get("vmaf", 0.0)),
                "vmaf_max": data.get("max", data.get("vmaf", 0.0)),
                "psnr_average": data.get("psnr_average", 0.0),
                "ssim_mean": data.get("ssim_mean", 0.0),
                "ms_ssim_mean": data.get("ms_ssim_mean", 0.0),
            }
        elif isinstance(data, list):
            scores = [d.get("vmaf", 0.0) for d in data if isinstance(d, dict)]
            if not scores:
                return None
            return {
                "vmaf": sum(scores) / len(scores),
                "vmaf_min": min(scores),
                "vmaf_max": max(scores),
            }
        return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def estimate_quality_fast(
    reference: str,
    distorted: str,
) -> dict[str, float]:
    """Quick quality estimation without full VMAF (uses PSNR/SSIM only).

    Much faster than full VMAF but less accurate perceptually.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-i",
        reference,
        "-i",
        distorted,
        "-filter_complex",
        "psnr,ssim",
        "-f",
        "null",
        "-",
    ]

    try:
        result = run_ffmpeg(cmd, capture_stderr=True)
        scores = _parse_fast_metrics(result.stderr)
        return scores
    except Exception:
        return {}


def _parse_fast_metrics(stderr: str) -> dict[str, float]:
    """Parse PSNR/SSIM from FFmpeg stderr."""
    import re

    scores: dict[str, float] = {}

    psnr_match = re.search(r"PSNR.*?=\s*([\d.]+)", stderr)
    if psnr_match:
        scores["psnr"] = float(psnr_match.group(1))

    ssim_match = re.search(r"SSIM.*?=\s*([\d.]+)", stderr)
    if ssim_match:
        scores["ssim"] = float(ssim_match.group(1))

    return scores


def estimate_quality_loss(
    original_size: int,
    new_size: int,
    quality: QualityResult | None = None,
    max_loss_pct: float = 5.0,
) -> tuple[bool, str]:
    """Estimate if quality loss is acceptable for the file size reduction.

    Args:
        original_size: Original file size in bytes
        new_size: New file size in bytes
        quality: Quality metrics result (optional)
        max_loss_pct: Maximum acceptable quality loss percentage

    Returns:
        (is_acceptable, reason)
    """
    if original_size == 0:
        return False, "Original size is zero"

    size_reduction = (original_size - new_size) / original_size * 100

    if quality is None or quality.mode == QualityMode.NONE:
        return True, f"Size reduced by {size_reduction:.1f}% (no quality check)"

    quality_loss = 100.0 - quality.score
    if quality_loss > max_loss_pct:
        return False, f"Quality loss {quality_loss:.1f}% exceeds max {max_loss_pct}%"

    return (
        True,
        f"Quality loss {quality_loss:.1f}% within limit, size reduced {size_reduction:.1f}%",
    )
