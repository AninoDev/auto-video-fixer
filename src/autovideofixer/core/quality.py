"""Auto Video Fixer - Video quality estimation using VMAF and other metrics."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum

from autovideofixer.core.ffmpeg_utils import run_ffmpeg


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
    # Set by estimate_ssim_psnr() (which has no VMAF score to report) so `.score`
    # has something meaningful to compare against `target` in TARGET mode --
    # without this, `.score` would fall back to vmaf_score=0.0 for that path.
    score_override: float | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    @property
    def score(self) -> float:
        """Return the score based on the quality mode."""
        if self.score_override is not None:
            return self.score_override
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


# Maps our short feature names to libvmaf's real `feature=name=...` values.
_VMAF_FEATURE_NAMES = {
    "psnr": "psnr",
    "ssim": "float_ssim",
    "ms_ssim": "float_ms_ssim",
}


def estimate_quality_vmaf(
    reference: str,
    distorted: str,
    model: str = "vmaf_v0.6.1",
    features: str = "psnr,ssim,ms_ssim,fast",
) -> QualityResult:
    """Estimate quality between reference and distorted video using VMAF.

    VMAF (Video Multi-Method Assessment Fusion) is a perceptual video quality
    metric developed by Netflix. It combines multiple metrics into a single
    score from 0-100 (higher is better). Requires an FFmpeg build with
    --enable-libvmaf.

    Args:
        reference: Path to original/high-quality reference video
        distorted: Path to processed/encoded video
        model: VMAF model version
        features: Additional metrics to compute (psnr, ssim, ms_ssim; "fast"
            is accepted for backwards compatibility but is not a real libvmaf
            feature and is ignored)

    Returns:
        QualityResult with scores
    """
    fd, json_path = tempfile.mkstemp(suffix=".json", prefix="avf_vmaf_")
    os.close(fd)

    feature_names = [
        _VMAF_FEATURE_NAMES[name]
        for f in features.split(",")
        if (name := f.strip()) in _VMAF_FEATURE_NAMES
    ]
    feature_opt = (
        ":feature=" + "|".join(f"name={name}" for name in feature_names) if feature_names else ""
    )

    try:
        # libvmaf's inputs are #0 "main" (the distorted stream) and #1 "reference" -
        # distorted must come first.
        filter_complex = (
            f"[0:v][1:v]libvmaf=log_path={json_path}:log_fmt=json:"
            f"model=version={model}{feature_opt}"
        )

        cmd = [
            "-i",
            distorted,
            "-i",
            reference,
            "-filter_complex",
            filter_complex,
            "-f",
            "null",
            "-",
        ]

        result = run_ffmpeg(cmd, timeout=600, capture_stderr=True)

        if result.returncode != 0:
            return QualityResult(
                vmaf_score=0.0,
                details={"error": "VMAF computation failed", "stderr": result.stderr[-2000:]},
            )

        scores = _parse_vmaf_json(json_path)
        if scores is None:
            return QualityResult(
                vmaf_score=0.0,
                details={"error": "Could not parse VMAF output", "stderr": result.stderr[-2000:]},
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
    """Parse libvmaf's JSON log output (log_fmt=json)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError, json.JSONDecodeError:
        return None

    pooled = data.get("pooled_metrics") if isinstance(data, dict) else None
    if not isinstance(pooled, dict) or "vmaf" not in pooled:
        return None

    def _mean(key: str) -> float:
        metric = pooled.get(key)
        return float(metric["mean"]) if isinstance(metric, dict) and "mean" in metric else 0.0

    return {
        "vmaf": _mean("vmaf"),
        "vmaf_min": float(pooled["vmaf"].get("min", 0.0)),
        "vmaf_max": float(pooled["vmaf"].get("max", 0.0)),
        "psnr_average": _mean("psnr_y"),
        "ssim_mean": _mean("float_ssim"),
        "ms_ssim_mean": _mean("float_ms_ssim"),
    }


def estimate_quality_fast(
    reference: str,
    distorted: str,
) -> dict[str, float]:
    """Quick quality estimation without full VMAF (uses PSNR/SSIM only).

    Much faster than full VMAF but less accurate perceptually.
    """
    cmd = [
        "-i",
        reference,
        "-i",
        distorted,
        "-filter_complex",
        "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr;[b][d]ssim",
        "-f",
        "null",
        "-",
    ]

    try:
        result = run_ffmpeg(cmd, capture_stderr=True)
        if result.returncode != 0:
            return {}
        return _parse_ssim_psnr_stderr(result.stderr)
    except Exception:
        return {}


def estimate_ssim_psnr(
    reference: str,
    distorted: str,
    max_frames: int = 100,
    target: float | None = None,
) -> QualityResult:
    """Estimate quality using SSIM and PSNR (no VMAF required).

    Much faster than full VMAF while still providing reliable metrics
    for encoding quality assessment. Does not compute a VMAF score -
    `vmaf_score` is left at its default (0.0); use `psnr`/`ssim` instead.

    Args:
        reference: Path to reference/original video.
        distorted: Path to processed/encoded video.
        max_frames: Maximum number of frames to compare (currently unused;
            reserved for future frame-limited comparison).
        target: If given, sets mode=QualityMode.TARGET with this target so
            meets_target() actually gates on the measured score instead of
            trivially returning True (the QualityMode.NONE default). The
            comparison score is SSIM scaled to 0-100 to match VMAF/config's
            0-100 convention (see quality_target.target in config.py).

    Returns:
        QualityResult with SSIM and PSNR scores populated (ms_ssim mirrors ssim,
        since this path does not compute a true multi-scale SSIM).
    """
    cmd = [
        "-i",
        reference,
        "-i",
        distorted,
        "-filter_complex",
        "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr;[b][d]ssim",
        "-f",
        "null",
        "-",
    ]

    mode = QualityMode.TARGET if target is not None else QualityMode.NONE

    try:
        result = run_ffmpeg(cmd, capture_stderr=True)
        if result.returncode != 0:
            return QualityResult(
                mode=mode,
                target=target or 0.0,
                details={"error": "PSNR/SSIM computation failed", "stderr": result.stderr[-2000:]},
            )

        scores = _parse_ssim_psnr_stderr(result.stderr)
        if not scores:
            return QualityResult(
                mode=mode,
                target=target or 0.0,
                details={"error": "Could not parse PSNR/SSIM output"},
            )

        ssim = scores.get("ssim", 0.0)
        return QualityResult(
            psnr=scores.get("psnr", 0.0),
            ssim=ssim,
            ms_ssim=ssim,
            mode=mode,
            target=target or 0.0,
            score_override=ssim * 100 if target is not None else None,
            details=scores,
        )

    except Exception as e:
        return QualityResult(
            mode=mode,
            target=target or 0.0,
            details={"error": str(e)},
        )


def _to_float(value: str) -> float:
    """Convert an ffmpeg metric value to float, handling 'inf'/'-inf'."""
    return float(value)


def _parse_ssim_psnr_stderr(stderr: str) -> dict[str, float]:
    """Parse PSNR and SSIM summary lines from FFmpeg stderr output.

    Real ffmpeg output (verified against ffmpeg n8.1.2):
        PSNR y:18.29 u:42.98 v:43.59 average:20.05 min:19.07 max:21.35
        SSIM Y:0.74 (5.85) U:0.98 (19.88) V:0.99 (20.50) All:0.82 (7.53)
    """
    num = r"(-?[\d.]+|inf|-inf)"
    scores: dict[str, float] = {}

    psnr_match = re.search(
        rf"PSNR\s+y:{num}\s+u:{num}\s+v:{num}\s+average:{num}\s+min:{num}\s+max:{num}",
        stderr,
    )
    if psnr_match:
        scores["psnr_y"] = _to_float(psnr_match.group(1))
        scores["psnr"] = _to_float(psnr_match.group(4))
        scores["psnr_min"] = _to_float(psnr_match.group(5))
        scores["psnr_max"] = _to_float(psnr_match.group(6))

    ssim_match = re.search(
        rf"SSIM\s+Y:{num}\s*\([^)]*\)\s+U:{num}\s*\([^)]*\)\s+V:{num}\s*\([^)]*\)\s+All:{num}",
        stderr,
    )
    if ssim_match:
        scores["ssim_y"] = _to_float(ssim_match.group(1))
        scores["ssim"] = _to_float(ssim_match.group(4))

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
