"""Unit tests for StabilizeStage._build_sharpen_suffix() (Feature 7).

Covers the configurable post-stabilization unsharp filter: default output,
disabled sharpening, custom values, and validation of ffmpeg's unsharp
constraints (odd matrix sizes in [3, 63], amounts in [-2.0, 5.0]).
"""

from __future__ import annotations

import pytest

from autovideofixer.config import Config
from autovideofixer.core.stages.stabilize import StabilizeStage


def _make_stage(tmp_path, overrides: dict | None = None) -> StabilizeStage:
    config = Config(tmp_path / "nonexistent.yaml")
    return StabilizeStage(config, overrides=overrides)


class TestBuildSharpenSuffix:
    def test_default_config(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage._build_sharpen_suffix() == ",unsharp=3:3:1:3:3:0"

    def test_disabled_returns_empty_string(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_enabled": False})
        assert stage._build_sharpen_suffix() == ""

    def test_custom_values(self, tmp_path):
        stage = _make_stage(
            tmp_path,
            {"sharpen_amount": 2.5, "sharpen_luma_size": 5},
        )
        assert stage._build_sharpen_suffix() == ",unsharp=5:5:2.5:3:3:0"

    def test_custom_chroma_values(self, tmp_path):
        stage = _make_stage(
            tmp_path,
            {"sharpen_chroma_amount": -1.5, "sharpen_chroma_size": 7},
        )
        assert stage._build_sharpen_suffix() == ",unsharp=3:3:1:7:7:-1.5"

    def test_even_luma_size_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_luma_size": 4})
        with pytest.raises(ValueError, match="sharpen_luma_size"):
            stage._build_sharpen_suffix()

    def test_luma_size_too_small_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_luma_size": 1})
        with pytest.raises(ValueError, match="sharpen_luma_size"):
            stage._build_sharpen_suffix()

    def test_luma_size_too_large_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_luma_size": 65})
        with pytest.raises(ValueError, match="sharpen_luma_size"):
            stage._build_sharpen_suffix()

    def test_chroma_size_out_of_range_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_chroma_size": 65})
        with pytest.raises(ValueError, match="sharpen_chroma_size"):
            stage._build_sharpen_suffix()

    def test_amount_too_large_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_amount": 6.0})
        with pytest.raises(ValueError, match="sharpen_amount"):
            stage._build_sharpen_suffix()

    def test_amount_too_small_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_amount": -2.5})
        with pytest.raises(ValueError, match="sharpen_amount"):
            stage._build_sharpen_suffix()

    def test_chroma_amount_out_of_range_raises(self, tmp_path):
        stage = _make_stage(tmp_path, {"sharpen_chroma_amount": 10.0})
        with pytest.raises(ValueError, match="sharpen_chroma_amount"):
            stage._build_sharpen_suffix()
