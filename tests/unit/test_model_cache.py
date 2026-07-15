"""Tests for AI model cache management."""

from pathlib import Path
from unittest.mock import patch

from autovideofixer.ai.model_cache import (
    MODEL_REGISTRY,
    clear_model_cache,
    download_model,
    ensure_model_available,
    get_model_dir,
    get_model_hash,
    get_model_path,
    list_available_models,
    list_cached_models,
)


class TestModelRegistry:
    """Test model registry contents."""

    def test_registry_has_models(self):
        """Test that the model registry has expected entries."""
        assert len(MODEL_REGISTRY) > 0
        assert "RealESRGAN_x4plus" in MODEL_REGISTRY
        assert "rife_v4.6" in MODEL_REGISTRY

    def test_model_metadata(self):
        """Test model metadata structure."""
        for name, meta in MODEL_REGISTRY.items():
            assert "url" in meta, f"Missing 'url' for {name}"
            assert "filename" in meta, f"Missing 'filename' for {name}"
            assert "description" in meta, f"Missing 'description' for {name}"
            assert meta["url"].startswith("https://"), f"Invalid URL for {name}"

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "RealESRGAN_x4plus" in models


class TestCompactSrvggRegistry:
    """Test the compact SRVGG (Real-ESRGAN v0.2.5.0) model registry entries."""

    COMPACT_MODEL_NAMES = (
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
        "realesr-animevideov3",
    )

    def test_compact_models_present(self):
        """All three compact SRVGG models are registered."""
        for name in self.COMPACT_MODEL_NAMES:
            assert name in MODEL_REGISTRY, f"Missing registry entry for {name}"

    def test_compact_models_well_formed(self):
        """Each compact model entry has a valid https URL, sha256, scale, and arch."""
        for name in self.COMPACT_MODEL_NAMES:
            meta = MODEL_REGISTRY[name]
            assert meta["url"].startswith("https://"), f"Non-https URL for {name}"
            sha256 = meta["sha256"]
            assert len(sha256) == 64, f"sha256 for {name} is not 64 chars: {sha256!r}"
            assert sha256 == sha256.lower(), f"sha256 for {name} is not lowercase: {sha256!r}"
            assert all(c in "0123456789abcdef" for c in sha256), (
                f"sha256 for {name} is not lowercase hex: {sha256!r}"
            )
            assert meta["scale"] == 4, f"Unexpected scale for {name}: {meta['scale']}"
            assert meta["arch"] == "srvgg", f"Unexpected arch for {name}: {meta['arch']}"
            assert meta["num_conv"] in (16, 32), (
                f"Unexpected num_conv for {name}: {meta['num_conv']}"
            )

    def test_general_x4v3_and_wdn_use_32_convs(self):
        """The two general-purpose compact models both use num_conv=32."""
        assert MODEL_REGISTRY["realesr-general-x4v3"]["num_conv"] == 32
        assert MODEL_REGISTRY["realesr-general-wdn-x4v3"]["num_conv"] == 32

    def test_animevideov3_uses_16_convs(self):
        """The anime compact model uses the smaller num_conv=16 variant."""
        assert MODEL_REGISTRY["realesr-animevideov3"]["num_conv"] == 16

    def test_compact_models_listed_as_available(self):
        """Compact models show up via list_available_models() like any other model."""
        models = list_available_models()
        for name in self.COMPACT_MODEL_NAMES:
            assert name in models

    def test_rrdb_models_unaffected_by_arch_field(self):
        """Existing RRDB-based models have no 'arch' field and must not have been touched."""
        for name in ("RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B"):
            meta = MODEL_REGISTRY[name]
            assert "arch" not in meta, f"{name} unexpectedly gained an 'arch' field"
            assert "num_conv" not in meta, f"{name} unexpectedly gained a 'num_conv' field"


class TestModelPath:
    """Test model path resolution."""

    def test_get_model_path_unknown(self):
        """Test getting path for unknown model."""
        result = get_model_path("nonexistent_model_xyz")
        assert result is None

    def test_get_model_path_not_cached(self, tmp_path):
        """Test getting path when model not cached (and not in project dir)."""
        with patch("autovideofixer.ai.model_cache.get_model_dir", return_value=tmp_path):
            result = get_model_path("RealESRGAN_x4plus")
            assert result is None


class TestModelDir:
    """Test model directory operations."""

    def test_get_model_dir(self):
        """Test model directory creation."""
        with patch("autovideofixer.ai.model_cache.get_data_dir") as mock_data:
            mock_data.return_value = Path("/tmp/test_avf_data")
            model_dir = get_model_dir()
            assert isinstance(model_dir, Path)


class TestModelDownload:
    """Test model download functionality."""

    def test_download_unknown_model(self):
        """Test downloading unknown model fails gracefully."""
        success, msg = download_model("totally_fake_model_xyz")
        assert success is False
        assert "Unknown model" in msg

    def test_ensure_model_with_custom_path(self, tmp_path):
        """Test ensure_model_available with valid custom path."""
        model_file = tmp_path / "test_model.pth"
        model_file.write_text("fake model weights")

        success, msg = ensure_model_available("fake", model_path=str(model_file))
        assert success is True

    def test_ensure_model_with_invalid_custom_path(self):
        """Test ensure_model_available with invalid custom path."""
        success, msg = ensure_model_available("fake", model_path="/nonexistent/model.pth")
        assert success is False
        assert "not found" in msg


class TestModelCache:
    """Test model cache operations."""

    def test_list_cached_models_empty(self, tmp_path):
        """Test listing cached models when cache is empty."""
        with patch("autovideofixer.ai.model_cache.get_model_dir", return_value=tmp_path):
            cached = list_cached_models()
            assert isinstance(cached, list)
            assert len(cached) == 0

    def test_model_hash_nonexistent(self):
        """Test computing hash for nonexistent file."""
        result = get_model_hash("/nonexistent/file.pth")
        assert result is None

    def test_model_hash_existing(self, tmp_path):
        """Test computing hash for existing file."""
        f = tmp_path / "test.pth"
        f.write_bytes(b"test data for hashing")
        result = get_model_hash(str(f))
        assert result is not None
        assert len(result) == 64  # SHA256 hex length

    def test_clear_cache_empty(self, tmp_path):
        """Test clearing empty cache."""
        with patch("autovideofixer.ai.model_cache.get_model_dir", return_value=tmp_path):
            removed = clear_model_cache()
            assert removed == 0

    def test_clear_specific_model(self, tmp_path):
        """Test clearing a specific model."""
        # Create a model file in the mock cache dir
        (tmp_path / "RealESRGAN_x4plus.pth").write_bytes(b"fake")
        with patch("autovideofixer.ai.model_cache.get_model_dir", return_value=tmp_path):
            removed = clear_model_cache("RealESRGAN_x4plus")
            assert removed == 1
            assert not (tmp_path / "RealESRGAN_x4plus.pth").exists()
