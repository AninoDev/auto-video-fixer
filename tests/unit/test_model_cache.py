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
