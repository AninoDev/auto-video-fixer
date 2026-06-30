"""Tests for AI model wrappers (Real-ESRGAN, RIFE)."""

from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator
from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler


class TestRealESRGANUpscaler:
    """Test Real-ESRGAN upscaler wrapper."""

    def test_upscaler_creation(self):
        """Test creating an upscaler."""
        upscaler = RealESRGANUpscaler(scale=4, model_name="RealESRGAN_x4plus")
        assert upscaler.scale == 4
        assert upscaler.model_name == "RealESRGAN_x4plus"
        assert upscaler.is_loaded is False

    def test_upscaler_default_scale(self):
        """Test default scale factor."""
        upscaler = RealESRGANUpscaler()
        assert upscaler.scale == 4

    def test_upscaler_anime_model(self):
        """Test anime model configuration."""
        upscaler = RealESRGANUpscaler(model_name="RealESRGAN_x4plus_anime_6B")
        assert upscaler.model_name == "RealESRGAN_x4plus_anime_6B"

    def test_upscaler_not_loaded_error(self):
        """Test upscale raises when model not loaded."""
        upscaler = RealESRGANUpscaler()
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype="uint8")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            upscaler.upscale(frame)

    def test_upscaler_unload(self):
        """Test unloading model."""
        upscaler = RealESRGANUpscaler()
        upscaler._model = MagicMock()
        upscaler._loaded = True
        upscaler.unload()
        assert upscaler._model is None
        assert upscaler._loaded is False

    @patch("autovideofixer.ai.wrappers.upscale.get_model_path", return_value=None)
    def test_load_model_no_path(self, mock_path):
        """Test loading when no model path available."""
        upscaler = RealESRGANUpscaler()
        result = upscaler.load_model()
        assert result is False

    @patch("autovideofixer.ai.wrappers.upscale.get_model_path")
    def test_load_model_file_not_found(self, mock_path, tmp_path):
        """Test loading when model file doesn't exist."""
        mock_path.return_value = tmp_path / "nonexistent.pth"
        upscaler = RealESRGANUpscaler()
        result = upscaler.load_model()
        assert result is False


class TestRIFEInterpolator:
    """Test RIFE frame interpolator wrapper."""

    def test_interpolator_creation(self):
        """Test creating an interpolator."""
        interp = RIFEInterpolator(model_name="rife_v4.6")
        assert interp.model_name == "rife_v4.6"
        assert interp.is_loaded is False

    def test_interpolator_not_loaded_error(self):
        """Test interpolation raises when model not loaded."""
        interp = RIFEInterpolator()
        import numpy as np

        frame0 = np.zeros((240, 320, 3), dtype="uint8")
        frame1 = np.zeros((240, 320, 3), dtype="uint8")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            interp.interpolate(frame0, frame1)

    def test_interpolator_unload(self):
        """Test unloading model."""
        interp = RIFEInterpolator()
        interp._model = MagicMock()
        interp._loaded = True
        interp.unload()
        assert interp._model is None
        assert interp._loaded is False

    @patch("autovideofixer.ai.wrappers.interpolate.get_model_path", return_value=None)
    def test_load_model_no_path(self, mock_path):
        """Test loading when no model path available."""
        interp = RIFEInterpolator()
        result = interp.load_model()
        assert result is False

    @patch("autovideofixer.ai.wrappers.interpolate.get_model_path")
    def test_load_model_file_not_found(self, mock_path, tmp_path):
        """Test loading when model file doesn't exist."""
        mock_path.return_value = tmp_path / "nonexistent.pkl"
        interp = RIFEInterpolator()
        result = interp.load_model()
        assert result is False

    def test_interpolate_video_single_frame(self):
        """Test interpolation with single frame (factor=2)."""
        interp = RIFEInterpolator()
        interp._loaded = True
        interp._model = MagicMock()
        interp._device = MagicMock()
        interp._device.type = "cpu"

        import numpy as np

        frames = [np.zeros((240, 320, 3), dtype="uint8")]

        # Mock the interpolate method
        interp.interpolate = MagicMock(return_value=np.zeros((240, 320, 3), dtype="uint8"))

        result = interp.interpolate_video(frames, factor=2)
        # With 1 frame and factor 2, we expect at least the original frame
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_interpolate_video_factor_1(self):
        """Test interpolation with factor 1 returns original frames."""
        interp = RIFEInterpolator()
        interp._loaded = True
        interp._model = MagicMock()
        interp._device = MagicMock()
        interp._device.type = "cpu"

        import numpy as np

        frames = [np.zeros((240, 320, 3), dtype="uint8")]
        result = interp.interpolate_video(frames, factor=1)
        assert len(result) == 1
