"""Tests for AI torch utilities."""

import sys
from unittest.mock import MagicMock, patch

from autovideofixer.ai.torch_utils import (
    frame_from_tensor,
    get_device,
    get_dtype,
    is_torch_available,
    tensor_from_frame,
)


class TestIsTorchAvailable:
    """Test PyTorch availability detection."""

    def test_torch_available(self):
        """Test detection when PyTorch is installed."""
        # PyTorch is in the optional deps; if installed, should return True
        result = is_torch_available()
        # Just verify it doesn't raise
        assert isinstance(result, bool)

    def test_torch_not_available(self):
        """Test detection when PyTorch is not installed."""
        with patch.dict(sys.modules, {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = is_torch_available()
                assert result is False


class TestGetDevice:
    """Test device detection."""

    def test_get_device_auto(self):
        """Test get_device auto-detection."""
        device = get_device("auto")
        assert device is not None

    def test_get_dtype(self):
        """Test dtype selection."""
        dtype_fp32 = get_dtype("fp32")
        assert dtype_fp32 is not None

        dtype_fp16 = get_dtype("fp16")
        assert dtype_fp16 is not None

    def test_get_device_cpu(self):
        """Test get_device CPU."""
        device = get_device("cpu")
        assert str(device) == "cpu"

    def test_tensor_from_frame_no_torch(self):
        """Test tensor_from_frame with mocked torch."""
        import numpy as np

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.float32 = "float32"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("autovideofixer.ai.torch_utils.is_torch_available", return_value=True):
                frame = np.zeros((240, 320, 3), dtype="uint8")
                result = tensor_from_frame(frame)
                assert result is not None

    def test_frame_from_tensor_no_torch(self):
        """Test frame_from_tensor with mocked torch."""
        mock_torch = MagicMock()
        mock_torch.nn.functional.interpolate.return_value = MagicMock()
        mock_torch.detach = MagicMock()

        mock_arr = MagicMock()
        mock_arr.squeeze.return_value = MagicMock()
        mock_arr.transpose.return_value = MagicMock()
        mock_arr.numpy.return_value = MagicMock()
        mock_arr.clip.return_value = MagicMock()
        mock_torch.from_numpy = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("autovideofixer.ai.torch_utils.is_torch_available", return_value=True):
                tensor = MagicMock()
                tensor.detach.return_value.cpu.return_value = MagicMock()
                tensor.squeeze.return_value.numpy.return_value.transpose.return_value = MagicMock()
                result = frame_from_tensor(tensor)
                assert result is not None
