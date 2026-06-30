"""Auto Video Fixer - AI/ML module for PyTorch-based video processing.

Provides AI upscaling (Real-ESRGAN), frame interpolation (RIFE),
and AI denoising capabilities with graceful fallback to traditional
methods when PyTorch or model files are unavailable.
"""

from autovideofixer.ai.frame_processor import FrameProcessor
from autovideofixer.ai.model_cache import (
    download_model,
    ensure_model_available,
    get_model_path,
    list_available_models,
)
from autovideofixer.ai.torch_utils import (
    frame_from_tensor,
    get_device,
    is_torch_available,
    tensor_from_frame,
)

__all__ = [
    "get_device",
    "tensor_from_frame",
    "frame_from_tensor",
    "is_torch_available",
    "get_model_path",
    "ensure_model_available",
    "list_available_models",
    "download_model",
    "FrameProcessor",
]
