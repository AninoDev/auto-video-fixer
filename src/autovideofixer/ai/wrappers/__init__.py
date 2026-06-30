"""Auto Video Fixer - AI model wrappers for video processing.

Provides PyTorch-based wrappers for Real-ESRGAN (upscaling) and
RIFE (frame interpolation) models.
"""

from autovideofixer.ai.wrappers.interpolate import RIFEInterpolator
from autovideofixer.ai.wrappers.upscale import RealESRGANUpscaler

__all__ = ["RealESRGANUpscaler", "RIFEInterpolator"]
