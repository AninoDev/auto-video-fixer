"""Auto Video Fixer - Real-ESRGAN upscaling model wrapper.

Implements the RRDBNet architecture used by Real-ESRGAN for
single-image super-resolution, adapted for video processing.

Architecture matches the official BasicSR implementation:
https://github.com/xinntao/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from autovideofixer.ai.model_cache import get_model_path

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.upscale")
    return _logger


class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block as used in Real-ESRGAN.

    Contains 5 conv layers with dense feature reuse.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 + x


class RRDB(torch.nn.Module):
    """Residual-in-Residual Dense Block.

    Contains 3 ResidualDenseBlocks in series with 0.2 scaling on residual.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    """RRDBNet architecture for Real-ESRGAN.

    Matches the official BasicSR implementation with:
    - conv_first: initial 3x3 conv
    - body: 23 RRDB blocks
    - conv_body: 3x3 conv on body output
    - conv_up1/conv_up2: upsampling stages
    - conv_hr: high-resolution feature
    - conv_last: final output conv
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample x4 using two x2 stages
        feat = torch.nn.functional.interpolate(
            self.lrelu(self.conv_up1(feat)), scale_factor=2, mode="nearest"
        )
        feat = torch.nn.functional.interpolate(
            self.lrelu(self.conv_up2(feat)), scale_factor=2, mode="nearest"
        )

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RealESRGANUpscaler:
    """Real-ESRGAN model wrapper for video upscaling.

    Wraps the RRDBNet architecture with frame-by-frame inference,
    proper color space handling, and optional test-time augmentation.

    Usage:
        upscaler = RealESRGANUpscaler(scale=4)
        upscaler.load_model("/path/to/RealESRGAN_x4plus.pth")
        result = upscaler.upscale(frame, tta_mode=7)
    """

    def __init__(
        self,
        scale: int = 4,
        model_name: str = "RealESRGAN_x4plus",
        tta_mode: int = 0,
        batch_size: int = 1,
    ):
        self.scale = scale
        self.model_name = model_name
        self.tta_mode = tta_mode
        self.batch_size = batch_size
        self._model: Any = None
        self._device: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None) -> bool:
        """Load the Real-ESRGAN model from disk.

        Args:
            model_path: Path to .pth file. If None, uses cached model.

        Returns:
            True if model loaded successfully.
        """
        try:
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                _get_logger().error("PyTorch not installed - cannot load model")
                return False
        except ImportError:
            _get_logger().error("PyTorch not installed - cannot load model")
            return False

        from autovideofixer.ai.torch_utils import get_device, load_model_from_state_dict

        if model_path is None:
            model_path = get_model_path(self.model_name)
            if model_path is None:
                _get_logger().error(
                    f"Model not found: {self.model_name}. Run ensure_model_available() first."
                )
                return False
            model_path = str(model_path)

        if not Path(model_path).is_file():
            _get_logger().error(f"Model file not found: {model_path}")
            return False

        self._device = get_device("auto")

        # Determine num_block based on model name
        num_block = 6 if "6B" in self.model_name else 23
        num_feat = 64

        self._model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=32,
            scale=self.scale,
        )

        self._model = load_model_from_state_dict(self._model, model_path, self._device)
        self._loaded = True

        _get_logger().info(f"Loaded {self.model_name} ({num_block} RRDB blocks) on {self._device}")
        return True

    def upscale(
        self,
        frame: Any,  # numpy array (H, W, 3) BGR uint8
        tta_mode: int | None = None,
    ) -> Any:
        """Upscale a single frame using Real-ESRGAN.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR, uint8.
            tta_mode: Test-time augmentation mode (0=off, 1, 2, 4, 7, 8, 15, etc.).
                      If None, uses the configured tta_mode.

        Returns:
            Upscaled frame as numpy array (H*scale, W*scale, 3) in BGR, uint8.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        from autovideofixer.ai.torch_utils import (
            apply_tta,
            frame_from_tensor,
            get_dtype,
            tensor_from_frame,
        )

        if tta_mode is None:
            tta_mode = self.tta_mode

        # Use fp16 for CUDA if available, otherwise fp32
        use_fp16 = self._device.type == "cuda"
        if use_fp16:
            self._model.half()

        tensor = tensor_from_frame(frame, device=self._device)
        if use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            output = self._model(tensor)
            if tta_mode and tta_mode >= 1:
                output = apply_tta(self._model, tensor, mode=tta_mode)

        if use_fp16:
            output = output.float()
            self._model.float()

        # Model already handles upscaling, so don't pass scale to frame_from_tensor
        result = frame_from_tensor(output, scale=1.0)
        return result

    def upscale_video(
        self,
        frames: list[Any],
        progress_callback=None,
    ) -> list[Any]:
        """Upscale a sequence of frames.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8.
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of upscaled numpy arrays.
        """
        results = []
        total = len(frames)

        for i, frame in enumerate(frames):
            result = self.upscale(frame)
            results.append(result)

            if progress_callback and total > 0:
                progress_callback(i + 1, total, f"Upscaling frame {i + 1}/{total}")

        return results

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
