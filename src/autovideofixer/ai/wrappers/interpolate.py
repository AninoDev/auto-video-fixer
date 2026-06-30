"""Auto Video Fixer - RIFE frame interpolation model wrapper.

Implements the RIFE (Real-Time Intermediate Flow Estimation) architecture
for video frame interpolation using bidirectional optical flow.
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
        _logger = logging.getLogger("autovideofixer.ai.interpolate")
    return _logger


class ConvBlock(torch.nn.Module):
    """Convolutional block with LeakyReLU."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.conv(x))


class DownBlock(torch.nn.Module):
    """Downsampling block with strided convolutions."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, stride=2, padding=1)
        self.conv2 = ConvBlock(out_ch, out_ch, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UpBlock(torch.nn.Module):
    """Upsampling block with bilinear interpolation."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv_skip = ConvBlock(skip_ch, out_ch, padding=1)
        self.conv_in = ConvBlock(in_ch, out_ch * 2, padding=1)
        self.conv_out = ConvBlock(out_ch * 2, out_ch, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        skip = self.conv_skip(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv_out(self.conv_in(x))


class CNet(torch.nn.Module):
    """Context network for multi-scale feature extraction.

    Extracts hierarchical features at 5 scales for flow estimation.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 10):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            ConvBlock(in_ch, base_ch, padding=1),
            ConvBlock(base_ch, base_ch, padding=1),
        )
        self.block2 = DownBlock(base_ch, base_ch * 2)
        self.block3 = DownBlock(base_ch * 2, base_ch * 4)
        self.block4 = DownBlock(base_ch * 4, base_ch * 8)
        self.block5 = DownBlock(base_ch * 8, base_ch * 16)
        self.block6 = DownBlock(base_ch * 16, base_ch * 32)

        self.output_channels = [
            base_ch,
            base_ch * 2,
            base_ch * 4,
            base_ch * 8,
            base_ch * 16,
            base_ch * 32,
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        h = self.block1(x)
        features.append(h)
        h = self.block2(h)
        features.append(h)
        h = self.block3(h)
        features.append(h)
        h = self.block4(h)
        features.append(h)
        h = self.block5(h)
        features.append(h)
        h = self.block6(h)
        features.append(h)
        return features


class IFNet(torch.nn.Module):
    """Intermediate Flow Network for estimating optical flow.

    Uses multi-scale feature pyramid from CNet to estimate
    bidirectional flow fields at multiple resolutions.
    """

    def __init__(self, cnet: CNet):
        super().__init__()
        self.cnet = cnet
        ch_list = cnet.output_channels

        # Flow estimation at each scale
        self.flow_conv1 = torch.nn.Sequential(
            ConvBlock(ch_list[0] * 2, ch_list[0], padding=1),
            ConvBlock(ch_list[0], 2, kernel_size=3, padding=1),  # 2 channels: flow (dx, dy)
        )
        self.flow_conv2 = torch.nn.Sequential(
            ConvBlock(ch_list[1] * 2 + 2, ch_list[1], padding=1),
            ConvBlock(ch_list[1], 2, kernel_size=3, padding=1),
        )
        self.flow_conv3 = torch.nn.Sequential(
            ConvBlock(ch_list[2] * 2 + 2, ch_list[2], padding=1),
            ConvBlock(ch_list[2], 2, kernel_size=3, padding=1),
        )
        self.flow_conv4 = torch.nn.Sequential(
            ConvBlock(ch_list[3] * 2 + 2, ch_list[3], padding=1),
            ConvBlock(ch_list[3], 2, kernel_size=3, padding=1),
        )
        self.flow_conv5 = torch.nn.Sequential(
            ConvBlock(ch_list[4] * 2 + 2, ch_list[4], padding=1),
            ConvBlock(ch_list[4], 2, kernel_size=3, padding=1),
        )

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate bidirectional flow fields.

        Returns:
            (flow01, flow10): Forward (img0->img1) and backward (img1->img0) flow fields.
        """
        feats0 = self.cnet(img0)
        feats1 = self.cnet(img1)

        # Top-scale flow estimation
        flow01 = self._est_flow(feats0[0], feats1[0], self.flow_conv1)
        flow10 = self._est_flow(feats1[0], feats0[0], self.flow_conv1)

        # Propagate flow to lower scales
        up_flow01 = (
            torch.nn.functional.interpolate(
                flow01, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )
        up_flow10 = (
            torch.nn.functional.interpolate(
                flow10, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )

        flow01 = flow01 + self._est_flow(
            feats0[1], feats1[1], self.flow_conv2, up_flow01, up_flow10
        )
        flow10 = flow10 + self._est_flow(
            feats1[1], feats0[1], self.flow_conv2, up_flow10, up_flow01
        )

        up_flow01 = (
            torch.nn.functional.interpolate(
                flow01, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )
        up_flow10 = (
            torch.nn.functional.interpolate(
                flow10, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )

        flow01 = flow01 + self._est_flow(
            feats0[2], feats1[2], self.flow_conv3, up_flow01, up_flow10
        )
        flow10 = flow10 + self._est_flow(
            feats1[2], feats0[2], self.flow_conv3, up_flow10, up_flow01
        )

        up_flow01 = (
            torch.nn.functional.interpolate(
                flow01, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )
        up_flow10 = (
            torch.nn.functional.interpolate(
                flow10, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )

        flow01 = flow01 + self._est_flow(
            feats0[3], feats1[3], self.flow_conv4, up_flow01, up_flow10
        )
        flow10 = flow10 + self._est_flow(
            feats1[3], feats0[3], self.flow_conv4, up_flow10, up_flow01
        )

        up_flow01 = (
            torch.nn.functional.interpolate(
                flow01, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )
        up_flow10 = (
            torch.nn.functional.interpolate(
                flow10, scale_factor=2, mode="bilinear", align_corners=False
            )
            * 2
        )

        flow01 = flow01 + self._est_flow(
            feats0[4], feats1[4], self.flow_conv5, up_flow01, up_flow10
        )
        flow10 = flow10 + self._est_flow(
            feats1[4], feats0[4], self.flow_conv5, up_flow10, up_flow01
        )

        return flow01, flow10

    def _est_flow(
        self,
        conv: torch.nn.Module,
        f0: torch.Tensor,
        f1: torch.Tensor,
        flow_conv: torch.nn.Module,
        prev_flow01: torch.Tensor | None = None,
        prev_flow10: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_flow01 is not None and prev_flow10 is not None:
            input_tensor = torch.cat([f0, f1, prev_flow01, prev_flow10], dim=1)
        else:
            input_tensor = torch.cat([f0, f1], dim=1)
        return flow_conv(input_tensor)


class EMD(torch.nn.Module):
    """Error Minimization Network for refining interpolated frames.

    Takes warped features and flow fields, then refines the output
    by minimizing reconstruction error.
    """

    def __init__(self, cnet: CNet):
        super().__init__()
        ch_list = cnet.output_channels
        self.block1 = torch.nn.Sequential(
            ConvBlock(ch_list[0] * 2 + 4, ch_list[0], padding=1),
            ConvBlock(ch_list[0], ch_list[0], padding=1),
        )
        self.block2 = torch.nn.Sequential(
            ConvBlock(ch_list[1] * 2 + 4, ch_list[1] * 2, padding=1),
            ConvBlock(ch_list[1] * 2, ch_list[1] * 2, padding=1),
        )
        self.up1 = UpBlock(ch_list[0], ch_list[1] * 2, ch_list[0])
        self.up2 = UpBlock(ch_list[0], ch_list[2] * 4, ch_list[0])
        self.conv_out = torch.nn.Sequential(
            ConvBlock(ch_list[0], ch_list[0], padding=1),
            torch.nn.Conv2d(ch_list[0], 3, 3, 1, 1),
        )

    def forward(
        self,
        feats0: list[torch.Tensor],
        feats1: list[torch.Tensor],
        flow01: torch.Tensor,
        flow10: torch.Tensor,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> torch.Tensor:
        """Refine the interpolated frame.

        Args:
            feats0, feats1: Multi-scale features from CNet.
            flow01, flow10: Bidirectional flow fields.
            img0, img1: Original input images.

        Returns:
            Refined output frame tensor.
        """
        # Warp features using flow
        warped0 = self._warp(feats0[0], flow01)
        warped1 = self._warp(feats1[0], flow10)

        # Top-scale processing
        h = self.block1(torch.cat([warped0, warped1, flow01, flow10], dim=1))
        h = self.up1(
            h,
            self.block2(
                torch.cat(
                    [
                        self._warp(feats1[1], _scale_flow(flow10, 2)),
                        self._warp(feats0[1], _scale_flow(flow01, 2)),
                        _scale_flow(flow01, 2),
                        _scale_flow(flow10, 2),
                    ],
                    dim=1,
                )
            ),
        )
        h = self.up2(
            h,
            torch.cat(
                [
                    self._warp(feats0[2], _scale_flow(flow01, 4)),
                    self._warp(feats1[2], _scale_flow(flow10, 4)),
                ],
                dim=1,
            ),
        )

        return torch.sigmoid(self.conv_out(h))

    def _warp(self, feats: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp features using optical flow (grid_sample)."""
        return torch.nn.functional.grid_sample(
            feats,
            flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


def _scale_flow(flow: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a flow field by a factor."""
    return (
        torch.nn.functional.interpolate(
            flow, scale_factor=scale, mode="bilinear", align_corners=False
        )
        * scale
    )


class RIFEModule(torch.nn.Module):
    """RIFE interpolation module combining IFNet + EMD.

    Takes two input frames and produces an interpolated frame.
    """

    def __init__(self, cnet: CNet | None = None):
        super().__init__()
        if cnet is None:
            cnet = CNet()
        self.ifnet = IFNet(cnet)
        self.emd = EMD(cnet)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> torch.Tensor:
        """Generate an interpolated frame between img0 and img1.

        Args:
            img0: First frame (B, 3, H, W), values in [0, 1].
            img1: Second frame (B, 3, H, W), values in [0, 1].

        Returns:
            Interpolated frame (B, 3, H, W), values in [0, 1].
        """
        flow01, flow10 = self.ifnet(img0, img1)
        return self.emd(
            self.ifnet.cnet(img0),
            self.ifnet.cnet(img1),
            flow01,
            flow10,
            img0,
            img1,
        )


class RIFEInterpolator:
    """RIFE model wrapper for video frame interpolation.

    Wraps the RIFE architecture with frame-pair inference and
    support for arbitrary interpolation factors.

    Usage:
        interpolator = RIFEInterpolator()
        interpolator.load_model("/path/to/rife_v4.6.pkl")
        result = interpolator.interpolate(frame_a, frame_b)
    """

    def __init__(
        self,
        model_name: str = "rife_v4.6",
        multi_scale: bool = True,
    ):
        self.model_name = model_name
        self.multi_scale = multi_scale
        self._model: RIFEModule | None = None
        self._device: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None) -> bool:
        """Load the RIFE model from disk.

        Args:
            model_path: Path to .pkl model file. If None, uses cached model.

        Returns:
            True if model loaded successfully.
        """
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
        self._model = RIFEModule()

        self._model = load_model_from_state_dict(self._model, model_path, self._device)
        self._loaded = True

        _get_logger().info(f"Loaded RIFE {self.model_name} on {self._device}")
        return True

    def interpolate(
        self,
        frame0: Any,
        frame1: Any,
    ) -> Any:
        """Interpolate a single frame between two input frames.

        Args:
            frame0: First frame as numpy array (H, W, 3) BGR uint8.
            frame1: Second frame as numpy array (H, W, 3) BGR uint8.

        Returns:
            Interpolated frame as numpy array (H, W, 3) in BGR, uint8.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from autovideofixer.ai.torch_utils import (
            frame_from_tensor,
            get_dtype,
            tensor_from_frame,
        )

        dtype = get_dtype("fp16") if self._device.type == "cuda" else None

        t0 = tensor_from_frame(frame0, device=self._device, dtype=dtype)
        t1 = tensor_from_frame(frame1, device=self._device, dtype=dtype)

        with torch.no_grad():
            output = self._model(t0, t1)
            if dtype is not None:
                output = output.float()

        return frame_from_tensor(output)

    def interpolate_video(
        self,
        frames: list[Any],
        factor: int = 2,
        progress_callback=None,
    ) -> list[Any]:
        """Interpolate frames to increase framerate by a given factor.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8.
            factor: Interpolation factor (2 = double framerate).
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of interpolated frames with factorx the original count.
        """
        if factor <= 1:
            return frames

        result: list[Any] = [frames[0]]

        total_pairs = len(frames) - 1
        for i in range(total_pairs):
            for _j in range(1, factor):
                # The model produces the actual interpolated frame
                frame = self.interpolate(frames[i], frames[i + 1])
                result.append(frame)
            result.append(frames[i + 1])

        # Remove duplicate of last frame
        if len(result) > len(frames) * factor:
            result = result[:-1]

        total = len(result)
        for idx in range(total):
            if progress_callback and total_pairs > 0:
                progress_callback(idx + 1, total, f"Interpolating frame {idx + 1}/{total}")

        return result

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
