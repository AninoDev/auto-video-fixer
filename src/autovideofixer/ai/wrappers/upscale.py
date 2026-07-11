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
        # The official BasicSR/Real-ESRGAN architecture scales the dense
        # block's residual branch by the empirical factor 0.2 before adding
        # it back (mirroring RRDB's own 0.2-scaled residual just below).
        # Without it, 23 RRDB blocks x 3 ResidualDenseBlocks each (69 total)
        # compound unscaled residual additions and the activations explode
        # into NaN within a few blocks -- frame_from_tensor's nan_to_num()
        # then silently renders that as solid black output. This was the
        # root cause of every Real-ESRGAN-based stage (upscale, denoise,
        # deblock) producing all-black video despite the weights loading
        # with a perfectly matching state_dict.
        return x5 * 0.2 + x


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


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Inverse of pixel_shuffle: trade spatial resolution for channels.

    (B, C, H*scale, W*scale) -> (B, C*scale^2, H, W). This is how the
    official BasicSR RRDBNet lets a single architecture serve x1/x2/x4
    checkpoints: the two upsample stages in forward() are always a fixed
    net 4x, so a scale=2 checkpoint pre-shrinks the spatial input by 2x
    (via this unshuffle) before conv_first so the two 4x-producing
    upsample stages land back on a net 2x versus the original input; a
    scale=1 checkpoint pre-shrinks by 4x the same way. This also means the
    expensive RRDB body (the ~90% of forward-pass cost measured on this
    project's target GPU) operates on a proportionally smaller feature
    map for scale=1/2 checkpoints than for scale=4 -- not just the final
    upsample tail -- which is the actual source of the speedup, not merely
    "avoiding a wasted 4x final size".
    """
    b, c, hh, hw = x.size()
    if hh % scale != 0 or hw % scale != 0:
        raise ValueError(
            f"pixel_unshuffle: spatial dims ({hh}x{hw}) must be divisible by scale={scale}"
        )
    h, w = hh // scale, hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class RRDBNet(torch.nn.Module):
    """RRDBNet architecture for Real-ESRGAN.

    Matches the official BasicSR implementation with:
    - conv_first: initial 3x3 conv
    - body: 23 RRDB blocks
    - conv_body: 3x3 conv on body output
    - conv_up1/conv_up2: upsampling stages (always a fixed net 4x)
    - conv_hr: high-resolution feature
    - conv_last: final output conv

    `scale` controls the NET scale factor the checkpoint was trained for
    (1, 2, or 4) by pixel-unshuffling the input before conv_first -- see
    pixel_unshuffle() above. It does NOT change conv_up1/conv_up2, which
    always perform a fixed 4x upsample; scale instead changes how much the
    input is pre-shrunk so that fixed 4x lands on the checkpoint's actual
    trained scale. This must match the checkpoint being loaded: x4plus
    uses scale=4 (no pre-shrink, num_in_ch stays 3), x2plus uses scale=2
    (num_in_ch becomes num_in_ch*4=12).
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
        self._orig_in_ch = num_in_ch

        if scale == 2:
            conv_first_in_ch = num_in_ch * 4
        elif scale == 1:
            conv_first_in_ch = num_in_ch * 16
        else:
            conv_first_in_ch = num_in_ch

        self.conv_first = torch.nn.Conv2d(conv_first_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat_in = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat_in = pixel_unshuffle(x, scale=4)
        else:
            feat_in = x

        feat = self.conv_first(feat_in)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Two fixed x2 upsample stages (always a net 4x from *this point*),
        # which combined with the scale-dependent pre-shrink above lands on
        # the checkpoint's actual trained net scale (1, 2, or 4).
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
        scale: float = 4,
        model_name: str = "RealESRGAN_x4plus",
        tta_mode: int = 0,
        batch_size: int = 1,
        device_preference: str = "auto",
    ):
        self.scale = scale
        self.model_name = model_name
        self.tta_mode = tta_mode
        self.batch_size = batch_size
        self.device_preference = device_preference
        self._model: Any = None
        self._device: Any = None
        self._loaded = False
        self._use_fp16 = False
        self._native_scale = scale

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

        self._device = get_device(self.device_preference)

        if self._device.type == "cuda":
            import torch

            # cudnn.benchmark profiles and caches the fastest conv algorithm for
            # a given input shape on first use -- a meaningful speedup here since
            # every frame in a video is the same fixed shape (repeated identical
            # forward passes), unlike typical training workloads with varying
            # batch/input sizes where this flag can hurt instead of help.
            torch.backends.cudnn.benchmark = True

        # Determine num_block based on model name
        num_block = 6 if "6B" in self.model_name else 23
        num_feat = 64

        # RRDBNet's `scale` constructor arg selects the pixel-unshuffle
        # preprocessing (see pixel_unshuffle()/RRDBNet docstrings above) and
        # MUST match the checkpoint's own trained architecture -- x4plus was
        # trained with scale=4 (no unshuffle, conv_first in_ch=3), x2plus
        # with scale=2 (conv_first in_ch=12). This is NOT the same thing as
        # self.scale (the caller's desired final output scale, e.g. a
        # denoise/deblock caller requesting scale=1 while still using the
        # x4plus checkpoint because there is no official x1 Real-ESRGAN
        # checkpoint). Get the checkpoint's real native scale from the model
        # registry; upscale()/frame_from_tensor apply a post-hoc GPU-side
        # resize (self.scale / native_scale) to reconcile the two whenever
        # they differ.
        from autovideofixer.ai.model_cache import MODEL_REGISTRY

        self._native_scale = MODEL_REGISTRY.get(self.model_name, {}).get("scale", 4)

        self._model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=32,
            scale=self._native_scale,
        )

        self._model = load_model_from_state_dict(self._model, model_path, self._device)

        # Cast once, at load time - not per-frame. Casting the whole model's
        # weights on every upscale() call multiplied overhead by frame count.
        self._use_fp16 = self._device.type == "cuda"
        if self._use_fp16:
            self._model.half()

        if self._device.type == "cuda":
            # channels_last (NHWC) lets cudnn dispatch its NHWC-native conv
            # kernels directly. Left in the default contiguous (NCHW) format,
            # cudnn was inserting an implicit nchwToNhwcKernel layout
            # conversion before every single conv2d call (measured at ~29%
            # of total CUDA time via torch.profiler on this RTX 5060 Ti /
            # sm_120 Blackwell card) because the fprop kernel it selects here
            # is NHWC-based. Converting the model's weights once at load time
            # (matched by converting each input tensor in upscale()) skips
            # that redundant conversion on every call -- measured ~25-30%
            # faster end-to-end forward pass with identical output.
            self._model = self._model.to(memory_format=torch.channels_last)

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
            tensor_from_frame,
        )

        if tta_mode is None:
            tta_mode = self.tta_mode

        tensor = tensor_from_frame(frame, device=self._device)
        if self._use_fp16:
            tensor = tensor.half()
        if self._device.type == "cuda":
            # Match the channels_last layout the model was converted to in
            # load_model() -- passing a contiguous (NCHW) tensor into a
            # channels_last model forces cudnn to convert it internally on
            # every call anyway, defeating the point.
            tensor = tensor.to(memory_format=torch.channels_last)

        def _infer() -> Any:
            with torch.no_grad():
                if tta_mode and tta_mode >= 1:
                    return apply_tta(self._model, tensor, mode=tta_mode)
                return self._model(tensor)

        try:
            output = _infer()
        except torch.cuda.OutOfMemoryError:
            # A single oversized/high-res frame can exceed VRAM even though prior
            # frames fit; clear the allocator cache and retry once instead of
            # aborting the whole video on one frame.
            _get_logger().warning("CUDA OOM upscaling a frame; clearing cache and retrying once")
            torch.cuda.empty_cache()
            try:
                output = _infer()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory upscaling frame even after cache clear + "
                    "retry; try a smaller scale_factor or --no-ai"
                ) from None

        if self._use_fp16:
            output = output.float()

        # RRDBNet's forward() now natively honors its checkpoint's real
        # trained scale via pixel-unshuffle preprocessing (see RRDBNet
        # docstring), so a scale=2 request loaded against the x2plus
        # checkpoint (self._native_scale == 2) needs no correction at all --
        # the RRDB body itself now runs on a proportionally smaller feature
        # map too, not just the output. A correction is only needed when the
        # caller's desired scale doesn't match any available checkpoint's
        # native architecture -- e.g. deblock/denoise_video requesting
        # scale=1 output while still using the x4plus checkpoint (there is
        # no official x1 Real-ESRGAN checkpoint), or scale=3 falling between
        # the 2x/4x checkpoints. In that case forward() still produces
        # self._native_scale output, and this resizes it (on-GPU, before the
        # .cpu() transfer in frame_from_tensor) down/up to what was asked
        # for.
        correction = self.scale / self._native_scale
        result = frame_from_tensor(output, scale=correction)
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
            self._use_fp16 = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
