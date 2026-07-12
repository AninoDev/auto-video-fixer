"""Auto Video Fixer - shared helpers for ncnn/Vulkan backends.

Frame convention matches the rest of `ai/`: numpy arrays, shape (H, W, 3),
BGR channel order, uint8 -- see `ai/frame_processor.py`. ncnn's own `Mat`
type is channel-first (C, H, W); the conversion helpers here handle both
directions plus the BGR<->RGB swap ncnn's Real-ESRGAN/RIFE ports expect
(both upstream projects train/export on RGB, matching the torch side of
this codebase -- see `ai/torch_utils.py:tensor_from_frame`).
"""

from __future__ import annotations

import logging
from typing import Any

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.backends.ncnn")
    return _logger


def is_ncnn_available() -> bool:
    """Check whether the `ncnn` Python package is installed and importable.

    This does NOT check for a usable Vulkan device -- a missing/broken
    Vulkan driver still imports fine and ncnn transparently falls back to
    CPU inference (see `get_vulkan_gpu_count()` for that check). Mirrors
    `torch_utils.is_torch_available()`'s narrow "is the package there"
    scope.
    """
    try:
        import ncnn  # noqa: F401

        return True
    except ImportError:
        return False


def get_vulkan_gpu_count() -> int:
    """Return the number of Vulkan-capable GPUs ncnn can see, or 0.

    0 covers both "ncnn not installed" and "installed but no Vulkan
    device/driver" -- callers that only care about "can I use Vulkan
    acceleration" can treat any exception here the same as 0 rather than
    needing two separate checks.
    """
    try:
        import ncnn

        return int(ncnn.get_gpu_count())
    except Exception as e:
        _get_logger().debug(f"ncnn Vulkan GPU probe failed: {e}")
        return 0


def frame_to_mat(frame: Any, target_w: int | None = None, target_h: int | None = None) -> Any:
    """Convert a BGR uint8 (H, W, 3) numpy frame to an ncnn.Mat (RGB, normalized to [0,1])."""
    import ncnn

    h, w = frame.shape[0], frame.shape[1]
    mat = ncnn.Mat.from_pixels_resize(
        frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, target_w or w, target_h or h
    )
    # Real-ESRGAN/RIFE ncnn ports both normalize input to [0, 1] (no mean
    # subtraction) -- matches tensor_from_frame()'s /255.0 on the torch side.
    mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    return mat


def mat_to_frame(mat: Any) -> Any:
    """Convert an ncnn.Mat (RGB, [0,1] float, C,H,W) back to a BGR uint8 (H, W, 3) numpy frame."""
    import numpy as np

    arr = np.array(mat)  # (C, H, W) float32, RGB, ~[0, 1]
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = arr.transpose(1, 2, 0)  # (H, W, C) RGB
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(arr).astype(np.uint8)


def make_net(use_vulkan: bool = True, vulkan_device: int = 0) -> Any:
    """Create an `ncnn.Net` configured for Vulkan (if available/requested) or CPU.

    Falls back to CPU inference (logged at WARNING, not an error -- ncnn
    handles this natively) whenever Vulkan was requested but no device is
    visible, e.g. no Vulkan ICD installed. Chunked/streaming behavior (the
    per-frame call pattern) is identical either way; only throughput
    differs, which is exactly the same "backend works, may just be slower"
    posture `frame_processor`/the torch path already have for CPU-only
    PyTorch.
    """
    import ncnn

    net = ncnn.Net()
    gpu_count = get_vulkan_gpu_count()
    if use_vulkan and gpu_count > 0:
        net.opt.use_vulkan_compute = True
        if hasattr(net, "set_vulkan_device"):
            net.set_vulkan_device(vulkan_device)
    else:
        if use_vulkan:
            _get_logger().warning(
                "ncnn backend: Vulkan requested but no Vulkan device visible "
                "(ncnn.get_gpu_count() == 0); falling back to CPU inference"
            )
        net.opt.use_vulkan_compute = False

    # Without these, ncnn's Option defaults leave blob storage/arithmetic in
    # fp32 on some builds -- roughly 2-4x the VRAM of the torch backend's fp16
    # path for the same RRDB feature maps, which is why a plain 1280x720 frame
    # (well under AUTO_TILE_THRESHOLD_PX) can OOM on a 16GB card even though
    # the equivalent torch/CUDA pass fits comfortably. fp16 arithmetic on a
    # super-resolution network is the same quality/precision tradeoff the
    # torch path already makes (RealESRGANUpscaler's own use_fp16 on CUDA).
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = True
    return net


def tiled_ncnn_inference(
    frame: Any,
    tile_size: int,
    overlap: int,
    out_scale: int,
    infer_fn: Any,
) -> Any:
    """Run `infer_fn` (BGR uint8 frame -> BGR uint8 frame, at `out_scale`x) tile-by-tile.

    Reuses the same tile-grid geometry as the torch path
    (`ai.wrappers.upscale.compute_tile_grid` -- pure pixel-coordinate math,
    no torch dependency) rather than re-deriving overlap/stitching logic,
    so both backends produce identically-shaped tiles for a given
    tile_size/overlap. Only the per-tile inference call and the
    numpy-vs-tensor stitching differ.
    """
    import numpy as np

    from autovideofixer.ai.wrappers.upscale import compute_tile_grid

    h, w = frame.shape[0], frame.shape[1]
    grid = compute_tile_grid(h, w, tile_size, overlap, out_scale=out_scale)

    out = np.empty((h * out_scale, w * out_scale, 3), dtype=np.uint8)
    for spec in grid:
        tile_in = frame[spec.in_y0 : spec.in_y1, spec.in_x0 : spec.in_x1]
        tile_out = infer_fn(tile_in)
        out[spec.out_y0 : spec.out_y1, spec.out_x0 : spec.out_x1] = tile_out[
            spec.crop_y0 : spec.crop_y1, spec.crop_x0 : spec.crop_x1
        ]
    return out
