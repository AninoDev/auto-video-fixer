"""Auto Video Fixer - PyTorch utility functions.

Provides device detection, tensor conversion, and model loading
utilities with graceful fallback when PyTorch is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.ai.torch_utils")
    return _logger


def is_torch_available() -> bool:
    """Check if PyTorch is installed and importable."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def get_device(preferred: str = "auto") -> Any:
    """Get the best available compute device.

    Args:
        preferred: 'auto', 'cuda', 'cpu', or 'mps'.

    Returns:
        torch.device for the selected device.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # This was previously silent, which made "AI upscaling is unexpectedly
        # slow" hard to self-diagnose: a GPU that torch doesn't see (e.g. a
        # PyTorch build with no kernels for the installed GPU's compute
        # capability, or CPU-only torch installed via plain `pip/uv install
        # torch` without a CUDA index URL) means every AI stage silently runs
        # a many-block CNN on CPU, ~1-2 orders of magnitude slower than GPU,
        # with no indication anything is wrong short of the job taking forever.
        # `avf gpu-info` now also surfaces this proactively (see cli.py).
        _get_logger().warning(
            "No CUDA or MPS GPU detected by PyTorch -- AI stages (upscale, "
            "interpolate, AI denoise) will run on CPU, which is dramatically "
            "slower. Run `avf gpu-info` to check what PyTorch sees, or pass "
            "--no-ai to use traditional (non-AI) methods instead."
        )
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            _get_logger().warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")

    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            _get_logger().warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(preferred: str = "fp32") -> Any:
    """Get PyTorch data type.

    Args:
        preferred: 'fp16', 'fp32', or 'bf16'.

    Returns:
        torch dtype.
    """
    import torch

    if preferred == "fp16":
        return torch.float16
    if preferred == "bf16":
        return torch.bfloat16
    return torch.float32


def tensor_from_frame(
    frame: "Any",  # numpy array (H, W, C) in BGR
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Convert a numpy frame to a PyTorch tensor.

    Converts from BGR (H, W, C) with range [0, 255] to CHW with range [0, 1].

    Args:
        frame: numpy array of shape (H, W, 3) in BGR, uint8.
        device: torch.device to place tensor on.
        dtype: torch dtype for the tensor.

    Returns:
        Tensor of shape (1, 3, H, W) with float values in [0, 1].
    """
    import numpy as np
    import torch

    if device is None:
        device = get_device("auto")
    if dtype is None:
        dtype = torch.float32

    arr = frame.astype("float32", copy=False) / 255.0
    # BGR -> RGB
    arr = arr[:, :, ::-1]
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    # Ensure contiguous array (torch doesn't support negative strides)
    arr = np.ascontiguousarray(arr)
    cpu_tensor = torch.from_numpy(arr).unsqueeze(0)

    if device.type == "cuda":
        # Copying from a pinned (page-locked) host buffer lets the CUDA driver
        # DMA it directly instead of first staging through an intermediate
        # pinned bounce buffer it allocates/frees per call -- non_blocking=True
        # then lets this H2D copy overlap with other CUDA-stream work queued
        # around it instead of forcing a host/device sync at every frame.
        cpu_tensor = cpu_tensor.pin_memory()
        tensor = cpu_tensor.to(device=device, dtype=dtype, non_blocking=True)
    else:
        tensor = cpu_tensor.to(device=device, dtype=dtype)
    return tensor


def frame_from_tensor(tensor: Any, scale: float = 1.0) -> "Any":  # numpy array
    """Convert a PyTorch tensor back to a numpy frame.

    Args:
        tensor: Tensor of shape (1, 3, H, W) with values in [0, 1].
        scale: Output scale factor (output_h = input_h * scale).

    Returns:
        numpy array of shape (H_out, W_out, 3) in RGB, uint8.
    """
    import numpy as np
    import torch

    tensor = tensor.detach().cpu()
    if scale != 1.0:
        tensor = torch.nn.functional.interpolate(
            tensor, scale_factor=scale, mode="bilinear", align_corners=False
        )

    arr = tensor.squeeze(0).numpy().transpose(1, 2, 0)
    # RGB -> BGR for OpenCV
    arr = arr[:, :, ::-1]
    # Sanitize NaN/Inf before clipping
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype("uint8")
    return arr


def load_model_from_state_dict(
    model: Any,
    state_dict_path: str,
    device: Any = None,
) -> Any:
    """Load model weights from a state dict file.

    Args:
        model: PyTorch model instance.
        state_dict_path: Path to .pth state dict file.
        device: Device to load model onto.

    Returns:
        Model with loaded weights, in eval mode.
    """
    import torch

    if device is None:
        device = get_device("auto")

    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    # Handle 'params_ema' wrapper (Real-ESRGAN format)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def apply_tta(
    model: Any,
    tensor: Any,
    mode: int = 7,
) -> Any:
    """Apply test-time augmentation for improved quality.

    Runs the model on several geometric augmentations of the input (flips,
    rotations) and averages the results for a slight quality improvement.
    Each augmented output MUST be transformed back to the canonical
    (un-augmented) orientation before averaging - a flipped output pixel
    isn't spatially aligned with an unflipped one, so averaging first and
    inverse-transforming the average afterward (as this used to do) mixes
    misaligned pixels and produces ghosting rather than a genuine
    improvement.

    Args:
        model: Loaded PyTorch model.
        tensor: Input tensor (1, C, H, W).
        mode: TTA mode (1-7). 7 = all 8 augmentations.

    Returns:
        Enhanced tensor.
    """
    import torch

    # Each entry is already transformed back to canonical orientation.
    canonical_outputs = [model(tensor)]

    if mode >= 2:
        # Horizontal flip
        out = model(torch.flip(tensor, [3]))
        canonical_outputs.append(torch.flip(out, [3]))

    if mode >= 4:
        # Vertical flip
        out = model(torch.flip(tensor, [2]))
        canonical_outputs.append(torch.flip(out, [2]))

    if mode >= 8:
        # Both flips
        out = model(torch.flip(tensor, [2, 3]))
        canonical_outputs.append(torch.flip(out, [2, 3]))

    if mode >= 16:
        # 90/180/270 degree rotations, each inverse-rotated back individually.
        rot90_in = torch.rot90(tensor, 1, [2, 3])
        rot180_in = torch.rot90(tensor, 2, [2, 3])
        rot270_in = torch.rot90(tensor, 3, [2, 3])
        canonical_outputs.append(torch.rot90(model(rot90_in), -1, [2, 3]))
        canonical_outputs.append(torch.rot90(model(rot180_in), -2, [2, 3]))
        canonical_outputs.append(torch.rot90(model(rot270_in), -3, [2, 3]))

    if len(canonical_outputs) == 1:
        return canonical_outputs[0]

    return sum(canonical_outputs) / len(canonical_outputs)


def infer_batch(
    model: Any,
    frames: list[Any],  # list of tensors (1, C, H, W)
    device: Any,
    batch_size: int = 1,
    dtype: Any = None,
) -> list[Any]:
    """Run inference on a batch of frames.

    Args:
        model: Loaded PyTorch model in eval mode.
        frames: List of input tensors.
        device: Device to run inference on.
        batch_size: Number of frames per batch.
        dtype: Optional dtype for half-precision inference.

    Returns:
        List of output tensors, same length as input.
    """
    import torch

    outputs: list[Any] = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            if dtype is not None:
                batch = [b.to(dtype=dtype) for b in batch]
            out = model(*batch) if len(batch) > 1 else model(batch[0])
            if not isinstance(out, (list, tuple)):
                out = [out]
            outputs.extend([o.detach().to(device=device, dtype=torch.float32) for o in out])
    return outputs
