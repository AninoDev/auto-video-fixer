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
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)
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
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
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

    Args:
        model: Loaded PyTorch model.
        tensor: Input tensor (1, C, H, W).
        mode: TTA mode (1-7). 7 = all 8 augmentations.

    Returns:
        Enhanced tensor.
    """
    import torch

    outputs = []
    # Original
    outputs.append(model(tensor))

    if mode >= 2:
        # Horizontal flip
        outputs.append(model(torch.flip(tensor, [3])))

    if mode >= 4:
        # Vertical flip
        outputs.append(model(torch.flip(tensor, [2])))

    if mode >= 8:
        # Both flips
        outputs.append(model(torch.flip(tensor, [2, 3])))

    if mode >= 16:
        # Rotations
        t90 = torch.rot90(tensor, 1, [2, 3])
        outputs.append(model(t90))
        outputs.append(model(torch.rot90(t90, 1, [2, 3])))
        outputs.append(model(torch.rot90(t90, 2, [2, 3])))

    if len(outputs) == 1:
        return outputs[0]

    result = sum(o for o in outputs) / len(outputs)

    # Reverse augmentations
    if mode >= 2:
        result = torch.flip(result, [3])
    if mode >= 4:
        result = torch.flip(result, [2])

    return result


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
