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

    # Transfer the raw uint8 BGR HWC frame as-is (smallest possible payload --
    # 1 byte/channel) and do the cast/normalize/BGR->RGB/HWC->CHW conversion on
    # the GPU tensor AFTER the H2D copy, not before: the mirror-image fix of
    # frame_from_tensor's GPU-side postprocessing below, for the same reason --
    # a CPU-bound numpy elementwise pass here was profiled (py-spy against a
    # live process) contributing to the same single-core CPU pinning that
    # starved the GPU between inference calls. Values differ from the old
    # CPU-numpy path by up to ~1 float32 ULP (GPU vs CPU division rounding),
    # not a real precision loss -- see torch_utils tests.
    arr = np.ascontiguousarray(frame)
    cpu_tensor = torch.from_numpy(arr).unsqueeze(0)

    if device.type == "cuda":
        # Copying from a pinned (page-locked) host buffer lets the CUDA driver
        # DMA it directly instead of first staging through an intermediate
        # pinned bounce buffer it allocates/frees per call -- non_blocking=True
        # then lets this H2D copy overlap with other CUDA-stream work queued
        # around it instead of forcing a host/device sync at every frame.
        cpu_tensor = cpu_tensor.pin_memory()
        tensor = cpu_tensor.to(device=device, non_blocking=True)
    else:
        tensor = cpu_tensor.to(device=device)

    tensor = tensor[:, :, :, [2, 1, 0]]  # BGR -> RGB
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # HWC -> CHW
    tensor = tensor.to(dtype=dtype).div(255.0)
    return tensor


def frame_from_tensor(tensor: Any, scale: float = 1.0) -> "Any":  # numpy array
    """Convert a PyTorch tensor back to a numpy frame.

    Args:
        tensor: Tensor of shape (1, 3, H, W) with values in [0, 1].
        scale: Output scale factor (output_h = input_h * scale).

    Returns:
        numpy array of shape (H_out, W_out, 3) in BGR, uint8.
    """
    import torch

    if scale != 1.0:
        tensor = torch.nn.functional.interpolate(
            tensor, scale_factor=scale, mode="bilinear", align_corners=False
        )

    # All elementwise post-processing (NaN/Inf sanitize, [0,1]->[0,255] scale+clip,
    # uint8 cast, RGB->BGR channel reorder, NCHW->NHWC layout) runs on the GPU tensor
    # BEFORE the device-to-host transfer, not after -- profiling a real upscale run
    # (py-spy dump against a live process) showed this function's old numpy-based
    # postprocessing pinning a full CPU core between inference calls, starving the GPU
    # (high utilization%, but low "effective" SM occupancy in nvtop -- the GPU was idle
    # waiting for the CPU-bound conversion of the PREVIOUS frame before the next
    # inference call could be issued). Two wins from moving this to torch/GPU ops:
    # the elementwise math itself runs across thousands of CUDA cores instead of one
    # CPU core running un-vectorized numpy passes over a negative-stride (reversed
    # channel) view, and the eventual .cpu() transfer moves 1 byte/channel (uint8)
    # instead of 4 (float32), cutting the PCIe transfer volume 4x too. Op order
    # (interpolate -> nan_to_num -> clamp+scale+cast -> channel/layout reorder)
    # matches the previous numpy implementation exactly, including nan_to_num's
    # posinf/neginf substitution values being in normalized [0,1] space (applied
    # before the *255 scale, same as before) and the uint8 cast truncating rather
    # than rounding (matching numpy .astype("uint8")'s truncation, not round-to-nearest).
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = tensor.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    tensor = tensor[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).contiguous().squeeze(0)

    arr = tensor.detach().cpu().numpy()
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
