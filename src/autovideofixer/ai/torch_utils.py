"""Auto Video Fixer - PyTorch utility functions.

Provides device detection, tensor conversion, and model loading
utilities with graceful fallback when PyTorch is not installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.ai.torch_utils")
    return _logger


# Process-wide semaphore bounding concurrent GPU AI inferences, driven by
# gpu.max_concurrent_inferences (Config.DEFAULTS, default 1). Currently only
# scene-mode's AI/RIFE interpolation path (core/scenes.py's
# interpolate_scene_clip) acquires this -- whole-video (non-scene) runs
# execute stages serially already, so gating them would be a no-op. This is
# the single-GPU placeholder for future multi-GPU inference distribution
# (see docs/ROADMAP.md).
_gpu_inference_semaphore: threading.Semaphore | None = None
_gpu_inference_semaphore_limit: int | None = None
_gpu_inference_semaphore_lock = threading.Lock()


def get_gpu_inference_semaphore(config: Any) -> threading.Semaphore:
    """Return the process-wide GPU inference semaphore, lazily created (or
    recreated, if the configured limit changed) from
    ``gpu.max_concurrent_inferences``.

    Not re-read on every call in the hot sense -- the limit is expected to
    be stable for a process's lifetime; recreating on a limit change exists
    mainly so tests can pass distinct configs without cross-test pollution.
    """
    global _gpu_inference_semaphore, _gpu_inference_semaphore_limit
    limit = config.get("gpu", "max_concurrent_inferences", default=1)
    try:
        limit = int(limit)
    except TypeError, ValueError:
        limit = 1
    if limit < 1:
        limit = 1
    with _gpu_inference_semaphore_lock:
        if _gpu_inference_semaphore is None or _gpu_inference_semaphore_limit != limit:
            _gpu_inference_semaphore = threading.Semaphore(limit)
            _gpu_inference_semaphore_limit = limit
        return _gpu_inference_semaphore


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


class PinnedStagingPool:
    """A small pool of persistent pinned (page-locked) CPU staging tensors.

    `tensor_from_frame`/`tensor_from_frames` copy the incoming numpy frame
    bytes into a reused pinned tensor from this pool instead of calling
    `.pin_memory()` fresh on a brand-new tensor every call -- page-locking a
    fresh host buffer is a real per-call cost (a full 4K BGR frame is
    ~25MB), and only needs to happen once per distinct (shape, dtype) as
    long as a pool slot isn't reused before its previous async H2D copy has
    actually finished reading from it.

    Keyed by `(shape, dtype)`. Capped at a small number of slots total
    (`MAX_KEYS * SLOTS_PER_KEY`) -- simple LRU-by-key eviction, not meant to
    cache many distinct resolutions simultaneously, just to avoid
    re-page-locking on every single frame/chunk of one video.

    Correctness (READ BEFORE CHANGING): a `non_blocking=True` H2D copy off a
    pinned tensor is asynchronous -- the CUDA driver DMAs directly out of
    that host memory on its own schedule, so the copy is NOT guaranteed to
    have finished reading by the time the Python call that queued it
    returns. If this pool's slot were overwritten (via a plain host-side
    `copy_()`, which is NOT ordered against a still-in-flight device read)
    before that read completes, the device would receive a torn/wrong
    frame -- silent, nondeterministic corruption, not a crash. This class
    avoids that by round-robining >= 2 slots per key and recording a
    `torch.cuda.Event` right after each slot's H2D copy is queued; before a
    slot is handed out again, `synchronize()` is called on the event
    recorded for its PREVIOUS use (a cheap no-op if that copy already
    finished, otherwise it blocks -- correctly -- until it has). CPU-only
    callers never use this pool at all (see `tensor_from_frame`/
    `tensor_from_frames`), so no event bookkeeping happens there.
    """

    SLOTS_PER_KEY = 2
    MAX_KEYS = 2  # cap: MAX_KEYS * SLOTS_PER_KEY == 4 pinned tensors total

    class _Slot:
        __slots__ = ("tensor", "event")

        def __init__(self, tensor: Any):
            self.tensor = tensor
            self.event: Any = None

    def __init__(self) -> None:
        import collections

        self._pools: collections.OrderedDict[tuple[Any, Any], dict[str, Any]] = (
            collections.OrderedDict()
        )

    def get_slot(self, shape: tuple[int, ...], dtype: Any) -> "PinnedStagingPool._Slot":
        """Return the next round-robin slot for `(shape, dtype)`, allocating the key if new.

        Blocks (via `torch.cuda.Event.synchronize()`) until this specific
        slot's previous H2D copy, if any, has completed -- see the class
        docstring's correctness note. Safe to call even if no previous copy
        was ever recorded for this slot (no-op in that case).
        """
        import torch

        key = (shape, dtype)
        entry = self._pools.get(key)
        if entry is None:
            while len(self._pools) >= self.MAX_KEYS:
                self._pools.popitem(last=False)  # evict least-recently-used key
            slots = [
                self._Slot(torch.empty(shape, dtype=dtype, pin_memory=True))
                for _ in range(self.SLOTS_PER_KEY)
            ]
            entry = {"slots": slots, "next": 0}
            self._pools[key] = entry
        else:
            self._pools.move_to_end(key)

        slot: "PinnedStagingPool._Slot" = entry["slots"][entry["next"]]
        entry["next"] = (entry["next"] + 1) % self.SLOTS_PER_KEY

        if slot.event is not None:
            slot.event.synchronize()
        return slot

    @staticmethod
    def record_copy(slot: "PinnedStagingPool._Slot") -> None:
        """Record a CUDA event marking "the H2D copy just queued off this slot".

        Call immediately after the `.to(device, non_blocking=True)` call
        that reads from `slot.tensor`.
        """
        import torch

        evt: Any = torch.cuda.Event()  # type: ignore[no-untyped-call]
        evt.record()
        slot.event = evt


_pinned_staging_pool: PinnedStagingPool | None = None


def get_pinned_staging_pool() -> PinnedStagingPool:
    """Return the process-wide `PinnedStagingPool` singleton, creating it on first use."""
    global _pinned_staging_pool
    if _pinned_staging_pool is None:
        _pinned_staging_pool = PinnedStagingPool()
    return _pinned_staging_pool


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
        #
        # Reuse a persistent pinned staging tensor from the pool instead of
        # calling .pin_memory() fresh (which page-locks a brand-new host
        # buffer every call -- a real cost at e.g. 4K, ~25MB/frame) --
        # get_slot() already blocks until that specific slot's PREVIOUS H2D
        # copy has finished reading from it, so overwriting it here via
        # copy_() is safe. See PinnedStagingPool's docstring for why this
        # is NOT safe to do without that synchronization.
        pool = get_pinned_staging_pool()
        slot = pool.get_slot(tuple(cpu_tensor.shape), cpu_tensor.dtype)
        slot.tensor.copy_(cpu_tensor)
        tensor = slot.tensor.to(device=device, non_blocking=True)
        pool.record_copy(slot)
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


def tensor_from_frames(
    frames: list[Any],  # list of numpy arrays (H, W, C) in BGR, all same shape
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Convert a list of numpy frames to a single batched PyTorch tensor.

    Genuinely batched sibling of `tensor_from_frame` above (stacks N frames
    along dim 0 instead of unsqueeze(0)-ing a single one), for the batched
    Real-ESRGAN inference path in ai/wrappers/upscale.py. NOT used by
    ncnn backends or RIFE interpolation -- those still call
    `tensor_from_frame` per-frame unchanged.

    Args:
        frames: List of numpy arrays, each (H, W, 3) in BGR, uint8, all the
            same H/W (caller's responsibility -- np.stack below raises if not).
        device: torch.device to place tensor on.
        dtype: torch dtype for the tensor.

    Returns:
        Tensor of shape (N, 3, H, W) with float values in [0, 1], frame order
        preserved (output index i corresponds to frames[i]).
    """
    import numpy as np
    import torch

    if device is None:
        device = get_device("auto")
    if dtype is None:
        dtype = torch.float32

    # Same H2D-then-convert-on-GPU strategy as tensor_from_frame (see its
    # comments) -- stack on CPU as raw uint8 BGR HWC (smallest payload), one
    # single H2D copy for the whole batch, then do cast/normalize/BGR->RGB/
    # HWC->CHW on the GPU tensor for all N frames at once.
    stacked = np.stack([np.ascontiguousarray(f) for f in frames], axis=0)
    cpu_tensor = torch.from_numpy(stacked)

    if device.type == "cuda":
        # See tensor_from_frame's comments -- same pinned-staging-pool reuse,
        # same slot-reuse-ordering correctness requirement (get_slot()
        # blocks on the slot's previous H2D copy before this copy_() may
        # overwrite it).
        pool = get_pinned_staging_pool()
        slot = pool.get_slot(tuple(cpu_tensor.shape), cpu_tensor.dtype)
        slot.tensor.copy_(cpu_tensor)
        tensor = slot.tensor.to(device=device, non_blocking=True)
        pool.record_copy(slot)
    else:
        tensor = cpu_tensor.to(device=device)

    tensor = tensor[:, :, :, [2, 1, 0]]  # BGR -> RGB
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    tensor = tensor.to(dtype=dtype).div(255.0)
    return tensor


def frames_from_tensor(tensor: Any, scale: float = 1.0) -> list[Any]:  # list of numpy arrays
    """Convert a batched PyTorch tensor back to a list of numpy frames.

    Genuinely batched sibling of `frame_from_tensor` above (splits dim 0 into
    a list instead of squeeze(0)-ing a single-item batch). Same op order and
    numerical semantics (nan_to_num before scale, truncating uint8 cast) as
    frame_from_tensor -- see its comments -- applied across the whole batch
    tensor at once instead of per-frame.

    Args:
        tensor: Tensor of shape (N, 3, H, W) with values in [0, 1].
        scale: Output scale factor (output_h = input_h * scale).

    Returns:
        List of N numpy arrays, each (H_out, W_out, 3) in BGR, uint8, in the
        same order as the input tensor's batch dimension (output[i]
        corresponds to tensor[i]).
    """
    import torch

    if scale != 1.0:
        tensor = torch.nn.functional.interpolate(
            tensor, scale_factor=scale, mode="bilinear", align_corners=False
        )

    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = tensor.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    tensor = tensor[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).contiguous()  # N,C,H,W -> N,H,W,C

    arr = tensor.detach().cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


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
