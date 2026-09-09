"""Auto Video Fixer - RIFE frame interpolation model wrapper.

Implements the real RIFE (Real-Time Intermediate Flow Estimation) IFNet
architecture, ported line-for-line from the official hzwer/Practical-RIFE
inference code (IFNet_HDv3.py / RIFE_HDv3.py, model version 4.25/4.26) so
that downloaded `flownet.pkl` checkpoints load with zero missing/unexpected
keys. Verified against a real checkpoint: state_dict loads with strict
matching (module-prefix stripped) aside from the training-only `teacher`/
`caltime` submodules the upstream loader also discards, and a forward pass
at two different timesteps produces genuinely different output (proving
timestep conditioning works, unlike a fixed-midpoint interpolator).
"""

from __future__ import annotations

import logging
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
import torch.nn.functional as F

from autovideofixer.ai.model_cache import get_model_path

_logger: logging.Logger | None = None

_backwarp_grids: dict[tuple[str, str, str], torch.Tensor] = {}


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.interpolate")
    return _logger


@dataclass(frozen=True)
class PlanEntry:
    """One output frame's origin, as produced by ``resample_plan()``
    (REQUIREMENTS.md § 16.1 -- per-gap adaptive AI interpolation).

    ``op``:
      - ``"emit"``: output the ORIGINAL input frame at index ``k`` verbatim
        (no inference at all) -- the output grid landed on, or within
        ``eps`` of, a real frame.
      - ``"synthesize"``: RIFE-interpolate between input frames ``k`` and
        ``k + 1`` at ``timestep=local``.
      - ``"hold"``: ``max_intermediates_per_gap`` was exceeded for this gap
        and ``gap_fallback="hold"`` -- repeat input frame ``k`` verbatim
        (no inference), exactly what the source did.
      - ``"blend"``: same cap, ``gap_fallback="blend"`` -- a linear
        crossfade of frames ``k``/``k + 1`` at weight ``local`` (no RIFE
        inference).

    ``k`` means different things depending on ``op``: for ``"emit"`` it is
    the EXACT input frame index to output; for the other three ops it is
    the bracket's LOWER index -- the pair referenced is always
    ``(k, k + 1)``.
    """

    op: str
    k: int
    local: float = 0.0


def resample_plan(
    in_timestamps: list[float],
    target_fps: float,
    *,
    eps: float = 1e-3,
    max_intermediates_per_gap: int = 8,
    gap_fallback: str = "hold",
) -> list[PlanEntry]:
    """Pure, side-effect-free timeline-resampling core (REQUIREMENTS.md § 16.1).

    Given real input frame timestamps ``in_timestamps`` (ascending, seconds)
    and a target framerate ``target_fps``, builds the output grid
    ``o[j] = t[0] + j / target_fps`` for ``j = 0 .. floor((t[-1] - t[0]) *
    target_fps)``, brackets each ``o[j]`` between consecutive input frames
    ``t[k] <= o[j] <= t[k + 1]``, and computes the normalized position
    ``local = (o[j] - t[k]) / (t[k + 1] - t[k])`` within that bracket:

    - ``local <= eps``      -> emit the ORIGINAL frame ``k`` (no inference).
    - ``local >= 1 - eps``  -> emit the ORIGINAL frame ``k + 1``.
    - otherwise             -> synthesize at ``timestep=local``.

    This is the grid formulation (not per-gap counting) because it is
    **inherently drift-free**: every ``o[j]`` is computed from the GLOBAL
    grid, never accumulated gap-by-gap, so rounding cannot walk the output
    duration off over a long video -- see REQUIREMENTS.md § 16.1 for the
    full rationale (also: real frames are reused wherever the grid lands on
    one, avoiding both needless inference cost and re-synthesizing a frame
    that already exists).

    ``max_intermediates_per_gap``: when a single bracket needs MORE
    synthesized frames than this (a stall or a hard cut, not motion -- RIFE
    quality degrades badly at large motion / extreme timesteps), every
    "synthesize" entry in that bracket is replaced by ``gap_fallback``
    (``"hold"`` -- repeat frame ``k``, default; or ``"blend"`` -- linear
    crossfade) instead of asking RIFE to invent motion across an
    unreasonably large gap.

    Degenerate inputs never raise and never divide by zero: an empty
    timeline returns ``[]``; a single-frame timeline or a non-positive
    ``target_fps`` returns one "emit" entry per input frame; a
    zero-or-negative overall duration (duplicate timestamps, or a
    non-monotonic timeline whose endpoints collapse) likewise degrades to
    "emit every original frame once, in order" rather than resampling.
    Non-monotonic timestamps and zero-length individual gaps are handled by
    clamping ``local`` to ``[0, 1]`` and treating a zero-or-negative
    individual gap as an immediate snap to frame ``k`` -- never a
    ZeroDivisionError.
    """
    if gap_fallback not in ("hold", "blend"):
        raise ValueError(f"gap_fallback must be 'hold' or 'blend', got {gap_fallback!r}")

    n = len(in_timestamps)
    if n == 0:
        return []
    if n == 1 or target_fps <= 0:
        return [PlanEntry(op="emit", k=i) for i in range(n)]

    t = in_timestamps
    t0 = t[0]
    duration = t[-1] - t0
    if duration <= 0:
        return [PlanEntry(op="emit", k=i) for i in range(n)]

    # Small epsilon guards the floor() against a value that's mathematically
    # exactly an integer but lands a hair under it due to float error.
    j_max = int(math.floor(duration * target_fps + 1e-9))

    entries: list[PlanEntry] = []
    bracket: list[PlanEntry] = []
    k = 0

    def flush() -> None:
        nonlocal bracket
        if not bracket:
            return
        synth_idx = [i for i, e in enumerate(bracket) if e.op == "synthesize"]
        if len(synth_idx) > max_intermediates_per_gap:
            for i in synth_idx:
                e = bracket[i]
                bracket[i] = PlanEntry(op=gap_fallback, k=e.k, local=e.local)
        entries.extend(bracket)
        bracket = []

    for j in range(j_max + 1):
        o = t0 + j / target_fps
        # Two-pointer advance: o[j] is monotonic non-decreasing, so k only
        # ever moves forward -- never re-scans from the start.
        while k < n - 2 and o > t[k + 1]:
            flush()
            k += 1

        gap = t[k + 1] - t[k] if k + 1 < n else 0.0
        if gap <= 0:
            local = 0.0
        else:
            local = (o - t[k]) / gap
            local = 0.0 if local < 0.0 else (1.0 if local > 1.0 else local)

        if local <= eps:
            bracket.append(PlanEntry(op="emit", k=k))
        elif local >= 1.0 - eps:
            bracket.append(PlanEntry(op="emit", k=k + 1))
        else:
            bracket.append(PlanEntry(op="synthesize", k=k, local=local))

    flush()
    return entries


def plan_stats(plan: list[PlanEntry]) -> dict[str, int]:
    """Summary counters for a ``resample_plan()`` result (REQUIREMENTS.md
    § 16.6 stage metadata): frames synthesized via RIFE inference, frames
    passed through unchanged (a real input frame, reused verbatim), and how
    many DISTINCT gaps hit ``max_intermediates_per_gap`` and fell back to
    "hold"/"blend" instead of synthesis.
    """
    frames_synthesized = 0
    frames_passed_through = 0
    capped_gaps: set[int] = set()
    for e in plan:
        if e.op == "synthesize":
            frames_synthesized += 1
        elif e.op == "emit":
            frames_passed_through += 1
        else:  # "hold" / "blend"
            capped_gaps.add(e.k)
    return {
        "frames_synthesized": frames_synthesized,
        "frames_passed_through": frames_passed_through,
        "gaps_capped": len(capped_gaps),
    }


def _linear_blend(frame_a: Any, frame_b: Any, weight: float) -> Any:
    """Linear crossfade between two frames at ``weight`` (0.0 -> a, 1.0 ->
    b) -- ``resample_plan()``'s "blend" ``gap_fallback`` (§ 16.3). Frames
    are numpy arrays (H, W, 3) uint8; blending is done in float32 and
    rounded back to the input dtype."""
    import numpy as np

    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    blended = a * (1.0 - weight) + b * weight
    return blended.round().astype(frame_a.dtype)


def execute_resample_plan(
    frame_source: Iterable[Any],
    plan: list[PlanEntry],
    interpolate_fn: Callable[[Any, Any, float], Any],
) -> Iterator[Any]:
    """Streaming executor for a ``resample_plan()`` result (§ 16.3
    "streaming discipline").

    ``frame_source`` yields decoded input frames, in ascending index order
    (frame 0, 1, 2, ...) -- e.g. a flat unroll of a chunked frame reader's
    batches. ``plan`` entries must reference non-decreasing frame indices
    (exactly what ``resample_plan()`` produces). ``interpolate_fn(frame_k,
    frame_k1, timestep)`` computes one synthesized frame, typically
    ``RIFEInterpolator.interpolate``.

    Yields one output frame per plan entry, in order, holding AT MOST the
    current bracket's two frames (``cur_frame``, ``next_frame``) at any
    time -- never an accumulated output list. Reintroducing full
    accumulation here would recreate the RIFE RAM blowup the AI stages'
    streaming path (``ai/frame_pipe.py``) was built to fix (see
    ``core/stages/interpolate.py``'s ``_execute_ai`` docstring).

    Fails open on a source that runs out of frames early (fewer frames than
    the plan expects, e.g. a stale/mismatched timeline): repeats the last
    available frame rather than raising -- this should not happen when the
    timeline probe's frame count matched the reader's actual output (see
    ``core/cadence.py``'s ``probe_frame_timestamps``), but a streaming
    executor must never crash a job over it.
    """
    it = iter(frame_source)

    def _next() -> Any:
        try:
            return next(it)
        except StopIteration:
            return None

    cur_idx = -1
    cur_frame: Any = None
    next_idx = -1
    next_frame: Any = _next()
    if next_frame is None:
        return
    cur_idx, cur_frame = 0, next_frame
    next_idx = 1
    next_frame = _next()

    def advance_to(target: int) -> None:
        nonlocal cur_idx, cur_frame, next_idx, next_frame
        while cur_idx < target and next_frame is not None:
            cur_idx, cur_frame = next_idx, next_frame
            next_idx = cur_idx + 1
            next_frame = _next()

    for entry in plan:
        if entry.op == "emit":
            advance_to(entry.k)
            yield cur_frame
            continue

        advance_to(entry.k)
        if next_frame is None:
            yield cur_frame
            continue

        if entry.op == "synthesize":
            yield interpolate_fn(cur_frame, next_frame, entry.local)
        elif entry.op == "hold":
            yield cur_frame
        elif entry.op == "blend":
            yield _linear_blend(cur_frame, next_frame, entry.local)
        else:  # pragma: no cover - resample_plan() never emits any other op
            yield cur_frame


def warp(tensor_input: torch.Tensor, tensor_flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `tensor_input` by the optical flow field `tensor_flow`.

    Ported from the official RIFE `model/warplayer.py::warp`. Grids are
    cached by (device, shape) like upstream, but built directly on
    `tensor_flow.device` rather than a module-level global device, so this
    also works correctly on MPS/multi-GPU instead of only the single device
    active at import time.
    """
    # Cache key includes dtype: a fp16 call after a fp32 call (or vice
    # versa) must not reuse a grid of the wrong dtype, which errors out of
    # grid_sample ("expected scalar type Half but found Float").
    key = (str(tensor_flow.device), str(tensor_flow.dtype), str(tensor_flow.size()))
    if key not in _backwarp_grids:
        horizontal = (
            torch.linspace(
                -1.0, 1.0, tensor_flow.shape[3], device=tensor_flow.device, dtype=tensor_flow.dtype
            )
            .view(1, 1, 1, tensor_flow.shape[3])
            .expand(tensor_flow.shape[0], -1, tensor_flow.shape[2], -1)
        )
        vertical = (
            torch.linspace(
                -1.0, 1.0, tensor_flow.shape[2], device=tensor_flow.device, dtype=tensor_flow.dtype
            )
            .view(1, 1, tensor_flow.shape[2], 1)
            .expand(tensor_flow.shape[0], -1, -1, tensor_flow.shape[3])
        )
        _backwarp_grids[key] = torch.cat([horizontal, vertical], 1)

    flow = torch.cat(
        [
            tensor_flow[:, 0:1, :, :] / ((tensor_input.shape[3] - 1.0) / 2.0),
            tensor_flow[:, 1:2, :, :] / ((tensor_input.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    grid = (_backwarp_grids[key] + flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tensor_input, grid=grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=True),
        torch.nn.LeakyReLU(0.2, True),
    )


class Head(torch.nn.Module):
    """Shallow per-frame feature encoder (state_dict key: `encode`)."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = torch.nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = torch.nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = torch.nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(torch.nn.Module):
    """Residual conv block with a learnable per-channel scale (`beta`)."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = torch.nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)
        self.relu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(torch.nn.Module):
    """One resolution stage of the coarse-to-fine flow estimator."""

    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = torch.nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = torch.nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            torch.nn.PixelShuffle(2),
        )

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor | None, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        out_flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        out_feat = tmp[:, 5:]
        return out_flow, mask, out_feat


class IFNet(torch.nn.Module):
    """Real RIFE v4.25/4.26 Intermediate Flow Network.

    Five coarse-to-fine `IFBlock` stages iteratively refine a bidirectional
    flow field and blend mask, conditioned on an explicit `timestep` channel
    so distinct timesteps (not just t=0.5) produce genuinely different
    interpolated frames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[float] = [8, 4, 2, 1, 1],  # noqa: B006 - matches upstream signature
    ) -> torch.Tensor:
        """Return the blended frame at `timestep` between img0 and img1 in [0, 1]."""
        timestep_map = (img0[:, :1].clone() * 0 + 1) * timestep
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow: torch.Tensor | None = None
        mask: torch.Tensor | None = None
        warped_img0, warped_img1 = img0, img1
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep_map), 1),
                    None,
                    scale=scale_list[i],
                )
            else:
                warped_f0 = warp(f0, flow[:, :2])
                warped_f1 = warp(f1, flow[:, 2:4])
                flow_delta, mask, feat = blocks[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            warped_f0,
                            warped_f1,
                            timestep_map,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + flow_delta
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        assert mask is not None
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)


def _load_rife_state_dict(model: IFNet, checkpoint_path: str) -> IFNet:
    """Load a real RIFE `flownet.pkl` checkpoint into `model`.

    Mirrors the official `RIFE_HDv3.Model.load_model()` loader: strips the
    `module.` prefix left by `DistributedDataParallel` training, and ignores
    the checkpoint's `teacher`/`caltime` submodule weights (training-only,
    intentionally absent from this inference-only `IFNet`).
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in raw.items() if "module." in k}
    if not state_dict:
        # Some re-exports omit the "module." prefix entirely.
        state_dict = raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing]
    real_unexpected = [
        k for k in unexpected if not (k.startswith("teacher") or k.startswith("caltime"))
    ]
    if real_missing:
        raise RuntimeError(f"RIFE checkpoint missing required keys: {real_missing[:5]}...")
    if real_unexpected:
        _get_logger().warning(f"RIFE checkpoint had unexpected keys: {real_unexpected[:5]}...")
    return model


def _resolve_checkpoint_path(path: str) -> str:
    """If `path` is a downloaded RIFE release zip, extract and cache `flownet.pkl`.

    Official RIFE releases ship the checkpoint bundled in a zip alongside
    its (training-only) source. The model registry points at that zip
    directly since no unpacked `.pkl`-only mirror is guaranteed to stay
    available; this extracts once and reuses the extracted file afterward.
    """
    src = Path(path)
    if src.suffix.lower() != ".zip":
        return path

    extracted = src.with_name(f"{src.stem}_flownet.pkl")
    if extracted.is_file():
        return str(extracted)

    # Cap the extracted member size -- a malicious/corrupted zip (reachable via
    # model_cache.py's --url override) could otherwise claim an oversized entry
    # and exhaust memory on a single unbounded member.read().
    max_extract_bytes = 512 * 1024 * 1024  # 512MB, generous for a flownet.pkl

    with zipfile.ZipFile(src) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("flownet.pkl")]
        if not candidates:
            raise RuntimeError(f"No flownet.pkl found inside RIFE archive: {path}")
        info = zf.getinfo(candidates[0])
        if info.file_size > max_extract_bytes:
            raise RuntimeError(
                f"flownet.pkl entry in {path} is {info.file_size} bytes, "
                f"exceeds the {max_extract_bytes} byte safety limit"
            )
        with zf.open(candidates[0]) as member, open(extracted, "wb") as out:
            while chunk := member.read(1024 * 1024):
                out.write(chunk)

    return str(extracted)


class RIFEInterpolator:
    """RIFE model wrapper for video frame interpolation.

    Usage:
        interpolator = RIFEInterpolator()
        interpolator.load_model()
        result = interpolator.interpolate(frame_a, frame_b, timestep=0.5)
    """

    def __init__(
        self,
        model_name: str = "rife_v4.6",
        device_preference: str = "auto",
        backend: str = "torch",
        vulkan_device: int = 0,
    ):
        """
        Args:
            backend: "torch" (default -- unchanged existing behavior) or
                "ncnn". "ncnn" delegates every method below to
                `ai.backends.ncnn_interpolate.NcnnInterpolateBackend`
                (which wraps the real `rife-ncnn-vulkan-python` package)
                instead of running the torch/IFNet code in this class.
                See ai/WIRING.md.
            vulkan_device: ncnn backend only -- which Vulkan physical
                device index to run on (see gpu.vulkan_device in
                Config.DEFAULTS). Irrelevant to the torch backend.
        """
        self.model_name = model_name
        self.device_preference = device_preference
        self.backend = backend
        self.vulkan_device = vulkan_device
        self._ncnn_backend: Any = None
        self._model: IFNet | None = None
        self._device: Any = None
        self._half = False
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_path: str | None = None) -> bool:
        """Load the RIFE model from disk.

        Args:
            model_path: Path to a `.pkl` checkpoint or a release `.zip`
                bundling one (torch backend). Ignored for the ncnn
                backend, which resolves its own .param/.bin via the ncnn
                model registry.

        Returns:
            True if the model loaded successfully. Never raises: an
            ncnn-specific failure is logged and returns False here, same
            contract as every existing torch failure path below.
        """
        if self.backend == "ncnn":
            from autovideofixer.ai.backends.ncnn_interpolate import NcnnInterpolateBackend

            self._ncnn_backend = NcnnInterpolateBackend(
                model_name=self.model_name, vulkan_device=self.vulkan_device
            )
            self._loaded = self._ncnn_backend.load_model()
            return self._loaded

        from autovideofixer.ai.torch_utils import get_device

        if model_path is None:
            cached = get_model_path(self.model_name)
            if cached is None:
                _get_logger().error(
                    f"Model not found: {self.model_name}. Run ensure_model_available() first."
                )
                return False
            model_path = str(cached)

        if not Path(model_path).is_file():
            _get_logger().error(f"Model file not found: {model_path}")
            return False

        try:
            checkpoint_path = _resolve_checkpoint_path(model_path)
        except Exception as e:
            _get_logger().error(f"Failed to prepare RIFE checkpoint: {e}")
            return False

        self._device = get_device(self.device_preference)
        if self._device.type == "cuda":
            import torch

            torch.backends.cudnn.benchmark = True
        model = IFNet()
        try:
            model = _load_rife_state_dict(model, checkpoint_path)
        except Exception as e:
            _get_logger().error(f"Failed to load RIFE weights: {e}")
            return False

        model.to(self._device)
        model.eval()
        self._half = self._device.type == "cuda"
        if self._half:
            model.half()
        self._model = model
        self._loaded = True

        _get_logger().info(f"Loaded RIFE {self.model_name} on {self._device}")
        return True

    def interpolate(
        self,
        frame0: Any,
        frame1: Any,
        timestep: float = 0.5,
    ) -> Any:
        """Interpolate a single frame between two input frames at `timestep`.

        Args:
            frame0: First frame as numpy array (H, W, 3) BGR uint8.
            frame1: Second frame as numpy array (H, W, 3) BGR uint8.
            timestep: Position between frame0 (0.0) and frame1 (1.0).

        Returns:
            Interpolated frame as numpy array (H, W, 3) BGR uint8.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.backend == "ncnn":
            assert self._ncnn_backend is not None
            return self._ncnn_backend.interpolate(frame0, frame1, timestep)

        assert self._model is not None

        from autovideofixer.ai.torch_utils import frame_from_tensor, get_dtype, tensor_from_frame

        dtype = get_dtype("fp16") if self._half else None

        t0 = tensor_from_frame(frame0, device=self._device, dtype=dtype)
        t1 = tensor_from_frame(frame1, device=self._device, dtype=dtype)

        # RIFE's internal downsampling (5 halvings) requires dimensions
        # divisible by 32; pad and crop back so odd resolutions don't crash.
        _, _, h, w = t0.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            t0 = F.pad(t0, (0, pad_w, 0, pad_h))
            t1 = F.pad(t1, (0, pad_w, 0, pad_h))

        def _infer() -> Any:
            with torch.no_grad():
                out = self._model(t0, t1, timestep=timestep)
                if pad_h or pad_w:
                    out = out[:, :, :h, :w]
                if dtype is not None:
                    out = out.float()
                return out

        try:
            output = _infer()
        except torch.cuda.OutOfMemoryError:
            _get_logger().warning(
                "CUDA OOM interpolating a frame pair; clearing cache and retrying once"
            )
            torch.cuda.empty_cache()
            try:
                output = _infer()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory interpolating frames even after cache clear "
                    "+ retry; try --no-ai"
                ) from None

        return frame_from_tensor(output)

    def interpolate_video(
        self,
        frames: list[Any],
        factor: int = 2,
        progress_callback=None,
    ) -> list[Any]:
        """Interpolate frames to increase framerate by a given factor.

        For `factor > 2`, each inserted frame uses a distinct timestep
        (`j / factor`), so they are genuinely different intermediate frames
        rather than repeated copies of a single fixed-midpoint result.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8.
            factor: Interpolation factor (2 = double framerate).
            progress_callback: Optional callback(current, total, message).

        Returns:
            List of interpolated frames with factor-x the original count.
        """
        if factor <= 1 or len(frames) < 2:
            return frames

        result: list[Any] = [frames[0]]
        total_pairs = len(frames) - 1
        total_out = total_pairs * factor + 1
        done = 0

        for i in range(total_pairs):
            for j in range(1, factor):
                timestep = j / factor
                frame = self.interpolate(frames[i], frames[i + 1], timestep=timestep)
                result.append(frame)
                done += 1
                if progress_callback:
                    progress_callback(done, total_out, f"Interpolating frame {done}/{total_out}")
            result.append(frames[i + 1])
            done += 1
            if progress_callback:
                progress_callback(done, total_out, f"Interpolating frame {done}/{total_out}")

        return result

    def unload(self) -> None:
        """Release model from memory."""
        if self.backend == "ncnn":
            if self._ncnn_backend is not None:
                self._ncnn_backend.unload()
                self._ncnn_backend = None
            self._loaded = False
            return
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
