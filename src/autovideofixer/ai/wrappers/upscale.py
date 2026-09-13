"""Auto Video Fixer - Real-ESRGAN upscaling model wrapper.

Implements the RRDBNet architecture used by Real-ESRGAN for
single-image super-resolution, adapted for video processing.

Architecture matches the official BasicSR implementation:
https://github.com/xinntao/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch

from autovideofixer.ai.model_cache import get_model_path

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.upscale")
    return _logger


class TileSpec(NamedTuple):
    """One tile's input crop (with overlap padding) and its placement in the output canvas.

    All coordinates are half-open ranges [start, end) in pixels. `in_*` locate the
    (overlap-padded) crop to feed the model; `out_*` locate where the *unpadded*
    portion of that tile's output belongs in the full-resolution output canvas;
    `crop_*` locate that same unpadded portion within the tile's own output (i.e.
    after running the padded input tile through the model at `out_scale`, crop
    `tensor[..., crop_y0:crop_y1, crop_x0:crop_x1]` before placing it at
    `out_y0:out_y1, out_x0:out_x1`).
    """

    in_y0: int
    in_y1: int
    in_x0: int
    in_x1: int
    out_y0: int
    out_y1: int
    out_x0: int
    out_x1: int
    crop_y0: int
    crop_y1: int
    crop_x0: int
    crop_x1: int


def compute_tile_grid(
    height: int, width: int, tile_size: int, overlap: int, out_scale: int = 1
) -> list[TileSpec]:
    """Compute a grid of overlapping tiles covering a `height` x `width` image.

    Standard tiled-inference layout (matches the approach used by the official
    Real-ESRGAN CLI's ``tile`` option): the image is divided into a grid of
    `tile_size` x `tile_size` cells (the last row/column may be smaller), each
    cell is padded by `overlap` pixels on every side (clamped to the image
    bounds) before being fed to the model, and only the unpadded center portion
    of each tile's output is kept when stitching -- this avoids the seam
    artifacts that plain non-overlapping tiling produces at tile boundaries
    (the model has no receptive-field context right at a hard-cropped edge).

    Args:
        height: Full input image height in pixels.
        width: Full input image width in pixels.
        tile_size: Target size (pixels) of each tile's non-overlap region.
            Must be > 0.
        overlap: Padding (pixels) added on each side of a tile before
            inference. Must be >= 0.
        out_scale: The model's own output scale factor (e.g. 4 for x4plus,
            2 for x2plus) -- output/placement coordinates are `out_scale`
            times the input coordinates.

    Returns:
        List of TileSpec, row-major order, covering the whole image with no
        gaps and no overlaps in the *output* placement regions.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if height <= 0 or width <= 0:
        raise ValueError(f"height/width must be > 0, got {height}x{width}")

    tiles: list[TileSpec] = []
    tiles_y = -(-height // tile_size)  # ceil div
    tiles_x = -(-width // tile_size)

    for ty in range(tiles_y):
        out_y0 = ty * tile_size
        out_y1 = min(out_y0 + tile_size, height)
        in_y0 = max(out_y0 - overlap, 0)
        in_y1 = min(out_y1 + overlap, height)
        for tx in range(tiles_x):
            out_x0 = tx * tile_size
            out_x1 = min(out_x0 + tile_size, width)
            in_x0 = max(out_x0 - overlap, 0)
            in_x1 = min(out_x1 + overlap, width)

            # Where the unpadded region sits within this tile's own
            # (padded-input-sized) output, scaled by the model's output factor.
            crop_y0 = (out_y0 - in_y0) * out_scale
            crop_y1 = crop_y0 + (out_y1 - out_y0) * out_scale
            crop_x0 = (out_x0 - in_x0) * out_scale
            crop_x1 = crop_x0 + (out_x1 - out_x0) * out_scale

            tiles.append(
                TileSpec(
                    in_y0=in_y0,
                    in_y1=in_y1,
                    in_x0=in_x0,
                    in_x1=in_x1,
                    out_y0=out_y0 * out_scale,
                    out_y1=out_y1 * out_scale,
                    out_x0=out_x0 * out_scale,
                    out_x1=out_x1 * out_scale,
                    crop_y0=crop_y0,
                    crop_y1=crop_y1,
                    crop_x0=crop_x0,
                    crop_x1=crop_x1,
                )
            )

    return tiles


def run_tiled_inference(
    tensor: "torch.Tensor",
    tile_size: int,
    overlap: int,
    out_scale: int,
    infer_fn: Callable[["torch.Tensor"], "torch.Tensor"],
    tile_batch_size: int = 1,
) -> "torch.Tensor":
    """Run `infer_fn` over `tensor` (N,C,H,W) tile-by-tile and stitch the result.

    Each tile is padded by `overlap` pixels (clamped at image bounds), passed
    through `infer_fn`, and the unpadded center of its output is written into
    the full-size output canvas.

    Args:
        tile_batch_size: When 1 (default), tiles are processed one at a time
            in a plain loop -- today's behavior, unchanged (each forward pass
            holds only one tile's activations in memory). When > 1, and
            `tensor`'s own batch dim is 1 (a single frame -- the tiling path
            only ever runs on one frame at a time), tiles sharing the same
            padded input shape (`compute_tile_grid` gives interior tiles a
            uniform shape; only the last row/column clamped at the image
            boundary can differ) are grouped and stacked into batches of up
            to `tile_batch_size`, each batch going through ONE `infer_fn`
            call instead of one-per-tile. This is the primary win for large
            frames (e.g. 4K, which at the default tile_size=512 needs a 5x8
            = 40-tile grid): 40 small sequential forward passes each pay
            Python/kernel-launch/host-device-sync overhead that a handful of
            larger batched passes mostly avoid. If `tensor`'s batch dim is
            not 1 (the whole-frame-batching path in
            RealESRGANUpscaler.upscale_batch never reaches here -- it
            deliberately falls back to per-frame processing whenever any
            frame in the chunk needs tiling, precisely to avoid this
            unsupported N>1-frames-plus-tiling combination), tile batching
            is skipped and the original one-tile-at-a-time loop runs
            instead. On a caught CUDA OOM during a tile batch, the batch is
            recursively halved and retried (mirroring the tile-size halving
            retry already used elsewhere in this file), down to batch=1
            which is exactly today's per-tile call.
    """
    n, c, h, w = tensor.shape
    grid = compute_tile_grid(h, w, tile_size, overlap, out_scale=out_scale)

    out = tensor.new_empty((n, c, h * out_scale, w * out_scale))

    def _infer_one(spec: TileSpec) -> None:
        tile_in = tensor[..., spec.in_y0 : spec.in_y1, spec.in_x0 : spec.in_x1]
        tile_out = infer_fn(tile_in)
        out[..., spec.out_y0 : spec.out_y1, spec.out_x0 : spec.out_x1] = tile_out[
            ..., spec.crop_y0 : spec.crop_y1, spec.crop_x0 : spec.crop_x1
        ]

    if tile_batch_size <= 1 or n != 1:
        for spec in grid:
            _infer_one(spec)
        return out

    import torch as _torch

    def _infer_group(specs: list[TileSpec]) -> None:
        if len(specs) == 1:
            _infer_one(specs[0])
            return
        tile_ins = [tensor[..., s.in_y0 : s.in_y1, s.in_x0 : s.in_x1] for s in specs]
        batched_in = _torch.cat(tile_ins, dim=0)
        oom = False
        batched_out = None
        try:
            batched_out = infer_fn(batched_in)
        except _torch.cuda.OutOfMemoryError:
            # Only set a flag here (nothing else) -- the actual cleanup/
            # recursion happens AFTER this try/except block, not inside the
            # except clause itself. While still inside an except clause,
            # Python keeps the exception's traceback alive (sys.exc_info()),
            # which transitively keeps the raising frame's own locals alive
            # too -- including infer_fn's own reference to this same batched
            # tensor -- so a `del batched_in` executed HERE would not
            # actually drop the tensor's last reference yet. Moving the
            # cleanup below, after the try/except has fully exited (and the
            # exception/traceback has been cleared), makes the release real
            # and immediate instead of merely appearing to happen.
            oom = True

        if oom:
            _get_logger().warning(
                f"CUDA OOM on a batch of {len(specs)} tiles; clearing cache and "
                "retrying with the tile batch split in half"
            )
            # Release the failed batch's input tensor BEFORE clearing the
            # allocator cache and recursing -- previously this stayed alive
            # through the entire halving recursion below it, so every retry
            # ran with LESS free VRAM than the attempt that just failed,
            # instead of getting back what the failed attempt would have
            # freed.
            del batched_in
            _torch.cuda.empty_cache()
            mid = len(specs) // 2
            _infer_group(specs[:mid])
            _infer_group(specs[mid:])
            return

        # oom is False here, so infer_fn() above returned normally and
        # batched_out was assigned -- this assert only narrows the type for
        # mypy (which can't otherwise see that `oom`/`batched_out` are set
        # together), it's not reachable as a real failure.
        assert batched_out is not None
        for j, spec in enumerate(specs):
            tile_out = batched_out[j : j + 1]
            out[..., spec.out_y0 : spec.out_y1, spec.out_x0 : spec.out_x1] = tile_out[
                ..., spec.crop_y0 : spec.crop_y1, spec.crop_x0 : spec.crop_x1
            ]
        # Release the successfully-stitched batch's tensors immediately
        # rather than waiting for the next iteration's rebinding to drop the
        # last reference -- keeps peak VRAM usage tighter across a long run
        # of many tile groups, same rationale as the OOM-path release above.
        del batched_in, batched_out

    # Group tiles by their padded input shape -- most interior tiles share
    # one shape; only the last row/column (clamped at the image boundary)
    # may be smaller and end up in their own group(s).
    groups: dict[tuple[int, int], list[TileSpec]] = {}
    for spec in grid:
        shape_key = (spec.in_y1 - spec.in_y0, spec.in_x1 - spec.in_x0)
        groups.setdefault(shape_key, []).append(spec)

    for specs in groups.values():
        n = len(specs)
        full_batches = (n // tile_batch_size) * tile_batch_size
        for i in range(0, full_batches, tile_batch_size):
            _infer_group(specs[i : i + tile_batch_size])
        # Remainder tiles (fewer than tile_batch_size left in this shape
        # group) go through the single-tile path instead of one last
        # partial-size batch: a partial batch produces a batched tensor
        # whose N differs from every other batched call made during this
        # run, and the CUDA caching allocator buckets by exact tensor shape
        # -- a one-off N fragments the allocator instead of reusing blocks
        # already sized for the uniform tile_batch_size batches. Processing
        # the remainder tile-by-tile (each call reuses the existing
        # single-tile-shape bucket) keeps every allocation shape seen during
        # tiled inference uniform. Simpler than padding the remainder up to
        # tile_batch_size with dummy tiles and slicing them back off, with
        # no difference in the stitched output either way.
        for spec in specs[full_batches:]:
            _infer_one(spec)

    return out


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


def resolve_arch(registry_entry: dict[str, Any]) -> str:
    """Return the architecture key ("rrdb" or "srvgg") for a MODEL_REGISTRY entry.

    Pure lookup, factored out of `RealESRGANUpscaler.load_model()` so the
    dispatch condition (which architecture a given model name resolves to)
    is unit-testable without needing actual model weights on disk. Entries
    with no explicit "arch" field (every RRDBNet-based model registered
    before the compact SRVGG models were added) default to "rrdb", so
    existing model names are unaffected.
    """
    return str(registry_entry.get("arch", "rrdb"))


class SRVGGNetCompact(torch.nn.Module):
    """SRVGGNetCompact architecture (Real-ESRGAN compact video models).

    Matches the official BasicSR implementation exactly (param names follow
    the flat `body.N` ModuleList convention of the released checkpoints, so
    strict state-dict loading works): a plain conv3x3 + PReLU stack of
    `num_conv` hidden layers, a final conv to num_out_ch*upscale^2 channels,
    PixelShuffle(upscale), and a nearest-upsampled residual add of the input.

    ~1.2M params at num_conv=32 (realesr-general-x4v3) vs RRDBNet's ~16.7M --
    an order-of-magnitude-plus less compute per frame, which is what matters
    for the AI stages since they are measured ~100% GPU-forward-bound.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
    ):
        super().__init__()
        self.upscale = upscale

        self.body = torch.nn.ModuleList()
        self.body.append(torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        self.body.append(torch.nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = torch.nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # Residual over the nearest-upsampled input: the network only has to
        # learn the correction, which is why so few parameters suffice.
        base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")
        result: torch.Tensor = out + base
        return result


class RealESRGANUpscaler:
    """Real-ESRGAN model wrapper for video upscaling.

    Wraps the RRDBNet architecture with frame-by-frame inference,
    proper color space handling, and optional test-time augmentation.

    Usage:
        upscaler = RealESRGANUpscaler(scale=4)
        upscaler.load_model("/path/to/RealESRGAN_x4plus.pth")
        result = upscaler.upscale(frame, tta_mode=7)
    """

    # Below this many input pixels, a whole-frame forward pass is safe on a
    # 16GB-class card even at native x4 (tail activations reach 16x this
    # pixel count at 64 channels). Above it, tile automatically rather than
    # rely solely on the reactive OOM handler -- proactively avoiding the
    # OOM is strictly cheaper than triggering, catching, and retrying one.
    # ~1280x720 (921,600px) at x4plus's 16x tail comfortably fits; this
    # threshold (2,097,152 = 2048x1024) leaves headroom below the point
    # 1920x1080 (2,073,600px) would auto-tile through x4plus, matching the
    # resolution at which OOM was actually observed.
    AUTO_TILE_THRESHOLD_PX = 2_097_152

    DEFAULT_TILE_SIZE = 512
    DEFAULT_TILE_OVERLAP = 32

    def __init__(
        self,
        scale: float = 4,
        model_name: str = "RealESRGAN_x4plus",
        tta_mode: int = 0,
        batch_size: int = 1,
        device_preference: str = "auto",
        tile_size: int = 0,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        tile_batch_size: int = 1,
        backend: str = "torch",
        vulkan_device: int = 0,
    ):
        """
        Args:
            backend: "torch" (default -- unchanged existing behavior) or
                "ncnn". "ncnn" delegates every method below to
                `ai.backends.ncnn_upscale.NcnnUpscaleBackend` instead of
                running the torch/RRDBNet code in this class -- the torch
                path is untouched either way, this only decides which
                implementation `load_model()`/`upscale()`/`unload()` run.
                See ai/WIRING.md for how a stage would plumb a config
                value through to this parameter.
        """
        self.scale = scale
        self.model_name = model_name
        self.tta_mode = tta_mode
        self.batch_size = batch_size
        self.device_preference = device_preference
        # tile_size semantics:
        #   0 (default): "auto" -- no fixed tiling is forced, but a large
        #     enough frame (see AUTO_TILE_THRESHOLD_PX) or a caught CUDA OOM
        #     still triggers tiling automatically at DEFAULT_TILE_SIZE (or
        #     smaller, on a second OOM).
        #   > 0: always tile at this size (skips the auto-threshold check).
        self.tile_size = tile_size
        self.tile_overlap = max(0, tile_overlap)
        # Number of tiles (sharing the same padded input shape) batched into
        # one forward pass inside run_tiled_inference(). 1 (default) = today's
        # behavior, one tile per forward call. This is the primary batching
        # knob for large (e.g. 4K) frames, which always take the tiling path
        # regardless of `batch_size` below -- see run_tiled_inference's
        # docstring. Separate from `batch_size` (whole-*frame* batching,
        # N different frames in one forward pass) since the two batch along
        # different axes and are mutually exclusive per call (upscale_batch()
        # falls back to per-frame processing -- which is where tile batching
        # kicks in -- whenever any frame in the chunk needs tiling).
        self.tile_batch_size = max(1, tile_batch_size)
        self.backend = backend
        self.vulkan_device = vulkan_device
        self._ncnn_backend: Any = None
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
            model_path: Path to .pth file (torch backend) or ignored
                (ncnn backend resolves its own .param/.bin via the ncnn
                model registry -- see ai/backends/ncnn_upscale.py).

        Returns:
            True if model loaded successfully. Never raises: an
            ncnn-specific failure (bindings missing, no such model, a
            graph load error) is logged and returns False here, exactly
            like every existing torch failure path below, so a calling
            stage's `_ai_fallback_or_fail()` handling needs no changes to
            cover this backend too.
        """
        if self.backend == "ncnn":
            from autovideofixer.ai.backends.ncnn_upscale import NcnnUpscaleBackend

            self._ncnn_backend = NcnnUpscaleBackend(
                model_name=self.model_name,
                scale=self.scale,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                vulkan_device=self.vulkan_device,
            )
            self._loaded = self._ncnn_backend.load_model()
            return self._loaded

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

        registry_entry = MODEL_REGISTRY.get(self.model_name, {})
        self._native_scale = registry_entry.get("scale", 4)

        arch = resolve_arch(registry_entry)
        if arch == "srvgg":
            # Compact video models (realesr-general-x4v3 etc.): plain conv
            # stack, no pixel-unshuffle -- `scale` is a genuine PixelShuffle
            # factor here, and num_conv comes from the registry because the
            # released checkpoints differ (32 for general-x4v3, 16 for
            # animevideov3).
            self._model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=num_feat,
                num_conv=registry_entry.get("num_conv", 32),
                upscale=int(self._native_scale),
            )
        else:
            self._model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=num_feat,
                num_block=num_block,
                num_grow_ch=32,
                scale=int(self._native_scale),
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

        if arch == "srvgg":
            _get_logger().info(
                f"Loaded {self.model_name} (SRVGG compact, "
                f"{registry_entry.get('num_conv', 32)} convs) on {self._device}"
            )
        else:
            _get_logger().info(
                f"Loaded {self.model_name} ({num_block} RRDB blocks) on {self._device}"
            )
        return True

    def _needs_tiling(self, height: int, width: int) -> bool:
        """Whether a frame of this size would take the tiled-inference path.

        Shared by `upscale()` (per-frame) and `upscale_batch()` (whole-frame
        batching, which must fall back to per-frame processing -- not attempt
        to combine whole-frame batching with tiling -- whenever any frame in
        a chunk needs this).
        """
        forced_tile = self.tile_size if self.tile_size > 0 else 0
        auto_tile = (
            forced_tile == 0
            and self._device is not None
            and self._device.type == "cuda"
            and height * width > self.AUTO_TILE_THRESHOLD_PX
        )
        return bool(forced_tile or auto_tile)

    def upscale(
        self,
        frame: Any,  # numpy array (H, W, 3) BGR uint8
        tta_mode: int | None = None,
        timer: Any = None,
    ) -> Any:
        """Upscale a single frame using Real-ESRGAN.

        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR, uint8.
            tta_mode: Test-time augmentation mode (0=off, 1, 2, 4, 7, 8, 15, etc.).
                      If None, uses the configured tta_mode.
            timer: Optional `ai.frame_processor.StageTimer` -- when given,
                H2D+preprocess (`tensor_from_frame`), GPU-forward (every
                model forward call, whole-frame or tiled -- honest
                `torch.cuda.Event`-based device time on CUDA), and
                D2H+postprocess (`frame_from_tensor`) durations are recorded
                into it. `None` (default) skips all instrumentation
                overhead entirely.

        Returns:
            Upscaled frame as numpy array (H*scale, W*scale, 3) in BGR, uint8.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.backend == "ncnn":
            assert self._ncnn_backend is not None
            return self._ncnn_backend.upscale(frame)

        import torch

        from autovideofixer.ai.frame_processor import gpu_forward_timer
        from autovideofixer.ai.torch_utils import (
            apply_tta,
            frame_from_tensor,
            tensor_from_frame,
        )

        if tta_mode is None:
            tta_mode = self.tta_mode

        if timer is not None:
            with timer.phase("h2d_preprocess"):
                tensor = tensor_from_frame(frame, device=self._device)
        else:
            tensor = tensor_from_frame(frame, device=self._device)
        if self._use_fp16:
            tensor = tensor.half()
        if self._device.type == "cuda":
            # Match the channels_last layout the model was converted to in
            # load_model() -- passing a contiguous (NCHW) tensor into a
            # channels_last model forces cudnn to convert it internally on
            # every call anyway, defeating the point.
            tensor = tensor.to(memory_format=torch.channels_last)

        def _infer_whole(t: Any) -> Any:
            with torch.no_grad():
                if timer is not None:
                    with gpu_forward_timer(self._device) as gt:
                        if tta_mode and tta_mode >= 1:
                            result = apply_tta(self._model, t, mode=tta_mode)
                        else:
                            result = self._model(t)
                    timer.record("gpu_forward", gt.elapsed_sec)
                    return result
                if tta_mode and tta_mode >= 1:
                    return apply_tta(self._model, t, mode=tta_mode)
                return self._model(t)

        def _infer_tiled(t: Any, tile_size: int, overlap: int) -> Any:
            from autovideofixer.ai.wrappers.upscale import run_tiled_inference

            return run_tiled_inference(
                t,
                tile_size=tile_size,
                overlap=overlap,
                out_scale=int(self._native_scale),
                infer_fn=_infer_whole,
                # TTA is a different batching axis (8 augmented views of ONE
                # tile) than batching N different tiles -- keep tile batching
                # off when TTA is enabled rather than trying to combine them.
                tile_batch_size=1 if (tta_mode and tta_mode >= 1) else self.tile_batch_size,
            )

        _, _, tensor_h, tensor_w = tensor.shape
        forced_tile = self.tile_size if self.tile_size > 0 else 0
        use_tile_size = forced_tile or (
            self.DEFAULT_TILE_SIZE if self._needs_tiling(tensor_h, tensor_w) else 0
        )

        try:
            if use_tile_size > 0:
                output = _infer_tiled(tensor, use_tile_size, self.tile_overlap)
            else:
                output = _infer_whole(tensor)
        except torch.cuda.OutOfMemoryError:
            # A single oversized/high-res frame can exceed VRAM even though prior
            # frames fit (or a static resolution-based auto-tile threshold didn't
            # trigger for this particular frame's actual peak usage). Clear the
            # allocator cache and retry with tiling instead of aborting the whole
            # video on one frame -- start from whatever tile size was already in
            # play (or the default), halved, and halve again once more if it's
            # still not enough.
            _get_logger().warning(
                "CUDA OOM upscaling a frame; clearing cache and retrying with tiling"
            )
            torch.cuda.empty_cache()
            retry_tile = (use_tile_size or self.DEFAULT_TILE_SIZE) // 2
            retry_tile = max(retry_tile, 64)
            output = None
            last_exc: BaseException | None = None
            for _attempt in range(2):
                try:
                    output = _infer_tiled(tensor, retry_tile, self.tile_overlap)
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_exc = exc
                    torch.cuda.empty_cache()
                    retry_tile = max(retry_tile // 2, 64)
            if output is None:
                raise RuntimeError(
                    "CUDA out of memory upscaling frame even after cache clear + "
                    "tiled retries; try a smaller scale_factor, a smaller "
                    "tile_size, or --no-ai"
                ) from last_exc

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
        if timer is not None:
            with timer.phase("d2h_postprocess"):
                result = frame_from_tensor(output, scale=correction)
        else:
            result = frame_from_tensor(output, scale=correction)
        return result

    def upscale_batch(
        self,
        frames: list[Any],  # list of numpy arrays (H, W, 3) BGR uint8
        tta_mode: int | None = None,
        timer: Any = None,
    ) -> list[Any]:
        """Upscale N different frames in one batched forward pass.

        Secondary/parallel deliverable to the primary tile-batching change in
        run_tiled_inference() -- this batches along a different axis (N
        different frames in one call) and only helps videos whose frames are
        small enough to skip the tiled path (a 4K frame ALWAYS takes tiling
        at the default AUTO_TILE_THRESHOLD_PX, so this path never even
        triggers for that case -- see upscale_video()).

        Falls back to the existing proven one-frame-at-a-time `upscale()`
        loop (unchanged) for every case explicitly out of scope for a first
        pass:
          - A chunk of a single frame (no batching benefit anyway).
          - The ncnn backend (not in scope -- torch path only).
          - TTA enabled (`tta_mode >= 1`): that's 8 forward passes on
            augmented views of ONE frame, a different batching axis than
            batching N frames; not combined here.
          - ANY frame in the chunk would need tiled inference (checked via
            `_needs_tiling` before building a batch tensor): tiling and
            whole-frame batching are not combined in this pass -- that
            combination is exactly what the tile_batch_size mechanism in
            run_tiled_inference() now handles per-frame instead.

        On a caught CUDA OOM while running the batched forward pass, the
        batch is recursively split in half and retried (mirroring the
        tile-size/tile-batch halving-retry pattern used elsewhere in this
        file), down to a single-frame chunk which then goes through the
        existing `upscale()` OOM/tiling-retry path unchanged.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8. May all be
                the same shape (required for the batched tensor path) or not
                (falls back to per-frame processing if shapes differ).
            tta_mode: Test-time augmentation mode. If None, uses the
                configured tta_mode.

        Returns:
            List of upscaled numpy arrays, same length and order as `frames`
            (output[i] corresponds to frames[i]).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not frames:
            return []

        if tta_mode is None:
            tta_mode = self.tta_mode

        if (
            len(frames) == 1
            or self.backend == "ncnn"
            or (tta_mode and tta_mode >= 1)
            or any(self._needs_tiling(f.shape[0], f.shape[1]) for f in frames)
        ):
            return [self.upscale(f, tta_mode=tta_mode, timer=timer) for f in frames]

        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            # tensor_from_frames() requires a uniform shape to stack; a mixed
            # chunk (shouldn't normally happen -- frames in one video share a
            # resolution -- but don't assume it) falls back per-frame too.
            return [self.upscale(f, tta_mode=tta_mode, timer=timer) for f in frames]

        import torch

        from autovideofixer.ai.frame_processor import gpu_forward_timer
        from autovideofixer.ai.torch_utils import frames_from_tensor, tensor_from_frames

        def _infer(chunk: list[Any]) -> list[Any]:
            if len(chunk) == 1:
                return [self.upscale(chunk[0], tta_mode=tta_mode, timer=timer)]

            if timer is not None:
                with timer.phase("h2d_preprocess"):
                    tensor = tensor_from_frames(chunk, device=self._device)
            else:
                tensor = tensor_from_frames(chunk, device=self._device)
            if self._use_fp16:
                tensor = tensor.half()
            if self._device.type == "cuda":
                tensor = tensor.to(memory_format=torch.channels_last)

            try:
                with torch.no_grad():
                    if timer is not None:
                        with gpu_forward_timer(self._device) as gt:
                            output = self._model(tensor)
                        timer.record("gpu_forward", gt.elapsed_sec)
                    else:
                        output = self._model(tensor)
            except torch.cuda.OutOfMemoryError:
                _get_logger().warning(
                    f"CUDA OOM on a batch of {len(chunk)} frames; clearing cache "
                    "and retrying with the batch split in half"
                )
                torch.cuda.empty_cache()
                mid = len(chunk) // 2
                return _infer(chunk[:mid]) + _infer(chunk[mid:])

            if self._use_fp16:
                output = output.float()
            correction = self.scale / self._native_scale
            if timer is not None:
                with timer.phase("d2h_postprocess"):
                    return frames_from_tensor(output, scale=correction)
            return frames_from_tensor(output, scale=correction)

        return _infer(frames)

    def upscale_video(
        self,
        frames: list[Any],
        progress_callback=None,
        timer: Any = None,
        operation_label: str = "Upscaling",
    ) -> list[Any]:
        """Upscale a sequence of frames.

        Args:
            frames: List of numpy arrays (H, W, 3) in BGR, uint8.
            progress_callback: Optional callback(current, total, message).
            timer: Optional `ai.frame_processor.StageTimer`, forwarded to
                `upscale()`/`upscale_batch()` -- see `upscale()`'s docstring.
            operation_label: Verb used in the progress message (e.g.
                "Upscaling frame N/total"). This wrapper's Real-ESRGAN
                implementation is shared by the `upscale`, `deblock`, and
                `denoise_video` stages, so the default "Upscaling" is only
                accurate for the `upscale` stage -- callers running it as a
                deblock or denoise pass should pass "Deblocking"/"Denoising"
                so progress messages describe the stage actually running.

        Returns:
            List of upscaled numpy arrays.
        """
        results = []
        total = len(frames)

        if self.batch_size <= 1:
            # Today's behavior, unchanged.
            for i, frame in enumerate(frames):
                result = self.upscale(frame, timer=timer)
                results.append(result)

                if progress_callback and total > 0:
                    progress_callback(i + 1, total, f"{operation_label} frame {i + 1}/{total}")

            return results

        processed = 0
        for i in range(0, total, self.batch_size):
            chunk = frames[i : i + self.batch_size]
            chunk_results = self.upscale_batch(chunk, timer=timer)
            results.extend(chunk_results)
            processed += len(chunk)

            if progress_callback and total > 0:
                progress_callback(processed, total, f"{operation_label} frame {processed}/{total}")

        return results

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
            self._use_fp16 = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
