"""Auto Video Fixer - Real-ESRGAN inference via ncnn/Vulkan.

Loads the official ncnn `.param`/`.bin` Real-ESRGAN graph (see
`ai/model_cache.py:NCNN_MODEL_REGISTRY`) through the generic `ncnn` PyPI
Python bindings and runs it in-process, frame by frame -- no subprocess
spawned per frame or per chunk (see AGENTS.md/WIRING.md: this must slot
into the existing chunked `stream_frames_prefetched()`/`AsyncVideoWriter`
loop the same way `RealESRGANUpscaler.upscale()` already does for torch).

Verified working in the implementing environment: a 64x64 random RGB
frame through the official `realesrgan-x4plus.param`/`.bin` produces a
256x256 output with a plausible value range via Vulkan on an AMD iGPU
(RADV RAPHAEL_MENDOCINO) -- see the implementing agent's report for the
full verification and a torch-vs-ncnn benchmark.
"""

from __future__ import annotations

import logging
from typing import Any

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.backends.ncnn_upscale")
    return _logger


class NcnnUpscaleBackend:
    """Real-ESRGAN ncnn/Vulkan backend, matching RealESRGANUpscaler's frame API.

    Usage mirrors the torch wrapper deliberately (same method names/shapes)
    so `RealESRGANUpscaler` can delegate to this with a thin dispatch,
    without either class needing to know about the other's internals beyond
    this shared shape:

        backend = NcnnUpscaleBackend(model_name="RealESRGAN_x4plus")
        if backend.load_model():
            out = backend.upscale(frame)  # BGR uint8 (H,W,3) -> BGR uint8 (H*scale,W*scale,3)
            backend.unload()
    """

    # Same rationale/threshold family as RealESRGANUpscaler.AUTO_TILE_THRESHOLD_PX,
    # but proactive-only (see AGENTS.md R4.5 / WIRING.md): ncnn/Vulkan does not
    # raise a catchable Python OOM exception the way torch.cuda does -- a Vulkan
    # allocation failure can present as a driver-level abort instead, so this
    # threshold is the only tiling trigger for this backend, not a reactive retry.
    AUTO_TILE_THRESHOLD_PX = 2_097_152
    DEFAULT_TILE_SIZE = 512
    DEFAULT_TILE_OVERLAP = 32

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        scale: float = 4,
        tile_size: int = 0,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        use_vulkan: bool = True,
        vulkan_device: int = 0,
    ):
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = max(0, tile_overlap)
        self.use_vulkan = use_vulkan
        self.vulkan_device = vulkan_device
        self._net: Any = None
        self._native_scale = 4
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, param_path: str | None = None, bin_path: str | None = None) -> bool:
        """Load the ncnn Real-ESRGAN graph + weights.

        If `param_path`/`bin_path` are not given, resolves and downloads
        them from `NCNN_MODEL_REGISTRY` via `model_cache.py` (mirrors
        `RealESRGANUpscaler.load_model()`'s `get_model_path()` fallback).
        Returns False (never raises) on any failure -- missing bindings, no
        model, or a graph load error -- so the calling stage's existing
        `_ai_fallback_or_fail()` handling (triggered on `load_model()`
        returning False) works unchanged for this backend too.
        """
        from autovideofixer.ai.backends.ncnn_common import is_ncnn_available, make_net

        if not is_ncnn_available():
            _get_logger().error("ncnn package not installed - cannot use ncnn backend")
            return False

        if param_path is None or bin_path is None:
            from autovideofixer.ai.model_cache import (
                NCNN_MODEL_REGISTRY,
                ensure_ncnn_model_available,
                get_ncnn_model_paths,
            )

            success, msg = ensure_ncnn_model_available(self.model_name)
            if not success:
                _get_logger().error(f"ncnn model unavailable: {msg}")
                return False
            paths = get_ncnn_model_paths(self.model_name)
            if paths is None:
                _get_logger().error(
                    f"ncnn model {self.model_name} reported available but paths not found"
                )
                return False
            param_path, bin_path = str(paths[0]), str(paths[1])
            self._native_scale = int(NCNN_MODEL_REGISTRY.get(self.model_name, {}).get("scale", 4))

        try:
            net = make_net(use_vulkan=self.use_vulkan, vulkan_device=self.vulkan_device)
            ret_param = net.load_param(param_path)
            ret_model = net.load_model(bin_path)
        except Exception as e:
            _get_logger().error(f"Failed to load ncnn Real-ESRGAN model: {e}")
            return False

        if ret_param != 0 or ret_model != 0:
            _get_logger().error(
                f"ncnn load_param/load_model failed for {self.model_name} "
                f"(param ret={ret_param}, model ret={ret_model}); the .param file may "
                "reference a custom layer this ncnn build doesn't register"
            )
            return False

        self._net = net
        self._loaded = True
        _get_logger().info(
            f"Loaded ncnn Real-ESRGAN model {self.model_name} "
            f"(vulkan={'yes' if self.use_vulkan else 'no'})"
        )
        return True

    def upscale(self, frame: Any) -> Any:
        """Upscale a single BGR uint8 (H, W, 3) frame. Raises on inference failure.

        Unlike load_model(), inference-time failures are allowed to raise
        (caught by the stage's existing broad `except Exception` around its
        per-frame AI call, same contract the torch path already relies on
        -- see core/stages/upscale.py's `except Exception as e:` around its
        chunked loop).
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("ncnn model not loaded. Call load_model() first.")

        h, w = frame.shape[0], frame.shape[1]
        forced_tile = self.tile_size if self.tile_size > 0 else 0
        auto_tile = forced_tile == 0 and (h * w) > self.AUTO_TILE_THRESHOLD_PX
        use_tile_size = forced_tile or (self.DEFAULT_TILE_SIZE if auto_tile else 0)

        if use_tile_size > 0:
            from autovideofixer.ai.backends.ncnn_common import tiled_ncnn_inference

            result = tiled_ncnn_inference(
                frame,
                tile_size=use_tile_size,
                overlap=self.tile_overlap,
                out_scale=self._native_scale,
                infer_fn=self._infer_whole,
            )
        else:
            result = self._infer_whole(frame)

        if self._native_scale != self.scale:
            import cv2

            out_h = int(round(h * self.scale))
            out_w = int(round(w * self.scale))
            result = cv2.resize(result, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        return result

    def _infer_whole(self, frame: Any) -> Any:
        from autovideofixer.ai.backends.ncnn_common import frame_to_mat, mat_to_frame

        mat_in = frame_to_mat(frame)
        ex = self._net.create_extractor()
        ex.input("data", mat_in)
        ret, mat_out = ex.extract("output")
        if ret != 0:
            raise RuntimeError(f"ncnn Real-ESRGAN extractor.extract() failed (ret={ret})")
        return mat_to_frame(mat_out)

    def unload(self) -> None:
        self._net = None
        self._loaded = False

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass
