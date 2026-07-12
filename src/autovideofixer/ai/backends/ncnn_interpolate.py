"""Auto Video Fixer - RIFE frame interpolation via ncnn/Vulkan.

Status: functional. Uses `rife-ncnn-vulkan-python` (PyPI), a SWIG wrapper around the official
upstream `rife-ncnn-vulkan` C++ project (github.com/nihui/rife-ncnn-vulkan, packaged for pip by
github.com/K4YT3X/rife-ncnn-vulkan-python) -- NOT the generic `ncnn` PyPI package that
`ncnn_upscale.py`/`ncnn_common.py` use. That distinction is the whole story here: RIFE's official
ncnn graph (see `ai/model_cache.py:NCNN_MODEL_REGISTRY["rife_v4.6"]`) references a custom ncnn
layer type, `rife.Warp`, implemented in the upstream C++ project's own source and compiled
directly into it. The generic `ncnn` package's Python bindings (`ncnn.Net`/`Extractor`) never
register that layer, so loading the same `.param`/`.bin` pair through them fails with "layer
rife.Warp not exists or registered" -- this was this module's previous, documented dead end.
`rife-ncnn-vulkan-python` bundles the compiled custom layer inside its own extension module
instead of relying on a generic binding to know about it, so it loads and runs correctly.

Verified working in the implementing environment (RTX 5060 Ti, Vulkan device index 1): loading
our own cached, hash-verified `rife_v4.6` ncnn model into `rife_ncnn_vulkan_python.Rife` and
calling `Rife.process()` on two real, distinct frames produces genuinely blended, non-degenerate
output (see the implementing report for the full pipeline-level benchmark).

Model directory note: `rife_ncnn_vulkan_python.Rife._load()` takes a *directory* (not individual
file paths), requires files literally named `flownet.param`/`flownet.bin` inside it, and picks
the model architecture (rife-v2/v3 vs rife-v4 vs the original two-network rife/HD/anime/UHD
family) by checking the directory *path string* for a `"rife-v2"`/`"rife-v3"`/`"rife-v4"`
substring -- there is no separate explicit architecture argument (see its
`rife_ncnn_vulkan.py::Rife.__init__`/`_load()`). Our `NCNN_MODEL_REGISTRY["rife_v4.6"]` entry
downloads nihui's official "rife-v4" release asset -- confirmed the same single-flownet
architecture as the package's own bundled `models/rife-v4/` directory, which (unlike the older
variants) contains only `flownet.{param,bin}`, no `contextnet`/`fusionnet`. `_prepare_model_dir()`
below bridges the naming mismatch: `get_ncnn_model_paths()` returns flat files named
`rife_v4.6.param`/`rife_v4.6.bin`, so this (re)creates a `rife-v4/` directory under the ncnn model
cache with `flownet.param`/`flownet.bin` symlinked (falling back to copying on platforms/
filesystems without symlink support) to those cached files -- cheap, idempotent, and the
`"rife-v4"` directory name is what makes `rife_ncnn_vulkan_python` select the matching
architecture. The package separately bundles its own `rife-v4` model under
`<site-packages>/rife_ncnn_vulkan_python/models/rife-v4/` (a different, slightly older training
checkpoint than our cached `rife_v4.6`/RIFEv4.26 release); `load_model()` always prefers our own
hash-verified cache, so the model version used here matches the torch backend's `rife_v4.6`
default rather than silently drifting to the bundled one.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("autovideofixer.ai.backends.ncnn_interpolate")
    return _logger


def is_rife_ncnn_available() -> bool:
    """Check whether the `rife_ncnn_vulkan_python` package is installed and importable.

    Mirrors `ncnn_common.is_ncnn_available()`'s narrow "is the package there" scope -- does NOT
    check for a usable Vulkan device (a missing/broken Vulkan driver still imports fine; see
    `load_model()`, which resolves `gpuid=-1` -- CPU inference -- whenever
    `ncnn_common.get_vulkan_gpu_count()` reports 0, the same posture `ncnn_common.make_net()`
    has for the upscale backend).
    """
    try:
        import rife_ncnn_vulkan_python  # noqa: F401

        return True
    except ImportError:
        return False


class NcnnInterpolateBackend:
    """RIFE ncnn/Vulkan backend, matching RIFEInterpolator's frame API.

    Delegates to `rife_ncnn_vulkan_python.Rife` (see module docstring for why this package, not
    the generic `ncnn` package `NcnnUpscaleBackend` uses). `interpolate()`'s frames are BGR
    uint8 (H, W, 3) numpy arrays, matching the rest of `ai/` (see `ai/frame_processor.py`);
    `Rife.process()` takes/returns PIL Images in RGB, so this converts both directions per call
    (mirroring the BGR<->RGB handling `ncnn_common.frame_to_mat()`/`mat_to_frame()` do for the
    other backend, just via PIL instead of an `ncnn.Mat`).

    Usage mirrors `NcnnUpscaleBackend` deliberately:

        backend = NcnnInterpolateBackend(model_name="rife_v4.6")
        if backend.load_model():
            out = backend.interpolate(frame0, frame1, timestep=0.5)
            backend.unload()
    """

    def __init__(
        self,
        model_name: str = "rife_v4.6",
        use_vulkan: bool = True,
        vulkan_device: int = 0,
    ):
        self.model_name = model_name
        self.use_vulkan = use_vulkan
        self.vulkan_device = vulkan_device
        self._rife: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, param_path: str | None = None, bin_path: str | None = None) -> bool:
        """Load the RIFE ncnn model via `rife_ncnn_vulkan_python.Rife`.

        If `param_path`/`bin_path` are not given, resolves and downloads them from
        `NCNN_MODEL_REGISTRY` via `model_cache.py` (mirrors
        `NcnnUpscaleBackend.load_model()`'s `get_ncnn_model_paths()` fallback). Returns False
        (never raises) on any failure -- missing package, no cached model, or a `Rife()`
        construction error -- so the calling stage's existing `_ai_fallback_or_fail()` handling
        (triggered on `load_model()` returning False) works unchanged for this backend too.
        """
        if not is_rife_ncnn_available():
            _get_logger().error(
                "rife_ncnn_vulkan_python package not installed - cannot use ncnn RIFE backend "
                "(pip install 'auto-video-fixer[ncnn]'; building it from source requires the "
                "swig system package and, on modern CMake, CMAKE_POLICY_VERSION_MINIMUM=3.5 -- "
                "see pyproject.toml's ncnn extra for the exact command)"
            )
            return False

        if param_path is None or bin_path is None:
            from autovideofixer.ai.model_cache import (
                ensure_ncnn_model_available,
                get_ncnn_model_paths,
            )

            success, msg = ensure_ncnn_model_available(self.model_name)
            if not success:
                _get_logger().error(f"ncnn RIFE model unavailable: {msg}")
                return False
            paths = get_ncnn_model_paths(self.model_name)
            if paths is None:
                _get_logger().error(
                    f"ncnn RIFE model {self.model_name} reported available but paths not found"
                )
                return False
            param_path, bin_path = str(paths[0]), str(paths[1])

        try:
            model_dir = self._prepare_model_dir(param_path, bin_path)
        except Exception as e:
            _get_logger().error(f"Failed to prepare ncnn RIFE model directory: {e}")
            return False

        from autovideofixer.ai.backends.ncnn_common import get_vulkan_gpu_count

        gpuid = -1
        if self.use_vulkan:
            gpu_count = get_vulkan_gpu_count()
            if gpu_count > 0:
                gpuid = self.vulkan_device
            else:
                _get_logger().warning(
                    "ncnn RIFE backend: Vulkan requested but no Vulkan device visible "
                    "(ncnn.get_gpu_count() == 0); falling back to CPU inference (gpuid=-1)"
                )

        try:
            import rife_ncnn_vulkan_python as rife_ncnn

            rife_obj = rife_ncnn.Rife(
                gpuid=gpuid,
                model=str(model_dir),
                scale=2,
                tta_mode=False,
                tta_temporal_mode=False,
                uhd_mode=False,
                num_threads=1,
            )
        except Exception as e:
            _get_logger().error(f"Failed to load ncnn RIFE model {self.model_name}: {e}")
            return False

        self._rife = rife_obj
        self._loaded = True
        _get_logger().info(
            f"Loaded ncnn RIFE model {self.model_name} from {model_dir} "
            f"(gpuid={gpuid}, vulkan={'yes' if gpuid != -1 else 'no'})"
        )
        return True

    @staticmethod
    def _prepare_model_dir(param_path: str, bin_path: str) -> Path:
        """Build (or reuse) the `rife-v4/flownet.{param,bin}` dir `Rife._load()` expects.

        See module docstring for why this indirection is needed: our cache stores
        `<logical_name>.param`/`.bin` flat files, but `rife_ncnn_vulkan_python` wants a
        directory containing exactly-named `flownet.param`/`flownet.bin` members, selecting the
        model architecture from the directory *name* itself.
        """
        from autovideofixer.ai.model_cache import get_ncnn_model_dir

        target_dir = get_ncnn_model_dir() / "rife-v4"
        target_dir.mkdir(parents=True, exist_ok=True)

        for member_name, src in (("flownet.param", param_path), ("flownet.bin", bin_path)):
            dest = target_dir / member_name
            src_path = Path(src).resolve()
            if dest.is_symlink() or dest.exists():
                already_linked = False
                if dest.is_symlink():
                    try:
                        already_linked = dest.resolve() == src_path
                    except OSError:
                        already_linked = False
                if already_linked:
                    continue
                dest.unlink()
            try:
                dest.symlink_to(src_path)
            except OSError:
                shutil.copyfile(src_path, dest)

        return target_dir

    def is_available(self) -> bool:
        """Real (not hardcoded) availability probe: attempt a full load and unload.

        Distinct from `load_model()` only in that it always cleans up after itself -- useful for
        `avf gpu-info`-style capability reporting without leaving a loaded model around.
        """
        try:
            ok = self.load_model()
        finally:
            self.unload()
        return ok

    def interpolate(self, frame0: Any, frame1: Any, timestep: float = 0.5) -> Any:
        if not self._loaded or self._rife is None:
            raise RuntimeError("ncnn RIFE model not loaded. Call load_model() first.")
        return self._infer_pair(frame0, frame1, timestep)

    def _infer_pair(self, frame0: Any, frame1: Any, timestep: float) -> Any:
        import cv2
        import numpy as np
        from PIL import Image

        img0 = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
        img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        out_img = self._rife.process(img0, img1, timestep=timestep)
        out_rgb = np.asarray(out_img)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def unload(self) -> None:
        self._rife = None
        self._loaded = False

    def __del__(self) -> None:
        try:
            self.unload()
        except Exception:
            pass
