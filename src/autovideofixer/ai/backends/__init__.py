"""Auto Video Fixer - inference backends for AI-capable stages.

An AI-capable stage (upscale, interpolate) talks to a model through the
existing wrapper classes in `ai/wrappers/` (`RealESRGANUpscaler`,
`RIFEInterpolator`). Those classes now accept a `backend: "torch" | "ncnn"`
constructor argument (default `"torch"`, today's only behavior, unchanged)
and internally delegate to one of the backend implementations in this
package when `backend="ncnn"` is requested:

- `ncnn_upscale.NcnnUpscaleBackend` -- Real-ESRGAN via the generic `ncnn`
  Python bindings + Vulkan. Verified working in this environment (see
  WIRING.md / the implementing agent's report for details and caveats).
- `ncnn_interpolate.NcnnInterpolateBackend` -- RIFE via
  `rife-ncnn-vulkan-python`, a *different* package from the generic `ncnn`
  bindings above: the official RIFE ncnn model graph depends on a custom
  `rife.Warp` ncnn layer that the generic `ncnn` PyPI package does not
  register, but `rife-ncnn-vulkan-python` wraps the upstream C++ tool that
  does. Verified working -- see that module's docstring.

Neither backend module is imported at package-import time -- both `ncnn`
and `rife-ncnn-vulkan-python` are optional dependencies (the `ncnn` extra),
imported lazily inside each backend's own functions, matching the existing
lazy `import torch` pattern used throughout `ai/`.
"""

from __future__ import annotations
