# NCNN backend: wiring needed outside `ai/`

Everything inside `ai/` is done and self-contained (see the implementing agents' reports for
verification details and benchmark numbers). Both the upscale/ncnn and RIFE/ncnn backends are
verified working -- RIFE's ncnn backend uses a separate package, `rife-ncnn-vulkan-python`, from
the generic `ncnn` bindings the upscale backend uses; see `ai/backends/ncnn_interpolate.py`'s
docstring for why. This file lists the exact, minimal changes needed in the files this
agent was *not* allowed to touch (`core/stages/*.py`, `config.py`, `cli/cli.py`, docs) to actually
expose `stages.<name>.backend: torch | ncnn` to users. Everything below is additive; nothing here
requires changing existing behavior when `backend` is left at its default.

## 1. `config.py` -- add the config key

Add `"backend": "torch"` to `DEFAULTS["stages"]["upscale"]` and
`DEFAULTS["stages"]["interpolate"]` (and, later, `denoise_video`/`deblock` if/when those stages
grow ncnn wrappers too -- out of scope for this pass, see "Not done" below).

```python
DEFAULTS["stages"]["upscale"]["backend"] = "torch"       # "torch" | "ncnn"
DEFAULTS["stages"]["interpolate"]["backend"] = "torch"    # "torch" | "ncnn"
```

## 2. `core/stages/upscale.py` -- pass `backend=` through

In `UpscaleStage._execute_ai()` (around where `RealESRGANUpscaler(...)` is constructed, currently
~line 451), add one line:

```python
upscaler = RealESRGANUpscaler(
    scale=scale_factor,
    model_name=pass_model,
    tta_mode=self._tt_mode,
    device_preference=self.config.get("gpu", "preferred_device", default="auto"),
    tile_size=self._stage_config.get("tile_size", 0),
    backend=self._stage_config.get("backend", "torch"),   # <-- add this
)
```

**Known gap to fix at the same time**: `_execute_ai()` currently calls
`ensure_model_available(pass_model)` *before* constructing `RealESRGANUpscaler` (~line 431), which
checks the torch `MODEL_REGISTRY` (`.pth` files) regardless of backend. When `backend="ncnn"`,
this pre-flight check is checking the wrong registry -- it may fail (blocking the ncnn attempt
before it starts, if no `.pth` happens to be cached) or spuriously succeed (if a `.pth` *is*
cached even though the `.param`/`.bin` aren't). The cleanest fix: skip that pre-flight
`ensure_model_available()` call entirely when `backend == "ncnn"` and let
`RealESRGANUpscaler.load_model()` do its own resolution/download (it already does, via
`ai.model_cache.ensure_ncnn_model_available()`) -- `load_model()` returning `False` already
routes through `_ai_fallback_or_fail()` exactly like every other AI-unavailable path, so no new
error-handling branch is needed, just don't gate on the wrong registry first.

## 3. `core/stages/interpolate.py` -- pass `backend=` through

Same pattern, wherever `RIFEInterpolator(...)` is constructed:

```python
interpolator = RIFEInterpolator(
    model_name=self._ai_model,
    device_preference=self.config.get("gpu", "preferred_device", default="auto"),
    backend=self._stage_config.get("backend", "torch"),   # <-- add this
)
```

Same pre-flight-registry caveat as above applies if `interpolate.py`'s stage also calls
`ensure_model_available()` before constructing the interpolator.

**Practical note**: `backend="ncnn"` on the interpolate stage requires the `rife-ncnn-vulkan-python`
package (the `ncnn` extra, see `pyproject.toml`) -- a different package from the generic `ncnn`
bindings the upscale backend uses. It is verified working. Without that package installed (or
without a Vulkan device), it falls back to torch/traditional per the `ai_fallback` policy, same as
any other AI-unavailable condition.

## 4. `cli/cli.py` -- optional CLI flag (not required, config-only is fine)

If a CLI flag is wanted (matching the existing `--ai`/`--no-ai` style), something like:

```python
@click.option(
    "--upscale-backend", type=click.Choice(["torch", "ncnn"]),
    help="Inference backend for the upscale stage (stages.upscale.backend)",
)
@click.option(
    "--interpolate-backend", type=click.Choice(["torch", "ncnn"]),
    help="Inference backend for the interpolate stage (stages.interpolate.backend)",
)
```
mapped to `stages.upscale.backend` / `stages.interpolate.backend` the same way `--codec` etc. map
to their config keys today. Purely additive -- fine to skip in the first wiring pass and add
later; `config.yaml`-only control is sufficient to satisfy R4.3.

## 5. Docs (`AGENTS.md`, `CHANGELOG.md`, `docs/config.example.yaml`, `docs/ROADMAP.md`)

Ready-to-paste snippets:

**`docs/config.example.yaml`** (under each relevant stage's block):
```yaml
stages:
  upscale:
    backend: torch  # torch | ncnn -- ncnn uses Vulkan (AMD/Intel/iGPU-friendly), in-process via
                     # the `ncnn` Python package; falls back per ai_fallback policy if unavailable.
  interpolate:
    backend: torch  # torch | ncnn -- see upscale.backend. RIFE's ncnn backend uses the
                     # separate `rife-ncnn-vulkan-python` package (part of the `ncnn` extra),
                     # not the generic `ncnn` package upscale uses; verified working.
```

**`AGENTS.md`** -- add a short subsection under "AI/Traditional Method Selection" (or a new
"Backend Selection" subsection right after it):
> Independent of the AI/traditional choice, AI-capable stages that support it (`upscale`,
> `interpolate`) also take `stages.<name>.backend: "torch" | "ncnn"` (default `"torch"`).
> `"ncnn"` runs inference in-process via the `ncnn` Python package + Vulkan instead of
> PyTorch/CUDA -- portable to AMD/Intel/integrated GPUs, no CUDA-matching torch build required.
> Falls back through the same `ai_fallback` policy as any other AI-unavailable condition if the
> required package, a Vulkan device, or the ncnn model files aren't available. Both the
> upscale/ncnn (generic `ncnn` package) and RIFE/ncnn (`rife-ncnn-vulkan-python` package)
> backends are verified working (see `ai/backends/ncnn_interpolate.py`'s docstring for why RIFE
> needs a different package).

**`CHANGELOG.md`** -- new entry:
> Added an optional ncnn/Vulkan inference backend for the upscale and interpolate stages
> (`stages.upscale.backend: ncnn`, `stages.interpolate.backend: ncnn`) -- an alternative to
> PyTorch/CUDA that works on AMD/Intel/integrated GPUs. Upscale runs in-process via the generic
> `ncnn` Python package; RIFE interpolation runs via `rife-ncnn-vulkan-python` (a separate
> package needed for its custom `rife.Warp` ncnn layer). Both are verified working.

**`docs/ROADMAP.md`** -- move "NCNN backend option" (feature 4 in `docs/REQUIREMENTS.md`) from
planned to implemented/verified for both upscale/ncnn and RIFE/ncnn.

## Not done (out of scope for this pass, noted for completeness)

- `denoise_video`/`deblock` stages also use `RealESRGANUpscaler` internally (per AGENTS.md's
  AI/Traditional section) but weren't wired to accept `backend=` at their call sites -- would need
  the same one-line change as #2 above once `denoise_video.py`/`deblock.py` want to expose it.
- ~~`gpu.vulkan_device`~~ -- done: `DEFAULTS["gpu"]["vulkan_device"] = 0` exists, and both
  `RealESRGANUpscaler`/`RIFEInterpolator` (and the stages that construct them) thread it through
  to their respective ncnn backends. Matters on multi-GPU machines where ncnn's default device 0
  isn't the fastest one visible (e.g. an iGPU enumerating before a passed-through dGPU) -- check
  `ncnn.get_gpu_count()`/`ncnn.get_gpu_info(i).device_name()` to find the right index.
- `avf gpu-info` doesn't report ncnn/Vulkan availability. `ai.backends.ncnn_common.
  is_ncnn_available()` / `get_vulkan_gpu_count()` are ready to call from there.
