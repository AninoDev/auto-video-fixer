# Auto Video Fixer - Agent Instructions

## Standing rule: keep docs in sync

Any change to behavior, config keys, CLI flags, or stage semantics MUST update
`AGENTS.md`, `CHANGELOG.md`, `docs/` (`ROADMAP.md`, `IMPLEMENTATION.md` as applicable), and
`docs/config.example.yaml` in the **same** change. Docs that drift from the code are worse than
no docs — the next agent (or human) will trust and propagate a stale claim.

## Setup & Commands

```bash
uv sync --all-extras        # install core + dev deps
uv run pytest tests/unit/ -v   # unit tests (no FFmpeg needed by default)
uv run pytest tests/ -v -m "integration"  # integration tests (requires FFmpeg)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ --ignore-missing-imports
uv run avf process video.mp4 -p 1080p60   # CLI entry
```

CI runs: `ruff check` -> `ruff format --check` -> `mypy` -> `pytest tests/unit/` -> `pytest -m integration` (3.14 only).

### Mixed Python/Rust: this is now a mixed-language project

`rust/avf_scenes/` is a PyO3/`maturin` Rust crate (scene-detection frame differencing, see
`docs/REQUIREMENTS.md` R5.1) — the first of the Rust rewrite candidates to land, and the pattern
future Rust work here should follow. It is a `[tool.uv.workspace]` member of the root
`pyproject.toml`, declared as a normal dependency (`avf-scenes`, `[tool.uv.sources]` pointing at
it as an editable workspace path) — **plain `uv sync --all-extras` builds and installs it
automatically**, no separate manual step needed. `rust/avf_scenes/pyproject.toml`'s
`build-system.requires = ["maturin>=1.14,<2.0"]` means uv fetches `maturin` itself into an
isolated PEP 517 build environment on demand (standard build-isolation behavior) — it does **not**
need to be pre-installed in the project's own venv for `uv sync` to work. The one thing that *does*
need to be present on `PATH` at sync time is a Rust toolchain (`cargo`/`rustc`, e.g. via `rustup`
or a system package) — `maturin` shells out to `cargo` to actually compile. If it's missing, `uv
sync` fails outright rather than silently skipping the extension, since `avf-scenes` is a required
dependency of the package, not an optional extra, matching REQUIREMENTS.md's R5.1 integration
plan. (For manual iteration, `uv pip install maturin` puts a runnable `maturin` CLI in the venv --
see the "iterate on the Rust side" note below -- but that's a convenience for `maturin develop`,
not something plain `uv sync` requires.)

At the Python call site, `core/analysis.py`'s `_detect_scene_changes()` still tries `import
avf_scenes` in a `try/except ImportError` and falls back to the original pure-Python/OpenCV
implementation (`_detect_scene_changes_python`) if that import fails for any reason (e.g. a
platform where the wheel didn't build) — this is a defense-in-depth fallback, not the primary way
the extension is expected to be missing day-to-day, since `uv sync` normally guarantees it's
built. `uv run pytest tests/unit/` does **not** need a separate Rust build step beyond the normal
`uv sync` — if you've run that, the extension is already installed into the venv.

To iterate on the Rust side without a full `uv sync` round-trip: `cd rust/avf_scenes && uv run
maturin develop --release` rebuilds and reinstalls just that crate into the current venv.
`cargo test` / `cargo clippy` / `cargo fmt --check` run from `rust/avf_scenes/` for Rust-side
gates (no Rust unit tests exist yet for this crate — correctness is verified via the Python-side
differential test `TestRustPythonParity` in `tests/unit/test_scene_detection.py`, which compares
the Rust and Python implementations' output directly).

A second crate, `rust/avf_hashing/` (perceptual-hash duplicate detection, `docs/REQUIREMENTS.md`
R5.2), followed the same workspace/build pattern (`[tool.uv.workspace]` member, `avf-hashing`
dependency + `[tool.uv.sources]` editable-workspace entry, built automatically by `uv sync
--all-extras`) with one structural difference worth knowing about: **it has real Rust unit tests**
(`cargo test` — DCT-matrix orthonormality, deterministic/distinguishing hashes, majority-vote
tie-breaking), unlike `avf_scenes`. A pyo3 crate built with the `extension-module` feature can't
link into a normal `cargo test` binary (that feature assumes libpython is provided by an embedding
CPython process, not the test harness itself) — undefined-symbol linker errors result if you try.
`rust/avf_hashing/Cargo.toml` works around this by making `extension-module` a Cargo feature
that's default-on (so `cargo build`/`maturin develop`/`uv sync` all still produce a real Python
extension) but can be turned off for testing: `cd rust/avf_hashing && cargo test
--no-default-features`. Do the same for `cargo clippy --no-default-features` if you touch that
crate's tests. `avf_scenes` doesn't need this because it has no Rust-level tests, only the
Python-side differential test described above — if it ever grows `cargo test` cases, apply the
same feature-flag split there too, for consistency.

At the Python call site, `core/analysis.py`'s `compute_video_phash()`/`hash_similarity()` follow
the identical lazy-import-with-fallback pattern as scene detection: `try: from avf_hashing import
...` / `except ImportError`, falling back to a pure-NumPy implementation of the *same* pHash
algorithm (not a different one) so results don't depend on whether the extension is built. See
that module's "Perceptual Hashing & Duplicate Detection" section and `rust/avf_hashing/src/lib.rs`
for the algorithm and why it replaced the old ahash/dhash implementation wholesale rather than
sitting alongside it (no backward-compatibility constraint — see R5.2 in
`docs/REQUIREMENTS.md`).

## Architecture

```
src/autovideofixer/
├── cli/cli.py              # Click CLI (`avf` entry, pyproject.scripts)
├── config.py               # Config + DEFAULTS
├── core/
│   ├── pipeline.py         # Pipeline orchestrator
│   ├── analysis.py         # VideoAnalyzer, VLM, scene detection, clip extraction, dedup
│   ├── ffmpeg_utils.py     # FFmpeg wrappers, probe, run
│   ├── presets.py          # Preset definitions (1080p60, 4k60, etc.)
│   └── stages/             # Processing stages
│       ├── base.py         # BaseStage, register_stage(), StageResult
│       ├── detect.py       # "detect" (analysis, priority=1, produces_output=False)
│       ├── stabilize.py    # "stabilize" — pipes raw YUV420P (decode→transform)
│       ├── crop.py         # "crop" (auto-crop, opt-in, off by default)
│       ├── deblock.py      # "deblock"
│       ├── denoise_video.py# "denoise_video"
│       ├── upscale.py      # "upscale" (AI or traditional)
│       ├── interpolate.py  # "interpolate" (AI or traditional)
│       ├── normalize_audio.py # "normalize_audio" + "normalize_volume" (two classes)
│       ├── encode.py       # "encode" (final stage)
│       ├── remux.py        # "remux" (only added for MKV input)
│       ├── speed.py        # "speed"
│       └── hdr.py          # "hdr"
```

FFmpeg must be in PATH. Verify with `avf gpu-info`.

## Stage Development

1. Create `src/autovideofixer/core/stages/<name>.py`. Subclass `BaseStage`.
2. Set class attributes: `name`, `display_name`, `description`, `category`, `priority`, `supports_gpu`.
3. Implement `should_run(input_info) -> (bool, reason | None)` and `execute(input_path, output_path, progress_callback, **kwargs) -> StageResult`.
4. Use `self._stage_config = config.get("stages", self.name, default={})` for stage config.
5. Report progress via `self._report_progress(0.0..1.0, message, callback)`.
6. **Register** in `src/autovideofixer/core/stages/__init__.py`: import and call `register_stage(MyStage)` at module level.
7. Add config to `DEFAULTS["stages"][<name>]` in `src/autovideofixer/config.py`.

## Pipeline Behavior

- **Stage ordering is hardcoded** in `Pipeline.optimize_stage_order()` (`core/pipeline.py`). Changing `DEFAULTS["pipeline"]["default_order"]` in config has **no effect**. Order:
  `detect, stabilize, crop, deblock, denoise_video, upscale, interpolate, normalize_volume,
  normalize_audio, speed, hdr, encode`. `crop` sits right after `stabilize` and before every
  other enhancement stage: stabilize's zoom-out correction can itself add a black border, so
  cropping after it removes both the original letterboxing/pillarboxing AND any residual
  stabilization border in one pass; running it before deblock/denoise/upscale/interpolate/encode
  means none of those (especially the AI-capable ones) spend compute on pixels about to be
  cropped away.
- **Stage name mismatch**: config `default_order` lists `"denoise"` but the registered name is `"denoise_video"`. Ignoring `default_order` avoids the bug.
- `remux` is **not** in the default pipeline. It is only added by `auto_determine_stages()` when the input is MKV.
- `detect` stage has `priority=1` (runs first) and `produces_output=False`.
- Default: `skip_stage_on_error: true` — pipeline continues on failure using the original input.
- **Temp files always use `.mkv`** (`generate_temp_path()` in `ffmpeg_utils.py`), regardless of
  the input or final output container — intermediate stages hardcode `libx264`, which not every
  container (e.g. WebM) permits, and MKV can hold essentially any codec. Written to
  `general.temp_dir` if set, else next to the input file. Some AI stages additionally use their
  own stage-internal `.mp4` temp file for the frame-extract/mux round trip (see
  `ai/wrappers/*`/stage `_execute_ai()` methods) — unrelated to the pipeline-level `.mkv` temps.
- **Final output extension** comes from `general.output_container` (default `"mp4"`), not from
  the input's extension — set via `Pipeline.add_job()` when no explicit `output_path` is given.
  `None`/empty keeps the input's own extension.
- **Temp cleanup is unconditional**, including on failure/cancel (`_cleanup_temp_files` /
  `_cleanup_generated_temp_paths` run in a `finally` in `Pipeline.execute_job()`). If the terminal
  stage was skipped or produced no output but an earlier stage's temp file is the best available
  result, that temp is *promoted* (moved, not copied — `_promote_temp_to_output()`) onto
  `job.output_path` instead of being deleted.
- `scan_directory()` (`core/analysis.py`) skips hidden files (dotfiles), including orphaned
  `.avf_*` intermediate temp files that may be left next to an input after a crash.
- Stage outputs chain: each stage's `output_path` becomes the next stage's `input_path`.

## Output Path Resolution

Output path is resolved in this priority order:
1. `job.output_path` explicitly set (e.g., via `add_job(input, output)`)
2. `config.get("general", "output_dir")` + `{stem}_enhanced{ext}` — set via CLI `-o` or config.yaml
3. `{input_dir}/{stem}_enhanced{ext}` (fallback)

The CLI `-o` flag sets `general.output_dir` in config. The pipeline reads it when creating jobs.

## AI/Traditional Method Selection

Stages with AI alternatives (upscale, denoise_video, interpolate, deblock) check `config.get("general", "use_ai")`:
- `None` (default) — use preset or auto-detect
- `True` (`--ai`) — force AI methods
- `False` (`--no-ai`) — force traditional methods (much faster, no GPU needed)

CLI: `--ai` / `--no-ai` (mutually exclusive flag_value pattern). Without either flag, falls back to preset/config.

### AI-fallback policy

Separately from *whether* to use AI, each AI-capable stage (upscale, interpolate,
denoise_video, deblock) has an **ai_fallback** policy controlling what happens when the AI path
is selected but can't actually run — PyTorch not installed, model download/load failure,
an inference exception, or CUDA OOM after tiling retries are exhausted:

- `general.ai_fallback` (default `True`) is the global default.
- `stages.<name>.ai_fallback` (default `null`) overrides it per-stage; `null` inherits the
  global value.
- `True` (default): silently fall back to the stage's traditional FFmpeg method, logged at
  WARNING.
- `False`: the stage FAILS outright with a named cause (e.g. `"AI processing unavailable for
  stage 'upscale': model not available: ..."`) instead of silently downgrading output quality.
- CLI: `--ai-fallback` / `--no-ai-fallback` overrides `general.ai_fallback` for the run; it does
  **not** override an explicit per-stage `stages.<name>.ai_fallback` set in config.
- Implemented via `BaseStage.is_ai_fallback_enabled()` / `BaseStage._ai_fallback_or_fail()`
  (`core/stages/base.py`) — call `_ai_fallback_or_fail()` at every point the AI path can't
  proceed; do NOT use it for a legitimate OOM-triggered tiling retry (still the AI path) or a
  genuine post-AI failure (e.g. a mux error after AI frames were already produced) — those keep
  failing regardless of this policy.

### Backend selection (torch vs ncnn)

Independent of the AI/traditional choice, the `upscale` and `interpolate` stages also take
`stages.<name>.backend: "torch" | "ncnn"` (default `"torch"`). `"ncnn"` runs inference
in-process via Vulkan instead of PyTorch/CUDA — portable to AMD/Intel/integrated GPUs with no
CUDA-matched torch build required (`ai/backends/`). Both stages use *different* Python packages
for this, both under the `ncnn` extra in `pyproject.toml`: upscale uses the generic `ncnn`
package; interpolate uses `rife-ncnn-vulkan-python`, because RIFE's official ncnn graph needs a
custom `rife.Warp` ncnn layer the generic package doesn't register (see
`ai/backends/ncnn_interpolate.py`'s docstring for the full story) — `rife-ncnn-vulkan-python`
wraps the upstream C++ tool that has it. `rife-ncnn-vulkan-python` has no prebuilt wheel for
Python 3.14 as of this writing; building it from source needs the `swig` system package and,
against modern CMake, `CMAKE_POLICY_VERSION_MINIMUM=3.5` set for the install command — see the
`ncnn` extra's comment in `pyproject.toml` for the exact command. It falls through the same
`ai_fallback` policy as any other AI-unavailable condition when the required package, a Vulkan
device, or the ncnn model files aren't available.

The upscale/ncnn backend is verified working (correct, non-black output) but is **not currently
competitive with `backend: torch`** even on a GPU where CUDA works well — see `docs/ROADMAP.md`'s
NCNN backend section for measured numbers (slower than torch at small frames, unpatched OOM on
plain 720p frames regardless of fp16 `Option` flags, ~36s/frame once tiling is forced to work
around that). Treat `backend: ncnn` for upscale as the AMD/Intel/iGPU portability option, not a
performance option, until the upstream generic Python bindings close those gaps.

RIFE's ncnn backend (`backend: ncnn` on `interpolate`) is verified working via
`rife-ncnn-vulkan-python` and, unlike upscale/ncnn, *is* competitive: ~82 fps at 1280x720 measured
directly against `NcnnInterpolateBackend`, and ~18 fps measured end-to-end through the full
`interpolate` stage (frame extraction + inference + encode) on an RTX 5060 Ti via Vulkan — see
`docs/ROADMAP.md`'s NCNN backend section for the full numbers. It uses the project's own cached,
hash-verified `rife_v4.6` ncnn model (nihui's official "rife-v4" release asset) rather than the
older checkpoint `rife-ncnn-vulkan-python` bundles internally, kept consistent with the torch
backend's default model.

ncnn models (.param/.bin) have their own registry in `ai/model_cache.py`
(`NCNN_MODEL_REGISTRY` / `ensure_ncnn_model_available()`), separate from the torch `.pth`
registry — stage pre-flight model checks must not gate an ncnn run on the torch registry.

### Batched inference (torch backend only)

`RealESRGANUpscaler` (`ai/wrappers/upscale.py`, used by `upscale`/`deblock`/`denoise_video` at
the torch backend) supports two independent, opt-in batching axes, both default `1` (today's
one-at-a-time behavior, unchanged unless configured):

- **`stages.<name>.batch_size`** — N different *frames* stacked into one forward pass
  (`RealESRGANUpscaler.upscale_batch()`/`upscale_video()`, `ai/torch_utils.py`'s
  `tensor_from_frames`/`frames_from_tensor`). Only applies when a frame is small enough to skip
  tiled inference — falls back to one-frame-at-a-time whenever any frame in a chunk would need
  tiling, `tta_mode` is enabled, or the ncnn backend is in use (none of those combine with
  whole-frame batching in this implementation).
- **`stages.<name>.tile_batch_size`** — N *tiles of one frame* stacked into one forward pass
  inside `run_tiled_inference()`, when a frame's resolution triggers tiled inference (see
  `tile_size`/`AUTO_TILE_THRESHOLD_PX` above). This is the batching knob that matters for large
  (e.g. 4K) input: at the default `tile_size=512`, a 4K frame needs a 5x8=40-tile grid, so 40
  small sequential forward passes (each paying Python/kernel-launch/host-device-sync overhead)
  collapse into a handful of larger batched calls. `compute_tile_grid()`'s interior tiles mostly
  share one padded input shape (only the last row/column, clamped at the image boundary, differ),
  so tiles are grouped by shape before being stacked — never mixes shapes into one `torch.cat`.

Both: on a caught `torch.cuda.OutOfMemoryError`, the batch is recursively halved and retried
(mirroring the existing tile-size halving-retry pattern), down to batch=1, which then goes
through the existing single-item OOM/tiling-retry path unchanged. Neither axis is combined with
the other in this pass — a chunk needing both is out of scope (whole-frame batching defers to
per-frame processing, where tile batching then applies per-frame instead). CLI: `--batch-size` /
`--tile-batch-size` set both keys across `upscale`/`deblock`/`denoise_video` at once (no
per-stage CLI override, matching the `--crop-limit` precedent's single-flag-multiple-keys shape
where a per-stage split isn't worth a flag each).

`run_tiled_inference()`'s OOM path (`_infer_group`) releases the failed batch's input tensor
(`del batched_in`) BEFORE clearing the CUDA allocator cache and recursing into the halved retry —
done outside the `except` clause itself (a flag is set inside `except`, the actual cleanup runs
after the `try/except` has fully exited), because an in-flight exception's traceback keeps the
raising frame's own locals alive, including `infer_fn`'s reference to that same tensor; a `del`
executed while still inside the `except` block would not actually drop the last reference yet.
Previously the tensor stayed alive through the entire halving recursion, so every retry ran with
LESS free VRAM than the attempt that had just failed. Remainder tiles in a shape group (fewer
than `tile_batch_size` left over) are now processed through the single-tile path (`_infer_one`)
rather than one last partial-size batch, keeping every batched-tensor shape seen during a run
uniform instead of fragmenting the CUDA caching allocator with a one-off `N`.

### AI stage temp-encode quality and the chunked streaming path

`upscale`/`deblock`/`denoise_video`/`interpolate`'s AI paths each write AI-processed frames to an
internal temp `.mp4` (`StreamingVideoWriter` or `FrameProcessor.frames_to_video()`), then mux that
temp file's video track against the original audio to produce the stage's real output. Both the
temp encode and the mux pass now take **explicit** `crf`/`preset` args instead of relying on
implicit defaults:
- The temp encode passes `stages.<name>.temp_crf` (new config key, default `16`) and the
  hardcoded preset `"medium"` (not itself a config key). Previously this write had NO `-crf`/
  `-preset` at all, silently landing on libx264's own default (CRF 23).
- The mux pass now uses `-c:v copy` (verified bit-identical to the temp file's video stream via
  stream-hash comparison), NOT a second `-crf 18` re-encode. Previously every AI stage muxed with
  `-c:v libx264 -crf 18`, meaning AI-processed frames were encoded TWICE at two different quality
  levels — a real quality bug (the temp's CRF-23 generation was baked in before the mux's CRF-18
  pass ever saw it) plus a wasted full x264 pass over the whole video for no benefit.
- `interpolate`'s AI/RIFE path had the identical double-encode pattern in its own internal temp +
  mux and got the same fix (also gained its own `stages.interpolate.temp_crf`, default 16) — it's
  architecturally separate from the other three (still buffers frames in a list before writing,
  not the streaming path below) but shared the exact same bug.

All AI-capable videos (`upscale`/`deblock`/`denoise_video`) now go through the chunked streaming
path (`stream_frames_prefetched` + `AsyncVideoWriter(StreamingVideoWriter(...))`, `chunk_size=25`)
regardless of frame count — the old `total_est > 1000` frame-count threshold and its full-buffer
`extract_frames()` → flat list → `frames_to_video()` route are gone from these three stages'
`_execute_ai()`/`_run_single_ai_pass()`. That threshold was resolution-blind: a short but
large-resolution (e.g. 4K) clip could still fall under 1000 frames while materializing its ENTIRE
frame set in RAM (a 33s 4K clip is ~25GB uncompressed), with zero decode/inference/write overlap.
Streaming has no measurable downside for short clips either, so there's no longer a reason to keep
two code paths. `extract_frames()`/`frames_to_video()` themselves are untouched in
`ai/frame_processor.py` — other callers (`interpolate`'s AI path, `frames_to_temp_video()`) still
use them, and they remain the eventual Phase-3 fallback machinery for a planned Rust frame-pipe
rewrite (see `docs/REQUIREMENTS.md` R5.3).

### Pinned staging pool (torch backend, CUDA only)

`ai/torch_utils.py`'s `PinnedStagingPool` (module-level singleton via `get_pinned_staging_pool()`)
holds a small, capped set of persistent `torch.empty(..., pin_memory=True)` CPU tensors, keyed by
`(shape, dtype)`, `SLOTS_PER_KEY=2` round-robin slots per key, `MAX_KEYS=2` (4 pinned tensors
total, LRU-evicted by key). `tensor_from_frame`/`tensor_from_frames` (`ai/torch_utils.py`) copy
the incoming numpy frame bytes into a pool slot (`copy_()`) and do the same `non_blocking=True`
H2D `.to(device)` as before, instead of calling `.pin_memory()` fresh (page-locking a brand-new
host buffer) on every single call — a real per-call cost at e.g. 4K (~25MB/frame). CPU-only
callers (`device.type != "cuda"`) don't touch the pool at all.

**Correctness-critical**: a pool slot is *reused*, and a `non_blocking=True` H2D copy reads from
its pinned host tensor asynchronously — the read is not guaranteed to have finished by the time
the Python call that queued it returns. `get_slot()` therefore records a `torch.cuda.Event` right
after each slot's H2D copy is queued (`record_copy()`) and `synchronize()`s on that event before
handing the same slot out again, so a later `copy_()` can never overwrite host memory the device
is still reading from. Do not remove or weaken this without understanding why — an
overwrite-before-copy race here would corrupt frames silently and nondeterministically, not
crash. See `tests/unit/test_torch_utils.py::TestPinnedStagingPoolCudaCorrectness` (real CUDA,
`@pytest.mark.integration`) and `TestPinnedStagingPoolKeying` (mocked, no GPU needed).

### AI-stage instrumentation (`ai/frame_processor.py`)

`StageTimer` and `gpu_forward_timer()` provide lightweight, always-on timing for the chunked AI
stage loop shared by `upscale`/`deblock`/`denoise_video` — instantiated once per stage run
(`StageTimer("upscale")` etc.) and threaded through as an optional `timer=` kwarg into
`RealESRGANUpscaler.upscale()`/`upscale_batch()`/`upscale_video()` (`ai/wrappers/upscale.py`).
Phases tracked: `decode_wait` (time blocked pulling the next chunk out of
`stream_frames_prefetched`, measured at the stage loop via manual `next()` calls instead of a
plain `for` loop), `h2d_preprocess`/`d2h_postprocess` (`tensor_from_frame(s)`/`frame_from_tensor(s)`
call time), `gpu_forward` (every model forward call — bracketed with `torch.cuda.Event` pairs via
`gpu_forward_timer()` for honest device-side time on CUDA, wall-clock fallback on CPU/MPS),
`write_wait` (time blocked in `AsyncVideoWriter.write()`). Per-chunk breakdown logs at DEBUG
(`StageTimer.end_chunk()`); an aggregate percentage-of-total breakdown logs at INFO once at stage
end (`StageTimer.summary()`). Independently, periodic throughput (`frames_done`, `elapsed_sec`,
`current_fps`) logs at INFO every ~10s or 10 chunks (whichever first) — this exists specifically
so a run killed early by an external timeout (e.g. a truncating `timeout` wrapper) still yields a
valid frames/sec curve instead of nothing; see `docs/REQUIREMENTS.md`'s R5.3 note on the 4K
benchmark this was retroactively needed for. `interpolate`'s RIFE path is NOT instrumented (it
doesn't share this loop shape — see "Kill the ≤1000-frame full-buffering path" below).

## Upscaling & Aspect Ratio

The upscale stage respects `quality.quality_target.keep_aspect_ratio` (default `True`):
- Rotates the preset bounding box to match input orientation (portrait input swaps w↔h)
- Scales to fit within the (possibly rotated) bounding box, preserving pixel count
- Square input uses the shorter preset edge for both dimensions
- Dimensions are rounded to even values (H.264 requirement)

Example: 1080p60 preset (1920×1080) + 9:16 portrait input → scales to ~1080×1920 (portrait).

`UpscaleStage._effective_target_bounds()` (`core/stages/upscale.py`) is the single shared
implementation of this rotation, used by `should_run()`'s "already at target?" gate, `execute()`'s
AI-vs-traditional method selection, and `_calculate_target_dimensions()`'s actual scale
computation — all three must agree on what "at target" means for a given input's orientation.
Previously `should_run()` and the method-selection logic each compared the input's raw resolution
against the *unrotated* preset target directly, while only `_calculate_target_dimensions()` did
the rotation. This meant a portrait input already at its rotated target (e.g. a 1080×1920 input
against a `[1920, 1080]` preset target) was misjudged as needing upscaling: `should_run()` let the
stage proceed, `execute()` picked the AI method, and `_execute_ai()`'s per-pass scale computation
— which rounds each pass to a power of 2 via `2 ** round(log2(sf)) if sf > 1 else 2` — silently
substituted a forced 2x scale whenever a pass's own needed scale computed to `<=1` (i.e. already
at/past target), instead of doing nothing. Real production repro: a 1080×1920 portrait 60fps
input against the `1080p60` preset (target `[1920, 1080]`, `keep_aspect_ratio: true`) ran
`RealESRGAN_x2plus` at 2x anyway, producing a wasted 2160×3840 output (~1fps, over an hour for a
63-second clip, ~99% GPU-forward time per instrumentation — pure wasted compute, since the input
was already exactly at target).

Fixed at three levels, all keyed off `UpscaleStage._SKIP_SCALE_THRESHOLD = 1.05` (5% linear, ~10%
area): `should_run()` and `execute()`'s method selection both compute the orientation-aware needed
scale (`max(bound_w/w, bound_h/h)` against the rotated bound) and treat `<= 1.05` as "already at
target" (skip / pick traditional over AI); `_execute_ai()` additionally guards itself directly
(defense in depth for `_execute_ai()` being invoked outside `should_run()`, e.g. `--stage
upscale`) and, within the per-pass loop, does a cheap `_resize_to_exact()` lanczos resize (or a
zero-loss `_copy_stream()` stream-copy if dimensions already match exactly) instead of an AI pass
whenever a pass's own computed `sf <= 1`. The 1.05 threshold specifically covers the "crop stage
shaved a few px off before upscale ran" case (e.g. 1072×1908 vs. a 1080×1920 rotated target is a
1.0075x scale) without masking any genuine upscale need (a real "needs upscaling" input is almost
always well above 1.05, e.g. 540×960 → 1080×1920 is 2.0x) — chosen to skip/resize rather than
still running a (much smaller, non-power-of-2) AI pass, since Real-ESRGAN's minimum discrete scale
is already 2x and a genuine sub-2x need this close to target isn't worth a full AI forward pass.

## Configuration

- Config file: `~/.config/auto-video-fixer/config.yaml` (Linux, `$XDG_CONFIG_HOME` if set),
  macOS/Windows paths in `config.py:get_config_dir()`.
- Data dir (models, cache): `get_data_dir()` — `$XDG_DATA_HOME` or `~/.local/share` on Linux.
- **State/log dir**: `get_state_dir()` — `$XDG_STATE_HOME` or `~/.local/state` on Linux
  (macOS/Windows fall back to the same base as the data dir, since neither platform has a
  distinct "state" convention). `get_log_dir()` = `<state_dir>/logs`.
- **Automatic per-run DEBUG log file**: every CLI invocation writes a timestamped log file
  (`avf-YYYYMMDD-HHMMSS.log`) to `get_log_dir()` at DEBUG level, independent of console
  verbosity — always on, no flag needed. The exact path is printed/logged at startup. Retention:
  newest 50 kept (`prune_old_logs()`, pruned at startup before the current run's file is
  created); `--log-file PATH` *replaces* the automatic location for that run (same always-DEBUG
  file logging, user-chosen path; no state-dir file is created).
- **Startup settings banner**: on every invocation, the CLI logs (`_log_effective_settings()` in
  `cli.py`) the version, full `sys.argv`, which config file is in use (default vs. explicit), and
  a redacted diff of effective config vs. `Config.DEFAULTS` at INFO; the full effective config
  dump (also redacted) at DEBUG. Secret-looking keys (containing `api_key`/`apikey`/`token`/
  `password`/`secret`, case-insensitive) are masked as `"***"` via `config.redact_secrets()`.
  Re-logged after `process`'s own preset/flag merging, since the group-level log only reflects
  what was loaded from disk.
- **`--config PATH`** (global flag, before the subcommand) / **`AVF_CONFIG`** env var select an
  alternate config file. Precedence: `--config` flag > `AVF_CONFIG` env var > default platform
  path. Unlike the default path (silently falls back to `Config.DEFAULTS` if missing), an
  explicitly-given path (flag or env var) is a **hard error** if it doesn't exist — no silent
  no-op.
- Config is read-once at `Config()` construction; `config.set()` marks dirty and `config.save()` writes YAML.
- Preset merging is recursive — preset values override config, but config values not in preset are preserved.
- A full, commented example covering every `Config.DEFAULTS` key lives at
  `docs/config.example.yaml`.

## Gotchas

- **stabilize.py `_get_video_dimensions` / `_get_video_framerate`**: Must use `stdout=subprocess.PIPE` (not `subprocess.DEVNULL`). Using DEVNULL discards output and forces fallback to 1920×1080 / 30fps, stretching portrait video to landscape.
- **stabilize pipe sizing**: The decode subprocess **must** include `-s {width}x{height}` to match the transform's `-s` input. Without it, anamorphic or non-standard resolution videos produce corrupted output due to stride mismatch.
- **vid.stab B-frame corruption**: Known bug (github.com/georgmartius/vid.stab#144). The pipeline pipes raw YUV420P between decode and transform to avoid it. Use accuracy=15 (higher causes FFmpeg return code 222).
- **FFmpeg output paths**: Stages must use an explicit output path. Using `output_path or input_path` on failure produces an empty file.
- **Stage disabled check**: `should_run()` must check `self.is_enabled()` which reads from `DEFAULTS["stages"][name]["enabled"]`.
- **`NormalizeVolumeStage`** (name `"normalize_volume"`) and **`NormalizeAudioStage`** (name `"normalize_audio"`) are two separate classes in `normalize_audio.py`.
- **Preset stage enable**: Presets define `enable_stages` which controls which stages run. A stage not listed in a preset's `enable_stages` will not execute, even if it's in the default order.

## Known bugs / pitfalls to avoid

The bugs formerly listed here (`cb` scoping crash in chunked AI paths, `upscaled`/`all_upscaled`
typo, missing `os`/`probe` imports in `deblock.py`/`denoise_video.py`, unused local `run_ffmpeg`
shadow import, F541 f-string lint errors, unused `fps_val`) are **fixed** — verified directly
against current `deblock.py`, `denoise_video.py`, `upscale.py`, `interpolate.py`: `cb` callbacks
are defined before their chunked loops in all four stages, `os` is imported at the top of
`deblock.py`, `denoise_video.py` imports `probe` at module level, and the AI upscale/deblock/
denoise chunked paths all correctly reference the frame lists they build. Kept here only as a
historical note in case of regression — do not assume these still need fixing.

Remaining lint/style notes (still current):
- **ruff I001** on `deblock.py` imports: blank line between `from __future__` and the stdlib imports is expected by the sort rule.
- **ruff E501** line length is 100. Several stage files (stabilize, interpolate, upscale, cli) exceed it with long f-strings; plan around this.
- **mypy `--ignore-missing-imports`** is required; `torch`/`cv2`/`rife` are optional and mypy will flag them without that flag.

## Scene mode

Opt-in, off by default (`scenes.enabled: false`). See `core/scenes.py`, `Config.DEFAULTS["scenes"]`
and `Config.DEFAULTS["analysis"]["llm"]`, and `Pipeline.execute_job()`'s scene-mode block (right
after `current_path = job.input_path`, before the main stage loop).

- **Trigger**: scene mode only engages when `scenes.enabled` is true AND the job's stage list
  includes `stabilize` and/or `interpolate`. Otherwise it's a strict no-op -- the whole-video path
  is completely unaffected.
- **Flow**: `run_scene_pipeline()` (1) runs the existing scene detector
  (`VideoAnalyzer.detect_events`, same `analysis.event_detection.*` config -- no separate
  scene-mode threshold), (2) optionally drops non-content scenes (see below), (3) re-encodes each
  kept scene as its own clip (`split_scene_video`/`split_scene_audio` -- re-encoded, not
  stream-copied, so cuts land exactly on the detected boundary instead of snapping to the nearest
  keyframe), (4) runs `stabilize` (tiered strength) and/or `interpolate` (never across a cut) on
  each scene clip independently, in parallel across scenes when there's more than one (bounded
  worker pool shared with per-scene interpolation chunking, see `_scene_worker_budget`), (5)
  concatenates the processed video clips and the original audio (cut at the same boundaries) back
  into one file. `Pipeline.execute_job()` then continues the normal per-file pipeline from that
  reassembled file, with `stabilize`/`interpolate` removed from the remaining stage list (they
  already ran) -- `upscale`/`denoise_video`/`deblock`/`normalize_*`/`encode` all run once on the
  whole reassembled video, not per-scene.
- **Fewer than 2 scenes detected, or any internal failure** (split/stabilize/interpolate/concat
  error, unexpected exception): `run_scene_pipeline()` returns `None` and the pipeline silently
  falls back to normal whole-video processing for that job -- scene mode can never turn a job that
  would have succeeded without it into a failure.
- **Per-scene stabilize strength tiering** (`scenes.stabilize.*`): each scene clip gets its own
  `StabilizeStage.execute()` call, which does its own `vidstabdetect` pass and reports
  `avg_shake` in its result metadata. "skip" tier = `StabilizeStage`'s own `needs_stab` gate
  (`avg_shake < stages.stabilize.threshold`) says no stabilization needed. "normal" tier =
  stabilized once with the stage's default config. "aggressive" tier = `avg_shake >=
  scenes.stabilize.aggressive_shake_threshold` (default 8.0px) -- re-run with `smoothness`
  multiplied by `scenes.stabilize.aggressive_smoothness_multiplier` (default 2.0). This means a
  shaky scene can get run through `vidstabdetect` twice (once for the normal-tier attempt, once
  more for the aggressive re-run) -- accepted cost, bounded to shaky scenes only.
- **Never-interpolate-across-a-cut** (`InterpolateStage` invoked per scene clip, never on the
  whole file when scene mode is on): each scene's own framerate is checked against the target via
  `should_run()` before interpolating, so a scene already at/above target framerate passes through
  unchanged instead of being redundantly processed.
- **Drop non-content scenes** (`scenes.drop_non_content: false` default, `--drop-non-content`
  CLI flag): per-scene VLM sampling (`VideoAnalyzer.run_vlm_analysis_for_scene` -- samples
  `min(max_sample_frames, ceil(scene_duration / sample_interval_sec))` frames from within just
  that scene's time range, via `_extract_sample_frames_range`) feeds a coordinating text-LLM pass
  (`run_scene_coordinator`, config `analysis.llm.*`: `provider`/`model`/`api_key`/`api_url`/
  `allow_http`, same HTTPS-required-for-non-loopback gate as `analysis.vlm.allow_http`). The
  coordinator returns which scene indices to drop and why; dropped scenes are logged at WARNING
  (index, time range, reason) and excluded from the reassembled output. **Fails open on ANY
  failure** -- unreachable endpoint, exception, unparseable JSON response, or a `drop` array
  containing a non-integer/out-of-range/negative index -- logging a WARNING and keeping every
  scene (`run_scene_coordinator` returns `{"drop": [], "failed": True, ...}`; `run_scene_pipeline`
  additionally catches any exception from the whole VLM+coordinator block and fails open the same
  way). If the coordinator response would drop literally every scene, that's also refused (treated
  as a bad response) and every scene is kept -- scene mode will never produce an empty output.
- **A/V sync**: audio is re-cut at the exact same scene boundaries as the video (both re-encoded,
  not stream-copied, from the original source) and concatenated separately, then muxed against
  the concatenated (processed) video track. Since neither stabilization nor interpolation changes
  a scene's time span (interpolation adds frames covering the *same* duration, it doesn't stretch
  it), this keeps sync without needing to re-derive timing from the processed video.
- **Known limitation**: concatenated output duration/frame-count is close to but not always
  bit-exact vs. the sum of scene durations -- `minterpolate`'s per-chunk (and per-scene) frame
  count is duration-based, not a fixed multiply, so independent per-scene/per-chunk retiming
  passes don't always sum to exactly what one continuous whole-file pass would have produced.
  Verified within ~1-2% on a synthetic multi-scene clip, with zero blend/ghosting artifacts at any
  scene boundary (the actual correctness-critical property) -- see CHANGELOG.

## Stabilization zoom coverage

`stages.stabilize.zoom_coverage` (float, default `1.0`) tunes how aggressively the stabilize
stage's zoom compensates for borders introduced by stabilization, once `zoom_enabled` /
`zoom_threshold`'s movement-extent gate (`apply_zoom` in `StabilizeStage.execute()`) has already
decided zoom applies at all -- `zoom_coverage` does NOT change that gate, only what zoom is used
once it fires:

- `1.0` (default, unchanged behavior): `vidstabtransform`'s own `optzoom=1` ("optimal static
  zoom"), sized to the single worst frame in the clip -- guaranteed no visible border on any
  frame, at the cost of being the least tight/most-cropped option.
- `0.0`: no zoom at all (`optzoom=0`, `zoom=0`) -- every border from camera motion stays visible,
  equivalent to `zoom_enabled=False`'s zoom behavior but without disabling the rest of the gate
  logic (movement is still analyzed, `apply_zoom` is still computed and reported in metadata).
- In between: `optzoom=0` plus a static `zoom=<pct>` computed by
  `StabilizeStage._compute_static_zoom_pct()` from the TRF vidstabdetect already parses
  (`_analyze_trf_file`/`_movement_extent`'s TRF-reading lineage) -- the `zoom_coverage`-th
  quantile of PER-FRAME required-zoom estimates, not the max. This lets a user trade "guaranteed
  no border, ever" for "less aggressive crop, with occasional brief borders on the most extreme
  motion" -- e.g. a violent-motion ending gets a border for a second or two while the rest of the
  clip stays zoomed less than the `1.0` case would force.

**Accuracy limitation (state honestly, this is not exact)**: what actually determines a frame's
visible border is `vidstabtransform`'s own internally-computed SMOOTHED camera path (a function of
`smoothing`/`maxshift`/`optalgo`/`interpol`), which this code has no access to -- it only sees the
raw per-block local-motion (LM) values in the TRF. `_compute_static_zoom_pct()` integrates those
into an approximate raw cumulative camera-path position, then applies a local moving-average
smoothing window (matching `stages.stabilize.smoothness`, the same window vidstabtransform itself
uses) to approximate the smoothed path, and measures each frame's deviation from that local
average as its "required zoom" (`zoom_pct = 200 * shift_px / dimension_px`, derived from
`vidstabtransform`'s `zoom=Z%` scaling the frame by `1+Z/100` around center). This is
**directionally correct and tunable** (verified: monotonically increasing with `coverage`,
verified via a synthetic shaky clip) but not an exact match to vidstabtransform's internal
computation -- empirically, on one synthetic test clip, this method's own `coverage=1.0` quantile
(the theoretical max) came out ~3x higher than vidstabtransform's own logged `optzoom=1` "Final
zoom" value, i.e. the approximation is conservative/over-corrects rather than under-corrects for
that clip. Individual frames' actual post-smoothing border requirements can come out higher or
lower than this estimate. This also does NOT account for rotation's contribution to border size
(same limitation as `_movement_extent()` -- no new rotation math was added).

**Verification methodology** (ffmpeg/CPU only, no GPU needed): a synthetic clip was generated with
`ffmpeg -f lavfi -i testsrc2=...` piped through a time-varying `crop=` filter simulating handheld
shake (sinusoidal jitter) plus one abrupt high-velocity displacement late in the clip (the
"violent-motion ending" scenario). Verified via whole-video-union `cropdetect=limit:round:reset=0`
(same methodology as the auto-crop stage) and per-frame `cropdetect=limit:round:reset=1` box
sampling: `zoom_coverage=1.0` produced zero border on any of 178 sampled frames (matches
`optzoom=1`'s guarantee); `zoom_coverage=0.0` reproduced the same borders as the fully-unzoomed
baseline (~43% of frames showing a small border, matching `crop_mode: black`'s border-fill
behavior); `zoom_coverage=0.6`'s computed static zoom quantile was measurably smaller than the
`coverage=1.0` quantile computed by the same function (10.4% vs. 25.5% on the test clip) while
still fully covering that particular clip's (small, ~2-4px) actual borders -- demonstrating the
dial responds correctly to `coverage` even though this specific synthetic clip's real border
requirement was too small to show a visible difference in cropdetect output at 0.6 vs. 1.0.

CLI: `--zoom-coverage FLOAT` (`stages.stabilize.zoom_coverage`), following the `--crop-limit`
precedent of a single per-stage override flag.

## Auto-crop

Opt-in, off by default (`stages.crop.enabled: false`). See `core/stages/crop.py`,
`Config.DEFAULTS["stages"]["crop"]`, `core/analysis.py::run_crop_vlm_check`/
`_parse_crop_vlm_response`, and `docs/REQUIREMENTS.md` feature 3.

- **What it does**: detects a video's true content bounds -- the union over the whole video (the
  furthest real content ever reaches toward each edge), NOT a per-frame crop -- via FFmpeg
  `cropdetect=limit=<L>:round=<R>:reset=0 -f null -`, and crops to that single window with a
  `crop=w:h:x:y` re-encode (`libx264 -crf 18`, `-c:a copy`). `reset=0` never resets cropdetect's
  accumulated box between frames, so the LAST `crop=` line in ffmpeg's stderr is the union of the
  whole scanned range -- exactly the whole-video bound this stage wants, not a window that
  flickers scene-to-scene.
- **Stage order**: right after `stabilize`, before every other enhancement/AI stage -- see the
  "Pipeline Behavior" note above for why.
- **Config** (`stages.crop`): `limit` (cropdetect luma threshold, default 24), `round` (even-
  dimension rounding, default 2), `min_crop_px` (skip entirely if the detected crop would save
  fewer than this many pixels in BOTH width and height, default 8 -- avoids a pointless 2px crop
  from encoder rounding noise), `analyze_duration_sec` (0 = full-video scan (default); >0 = only
  scan the first N seconds, for very long inputs where a full scan is too slow), `vlm_check`
  (default false), `vlm_policy` (`"warn"` default | `"skip"`).
- **`should_run()` vs. `execute()`**: `should_run()` does a cheap ~10s cropdetect pre-filter
  against `input_info`'s (possibly stale -- see "Pipeline Behavior") filepath purely to skip
  obviously-nothing-to-do cases early; it is NOT authoritative. `execute()` always re-runs a full
  cropdetect pass (respecting `analyze_duration_sec`) against the actual file it's handed and can
  independently return `SKIPPED` (e.g. "cropdetect produced no result", "saves only Nx Mpx, below
  min_crop_px", or an invalid/larger-than-input crop window) -- a false "proceed" from
  `should_run()` is harmless (`execute()` still catches it), a false "skip" (missing a border that
  only appears after an earlier stage, e.g. stabilize) is the accepted risk of keeping the
  pre-filter cheap.
- **VLM assist** (`stages.crop.vlm_check`, requires `analysis.vlm.enabled: true`): a watermark/
  logo positioned relative to a letterboxed frame (i.e. sitting in the border area cropdetect
  would otherwise remove) can fool naive cropdetect either way -- brightness above `limit` widens
  the kept region to "protect" it, or a dim watermark below `limit` gets cropped away along with
  the border with no way for cropdetect alone to know that was meaningful content. When enabled,
  one frame is rendered twice (plain, and the same frame with the proposed crop box drawn via
  `drawbox`) and sent to the VLM with a fixed internal prompt asking whether the region OUTSIDE
  the box contains meaningful content (`core.analysis.run_crop_vlm_check`, its own system/user
  prompt pair -- `analysis.vlm.prompt_override`/`prompt_append` do NOT apply here). Parsed with a
  tolerant JSON-then-yes/no-text fallback (`_parse_crop_vlm_response`). If the VLM says yes:
  `vlm_policy: "warn"` (default) logs a WARNING and crops anyway; `"skip"` skips cropping this
  video entirely. **Fails open** on any VLM error (unreachable endpoint, exception, unparseable
  response) -- logs a WARNING and proceeds with the plain cropdetect result, never blocks on VLM
  availability.
- **`"expand"` policy was considered and rejected**: growing the crop box to include only the
  flagged region isn't feasible because VLMs don't reliably return pixel coordinates for what they
  flag -- there's nothing to expand *to*. Only `"warn"`/`"skip"` are implemented.
- **CLI**: `--enable-stage crop` (or `stages.crop.enabled: true` in config/preset) opts in;
  `--crop-limit INT` overrides `stages.crop.limit` for the run (still requires the stage to be
  enabled some other way -- it's a threshold override, not an enable flag).
- **Aspect-ratio interaction**: crop always runs before `upscale`, so `quality.quality_target`
  dimensions apply to the already-cropped frame -- the target resolution is a post-crop target,
  which is almost always what's wanted (the user wants the *content* at the target resolution, not
  the pre-crop letterboxed frame).

## Rust rewrite candidates

Three CPU-hot-path targets are scoped for PyO3/`maturin`-based Rust rewrites, prioritized soon on
the roadmap. **Scene-detection frame differencing** (`_detect_scene_changes()`, `core/
analysis.py`) is **implemented** — see "Mixed Python/Rust" above and `rust/avf_scenes/`. Still
planned: perceptual-hash duplicate detection (`compute_video_hash`/`compute_video_dhash`, `core/
analysis.py`) and the chunked AI frame prefetch/writer threading (`ai/frame_processor.py`). See
`docs/REQUIREMENTS.md`'s "5. Rust rewrite candidates" section for full scope, interface
signatures, and verification bars per target, and `docs/ROADMAP.md`'s "Planned" section for
priority context. Two other pieces (FFmpeg subprocess orchestration, AI inference calls
themselves) were explicitly considered and rejected as Rust targets — see REQUIREMENTS.md for why
before proposing them again absent a new non-performance justification.

## CLI Flags

Global (before the subcommand):
- `--verbose, -v`: Enable DEBUG level logging (console and the always-on log file)
- `--log-level LEVEL`: Set console logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file PATH`: Write the run's log file here instead of the automatic timestamped file in
  the platform log dir (see "Configuration" above)
- `--file-log-level LEVEL`: Log level for the file log, if different from the console level
- `--config PATH` (env: `AVF_CONFIG`): Use this config file instead of the default platform path;
  must already exist (hard error if not)

`avf process`:
- `--preset, -p NAME`: Apply preset (e.g., `1080p60`, `4k60`, `size_reduction`)
- `--output, -o DIR`: Output directory
- `--output-name NAME`: Explicit output filename (only valid with exactly one input file)
- `--recursive, -r`: Scan directories recursively
- `--stage NAME`: Run only these stage(s), replacing the preset/auto-determined list (can
  repeat); also forces `stages.<name>.enabled=false` to be bypassed for explicitly-requested
  stages (see `BaseStage._force_enabled` in `core/stages/base.py`)
- `--enable-stage NAME` / `--disable-stage NAME`: force-enable/disable a stage on top of the
  preset/default set (can repeat) — unlike `--stage`, this doesn't replace the whole stage list
- `--dry-run`: Show what would be done without processing
- `--list-presets`: List available presets
- `--ai` / `--no-ai`: Force AI or traditional methods (`general.use_ai`)
- `--ai-fallback` / `--no-ai-fallback`: Allow/forbid AI stages from silently falling back to
  traditional methods when the AI path can't run (`general.ai_fallback`; see "AI-fallback
  policy" above) — does not override a per-stage `stages.<name>.ai_fallback` in config
- `--overwrite` / `--no-overwrite`: Allow overwriting an existing output file
- `--fps FLOAT`: Target output framerate (`quality.quality_target.target_framerate`)
- `--resolution WIDTHxHEIGHT`: Target output resolution, e.g. `3840x2160`
  (`quality.quality_target.target_resolution`)
- `--codec` / `--audio-codec` / `--crf` / `--encoder-preset`: encode-stage overrides
  (`encoding.video_codec`/`audio_codec`/`crf`/`preset`) — `--encoder-preset` is NOT the same as
  `--preset` (which selects a named avf processing bundle)
- `--hwaccel {auto,none,cuda,vaapi,qsv,vulkan,videotoolbox}`: FFmpeg hwaccel for encode/decode
  (`ffmpeg.hwaccel`)
- `--gpu-device {auto,cpu,cuda,mps}`: PyTorch device for AI stages (`gpu.preferred_device`) —
  separate from `--hwaccel`
- `--threads N`: sets `general.max_concurrent_jobs`
- `--scene-mode` / `--no-scene-mode`: run `stabilize`/`interpolate` per-scene instead of
  whole-video (`scenes.enabled`) — see "Scene mode" above
- `--drop-non-content` / `--no-drop-non-content`: with `--scene-mode`, drop scenes flagged as
  non-content by the VLM+coordinator pass (`scenes.drop_non_content`) — fails open on any
  failure, see "Scene mode" above
- `--crop-limit INT`: cropdetect luma threshold override for the auto-crop stage
  (`stages.crop.limit`) — auto-crop itself is still opt-in, enable it via `--enable-stage crop`
  or `stages.crop.enabled: true` in config; see "Auto-crop" above
- `--zoom-coverage FLOAT`: fraction of frames (0.0-1.0) that should end up border-free once the
  stabilize stage's zoom gate decides zoom applies at all (`stages.stabilize.zoom_coverage`) —
  see "Stabilization zoom coverage" above
- `--batch-size INT` / `--tile-batch-size INT`: set `stages.{upscale,deblock,denoise_video}.
  batch_size`/`tile_batch_size` (all three at once) — see "Batched inference" above

`avf analyze PATHS...`: like `process`, accepts multiple files and/or directories (directories
scanned via `scan_directory`, hidden files skipped) and analyzes each in sequence; a per-file
failure is logged and skipped, remaining files still run, and the command exits non-zero if any
failed.
- `--vlm` / `--no-vlm`: Enable/disable VLM analysis
- `--events` / `--no-events`: Enable/disable event/scene detection
- `--classify`: Classify detected events with VLM
- `--clip DIR` / `--no-clip`: Extract detected scenes as clips to a directory
- `--recursive, -r`: Scan directories recursively (same as `process`)
- `--scene-threshold FLOAT`: Override `analysis.event_detection.scene_change_threshold` (default
  0.15) for this run -- mean fractional per-pixel luma change between consecutive downscaled
  frames, 0-1; lower is more sensitive (risks false positives from motion/noise), higher only
  catches hard cuts. See `_detect_scene_changes()` in `core/analysis.py` for the full metric
  writeup and calibration data (0.3, the previous default, under-detected real footage).
- `--min-scene-duration FLOAT`: Override `analysis.event_detection.min_scene_duration_sec`
  (default 2.0) for this run -- doesn't affect cut *detection*, only whether a cut close to the
  previous one gets its own scene vs. being merged into the next.
- `--full`: Print the complete, untruncated VLM summary per file in a Rich panel below the table
  (the table's own "VLM Summary" row is always a truncated preview with a "(use --full for full
  text)" hint). The full summary is also always written at INFO to the log (auto per-run DEBUG
  log file always has it; console has it at INFO too unless `--log-level` raises the threshold).
- `--csv PATH`: Write one row per analyzed video (filepath, filename, duration, resolution,
  framerate, codec, has_video/has_audio/hdr, scenes_detected, `scene_boundaries` (see below), and
  — when VLM ran — the FULL untruncated summary, `;`-joined tags/objects, and content_rating) to
  a UTF-8 CSV. Overwrites `PATH` if it exists; does not append across runs.
- `--prompt-append TEXT`: Extra text appended to the VLM user prompt for this run (overrides
  `analysis.vlm.prompt_append`) -- e.g. job-specific context like "these are trail-camera clips".
- `--prompt-override TEXT`: Replace the VLM user prompt entirely for this run (overrides
  `analysis.vlm.prompt_override`; `--prompt-append`/`prompt_append` still appends after an
  override). Changing the requested output format away from JSON degrades gracefully --
  `_parse_vlm_response()`'s fallback treats non-JSON text as the summary (empty tags/objects/
  rating) instead of erroring. There's no CLI flag for the system prompt -- use
  `analysis.vlm.system_prompt_override` in config.
- `--max-sample-frames INT`: Override `analysis.vlm.max_sample_frames` (default 8) for this run.
- `--sample-interval FLOAT`: Override `analysis.vlm.sample_interval_sec` (default 10.0) for this
  run. (Prior to this flag being added, `sample_interval_sec` in config was never actually read
  -- `run_vlm_analysis()` hardcoded 10.0 -- so this also fixes a dead config key.)
- `--vlm-model TEXT`: Override `analysis.vlm.model` (e.g. `llava`, `gpt-4o`) for this run.
- `--vlm-url TEXT`: Override `analysis.vlm.api_url` for this run. Still goes through the same
  HTTPS-required-for-non-loopback-hosts gate as the config value (`_call_custom_api`'s `urlparse`
  check) -- `analysis.vlm.allow_http` is what permits plain HTTP, and deliberately has no CLI
  flag, same as `analysis.vlm.api_key`; both stay config-file-only (secrets/security posture
  shouldn't be one flag away from a shell history entry).

**Scene-score visibility** (added after a real-world report of a flat scene count across
`--scene-threshold` 0.01-0.30 that turned out to need the raw scores to diagnose): every
`detect_events()`/`_detect_scene_changes()` call now logs at INFO, per file:
  - The effective `threshold`/`min_duration` actually used (config value or CLI override,
    whichever applied) -- confirms an override took effect.
  - A one-line cut-score summary: number of cuts found, min/median/max `diff_score` among them.
  - A "near-miss" line (only if any exist): up to the 10 highest diff_scores that fell *below*
    threshold but *above* threshold/4 -- i.e. plausible cuts a lower threshold would catch, with
    their timestamps, so "would lowering the threshold help, and where?" doesn't require a rerun.
`avf analyze --full` also prints each scene's boundary `diff_score` in the console scene listing
(`(score=0.XXX)`; the default view omits it to stay uncluttered), and `--csv`'s
`scene_boundaries` column has one `t=<seconds>s@<confidence>` entry per scene, `;`-joined.
`SceneEvent.confidence` is the `diff_score` of the cut that *ended* that scene (not started it);
the final scene per video has a fixed placeholder confidence of 0.5 (nothing ends it) -- see
`_scene_boundaries_str()`'s docstring in `cli/cli.py`.

Other subcommands: `avf find-duplicates REFERENCE DIRECTORY [--threshold FLOAT]`,
`avf presets-cmd` (lists presets), `avf gpu-info`, `avf model-info [--model NAME]`,
`avf model-download --model NAME [--url URL] [--force]`.
