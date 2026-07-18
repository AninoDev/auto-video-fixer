# Auto Video Fixer - Project Roadmap

## Vision

Auto Video Fixer aims to be the most intelligent, automated video enhancement tool available. It combines AI-powered processing with traditional techniques to deliver professional-quality results with minimal user effort.

## Current Status: v0.3.0 (Alpha)

This section is written to be honest about what is *implemented and verified working* versus
*implemented but not independently verified* versus *not built yet*. Earlier versions of this
document marked AI upscaling as "completed" while it was, in fact, producing solid-black output
on GPU due to a missing residual-scaling factor in the RRDB block — that bug (and several
related ones) are now fixed and spot-checked on GPU, but the incident is why this document now
draws an explicit VERIFIED / UNVERIFIED line instead of a flat done/not-done checklist.

### Implemented and verified working

- **Core pipeline architecture** — stage registry, `Pipeline.execute_job()`/`execute_all()`,
  config-driven stage ordering/omission/repetition (`pipeline.default_order`, resolved by
  `Pipeline.resolve_stage_order()`), per-job temp file lifecycle (`.mkv`
  intermediates, unconditional cleanup, orphan-temp promotion to output), quality gate
  (SSIM/PSNR via FFmpeg with `measurement_failed` reported instead of a fake 0.0 score).
- **Traditional (FFmpeg-only) processing end-to-end**: stabilize (vidstab, raw-pipe
  decode→transform to avoid the vid.stab B-frame corruption bug, `optzoom=1` auto zoom-out for
  shaky footage), deblock (`unsharp`), denoise (`hqdn3d`), upscale (`scale=...:flags=lanczos`),
  interpolate (`minterpolate`), audio normalize (EBU R128), encode, remux, speed, HDR-to-SDR.
- **AI upscaling — Real-ESRGAN (RRDBNet)**: x4plus/x2plus/anime_6B model support, x2plus
  auto-selected for scale≤2 passes (upscale) and scale=1 passes (deblock/denoise) for a
  measured ~4-5x speedup over running x4plus and discarding the extra resolution, tiled
  inference with reactive CUDA-OOM retry and a `stages.<name>.tile_size` override,
  `channels_last` memory format (measured ~1.6→11 fps for a 720p→1080p pass on the project's
  target GPU), TTA, FP16 on CUDA. The previous black-output bug (missing `0.2` residual scaling
  in the RRDB block → NaN collapse) is fixed and confirmed producing real output on GPU.
- **AI upscaling — compact SRVGG models**: `realesr-general-x4v3`/`realesr-general-wdn-x4v3`
  (`num_conv=32`)/`realesr-animevideov3` (`num_conv=16`), the official xinntao/Real-ESRGAN
  v0.2.5.0 compact video checkpoints, dispatched alongside RRDBNet off a registry `arch` field
  (`resolve_arch()` in `ai/wrappers/upscale.py`). ~1.2M/~0.6M params vs RRDBNet's ~16.7M — an
  order-of-magnitude-plus less GPU compute per frame at some quality cost; RRDB remains the
  default for `upscale`/`deblock`/`denoise_video`. Verified on GPU: strict state-dict load of
  all three checkpoints, correct x4 output shapes, non-black fp16 output, tiled inference.
- **AI frame interpolation — RIFE (IFNet + EMD)**: the previous fps bug (writing interpolated
  output at the *input* fps instead of `input_fps * factor`, which stretched clip duration
  instead of increasing framerate) is fixed.
- **AI/traditional selection & fallback policy**: `--ai`/`--no-ai` override, plus a separate
  `general.ai_fallback` / `stages.<name>.ai_fallback` policy and `--ai-fallback`/
  `--no-ai-fallback` CLI flags — when fallback is disabled, an AI stage that can't run FAILS
  with a named cause instead of silently downgrading to traditional output.
- **Quality gate**: reference is scaled to the output's resolution before SSIM/PSNR (previously
  a straight compare against differing resolutions would either error or silently no-op);
  measurement failures are reported as "not checked" rather than a fake 0.0.
- **CLI**: `process`, `analyze`, `find-duplicates`, `presets-cmd`, `gpu-info`, `model-info`,
  `model-download`; global `--config`/`AVF_CONFIG`; automatic per-run DEBUG log file with
  retention (newest 50) under the platform state dir; startup settings banner (redacted
  effective-config diff at INFO, full dump at DEBUG).
- **Preset system**: 7 built-in presets (`max_quality`, `4k60`, `4k30`, `1080p60`,
  `size_reduction`, `remux_only`, `hdr_enhance`), recursive config merging (a preset no longer
  clobbers unrelated user config under the same top-level key).
- **VFR/YouTube-origin input handling**: `_parse_fps`/stabilize's framerate probing now prefer
  `avg_frame_rate` over `r_frame_rate` ("tbr"), fixing a 2x-speed-then-freeze artifact on VFR
  sources where the two rates diverge; hardened against ffprobe's `"0/0"` undefined-rate value.
- **VLM (Vision Language Model) classification** — the generic OpenAI-compatible `api` provider
  (`analysis.vlm.provider: api`) has been run end-to-end against a live llama.cpp server on the
  LAN (via `analysis.vlm.allow_http`), producing real summary/tags/objects/rating output. Ollama
  and OpenAI Vision providers share the same request/response code path
  (`_call_ollama`/`_call_openai` vs. `_call_custom_api`) and are unit-tested
  (`tests/unit/test_vlm.py`) but have not themselves been run against a live instance/key.

### Implemented and verified working (recently added)

- **Scene-based processing pipeline (opt-in, `scenes.enabled`)** — `core/scenes.py` splits a
  video at existing scene-detection boundaries, runs `stabilize`/`interpolate` per-scene, and
  concatenates the result (audio cut at the same boundaries) before the remaining whole-video
  stages run. Verified end-to-end on a synthetic multi-scene clip: zero blend/ghosting frames at
  any scene boundary (frame interpolation never crosses a cut), output duration/fps within
  tolerance of the original, and via `Pipeline.execute_job()`'s full stage loop (not just the
  standalone module). See AGENTS.md's "Scene mode" section.
- **Per-scene stabilization strength tiering** (`scenes.stabilize.*`) — verified on the same
  synthetic clip: a scene with synthetic camera-shake (crop-jitter) was correctly tiered
  "aggressive" (re-stabilized with higher smoothness) while static/low-motion scenes were tiered
  "skip" or "normal" as expected, confirmed via stage logs.
- **Drop non-content scenes** (`scenes.drop_non_content`, requires `analysis.vlm` +
  `analysis.llm`) — verified live against a real LAN VLM endpoint: per-scene VLM sampling
  correctly identified a synthetic "LIKE AND SUBSCRIBE" interstitial scene by content. The
  coordinating text-LLM pass was also exercised live against the same endpoint; in that specific
  run the coordinator model's response was truncated (its reasoning tokens exhausted the
  response token budget before it emitted the JSON answer) — this is exactly the kind of
  real-world failure the fail-open design targets, and it correctly triggered fail-open (kept
  every scene, logged a WARNING) rather than dropping anything. The coordinator's parsing/
  drop-index validation is additionally unit-tested with mocked responses (correct-drop, and
  every failure mode: unreachable, malformed JSON, out-of-range index). **Not yet verified**: a
  coordinator response that actually returns a non-empty, valid drop list end-to-end against a
  live endpoint (blocked on the above token-budget issue in this environment, not a code defect
  — a shorter prompt against the same endpoint did return valid JSON, so the parsing/prompting
  approach itself works; a production run should raise the endpoint's `max_tokens` for
  reasoning-style coordinator models).
- **Parallel-chunked traditional frame interpolation** (`stages.interpolate.parallel_chunks`) —
  benchmarked (~4.2-4.7x speedup on an 8-core machine, 60s 720p30→60fps) and boundary-verified
  (near-identical pixel content across chunk seams on a synthetic clip; see CHANGELOG). Frame
  counts between serial and parallel runs are close but not bit-exact (~1-2%, from
  `minterpolate`'s own per-chunk duration-based frame-count rounding) — see `AGENTS.md`'s Scene
  mode section and `_execute_traditional_parallel`'s docstring for detail.
- **Auto-crop stage (opt-in, `stages.crop.enabled`)** — `core/stages/crop.py` detects a video's
  true content bounds via FFmpeg `cropdetect` with `reset=0` (whole-video union, not a per-frame
  crop) and crops to that window, right after `stabilize` in `Pipeline.optimize_stage_order()`.
  Verified against real FFmpeg-generated fixtures: a 640x360-in-640x480 letterboxed video crops
  back to ~640x360 with cropdetect finding no remaining border on a post-crop re-scan; a
  border-free video is correctly left uncropped; a letterboxed video with a dim in-border overlay
  is cropped away in plain mode (documented limitation, not a bug — see AGENTS.md's "Auto-crop"
  section). **VLM-assisted watermark/content disambiguation** (`stages.crop.vlm_check`) is
  implemented and unit-tested with mocked VLM responses (both providers dispatch, JSON + tolerant
  text-fallback parsing, fail-open on error) and integration-tested with a mocked
  `run_crop_vlm_check` (`vlm_policy: "skip"` correctly leaves the video uncropped) — like the
  rest of the VLM foundation (see "0. Foundation" in REQUIREMENTS.md), it has **not** been
  exercised against a live VLM endpoint as part of this pass. **Arbitrary-color border
  detection** (`stages.crop.detector: "rust"`, default `"auto"`) is now also implemented — a new
  Rust extension (`rust/avf_borders/`) detects per-edge borders of ANY color (not just black/dark,
  which is `cropdetect`'s limit), reporting both a dominant color and a "solidity" percentage per
  edge, logged at INFO on every run. See REQUIREMENTS.md's R3.5 and AGENTS.md's "Mixed
  Python/Rust"/"Auto-crop" sections.

### Implemented but not independently verified

These have working code paths and existing unit tests, but have not been checked end-to-end
against real-world inputs the way the traditional pipeline and the AI upscale/interpolate paths
above have been. Treat their behavior as "probably correct, unconfirmed" rather than "done":

- **Scene detection quality** — frame-differencing scene-change detection
  (`_detect_scene_changes`) and heuristic event classification exist and are unit-tested
  (`tests/unit/test_scene_detection.py`); detection accuracy/threshold tuning on real footage is
  unverified. The frame-differencing hot loop itself is now a Rust extension (see "Rust rewrites"
  below) with a verified-identical pure-Python fallback — that part is no longer unverified
  plumbing, just the threshold-tuning-on-real-footage question above.
- **Duplicate detection accuracy** — the ahash/dhash implementation was replaced by a Rust-backed
  pHash (see "Rust rewrites" below); accuracy verified against a real ffmpeg-generated near-
  duplicate/non-duplicate test set (`tests/unit/test_hashing.py::TestHashSeparation`), not just
  real-world usage. Genuinely-unverified real-world false-positive/negative rates on arbitrary
  user footage remain an open question, same caveat as scene detection above.
- **The Qt (PySide6) GUI** (`gui/main_window.py`, `avf-gui` entry point) — has a test file
  (`tests/unit/test_gui.py`) and basic job-queue/preset/progress wiring, but has not been
  manually driven end-to-end as part of this pass.

### Planned (not implemented)

See `docs/REQUIREMENTS.md` for full detail on the original design considerations. Features 1-3
(scene-based processing, per-scene strength, auto-crop) have all moved to "Implemented and
verified working" above; nothing remains in this list.

**Rust rewrite targets**, scoped in full in `docs/REQUIREMENTS.md`'s "5. Rust rewrite candidates"
section. Integration approach for all three: PyO3 + `maturin`, narrow per-function bindings (not a
wholesale module port) so existing call sites change minimally. Two other pieces were considered
and explicitly rejected for now (FFmpeg subprocess orchestration, AI model inference calls
themselves) — see REQUIREMENTS.md for why.

1. **Scene-detection frame differencing** (`_detect_scene_changes()`, `core/analysis.py`) — **IMPLEMENTED
   2026-07-13.** New crate `rust/avf_scenes/` (PyO3 + maturin, workspace member of the root
   `pyproject.toml`/`uv.lock` via `[tool.uv.workspace]`/`[tool.uv.sources]`, so `uv sync` builds
   and installs it automatically). Decodes via a piped `ffmpeg -f rawvideo` subprocess
   (`format=gray,scale=320:180:flags=bilinear`) instead of `cv2.VideoCapture`, computes the same
   mean-absolute-luma-diff metric, and releases the GIL for the whole decode/diff loop
   (`Python::detach`), re-acquiring only to fire the progress callback. `_detect_scene_changes()`
   in `core/analysis.py` now dispatches to the Rust path when `avf_scenes` imports successfully,
   falling back to the original pure-Python/OpenCV implementation
   (`_detect_scene_changes_python`) with a DEBUG log line otherwise — see AGENTS.md's Setup &
   Commands for the build step. Verified: identical scene boundaries and cut counts to the Python
   implementation on the calibration fixture and a synthetic pan+hard-cut motion clip
   (`TestRustPythonParity` in `tests/unit/test_scene_detection.py`), with per-cut `diff_score`
   divergence under 0.002 (decode/scale path differences: ffmpeg's bilinear scale + `format=gray`
   vs. OpenCV's `INTER_LINEAR` + BT.601 `cvtColor`). Real-world speed on a 54s/1620-frame 1080p60
   test clip: Python 2.55s (~635 fps) vs. Rust 1.73s (~936 fps), ~1.5x.
2. **Perceptual hashing / duplicate detection** (`compute_video_hash`/`compute_video_dhash`,
   `core/analysis.py`) — **IMPLEMENTED 2026-07-13.** New crate `rust/avf_hashing/` (same
   workspace/build pattern as `avf_scenes` above). Never used in production (confirmed by user
   2026-07-12: no stored hash values exist anywhere), so this was free to pick the best algorithm
   rather than port ahash/dhash as-is: landed as **pHash** (DCT-based), hand-rolled in Rust rather
   than via the `img_hash` crate (evaluated and rejected -- it would pull in the `image` crate's
   ~24 transitive codec dependencies just to wrap raw bytes this crate already gets from its own
   ffmpeg pipe). `compute_video_hash()`/`compute_video_dhash()`/`hash_similarity()` are replaced
   wholesale by `compute_video_phash()`/`hash_similarity()` (hex-string 64-bit hashes); the
   pure-Python fallback implements the identical pHash algorithm in NumPy, not the retired ahash/
   dhash. `analysis.duplicate_detection.hash_type` config key removed (no longer meaningful with
   one algorithm); `similarity_threshold` default changed 0.95 -> 0.85, recalibrated against real
   measured scores. Verified behaviorally against a real ffmpeg-generated near-duplicate/non-
   duplicate test set (`tests/unit/test_hashing.py::TestHashSeparation`): near-duplicate pairs
   (same source, different CRF/resolution/trim) scored 0.9375-1.0 similarity; non-duplicate pairs
   (distinct lavfi sources, including cross-comparing near-duplicate variants of different
   sources) scored 0.3438-0.5781 -- a wide margin either side of the 0.85 threshold. See
   `docs/REQUIREMENTS.md` R5.2 for the full algorithm-choice writeup.
3. **Chunked AI frame I/O overlap** (`PrefetchIterator`/`AsyncVideoWriter`,
   `ai/frame_processor.py`) — **IMPLEMENTED (lean v1 scope).** New crate `rust/avf_framepipe/`
   (same workspace/build pattern as `avf_scenes`/`avf_hashing` above): `FrameReader`/`FrameWriter`
   PyO3 classes, each a background OS thread + piped `ffmpeg` subprocess handing frames across a
   bounded `sync_channel` (no NVDEC, no buffer-lease pooling -- the >85% compute-bound scope gate
   in `docs/REQUIREMENTS.md` R5.3's 2026-07-14 measurement meant the win here is architecture
   robustness and headroom for lighter/faster AI passes, not raw throughput). `ai/frame_pipe.py`'s
   `get_frame_reader()`/`get_frame_writer()` factories wire it into `upscale`/`deblock`/
   `denoise_video`'s stage loops, falling back to the pre-existing `frame_processor.py` machinery
   (unchanged) when the extension isn't built -- same lazy-import-with-fallback contract as
   `avf_scenes`/`avf_hashing`. New config keys `stages.<name>.read_ahead` (default 2) and
   `stages.<name>.write_queue_depth` (default 4) on the three stages. Verified: a differential
   identity test proves the Rust and Python transports are bit-exact-interchangeable
   (`tests/unit/test_frame_pipe.py::TestTransportIdentity`), and a 1200-frame soak test confirms
   bounded (not full-video) memory growth (`TestFramePipeSoak`). `interpolate`'s AI/RIFE path is
   unchanged (still buffers frames in a list via `frame_processor.py` directly, not this adapter).

Feature 4 (NCNN backend) is now implemented: `stages.upscale.backend: ncnn` runs Real-ESRGAN
in-process over Vulkan. Output parity with torch is confirmed on a real frame. Re-verified after
fixing this dev box's Vulkan/NVIDIA driver-version passthrough mismatch (a host/LXC config issue,
not a project bug) so Vulkan now sees the RTX 5060 Ti directly, not just the iGPU it fell back to
before — and the dGPU numbers are **not** a win for this backend on this hardware:
- 320x240 untiled: 5.1 fps on the RTX 5060 Ti via ncnn/Vulkan, vs. 7.33 fps via the torch/CUDA
  backend at the same size on the same card -- ncnn is slower here even on the "real" GPU.
- 1280x720 untiled: reliably raises a Vulkan out-of-memory error (`vkAllocateMemory failed -2`)
  despite 16GB of VRAM being available. Explicitly enabling `net.opt.use_fp16_storage` /
  `use_fp16_packed` / `use_fp16_arithmetic` (now done in `ai/backends/ncnn_common.py`) made no
  measurable difference to either the OOM or the timing -- the flags verifiably persist at the
  Python level, so this looks like a per-layer Vulkan shader capability gap in the generic
  `ncnn` PyPI package's bundled shaders for this graph (the same class of limitation as RIFE's
  missing custom layer below), not a config problem.
- 1280x720 with `stages.upscale.tile_size` forced to 512: succeeds (correct output, valid pixel
  content) but took ~36 seconds for a single frame in local testing -- tiling avoids the crash
  but at a cost that makes it impractical for real use at this resolution on this backend/binding
  combination today.

Net assessment for upscale/ncnn: on hardware where CUDA already works well (this project's
primary dev target), `backend: torch` remains the right default and this doesn't change. The
ncnn backend's value proposition is still portability to AMD/Intel/iGPU systems without a working
CUDA path, not speed -- and even there, expect it to be usable mainly at modest frame sizes until
the upstream generic Python bindings close the shader/layer gaps noted above.

RIFE's ncnn backend (`stages.interpolate.backend: ncnn`) is a different story: it's implemented
via `rife-ncnn-vulkan-python` (a *separate* PyPI package from the generic `ncnn` bindings above --
see `ai/backends/ncnn_interpolate.py`'s docstring for why RIFE needs its own package: the official
RIFE ncnn graph depends on a custom `rife.Warp` ncnn layer that only this package's SWIG-wrapped
upstream C++ build registers) and is verified working *and* fast:
- Raw backend (`NcnnInterpolateBackend.interpolate()`, includes BGR<->RGB conversion): 81.8 fps
  interpolating a real 1280x720 frame pair on the RTX 5060 Ti (Vulkan device index 1), warm.
- Full `interpolate` stage end-to-end (frame extraction, inference, encode) on the same hardware
  and resolution, real `avf process` run through the AI path: ~18 fps throughput (47 interpolated
  frames + I/O + libx264 encode in ~2.6s), producing a correctly-timed (48 tbr), non-black, valid
  output.
- On the AMD iGPU (RADV RAPHAEL_MENDOCINO, Vulkan device index 0): ~4.5 fps, notably slower than
  the dGPU but still correct, non-degenerate output. `gpu.vulkan_device` (default `0`) now
  selects which Vulkan device the ncnn backend uses for both upscale and interpolate -- set it to
  the dGPU's index (find it via `ncnn.get_gpu_count()`/`get_gpu_info(i).device_name()`, or `avf
  gpu-info` once that reports ncnn devices -- still a TODO, see `ai/WIRING.md`) to get the faster
  numbers above instead of the iGPU default.

This backend resolves our own cached `rife_v4.6` ncnn model (nihui's official "rife-v4"
single-flownet release asset already in `NCNN_MODEL_REGISTRY`, kept consistent with the torch
backend's default) into the `flownet.param`/`flownet.bin`-named directory layout
`rife-ncnn-vulkan-python` expects, rather than falling back to the older checkpoint that package
bundles internally under its own `rife-v4` model directory.

Also still open from earlier planning and not superseded by the above:
- Smart/auto-tuned quality parameters based on content analysis.
- Hardware encoding paths beyond what FFmpeg's `-hwaccel` already covers opportunistically
  (explicit NVENC/QSV/VAAPI/VideoToolbox encoder selection, distinct from decode-side hwaccel).
- **Multi-GPU inference** (IN SCOPE, deferred): distributing scene/chunk AI inference across
  multiple GPUs, including heterogeneous VRAM sizes (e.g. don't schedule the same chunk size onto
  a 8GB card and a 24GB card). Today, `gpu.max_concurrent_inferences` (default `1`, a process-wide
  `threading.Semaphore` in `ai/torch_utils.get_gpu_inference_semaphore()`) bounds concurrent GPU
  AI inferences in scene mode to avoid VRAM contention on a *single* GPU -- that's the single-GPU
  placeholder this feature will generalize into a scheduler that's aware of which physical device
  each concurrent inference lands on and how much VRAM it has, rather than one process-wide count.
- **Output handling, run reporting, and config tooling** (APPROVED 2026-07-18, next up): the
  full requirement set lives in `docs/REQUIREMENTS.md` feature 6 (6.1-6.8) — skip-not-fail for
  existing outputs, spec-checking existing outputs with rename-or-overwrite of mismatches,
  input probe failure policy, per-video stage/mode summary, media-info + timing
  instrumentation, structured JSON run report, PII-clean log variant, and
  `avf config clean|upgrade|dump`. Agreed delivery: four commits, grouped as documented there.
- Plugin/custom-stage system, face restoration (GFPGAN/CodeFormer), audio enhancement
  (Demucs-based denoise is a config stub — `stages.denoise_audio` exists in `Config.DEFAULTS` but
  there is no corresponding registered stage implementing it), enterprise/cloud/API-service
  features.

## Contribution Areas

1. **AI Models**: Help integrate and optimize AI models (including the planned ncnn backend)
2. **GUI Development**: Verify and extend the Qt interface
3. **Testing**: Cross-platform testing and bug reports, especially around VLM providers and the
   GUI, which are the least-verified surfaces today
4. **Documentation**: Keep `AGENTS.md`, this file, and `docs/config.example.yaml` in sync with
   code — see the standing rule at the top of `AGENTS.md`
5. **Presets**: Create and share processing presets

## Feedback Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Architecture decisions and planning
