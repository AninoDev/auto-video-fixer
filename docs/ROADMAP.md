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
  hardcoded stage ordering (`optimize_stage_order()`), per-job temp file lifecycle (`.mkv`
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
  exercised against a live VLM endpoint as part of this pass.

### Implemented but not independently verified

These have working code paths and existing unit tests, but have not been checked end-to-end
against real-world inputs the way the traditional pipeline and the AI upscale/interpolate paths
above have been. Treat their behavior as "probably correct, unconfirmed" rather than "done":

- **Scene detection quality** — frame-differencing scene-change detection
  (`_detect_scene_changes`) and heuristic event classification exist and are unit-tested
  (`tests/unit/test_scene_detection.py`); detection accuracy/threshold tuning on real footage is
  unverified.
- **Duplicate detection accuracy** — ahash/dhash + Hamming-distance similarity scoring is
  implemented; real-world false-positive/negative rates are unverified.
- **The Qt (PySide6) GUI** (`gui/main_window.py`, `avf-gui` entry point) — has a test file
  (`tests/unit/test_gui.py`) and basic job-queue/preset/progress wiring, but has not been
  manually driven end-to-end as part of this pass.

### Planned (not implemented)

See `docs/REQUIREMENTS.md` for full detail on the original design considerations. Features 1-3
(scene-based processing, per-scene strength, auto-crop) have all moved to "Implemented and
verified working" above; nothing remains in this list.

**Next up, prioritized soon (2026-07-12):** three Rust rewrite targets, scoped in full in
`docs/REQUIREMENTS.md`'s "5. Rust rewrite candidates" section:

1. **Scene-detection frame differencing** (`_detect_scene_changes()`, `core/analysis.py`) —
   eliminate per-frame Python/GIL overhead and enable true pipelined decode+diff; directly gates
   how usable the scene-based processing feature is on long videos.
2. **Perceptual hashing / duplicate detection** (`compute_video_hash`/`compute_video_dhash`,
   `core/analysis.py`) — never used in production (confirmed by user 2026-07-12: no stored hash
   values exist anywhere), so this is free to pick the best algorithm rather than port ahash/dhash
   as-is; evaluate pHash (more robust to re-encodes/crops than ahash/dhash) and existing Rust
   crates (e.g. `img_hash`) before hand-rolling. Verification is behavioral (separates a
   near-duplicate test set correctly), not bit-identical output.
3. **Chunked AI frame I/O overlap** (`PrefetchIterator`/`AsyncVideoWriter`,
   `ai/frame_processor.py`) — replace GIL-bound Python threading with real OS-thread parallelism
   for CPU-side frame marshalling, now that GPU inference itself is fast enough (7-92 fps
   depending on backend) that CPU-side handling is an increasingly real bottleneck. Most
   architecturally involved of the three; needs its own design pass before implementation.

Integration approach for all three: PyO3 + `maturin`, narrow per-function bindings (not a
wholesale module port) so existing call sites change minimally. Two other pieces were considered
and explicitly rejected for now (FFmpeg subprocess orchestration, AI model inference calls
themselves) — see REQUIREMENTS.md for why.

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
- Multi-GPU support, plugin/custom-stage system, face restoration (GFPGAN/CodeFormer), audio
  enhancement (Demucs-based denoise is a config stub — `stages.denoise_audio` exists in
  `Config.DEFAULTS` but there is no corresponding registered stage implementing it), enterprise/
  cloud/API-service features.

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
