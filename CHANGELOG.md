# Auto Video Fixer - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`gpu.vulkan_device` (default `0`)**: selects which Vulkan physical device index the ncnn
  backend (`stages.upscale.backend: ncnn`, `stages.interpolate.backend: ncnn`) runs on. Both
  `RealESRGANUpscaler`/`RIFEInterpolator` and the stages constructing them now thread this
  through; previously it was hardcoded to device 0, which on a multi-GPU machine may not be the
  fastest one visible (e.g. an integrated GPU can enumerate before a passed-through discrete
  GPU) -- see `avf gpu-info`/`ncnn.get_gpu_count()` to find the right index.
- **RIFE's ncnn/Vulkan interpolation backend is now real and fast**: replaces the previous
  generic-`ncnn`-bindings implementation (which could never load the official RIFE graph -- it
  needs a custom `rife.Warp` ncnn layer the generic bindings don't register) with the
  `rife-ncnn-vulkan-python` package, a SWIG wrapper around the actual upstream C++ tool. No
  prebuilt wheel exists for Python 3.14 yet; building from source needs the `swig` system package
  and `CMAKE_POLICY_VERSION_MINIMUM=3.5` set for the install (works around a stale vendored
  `cmake_minimum_required()` in the package's bundled ncnn snapshot that modern CMake rejects --
  see the `ncnn` extra's comment in `pyproject.toml`). Verified: 81.8 fps interpolating a real
  1280x720 frame pair on an RTX 5060 Ti (raw backend), ~18 fps through the full `interpolate`
  stage end-to-end, correct non-black output.
- **Planned: three Rust rewrite targets** (not yet implemented, scoped for a near-term push) --
  scene-detection frame differencing, perceptual-hash duplicate detection, and the chunked AI
  frame prefetch/writer threading. See `docs/REQUIREMENTS.md`'s "5. Rust rewrite candidates" for
  full scope and `AGENTS.md` for a pointer.
- **`analysis.vlm.max_tokens` (default 1024) and `analysis.llm.max_tokens` (default 4096)**:
  the VLM/LLM response token budget was previously hardcoded to 1000 in the request payload
  (client-side -- not a server setting), which truncated reasoning models mid-thought (their
  thinking tokens count against the same budget) and made the scene coordinator fail open.
  Now configurable; Ollama requests map it to `options.num_predict`.
- **Auto-crop stage (opt-in, off by default)**: new `crop` stage (`core/stages/crop.py`) detects
  a video's true content bounds -- the union over the whole video (the furthest real content
  ever reaches toward each edge), not a per-frame crop -- via FFmpeg
  `cropdetect=limit=<L>:round=<R>:reset=0` and crops to that single window
  (`crop=w:h:x:y`, `libx264 -crf 18`, `-c:a copy`). Strips letterboxing/pillarboxing from source
  video and any residual black border stabilization can itself introduce. Enable with
  `--enable-stage crop` or `stages.crop.enabled: true`; runs right after `stabilize` and before
  every other enhancement/AI stage in `Pipeline.optimize_stage_order()` so deblock/denoise/
  upscale/interpolate never spend compute on pixels about to be cropped away. New config:
  `stages.crop.limit` (default 24), `.round` (default 2), `.min_crop_px` (default 8 -- skip if
  the crop would save fewer pixels than this in both dimensions), `.analyze_duration_sec` (0 =
  full-video scan, default; >0 = sample only the first N seconds), `.vlm_check` (default false),
  `.vlm_policy` (`"warn"` default | `"skip"`). New CLI flag `--crop-limit INT`.
  - **VLM-assisted watermark/content disambiguation (opt-in, `stages.crop.vlm_check`)**: a
    watermark/logo sitting in the border area can fool naive cropdetect either way -- bright
    enough to widen the kept region, or dim enough to get cropped away with no way for cropdetect
    alone to flag it as meaningful. When enabled (requires `analysis.vlm.enabled: true`), one
    frame is rendered twice (plain, and with the proposed crop box drawn via `drawbox`) and sent
    to the VLM with a fixed internal prompt (`core.analysis.run_crop_vlm_check`, parsed via a
    tolerant JSON-then-text fallback, `_parse_crop_vlm_response`). `vlm_policy: "warn"` logs a
    WARNING and crops anyway (default); `"skip"` skips cropping the video entirely. Fails open on
    any VLM error (unreachable endpoint, exception, unparseable response) -- logs a WARNING and
    proceeds with the plain cropdetect result. An `"expand"` policy (grow the crop box to include
    just the flagged region) was considered and rejected: VLMs don't reliably return pixel
    coordinates, so there's nothing to expand to.
  - Verified against real FFmpeg-generated fixtures: a 640x360-in-640x480 letterboxed video crops
    back to ~640x360 with no border left on re-scan; a border-free video is left uncropped
    (via `should_run()`'s quick pre-filter or `execute()`'s `min_crop_px` skip); a letterboxed
    video with a dim in-border overlay is cropped away in plain mode (documented limitation --
    plain cropdetect has no notion of "meaningful overlay" vs. background), while
    `vlm_check` + a mocked VLM response + `vlm_policy: "skip"` leaves that video uncropped. See
    `tests/integration/test_integration.py::TestCropIntegration`.
- **Optional ncnn/Vulkan inference backend** for the upscale and interpolate stages
  (`stages.upscale.backend: ncnn`, `stages.interpolate.backend: ncnn`) — an alternative to
  PyTorch/CUDA that works on AMD/Intel/integrated GPUs with no CUDA-matched torch build. Upscale
  runs in-process via the generic `ncnn` Python package. RIFE interpolation runs via a separate
  package, `rife-ncnn-vulkan-python` (also part of the new `ncnn` extra) — RIFE's official ncnn
  graph needs a custom `rife.Warp` ncnn layer that only this package's wrapped upstream C++ build
  registers; the generic `ncnn` package doesn't. `rife-ncnn-vulkan-python` has no prebuilt wheel
  for Python 3.14 yet: building it from source needs the `swig` system package and, against
  modern CMake, `CMAKE_POLICY_VERSION_MINIMUM=3.5` set for the install command (see the `ncnn`
  extra's comment in `pyproject.toml`). Both backends are verified working (correct, non-black
  output); RIFE/ncnn is also fast — ~82 fps interpolating a real 1280x720 frame pair via the raw
  backend, ~18 fps end-to-end through the full `interpolate` stage, both on an RTX 5060 Ti over
  Vulkan. ncnn models (.param/.bin) get their own SHA256-verified registry and download path.
  Unavailability (missing package, no Vulkan device, no models) flows through the existing
  `ai_fallback` policy.
- **Scene-based processing (opt-in, off by default)**: new `core/scenes.py` splits a video at
  existing scene-detection boundaries (`VideoAnalyzer.detect_events`), re-encodes each scene as
  its own clip, runs `stabilize`/`interpolate` per-scene, and concatenates the result (video and
  audio cut at the same boundaries, remuxed together) before the remaining whole-video stages
  (upscale/denoise/deblock/normalize/encode) run. Enable with config `scenes.enabled: true` or
  CLI `--scene-mode`; the whole-video path is completely unaffected when disabled (zero behavior
  change). See AGENTS.md's "Scene mode" section for the full design.
  - **Never interpolate across a cut**: frame interpolation now always runs per-scene when scene
    mode is on, so RIFE/minterpolate never synthesizes a blend frame between two unrelated shots.
    Verified on a synthetic multi-scene clip: zero ghosting/blend frames at any sampled boundary
    frame; output duration/fps stay within the documented tolerance of the original.
  - **Per-scene stabilization strength tiering**: each scene gets its own `vidstabdetect` pass;
    scenes below `stages.stabilize.threshold` skip stabilization entirely, scenes above
    `scenes.stabilize.aggressive_shake_threshold` get a second pass with smoothness multiplied by
    `scenes.stabilize.aggressive_smoothness_multiplier` (default tiers: skip / normal /
    aggressive). New config: `scenes.stabilize.enabled`, `.aggressive_shake_threshold`,
    `.aggressive_smoothness_multiplier`.
  - **Drop non-content scenes (opt-in, requires scene mode)**: `scenes.drop_non_content: true` (+
    CLI `--drop-non-content`) runs per-scene VLM sampling (`VideoAnalyzer.
    run_vlm_analysis_for_scene`, frame count scaled down for short scenes) followed by a
    coordinating text-LLM pass (`run_scene_coordinator`, new `analysis.llm` config section --
    provider/model/api_url/api_key/allow_http, same HTTPS/allow_http gate as `analysis.vlm`) that
    reviews all per-scene summaries together and flags scenes that aren't part of the main
    content (e.g. a "like and subscribe" interstitial). **Fails open on any failure** -- unparseable
    response, out-of-range/non-integer drop index, unreachable endpoint, or an exception --
    logging a WARNING and keeping every scene; verified live against a real (LAN) VLM endpoint,
    including a real failure case (the coordinator model's reasoning tokens exhausted its
    response budget before emitting JSON) that correctly triggered fail-open.
  - `avf process --scene-mode/--no-scene-mode` and `--drop-non-content/--no-drop-non-content` CLI
    flags (override `scenes.enabled`/`scenes.drop_non_content` for the run).
- **Parallel-chunked traditional frame interpolation**: `minterpolate` is single-threaded per
  ffmpeg process and slow on long clips. `InterpolateStage`'s traditional path now splits a long
  enough input into N frame-index-aligned chunks (1-frame overlap, trimmed on concat) and runs
  them as parallel ffmpeg processes, bounded by `stages.interpolate.parallel_chunks` (0 = auto,
  `min(cpu_count, 8)`; 1 = previous serial behavior) and `stages.interpolate.
  min_chunk_duration_sec`. Applies to whole-video interpolation AND to each scene's interpolation
  pass under scene mode (worker budget shared between scenes-in-parallel and chunks-per-scene so
  the two pools don't oversubscribe each other -- see `core/scenes.py::_scene_worker_budget`).
  The AI/RIFE path is unaffected and stays serial (GPU-bound). Benchmarked on a 60s 720p30 clip
  interpolated to 60fps on an 8-core machine: ~270s serial vs. ~58-64s with `parallel_chunks=8`
  (~4.2-4.7x). Frame counts between serial and chunked runs are close but not bit-exact (within
  ~1-2%, from `minterpolate`'s own per-chunk duration-based frame-count rounding, not from
  dropped/duplicated content) -- see `_execute_traditional_parallel`'s docstring.
- **`avf analyze` multi-file support**: now accepts multiple files and/or directories
  (`avf analyze PATHS...`), with `--recursive`/`-r` mirroring `process`'s directory scanning. A
  failure analyzing one file is logged and skipped; the remaining files still run, and the
  command exits non-zero if any file failed.
- **`avf analyze --full`**: prints the complete, untruncated VLM summary per file in a Rich
  panel below the results table (the table's own "VLM Summary" row stays a truncated preview,
  now with an explicit "(use --full for full text)" hint instead of a bare `...`). The full
  summary (plus tags/objects/rating) is now also always logged at INFO — previously it was
  truncated in the table and never appeared anywhere in full, including the always-on per-run
  DEBUG log file.
- **`avf analyze --csv PATH`**: writes one row per analyzed video (filepath, filename, duration,
  resolution, framerate, codec, has_video/has_audio/hdr, scenes_detected, and — when VLM ran —
  the FULL summary, `;`-joined tags/objects, and content_rating) to a UTF-8 CSV via Python's
  `csv` module (proper quoting for summaries containing commas/newlines). Overwrites `PATH` if
  it already exists rather than appending across runs.
- **`analysis.vlm.prompt_append` / `prompt_override` / `system_prompt_override`** config keys,
  plus `avf analyze --prompt-append TEXT` / `--prompt-override TEXT` CLI flags (CLI overrides
  config for that run): lets a user add job-specific context to the VLM prompt (e.g. "these are
  trail-camera clips, focus on wildlife species") or replace it/the system prompt entirely.
  `_parse_vlm_response()`'s existing non-JSON fallback (treats the raw response as the summary)
  means an override that changes the requested JSON output format degrades gracefully instead
  of erroring.
- **`avf analyze` progress reporting**: `VideoAnalyzer.analyze()`/`detect_events()`/
  `run_vlm_analysis()` now accept an optional `progress_callback(phase, detail)` invoked at
  phase transitions (probing, scene-detection start/periodic-progress/done, VLM frame
  sampling/request, done) — scene detection's frame-differencing loop reports roughly every 5%.
  The CLI wires this to a live Rich status line on a TTY (plain scrolling INFO log lines
  otherwise), and every phase transition is logged at INFO regardless, so slow videos/VLM
  endpoints no longer look hung with no feedback.
- **`avf analyze --scene-threshold FLOAT` / `--min-scene-duration FLOAT`**: per-run overrides
  for `analysis.event_detection.scene_change_threshold`/`min_scene_duration_sec` (which now
  actually flow into `_detect_scene_changes()` as a `threshold` param on
  `VideoAnalyzer.detect_events()` — previously the threshold was config-only with no override
  path). See "Changed" below for the recalibrated default.
- **Scene-detection score visibility**: `detect_events()` now logs, per file, the effective
  `threshold`/`min_duration` actually used (config or CLI override — makes it obvious an
  override took effect) and a one-line cut-score summary (count, min/median/max `diff_score`).
  `_detect_scene_changes()` additionally tracks and logs up to the 10 highest "near-miss"
  `diff_score`s that fell *below* threshold but *above* threshold/4, with timestamps — i.e.
  plausible cuts a lower `--scene-threshold` would catch, without having to rerun detection to
  find out. `avf analyze --full` now prints each scene's boundary `diff_score` in the console
  scene listing, and `--csv` gained a `scene_boundaries` column (`;`-joined
  `t=<seconds>s@<confidence>` per scene). Prompted by a real-world report of a flat scene count
  across `--scene-threshold` 0.01-0.30 on a real clip — this makes that kind of bimodal-vs-buggy
  question answerable from the log/CSV instead of guesswork.
- **`avf analyze --max-sample-frames INT` / `--sample-interval FLOAT` / `--vlm-model TEXT` /
  `--vlm-url TEXT`**: per-run overrides for `analysis.vlm.max_sample_frames`/
  `sample_interval_sec`/`model`/`api_url`. `--vlm-url` goes through the same
  HTTPS-required-for-non-loopback gate as the config value; there is deliberately no CLI flag for
  `analysis.vlm.allow_http` or `api_key` — those stay config-file-only. (Fixes a latent bug along
  the way: `analysis.vlm.sample_interval_sec` in config was never actually read —
  `run_vlm_analysis()` hardcoded a 10.0 default regardless — so this also makes that config key
  do something.)
- **`analysis.vlm.allow_http`** (default `false`): the OpenAI-compatible `api` VLM provider
  refuses plain-HTTP endpoints on non-loopback hosts by default (frames and the API key would
  travel unencrypted); setting this to `true` permits plain HTTP for e.g. a LAN inference box
  (llama.cpp, LM Studio, vLLM) on a private subnet. HTTPS and loopback never need it. The
  refusal log message now names the override.
- **AI-fallback policy** for AI-capable stages (upscale, interpolate, denoise_video, deblock):
  `general.ai_fallback` (default `true`) plus per-stage `stages.<name>.ai_fallback` (default
  `null` = inherit) control whether a stage silently falls back to its traditional FFmpeg
  method when the AI path can't run (PyTorch missing, model load failure, inference exception,
  CUDA OOM after tiling retries), or FAILS outright with a named cause. New CLI flags
  `--ai-fallback` / `--no-ai-fallback`.
- **Startup settings banner**: every run now logs the avf version, full invocation (`argv`), which
  config file is in use, and a redacted diff of effective config vs. defaults at INFO, plus the
  full effective config at DEBUG.
- **Automatic per-run DEBUG log file**: every invocation writes a timestamped log file under the
  platform state directory (`~/.local/state/auto-video-fixer/logs/` on Linux), independent of
  console verbosity; retains the newest 50 and prunes older ones at startup. `--log-file PATH`
  redirects the run's log file to a custom path instead of the automatic location.
- **Global `--config PATH` flag and `AVF_CONFIG` environment variable** to select an alternate
  config file (flag > env var > default platform path); an explicitly-given path that doesn't
  exist is now a hard error instead of a silent fallback to defaults.
- Tiled AI inference with reactive CUDA-OOM retry and a per-stage `stages.<name>.tile_size`
  override, for upscale/deblock/denoise_video.
- `RealESRGAN_x2plus` added to the AI model registry, and auto-selected in place of x4plus
  whenever a pass only needs scale ≤2 (upscale) or scale=1 (deblock/denoise) — measurably faster
  since x2plus's architecture natively shrinks its working resolution instead of computing a
  native 4x result and discarding most of it.

### Changed
- **`analysis.event_detection.scene_change_threshold` default lowered from `0.3` to `0.15`** —
  calibrated against a synthetic ground-truth clip (12 visually distinct 5s segments, 11 known
  hard cuts): the old default found only 9/12 segments (missed 3 real cuts scoring 0.18-0.27 on
  the frame-differencing metric), while 0.15 finds all 12 with zero false positives (measured
  max within-segment score 0.040, min actual-cut score 0.184). Matches a user report of a real
  3-minute clip (~30 real cuts) where the old default found only ~5. See
  `_detect_scene_changes()`'s docstring in `core/analysis.py` for the full metric writeup, and
  `avf analyze --scene-threshold`/`--min-scene-duration` above for per-run overrides.
- **Intermediate pipeline temp files always use `.mkv`**, regardless of input/output container
  (MKV can hold any codec the pipeline's hardcoded intermediate encodes use; some source
  containers, e.g. WebM, cannot). Final output extension is now driven by
  `general.output_container` (default `"mp4"`) rather than always mirroring the input's
  extension.
- Temp file cleanup is now unconditional, including on stage failure or job cancellation; a
  surviving intermediate temp file is promoted (moved) to the job's output path when the
  terminal stage was skipped or produced no output, instead of being silently deleted.
- `scan_directory()` now skips hidden files, including orphaned `.avf_*` intermediate temp files.
- `--stage NAME` now overrides a stage's `enabled: false` in config for stages explicitly
  requested by name, matching its documented "replaces the preset/auto-determined list"
  behavior.
- The quality gate (SSIM/PSNR) now scales the reference video to the output's resolution before
  comparing, and reports "measurement failed" (leaving quality fields as "not checked") instead
  of a fake `0.0` score when the FFmpeg comparison itself fails.
- AI upscale/deblock/denoise now use the `channels_last` (NHWC) tensor memory format on CUDA,
  measured at roughly 1.6→11 fps for a 720p→1080p pass on the project's target GPU.
- Frame decode/GPU-inference/write are now overlapped via a prefetching decode thread and an
  async writer thread instead of serializing read → infer → write per chunk.

### Fixed
- **`avf analyze`'s per-scene console listing was silently dropping its `[event_type]` prefix**
  (e.g. `[scene_change]`, `[talking_head]`) — Rich's console markup parser (on by default)
  swallows any `[...]` segment that isn't a recognized style tag instead of erroring, so the tag
  just vanished. Found while adding the `--full` boundary-confidence display (which used the
  same bracket pattern and would have had the identical problem). Fixed by printing that line
  with `markup=False`.
- **AI upscaling produced solid-black output on GPU** — the Real-ESRGAN RRDB residual-dense
  block was missing its `0.2` residual-scaling factor, causing activations to compound/explode
  into NaNs on real (non-toy) inputs. Verified fixed: AI upscale now produces real, non-black
  output on GPU.
- **Real-ESRGAN x2 support**: added pixel-unshuffle preprocessing for genuine 2x-scale models
  (`RealESRGAN_x2plus`) rather than only supporting native-4x checkpoints downscaled after the
  fact.
- **RIFE frame interpolation duration bug** — interpolated output was written at the *input*
  video's fps instead of `input_fps * factor`, which stretched the clip's duration instead of
  increasing its framerate while preserving duration.
- **Stabilize zoom could only zoom OUT, never in** — the previous hand-computed zoom percentage
  had an inverted sign and could not actually remove borders introduced by stabilization. Now
  delegates the zoom *amount* to `vidstabtransform`'s own `optzoom=1` ("optimal static zoom")
  when `stages.stabilize.zoom_enabled` is true, which operates on the filter's own smoothed
  camera path instead of raw per-block motion vectors.
- **VFR / YouTube-origin input played at ~2x speed then froze** — framerate probing (both
  `ffmpeg_utils._parse_fps` and `stabilize.py`'s internal probe) now prefers `avg_frame_rate`
  (the honest "frame count / duration" rate) over `r_frame_rate` ("tbr", ffmpeg's declared rate,
  which can be double the true average for VFR/YouTube sources) anywhere frame count is
  reconciled with wall-clock time, such as the stabilize stage's raw-pipe decode→transform
  handoff. Both also now handle ffprobe's `"0/0"` (undefined rate) response without crashing.
- Chunked AI code paths (upscale/deblock/denoise_video/interpolate, used for inputs with >1000
  frames) no longer raise `UnboundLocalError` on the progress-callback variable.
- Removed a stray reference to the wrong local variable name when muxing AI-upscaled frames back
  to video.
- Missing top-level imports in `deblock.py` (`os`) and `denoise_video.py` (`probe`) that only
  worked by accident via a later local import.

### Added (from earlier Unreleased entries, retained)
- Initial project structure
- Configuration system with YAML storage
- Core pipeline engine
- CLI interface with Click
- Basic processing stages (detect, stabilize, deblock, denoise, upscale, interpolate, normalize, encode, remux, speed, hdr)
- Stage registry and automatic discovery
- Smart stage ordering
- Preset system with 7 built-in presets
- Video analysis utilities
- FFmpeg integration with hardware acceleration detection
- Quality estimation (VMAF, PSNR, SSIM)
- Scene detection
- Duplicate detection
- Comprehensive test suite
- CI/CD pipeline
- Documentation (User Guide, Developer Guide, API, Roadmap, Implementation Plan)

### Changed
- Initial release
- Raised minimum supported Python version to 3.14

### Fixed
- Debounce filter compatibility across FFmpeg versions
- Temp file management
- Stage registration
- **Stabilization B-frame artifacting** - Fixed vid.stab buffer corruption bug (github.com/georgmartius/vid.stab#144) by piping raw video between decode and transform processes instead of feeding decoder reference frames directly to vidstabtransform
- **Preset merge overwriting config** - Fixed `_merge_config()` to recursively merge nested dicts instead of replacing entire stage configs. Previously, applying a preset like `-p 1080p60` would overwrite `stages.stabilize` with just `{"enabled": true}`, losing all custom settings (smoothness, maxshift, zoom_enabled, etc.)
- **CLI model-info crash** - Fixed `set(list_cached_models())` error where cached models are dicts, not strings. Now extracts model names with set comprehension
- **CLI styling** - Fixed Rich markup error (`[Cyan]` not recognized, changed to `[blue]`)
- **Test isolation** - Fixed `test_get_nested_value` to use `tmp_path` config instead of reading from system config directory
- **Dead code in upscale stage** - Removed unreachable code in `_execute_ai()` that referenced undefined `sf` and `target_width`/`target_height` variables from a previous refactoring
- **Unused imports/variables** - Removed unused `logging` import, `max_mag` and `prev_time` unused variables in stabilize stage

---

## [0.2.0] - 2026-06-28

### Added
- **AI upscaling with Real-ESRGAN** - PyTorch-based super-resolution using RRDBNet architecture
  - Real-ESRGAN x4plus, x2plus, and anime 6B model support
  - Automatic model download and caching
  - Test-time augmentation (TTA) for improved quality
  - FP16 inference on CUDA for faster processing
- **AI frame interpolation with RIFE** - Real-Time Intermediate Flow Estimation
  - Bidirectional optical flow estimation
  - Multi-scale feature pyramid for accurate motion estimation
  - Warp-based frame synthesis with error minimization
  - RIFE v4.6 and v4.11 model support
- **AI denoising** - Real-ESRGAN-based noise reduction (denoise mode at scale=1)
- **AI/ML module** (`autovideofixer.ai/`) with:
  - PyTorch device and tensor utilities
  - Model cache and download management
  - Frame extraction and conversion utilities
  - Real-ESRGAN wrapper (RRDBNet implementation)
  - RIFE wrapper (IFNet + EMD architecture)
- **Improved quality estimation** - SSIM/PSNR via FFmpeg with per-frame and aggregate metrics
- AI config options: scale_factor, tta_mode for upscaling stage
- Graceful fallback to traditional FFmpeg methods when AI dependencies unavailable

### Changed
- **upscale** stage: `_execute_ai()` now uses Real-ESRGAN instead of placeholder
- **interpolate** stage: `_execute_ai()` now uses RIFE instead of placeholder
- **denoise_video** stage: `_execute_ai()` now uses Real-ESRGAN denoise instead of placeholder
- Version bumped to 0.2.0

### Fixed
- Model loading with params_ema wrapper format (Real-ESRGAN compatibility)

### Security
- No security advisories

---

## [0.3.0] - 2026-06-30

### Added
- **VLM (Vision Language Model) integration** - Analyze video content using AI language models
  - Ollama support (local models, default at localhost:11434)
  - OpenAI Vision API support (GPT-4o, etc.)
  - Custom API endpoint support (OpenAI-compatible format)
  - Local VLM server support
  - JSON response parsing with fallback for non-JSON responses
  - Configurable model, API URL, API key, and sample frame count
- **Event classification** - Automatic classification of detected scenes (talking_head, action, landscape, text_overlay, transition) using frame analysis heuristics
- **Clip extraction** - Extract detected scenes as separate video clips via FFmpeg
  - `VideoClip` dataclass with source path, timing, and output path
  - `extract_scenes_as_clips()` for batch extraction
- **Improved perceptual hashing** - Added difference hash (dhash) in addition to average hash (ahash)
  - dhash compares adjacent pixels for structural pattern detection
  - More robust for videos with similar content but different lighting
- **Batch deduplication** - `find_duplicates()` method to find all duplicate groups in a file batch
  - Groups files by perceptual hash similarity
  - Returns deduplicated groups of 2+ duplicate files
- **New CLI flags for `analyze` command**:
  - `--events` / `--no-events` to enable/disable event detection
  - `--classify` to enable VLM-based event classification
  - `--clip DIR` to extract scenes as clips to a directory
- **Analysis config options**: `max_sample_frames`, `sample_interval_sec`, `classify_events`, `hash_type`

### Changed
- Version bumped to 0.3.0
- `SceneEvent` dataclass: added `duration` property
- `analyze` CLI command: enhanced output with scene duration, event type descriptions, content rating
- VLM frame extraction: now uses `max_frames` parameter instead of fixed limit

### Fixed
- Video analysis `analyze()` now properly respects `include_vlm` and `include_events` boolean flags with None-sentinel pattern

---

## [0.1.0] - 2026-06-27

### Added
- Core architecture and pipeline engine
- 12 processing stages
- CLI interface (process, analyze, presets, find-duplicates, gpu-info)
- Configuration management
- Preset system
- Basic video analysis
- FFmpeg integration
- Hardware acceleration detection
- Unit test suite
- CI/CD configuration
- Project documentation

[Unreleased]: https://github.com/yourusername/auto-video-fixer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/auto-video-fixer/releases/tag/v0.1.0
