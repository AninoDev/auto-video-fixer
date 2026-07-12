# Auto Video Fixer - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
