# Auto Video Fixer - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
