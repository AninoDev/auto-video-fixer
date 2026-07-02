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
