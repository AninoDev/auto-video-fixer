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

### Implemented but not independently verified

These have working code paths and existing unit tests, but have not been checked end-to-end
against real-world inputs the way the traditional pipeline and the AI upscale/interpolate paths
above have been. Treat their behavior as "probably correct, unconfirmed" rather than "done":

- **VLM (Vision Language Model) classification** — Ollama, OpenAI Vision, and generic
  OpenAI-compatible custom-API providers (`analysis.vlm.provider`) are implemented in
  `core/analysis.py` with unit test coverage (`tests/unit/test_vlm.py`), but have not been run
  against a live Ollama instance or OpenAI API key as part of this verification pass.
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

See `docs/REQUIREMENTS.md` for full detail on these — they are user-specified requirements with
design considerations captured, not yet built:

1. **Scene-based processing pipeline** — per-scene VLM sampling + a coordinating LLM pass that
   determines a video's main content and optionally drops non-content scenes (e.g. "like and
   subscribe" interstitials). Strictly opt-in.
2. **Per-scene processing strength** — stabilize shaky scenes more aggressively than stable
   ones; never interpolate across a scene cut.
3. **Auto-crop mode** — `cropdetect`-based true-content-bounds detection, with optional VLM
   assistance to distinguish real content bounds from watermarks/overlays.
4. **NCNN backend option** — in-process (no subprocess spawning) ncnn/Vulkan Real-ESRGAN/RIFE
   as an alternative to the PyTorch/CUDA backend, selectable per-stage
   (`stages.<name>.backend: torch|ncnn`).

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
