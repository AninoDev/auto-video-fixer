# Requirements: Planned Feature Set

Status: **ALL features in this document are PLANNED / NOT IMPLEMENTED** unless explicitly noted
otherwise. This is a requirements/design doc capturing user-specified intent for future work, not
a description of current behavior. For what currently exists, see `docs/ROADMAP.md`'s
"Implemented and verified working" / "Implemented but not independently verified" sections and
`AGENTS.md`.

These four features share a common dependency: the VLM (Vision Language Model) connection layer
described in "0. Foundation" below. That layer already exists in code (`core/analysis.py`,
`analysis.vlm` config block) but is itself unverified end-to-end (see ROADMAP) — so every feature
here that depends on it inherits that same unverified-foundation risk, on top of being unbuilt.

---

## 0. Foundation: VLM config surface (exists, unverified)

All four planned features below build on the existing VLM connection layer. Recorded here for
reference since the new features assume/extend it.

Config block (`Config.DEFAULTS["analysis"]["vlm"]`, `config.py`):

```yaml
analysis:
  vlm:
    enabled: false
    provider: local          # local, ollama, openai, api
    model: llava
    api_key: ""
    api_url: ""
    max_sample_frames: 8
    sample_interval_sec: 10.0
```

- `provider: local` and `provider: ollama` both currently route to the same Ollama-compatible
  chat endpoint (`_run_ollama_vlm` / `_call_ollama`, default base URL
  `http://localhost:11434`) — `local` is effectively an alias for `ollama` today, not a
  separate local-inference path.
- `provider: openai` calls the OpenAI Vision chat-completions endpoint
  (`_run_openai_vlm` / `_call_openai`), requires `api_key`.
- `provider: api` calls a user-supplied OpenAI-compatible endpoint at `api_url`
  (`_run_api_vlm` / `_call_custom_api`); refuses non-HTTPS URLs for non-loopback hosts.
- `max_sample_frames` / `sample_interval_sec` control how many frames are pulled from the video
  and how far apart, via `_extract_sample_frames()`.
- Entry point: `VideoAnalyzer.run_vlm_analysis()` in `core/analysis.py`.
- **Status: implemented, unverified** — has unit tests (`tests/unit/test_vlm.py`) mocking the
  HTTP calls, but has not been exercised against a live Ollama instance or a real OpenAI API key
  as part of this documentation pass. Anything below that depends on VLM output quality
  (scene-drop decisions, watermark/content-bound disambiguation) inherits this same
  unverified-in-practice caveat even once its own code is written.

---

## 1. Scene-based processing pipeline (optional, opt-in)

### Intent

Split a video into scenes (scene detection already exists — `_detect_scene_changes` in
`core/analysis.py`), run VLM analysis on a small number of sampled frames per scene (fewer
frames for very short scenes), then have a coordinating LLM agent review *all* per-scene VLM
results together to determine the video's main subject/purpose, and optionally **drop scenes
that aren't part of the main content** — e.g. a "like and subscribe" interstitial, a
channel-branding bumper, or an unrelated promotional insert spliced into the middle of the video.

### Requirements

- **R1.1** Must be strictly optional — off by default, a config flag (proposed:
  `analysis.scene_pipeline.enabled: false`) and/or a CLI flag (proposed: `--scene-pipeline`) must
  enable it. Existing whole-video processing must be unaffected when disabled.
- **R1.2** Scene boundaries reuse the existing scene-detection code path (`_detect_scene_changes`
  / `detect_events`), not a new detector — avoid duplicating logic that already exists.
- **R1.3** Per-scene VLM sampling: sample N frames per scene, where N scales down for short
  scenes (proposed: `min(max_sample_frames, ceil(scene_duration_sec / sample_interval_sec))`,
  reusing the existing `max_sample_frames`/`sample_interval_sec` config knobs rather than adding
  parallel ones unless a genuine need for scene-specific tuning emerges).
- **R1.4** A coordinating pass (a second LLM call, potentially text-only over the collected
  per-scene VLM descriptions rather than a second vision pass) reviews all scenes' VLM output
  together and produces: (a) a summary of the video's main content/purpose, (b) a per-scene
  keep/drop decision with a stated reason.
- **R1.5** Dropped scenes are excluded from the final output; the remaining scenes are
  concatenated (see feature 2 for why concatenation, not a single continuous encode, is required
  once per-scene processing is in play).
- **R1.6** The keep/drop decision must be auditable — log (at INFO or above) which scenes were
  dropped and why, so a bad drop is diagnosable and not silently destructive. Consider a
  dry-run/report mode that shows the keep/drop plan without actually removing anything, given the
  destructive/lossy nature of dropping content.
- **R1.7** Failure mode: if the VLM/coordinating LLM call fails or is unavailable, the pipeline
  must fail closed to "keep everything" (skip the scene-pipeline step, process the whole video
  normally) rather than fail closed to "drop everything."

### Design considerations

- The coordinating LLM step needs its own model/provider config, which may or may not be the
  same provider as the per-scene VLM (a cheaper/faster text-only model may suffice for
  coordination once per-scene descriptions already exist as text).
- Cost/latency: N scenes × M sampled frames × VLM calls, plus one coordination call, is
  meaningfully more expensive than today's single-pass `analyze`. Needs a cost/time estimate
  surfaced to the user before running on a long video, or a `--dry-run`-style preview.
- False-positive scene drops (dropping real content that merely *resembles* an interstitial) are
  the primary risk — R1.6's auditability requirement exists specifically to make this
  recoverable rather than silent data loss.

---

## 2. Per-scene processing strength (never interpolate across a cut)

### Intent

Different scenes in the same video can need very different processing. A scene shot handheld and
shaky should be stabilized aggressively; a tripod-locked scene should be stabilized lightly or
not at all. Separately — and this is a hard correctness requirement, not a tuning
preference — frame interpolation must **never** run across a scene boundary: interpolating
between the last frame of one shot and the first frame of the next (a hard cut) asks the
interpolation model to synthesize motion between two unrelated images, which produces garbage
frames (ghosting/warping artifacts at the cut).

### Requirements

- **R2.1** Per-scene stabilize strength: shake magnitude already computed per-video via TRF
  analysis (`_analyze_trf_file`/`_movement_extent` in `stabilize.py`) must instead be computed
  **per scene**, and stabilize parameters (`smoothness`, `maxshift`, `zoom_enabled`/
  `zoom_threshold`) adjusted per scene rather than applied uniformly across the whole video.
- **R2.2** Frame interpolation must treat each scene as an independent clip: run interpolation
  within a scene's frame range only, never across a detected scene-change boundary. This applies
  regardless of whether feature 1 (scene-based pipeline / content dropping) is enabled — this is
  a correctness fix for interpolation specifically, not tied to the drop-scenes feature.
- **R2.3** Implementation approach: split the source into per-scene clips (reusing
  `extract_scenes_as_clips`/`VideoClip` from `core/analysis.py`), process each clip
  independently through the relevant stages (at minimum stabilize and interpolate; other stages
  may also benefit from per-scene tuning, e.g. denoise strength for a grainy low-light scene vs.
  a clean daylight scene), then concatenate the processed clips back into one output (FFmpeg
  concat demuxer/filter, matching codec/timebase across segments).
- **R2.4** Concatenation must preserve audio sync — if scenes are processed as separate video-only
  clips and audio is normalized once over the whole original file, the concat step must
  re-attach a single continuous audio track rather than concatenating N independently-cut audio
  segments (which would risk audible seams).
- **R2.5** This feature is largely orthogonal to feature 1: per-scene strength tuning and the
  never-interpolate-across-cuts rule are useful even when the "drop non-content scenes" behavior
  is off. Should not require `analysis.scene_pipeline.enabled` — only requires scene detection to
  have run.

### Design considerations

- Per-scene clip splitting + independent stage runs + concat is a significant pipeline
  restructuring (today `Pipeline.execute_job()` runs stages over one whole-file path per job).
  Likely needs either: (a) a scene-aware wrapper that invokes today's per-file pipeline once per
  scene clip and concatenates results, or (b) deeper stage-level awareness of scene boundaries
  (e.g. interpolate stage taking a list of frame-index ranges it must not cross). (a) is simpler
  and reuses existing stage code unmodified; (b) avoids redundant encode/decode at scene
  boundaries but touches every AI-capable stage.
- Very short scenes (a few frames) may not have enough frames for meaningful stabilization
  analysis or AI interpolation — needs a minimum-scene-length floor below which a scene is left
  unprocessed or merged with a neighbor.
- Concat-induced re-encoding at scene boundaries could introduce its own quality loss if not
  done carefully (matching CRF/profile across segments, or using a lossless intermediate for the
  per-scene passes and only doing the final encode once, after concat).

---

## 3. Auto-crop mode

### Intent

Automatically detect a video's true content bounds (e.g. strip letterboxing/pillarboxing) using
FFmpeg's `cropdetect` filter as the baseline signal, with VLM assistance to distinguish genuine
content bounds from watermarks or overlays that extend beyond the actual video content (e.g. a
watermark logo positioned relative to a letterboxed frame, which naive `cropdetect` might
interpret as "content" because it's non-black pixel data outside the true picture area).
User-controllable — not forced on automatically.

### Requirements

- **R3.1** Baseline detection: `cropdetect` run with `reset=0` (accumulate the tightest safe crop
  across the whole analyzed range rather than resetting per-frame/per-GOP, which would otherwise
  flicker the detected crop window scene-to-scene). Proposed config:
  `stages.autocrop.enabled`, `stages.autocrop.reset` (default `0`), `stages.autocrop.round`
  (even-dimension rounding, matching the existing `_round_to_even` pattern in `upscale.py`).
  Sample duration/frame count for the detect pass needs a cap — running `cropdetect` over an
  entire long video is expensive; proposed a configurable sample window (e.g. first N seconds,
  or evenly-spaced samples) with a fallback to full-video if content bounds seem to vary.
- **R3.2** VLM-assisted disambiguation: when a `cropdetect` result includes an isolated
  non-black region that a naive crop would keep (e.g. a corner watermark sitting outside the
  main letterboxed frame), a VLM pass over sampled frames should be able to flag "this bright
  region in the corner is a watermark/logo, not part of the main content" and have the final crop
  bounds exclude it. This requires the VLM to reason about *spatial regions* of a frame, not just
  classify the frame as a whole — may need a prompt that asks for a bounding-box-style answer or
  at minimum a yes/no per detected crop-candidate edge.
  This is the most speculative requirement in this document; may need experimentation to find a
  workable prompt design, since the current VLM code path (`core/analysis.py`) has no coordinate/
  region-output support today.
- **R3.3** Must be user-controllable: an explicit opt-in flag/config, a way to preview the
  detected crop before committing (given crop is inherently lossy — cropped pixels are gone), and
  a manual-override path to supply exact crop dimensions instead of relying on detection.
- **R3.4** Failure mode: if `cropdetect` finds no consistent crop (e.g. content genuinely fills
  the frame, or bounds vary too much across the sampled range) auto-crop must no-op rather than
  apply a wrong/degenerate crop.

### Design considerations

- Where this fits in stage ordering matters: crop should almost certainly run before
  upscale/deblock/denoise (don't waste AI compute upscaling letterbox bars) but the exact
  position needs to respect `Pipeline.optimize_stage_order()`'s hardcoded ordering, which today
  has no crop stage at all.
- Aspect-ratio interaction with `quality.quality_target.keep_aspect_ratio` (used by
  `UpscaleStage._calculate_target_dimensions`) needs a defined interaction: does a target
  resolution apply to the pre-crop or post-crop frame? Post-crop is almost certainly the correct
  answer (the user wants the *content* at the target resolution) but this must be made explicit
  in the eventual implementation, not left implicit.
- VLM region-reasoning (R3.2) may simply not be reliable enough with current-generation
  vision-language models at low resolution/small sample counts; the design should allow
  `cropdetect`-only operation (no VLM) as a supported mode, with VLM assistance as a strict
  enhancement on top, not a hard dependency.

---

## 4. NCNN backend option

### Intent

Add ncnn/Vulkan variants of Real-ESRGAN and RIFE (`realesrgan-ncnn-vulkan`,
`rife-ncnn-vulkan`) as an alternative inference backend to the existing PyTorch/CUDA path.
Motivation: ncnn/Vulkan is often faster than CUDA in practice for these specific models, and
works out-of-the-box on AMD/Intel/integrated GPUs via Vulkan without requiring a CUDA-capable
NVIDIA card or a matching PyTorch/CUDA build (a real support burden today — see `gpu-info`'s
CPU-fallback warning text in `cli.py` about torch builds not matching a GPU's compute
capability).

### Requirements

- **R4.1** Must be in-process via Python bindings — **not** shelling out to a per-frame
  executable. The existing `rife-ncnn-vulkan`/`realesrgan-ncnn-vulkan` CLI tools process whole
  files or frame-dump directories; spawning a subprocess per frame (or even per chunk) would
  reintroduce process-startup overhead per call and lose the in-process pipelining
  (`AsyncVideoWriter`, `stream_frames_prefetched`) the PyTorch path already has. Python bindings
  (e.g. the `ncnn` Python package with a hand-written inference wrapper, or an existing
  ncnn-vulkan Python binding if one has adequate coverage) are required so a single long-lived
  process handles the whole video.
- **R4.2** Must preserve the existing chunked/streaming frame processing model — never load all
  frames of a video into memory at once. The ncnn wrapper needs to slot into the same
  `stream_frames_prefetched()` / `AsyncVideoWriter` chunked pipeline the torch wrappers use
  today (see `upscale.py`/`deblock.py`/`denoise_video.py`'s `use_chunked` branches), not a
  separate all-frames-in-memory code path.
- **R4.3** Config surface: `stages.<name>.backend: torch | ncnn` per AI-capable stage (upscale,
  interpolate, denoise_video, deblock), defaulting to `torch` (today's only backend, so this is
  backward compatible with no config changes required for existing users). A backend-unavailable
  situation (ncnn not installed, no Vulkan device) should route through the *same*
  `ai_fallback`/`_ai_fallback_or_fail` mechanism the torch path already uses — not a separate
  error-handling path — so `general.ai_fallback`/`stages.<name>.ai_fallback` continues to be the
  single place this policy is controlled regardless of backend.
- **R4.4** Model compatibility: ncnn Real-ESRGAN/RIFE releases ship their own `.param`/`.bin`
  model files, distinct from the PyTorch `.pth` checkpoints `ai/model_cache.py` currently
  downloads/verifies. The model registry/cache system needs either a parallel ncnn model
  registry or an extension of `MODEL_REGISTRY` to carry per-backend download URLs/checksums for
  the same logical model name (e.g. `RealESRGAN_x4plus` resolving to a `.pth` for `backend:
  torch` and a `.param`+`.bin` pair for `backend: ncnn`).
- **R4.5** Tiling: the torch path's OOM-driven tiling (`compute_tile_grid`,
  `AUTO_TILE_THRESHOLD_PX`, `stages.<name>.tile_size`) is CUDA-OOM-specific reactive logic:
  ncnn/Vulkan doesn't raise the same exception type and Vulkan devices have their own (often
  smaller, e.g. integrated GPU shared memory) VRAM constraints, so tiling for the ncnn backend
  needs its own trigger condition — likely proactive (based on `AUTO_TILE_THRESHOLD_PX`-style
  resolution thresholds) rather than reactive-on-exception, since a Vulkan OOM may present
  differently (driver crash/hang) rather than a catchable Python exception.

### Design considerations

- `gpu.preferred_device` (today `auto|cpu|cuda|metal` per `HWAccel`/`get_device()` semantics in
  `ai/torch_utils.py`) is a torch-specific concept; ncnn/Vulkan device selection is a separate
  axis (which Vulkan-capable GPU, if multiple) that doesn't map cleanly onto it. Needs either a
  parallel `gpu.vulkan_device` config or a documented statement that `preferred_device` only
  applies to `backend: torch` and ncnn always picks the first available Vulkan device (with a
  future config knob if that proves insufficient).
- Packaging: ncnn Python bindings are a new dependency surface distinct from the existing
  optional `ai` extra (PyTorch/OpenCV). Needs its own optional extras group (e.g. `ncnn`) so
  users who only want the torch path aren't forced to install Vulkan-related dependencies, and
  vice versa.
- Quality parity between the torch and ncnn builds of the "same" model is not guaranteed
  (different runtime, potentially different precision/quantization) — the design should not
  assume bit-identical output between backends, and any documentation of this feature once built
  should state that explicitly rather than implying the backend choice is purely a performance
  knob with no quality trade-off.
