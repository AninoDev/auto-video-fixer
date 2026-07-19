# Requirements: Planned Feature Set

Status: features 1, 2, and 3 below (scene-based processing pipeline, per-scene processing
strength, auto-crop mode) are now **IMPLEMENTED** — see `core/scenes.py` / `core/stages/crop.py`,
`AGENTS.md`'s "Scene mode" and "Auto-crop" sections, and `docs/ROADMAP.md`'s "Implemented and
verified working" section for what's built and verified vs. still open within those features
(marked inline below with **[IMPLEMENTED]** / **[UNVERIFIED]** / **[DEFERRED]** tags per
requirement). Feature 4 (NCNN backend) is also implemented — see `docs/ROADMAP.md`. This is a
requirements/design doc capturing user-specified intent, not a full description of current
behavior — for that, see `docs/ROADMAP.md` and `AGENTS.md`.

Feature 5 (Rust rewrite candidates) is a **NEW, unbuilt** planned initiative, scoped below —
prioritized soon on the roadmap per user direction (2026-07-12): the user has substantial prior
Rust experience (multi-thousand-line hand-written programs, prior production Python/Rust FFI
work) and wants performance/safety-critical hot paths considered for Rust rewrites as standard
practice going forward, not a one-off. Two other candidates considered (chunked AI frame I/O
overlap in `ai/frame_processor.py`, and the FFmpeg subprocess orchestration in `core/pipeline.py`)
were explicitly assessed and rejected for now — see the note at the end of section 5.

These four features share a common dependency: the VLM (Vision Language Model) connection layer
described in "0. Foundation" below. That layer exists in code (`core/analysis.py`, `analysis.vlm`
config block) and has now been exercised live for features 1-2 (see ROADMAP) -- features 3-4
still inherit the original unverified-foundation caveat on top of being unbuilt.

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

- **R1.1** [IMPLEMENTED] Must be strictly optional — off by default, a config flag (proposed:
  `analysis.scene_pipeline.enabled: false`) and/or a CLI flag (proposed: `--scene-pipeline`) must
  enable it. Existing whole-video processing must be unaffected when disabled.
- **R1.2** [IMPLEMENTED] Scene boundaries reuse the existing scene-detection code path (`_detect_scene_changes`
  / `detect_events`), not a new detector — avoid duplicating logic that already exists.
- **R1.3** [IMPLEMENTED] Per-scene VLM sampling: sample N frames per scene, where N scales down for short
  scenes (proposed: `min(max_sample_frames, ceil(scene_duration_sec / sample_interval_sec))`,
  reusing the existing `max_sample_frames`/`sample_interval_sec` config knobs rather than adding
  parallel ones unless a genuine need for scene-specific tuning emerges).
- **R1.4** [IMPLEMENTED, coordinator drop-path UNVERIFIED live-end-to-end -- see ROADMAP] A coordinating pass (a second LLM call, potentially text-only over the collected
  per-scene VLM descriptions rather than a second vision pass) reviews all scenes' VLM output
  together and produces: (a) a summary of the video's main content/purpose, (b) a per-scene
  keep/drop decision with a stated reason.
- **R1.5** [IMPLEMENTED] Dropped scenes are excluded from the final output; the remaining scenes are
  concatenated (see feature 2 for why concatenation, not a single continuous encode, is required
  once per-scene processing is in play).
- **R1.6** [IMPLEMENTED (WARNING-level log); dry-run/report-only preview mode DEFERRED] The keep/drop decision must be auditable — log (at INFO or above) which scenes were
  dropped and why, so a bad drop is diagnosable and not silently destructive. Consider a
  dry-run/report mode that shows the keep/drop plan without actually removing anything, given the
  destructive/lossy nature of dropping content.
- **R1.7** [IMPLEMENTED, verified live -- see ROADMAP] Failure mode: if the VLM/coordinating LLM call fails or is unavailable, the pipeline
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

- **R2.1** [PARTIALLY IMPLEMENTED -- shake magnitude and `smoothness` are tiered per scene
  (skip/normal/aggressive, see `core/scenes.py::stabilize_scene_clip`); `maxshift`/`zoom_enabled`/
  `zoom_threshold` are NOT yet varied per scene, still the stage's uniform config for every
  scene] Per-scene stabilize strength: shake magnitude already computed per-video via TRF
  analysis (`_analyze_trf_file`/`_movement_extent` in `stabilize.py`) must instead be computed
  **per scene**, and stabilize parameters (`smoothness`, `maxshift`, `zoom_enabled`/
  `zoom_threshold`) adjusted per scene rather than applied uniformly across the whole video.
- **R2.2** [IMPLEMENTED, verified -- zero blend/ghost frames at scene boundaries] Frame interpolation must treat each scene as an independent clip: run interpolation
  within a scene's frame range only, never across a detected scene-change boundary. This applies
  regardless of whether feature 1 (scene-based pipeline / content dropping) is enabled — this is
  a correctness fix for interpolation specifically, not tied to the drop-scenes feature.
- **R2.3** [IMPLEMENTED -- re-encode approach, not extract_scenes_as_clips's stream-copy] Implementation approach: split the source into per-scene clips (reusing
  `extract_scenes_as_clips`/`VideoClip` from `core/analysis.py`), process each clip
  independently through the relevant stages (at minimum stabilize and interpolate; other stages
  may also benefit from per-scene tuning, e.g. denoise strength for a grainy low-light scene vs.
  a clean daylight scene), then concatenate the processed clips back into one output (FFmpeg
  concat demuxer/filter, matching codec/timebase across segments).
- **R2.4** [IMPLEMENTED -- per-scene re-encoded audio segments, not a single continuous track] Concatenation must preserve audio sync — if scenes are processed as separate video-only
  clips and audio is normalized once over the whole original file, the concat step must
  re-attach a single continuous audio track rather than concatenating N independently-cut audio
  segments (which would risk audible seams).
- **R2.5** [IMPLEMENTED] This feature is largely orthogonal to feature 1: per-scene strength tuning and the
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

## 3. Auto-crop mode [IMPLEMENTED]

### Intent

Automatically detect a video's true content bounds (e.g. strip letterboxing/pillarboxing) using
FFmpeg's `cropdetect` filter as the baseline signal, with VLM assistance to distinguish genuine
content bounds from watermarks or overlays that extend beyond the actual video content (e.g. a
watermark logo positioned relative to a letterboxed frame, which naive `cropdetect` might
interpret as "content" because it's non-black pixel data outside the true picture area).
User-controllable — not forced on automatically.

See `core/stages/crop.py`, `Config.DEFAULTS["stages"]["crop"]`, `core/analysis.py`'s
`run_crop_vlm_check`/`_parse_crop_vlm_response`, and `AGENTS.md`'s "Auto-crop" section for the
built implementation.

### Requirements

- **R3.1** [IMPLEMENTED] Baseline detection: `cropdetect` run with `reset=0` (accumulate the
  tightest safe crop across the whole analyzed range rather than resetting per-frame/per-GOP,
  which would otherwise flicker the detected crop window scene-to-scene). Implemented as
  `stages.crop.enabled`, `stages.crop.limit` (cropdetect luma threshold, default 24),
  `stages.crop.round` (even-dimension rounding, default 2, matching the existing
  `_round_to_even` pattern in `upscale.py`) — `reset` itself is not separately configurable, it's
  always `0` (unconditional whole-scan union is the whole point of this feature; a per-frame/
  per-GOP reset would defeat it). `stages.crop.analyze_duration_sec` (0 = full video, default;
  >0 = seconds sampled from the start) caps the detect pass for long videos, with `min_crop_px`
  (default 8) as an additional no-op guard when the detected savings are negligible in both
  dimensions.
- **R3.2** [IMPLEMENTED] VLM-assisted disambiguation: rather than asking the VLM to reason about
  arbitrary spatial regions/bounding boxes (which real vision-language models don't reliably
  produce — see the rejected "expand" policy below), the implemented approach renders one frame
  two ways (plain, and the same frame with the *already-detected* crop box drawn via `drawbox`)
  and asks a narrow yes/no question: does the region OUTSIDE the drawn box contain meaningful
  content (logo/watermark/text) or only black/empty space? This sidesteps needing coordinate
  output from the VLM entirely — the crop candidate is already known from `cropdetect`; the VLM
  only has to judge one binary question about it. `stages.crop.vlm_check` (default false, requires
  `analysis.vlm.enabled: true`) opts in; `stages.crop.vlm_policy` is `"warn"` (log + crop anyway,
  default) or `"skip"` (don't crop this video). Parsed with a tolerant JSON-then-text fallback
  (`_parse_crop_vlm_response`) and fails open (proceed with the plain cropdetect result + WARNING)
  on any VLM error.
- **R3.3** [IMPLEMENTED, preview/manual-override DEFERRED] User-controllable: an explicit opt-in
  flag/config (`stages.crop.enabled`, `--enable-stage crop`) is implemented. **Not implemented**:
  a dedicated preview mode showing the detected crop before committing, and a manual-override path
  to supply exact crop dimensions instead of relying on detection — cropping is inherently lossy
  (cropped pixels are gone), so a user who wants to sanity-check first today has to run with
  `--dry-run` awareness of the feature or inspect the stage's logged `detected_crop`/
  `cropped_resolution` metadata after the fact, not before the encode happens. A future manual-crop
  override would most naturally be a separate `stages.crop.manual_crop: "w:h:x:y"` config key that
  bypasses `cropdetect` entirely when set.
- **R3.4** [IMPLEMENTED] Failure mode: if `cropdetect` finds no consistent crop (unreadable video,
  filter failure), or the detected crop's savings are below `min_crop_px` in both dimensions, or
  the detected window is degenerate (zero/negative or larger than the input), the stage returns
  `StageStatus.SKIPPED` with a specific reason rather than applying a wrong/degenerate crop —
  the original video passes through to the next stage unchanged.
- **R3.5** [IMPLEMENTED] Arbitrary-color border detection: `cropdetect` (R3.1) is luma-threshold-
  only — white, gray, or otherwise colored letterbox/pillarbox borders are invisible to it. A new
  Rust extension, `rust/avf_borders/` (`detect_border_frames()`), detects per-edge borders of ANY
  color: per sampled frame, it computes the DOMINANT color of the outer border strip (4-bit-per-
  channel quantization, largest bin's mean actual color) plus a "solidity" percentage, then walks
  inward line-by-line while a majority of each line still matches that color — a logo/overlay
  occupying a minority of a border line doesn't stop the walk, and an edge whose solidity never
  clears `stages.crop.border_solidity_min` (default `0.60`) is treated as having no solid border
  at all (protects blurred-video-background pillarboxing, a real but non-uniform edge, from being
  cropped away in v1). Its per-frame windows feed the SAME `aggregate_crop_windows()` transition-
  exclusion/union aggregation the cropdetect path uses (R3.1) — only the per-frame detection step
  differs. Selected via `stages.crop.detector` (`"auto"` default, `"rust"`, `"cropdetect"`),
  following the same lazy-import-with-fallback convention as `avf_scenes`/`avf_hashing`/
  `avf_framepipe` (R5.1-R5.3) — see AGENTS.md's "Mixed Python/Rust" and "Auto-crop" sections for
  the full algorithm, config keys, and `should_run()` prefilter interaction (the luma-only
  prefilter is skipped whenever the resolved detector isn't `"cropdetect"`, since it would
  otherwise wrongly skip a white/colored-bordered video the rust pass could actually crop).

### Design considerations

- Where this fits in stage ordering: **resolved** — `crop` runs immediately after `stabilize` and
  before every other enhancement/AI stage in `Pipeline.optimize_stage_order()`. Stabilize's
  zoom-out shake correction can itself add a black border, so cropping after it removes both the
  original letterboxing/pillarboxing and any residual stabilization border in one pass; running it
  before deblock/denoise/upscale/interpolate/encode means none of those (especially the
  AI-capable ones) spend compute on pixels about to be cropped away.
- Aspect-ratio interaction with `quality.quality_target.keep_aspect_ratio`: **resolved** as
  post-crop — since `crop` always runs before `upscale` in the fixed stage order, a configured
  target resolution is naturally applied to the already-cropped frame, matching the design
  consideration's expected answer (the user wants the *content* at the target resolution).
- VLM region-reasoning: **resolved by avoiding it** — rather than asking the VLM to output
  coordinates (R3.2's original speculative framing), the implementation only ever asks a fixed
  yes/no question against an already-computed crop candidate rendered as a visible box overlay.
  This is why the **"expand" policy is NOT implemented and was deliberately rejected**: growing
  the crop box to include just the flagged region would require the VLM to report *where* that
  region is (pixel coordinates or at least a sub-box), which current-generation VLMs don't do
  reliably — there's nothing reliable to expand *to*. Only `"warn"`/`"skip"` exist.
  `cropdetect`-only operation (no VLM) remains the default and fully supported mode;
  `stages.crop.vlm_check` is a strict, off-by-default enhancement on top, never a hard dependency.

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

## 5. Rust rewrite candidates (PLANNED, prioritized soon)

**Status: not started.** Scoped 2026-07-12 per user direction — see the note under "Status" at
the top of this document. Integration approach for all three: PyO3 + `maturin`, exposing a
narrow Python-callable function/class per target (not a wholesale module rewrite) so each stage's
existing call site changes minimally — e.g. `_detect_scene_changes()` keeps its current Python
signature and callers, but its body becomes a thin call into a compiled extension. Build via a
`maturin`-built wheel, added as a normal (non-optional — these aren't GPU/Vulkan-style optional
hardware deps, they're a portable CPU-only compiled extension) dependency once implemented, with
a pure-Python fallback path kept ONLY if compiling the extension turns out to meaningfully
complicate the install story on a target platform (decide per-target during implementation, not
speculatively now).

### R5.1 Scene-detection frame differencing [IMPLEMENTED 2026-07-13]

**Landed as**: `rust/avf_scenes/` (PyO3 + maturin crate, `uv.lock` workspace member) exposing
`avf_scenes.detect_scene_changes_rs(filepath, threshold, min_duration_sec, ffmpeg_path,
ffprobe_path, progress_callback) -> (scenes, cut_scores, near_misses)`. `_detect_scene_changes()`
in `core/analysis.py` dispatches to it when the extension imports successfully
(`_detect_scene_changes_rust`), otherwise falls back to the original pure-Python/OpenCV loop
(renamed `_detect_scene_changes_python`, kept verbatim) — both share the same cut-score/near-miss
logging in the dispatcher. Decode is a piped `ffmpeg -f rawvideo -pix_fmt gray` subprocess
(`-vf format=gray,scale=320:180:flags=bilinear`) rather than an `ffmpeg-next`/`ac-ffmpeg` binding,
per this section's stated preference — zero new runtime dependencies, consistent with the rest of
the project's FFmpeg-subprocess architecture. The Rust side releases the GIL for the whole
decode/diff loop (`Python::detach`) and only reacquires it (`Python::attach`) to fire the
progress callback, matching the "true multithreaded decode+diff, not GIL-bound" motivation below.

**Verification performed**: `TestRustPythonParity` (`tests/unit/test_scene_detection.py`) is a
direct differential test — same calibration fixture plus a synthetic pan+hard-cut motion clip,
asserting identical scene boundary counts/timestamps (0.05s tolerance) and confidence values
(0.01 tolerance) between `_detect_scene_changes_python` and `_detect_scene_changes_rust`. Measured
per-cut `diff_score` divergence: at most ~0.002 on the calibration fixture (decode/scale path
difference: ffmpeg's bilinear `scale`+`format=gray` vs. OpenCV's `INTER_LINEAR` `resize` + BT.601
`cvtColor` — both approximate the same luma transform slightly differently). Real-world speed on a
54s/1620-frame 1080p60 clip (`avf analyze`, single-shot content): Python 2.55s (~635 fps) vs. Rust
1.73s (~936 fps), ~1.5x wall-clock. `cargo clippy`/`cargo fmt --check` clean.

**Original scope** (for reference — target file/line, current-shape rationale, interface, and
verification bar as originally planned; see "Landed as" above for what actually shipped):

**Target**: `_detect_scene_changes()` in `core/analysis.py` (~line 1140) — the per-frame loop
backing `avf analyze`'s scene detection and the scene-based processing pipeline's (`core/
scenes.py`) split points. Recently recalibrated (see CHANGELOG/ROADMAP: default threshold
0.3→0.15) and now core to the flagship scene-based feature (R1/R2 above), so its performance
directly gates how usable that whole feature is on long videos.

- **Current shape**: `cv2.VideoCapture` frame-by-frame read loop; per frame, grayscale convert +
  resize to 320x180 + `cv2.absdiff` against the previous frame + `.mean()` — already delegates
  the actual pixel math to OpenCV's C++ implementation, so the walk itself (Python-level loop
  overhead, per-frame `cap.read()`/timestamp bookkeeping, the near-miss-score tracking added in
  the CLI-parity round) is the Python-side cost, not the pixel arithmetic. A Rust rewrite's win
  here is less "faster pixel math" (OpenCV is already fast) and more: eliminating per-frame
  Python/GIL overhead, and enabling true multi-threaded decode+diff (a single `cv2.VideoCapture`
  loop is inherently serial; a Rust implementation using e.g. `ffmpeg-next`/`ac-ffmpeg` bindings
  or decoding via a piped raw-frame stream could pipeline decode and diff across threads without
  fighting the GIL the way a Python `threading`-based approach would).
- **Interface**: `detect_scene_changes_rs(filepath: str, threshold: float, min_duration_sec:
  float, progress_callback: Callable | None) -> list[tuple[float, float, float]]` (start_time,
  end_time, confidence per scene) — mirrors the existing Python function's return shape
  (`SceneEvent` construction stays in Python, just fed by the Rust-computed boundary list) so the
  near-miss-score logging/CSV export work built on top of it doesn't need to change.
- **Verification bar before landing**: identical (or documented, bounded-difference) scene
  boundaries vs. the current Python implementation on the existing synthetic ground-truth fixture
  (`tests/unit/test_scene_detection.py`'s `TestSceneDetectionCalibration`) — this is a
  recently-calibrated, security/correctness-adjacent path (scene-based processing depends on it
  being right, not just fast), so a rewrite must not silently drift the detection behavior.

### R5.2 Perceptual hashing / duplicate detection [IMPLEMENTED 2026-07-13]

**Landed as**: `rust/avf_hashing/` (PyO3 + maturin crate, `[tool.uv.workspace]` member, same
build/lazy-import/fallback pattern as `avf_scenes`/R5.1) exposing `avf_hashing.compute_phash_rs
(filepath, num_frames, ffmpeg_path, ffprobe_path) -> str` and `avf_hashing.hash_similarity_rs
(hash1, hash2) -> float`. `compute_video_hash()`/`compute_video_dhash()`/`hash_similarity()` in
`core/analysis.py` are **replaced wholesale** (not kept alongside) by `compute_video_phash()`
(dispatches to the Rust extension when importable, else a pure-NumPy fallback implementing the
identical algorithm) and an updated `hash_similarity()` operating on 16-character hex pHash
strings.

**Algorithm deviation from the original scope (the significant design decision here)**: the
section below originally scoped a straight ahash/dhash *port*, evaluating the `img_hash` crate as
a pre-built implementation. Since this feature has never shipped hash values anyone depends on
(confirmed with the user 2026-07-12), the port constraint was dropped in favor of picking the best
algorithm outright: **pHash** (DCT-based perceptual hash), hand-rolled in Rust rather than via
`img_hash`. Two reasons:

1. **Accuracy**: ahash (mean-threshold) and dhash (adjacent-pixel gradient) are spatial-domain
   hashes, sensitive to exactly the pixel-level noise real near-duplicate video files have
   (re-encode artifacts, resize/crop shifts, color-grading tweaks). pHash instead thresholds the
   low-frequency 2D DCT coefficients of the downscaled frame, which encode coarse picture
   structure and are naturally robust to that kind of high-frequency noise — the standard choice
   when accuracy, not raw speed, is the goal (and speed stopped being a real constraint the moment
   this became compiled Rust code, regardless of which algorithm was picked).
2. **`img_hash` was evaluated and rejected**: it operates on `image::GenericImageView`, so using
   it would pull in the `image` crate and its per-format codec dependencies (~24 transitive crates
   as of this writing) purely to wrap raw grayscale bytes this crate already gets from its own
   ffmpeg pipe — image *decoding* isn't a need here, ffmpeg already handles it upstream. pHash
   itself is small and well-understood enough (downscale, 2D DCT, threshold the low-frequency
   block against its own mean) that hand-rolling it avoided that dependency weight for no loss of
   correctness. See `rust/avf_hashing/src/lib.rs`'s module docstring for the full writeup.

Frame sampling stays conceptually the same as before (`num_frames`, default 30, evenly spaced
across the video), and per-frame hashes still combine into one video-level hash via majority-vote
bit combination — both were kept because they're still reasonable choices with no format to
preserve, not because of a compatibility requirement.

**Config**: `analysis.duplicate_detection.hash_type` (`perceptual`/`dhash`/`combined` in
`config.py`/`docs/config.example.yaml`) is **removed** — it was never read by any call site
(`find_similar()`/`find_duplicates()`/the CLI only ever used `similarity_threshold`), and doesn't
map onto a single-algorithm design. `similarity_threshold`'s default changed from `0.95` to
`0.85`, recalibrated against real measured pHash similarity scores (see "Verification performed"
below) rather than carried over from the old ahash/dhash-tuned default.

**Verification performed**: `tests/unit/test_hashing.py` — real ffmpeg-generated fixtures, not
just hash-math unit tests. `TestHashSeparation` builds genuine near-duplicates (one `testsrc2`
source re-encoded at CRF 18 vs. CRF 32, at 320x240 vs. 160x120, and trimmed ~0.3s off the start)
and genuine non-duplicates (four distinct lavfi sources: `testsrc2`, `smptebars`, solid red,
`mandelbrot`), asserting the near-duplicate pairs score >= the 0.85 default threshold and the
non-duplicate pairs (including cross-comparing each source's own near-duplicate variants against
each other, not just against their own base) score below it. Measured similarity scores: near-
duplicate pairs ranged 0.9375-1.0; non-duplicate pairs ranged 0.3438-0.5781 — a wide margin on
both sides of 0.85. `TestRustPythonHashParity` confirms the Rust extension and pure-Python
fallback agree closely (>= 0.85 similarity hashing the same video) despite decoding frames via
different paths (ffmpeg `select` filter vs. `cv2.VideoCapture` seeking) that land on slightly
different sampled frames — full bit-for-bit equality isn't the right bar here, unlike R5.1's
frame-differencing parity test, since neither decode path is "the reference" the other must match
exactly. Speed (secondary to accuracy per this section's original verification bar): comparable
between the Rust and Python paths for small (a few seconds, num_frames=30) test clips — both are
dominated by ffmpeg/OpenCV subprocess and decode overhead at this scale rather than the hash math
itself, which is cheap either way once frames are in hand.

**Original scope** (for reference — target file/line and originally-planned interface/verification
bar; see "Landed as" and the algorithm-deviation writeup above for what actually shipped and why it
differs):

**Target**: `compute_video_hash()` (ahash, ~line 1843) and `compute_video_dhash()` (dhash, ~line
1900) in `core/analysis.py`, backing `avf find-duplicates`. Both build their hash bit-string with
a Python-level `str.join()` over a per-pixel/per-comparison generator expression (ahash:
256 pixel comparisons per sampled frame; dhash: width×(width+1-1) adjacent-pixel comparisons per
frame), then do majority-vote bit combination across all sampled frames with another nested
Python loop over every bit position × every frame's hash. This is a textbook case of Python loop
overhead dominating trivial per-element work — genuinely embarrassingly parallel and a strong
Rust target.

**No backward-compatibility constraint (confirmed by user 2026-07-12): this feature has never
been used in production by anyone, including the user — no hash values exist anywhere that need
to keep matching.** This means the Rust rewrite is NOT bound to reproducing ahash/dhash bit-for-
bit; pick whatever algorithm is actually best (fastest, most memory-efficient, most accurate,
most flexible) rather than porting the existing one as-is. Concretely, worth evaluating instead
of a literal ahash/dhash port:

- **pHash (DCT-based perceptual hash)**: substantially more robust than ahash/dhash to the kind
  of near-duplicate variation video files actually have (re-encodes, minor crops, compression
  artifact differences, slight color grading) — the standard choice when accuracy matters more
  than raw speed, and speed is not actually a concern once this is in Rust regardless of which
  algorithm is chosen.
- Existing, well-vetted Rust crates rather than a hand-rolled implementation where reasonable —
  e.g. the `img_hash` crate implements aHash/dHash/pHash/blockhash with a shared, already-
  optimized `HasherConfig` API and its own Hamming-distance comparison; evaluate whether it (or
  an equivalent maintained crate) covers this need before writing hashing logic from scratch.
  Frame extraction/sampling (decode + resize, likely via `ffmpeg-next` or piping raw frames)
  still needs its own implementation either way.
- A combined/ensemble hash (e.g. running both ahash and pHash and requiring agreement, or a
  single stronger hash) is worth considering given there's no format to preserve — pick whatever
  gives the best true-positive/false-positive tradeoff for near-duplicate video detection, not
  necessarily whatever the current Python code happens to do.

- **Interface**: still a Python-callable function per algorithm chosen (e.g.
  `compute_phash_rs(filepath: str, num_frames: int) -> str` or whatever shape the chosen
  algorithm naturally produces) plus a matching similarity/distance function — exact signature
  follows from the algorithm decision above, not fixed in advance the way R5.1's is.
- **Verification bar**: since there's no legacy output to match, correctness verification should
  instead be behavioral — construct a small test set of genuine near-duplicates (e.g. the same
  clip re-encoded at a different bitrate/resolution, or with a few seconds trimmed) and genuine
  non-duplicates, and confirm the chosen algorithm actually separates them with a sensible
  threshold. Don't just benchmark speed; confirm the accuracy case this feature exists for.

### R5.3 Chunked AI frame I/O overlap

> **2026-07-13 reprioritization note**: live-monitoring a real production job (RTX 5060 Ti,
> `nvidia-smi --query-gpu=utilization.gpu` sampled every second for 40s during `deblock`) found
> GPU utilization pinned near 100% almost continuously, meaning there's little idle GPU time left
> for R5.3's I/O-overlap approach to reclaim for *that* stage. Separately, the actual real-world
> job that prompted this investigation was a single 4K (3840x2160) input stuck in `deblock` for
> hours -- at that resolution every frame takes the tiled-inference path (`run_tiled_inference`,
> `ai/wrappers/upscale.py`), which at the default `tile_size=512` needs a 5x8=40-tile grid
> processed one tile at a time: 40 small sequential forward passes per frame, each paying Python/
> kernel-launch/host-device-sync overhead. That -- not I/O overlap -- is almost certainly the
> dominant real-world cost for large-resolution jobs, and was closed by adding
> `tile_batch_size` (batches same-shaped tiles from `compute_tile_grid` into fewer, larger forward
> calls; see `RealESRGANUpscaler`/`run_tiled_inference` in `ai/wrappers/upscale.py` and the
> `stages.{upscale,deblock,denoise_video}.tile_batch_size` config key) plus a whole-*frame*
> `batch_size` for videos small enough to skip tiling entirely (`RealESRGANUpscaler.upscale_batch`/
> `upscale_video`). R5.3 is not superseded -- 100% GPU utilization with only ~70-80% SM occupancy
> (per `nvtop`) is still consistent with I/O/CPU-side stalls between kernel launches on top of the
> now-fixed small-batch-launch overhead -- but it's de-prioritized below the tile/frame batching
> fix above, which was the higher-leverage, lower-complexity win for the actual reported workload.
>
> **2026-07-13 correction -- the "real-4K benchmark result" below was invalid; retracted**: the
> A/B described in the previous version of this note (tile_batch_size 1/4/8 against a real
> 3840x2160/150-frame clip through `deblock --ai`) was run under `timeout 400` (6m40s). All three
> runs were killed by that timeout at exactly 6m40.3-6m40.4s -- none of them reached a completion
> marker, and all three output directories were empty. The "virtually identical wall-clock time"
> finding was therefore an artifact of the runs being cut off at the same external time limit, not
> a measurement of anything about tile batching's effect on throughput. **The throughput
> conclusions in that note (tile/frame batching "did not measurably help" a real 4K workload) are
> void and must not be relied on.** A corrected, untruncated 4K measurement is tracked as Phase 0
> item 3 of the R5.3 quick-wins-then-Rust plan; do not re-cite the numbers below as evidence until
> that rerun lands.
>
> What remains valid from those three (truncated) runs, since it doesn't depend on how long they
> ran: (a) `sys` time was a high share (~3m50-3m55s out of the ~6m40s each run reached) across all
> three -- worth investigating, but the previous note's attribution of that share to raw-frame
> I/O/marshalling overhead was speculation, not measurement; CUDA's blocking-sync driver ioctls
> also show up as `sys` time and haven't been ruled out as a contributor. (b) The batched runs
> (`tile_batch_size` 4 and 8) both hit CUDA OOM cascades roughly 4 minutes in, while the unbatched
> baseline (`tile_batch_size=1`) ran clean up to the point it too was killed at 6m40s. That
> asymmetry is real and is consistent with (not proof of, but a useful lead toward) the OOM-retry
> tensor-lifetime defect fixed in Phase 1.2 of the R5.3 plan (`_infer_group` keeping the failed
> batch tensor alive across the halving retry, and remainder-batch shape fragmentation).
>
> R5.3 (the Rust `avf_framepipe` transport rewrite) is not decided by this note either way --
> its case rests on the architecture gap (Python `threading`/GIL vs. real OS threads) documented
> above, not on the retracted A/B. A valid conclusion about whether transport or compute dominates
> the real 4K budget requires the corrected, untruncated measurement.

> **2026-07-14 valid post-Phase-1 measurement (the corrected rerun)**: 150-frame synthetic
> 3840x2160 clip (testsrc2+noise, CRF 40 for real blocking) through `deblock --ai`,
> `tile_batch_size=1`, RTX 5060 Ti, NO timeout, exit 0, stage completed, output verified.
> Per-phase budget from the Phase-0 instrumentation: **gpu_forward=592.4s (~100%)**,
> decode_wait=0.2s, h2d_preprocess=0.5s, d2h_postprocess=1.2s, write_wait=0.0s;
> avg 0.25 fps (matches real-world 4K deblock observations). Small-res contrast
> (576x320, same clip recipe): avg 13.87 fps, gpu_forward=10.1s (95%), h2d 4%, d2h 1%.
> Conclusion: post-Phase-1, deblock is compute-bound at every tested resolution — transport
> overhead is ~0% at 4K and ~5% at 576x320. The >85% scope gate triggered: `avf_framepipe`
> lands in lean v1 form only (no NVDEC, no buffer-lease pooling), and its expected speed
> benefit for deblock is small; its case is architecture robustness (real OS threads, bounded
> memory, diagnosable ffmpeg errors) and headroom for lighter/faster AI passes.
>
> **2026-07-14 final A/B (Python transport vs Rust `avf_framepipe`, same clips/settings/machine,
> both runs exit 0 + completed)**: 4K deblock — 0.25 fps on both transports (identical, as the
> ~100% gpu_forward budget predicted). 576x320 deblock — 13.87 fps (Python) → 14.41 fps (Rust),
> **+3.9%**, gpu_forward share 95% → 98%. Transport is now effectively invisible in the phase
> budget at all tested resolutions. Output parity proven separately by the bit-exact
> differential identity test (`tests/unit/test_frame_pipe.py`, integration-marked); bounded
> memory by the 1200-frame soak test (~6.8MB RSS growth).
>
> **2026-07-14 compact-model follow-up (the change that actually moves 4K throughput)**: since
> the budget is ~100% gpu_forward, the fix is a lighter model, not a faster runtime — the
> ncnn-wrapper idea (#24) stays rejected on this hardware (repo's own A/B: ncnn/Vulkan 5.1 fps
> vs torch/CUDA 7.33 fps at 320x240, OOM at 720p). With the new SRVGG compact models
> (`ai_model: realesr-general-x4v3`), same 4K/small clips, same discipline (exit 0 + completed):
> 4K deblock 0.25 → **0.81 fps (3.2x)**; 576x320 deblock 14.4 → **41.7 fps (2.9x)**, where
> transport (h2d 15%) is now visible — the avf_framepipe work pays off exactly there. Quality
> vs the RRDB reference output: SSIM 0.973 (4K) / 0.911 (small), healthy signalstats. RRDB
> remains the default; compact is opt-in per stage via `stages.<name>.ai_model`.

> **Status: implemented.** `rust/avf_framepipe/` (lean v1 scope per the gate above -- no NVDEC,
> no buffer-lease pooling) landed as a `FrameReader`/`FrameWriter` PyO3 crate, and `ai/frame_pipe.py`
> wires it into `upscale`/`deblock`/`denoise_video`'s chunked stage loops via `get_frame_reader()`/
> `get_frame_writer()` factories, with a fallback to the pre-existing `ai/frame_processor.py`
> machinery when the extension isn't built (see AGENTS.md's "Mixed Python/Rust" section for the
> adapter contract and fallback-parity gaps). Verification: a differential identity test
> (`tests/unit/test_frame_pipe.py::TestTransportIdentity`) proves the Rust and Python transports
> are bit-exact-interchangeable under identical passthrough processing and encode settings; a
> soak test (`TestFramePipeSoak`, 1200 frames) confirms bounded (not full-video) memory growth
> streaming through the reader/writer pair. `interpolate`'s AI/RIFE path was explicitly left on
> `frame_processor.py` directly (still buffers frames in a list, not the streaming path) --
> out of scope per the "Scope carefully" note below, unchanged by this work.

**Target**: `PrefetchIterator` and `AsyncVideoWriter` in `ai/frame_processor.py` (`stream_frames_prefetched`,
~line 247), the background-thread decode-prefetch / async-write machinery added to keep the GPU
fed during AI stage processing (upscale/interpolate/denoise/deblock). Currently Python
`threading`-based, which works for I/O-bound prefetch (decode subprocess I/O releases the GIL)
but is fighting an uphill battle now that GPU inference is fast (7-92 fps depending on
backend/model per the NCNN benchmarking round) — CPU-side frame marshalling (color conversion,
tensor packing/unpacking, queue handoff) increasingly matters at those speeds, and Python
threading's GIL means CPU-bound portions of that pipeline don't actually run concurrently no
matter how many threads are spun up.

- **Scope carefully**: NOT a rewrite of the whole AI stage — just the producer/consumer queue +
  prefetch/writer thread pair, with the actual model inference call staying exactly where it is
  (torch/ncnn, Python-orchestrated either way). A true multi-stage pipeline (decode thread → CPU
  preprocess thread → GPU inference → CPU postprocess thread → write thread) implemented in Rust
  with real OS threads (no GIL) around a Python-callable inference step (via a callback or
  channel) is the target shape — this is the most architecturally involved of the three
  candidates and should be scoped as its own design pass before implementation starts, not
  assumed to be a drop-in swap the way R5.1/R5.2 are.
- **Verification bar**: measured end-to-end stage throughput (fps) improvement on the existing
  benchmark methodology used throughout this project's AI-stage work (real ffmpeg-generated test
  clips, signalstats non-black checks, before/after fps numbers) — this candidate's whole
  justification is a performance number, so it needs one, not just "should be faster in theory."

**Evidence backing this scope (2026-07-12, real hardware — RTX 5060 Ti)**: `frame_from_tensor`/
`tensor_from_frame` in `ai/torch_utils.py` were rewritten first (GPU-side elementwise
post/pre-processing instead of CPU-side numpy, before/after the H2D/D2H transfer respectively —
bit-exact/ULP-level output verified, see the module's docstrings and `tests/unit/
test_torch_utils.py`). Isolated timing: `frame_from_tensor` at a real 2304x1280 (4x-upscaled)
output size went from 41.4ms to 2.9ms/call (~14x). Real end-to-end impact, both measured via an
old-vs-new A/B at identical config (git-stashing just the fix, nothing else varying):
- `upscale` stage, 1080p60 preset, 576x320→1920x1066 input: 300 frames, 121.4s → 108.1s
  (**~11% faster**).
- `denoise_video` stage (scale=1, no spatial upscaling, native 576x320 output — the "ESRGAN
  stage" the user originally reported as CPU-bottlenecked at 14-15fps): 300 frames, 23.2s → 22.3s
  (**~4% faster** with both `frame_from_tensor` and `tensor_from_frame` fixed) — a real but much
  smaller win than `upscale`, because the postprocessing array is 16x smaller at native
  resolution than at a 4x-upscaled output, so the *absolute* CPU time saved per frame is smaller
  even though the isolated per-call speedup is the same ~14x.

Five `py-spy dump` samples against the live (fixed-code) `denoise_video` process (sudo, real PID,
zero-overhead sampling) landed: 2/5 in genuine GPU compute (`torch/nn/modules/conv.py`'s
`_conv_forward`, i.e. the RRDBNet convolutions actually running), 2/5 in `frame_from_tensor`,
1/5 in `tensor_from_frame` — with **both helper threads idle in every single sample** (the
prefetch thread blocked on a full queue's `put()`, meaning it decoded ahead and is waiting for
the main thread to consume; the writer thread blocked on an empty queue's `get()`, meaning it has
nothing new to write). Given the isolated timing already proved `frame_from_tensor`'s own CPU
math is now ~2.9ms, landing samples there is most plausibly catching the mandatory GPU-sync wait
at `.cpu()` (blocking until all previously-queued CUDA kernels, including the model's own
convolutions, finish) rather than leftover CPU-bound work — i.e. the CPU thread is idle *waiting
for GPU compute*, with nothing scheduled to fill that wait. This is the concrete confirmation
R5.3 targets the right problem: the existing prefetch/writer threads can't help because the main
loop never hands them frame N+1's decode or frame N-1's write to do *while* frame N sits on the
GPU — it processes one frame fully serially (decode→H2D→forward→sync→postprocess→write) before
starting the next, so there's structurally nothing for the helper threads to overlap with. This
is exactly the "true multi-stage pipeline" shape scoped above, not a numpy/torch-op-level fix —
the two op-level fixes above are already merged and are a distinct, smaller, already-realized
win.

### Rejected candidates (revisit only if a new non-performance reason emerges)

Two other pieces of the codebase were considered during scoping and explicitly set aside — not
because Rust couldn't do them, but because the payoff doesn't currently justify the added build
complexity (a Rust toolchain requirement, cross-platform wheel building via `maturin`, and a
second language in the codebase) for these specific pieces:

- **FFmpeg subprocess orchestration** (`core/pipeline.py`, the stage classes' `subprocess`/`Popen`
  calls) — this is I/O-bound glue code shelling out to `ffmpeg`/`ffprobe`; the actual work
  happens inside the FFmpeg process, not in the Python orchestrating it. A Rust rewrite here
  would move where the `Popen` calls live without changing what dominates wall-clock time.
- **The AI model inference calls themselves** — these delegate to PyTorch or ncnn either way;
  there's no "the Python is slow" problem here to solve, since the compute already happens in
  optimized C++/CUDA/Vulkan code outside the Python interpreter.

If either of these ever needs a rewrite for a reason OTHER than raw speed (e.g. a memory-safety
bug in the subprocess lifecycle management, or a need for true OS-level concurrency that Python's
`subprocess` module can't provide), revisit — but "rewrite it in Rust" alone isn't sufficient
justification for these two given the current design.

## 6. Output handling, run reporting, and config tooling (PLANNED 2026-07-18, user-approved)

User-approved requirement set from the 2026-07-18 session (full detail preserved here because
the approving conversation is closed; treat this section as the authoritative spec). Agreed
delivery order — four commits:

1. **Commit ①**: 6.1 + 6.2 + 6.3 (one decision path in `Pipeline.execute_job()`).
2. **Commit ②**: 6.4 + 6.5 + 6.6 (one reporting/data-model effort — summary, timing, and JSON
   all read the same new provenance fields).
3. **Commit ③**: 6.7 (log infrastructure, independent).
4. **Commit ④**: 6.8 (config tooling, independent).

Standing process for all four: Sonnet subagent implements from a written spec; coordinator
independently verifies (gates: `uv run pytest tests/unit -q` — baseline 720 passing;
`uv run ruff check .`; `uv run ruff format --check .`;
`uv run mypy src --ignore-missing-imports` <= 172 errors; cargo gates only if Rust touched),
plus a live end-to-end test per commit; docs synced every commit (AGENTS.md, CHANGELOG.md,
docs/config.example.yaml); user signs/pushes each commit (agents NEVER run git
stash/checkout/restore/reset/add/rm/mv/commit).

### 6.1 Existing output with overwrite disabled = SKIPPED, not FAILED (IMPLEMENTED 2026-07-18)

Today `execute_job()` returns `success=False` + ERROR log ("Output already exists and
general.overwrite is False") — the user found "failed" deeply confusing for videos that were
simply already done from a previous run.

- New first-class job outcome **SKIPPED**, distinct from COMPLETED/FAILED: carried on
  `JobResult` (with a machine-readable skip sub-reason), counted separately in the end-of-run
  summary, logged at INFO not ERROR.
- `general.existing_output: "skip" | "fail"` — default **"skip"**; `"fail"` restores the old
  classification exactly.
- Exit code: a run whose jobs are all completed-or-skipped exits 0; only true failures make
  the run exit non-zero.

### 6.2 Spec-check existing outputs; rename-or-overwrite mismatches (IMPLEMENTED 2026-07-18)

When an output exists and overwrite is false, optionally verify the existing file actually
satisfies the CURRENT effective targets before deciding to skip.

- `general.check_existing_target: true` — **default ON** (user decision: ffprobe is cheap and
  the info is valuable). ffprobe the existing output; compare against effective targets.
- **v1 spec set**: container (general.target_format), resolution, framerate, video codec,
  audio codec. Codecs compared only against what the user's encoding settings actually
  specify; bitrate deliberately excluded (nothing targets it). Extensible.
- **Resolution comparison must be orientation-aware and tolerant**: reuse the same semantics
  as `UpscaleStage._effective_target_bounds()` + the 1.05 `_SKIP_SCALE_THRESHOLD` — a
  1080x1918 output SATISFIES a [1920,1080] target (rotated bounds; exact-aspect fits
  legitimately land a few px short). Framerate satisfied within a small epsilon
  (29.97 ~= 30). An unreadable/corrupt existing output counts as a MISMATCH.
- Check on + reprocessing off => mismatch logged loudly, job still SKIPPED (classified per
  6.1, with a distinct "exists-but-mismatched" sub-reason).
- `general.reprocess_mismatched: false` — default OFF; when true, a mismatch triggers
  reprocessing of that one video.
- `general.existing_mismatched: "rename" | "overwrite"` — default **"rename"** — what happens
  to the old mismatched file when reprocessing:
  - rename: the existing file is renamed (NEVER overwritten) to `<stem><suffix><N><ext>`,
    N starting at 1 and incrementing until an unused name is found; the new output is then
    written at the original name.
  - overwrite: replace directly.
- `general.mismatched_rename_suffix: "_mismatched-"` — user-overridable; the stem+suffix+N+ext
  pattern and the incrementing-N contract must hold for any custom suffix.
- `general.mismatched_max_renames: null` — cap on N. null = unlimited (default). Exceeding
  the cap => job FAILED with an explicit reason. **0 = renaming fully disabled**: a mismatch
  in rename mode is then FAILED (documented as the intentional "never silently rename or
  overwrite" strict posture — user explicitly accepted FAIL semantics for this combination;
  it is deliberately allowed, not rejected at config-validation time).
- **Logging contract**: every decision states what the existing file measured vs. what was
  targeted, which specs mismatched, which flags drove the outcome, and — on rename — the
  original name and the exact new name. Example shape: "output existed but mismatched
  (framerate 30<60, vcodec h264!=libx265); renamed to video_enhanced_mismatched-2.mp4;
  reprocessing".

### 6.3 Input probe failure policy (IMPLEMENTED 2026-07-18)

- **Audit current behavior first** (the user does not know what AVF does today), then
  enforce: an input ffprobe CANNOT analyze fails that job (FAILED, ffprobe stderr surfaced in
  log and summary); ffprobe WARNINGS never fail a job by default and are always surfaced to
  the user. Applies uniformly to every input video.
- `general.skip_invalid_inputs: false` — opt-in scavenge mode: unanalyzable inputs become
  SKIPPED (sub-reason: invalid/unreadable input) instead of FAILED so a batch with
  known-corrupt members completes; still listed distinctly in the summary (the failure is
  noted either way).
- `general.fail_on_probe_warnings: false` — opt-in strict mode; explicitly non-default.

### 6.4 Per-video stage/mode summary (IMPLEMENTED 2026-07-19)

- Per job and aggregated at end of run, each stage classified as: **ran+AI**,
  **ran+traditional (chosen)**, **ran+traditional (fallback — the AI attempt failed first;
  MUST be distinguished from chosen-traditional)**, **failed**, or **skipped (with reason)**.
- Requires uniform provenance in `StageResult.metadata`: most stages already record
  `method`; a fallback marker must be added at the `BaseStage._ai_fallback_or_fail()` seam
  (e.g. `metadata["ai_fallback_used"] = True` + the failure cause) so reporting can tell
  fallback-traditional from chosen-traditional.
- Per input video at a glance: **completed** (output created) / **failed** (no output) /
  **skipped** with sub-reason (already-exists; exists-but-mismatched-not-reprocessed;
  invalid input). Reprocessed-mismatch jobs are ordinary completed jobs noted as reprocessed.
- Scene-mode stats per video when it ran: scenes detected / kept / dropped.
- Rendered as a Rich table on the console, plain lines in the log file, and stored on
  `JobResult` for future GUI use.

### 6.5 Media info printouts + timing instrumentation (IMPLEMENTED 2026-07-19)

- **Media info**: at each job's start log the input's resolution, framerate, duration,
  filesize, bitrate, video/audio codecs; at completion the same for the output
  (side-by-side comparison). INFO level.
- **Per-stage timing**: after each stage completes/fails/skips, log its wall-clock duration
  (`StageResult.duration_sec` already exists — surface it).
- **Per-video timing in the end summary, millisecond precision, two DISTINCT numbers**:
  - job total wall time: from the moment the video's turn starts (INCLUDING the 6.1/6.2
    exists/spec-check decision phase and input probing) to the moment the job result is
    finalized;
  - processing time: the stage-pipeline portion only (zero for skipped videos — the reason
    the two must be separate).
  - Whole-run elapsed shown once at the bottom.
- **Stage-timing summary flags** (config keys + CLI flags, all off by default; three views):
  per-video per-stage duration table; per-stage totals summed across the run; per-stage
  average per video. **The average's divisor is the number of videos that actually RAN that
  stage** — never total video count (a video that failed at stage 3 contributes nothing to
  stages 4+; a skipped stage contributes nothing anywhere). **Failed stage executions are
  excluded from the success-timing stats and shown in their own separate section** (a stage
  that died in 2s must not drag the average down). Purpose: identify which stages dominate
  runtime and correlate video characteristics with stage cost.

### 6.6 Structured JSON run report (IMPLEMENTED 2026-07-19)

- Optional: `--report-json PATH` CLI flag + config key. One JSON document per run:
  run metadata (avf version, effective non-default settings, timestamps), per-job records
  (input/output media info, outcome + sub-reason, the full 6.1/6.2 decision trail, scene
  stats, job total/processing ms), per-stage records within each job (status, method,
  fallback provenance, duration ms, skip reason, error).
- **Non-redundancy is a hard design requirement**: each fact lives in exactly ONE canonical
  place. Aggregates (per-stage totals/averages) are NOT stored — they are derivable from the
  per-job stage records, and storing both invites selecting the wrong attribute during
  analysis (the user's stated purpose is offline data analysis to guide performance work).
- Machine-oriented: stable key names, numbers as numbers, no Rich formatting artifacts.
- The JSON always contains the raw per-stage data regardless of the 6.5 display flags.

### 6.7 PII-clean log variant

- `general.log_type: "raw" | "clean" | "both" | "none"` (config + CLI flag; default "raw" =
  today's behavior). Applies to FILE logging; the console stays raw.
- **Clean mode** substitutes private values via a logging filter holding a per-run
  consistent mapping (same real value -> same placeholder everywhere, numbered by first
  appearance, so cross-referencing within the log still works for diagnosis):
  - input filenames -> `input_video_01.mkv` (original extension preserved; 2-digit numbering
    by first appearance);
  - output filenames -> `output_video_01.mp4`;
  - directories -> role-based placeholders, CONSISTENT naming: `/path/to/input/`,
    `/path/to/output/`, `/path/to/config/` (note: "input" not "input_videos" — user
    explicitly corrected this for consistency with `/path/to/output/`);
  - network endpoints (VLM/LLM api_url hosts/IPs) -> e.g. `http://vlm-endpoint/...`;
  - embedded video titles/metadata values when they appear in log text;
  - stage temp filenames (`/tmp/avf_*`) left untouched (no PII by construction);
  - secrets already covered by the existing `redact_secrets` convention.
- Mapping is derived from KNOWN values (actual input/output/config paths, configured
  endpoints) — substitution, not guesswork regexes; that is what makes it reliable enough
  for users to publish logs when asking for help. Free-text PII inside e.g. DEBUG-logged VLM
  responses cannot be reliably caught and is documented as out of scope for v1.
- **both mode**: two files. Suffix insertion into a custom `--log-file` name: before the
  extension if one exists, appended to the end otherwise (extensionless Unix names).
  `general.log_suffix_raw` (default `""`) and `general.log_suffix_clean` (default
  `"-clean"`); empty string = no suffix. If `both` would produce two identical paths (both
  suffixes empty), that is a config error at startup — never a silent overwrite.

### 6.8 Config tooling: `avf config clean | upgrade | dump`

Three subcommands under a new `avf config` group (chosen over a standalone script for CLI
consistency; user approved).

- **clean**: input YAML -> normalized bare YAML: comments stripped, canonical formatting,
  containing ONLY the keys actually present in the input (no defaults merged in).
- **dump**: emit the EFFECTIVE config — defaults + config file(s) + preset(s) + --set/flags,
  the full cascade — as clean YAML. Accepts the same --config/--preset/--set layering as
  `process` so the user can dump exactly what a given invocation would run with. Secrets
  (api_key etc.) redacted to "***" by default; explicit `--with-secrets` writes real values.
- **upgrade**: apply an input config onto a template — default `docs/config.example.yaml`
  (it is updated with every AVF release), optional explicit template path. Output = the
  template's FULL text (comments, ordering, structure, new options at their documented
  defaults) with each LEAF value present in the input replaced in place at the same tree
  path. Strictly leaf-by-leaf, never subtree replacement — EXCEPT lists are atomic leaves
  (e.g. `pipeline.default_order`: order and presence matter; replace wholesale). Requires
  round-trip comment-preserving YAML editing of the template: use **ruamel.yaml** (one new
  dependency) — plain PyYAML cannot preserve comments. Purpose: upgrading an existing user
  config to a new AVF version's template so new options and improved comments become
  visible while user settings are kept.
- **Mismatch safety**: input keys absent from the template (renamed/removed/moved options)
  are listed explicitly and the tool REFUSES to write unless `--drop-unknown` is passed,
  which omits them with a per-key warning. (Semantic changes hiding behind unchanged key
  names are undetectable; documented limitation.)
- **File safety** (all three subcommands): never overwrite an existing destination file by
  default. `--force` overwrites; `--backup` renames the existing file to the first unused
  incremental name (`config.yaml.1`, `config.yaml.2`, ...), writes the new file, and prints
  exactly what was renamed to what.

### Design considerations / non-obvious implementation notes

- 6.1/6.2's decision phase happens in `Pipeline.execute_job()` where the current
  "Output already exists" ERROR-and-fail block sits (after scene mode, before stage loop as
  of 2026-07-18 — search for `general.overwrite`). The per-stage input_info re-probe and the
  plan-membership fix (see CHANGELOG 2026-07-18) already landed; do not regress them.
- A SKIPPED job must not run ANY stage, must still produce a complete JobResult (for 6.4/6.6
  reporting), and its processing time is 0 while its total wall time is real.
- 6.2's spec comparison should live in a small dedicated helper (pure, unit-testable) taking
  (existing_probe, effective_targets) -> (matches, mismatch_reasons list) — the logging
  contract needs the reasons list anyway.
- 6.4's fallback provenance: `_ai_fallback_or_fail()` in `core/stages/base.py` is the ONLY
  seam through which AI->traditional fallback flows; mark metadata there once rather than in
  each stage.
- GPU note for agents: check `nvidia-smi --query-compute-apps` before any GPU-touching live
  test; skip GPU tests if the user's jobs are running.
- Rich console output is line-wrapped and unfriendly to grep — verification workflows should
  use `--log-file` and grep the plain file (established practice).
- Config keys should all be added to `DEFAULTS` in `config.py` AND documented in
  docs/config.example.yaml with rationale comments, per the standing docs-sync rule.
