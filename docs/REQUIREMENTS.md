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

### 6.7 PII-clean log variant (IMPLEMENTED 2026-07-19)

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

### 6.8 Config tooling: `avf config clean | upgrade | dump` (IMPLEMENTED 2026-07-20)

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

## 7. Resolution downscaling & fit modes [IMPLEMENTED 2026-07-21]

**Feature 1 — `downscale` stage**: a new pure-FFmpeg (no AI), opt-in stage
(`stages.downscale.enabled`, default `false`) that shrinks an OVERSIZED input down to the
configured target resolution box. Sits in `pipeline.default_order` immediately after `crop`
and before the heavier `denoise_video`/`upscale`/`interpolate` stages, so they never spend
compute on pixels a downscale would shrink away anyway. `should_run()` never fires on an input
that needs upscaling instead (guarded by the same `SKIP_SCALE_THRESHOLD` tolerance the upscale
stage uses) — `downscale` and `upscale` are complementary: an oversized input gets shrunk by
`downscale` and then skipped by `upscale` ("already at target"); a small input is skipped by
`downscale` and (if enabled) upscaled as usual.

**Feature 2 — shared dimension-fitting helper + two fit modes**: `core/output_check.py:
compute_fitted_dimensions()` is the one implementation of "fit an input into an
orientation-aware target box" shared by `UpscaleStage` and `DownscaleStage`. Two
`quality.quality_target.resolution_fit_mode` values:
  - `preserve_aspect` (default): fits the input's exact aspect ratio within the target box,
    rounds each dimension UP to `dimension_multiple`. Reproduces the pre-existing upscale
    behavior exactly (byte-identical output dims) when `dimension_multiple == 2`.
  - `snap_limiting` ("snap-to-box-when-close"): the LIMITING axis (the one binding the
    preserve_aspect min-fit scale) always lands exactly onto its target bound. The OTHER
    (derived) axis's exact aspect-preserving float value falls short of its own bound by some
    gap fraction; if that gap is `<= snap_tolerance` (default 0.01 = 1%), the derived axis is
    ALSO snapped exactly onto its bound -- both dimensions land on the full target box (e.g. a
    1440x812 input against `[1920, 1080]` snaps to exactly 1920x1080 instead of ~1920x1078).
    Beyond `snap_tolerance` (a genuinely different aspect ratio), the derived axis is left at
    its aspect-preserving value, rounded to the NEAREST `dimension_multiple` (e.g. a 3840x2106
    input -- a 2.5% gap -- stays 1920x1052, not snapped to 1920x1080).

`quality.quality_target.dimension_multiple` (default 2, required by H.264/yuv420p) is the
rounding granularity for both fit modes and both stages.
`quality.quality_target.snap_tolerance` (default 0.01) only applies to `snap_limiting`.

**Feature 4 — upscale routes through the shared helper**: `UpscaleStage._calculate_target_
dimensions()` now delegates to `compute_fitted_dimensions()` instead of its own inline
scale+round logic, picking up both fit modes, `dimension_multiple`, and `snap_tolerance` for
free, and fixing the same rounding artifact in upscale's own output when
`resolution_fit_mode: snap_limiting` is set. `should_run()`'s "already at target" skip logic is
unchanged.

CLI: `--downscale`/`--no-downscale` (`stages.downscale.enabled`), `--resolution-fit-mode
{preserve_aspect,snap_limiting}` (`quality.quality_target.resolution_fit_mode`),
`--dimension-multiple INT` (`quality.quality_target.dimension_multiple`), `--snap-tolerance
FLOAT` (`quality.quality_target.snap_tolerance`).

## 8. Pipeline default tuning (crop-first, deblock via denoise-optimized model, denoise off by default) [IMPLEMENTED 2026-07-22]

Two independent default-behavior changes, both motivated by feature 7's `downscale` stage and
by `stabilize`'s now-borderless zoom (percentile-based, not the old motion-guess zoom):

**Change 1 — `crop` moves to the front of `pipeline.default_order`**: previously `crop` ran
after `stabilize` because stabilize's zoom-out correction could itself add a black border that
only a subsequent crop would clean up. That's no longer true — the percentile-based zoom is
borderless by construction, so there's nothing left for a post-stabilize crop to remove. `crop`
now runs immediately after `detect`, with `downscale` right behind it (unchanged from feature
7). New default order: `detect, crop, downscale, deblock, stabilize, denoise_video, upscale,
interpolate, normalize_volume, normalize_audio, speed, hdr, encode`. Running crop first means
every downstream stage — not just `downscale` — sizes off the already-cropped frame instead of
the original, so `deblock`/`stabilize`/`denoise_video`/`upscale`/`interpolate` never spend
compute on pixels that would just be cropped away. `deblock` still runs before `stabilize` for
the pre-existing reason (deblocking before stabilization's perspective warping keeps the
deblock model's input accurate, and gives the stabilizer cleaner detail to track motion
against).

**Change 2 — `deblock` defaults to the compact `realesr-general-wdn-x4v3` model;
`denoise_video` defaults to disabled**: `realesr-general-wdn-x4v3` is a denoise-optimized
Real-ESRGAN general-v3 SRVGG checkpoint (already present in `ai/model_cache.py`'s
`MODEL_REGISTRY`) trained on real-world degradation that includes both compression/blocking
artifacts and general noise. It's ~3x faster than the RRDB `RealESRGAN_x4plus` checkpoint
`deblock` previously defaulted to, and — because of what it was trained on — doubles as both a
deblock and a denoise pass. Making it the default `stages.deblock.ai_model` means one early AI
pass now handles both artifact classes that used to need two separate stages, so
`stages.denoise_video.enabled` now defaults to `false`: the other default-enabled stages don't
introduce new noise, so a dedicated second denoise pass is redundant in the common case.
`denoise_video` stays in `pipeline.default_order` at its existing slot (after `stabilize`,
before `upscale`) — disabling via `enabled: false` is what actually drops it from a run, per the
pipeline's "omission from `default_order` != disabled" semantics; a user who wants a separate
denoise pass, or who sets `stages.deblock.ai_model` to something that doesn't cover denoising,
can flip it back on. The `RealESRGAN_x4plus`→`RealESRGAN_x2plus` OOM-avoidance swap in
`deblock.py` (a strict `== "RealESRGAN_x4plus"` string check) does not fire for the new compact
default — it only matters when a user explicitly configures `RealESRGAN_x4plus`.

**`max_quality` preset is the deliberate exception**: it keeps `denoise_video` enabled and pins
`stages.deblock.ai_model` / `stages.upscale.ai_model` to `RealESRGAN_x4plus`, preserving the old
RRDB-everywhere, denoise-as-a-separate-pass behavior for users who explicitly asked for maximum
quality over speed. `1080p60`/`4k60`/`4k30` (which previously enabled `denoise_video` in their
`enable_stages`) now leave it disabled, inheriting the new default. `size_reduction`/
`remux_only` (denoise already off/not applicable) and `hdr_enhance` (enables `denoise_video` but
not `deblock` — denoise is its only artifact-removal stage) are unchanged.

## 9. AI interpolation to exact non-integer target framerates (RIFE + minterpolate hybrid) [IMPLEMENTED 2026-07-22]

**Problem**: the traditional `interpolate` path (FFmpeg `minterpolate`, `mi_mode=mci`) already
retimes to any target fps exactly, since `minterpolate`'s own `fps=` sub-option does true
motion-compensated interpolation to an arbitrary rate. The AI/RIFE path did not: RIFE
(`ai/wrappers/interpolate.py`) only supports integer factors (`current_fps * N`), but
`InterpolateStage._execute_ai()` computed `factor = int(target_fps / current_fps)` and forced
`factor = 2` whenever that floored to `<= 1` — always overshooting rather than ever landing on
the labeled target. 50fps→60fps forced `factor=2` and produced 100fps output (never retimed
down); 24fps→60fps produced 48fps, never 60.

**Fix — "RIFE under, then minterpolate up"**: `InterpolateStage._plan_ai_interpolation(
current_fps, target_fps, hybrid_enabled)` (pure, unit-tested helper in `core/stages/
interpolate.py`) computes the largest integer RIFE factor that stays at or below the target
(`floor(target_fps / current_fps)`), runs RIFE at that factor, then — only if the result still
falls short of the exact target — runs one `minterpolate` finish pass over the RIFE output to
reach it exactly. When no integer RIFE factor helps at all without overshooting (e.g. 50→60,
24→30, 60→75 all floor to `factor <= 1`), RIFE is skipped entirely and the AI path delegates to
the traditional/minterpolate-only path, which already reaches the exact target. This is a
deliberate strategy choice, not an AI-unavailable condition — it does not go through the
`ai_fallback` policy, and the resulting `StageResult.metadata["method"]` is accurately
`"traditional"` in that case.

Worked examples (hybrid enabled): 24→60 → RIFE factor 2 (→48fps) + minterpolate finish to 60;
30→60 → RIFE factor 2 (→60fps exactly), no finish; 50→60 → RIFE skipped, minterpolate-only;
30→120 → RIFE factor 4 (→120fps exactly), no finish; 24→30 → RIFE skipped, minterpolate-only;
60→75 → RIFE skipped, minterpolate-only; 30→75 → RIFE factor 2 (→60fps) + minterpolate finish to
75.

**Config/CLI**: `stages.interpolate.hybrid_ai_minterpolate` (default `true`) / CLI
`--interpolate-hybrid`/`--no-interpolate-hybrid`. Disabling it restores the exact pre-hybrid
behavior: RIFE factor forced to 2 whenever the natural floor is `<= 1`, no finish pass, output
overshoots the labeled target. When the finish pass runs, it's a second crash-resilient `.mkv`
temp (same rationale as the RIFE temp — see feature notes on the streaming AI-frame pipeline in
AGENTS.md), muxed with the original audio in place of the RIFE temp; both temps are cleaned up
on every success/failure path. Returned stage metadata gains `rife_factor`,
`minterpolate_finish: bool`, and `fps_out`.

## 10. Input file lists & pluggable parsers [IMPLEMENTED 2026-07-22]

**Problem**: `process`'s positional `PATHS` argument only accepts filenames/directories typed (or
shell-globbed) directly on the command line — there was no way to hand it a prepared list of
files, e.g. a batch selected in a file manager and pasted into a terminal, or a manifest
pairing specific inputs with specific output names.

**Fix**: `process --from-file PATH` (repeatable) reads `PATH`'s contents and parses it into more
input paths via a pluggable parser (`core/input_parsers/`, registered the same way `core/stages/`
registers stages). `--from-file-parser NAME` selects the parser statefully (applies to every
`--from-file` that follows it, until the next `--from-file-parser`); an inline `NAME:PATH` value
on `--from-file` overrides the parser for just that one file. Default parser: `shlex` —
deliberately chosen because `shlex.split()`'s whitespace/newline-splitting-with-quote-honoring is
exactly what a Dolphin (or most file managers') drag-and-drop-onto-a-terminal paste produces.
Three more built-ins ship (`lines` — one path per line, `#`-comments; `csv` — positional or
headered `input[,output][,recursive]`; `json` — array of strings/objects or `{"inputs": [...]}`),
and parsers are user-extensible exactly like stages: subclass `InputParser`, set a `name`, register
it, import the module in `core/input_parsers/__init__.py`.

`--from-file` entries combine with positional `PATHS` (both contribute to the same job list); a
list-file entry may carry its own output path (csv `output` column / json `output` key) and/or a
per-entry `recursive` override for directory entries — everything else falls through to the normal
`-o`/`general.output_dir` resolution. `--output-name` still requires exactly one input total across
both sources, and errors if combined with any per-file output (ambiguous). See AGENTS.md's "Input
file lists & pluggable parsers" section for the full CLI-side mechanics (argv-order recovery,
inline-override disambiguation, fallback behavior).

## 11. Live progress bars [IMPLEMENTED 2026-07-22]

**Problem**: a long `avf process` batch gave no live sense of how far along the current video or
the whole batch was — only start/finish lines and the per-job report table printed after each
job completed.

**Fix**: `process` gets two independent live `rich.progress` bars in one shared console region
(`cli/progress.py::ProgressReporter`): a BATCH bar (total = job count; completed =
fully-finished jobs + the current job's own 0..1 progress, so it advances smoothly within a
video, not just once per video) and a PER-FILE bar (total = 1.0, reset at the start of each job,
tracking that job's own progress as reported by `Pipeline.execute_job`'s `progress_callback` on
every per-stage progress update). Each bar is independently controlled by
`reporting.progress_batch` / `reporting.progress_file` (`--progress-batch/--no-progress-batch`,
`--progress-file/--no-progress-file`) — both `True` by default — but a bar only actually renders
when `console.is_terminal` is true; piped/redirected/non-TTY runs get no bars regardless of
config, identical to pre-feature behavior. Per-job Rich report tables print via the same console
during the live region (`rich.progress` supports interleaved `console.print`); the end-of-run
summary/stage-timing/JSON-report tail is printed only after the region stops, so it's never
overwritten by the bars. v1 limitation: with `general.max_concurrent_jobs` > 1 there is still
only one file bar, which reflects whichever job most recently reported progress — no per-job
bars yet. See AGENTS.md's "Feature 6 — live progress bars" section for implementation details.

## 12. True source cadence recovery & timestamp-safe pipeline (PLANNED 2026-09-08, user-approved)

**Problem**: AVF treats the *encoded* framerate as the truth about a video's motion. Many real
inputs lie about it. A 24fps recording published at 60fps CFR carries 60 frames per second of
which only 24 are distinct — the other 36 are duplicates inserted by whoever transcoded it.
Variable-framerate captures (phone video, screen recordings) that were converted to CFR for
publishing are the same story with an irregular original cadence. Encoder noise means the
duplicates are rarely bit-identical, so naive equality checks miss them.

Two consequences, both bad:

1. **Interpolation is defeated.** `InterpolateStage` reads `current_fps` from the probe
   (`avg_frame_rate` = 60) and interpolates 60→120. But 36 of every 60 input frames carry no new
   motion, so RIFE/minterpolate spend their effort synthesizing intermediate frames *between two
   identical images* — producing nothing, at full GPU/CPU cost. The genuinely useful work
   (24→60, real new motion) never happens because AVF never learns the content is 24fps.
2. **Every stage overpays.** `upscale`, `deblock`, `denoise_video` and `stabilize` each process
   the duplicate frames at full per-frame cost. On a 24-in-60 input that is **60% of all AI
   inference wasted** on frames that are copies of their predecessor.

**Fix**: recover the true content cadence up front, drop the duplicate frames, and carry the
recovered timeline through the pipeline as genuine VFR (original presentation timestamps
preserved) rather than re-flattening it to CFR.

### 12.1 Cadence analysis (`core/cadence.py`, new module)

`analyze_cadence(path, config, sample_sec=...) -> CadenceAnalysis` runs one **decode-only**
ffmpeg pass (no encode, so it is cheap relative to any processing stage):

```
ffmpeg -i IN -map 0:v:0 -an -sn \
       -vf mpdecimate=hi=<hi>:lo=<lo>:frac=<frac>,showinfo \
       -f null -
```

`mpdecimate` drops near-duplicate frames; the `showinfo` filter immediately after it therefore
logs one line per **surviving (unique)** frame to stderr, each carrying that frame's original
`pts_time`. Parsing those `pts_time` values yields the recovered timeline directly — no pixel
comparison code of our own, and `mpdecimate`'s `hi`/`lo`/`frac` thresholds are exactly the
tunables needed to tolerate encoder noise between "identical" frames.

**Do not use `metadata=print` here.** An earlier draft of this spec specified
`mpdecimate,metadata=print:file=-`; it emits **nothing at all**. `metadata=print` only prints for
frames that already carry an attached metadata key, and `mpdecimate` never attaches one.
Verified on ffmpeg n9.0.1: the `metadata=print` form yields 0 `pts_time` records for a file whose
`showinfo` form correctly yields 48. `showinfo` writes to **stderr at the default log level**, so
the analysis pass must capture stderr rather than stdout.

**Native VFR vs. a CFR encode of a VFR original — these are different inputs.** A direct phone or
screen capture is genuinely VFR: irregular gaps, but **no duplicate frames**, so `retime` must
SKIP. A YouTube-style download of that same content is CFR *padded with real duplicates*, so
`retime` must run. Distinguishing them requires a MEASURED total frame count: deriving it as
`duration * encoded_fps` uses `avg_frame_rate`, which § 12.3 documents as unreliable for VFR — a
genuine 123-frame capture advertising 60fps over 3s "measured" 180 total frames and reported a
0.317 duplicate ratio for a file with zero duplicates, triggering a needless full re-encode and
pinning the CFR deliverable to a meaningless mean rate. The analysis chain is therefore
`showinfo,mpdecimate,showinfo`: instance 0 counts every decoded frame, instance 2 counts the
survivors, both from the same single decode pass, attributed via ffmpeg's `[Parsed_showinfo_<n>]`
log prefix. Note a showinfo record can wrap across multiple log lines, so count `pts_time`
occurrences, not prefix occurrences (123 frames produced 251 prefix lines). Verified:
native VFR 123/123 (0.000, skips), VFR-transcoded-to-CFR 180/123 (0.317, runs), 24-in-60 120/48
(0.600, runs).

`CadenceAnalysis` (frozen dataclass) reports:

- `encoded_fps` — what the container claims (today's `probe().framerate`).
- `total_frames`, `unique_frames`, `duplicate_ratio` = `1 - unique/total`.
- `unique_timestamps: list[float]` — the recovered timeline.
- `detected_fps` — `unique_frames / analysed_duration`.
- `nominal_fps` — `detected_fps` snapped to the nearest standard rate (23.976, 24, 25, 29.97,
  30, 48, 50, 59.94, 60, 100, 120) when within `snap_tolerance`, else `detected_fps` unchanged.
- `is_padded` — `duplicate_ratio >= min_duplicate_ratio`.
- `is_regular` — coefficient of variation of the inter-frame gaps is under `regularity_tolerance`.
  Note this is computed on the *raw* recovered gaps, so grid quantization (see 12.6) makes a
  uniform-cadence source read as mildly irregular; `grid_rate` disambiguates.
- `grid_rate` — the rate whose period divides all recovered gaps (i.e. the original encoded
  grid). Present when the timeline is a subset of a uniform grid, which is true for any input
  that was itself CFR-encoded.

Analysis may be limited to the first `analysis_sample_sec` seconds (default 120; `0` = whole
file) because its only job is the *decision*; the retime pass itself always runs `mpdecimate`
over the whole input.

### 12.2 The `retime` stage (`core/stages/retime.py`, new)

Runs **immediately after `detect`**, so every later stage sees the reduced frame set and the
compute saving compounds across the whole pipeline. One ffmpeg pass:

```
ffmpeg -i IN -vf mpdecimate=hi=<hi>:lo=<lo>:frac=<frac> -fps_mode vfr \
       -c:v <temp codec> -crf <temp_crf> -c:a copy -y OUT.mkv
```

`-fps_mode vfr` is load-bearing: it tells ffmpeg to honour the surviving frames' own timestamps
instead of conforming the output back to a constant rate. The temp is `.mkv` for crash
resilience, matching the convention the AI stages already use.

The stage **SKIPs** (never fails) when `analyze_cadence` reports `is_padded == False` — an input
that is already honest costs one cheap decode pass and no re-encode. It also skips when the
input has no video stream, and degrades to SKIPPED (with a warning, never a hard failure) if the
analysis pass errors, since a cadence miss must not take down an otherwise fine job.

Stage metadata: `method` (`"mpdecimate"`), `encoded_fps`, `detected_fps`, `nominal_fps`,
`frames_before`, `frames_after`, `frames_removed`, `duplicate_ratio`, `is_regular`, `grid_rate`.

### 12.3 Propagating the recovered timeline

The recovered cadence is published into the job's `input_info` as `true_framerate`,
`cadence` (the `CadenceAnalysis`), and `is_vfr`.

**`InterpolateStage` must read `input_info["true_framerate"]` in preference to
`input_info["framerate"]`** when computing `current_fps`. This is the entire payoff of the
feature: on a 24-in-60 input targeting 60fps, today's code sees 60→60 and skips as "already at
target"; with the recovered cadence it correctly sees 24→60 and does real interpolation.

**Critical gotcha — do not re-probe for fps downstream of `retime`.** `probe()._parse_fps()`
prefers `avg_frame_rate`, and for a VFR Matroska file that value is a container-level nominal
that *misreports* the real cadence (empirically: a 48-frame, 2-second VFR MKV still advertised
`avg_frame_rate=60/1`). Any stage needing the rate after `retime` must take it from
`input_info`, not from a fresh probe.

### 12.4 Timestamp-safety contract (applies to every stage)

Once an intermediate is VFR, any stage that silently re-conforms it to CFR re-inserts the
duplicate frames and undoes the feature. Each stage falls into one of three classes:

**(a) Filter-only ffmpeg stages** — `crop`, `downscale`, `denoise_video`, `deblock`
(traditional), `hdr`, `stabilize` (transform pass), `encode`. These pass PTS through correctly
under ffmpeg 9's default `fps_mode=auto` for Matroska, but the behaviour is muxer-dependent and
must not be left implicit: add an explicit `-fps_mode passthrough` to the output args via a
shared helper (`ffmpeg_utils.timing_output_args(...)`) so it is deterministic across muxers and
ffmpeg versions.

**(b) Raw-frame pipe stages** — `upscale` (AI), `deblock` (AI), `interpolate` (AI/RIFE),
`stabilize`'s raw pipe. These are the dangerous class.

- **Reader side (mandatory fix).** The decoder feeding the raw pipe *re-expands VFR back to CFR
  by duplicating frames* unless told otherwise. Verified: piping a 48-frame VFR file to
  `-f rawvideo` yielded **120 frames** — every duplicate `retime` had just removed was silently
  put back, at full AI inference cost. The reader's output args must include
  `-fps_mode passthrough`, which restores the correct 48. This single omission would negate the
  entire feature while appearing to work.
- **Writer side.** `rawvideo` carries no timestamps, so the writer must reconstruct them. There
  is no reliable ffmpeg-native mechanism to graft an arbitrary per-frame PTS table onto a raw
  stream — both candidate approaches were tested and rejected (see 12.6). The writer therefore
  uses a **carrier rate equal to `nominal_fps`**, which reproduces the intended timeline
  *exactly* whenever the content cadence is uniform (the overwhelmingly common padded case) and
  **linearizes** genuinely irregular cadence to a constant rate. Linearization is a real,
  bounded limitation: it must be recorded in stage metadata as `timeline_linearized: true` and
  surfaced in the run report rather than passing silently.

**(c) Timestamp-rewriting stages** — `speed` (`setpts`). Already correct: `setpts` scales
whatever PTS it is given, so it composes with VFR without change. It must still carry
`-fps_mode passthrough` so the scaled timestamps survive to the muxer.

### 12.5 Final output timing

`general.output_timing` — `cfr` (default) | `vfr` | `passthrough`. The default keeps the final
deliverable CFR, preserving the MP4-for-compatibility posture; `vfr` carries the recovered
timeline all the way out. Intermediates are always VFR regardless — this key governs only the
last encode.

**A CFR deliverable must be pinned to the stream's TRUE cadence.** `-fps_mode cfr` on its own
conforms the output to whatever rate the *container* still advertises, which for a retimed
24-in-60 input is still 60 — so ffmpeg faithfully re-inserts exactly the duplicate frames
`retime` just removed. Verified end-to-end: 120 frames in → 48 after retime → **120 back out**,
i.e. running `retime` appeared to do nothing at all. The encode stage therefore also emits
`-r <true_framerate>` in `cfr` mode, making the deliverable honest 24fps CFR. `vfr`/`passthrough`
carry real timestamps and must never be rate-pinned.

This makes `true_framerate` mean **"the stream's genuine CURRENT cadence"**, not "the original
source cadence" — so every stage that legitimately retimes the stream must refresh it.
`interpolate` does, via `fps_out`, which it must report on **every** completed path (AI *and*
traditional). Omitting it on the traditional paths caused a retimed 24-in-60 input interpolated
to 60fps to be encoded back down to `-r 24`, silently discarding every frame minterpolate had
just synthesized.

**Existing user configs will not pick this stage up.** `pipeline.default_order` is a list, and
list-valued config keys are REPLACED WHOLESALE by a later cascade layer (AGENTS.md, "Config
cascade") — so any user whose `~/.config/auto-video-fixer/config.yaml` pins its own
`default_order` gets a pipeline with no `retime` in it, and `--stage retime` lands the stage at
the wrong position (after `interpolate`, where it is useless). `avf config upgrade` is the
supported remedy. This is the same migration hazard `denoise_video`'s default flip hit.

### 12.6 Empirically validated ffmpeg behaviour (ffmpeg n9.0.1)

Recorded so future work does not have to re-derive it. Test case: `testsrc2` at 24fps padded to
60fps CFR, 2 seconds, 120 frames.

| Behaviour | Result |
|---|---|
| `mpdecimate` + `-fps_mode vfr` | 120 → **48** frames; PTS gaps alternate 0.050/0.033s — the true 24-on-60 pulldown pattern, exactly recovered |
| Filter stage → MKV, default `fps_mode` | VFR preserved (48 frames, PTS intact) |
| Filter stage → MP4, default `fps_mode` | VFR preserved; `avg_frame_rate` reported as `2880/119` |
| VFR file → `-f rawvideo` pipe, no `fps_mode` | **48 → 120 frames** (duplicates silently reinserted) |
| VFR file → `-f rawvideo` pipe, `-fps_mode passthrough` on the reader's output args | 48 frames preserved — the required fix |
| VFR MKV `avg_frame_rate` | **Unreliable** — advertised `60/1` for a 48-frame 2s file |
| `mpdecimate,metadata=print:file=-` | **Emits nothing** (0 records) — `metadata=print` needs a pre-attached metadata key; use `showinfo` (48 records) |

Rejected write-side timestamp-graft mechanisms:

- **`sendcmd` + `setpts expr`** (file-driven per-frame command table): command windows do not
  align reliably with frame boundaries; produced duplicated and out-of-order PTS
  (`0, 0.042, 0.042, 0.125, …` against a target of `0, 0.050, 0.083, 0.133, …`). Rejected.
- **`setts` bitstream filter with a closed-form expression**: `N` counts packets in *decode*
  order, so with B-frames the emitted PTS are scrambled (`0, 11.25, 7.5, 15.0, 3.75, …`).
  Rejected.
- **`mkvmerge --timestamps`** (would work correctly): rejected as a new external dependency;
  not present on the reference machine.

`-fps_mode` is an **output** option. Placing it before `-i` corrupts input parsing (observed as
spurious `Duplicate element` / EBML errors), so it must be emitted after the input spec.

**Build gotcha — the reader fix lives in Rust.** The `-fps_mode passthrough` reader fix in 12.4b
must be applied in `rust/avf_framepipe/src/lib.rs` (`FrameReader::new`), because that crate — not
`ai/frame_processor.py` — is the active reader whenever the extension is installed. The Python
fallback reads via `cv2.VideoCapture`, which returns the real decoded frames and is therefore
unaffected by this class of bug.

After editing that Rust source, **`uv sync` / `uv run` serve a STALE CACHED WHEEL** and silently
revert the fix. Verified: the same reader returned 48 frames from `.venv/bin/python` immediately
after `maturin develop --release`, but 120 frames again through `uv run`, which reinstalled the
cached build. Force a real rebuild with:

```
uv sync --all-extras --reinstall-package avf_framepipe
```

This failure mode is silent and looks exactly like the bug the fix addresses, so verify the frame
count through `uv run` (not just a direct venv python) after any change to the Rust reader.

### 12.7 Config surface

```yaml
stages:
  retime:
    enabled: true            # on by default, per user decision
    hi: 768                  # mpdecimate hi   (64*12)
    lo: 320                  # mpdecimate lo   (64*5)
    frac: 0.33               # mpdecimate frac
    min_duplicate_ratio: 0.05   # below this the stage SKIPs (input already honest)
    analysis_sample_sec: 120    # 0 = analyse whole file
    snap_tolerance: 0.02        # nominal_fps snapping window
    regularity_tolerance: 0.15  # gap CV below this counts as uniform cadence
    temp_crf: 16
general:
  output_timing: cfr         # cfr | vfr | passthrough
```

CLI: `--retime/--no-retime`, `--retime-min-duplicate-ratio`, `--output-timing`. Overridable at
every layer the user already expects (config file, preset `enable_stages` + per-stage config,
`pipeline.default_order` per-occurrence overrides, per-input overrides from `--from-file`
manifests, and the CLI flags above).

### 12.8 Non-goals / limitations (v1)

- Genuinely irregular cadence is **linearized** through the raw-pipe AI stages (12.4b). Filter
  stages preserve it exactly.
- RIFE is not made timestamp-aware — it interpolates on the uniform recovered cadence rather
  than synthesizing a variable number of frames per gap. `minterpolate` already handles variable
  gaps natively. Timestamp-aware RIFE is deferred.
- Interlaced / telecined sources are out of scope; `mpdecimate` is not a field-aware inverse
  telecine (`fieldmatch`/`decimate` would be the right tools) and no IVTC is attempted.

## 13. Stage-level progress reporting (chunked and ffmpeg-driven) (PLANNED 2026-09-08, user-approved)

**Problem**: feature §11 gave `process` a batch bar and a per-file bar, but the per-file bar only
advances when a *stage* reports progress. Stages that do one long opaque operation — a single
`minterpolate` ffmpeg invocation, a whole-file `stabilize` transform pass — jump from 0 to 1 with
a multi-minute silence in between. The user cannot tell a slow stage from a hung one.

### 13.1 Chunked stages (straightforward)

Stages that already split work into chunks know their own denominator and simply are not
reporting it. `interpolate`'s traditional path splits into `parallel_chunks` time-chunks; the AI
stages (`upscale`, `deblock`, `interpolate`) stream in fixed frame-chunks and already log
`chunk #N (25 frames)` lines via `ai/frame_processor.py`'s instrumentation. Each must call the
stage `progress_callback` with `chunks_done / chunks_total` (for parallel chunks, count
completions, not the running index — they finish out of order).

This is the cheap, high-value half and should land first.

### 13.2 Non-chunked ffmpeg stages (parse the live progress stream)

For a single long ffmpeg invocation, use ffmpeg's `-progress pipe:<fd>` machine-readable stream
rather than scraping the human-readable stderr log. It emits `frame=`, `out_time_us=`,
`speed=` blocks terminated by `progress=continue` / `progress=end`. `core/ffmpeg_utils.py`
already has `_parse_ffmpeg_progress()` and `run_ffmpeg()` accepts a progress callback — extend
that path rather than adding a second mechanism.

**The hard part the user correctly identified is the denominator**, not the numerator. Getting
"how far along" requires knowing the expected output duration or frame count, which several
stages change:

- `interpolate` — output frame count is `input_frames * target_fps / current_fps`; duration is
  unchanged. Prefer progressing on **`out_time_us` against input duration**, which is invariant
  under framerate change and therefore correct for both minterpolate and RIFE.
- `speed` — duration changes by exactly `1/speed`; expected output duration is
  `input_duration / speed`. Known exactly up front.
- `retime` (§12) — drops frames, so frame-count denominators are wrong, but `out_time_us` still
  tracks the input timeline. Another argument for timing over frame counting.
- `crop`/`downscale`/`deblock`/`denoise_video`/`hdr`/`encode` — duration and frame count both
  preserved; either denominator works.

**Therefore: use `out_time_us / expected_output_duration` as the universal progress metric**, with
`expected_output_duration` defaulting to the input duration and overridden per stage only where
the stage provably changes it (`speed`). Frame-count progress is a fallback for stages where
duration is unknown. Under VFR intermediates (§12) frame counting is actively misleading, so the
time-based metric is the right primitive regardless.

Scope note: `stabilize` runs two passes (detect + transform); it must report progress across the
pair as a weighted sum, not restart at 0 for the second pass.

## 14. Per-scene output files (PLANNED 2026-09-08, user-approved)

**Problem**: scene detection and scene-mode processing already exist, but a run always produces
one output file. For compilation videos — the user's common case — the individually useful
artifact is one file per detected scene, so scenes can be kept, dropped, or reordered by hand.

### 14.1 Requirements

- Opt-in `--split-scenes` (plus config key) making `process` emit one output file per detected
  scene instead of (or in addition to — this must be a distinct, explicit choice) the single
  combined output.
- Naming must encode **source order**, because scenes may be processed in parallel and therefore
  file mtimes do NOT reflect scene order (the user flagged this explicitly). Use a zero-padded
  ordinal plus the source stem: `<stem>_scene_0001.mp4`. The ordinal is the contract §15 relies on.
- Write a **sidecar manifest** (JSON) beside the clips recording, per scene: source path, scene
  index, source start/end timestamps, duration, resolution, and the crop rectangle actually
  applied. Deriving order from filenames alone is fragile; the manifest is authoritative and makes
  §15 robust.
- Reuse `core/scenes.py::split_scene_video()` rather than adding a second splitting path.

### 14.2 Per-scene independent auto-crop

Compilation videos splice sources with different aspect ratios, so a single whole-video crop
rectangle is wrong for most scenes. When `--split-scenes` is active, auto-crop must be
computable **per scene** rather than once for the whole input.

Note this specifically contradicts the existing whole-video `cropdetect=reset=0` approach
documented in §3 / AGENTS.md's "Auto-crop" section, which deliberately computes one maximal
rectangle for the entire file. Per-scene cropping therefore needs its own detection pass scoped
to each scene's time range, and the two modes must not be silently mixed. The VLM assist
described in §3 (distinguishing true content bounds from watermarks/overlays) applies per scene
as well.

## 15. Scene clip concatenation tool (PLANNED 2026-09-08, user-approved)

**Problem**: having produced per-scene clips (§14), the user wants to hand a selected subset back
to a tool and get one contiguous video.

### 15.1 Requirements

- New CLI command (e.g. `avf concat`) accepting clip paths, directories, or a `--from-file` list
  (reuse the existing `core/input_parsers/` registry from §10 — do not invent a second list format).
- **Ordering.** Default to reconstructing the original scene order from the §14.1 manifest when one
  is present, falling back to the filename ordinal. **Never order by mtime** — parallel scene
  processing makes timestamps meaningless, which the user called out directly. An explicit
  `--order` override (`manifest` | `name` | `given`) must exist.
- **Mixed sources.** When clips come from more than one source video, original scene order is
  undefined. Default to an error explaining the ambiguity and naming the conflicting sources,
  with `--order given` as the documented escape hatch (the user's stated position: the intended
  behaviour is not obvious and should be user-specified).
- **Mismatched resolution/framerate/codec.** Concatenating streams that differ cannot be done with
  a stream copy. Default to erroring with a clear diff of what differs, and offer an opt-in
  `--normalize` that re-encodes all inputs to a common spec. Silent re-encoding must not be the
  default — it is lossy and slow, and the user should choose it.
- Use the ffmpeg `concat` **demuxer** with `-c copy` on the fast path; `core/scenes.py` already has
  `_concat_demuxer()` and `concat_video_clips()` to build on.
- Audio must be handled explicitly: clips with and without audio tracks cannot be concat-copied
  together; detect and report rather than producing a silent or truncated result.
