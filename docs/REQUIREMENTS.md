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

### R5.1 Scene-detection frame differencing

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

### R5.2 Perceptual hashing / duplicate detection

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
