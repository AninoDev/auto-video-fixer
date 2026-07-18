# Auto Video Fixer - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Stages after the first geometry-changing one saw stale `input_info` (post-crop upscale bug)**:
  `Pipeline.execute_job()` probed `input_info` once up front (plus once more after scene mode)
  and never again, so every later stage's `should_run()` judged the ORIGINAL input's resolution
  even after an earlier stage had already changed it. Concretely, once `crop` removes
  letterboxing/pillarboxing: a 1080x1080 input with letterboxed 16:9 content cropped to
  1080x608, but `upscale.should_run()` still saw 1080x1080, misjudged the orientation-aware
  target bounds, and the final output stayed 1080x608 instead of upscaling to 1920x1080; a
  1920x1080 input with pillarboxed 9:16 content cropped to 608x1080, but `upscale` still saw
  1920x1080 == target and skipped entirely, leaving the output at 608x1080 instead of the
  correct portrait 1080x1920. `UpscaleStage.should_run()`/`_effective_target_bounds()` were
  themselves correct -- the input they were given was stale. `execute_job()` now re-probes
  `input_info` (re-injecting any pipeline-layered keys, e.g. `general.target_format`) whenever a
  stage completes with a new output file; a re-probe failure logs a WARNING and keeps the
  previous `input_info` rather than failing the job. See AGENTS.md's "Pipeline Behavior" for the
  mechanism.

  A second half of the same bug survived the re-probe fix: `Pipeline.auto_determine_stages()`
  decided whether `"upscale"` even entered the job's PLAN using that same up-front, pre-crop
  probe (`if resolution[0] < target_w or resolution[1] < target_h: stages.append("upscale")`) --
  orientation-blind, and evaluated before `crop` has run. For the 1920x1080-pillarboxed-to-
  608x1080 case above, 1920x1080 already satisfies a `[1920, 1080]` target, so `"upscale"` never
  entered `job.stages` at all; the re-probe fix couldn't help because a stage absent from the
  plan is never instantiated, so its (now-correctly-fresh) `should_run()` never even runs. Live
  end-to-end repro: `Processing pillarbox_wide.mp4: stages=['crop', 'detect', 'stabilize', ...]`
  -- `upscale` absent from the plan, final output stuck at 608x1080 instead of ~1080x1918.
  `auto_determine_stages()` now appends `"upscale"` to the plan whenever
  `quality.quality_target.target_resolution` is configured at all, regardless of the up-front
  probe's resolution, and leaves the actual run/skip decision entirely to the in-loop,
  orientation-aware, freshly-reprobed `should_run()`. Behavior preserved: with no
  `target_resolution` configured, `"upscale"` still never enters the auto-determined plan.
  Audited the rest of `auto_determine_stages()` and the preset `enable_stages` path for the same
  class of bug -- no other stage's plan membership is geometry-gated (`interpolate`/`hdr`'s
  plan-time checks use framerate/HDR-ness, which `crop` doesn't alter, and were left unchanged).
- **Scene-mode OOM incident: per-scene `parallel_chunks` override was silently swallowed**: a
  real 2-minute 4K/30fps run in scene mode OOM-killed a 56 GiB container. Root cause:
  `InterpolateStage.execute()` accepted `parallel_chunks` only via `**kwargs`, never forwarding it
  to `_execute_traditional()` -- so scene mode's per-scene chunk-budget bound
  (`_scene_worker_budget()`, `core/scenes.py`) was lost and every concurrently-processed scene's
  traditional interpolation fell back to `stages.interpolate.parallel_chunks`'s own independent
  auto-chunking (0 = auto = `min(cpu, 8)`) PER SCENE. Log evidence: 6 scenes processed
  concurrently, each ALSO auto-chunking up to 8 ways, peaking at ~12 concurrent 4K `minterpolate`
  ffmpeg processes. `execute()` now accepts `parallel_chunks` as a real named parameter (default
  `None`) and forwards it through.
- **Scene VLM classification ignored `analysis.vlm.enabled`**: `scenes.drop_non_content` called
  the per-scene VLM sampling + coordinator pass unconditionally, even when VLM was disabled. With
  the VLM endpoint down, this meant ~8 minutes of 120s HTTP timeouts before "keeping all scenes
  anyway" (the same incident run above). `run_scene_pipeline()` now checks `analysis.vlm.enabled`
  first: when `drop_non_content` is true but VLM is disabled, it logs one WARNING and keeps all
  scenes without any VLM/network call.

- **Volume/audio normalization no longer fails on silent or near-silent audio**: inputs with no
  real audio track get a silent stereo track added earlier in the pipeline, so by the time
  `normalize_volume`/`normalize_audio` (`core/stages/normalize_audio.py`) run there IS an audio
  stream -- just silent, or near-silent if the source added dithering noise. Two-pass loudnorm's
  first pass measured `input_i = -inf` on pure digital silence (or produced other unusable
  values on near-silence), and the second pass blew up feeding that back in as infinite gain,
  failing the stage outright. Both stages now parse the measured integrated loudness robustly
  (`-inf`/`inf`/`nan`/unparseable all collapse to a silence sentinel rather than raising) and, if
  it's the sentinel or `<=` a new `stages.normalize_volume.silence_threshold_db` /
  `stages.normalize_audio.silence_threshold_db` config key (default `-80.0` LUFS, chosen to sit
  just above 2-3 LSBs of 16-bit dither noise: `20*log10(3/32768) ~= -80.8 dBFS`), skip
  normalization and pass the input through unchanged (logged at INFO, distinguishing perfect
  silence from near-silence) instead of failing -- matching the mid-execute skip convention
  `UpscaleStage`'s "already at target resolution" gate and `StabilizeStage`'s "no stabilization
  needed" path already use (`StageStatus.COMPLETED` with `skipped_reason` set, pipeline
  continues to later stages). Implemented once in `NormalizeAudioStage` and reused by
  `NormalizeVolumeStage` (both classes run the identical loudnorm algorithm), so a
  per-occurrence `pipeline.default_order` `config:` override on either stage section works
  automatically.

- **Real-world long videos no longer die at 600s ("timeout reached")**: nearly every stage's
  main ffmpeg processing/mux pass hardcoded a fixed wall-clock `timeout=600` (some
  `1800`/`3600`) in `core/stages/*.py` and `core/quality.py` -- wrong-shaped for a whole-video
  pass, since a legitimately long input (not a stuck/hung ffmpeg process) would simply exceed
  it and fail. These now resolve a configurable timeout that defaults to `null` (unlimited):
  new `pipeline.stage_timeout` (global default) and per-stage `stages.<name>.timeout`
  (`BaseStage.stage_timeout()` in `core/stages/base.py`, resolution order stage -> global ->
  `None`), plus `quality.timeout` for `core/quality.py`'s VMAF/SSIM quality-gate comparison
  pass (not a stage, so it gets its own key). All three share identical null/0-means-unlimited,
  positive-number-is-seconds, negative/non-numeric-is-a-config-error semantics via the new
  `resolve_timeout()` helper in `config.py`. `run_ffmpeg()` (`core/ffmpeg_utils.py`) now accepts
  `timeout=None` to mean "wait forever" (`subprocess.Popen.wait(timeout=None)` already blocks
  indefinitely -- no behavior change needed there). No new per-occurrence machinery was
  needed: the existing `pipeline.default_order` mapping entry's `config: {timeout: ...}`
  override already reaches a stage's resolved timeout through the same mechanism every other
  per-occurrence config key uses. Short, genuinely-bounded helper calls (probes, hwaccel
  detection, single-frame extraction, cropdetect quick samples, loudness measurement) were
  deliberately left at their small fixed timeouts -- a hang there indicates real breakage, not
  a long input. See AGENTS.md's "Stage/quality ffmpeg timeouts" section for the full list of
  wired-up vs. kept-fixed call sites, and `docs/config.example.yaml` for worked examples
  (including the per-occurrence form). Scene mode's split/concat/mux helpers
  (`core/scenes.py`, previously fixed at 300/600/1800s) are wired to `pipeline.stage_timeout`
  the same way.

- **ffmpeg/ffprobe subprocess spawns could leave the user's terminal stuck in raw mode**: no
  spawn site in this codebase passed `-nostdin` (ffmpeg) or detached stdin, so whenever a spawned
  ffmpeg's stdin happened to be the controlling terminal, ffmpeg would switch that tty to raw mode
  to poll for interactive keyboard commands (`q` to quit, etc.) -- and, if the process was
  killed/crashed/backgrounded before exiting cleanly (an interrupted `avf process` being the
  common case), never restore it. Symptom: after such a run, the shell shows no keystroke echo and
  an extra blank line per command until the user manually runs `reset`/`stty sane`. Root-caused via
  `grep -rn "subprocess.run\|subprocess.Popen" src/` plus `Command::new` in the Rust crates -- no
  call site anywhere passed `-nostdin` or `stdin=DEVNULL`/`Stdio::null()`.
  - **Central fix**: `core/ffmpeg_utils.py`'s `run_ffmpeg()` -- the shared runner nearly every
    stage's ffmpeg pass routes through -- now always prepends `-nostdin` to the built command and
    passes `stdin=subprocess.DEVNULL` to `Popen()`, fixing every stage in one place. `probe()`
    (ffprobe) and `detect_hardware_acceleration()` (ffmpeg `-hwaccels`), the two other central
    helpers in the same module, got the same treatment (`-nostdin` where applicable -- ffprobe has
    no such flag, so it's `stdin=DEVNULL` only there).
  - **Per-site fixes** for the spawns that don't route through the central helpers:
    `stages/stabilize.py`'s two direct `ffprobe` calls (`_get_video_dimensions`/
    `_get_video_framerate`) and its manual decode→transform raw-pipe `ffmpeg` (the decode side --
    the transform side already reads its video from a pipe, which already prevents tty capture, so
    it's intentionally left alone); `ai/model_cache.py`'s `curl` fallback download.
  - **Rust**: `rust/avf_scenes/src/lib.rs`'s ffprobe/ffmpeg spawns and `rust/avf_framepipe/src/lib.rs`'s
    `FrameReader` decode ffmpeg gained `-nostdin` (the ffmpeg one already had `Stdio::null()` set;
    the ffprobe one gained it). `rust/avf_hashing/src/lib.rs`'s ffprobe/ffmpeg spawns (found by the
    same grep audit, not originally called out for this fix but in scope for "no spawn site")
    got the identical treatment. `FrameWriter`'s ffmpeg (piped stdin, by design) is untouched, same
    as `stabilize.py`'s transform process and `ai/frame_processor.py`'s `StreamingVideoWriter`/
    `frames_to_video()` writer pipes -- a deliberately piped stdin already can't capture a tty.
  - **Defense in depth**: new `sanitize_console_text()` (`config.py`, alongside `redact_secrets()`)
    strips/replaces C0 controls (except `\t`/`\n`), DEL, the C1 range, and ESC with U+FFFD, without
    touching any other Unicode (CJK/RTL/combining marks/emoji preserved, no normalization) --
    covers the secondary vector where a crafted filename's raw escape bytes get echoed into console
    output. Wired into `logger.py` as `_SanitizingConsoleFormatter`, attached to the console
    (Rich) handler only -- file/log-file handlers keep raw text for debugging. `cli.py`'s direct
    `console.print()` calls that interpolate external data (filenames, exception text, VLM
    summaries/tags) now route that data through the same helper (`_safe()`) at the interpolation
    site, since Rich's own `[style]` markup tags need to keep working and can't go through a
    blanket formatter the way the logging path does.

### Added
- **Per-stage `use_ai` tristate, shared AI/traditional method resolution, and loud method-choice
  logging**: the four AI-capable stages (upscale, deblock, denoise_video, interpolate) now resolve
  "ai" vs "traditional" through one shared `BaseStage.resolve_ai_method(explicit_method,
  auto_default)` helper (`core/stages/base.py`), called once at the top of each stage's
  `execute()`. Precedence: an explicit `method=` kwarg from a caller (scene mode, tests) wins
  outright; then the new `stages.<name>.use_ai` (tristate, default `null`); then `general.use_ai`
  (unchanged, set by `--ai`/`--no-ai` -- per the existing `ai_fallback` convention, the global flag
  does not override an explicit per-stage config value); then the stage's own hardcoded auto
  default (upscale/deblock -> AI, denoise_video/interpolate -> traditional -- all four deliberate,
  documented in `AGENTS.md`). `interpolate`'s `method` parameter changed from `str = "traditional"`
  to `str | None = None`; `denoise_video`'s likewise, preserving its `"traditional"` auto default
  exactly. Every stage now logs the resolved method + why at **INFO** (`_log_ai_method_choice()`)
  -- visible in a normal `--verbose` run, so a grep for "using traditional"/"using AI" always
  explains what happened, including an opt-in hint (e.g. "set stages.interpolate.use_ai: true or
  pass --ai to use it") when the auto default landed on traditional.
- **`scenes.interpolate.use_ai`** (tristate, default `null`): scene-mode-only override for
  per-scene interpolation method selection. `null` resolves via the same shared
  `resolve_ai_method()` the standard `interpolate` stage uses (so both paths respond together to
  `stages.interpolate.use_ai`/`general.use_ai`, fixing scene mode previously hardcoding its method
  from `general.use_ai` alone and mapping "auto" to traditional independently of the stage);
  `true`/`false` forces AI/traditional for the scene path only.
- **`gpu.max_concurrent_inferences`** (default `1`): process-wide cap on concurrent GPU AI
  inferences, enforced by a lazily-created `threading.Semaphore`
  (`ai.torch_utils.get_gpu_inference_semaphore()`). Scene mode acquires it around the AI/RIFE
  inference call only (per-scene, when the resolved method is "ai") -- otherwise scene mode's
  thread pool (`scene_workers` threads) would launch concurrent RIFE inferences contending for
  VRAM instead of parallelizing. Whole-video (non-scene) runs execute stages serially already, so
  they're unaffected. A thread blocking on the semaphore logs at DEBUG.
- **Resolution-aware scene worker budget**: `_scene_worker_budget()` (`core/scenes.py`) was
  memory-blind, splitting only the CPU count -- even with the `parallel_chunks`-forwarding fix
  above, a 4K input could still run several concurrent multi-GB-footprint whole-scene
  `minterpolate` processes. It now takes the input's already-probed resolution and scales the
  total CPU budget down before splitting (`mem_scale = max(1.0, pixels / (2 * 1920*1080))` --
  inputs up to ~2x 1080p keep the full budget, 4K halves it, 8K quarters it again). New
  `scenes.max_workers` (default `null`): a positive int caps the resulting `scene_workers`
  explicitly without changing the underlying total budget the per-scene chunk count still derives
  from.

- **CSS-like config cascade for `avf process`, plus `--set KEY=VALUE`**: config now resolves as
  an ordered stack of layers -- `Config.DEFAULTS` < the user `config.yaml` (or an explicit
  top-level `--config`/`AVF_CONFIG` path, unchanged) < each `process --preset NAME_OR_PATH` /
  `process --config PATH` layer, applied in the order those flags actually appear on the command
  line (interleaved -- `--preset A --config B --preset C` applies A, then B, then C, not "both
  presets then the config") < one final layer folding in every other config-affecting CLI flag
  (including the new `--set`), always applied LAST regardless of where it was typed, with a
  later-typed flag beating an earlier-typed one on conflict. Each layer deep-merges onto the
  previous one (only clobbers the keys it specifies); list-valued keys (e.g.
  `pipeline.default_order`, a preset's `enable_stages`) are replaced wholesale by the last layer
  that sets them, never merged element-wise.
  - `Config` gained `apply_layer(layer, source_label)` (module-level `deep_merge()` under the
    hood, same merge semantics `_deep_update`/the `pipeline.default_order` occurrence-`config`
    override already used) and a `sources` property recording every layer's label in order, for
    DEBUG-level tracing of which layer set a given effective value. `cli.py`'s
    `_log_effective_settings()`/preset-merge call sites and `gui/main_window.py`'s
    `_on_preset_changed()` now go through this instead of the old bespoke `_merge_config()`
    helper duplicated in both files (removed).
  - `process --preset` and `process --config` are both now repeatable (`multiple=True`). `--preset`
    accepts either a registered name or a path to a preset file (JSON, via the existing
    `load_preset()`) -- disambiguated by a path separator, a `.yaml`/`.yml` extension, or the path
    existing on disk. The new per-command `process --config PATH` is distinct from the existing
    top-level `avf --config PATH process ...` flag: the top-level flag still selects *which file*
    fills the base "user config.yaml" slot (unchanged); the new one adds an *additional* layer on
    top of that.
  - `--set KEY=VALUE` (repeatable): dot-notation key, value parsed as a YAML scalar (`true`/`16`/
    `null`/quoted strings/inline lists all work). Malformed input (no `=`, empty key) is a
    `click.BadParameter` error. Secret-looking keys (`api_key`/`token`/`password`/etc., matching
    the same check `redact_secrets()` uses) are rejected -- config-file-only, to keep secrets out
    of shell history.
  - Cross-option command-line ordering (which Click itself loses -- it parses each `multiple=True`
    option into its own tuple) is recovered via a `sys.argv` scan (`_scan_process_argv()` in
    `cli.py`); if `sys.argv` doesn't actually correspond to the invocation being processed (e.g. a
    programmatic `CliRunner.invoke()` call in a test) the scan is discarded and a fixed, documented
    fallback order is used instead (all `--preset` values then all `--config` values for step 3;
    declared-option order for step 4).
  - See AGENTS.md's new "Config cascade" section and `docs/config.example.yaml`'s header for the
    full model plus worked CLI examples.
- **`pipeline.default_order` now actually drives stage order, omission, and repetition**
  (previously "informational only" -- `Pipeline.optimize_stage_order()` hardcoded the real order
  and silently ignored this config key entirely, a stale note config.py itself used to document).
  `Pipeline.resolve_stage_order()` (new; `optimize_stage_order()` becomes a thin backward-compatible
  wrapper returning a flattened `list[str]` of occurrence labels) reads `pipeline.default_order` and
  resolves each entry into a run/skip decision independently:
  - A plain stage name string behaves exactly as before (runs iff the stage is in the
    requested/auto-determined set).
  - A mapping `{stage, enabled?, config?}` adds explicit per-occurrence control: `enabled: true`
    force-runs the occurrence regardless of `stages.<name>.enabled`/preset `enable_stages`/
    auto-determination (mirroring the existing `--stage`/`explicit_stage_request` bypass
    mechanism; internal `should_run()` dependency gates still apply); `enabled: false` hard-drops
    it (the only way to drop a stage that's otherwise requested -- plain omission from the list
    still gets it appended at the end, for `--stage` compat); `enabled: null`/omitted defers to
    global gating; `config: {...}` deep-merges per-occurrence overrides onto the cascaded
    `stages.<name>` dict (`stages.<name>` &larr; `job.stage_overrides[name]` &larr; occurrence
    `config`) for that occurrence only.
  - The same stage name may repeat in the list; each occurrence resolves/runs independently and
    chains off the previous occurrence's output. Repeated occurrences get an occurrence-qualified
    label (`"deblock"`, `"deblock#2"`, ...) used for `JobResult.stage_results` keys, generated temp
    filenames, progress, and logging -- the plain name is used everywhere a stage has exactly one
    occurrence (the common case is unaffected). `pipeline.max_stages` now counts occurrences, not
    unique stage names.
  - `BaseStage.__init__` gained an optional `overrides: dict[str, Any] | None` param (deep-merged
    onto a deep-copied `self._stage_config`, never mutating the shared `Config` data), and
    `create_stage()`/every built-in stage subclass's `__init__` now threads it through, so
    `__init__`-cached config fields (e.g. `ai_model`, `tile_size`) see the fully-cascaded value.
    The deep-merge logic itself was extracted from `Config._deep_update` into a new module-level
    `autovideofixer.config.deep_merge()`, reused by both.
  - Malformed `default_order` entries (a mapping missing a string `stage` key, a non-bool/non-null
    `enabled`, a non-mapping `config`, or an entry that's neither a string nor a mapping) raise
    `ValueError` at order-resolution time, surfaced by `execute_job()` as a failed `JobResult`
    rather than an uncaught exception.
  - **`DEFAULTS["pipeline"]["default_order"]` changed**: `deblock` now runs **before**
    `stabilize` (previously the reverse, and previously moot since the config value was ignored).
    Blocking artifacts come from the source video, so deblocking before stabilization's
    perspective warping keeps the deblock model's input accurate (no warped block edges) and
    gives the stabilizer cleaner detail to track motion against. New order: `detect, deblock,
    stabilize, crop, denoise_video, upscale, interpolate, normalize_volume, normalize_audio,
    speed, hdr, encode` (also fixes `crop` having been entirely absent from `DEFAULTS`'s list,
    even though the hardcoded order always included it). The historical `default_order` typo
    (`"denoise"` instead of the registered `"denoise_video"`) is also fixed, now that the list is
    live rather than ignored.
  - See `AGENTS.md`'s "Pipeline Behavior" section and `docs/config.example.yaml`'s
    `pipeline.default_order` comment (full syntax + worked repetition example) for details, and
    `tests/unit/test_pipeline.py`'s `TestResolveStageOrder`/`TestOccurrenceAwareExecution` for
    coverage.
- **Compact SRVGG Real-ESRGAN models** (`ai/wrappers/upscale.py`'s new `SRVGGNetCompact`
  architecture, matching the official BasicSR flat `body.N` ModuleList layout so strict
  state-dict loading works against the real checkpoints): three new `MODEL_REGISTRY` entries in
  `ai/model_cache.py` -- `realesr-general-x4v3` and `realesr-general-wdn-x4v3` (`num_conv=32`,
  the latter a denoise-strength companion checkpoint; the official denoise-strength blending of
  the two state dicts is not implemented, this checkpoint is usable standalone) and
  `realesr-animevideov3` (`num_conv=16`, animation-tuned) -- all official xinntao/Real-ESRGAN
  v0.2.5.0 release assets, hash-verified. `RealESRGANUpscaler.load_model()` dispatches between
  `SRVGGNetCompact` and the existing `RRDBNet` off each registry entry's new `arch` field (a
  pure `resolve_arch()` helper; entries without it default to `"rrdb"`, so every existing model
  is unaffected). ~1.2M/~0.6M params vs RRDB's ~16.7M -- an order-of-magnitude-plus less GPU
  compute per frame (the AI stages are measured ~100% `gpu_forward`-bound) at some restoration
  quality cost; RRDB remains the default for `upscale`/`deblock`/`denoise_video`. The existing
  `RealESRGAN_x4plus` -> `RealESRGAN_x2plus` model-swap optimization in `deblock`/
  `denoise_video`/`upscale` is a strict string-equality check against the literal
  `"RealESRGAN_x4plus"`, so it does not fire for (and does not need changes to skip) the new
  compact model names. Verified (GPU): strict state-dict load of all three official checkpoints,
  correct x4 output shapes, non-black fp16 output, and the tiled-inference path.
- **Rust-backed AI frame transport (`rust/avf_framepipe/`, docs/REQUIREMENTS.md R5.3)**: a new
  PyO3/`maturin` extension providing threaded, bounded-channel ffmpeg frame I/O (`FrameReader`/
  `FrameWriter`, each a background OS thread + piped `ffmpeg` subprocess), following the same
  `[tool.uv.workspace]`/lazy-import-with-fallback pattern as `avf_scenes` (R5.1) and `avf_hashing`
  (R5.2) -- see AGENTS.md's "Mixed Python/Rust" section. A new adapter module, `ai/frame_pipe.py`,
  exposes `get_frame_reader()`/`get_frame_writer()` factories that pick the Rust backend when
  built, else fall back to thin wrappers around the existing `ai/frame_processor.py` machinery
  (`stream_frames_prefetched`/`AsyncVideoWriter`/`StreamingVideoWriter`, left unmodified) --
  exposing an identical `next_batch()`/`frames_read()`/`close()` (reader) and
  `write_batch()`/`frames_written()`/`close()` (writer) surface either way, so call sites never
  branch on backend. The `upscale`/`deblock`/`denoise_video` AI stages' chunked streaming loops
  now go through this adapter (their `StageTimer` phase instrumentation, mux/temp-file handling,
  and error paths are unchanged -- only the transport underneath `decode_wait`/`write_wait` was
  swapped). Per docs/REQUIREMENTS.md R5.3's 2026-07-14 measurement (deblock is compute-bound,
  ~0-5% transport overhead at tested resolutions), this landed as a lean v1 (no NVDEC, no
  buffer-lease pooling) -- its case is architecture robustness (real OS threads instead of
  GIL-bound Python threading, bounded memory, diagnosable ffmpeg errors), not a throughput win
  for existing GPU-bound workloads. New config keys `stages.{upscale,deblock,denoise_video}.
  read_ahead` (default `2`) and `.write_queue_depth` (default `4`) control the reader/writer's
  backpressure depth (only the Rust backend's channels actually vary with these -- the Python
  fallback's writer queue depth is a fixed constant, see `ai/frame_pipe.py`'s docstring).
  `interpolate`'s AI/RIFE path is unchanged (still uses `frame_processor.py` directly, not this
  adapter). Verified: a differential identity test proves the Rust and Python transports are
  bit-exact-interchangeable under identical passthrough processing and encode settings
  (`tests/unit/test_frame_pipe.py::TestTransportIdentity`), and a 1200-frame soak test confirms
  bounded (not full-video) memory growth streaming through the reader/writer pair
  (`TestFramePipeSoak`).
- **`stages.stabilize.zoom_coverage`** (float, 0.0-1.0, default `1.0`): tunes how aggressively the
  stabilize stage's zoom compensates for stabilization-introduced borders, once
  `zoom_enabled`/`zoom_threshold`'s movement-extent gate has decided zoom applies at all (the gate
  itself is unchanged). `1.0` = today's exact `optzoom=1` behavior (zoom sized to the single worst
  frame -- guaranteed no border on any frame). `0.0` = no zoom at all (every border stays
  visible). In between: `optzoom=0` plus a static `zoom=<pct>` computed as the `zoom_coverage`-th
  quantile of per-frame required-zoom estimates (`StabilizeStage._compute_static_zoom_pct()`,
  built from the already-parsed TRF local-motion data plus a local moving-average matching
  `smoothness`) instead of the max -- trades the "no border, ever" guarantee for a less aggressive
  crop with occasional brief borders during the most extreme motion (e.g. a violent-motion
  ending), addressing real user feedback that `optzoom=1` (added to fix the opposite problem --
  see the "Stabilize zoom could only zoom OUT, never in" fix below) now over-zooms and crops
  content that should stay visible, when some borders would be an acceptable tradeoff. This is an
  **approximation** of vidstabtransform's own internally-computed smoothed camera path, not an
  exact read of it (stated explicitly in code comments/AGENTS.md); verified via a synthetic shaky
  clip with cropdetect: `coverage=1.0` shows zero border on any sampled frame, `coverage=0.0`
  reproduces the fully-unzoomed baseline's borders, and `coverage=0.6`'s computed zoom quantile is
  measurably smaller than the same function's `coverage=1.0` quantile (10.4% vs. 25.5% on the test
  clip) -- see AGENTS.md's "Stabilization zoom coverage" section for the full verification
  writeup and the accuracy caveat. CLI: `--zoom-coverage FLOAT`.
- **Rust perceptual-hash duplicate detection (`rust/avf_hashing/`, docs/REQUIREMENTS.md R5.2)**:
  `compute_video_hash()`/`compute_video_dhash()`/`hash_similarity()` (`core/analysis.py`) --
  Python-loop-heavy ahash/dhash implementations backing `avf find-duplicates` -- are replaced
  wholesale by `compute_video_phash()`/`hash_similarity()`, a DCT-based **pHash** implementation
  backed by a new PyO3/`maturin` extension (`avf_hashing`), following the exact same
  `[tool.uv.workspace]`/lazy-import-with-fallback pattern as `avf_scenes` (R5.1) -- see AGENTS.md's
  "Mixed Python/Rust" section. This feature has never been used in production (confirmed with the
  user 2026-07-12), so there was no bit-for-bit compatibility constraint; pHash was chosen over a
  literal ahash/dhash port because it hashes low-frequency 2D DCT coefficients rather than raw
  pixel values, making it substantially more robust to the re-encodes/resolution changes/trims
  that make up real-world near-duplicate video variation (see `rust/avf_hashing/src/lib.rs`'s
  module docstring for the full rationale, including why the `img_hash` crate was evaluated and
  rejected in favor of a small hand-rolled implementation). Frames are sampled the same way as
  before (`num_frames`, default 30, evenly spaced), pHashed individually, and combined into one
  video-level hash via majority-vote bit combination. The pure-Python fallback (used when the
  compiled extension isn't available) implements the *identical* pHash algorithm in NumPy, not the
  retired ahash/dhash, so results no longer depend on which path runs. `analysis.duplicate_
  detection.hash_type` (`perceptual`/`dhash`/`combined`) is removed from `config.py`/`docs/
  config.example.yaml` -- it's no longer meaningful with a single algorithm -- and `analysis.
  duplicate_detection.similarity_threshold`'s default drops from `0.95` to `0.85`, recalibrated
  against real measured near-duplicate/non-duplicate pHash similarity scores (see `tests/unit/
  test_hashing.py::TestHashSeparation`'s docstring for the actual numbers: near-duplicate pairs --
  same source re-encoded at a different CRF/resolution or trimmed -- scored >= 0.9375; distinct
  lavfi sources, including cross-comparing each source's own near-duplicate variants, scored <=
  0.5781). Verified with real ffmpeg-generated fixtures, not just unit-level hash math.
- **Rust scene-detection extension (`rust/avf_scenes/`, docs/REQUIREMENTS.md R5.1)**: the first
  Rust code in this project. `_detect_scene_changes()`'s per-frame decode/diff loop (`core/
  analysis.py`) is now backed by a PyO3/`maturin` extension (`avf_scenes`) that pipes video
  through `ffmpeg -f rawvideo` (grayscale, downscaled to 320x180) instead of `cv2.VideoCapture`,
  computes the same mean-absolute-luma-diff `diff_score` metric, and releases the GIL for the
  whole decode/diff loop. `avf_scenes` is a normal (non-optional) dependency built automatically
  by `uv sync` via a `[tool.uv.workspace]` member at `rust/avf_scenes/` -- see AGENTS.md's Setup &
  Commands. If the compiled extension isn't importable for any reason, `_detect_scene_changes()`
  falls back to the original pure-Python/OpenCV implementation (kept in place, unchanged, as
  `_detect_scene_changes_python`) with a DEBUG log line. Verified via a new differential test
  (`TestRustPythonParity`, `tests/unit/test_scene_detection.py`) that the Rust and Python paths
  find identical scene boundaries on the calibration fixture and a synthetic motion+cut clip, with
  per-cut `diff_score` divergence under 0.002 (decode/scale path rounding differences between
  ffmpeg's `scale`+`format=gray` and OpenCV's `resize`+`cvtColor`). Real-world speed on a
  54s/1620-frame 1080p60 clip: ~1.5x wall-clock (2.55s Python vs. 1.73s Rust; ~635 vs. ~936 fps).
- **Batched Real-ESRGAN inference (`upscale`/`deblock`/`denoise_video`, torch backend)**: two new
  opt-in (default `1` = today's behavior, unchanged) per-stage config keys,
  `stages.<name>.batch_size` and `stages.<name>.tile_batch_size`, plus matching `--batch-size`/
  `--tile-batch-size` CLI flags on `avf process`. Root cause: `RealESRGANUpscaler`/
  `FrameProcessor` accepted a `batch_size` constructor arg that was never actually read anywhere
  -- every frame (and, when tiling, every tile) went through the model one at a time. Two
  independent batching axes were added:
  - `tile_batch_size` batches N tiles of the SAME frame (sharing `compute_tile_grid`'s padded
    input shape) into one forward pass inside `run_tiled_inference()` (`ai/wrappers/upscale.py`).
    This is the primary real-world fix: a 4K (3840x2160) input always takes the tiled path at the
    default `tile_size=512` (5x8=40 tiles/frame), so a multi-hour `deblock` job stuck on a real 4K
    video was paying 40 small sequential forward passes per frame, each with its own Python/
    kernel-launch/host-device-sync overhead -- confirmed via `nvidia-smi` showing GPU utilization
    pinned near 100% (no idle time for I/O overlap) alongside `nvtop`-reported SM occupancy of
    only ~70-80%, the classic signature of small serialized kernel launches.
  - `batch_size` batches N different frames into one forward pass
    (`RealESRGANUpscaler.upscale_batch()`/`upscale_video()`, plus new
    `tensor_from_frames`/`frames_from_tensor` batched-tensor helpers in `ai/torch_utils.py`,
    siblings of the existing single-frame `tensor_from_frame`/`frame_from_tensor`). Secondary to
    tile batching -- only helps frames small enough to skip tiled inference in the first place.
  - Both axes fall back to today's one-at-a-time processing for tiling+whole-frame-batching
    combinations, TTA (`tta_mode >= 1`, a different batching axis -- 8 augmented passes of ONE
    item), and the ncnn backend; on a caught `torch.cuda.OutOfMemoryError` the batch is
    recursively halved and retried (mirroring the existing tile-size OOM-retry pattern), down to
    a single item which then goes through the existing proven single-item OOM/tiling-retry path
    unchanged.
  - Verified: order preservation (distinguishable per-frame content), non-evenly-divisible tail
    chunking (e.g. 10 frames / batch_size=4 -> 4,4,2), and TTA/tiling fallback all covered by new
    mocked-torch unit tests (`tests/unit/test_ai_wrappers.py`, `tests/unit/test_torch_utils.py`);
    numeric equivalence against the existing one-at-a-time path is covered by new GPU-gated
    integration tests (skip without a cached checkpoint or CUDA) asserting max-abs-diff <= 12
    uint8 levels at batch_size/tile_batch_size 4 and 8 vs 1 -- ran once against real hardware
    (passed) but full A/B throughput benchmarking against a real 4K clip is still pending: the
    user's own multi-hour production job was actively using the GPU (`nvidia-smi
    --query-compute-apps` non-empty, ~100% utilization) throughout this change, so the timed
    before/after benchmark was deliberately deferred rather than contend with it -- see
    `docs/REQUIREMENTS.md`'s R5.3 note for the full context and re-run instructions.
- **`stages.{upscale,deblock,denoise_video,interpolate}.temp_crf`** (default `16`): CRF for
  each AI stage's internal temp-file encode (the chunked write to a temp `.mp4` before the
  final mux pass), now passed explicitly instead of the temp encode getting no `-crf`/`-preset`
  at all (silently landing on libx264's own default, CRF 23). The mux pass's preset is a fixed
  `"medium"` (not itself configurable). See "Fixed" below for why this matters (double-encode
  quality bug).
- **AI-stage per-phase timing + periodic throughput instrumentation** (`ai/frame_processor.py`'s
  `StageTimer`/`gpu_forward_timer()`, threaded through `RealESRGANUpscaler.upscale()`/
  `upscale_batch()`/`upscale_video()` via an optional `timer=` kwarg): the chunked AI loop shared
  by `upscale`/`deblock`/`denoise_video` now times `decode_wait`, `h2d_preprocess`, `gpu_forward`
  (honest `torch.cuda.Event`-bracketed device time on CUDA, wall-clock fallback otherwise),
  `d2h_postprocess`, and `write_wait` per chunk (DEBUG) plus an aggregate percentage breakdown at
  stage end (INFO). Independently, INFO-level throughput (`frames_done`, `elapsed_sec`,
  `current_fps`) logs every ~10s/10 chunks regardless of the phase breakdown, specifically so a
  run killed early by an external timeout still yields a valid frames/sec curve -- see
  `docs/REQUIREMENTS.md`'s R5.3 note on the 4K benchmark this was retroactively needed for. No
  behavior change; always-on but cheap (`time.perf_counter()`/CUDA events only).

### Changed
- **`frame_from_tensor`/`tensor_from_frame` (`ai/torch_utils.py`) now do their elementwise
  pre/post-processing on the GPU tensor instead of CPU numpy**, moving the device transfer to the
  end (post-process) or start (pre-process) of each conversion instead of the middle. Found via
  live `py-spy` profiling of a real upscale run: the CPU-side numpy postprocessing (NaN sanitize,
  scale, clip, uint8 cast, channel reorder) was pinning a full CPU core between GPU inference
  calls, starving the GPU (high utilization%, low effective occupancy). Bit-exact (postprocess)
  / ULP-level-equivalent (preprocess, ~5.96e-8 max diff -- ordinary GPU-vs-CPU float rounding,
  not a precision loss) output verified against the old implementation before landing, including
  NaN/Inf/out-of-range edge cases. Real measured impact (old-vs-new A/B, RTX 5060 Ti): `upscale`
  stage ~11% faster end-to-end (a 2304x1280 output frame's postprocessing alone dropped from
  41.4ms to 2.9ms in isolation); `denoise_video`/`deblock` (scale=1, native-resolution output)
  ~4% faster -- a smaller win since the postprocessing array is 16x smaller at native resolution
  than at a 4x-upscaled output. See `docs/REQUIREMENTS.md`'s R5.3 for the follow-up: the
  remaining bottleneck is architectural (frames processed fully serially, so the existing
  prefetch/write helper threads have nothing to overlap with a frame's GPU-sync wait), not
  further op-level tuning.
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
- **Upscale stage ran a full AI pass on a video already at its target resolution** — a portrait
  input already exactly at its orientation-rotated target (e.g. 1080x1920 against a `1080p60`
  preset's `[1920, 1080]` target, which rotates to 1080x1920 for a portrait input) was not
  recognized as "already at target" and ran a wasted `RealESRGAN_x2plus` pass, producing a
  2160x3840 output (~1fps, over an hour for a 63-second clip; confirmed via GPU-forward
  instrumentation as ~99% wasted compute). Root cause was three separate spots comparing the input
  against the *unrotated* preset target instead of the orientation-aware rotated target: `Upscale
  Stage.should_run()`'s "already at target?" check, `execute()`'s AI-vs-traditional method
  selection, and — the actual mechanism that produced exactly a 2x output — `_execute_ai()`'s
  per-pass scale-factor rounding (`2 ** round(log2(sf)) if sf > 1 else 2`), which silently
  substituted a forced 2x scale whenever a pass's own needed scale computed to `<=1` (already
  at/past target) instead of doing nothing. Fixed at all three levels via a shared
  `_effective_target_bounds()` helper and a new `_SKIP_SCALE_THRESHOLD = 1.05` (5% linear, ~10%
  area — chosen to also cover the "crop stage shaved a few px off before upscale ran" case, e.g.
  1072x1908 vs. a 1080x1920 rotated target, without masking a genuine upscale need): at/below the
  threshold, the stage now skips (already exact) or does a cheap lanczos resize (near target, not
  exact) instead of ever running a wasted AI pass. See AGENTS.md's "Upscaling & Aspect Ratio"
  section for the full root-cause writeup.
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
- **Double lossy encode in every AI stage** (`upscale`/`deblock`/`denoise_video`/`interpolate`):
  each AI stage wrote its internal temp file with no `-crf`/`-preset` (silently libx264's default,
  CRF 23), then re-encoded that temp file again during the mux pass at `-crf 18` -- two lossy
  generations plus a wasted full x264 pass over the whole video for no quality benefit. Fixed by
  giving the temp encode an explicit `-crf <stages.<name>.temp_crf>` (new config key, default 16)
  `-preset medium`, and switching every AI stage's mux pass to `-c:v copy` (stream-copy, since the
  temp file is now the only real encode). Verified: mux output's video stream is bit-identical to
  the temp file's video stream (ffmpeg stream-hash comparison, `-map 0:v -c copy -f md5`) across
  `deblock`/`denoise_video`/`upscale`, all against a real ffmpeg-generated clip through the actual
  AI path (cached `RealESRGAN_x2plus` checkpoint, not mocked).
- **Tile-batch OOM retry held the failed batch tensor alive across the halving recursion**
  (`run_tiled_inference()`'s `_infer_group`, `ai/wrappers/upscale.py`): every retry after a caught
  `torch.cuda.OutOfMemoryError` ran with LESS free VRAM than the attempt that had just failed,
  instead of getting back the memory that attempt would have freed, because `batched_in` (and,
  implicitly, `infer_fn`'s own reference to it via the live exception traceback) stayed referenced
  through the whole recursion. Fixed by moving the `del batched_in` + cache-clear + recursion
  OUTSIDE the `except` clause (an in-flight exception's traceback keeps the raising frame's locals
  alive for as long as the `except` block is still executing, so a `del` inside it doesn't
  actually drop the reference yet). Also: remainder tiles in a shape group (fewer than
  `tile_batch_size` left over) now go through the single-tile path instead of one last
  partial-size batch, avoiding a batched-tensor shape that differs from every other call in the
  run and fragments the CUDA caching allocator. Verified via a new mocked-torch regression test
  (`tests/unit/test_ai_wrappers.py::TestRunTiledInference::
  test_tile_batching_oom_releases_failed_batch_before_retry`) that weakref-tracks the failed batch
  tensor and asserts it's already released by the time `torch.cuda.empty_cache()` runs -- confirmed
  to actually fail against the pre-fix code (moving the `del` back inside the recursion makes the
  test fail as expected).
- **≤1000-frame videos buffered ALL frames into RAM before AI inference**
  (`upscale`/`deblock`/`denoise_video`): a frame-count-only threshold (`total_est > 1000`) was
  resolution-blind, so a short but large-resolution clip (e.g. a 33s 4K clip, ~25GB uncompressed)
  could still fall under 1000 frames while materializing its entire frame set via
  `extract_frames()` before a single inference call, with zero decode/inference/write overlap. All
  three stages now go through the chunked streaming path unconditionally (`use_chunked` branch and
  its full-buffer `extract_frames()`/`frames_to_video()` route removed); streaming has no
  measurable downside for short clips. Verified: a 30-frame real clip end-to-end through each of
  the three stages' AI path still produces valid, correctly-encoded output via the streaming path.

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
