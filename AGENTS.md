# Auto Video Fixer - Agent Instructions

## Standing rule: keep docs in sync

Any change to behavior, config keys, CLI flags, or stage semantics MUST update
`AGENTS.md`, `CHANGELOG.md`, `docs/` (`ROADMAP.md`, `IMPLEMENTATION.md` as applicable), and
`docs/config.example.yaml` in the **same** change. Docs that drift from the code are worse than
no docs — the next agent (or human) will trust and propagate a stale claim.

## Setup & Commands

```bash
uv sync --all-extras        # install core + dev deps
uv run pytest tests/unit/ -v   # unit tests (no FFmpeg needed by default)
uv run pytest tests/ -v -m "integration"  # integration tests (requires FFmpeg)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ --ignore-missing-imports
uv run avf process video.mp4 -p 1080p60   # CLI entry
```

CI runs: `ruff check` -> `ruff format --check` -> `mypy` -> `pytest tests/unit/` -> `pytest -m integration` (3.14 only).

### Mixed Python/Rust: this is now a mixed-language project

`rust/avf_scenes/` is a PyO3/`maturin` Rust crate (scene-detection frame differencing, see
`docs/REQUIREMENTS.md` R5.1) — the first of the Rust rewrite candidates to land, and the pattern
future Rust work here should follow. It is a `[tool.uv.workspace]` member of the root
`pyproject.toml`, declared as a normal dependency (`avf-scenes`, `[tool.uv.sources]` pointing at
it as an editable workspace path) — **plain `uv sync --all-extras` builds and installs it
automatically**, no separate manual step needed. `rust/avf_scenes/pyproject.toml`'s
`build-system.requires = ["maturin>=1.14,<2.0"]` means uv fetches `maturin` itself into an
isolated PEP 517 build environment on demand (standard build-isolation behavior) — it does **not**
need to be pre-installed in the project's own venv for `uv sync` to work. The one thing that *does*
need to be present on `PATH` at sync time is a Rust toolchain (`cargo`/`rustc`, e.g. via `rustup`
or a system package) — `maturin` shells out to `cargo` to actually compile. If it's missing, `uv
sync` fails outright rather than silently skipping the extension, since `avf-scenes` is a required
dependency of the package, not an optional extra, matching REQUIREMENTS.md's R5.1 integration
plan. (For manual iteration, `uv pip install maturin` puts a runnable `maturin` CLI in the venv --
see the "iterate on the Rust side" note below -- but that's a convenience for `maturin develop`,
not something plain `uv sync` requires.)

At the Python call site, `core/analysis.py`'s `_detect_scene_changes()` still tries `import
avf_scenes` in a `try/except ImportError` and falls back to the original pure-Python/OpenCV
implementation (`_detect_scene_changes_python`) if that import fails for any reason (e.g. a
platform where the wheel didn't build) — this is a defense-in-depth fallback, not the primary way
the extension is expected to be missing day-to-day, since `uv sync` normally guarantees it's
built. `uv run pytest tests/unit/` does **not** need a separate Rust build step beyond the normal
`uv sync` — if you've run that, the extension is already installed into the venv.

To iterate on the Rust side without a full `uv sync` round-trip: `cd rust/avf_scenes && uv run
maturin develop --release` rebuilds and reinstalls just that crate into the current venv.
`cargo test` / `cargo clippy` / `cargo fmt --check` run from `rust/avf_scenes/` for Rust-side
gates (no Rust unit tests exist yet for this crate — correctness is verified via the Python-side
differential test `TestRustPythonParity` in `tests/unit/test_scene_detection.py`, which compares
the Rust and Python implementations' output directly).

A second crate, `rust/avf_hashing/` (perceptual-hash duplicate detection, `docs/REQUIREMENTS.md`
R5.2), followed the same workspace/build pattern (`[tool.uv.workspace]` member, `avf-hashing`
dependency + `[tool.uv.sources]` editable-workspace entry, built automatically by `uv sync
--all-extras`) with one structural difference worth knowing about: **it has real Rust unit tests**
(`cargo test` — DCT-matrix orthonormality, deterministic/distinguishing hashes, majority-vote
tie-breaking), unlike `avf_scenes`. A pyo3 crate built with the `extension-module` feature can't
link into a normal `cargo test` binary (that feature assumes libpython is provided by an embedding
CPython process, not the test harness itself) — undefined-symbol linker errors result if you try.
`rust/avf_hashing/Cargo.toml` works around this by making `extension-module` a Cargo feature
that's default-on (so `cargo build`/`maturin develop`/`uv sync` all still produce a real Python
extension) but can be turned off for testing: `cd rust/avf_hashing && cargo test
--no-default-features`. Do the same for `cargo clippy --no-default-features` if you touch that
crate's tests. `avf_scenes` doesn't need this because it has no Rust-level tests, only the
Python-side differential test described above — if it ever grows `cargo test` cases, apply the
same feature-flag split there too, for consistency.

At the Python call site, `core/analysis.py`'s `compute_video_phash()`/`hash_similarity()` follow
the identical lazy-import-with-fallback pattern as scene detection: `try: from avf_hashing import
...` / `except ImportError`, falling back to a pure-NumPy implementation of the *same* pHash
algorithm (not a different one) so results don't depend on whether the extension is built. See
that module's "Perceptual Hashing & Duplicate Detection" section and `rust/avf_hashing/src/lib.rs`
for the algorithm and why it replaced the old ahash/dhash implementation wholesale rather than
sitting alongside it (no backward-compatibility constraint — see R5.2 in
`docs/REQUIREMENTS.md`).

A third crate, `rust/avf_framepipe/` (threaded, bounded-channel ffmpeg frame I/O for the AI stage
loop, `docs/REQUIREMENTS.md` R5.3), followed the same workspace/build pattern (`[tool.uv.workspace]`
member, `avf-framepipe` dependency + `[tool.uv.sources]` editable-workspace entry, built
automatically by `uv sync --all-extras`). Unlike `avf_scenes`/`avf_hashing`, which each expose one
function replacing a hot loop, `avf_framepipe` exposes two classes, `FrameReader` and
`FrameWriter` (see `rust/avf_framepipe/src/lib.rs`'s module docs for exact signatures): each owns a
background OS thread plus a piped `ffmpeg` subprocess, handing frames across a bounded
`std::sync::mpsc::sync_channel` so a Python-side inference loop can pull/push frames without ever
blocking on the ffmpeg pipe directly. `FrameReader(path, width, height, chunk_size, read_ahead,
ffmpeg_path).next_batch() -> list[np.ndarray] | None` decodes rawvideo BGR24 frames in
`chunk_size`-sized batches, buffering at most `read_ahead` batches ahead of the consumer.
`FrameWriter(path, width, height, fps, codec, crf, preset, write_queue,
ffmpeg_path).write_batch(list)` mirrors `StreamingVideoWriter`'s ffmpeg argument shape (including
not setting `-movflags faststart`). Both have `.close()` (writer's returns `bool` success) that are
idempotent and safe to call from `Drop` as a backstop.

At the Python call site, `ai/frame_pipe.py`'s `get_frame_reader()`/`get_frame_writer()` factories
follow the identical lazy-import-with-fallback pattern as scene detection/hashing: `try: import
avf_framepipe` / `except ImportError`, falling back to thin Python wrapper classes
(`_PythonFrameReader`/`_PythonFrameWriter`) around the *existing* `ai/frame_processor.py` machinery
(`FrameProcessor.stream_frames_prefetched()` for reading, `AsyncVideoWriter(StreamingVideoWriter(...))`
for writing) — exposing the identical `next_batch()`/`frames_read()`/`close()` (reader) and
`write_batch()`/`frames_written()`/`close()` (writer) surface either way, so `upscale`/`deblock`/
`denoise_video`'s stage loops never branch on backend. `frame_processor.py` itself is **untouched**
by this — it's still the fallback implementation and still directly usable by anything that doesn't
need transport-backend selection (e.g. `interpolate`'s AI/RIFE path, which still buffers frames in
a list rather than streaming — see "Kill the ≤1000-frame full-buffering path" below). Two fallback
parity gaps worth knowing: `write_queue`/`write_queue_depth` only actually varies the Rust
backend's channel bound (`AsyncVideoWriter`'s queue depth is a hardcoded constant `frame_processor.py`
wasn't modified to expose); and the Python fallback's `frames_written()` counts frames as they're
*enqueued* rather than as they're actually flushed to the ffmpeg pipe (both converge to the same
final count once `close()` returns). New config keys `stages.<name>.read_ahead` (default `2`) and
`stages.<name>.write_queue_depth` (default `4`) on `upscale`/`deblock`/`denoise_video` plumb into
these factories; `chunk_size` (`25`) stays a call-site constant, not a config key.

A fourth crate, `rust/avf_borders/` (per-edge, arbitrary-color border/letterbox detection for the
`crop` stage), followed the same workspace/build pattern (`[tool.uv.workspace]` member,
`avf-borders` dependency + `[tool.uv.sources]` editable-workspace entry, built automatically by `uv
sync --all-extras`) and the same `extension-module` default-on Cargo feature split as
`avf_hashing`/`avf_framepipe` (`cargo test --no-default-features` from `rust/avf_borders/` for a
linkable test binary; `cargo clippy --no-default-features -- -D warnings` / `cargo fmt --check` for
the rest of the Rust-side gates). It exists because FFmpeg's `cropdetect` filter (used by the rest
of the `crop` stage, see "Auto-crop" below) is luma-threshold-only -- white, gray, or otherwise
colored borders are invisible to it, only black/dark ones. `avf_borders` exposes one function,
`detect_border_frames(path, ffmpeg_path, sample_fps, strip_px, tolerance, majority, solidity_min,
ffprobe_path="ffprobe") -> list[dict]`: per sampled frame and per edge (top/bottom/left/right), it
computes the DOMINANT color of the outermost `strip_px`-deep strip (4-bit-per-channel quantization,
largest bin's mean actual color -- robust to slight gradients/compression noise) plus a "solidity"
percentage (fraction of the strip matching that color within `tolerance`, using max-per-channel-
absolute-difference as the distance metric), then -- only if solidity is >= `solidity_min` (default
0.60, keeps blurred-video-background pillarboxing, a real but non-uniform edge, from being cropped
in v1) -- walks inward line by line while >= `majority` (default 0.90) of each line still matches,
capped at 45% of that edge's dimension (never deeper -- a "border" spanning close to half the frame
isn't a border; without this cap a solid-color transition frame would report a near-full-depth
border on every edge). Self-contained like `avf_scenes`/`avf_hashing`: dimensions and fps are
probed internally via `ffprobe`, not passed in from Python. Unlike `avf_scenes`/`avf_hashing`
(which silently return empty/zero on any ffmpeg failure), this crate's contract requires a real
diagnosable error -- ffmpeg spawn failure, non-zero exit, or a truncated mid-frame read all raise
`RuntimeError` in Python with an ffmpeg stderr tail (captured on a background thread, same
`StderrTail` shape as `avf_framepipe`, not discarded to `Stdio::null()` like the other two crates);
the child is always reaped (no zombies) and a clean zero-frame decode (exit 0) returns an empty
list rather than erroring. Has real Rust unit tests (5 ffmpeg-lavfi-generated-fixture cases per the
spec: black letterbox, white letterbox -- the case cropdetect literally cannot detect --, a small
(~8%-width) logo that the default `majority=0.90` tolerates without stopping the walk, the same
large (~25%-width) logo tested at both the default majority (walk stops at the logo) and an
explicitly relaxed `majority=0.70` (walk tolerates it and reaches the true border), and a
corrupt/nonexistent input erroring without a hang or zombie -- plus small pure-function tests for
the quantization/distance helpers).

At the Python call site, `core/stages/crop.py`'s `CropStage` follows the identical lazy-import-
with-fallback pattern: `try: from avf_borders import detect_border_frames as
_detect_border_frames_rs_native` / `except ImportError`, logged at DEBUG on failure. New
`stages.crop.detector` config key (`"auto"` (default) | `"rust"` | `"cropdetect"`) picks which
per-frame detector `execute()` uses -- `"auto"` resolves to `"rust"` when the extension is
importable, else falls back to `"cropdetect"`; an explicit `"rust"` that turns out unavailable also
falls back, but logs a WARNING instead of silence (the user asked for it by name). The rust
detector's per-frame `(t, x, y, w, h)` rects feed the SAME `aggregate_crop_windows()` (transition-
exclusion + union) the cropdetect path uses -- see "Auto-crop" below -- only the per-frame detection
step differs; a rust decode failure (`RuntimeError`) falls back to the cropdetect path for that run
(fail-open, same spirit as the VLM check), logged at WARNING. `should_run()`'s cheap prefilter is
luma-threshold-only (it always uses the old single-pass `cropdetect`, cheap specifically because
it's the least-precise check available) -- when the resolved detector isn't `"cropdetect"`, a
white/colored-bordered video would look like "no border" to that prefilter and get skipped before
the rust pass (which WOULD see it) ever runs, so `should_run()` skips the prefilter entirely and
returns `True` whenever the resolved detector != `"cropdetect"`, deferring to `execute()`'s full
rust pass -- simpler than teaching the quick sample to run a bounded rust pass itself, and
acceptable because the quick sample's whole point is staying cheap, not being authoritative (same
tradeoff `should_run()` already accepts elsewhere in this stage).

## Architecture

```
src/autovideofixer/
├── cli/cli.py              # Click CLI (`avf` entry, pyproject.scripts)
├── config.py               # Config + DEFAULTS
├── core/
│   ├── pipeline.py         # Pipeline orchestrator
│   ├── analysis.py         # VideoAnalyzer, VLM, scene detection, clip extraction, dedup
│   ├── ffmpeg_utils.py     # FFmpeg wrappers, probe, run
│   ├── presets.py          # Preset definitions (1080p60, 4k60, etc.)
│   └── stages/             # Processing stages
│       ├── base.py         # BaseStage, register_stage(), StageResult
│       ├── detect.py       # "detect" (analysis, priority=1, produces_output=False)
│       ├── stabilize.py    # "stabilize" — pipes raw YUV420P (decode→transform)
│       ├── crop.py         # "crop" (auto-crop, opt-in, off by default)
│       ├── deblock.py      # "deblock"
│       ├── denoise_video.py# "denoise_video"
│       ├── upscale.py      # "upscale" (AI or traditional)
│       ├── interpolate.py  # "interpolate" (AI or traditional)
│       ├── normalize_audio.py # "normalize_audio" + "normalize_volume" (two classes)
│       ├── encode.py       # "encode" (final stage)
│       ├── remux.py        # "remux" (only added for MKV input)
│       ├── speed.py        # "speed"
│       └── hdr.py          # "hdr"
```

FFmpeg must be in PATH. Verify with `avf gpu-info`.

## Stage Development

1. Create `src/autovideofixer/core/stages/<name>.py`. Subclass `BaseStage`.
2. Set class attributes: `name`, `display_name`, `description`, `category`, `priority`, `supports_gpu`.
3. Implement `should_run(input_info) -> (bool, reason | None)` and `execute(input_path, output_path, progress_callback, **kwargs) -> StageResult`.
4. Use `self._stage_config = config.get("stages", self.name, default={})` for stage config.
5. Report progress via `self._report_progress(0.0..1.0, message, callback)`.
6. **Register** in `src/autovideofixer/core/stages/__init__.py`: import and call `register_stage(MyStage)` at module level.
7. Add config to `DEFAULTS["stages"][<name>]` in `src/autovideofixer/config.py`.

## Pipeline Behavior

- **Stage ordering, omission, and repetition are driven by config**:
  `Pipeline.resolve_stage_order()` (`core/pipeline.py`, called by both `optimize_stage_order()`
  and `execute_job()`) reads `config.get("pipeline", "default_order")` and resolves it into an
  ordered list of `StageOrderEntry` occurrences. The old hardcoded-and-ignored-config behavior is
  gone -- `DEFAULTS["pipeline"]["default_order"]` (and a matching `DEFAULT_STAGE_ORDER` constant
  in `core/pipeline.py`, used only as a fallback when the config key is missing/empty) now
  **is** the actual execution order. Default order:
  `detect, deblock, stabilize, crop, denoise_video, upscale, interpolate, normalize_volume,
  normalize_audio, speed, hdr, encode`. Two changes from the previous hardcoded order:
  - `deblock` now runs **before** `stabilize` (previously the reverse): blocking artifacts come
    from the source video, so deblocking before stabilization's perspective warping keeps the
    deblock model's input accurate (no warped block edges), and gives the stabilizer cleaner
    detail to track motion against.
  - `crop` sits right after `stabilize` and before every other enhancement stage: stabilize's
    zoom-out correction can itself add a black border, so cropping after it removes both the
    original letterboxing/pillarboxing AND any residual stabilization border in one pass; running
    it before denoise/upscale/interpolate/encode means none of those (especially the AI-capable
    ones) spend compute on pixels about to be cropped away.

  Each `default_order` entry is either a plain stage name string (behaves exactly as before:
  this occurrence runs iff the stage is in the requested/auto-determined stage set) or a mapping
  `{stage, enabled?, config?}` for explicit per-occurrence control:
  - `enabled: true` forces the occurrence to run regardless of `stages.<name>.enabled`, preset
    `enable_stages`, or auto-determination membership (`should_run()`'s own internal
    dependency/sanity gates still apply -- this only bypasses the config `enabled` flag, mirroring
    the existing `explicit_stage_request`/`_force_enabled` mechanism `--stage` already uses).
  - `enabled: false` means this occurrence **never** runs -- this is the **only** way to
    hard-drop a stage that's otherwise in the requested set. Omitting a stage from
    `default_order` entirely does **not** drop it: for plain-`--stage` compatibility, any
    requested stage never mentioned anywhere in `default_order` (as a string or inside a
    mapping's `stage` key) is still appended at the end. `encode` always stays last regardless.
  - `enabled: null`/omitted defers to global gating, identical to a plain string entry.
  - `config: {...}` deep-merges onto the cascaded `stages.<name>` dict for **that occurrence
    only** (order: `stages.<name>` &larr; `job.stage_overrides[name]` &larr; this occurrence's
    `config`), via `BaseStage.__init__`'s optional `overrides` param -- see `core/stages/base.py`.

  The same stage name may appear more than once in `default_order`; each occurrence resolves
  and runs independently (in list order), chaining off the previous occurrence's output like any
  other stage. A repeated stage's second+ occurrence is temp-filed/logged/keyed in
  `JobResult.stage_results` under an occurrence-qualified label (`"deblock"`, `"deblock#2"`, ...)
  -- the plain stage name is used everywhere a stage has just one occurrence (the common case),
  so this is invisible unless `default_order` actually repeats a name. `pipeline.max_stages`
  counts occurrences, not unique stage names. Malformed `default_order` entries (a mapping
  without a string `stage` key, a non-bool/non-null `enabled`, a non-mapping `config`, or an
  entry that's neither a string nor a mapping) raise `ValueError` at order-resolution time,
  surfaced by `execute_job()` as a failed `JobResult` (not an uncaught exception). See
  `docs/config.example.yaml`'s `pipeline.default_order` comment for the full syntax and a worked
  example, and `tests/unit/test_pipeline.py`'s `TestResolveStageOrder`/
  `TestOccurrenceAwareExecution` for behavioral coverage.
- **Stage name mismatch (historical, fixed)**: `default_order` used to list `"denoise"` while the
  registered stage name is `"denoise_video"` -- back when the config list was ignored entirely,
  this typo was harmless. Now that `default_order` is live, `DEFAULTS` was audited and uses the
  correct `"denoise_video"` name; a raw `"denoise"` entry in a *user* config would simply never
  match any requested stage (silently a no-op, not an error, since unmatched plain-string entries
  aren't validated against the stage registry until `execute_job()`'s "Unknown stage" warn+skip
  pass -- which only fires for entries that actually resolve to a running occurrence).
- `remux` is **not** in the default pipeline. It is only added by `auto_determine_stages()` when the input is MKV.
- `detect` stage has `priority=1` (runs first) and `produces_output=False`.
- Default: `skip_stage_on_error: true` — pipeline continues on failure using the original input.
- **Temp files always use `.mkv`** (`generate_temp_path()` in `ffmpeg_utils.py`), regardless of
  the input or final output container — intermediate stages hardcode `libx264`, which not every
  container (e.g. WebM) permits, and MKV can hold essentially any codec. Written to
  `general.temp_dir` if set, else next to the input file. Some AI stages additionally use their
  own stage-internal `.mp4` temp file for the frame-extract/mux round trip (see
  `ai/wrappers/*`/stage `_execute_ai()` methods) — unrelated to the pipeline-level `.mkv` temps.
- **Final output extension** comes from `general.output_container` (default `"mp4"`), not from
  the input's extension — set via `Pipeline.add_job()` when no explicit `output_path` is given.
  `None`/empty keeps the input's own extension.
- **Temp cleanup is unconditional**, including on failure/cancel (`_cleanup_temp_files` /
  `_cleanup_generated_temp_paths` run in a `finally` in `Pipeline.execute_job()`). If the terminal
  stage was skipped or produced no output but an earlier stage's temp file is the best available
  result, that temp is *promoted* (moved, not copied — `_promote_temp_to_output()`) onto
  `job.output_path` instead of being deleted.
- `scan_directory()` (`core/analysis.py`) skips hidden files (dotfiles), including orphaned
  `.avf_*` intermediate temp files that may be left next to an input after a crash.
- Stage outputs chain: each stage's `output_path` becomes the next stage's `input_path`.
- **`input_info` is re-probed after every stage that produces a new output file**, not just
  once up front. `execute_job()` probes `job.input_path` once before the loop starts, then
  after a stage completes with a NEW `output_path` (different from the file that was just fed
  in — a skipped stage or a passthrough that reuses the same path is not re-probed), calls
  `get_video_info()` again on that new file and re-injects any pipeline-layered keys (currently
  `general.target_format`, via `Pipeline._reprobe_input_info()`/`_INJECTED_INPUT_INFO_KEYS`)
  before the next stage's `should_run()`/`execute()` see it. A probe failure on the re-probe
  logs a WARNING and keeps the previous `input_info` rather than failing the job (fail-open, same
  convention as other auxiliary info). This fixes a real bug: `crop` shrinking a
  1080x1080 letterboxed input down to 1080x608 (or a 1920x1080 pillarboxed input down to
  608x1080) used to be invisible to `upscale.should_run()`/`_effective_target_bounds()`, which
  saw the ORIGINAL pre-crop resolution and either misjudged the orientation-aware target or
  concluded (wrongly) that the input was already at target and skipped upscaling entirely. The
  post-scene-mode re-probe (`scenes.enabled`, ~line 671 in `pipeline.py`) is unchanged and
  covers the reassembled scene-mode file; the general per-stage mechanism above then re-probes
  again from there on subsequent stages, same as any other stage transition.
- **`auto_determine_stages()` never geometry-gates a stage's PLAN MEMBERSHIP against the
  up-front probe** -- fixed alongside the re-probe bug above, since the two are the same root
  cause wearing two hats. Previously it only appended `"upscale"` to the auto-determined stage
  list when the original probe's resolution was already below `quality.quality_target
  .target_resolution`; for an input already at (or, orientation-swapped, effectively at) that
  target -- e.g. 1920x1080 vs. a `[1920, 1080]` target -- `"upscale"` never entered `job.stages`
  at all, so the in-loop `should_run()` (even with fresh post-crop geometry) never got a chance
  to run: a stage absent from the plan is never instantiated. Concretely, a 1920x1080 input with
  pillarboxed 9:16 content that `crop` shrinks to 608x1080 needs upscaling back up to ~1080x1920,
  but never got it, because `crop`'s ORIGINAL resolution comparison found nothing to do. Now,
  `"upscale"` is appended to the plan whenever `target_resolution` is configured AT ALL,
  regardless of the up-front probe's resolution -- `UpscaleStage.should_run()`/
  `_effective_target_bounds()` (orientation-aware, and re-probed per-stage per the fix above) is
  the sole authority on whether it actually runs. With no `target_resolution` configured,
  `"upscale"` still never enters the auto-determined plan (unchanged). `interpolate`'s
  `target_framerate`-vs-`framerate` check and `hdr`'s `is_hdr` check were deliberately left as
  plan-time gates -- crop doesn't alter a video's framerate or HDR-ness, so those two remain
  accurate at plan time. Audited the rest of `auto_determine_stages()` and the preset
  `enable_stages` path (`core/presets.py`) for the same class of bug: no other stage's plan
  membership is geometry-gated -- `enable_stages` entries are static per-stage `enabled: bool`
  config writes, not probe-conditioned, and every other `auto_determine_stages()` branch
  (`stabilize`/`denoise_video`/`deblock`/`normalize_volume`/`normalize_audio`/`speed`/`crop`) is
  gated on `stages.<name>.enabled`/`speed.enabled`/`crop.enabled` config flags, not on the
  input's probed geometry.

## Output Path Resolution

Output path is resolved in this priority order:
1. `job.output_path` explicitly set (e.g., via `add_job(input, output)`)
2. `config.get("general", "output_dir")` + `{stem}_enhanced{ext}` — set via CLI `-o` or config.yaml
3. `{input_dir}/{stem}_enhanced{ext}` (fallback)

The CLI `-o` flag sets `general.output_dir` in config. The pipeline reads it when creating jobs.

## Job outcomes (SKIPPED, existing-output spec-check, input-probe policy)

`docs/REQUIREMENTS.md` § 6.1/6.2/6.3 (implemented as one decision path in
`Pipeline.execute_job()`, right after the input probe and before scene mode/any stage runs — see
`Pipeline._decide_existing_output()`/`_terminal_probe_failure_result()`/`_handle_probe_warning()`
in `core/pipeline.py`):

- **`JobResult.outcome`**: `"completed" | "failed" | "skipped"` — canonical; `JobResult.success`
  (`bool`) is kept in sync automatically (`success == (outcome == "completed")`, enforced in
  `JobResult.__post_init__`). `PipelineStatus.SKIPPED` mirrors this on `job.status`.
  `JobResult.skip_reason` is a machine-readable sub-reason when `outcome == "skipped"`:
  `"output-exists"` / `"output-exists-mismatched"` / `"invalid-input"`. `JobResult.decision_log`
  (`list[str]`) records the human-readable decision trail (what was measured vs. targeted, which
  flags drove the outcome, rename actions) for every job, not just skipped ones — this is what a
  future structured JSON report (§ 6.6, not yet implemented) will serialize.
- **Existing output, `general.overwrite: false`**: no longer an automatic FAILED. Controlled by
  `general.existing_output` (`"skip"` default / `"fail"` restores the old FAILED+ERROR behavior
  exactly). When `"skip"` and `general.check_existing_target` (default `true`) is on, the existing
  file is ffprobed and spec-checked against the job's effective targets (container, resolution,
  framerate, video/audio codec — bitrate deliberately excluded) via
  `core/output_check.py`'s `effective_output_targets()`/`check_output_spec()`. A match is SKIPPED
  `"output-exists"`; a verified mismatch is SKIPPED `"output-exists-mismatched"` (logged at
  WARNING) unless `general.reprocess_mismatched: true`, in which case
  `general.existing_mismatched` (`"rename"` default / `"overwrite"`) decides whether the old file
  is renamed aside (`<stem><general.mismatched_rename_suffix><N><ext>`, N from 1, first unused —
  `general.mismatched_max_renames` caps N; `null` unlimited, `0` disables renaming entirely and
  turns a rename-mode mismatch into a FAILED job — a deliberate strict "never silently rename or
  overwrite" posture, allowed at config-validation time, not rejected) or overwritten directly.
  Resolution comparison reuses `UpscaleStage`'s orientation-aware bounds/threshold math, now
  shared via `core/output_check.py`'s `effective_target_bounds()`/`SKIP_SCALE_THRESHOLD` (also
  aliased as `UpscaleStage._SKIP_SCALE_THRESHOLD`) so both stay in lockstep. Framerate uses a
  small epsilon (`FRAMERATE_EPSILON = 0.11`, e.g. 29.97 ≈ 30); codec comparison normalizes to
  codec FAMILY (e.g. `libx265`/`hevc_nvenc` both satisfy an `"hevc"` target), never the literal
  encoder string. An unspecified target (e.g. no `quality.quality_target.target_resolution` set)
  is NEVER a mismatch — `effective_output_targets()` is deliberately conservative about what
  counts as "the user actually specified this".
- **Input probe failure policy** (`core/ffmpeg_utils.probe()` now runs ffprobe with `-v error`,
  not the old `-v quiet`, so a raised `RuntimeError`'s message actually carries ffprobe's stderr):
  an unreadable input FAILS the job by default (stderr surfaced in the log/`JobResult.errors`);
  `general.skip_invalid_inputs: true` makes it SKIPPED (`"invalid-input"`) instead, so one corrupt
  file doesn't need to fail an entire batch. Applied uniformly whether `auto_determine_stages()`'s
  own probe or `execute_job()`'s explicit probe hits the bad file first (both funnel through
  `Pipeline._terminal_probe_failure_result()`). Separately, a *successful* probe can still have
  non-empty ffprobe stderr (a real warning ffprobe recovered from) — always surfaced as a WARNING
  log; `general.fail_on_probe_warnings: true` promotes that to a hard FAILED instead. `ProbeResult`
  carries this as `.stderr` / `to_info_dict()["probe_stderr"]`.
- **CLI/GUI**: `avf process`'s exit code is driven by `cli.py`'s `_count_failed()`
  (`outcome == "failed"` only) — a run whose jobs are all completed-or-skipped exits 0.
  `_print_summary()` reports Skipped as its own line (with sub-reasons), separate from Failed. The
  GUI's job table renders a SKIPPED job as "Skipped", not "Failed"
  (`gui/main_window.py::_on_job_complete`).
- Not yet implemented: § 6.4 (per-video stage/mode summary table), § 6.5 (media info/timing
  instrumentation), § 6.6 (structured JSON report), § 6.7 (PII-clean log variant), § 6.8 (`avf
  config clean|upgrade|dump`) — see `docs/REQUIREMENTS.md` § 6 for the full planned set and
  delivery order.

## AI/Traditional Method Selection

The four AI-capable stages (upscale, deblock, denoise_video, interpolate) all resolve their
`method` ("ai" vs "traditional") through one shared helper, `BaseStage.resolve_ai_method(
explicit_method, auto_default)` (`core/stages/base.py`), called at the top of each stage's
`execute()`. Precedence, highest first:

1. **An explicit `method=` kwarg** passed by a caller (scene mode, tests, direct `--stage`
   invocations) — wins outright, no matter what config says.
2. **`stages.<name>.use_ai`** (tristate: `null` (default) / `true` / `false`) — per-stage
   override, set in config only (no dedicated CLI flag; use `--set stages.<name>.use_ai=true`).
3. **`general.use_ai`** (tristate, same as before) — set by the CLI's `--ai`/`--no-ai`
   (mutually exclusive flag_value pattern). Per the same convention as `ai_fallback` below, the
   global CLI flag does **not** override an explicit per-stage config value — step 2 already won
   if it applied.
4. **The stage's own hardcoded auto default** — used only when nothing above resolved it.
   These are deliberate, not oversights: **upscale** defaults to AI (its own scale-threshold
   logic in `_auto_method_for_scale()` still decides "already at target" → traditional even at
   this step), **deblock** defaults to AI (better quality), **denoise_video** and **interpolate**
   both default to **traditional** (hqdn3d/minterpolate are far cheaper with no GPU/model
   dependency, and minterpolate is fast with decent quality — a deliberate asymmetry vs.
   upscale/deblock).

`resolve_ai_method()` returns `(method, source)`; every stage immediately logs the choice at
**INFO** via `BaseStage._log_ai_method_choice()` — this fires on every `--verbose` run (no need
for DEBUG), so a grep for "using traditional"/"using AI" in a log always explains what happened.
Format: stage name, chosen method + a stage-specific technology description, and the resolution
source; when the auto default landed on traditional, the message also appends an opt-in hint
(e.g. `set stages.interpolate.use_ai: true or pass --ai to use it`) — the hint is omitted once
the traditional choice came from an explicit `use_ai: false`/`--no-ai` rather than the default,
since there's nothing to "opt into" from an explicit choice. Example lines:
```
interpolate: using traditional minterpolate (auto default; AI/RIFE model 'rife_v4.6' is
configured but not selected -- set stages.interpolate.use_ai: true or pass --ai to use it)
interpolate: using traditional minterpolate (stages.interpolate.use_ai: false)
interpolate: using AI interpolation (RIFE 'rife_v4.6', backend torch) (selected by
stages.interpolate.use_ai: true)
```

### AI-fallback policy

Separately from *whether* to use AI, each AI-capable stage (upscale, interpolate,
denoise_video, deblock) has an **ai_fallback** policy controlling what happens when the AI path
is selected but can't actually run — PyTorch not installed, model download/load failure,
an inference exception, or CUDA OOM after tiling retries are exhausted:

- `general.ai_fallback` (default `True`) is the global default.
- `stages.<name>.ai_fallback` (default `null`) overrides it per-stage; `null` inherits the
  global value.
- `True` (default): silently fall back to the stage's traditional FFmpeg method, logged at
  WARNING.
- `False`: the stage FAILS outright with a named cause (e.g. `"AI processing unavailable for
  stage 'upscale': model not available: ..."`) instead of silently downgrading output quality.
- CLI: `--ai-fallback` / `--no-ai-fallback` overrides `general.ai_fallback` for the run; it does
  **not** override an explicit per-stage `stages.<name>.ai_fallback` set in config.
- Implemented via `BaseStage.is_ai_fallback_enabled()` / `BaseStage._ai_fallback_or_fail()`
  (`core/stages/base.py`) — call `_ai_fallback_or_fail()` at every point the AI path can't
  proceed; do NOT use it for a legitimate OOM-triggered tiling retry (still the AI path) or a
  genuine post-AI failure (e.g. a mux error after AI frames were already produced) — those keep
  failing regardless of this policy.

### Backend selection (torch vs ncnn)

Independent of the AI/traditional choice, the `upscale` and `interpolate` stages also take
`stages.<name>.backend: "torch" | "ncnn"` (default `"torch"`). `"ncnn"` runs inference
in-process via Vulkan instead of PyTorch/CUDA — portable to AMD/Intel/integrated GPUs with no
CUDA-matched torch build required (`ai/backends/`). Both stages use *different* Python packages
for this, both under the `ncnn` extra in `pyproject.toml`: upscale uses the generic `ncnn`
package; interpolate uses `rife-ncnn-vulkan-python`, because RIFE's official ncnn graph needs a
custom `rife.Warp` ncnn layer the generic package doesn't register (see
`ai/backends/ncnn_interpolate.py`'s docstring for the full story) — `rife-ncnn-vulkan-python`
wraps the upstream C++ tool that has it. `rife-ncnn-vulkan-python` has no prebuilt wheel for
Python 3.14 as of this writing; building it from source needs the `swig` system package and,
against modern CMake, `CMAKE_POLICY_VERSION_MINIMUM=3.5` set for the install command — see the
`ncnn` extra's comment in `pyproject.toml` for the exact command. It falls through the same
`ai_fallback` policy as any other AI-unavailable condition when the required package, a Vulkan
device, or the ncnn model files aren't available.

The upscale/ncnn backend is verified working (correct, non-black output) but is **not currently
competitive with `backend: torch`** even on a GPU where CUDA works well — see `docs/ROADMAP.md`'s
NCNN backend section for measured numbers (slower than torch at small frames, unpatched OOM on
plain 720p frames regardless of fp16 `Option` flags, ~36s/frame once tiling is forced to work
around that). Treat `backend: ncnn` for upscale as the AMD/Intel/iGPU portability option, not a
performance option, until the upstream generic Python bindings close those gaps.

RIFE's ncnn backend (`backend: ncnn` on `interpolate`) is verified working via
`rife-ncnn-vulkan-python` and, unlike upscale/ncnn, *is* competitive: ~82 fps at 1280x720 measured
directly against `NcnnInterpolateBackend`, and ~18 fps measured end-to-end through the full
`interpolate` stage (frame extraction + inference + encode) on an RTX 5060 Ti via Vulkan — see
`docs/ROADMAP.md`'s NCNN backend section for the full numbers. It uses the project's own cached,
hash-verified `rife_v4.6` ncnn model (nihui's official "rife-v4" release asset) rather than the
older checkpoint `rife-ncnn-vulkan-python` bundles internally, kept consistent with the torch
backend's default model.

ncnn models (.param/.bin) have their own registry in `ai/model_cache.py`
(`NCNN_MODEL_REGISTRY` / `ensure_ncnn_model_available()`), separate from the torch `.pth`
registry — stage pre-flight model checks must not gate an ncnn run on the torch registry.

### Real-ESRGAN model registry (RRDB vs compact SRVGG)

`ai/model_cache.py`'s `MODEL_REGISTRY` (torch `.pth` models, used by `upscale`/`deblock`/
`denoise_video` at `backend: torch`) has two architecture families, dispatched in
`RealESRGANUpscaler.load_model()` (`ai/wrappers/upscale.py`) off each entry's `arch` field
(default `"rrdb"` when absent, so every pre-existing entry is unaffected):

- **RRDB (`RRDBNet`, default)** — `RealESRGAN_x4plus`, `RealESRGAN_x2plus`,
  `RealESRGAN_x4plus_anime_6B`. ~16.7M params, the highest-quality restoration, and the default
  for all three AI-capable stages. `deblock`/`denoise_video` (both run Real-ESRGAN at scale<=2)
  and `upscale` (for scale<=2 requests) transparently substitute `RealESRGAN_x2plus` whenever the
  configured model is literally `"RealESRGAN_x4plus"` (a strict `==` string check, e.g.
  `core/stages/deblock.py`, `core/stages/denoise_video.py`) — this swap is model-name-string-based
  and does NOT trigger for any other model name, compact or RRDB.
- **Compact (`SRVGGNetCompact`)** — `realesr-general-x4v3` (`num_conv=32`), `realesr-general-wdn-x4v3`
  (`num_conv=32`, denoise-strength companion — official usage blends the two checkpoints' state
  dicts for a tunable `denoise_strength`; that blending is **not implemented**, this checkpoint is
  only usable standalone here), `realesr-animevideov3` (`num_conv=16`, animation-tuned). ~1.2M/
  ~0.6M params — an order-of-magnitude-plus less GPU compute per frame than RRDB (the AI stages
  are measured ~100% `gpu_forward`-bound, see "AI-stage instrumentation" above), at some
  restoration-quality cost versus RRDB. `arch: "srvgg"` and `num_conv` are registry-only fields
  (RRDB entries have neither); `RealESRGANUpscaler.load_model()` reads them via the pure helper
  `resolve_arch()` to pick `SRVGGNetCompact(num_conv=..., upscale=native_scale)` over
  `RRDBNet(scale=native_scale)`. All three compact checkpoints are the official
  xinntao/Real-ESRGAN v0.2.5.0 release assets, hash-verified the same way as every other registry
  entry.

RRDB remains the default for all three stages — set `stages.<name>.ai_model` explicitly to opt
into a compact model (see `docs/config.example.yaml`'s `ai_model` comment for the full list of
valid values).

### Batched inference (torch backend only)

`RealESRGANUpscaler` (`ai/wrappers/upscale.py`, used by `upscale`/`deblock`/`denoise_video` at
the torch backend) supports two independent, opt-in batching axes, both default `1` (today's
one-at-a-time behavior, unchanged unless configured):

- **`stages.<name>.batch_size`** — N different *frames* stacked into one forward pass
  (`RealESRGANUpscaler.upscale_batch()`/`upscale_video()`, `ai/torch_utils.py`'s
  `tensor_from_frames`/`frames_from_tensor`). Only applies when a frame is small enough to skip
  tiled inference — falls back to one-frame-at-a-time whenever any frame in a chunk would need
  tiling, `tta_mode` is enabled, or the ncnn backend is in use (none of those combine with
  whole-frame batching in this implementation).
- **`stages.<name>.tile_batch_size`** — N *tiles of one frame* stacked into one forward pass
  inside `run_tiled_inference()`, when a frame's resolution triggers tiled inference (see
  `tile_size`/`AUTO_TILE_THRESHOLD_PX` above). This is the batching knob that matters for large
  (e.g. 4K) input: at the default `tile_size=512`, a 4K frame needs a 5x8=40-tile grid, so 40
  small sequential forward passes (each paying Python/kernel-launch/host-device-sync overhead)
  collapse into a handful of larger batched calls. `compute_tile_grid()`'s interior tiles mostly
  share one padded input shape (only the last row/column, clamped at the image boundary, differ),
  so tiles are grouped by shape before being stacked — never mixes shapes into one `torch.cat`.

Both: on a caught `torch.cuda.OutOfMemoryError`, the batch is recursively halved and retried
(mirroring the existing tile-size halving-retry pattern), down to batch=1, which then goes
through the existing single-item OOM/tiling-retry path unchanged. Neither axis is combined with
the other in this pass — a chunk needing both is out of scope (whole-frame batching defers to
per-frame processing, where tile batching then applies per-frame instead). CLI: `--batch-size` /
`--tile-batch-size` set both keys across `upscale`/`deblock`/`denoise_video` at once (no
per-stage CLI override, matching the `--crop-limit` precedent's single-flag-multiple-keys shape
where a per-stage split isn't worth a flag each).

`run_tiled_inference()`'s OOM path (`_infer_group`) releases the failed batch's input tensor
(`del batched_in`) BEFORE clearing the CUDA allocator cache and recursing into the halved retry —
done outside the `except` clause itself (a flag is set inside `except`, the actual cleanup runs
after the `try/except` has fully exited), because an in-flight exception's traceback keeps the
raising frame's own locals alive, including `infer_fn`'s reference to that same tensor; a `del`
executed while still inside the `except` block would not actually drop the last reference yet.
Previously the tensor stayed alive through the entire halving recursion, so every retry ran with
LESS free VRAM than the attempt that had just failed. Remainder tiles in a shape group (fewer
than `tile_batch_size` left over) are now processed through the single-tile path (`_infer_one`)
rather than one last partial-size batch, keeping every batched-tensor shape seen during a run
uniform instead of fragmenting the CUDA caching allocator with a one-off `N`.

### AI stage temp-encode quality and the chunked streaming path

`upscale`/`deblock`/`denoise_video`/`interpolate`'s AI paths each write AI-processed frames to an
internal temp `.mp4` (`StreamingVideoWriter` or `FrameProcessor.frames_to_video()`), then mux that
temp file's video track against the original audio to produce the stage's real output. Both the
temp encode and the mux pass now take **explicit** `crf`/`preset` args instead of relying on
implicit defaults:
- The temp encode passes `stages.<name>.temp_crf` (new config key, default `16`) and the
  hardcoded preset `"medium"` (not itself a config key). Previously this write had NO `-crf`/
  `-preset` at all, silently landing on libx264's own default (CRF 23).
- The mux pass now uses `-c:v copy` (verified bit-identical to the temp file's video stream via
  stream-hash comparison), NOT a second `-crf 18` re-encode. Previously every AI stage muxed with
  `-c:v libx264 -crf 18`, meaning AI-processed frames were encoded TWICE at two different quality
  levels — a real quality bug (the temp's CRF-23 generation was baked in before the mux's CRF-18
  pass ever saw it) plus a wasted full x264 pass over the whole video for no benefit.
- `interpolate`'s AI/RIFE path had the identical double-encode pattern in its own internal temp +
  mux and got the same fix (also gained its own `stages.interpolate.temp_crf`, default 16) — it's
  architecturally separate from the other three (still buffers frames in a list before writing,
  not the streaming path below) but shared the exact same bug.

All AI-capable videos (`upscale`/`deblock`/`denoise_video`) now go through the chunked streaming
path (`ai/frame_pipe.get_frame_reader()`/`get_frame_writer()`, `chunk_size=25` — see "Mixed
Python/Rust" above for the Rust-backed `avf_framepipe` transport and its Python fallback)
regardless of frame count — the old `total_est > 1000` frame-count threshold and its full-buffer
`extract_frames()` → flat list → `frames_to_video()` route are gone from these three stages'
`_execute_ai()`/`_run_single_ai_pass()`. That threshold was resolution-blind: a short but
large-resolution (e.g. 4K) clip could still fall under 1000 frames while materializing its ENTIRE
frame set in RAM (a 33s 4K clip is ~25GB uncompressed), with zero decode/inference/write overlap.
Streaming has no measurable downside for short clips either, so there's no longer a reason to keep
two code paths. `extract_frames()`/`frames_to_video()` themselves are untouched in
`ai/frame_processor.py` — other callers (`interpolate`'s AI path, `frames_to_temp_video()`) still
use them directly (not through `ai/frame_pipe.py`).

### Pinned staging pool (torch backend, CUDA only)

`ai/torch_utils.py`'s `PinnedStagingPool` (module-level singleton via `get_pinned_staging_pool()`)
holds a small, capped set of persistent `torch.empty(..., pin_memory=True)` CPU tensors, keyed by
`(shape, dtype)`, `SLOTS_PER_KEY=2` round-robin slots per key, `MAX_KEYS=2` (4 pinned tensors
total, LRU-evicted by key). `tensor_from_frame`/`tensor_from_frames` (`ai/torch_utils.py`) copy
the incoming numpy frame bytes into a pool slot (`copy_()`) and do the same `non_blocking=True`
H2D `.to(device)` as before, instead of calling `.pin_memory()` fresh (page-locking a brand-new
host buffer) on every single call — a real per-call cost at e.g. 4K (~25MB/frame). CPU-only
callers (`device.type != "cuda"`) don't touch the pool at all.

**Correctness-critical**: a pool slot is *reused*, and a `non_blocking=True` H2D copy reads from
its pinned host tensor asynchronously — the read is not guaranteed to have finished by the time
the Python call that queued it returns. `get_slot()` therefore records a `torch.cuda.Event` right
after each slot's H2D copy is queued (`record_copy()`) and `synchronize()`s on that event before
handing the same slot out again, so a later `copy_()` can never overwrite host memory the device
is still reading from. Do not remove or weaken this without understanding why — an
overwrite-before-copy race here would corrupt frames silently and nondeterministically, not
crash. See `tests/unit/test_torch_utils.py::TestPinnedStagingPoolCudaCorrectness` (real CUDA,
`@pytest.mark.integration`) and `TestPinnedStagingPoolKeying` (mocked, no GPU needed).

### AI-stage instrumentation (`ai/frame_processor.py`)

`StageTimer` and `gpu_forward_timer()` provide lightweight, always-on timing for the chunked AI
stage loop shared by `upscale`/`deblock`/`denoise_video` — instantiated once per stage run
(`StageTimer("upscale")` etc.) and threaded through as an optional `timer=` kwarg into
`RealESRGANUpscaler.upscale()`/`upscale_batch()`/`upscale_video()` (`ai/wrappers/upscale.py`).
Phases tracked: `decode_wait` (time blocked pulling the next chunk out of the `ai/frame_pipe.py`
reader's `next_batch()` — Rust `avf_framepipe.FrameReader` or its Python fallback, whichever
backend is active — measured at the stage loop via manual calls instead of a plain `for` loop),
`h2d_preprocess`/`d2h_postprocess` (`tensor_from_frame(s)`/`frame_from_tensor(s)` call time),
`gpu_forward` (every model forward call — bracketed with `torch.cuda.Event` pairs via
`gpu_forward_timer()` for honest device-side time on CUDA, wall-clock fallback on CPU/MPS),
`write_wait` (time blocked in the `ai/frame_pipe.py` writer's `write_batch()`). Per-chunk
breakdown logs at DEBUG
(`StageTimer.end_chunk()`); an aggregate percentage-of-total breakdown logs at INFO once at stage
end (`StageTimer.summary()`). Independently, periodic throughput (`frames_done`, `elapsed_sec`,
`current_fps`) logs at INFO every ~10s or 10 chunks (whichever first) — this exists specifically
so a run killed early by an external timeout (e.g. a truncating `timeout` wrapper) still yields a
valid frames/sec curve instead of nothing; see `docs/REQUIREMENTS.md`'s R5.3 note on the 4K
benchmark this was retroactively needed for. `interpolate`'s RIFE path is NOT instrumented (it
doesn't share this loop shape — see "Kill the ≤1000-frame full-buffering path" below).

## Upscaling & Aspect Ratio

The upscale stage respects `quality.quality_target.keep_aspect_ratio` (default `True`):
- Rotates the preset bounding box to match input orientation (portrait input swaps w↔h)
- Scales to fit within the (possibly rotated) bounding box, preserving pixel count
- Square input uses the shorter preset edge for both dimensions
- Dimensions are rounded to even values (H.264 requirement)

Example: 1080p60 preset (1920×1080) + 9:16 portrait input → scales to ~1080×1920 (portrait).

`UpscaleStage._effective_target_bounds()` (`core/stages/upscale.py`) is the single shared
implementation of this rotation, used by `should_run()`'s "already at target?" gate, `execute()`'s
AI-vs-traditional method selection, and `_calculate_target_dimensions()`'s actual scale
computation — all three must agree on what "at target" means for a given input's orientation.
Previously `should_run()` and the method-selection logic each compared the input's raw resolution
against the *unrotated* preset target directly, while only `_calculate_target_dimensions()` did
the rotation. This meant a portrait input already at its rotated target (e.g. a 1080×1920 input
against a `[1920, 1080]` preset target) was misjudged as needing upscaling: `should_run()` let the
stage proceed, `execute()` picked the AI method, and `_execute_ai()`'s per-pass scale computation
— which rounds each pass to a power of 2 via `2 ** round(log2(sf)) if sf > 1 else 2` — silently
substituted a forced 2x scale whenever a pass's own needed scale computed to `<=1` (i.e. already
at/past target), instead of doing nothing. Real production repro: a 1080×1920 portrait 60fps
input against the `1080p60` preset (target `[1920, 1080]`, `keep_aspect_ratio: true`) ran
`RealESRGAN_x2plus` at 2x anyway, producing a wasted 2160×3840 output (~1fps, over an hour for a
63-second clip, ~99% GPU-forward time per instrumentation — pure wasted compute, since the input
was already exactly at target).

Fixed at three levels, all keyed off `UpscaleStage._SKIP_SCALE_THRESHOLD = 1.05` (5% linear, ~10%
area): `should_run()` and `execute()`'s method selection both compute the orientation-aware needed
scale (`max(bound_w/w, bound_h/h)` against the rotated bound) and treat `<= 1.05` as "already at
target" (skip / pick traditional over AI); `_execute_ai()` additionally guards itself directly
(defense in depth for `_execute_ai()` being invoked outside `should_run()`, e.g. `--stage
upscale`) and, within the per-pass loop, does a cheap `_resize_to_exact()` lanczos resize (or a
zero-loss `_copy_stream()` stream-copy if dimensions already match exactly) instead of an AI pass
whenever a pass's own computed `sf <= 1`. The 1.05 threshold specifically covers the "crop stage
shaved a few px off before upscale ran" case (e.g. 1072×1908 vs. a 1080×1920 rotated target is a
1.0075x scale) without masking any genuine upscale need (a real "needs upscaling" input is almost
always well above 1.05, e.g. 540×960 → 1080×1920 is 2.0x) — chosen to skip/resize rather than
still running a (much smaller, non-power-of-2) AI pass, since Real-ESRGAN's minimum discrete scale
is already 2x and a genuine sub-2x need this close to target isn't worth a full AI forward pass.

## Configuration

- Config file: `~/.config/auto-video-fixer/config.yaml` (Linux, `$XDG_CONFIG_HOME` if set),
  macOS/Windows paths in `config.py:get_config_dir()`.
- Data dir (models, cache): `get_data_dir()` — `$XDG_DATA_HOME` or `~/.local/share` on Linux.
- **State/log dir**: `get_state_dir()` — `$XDG_STATE_HOME` or `~/.local/state` on Linux
  (macOS/Windows fall back to the same base as the data dir, since neither platform has a
  distinct "state" convention). `get_log_dir()` = `<state_dir>/logs`.
- **Automatic per-run DEBUG log file**: every CLI invocation writes a timestamped log file
  (`avf-YYYYMMDD-HHMMSS.log`) to `get_log_dir()` at DEBUG level, independent of console
  verbosity — always on, no flag needed. The exact path is printed/logged at startup. Retention:
  newest 50 kept (`prune_old_logs()`, pruned at startup before the current run's file is
  created); `--log-file PATH` *replaces* the automatic location for that run (same always-DEBUG
  file logging, user-chosen path; no state-dir file is created).
- **Startup settings banner**: on every invocation, the CLI logs (`_log_effective_settings()` in
  `cli.py`) the version, full `sys.argv`, which config file is in use (default vs. explicit), and
  a redacted diff of effective config vs. `Config.DEFAULTS` at INFO; the full effective config
  dump (also redacted) at DEBUG. Secret-looking keys (containing `api_key`/`apikey`/`token`/
  `password`/`secret`, case-insensitive) are masked as `"***"` via `config.redact_secrets()`.
  Re-logged after `process`'s own preset/flag merging, since the group-level log only reflects
  what was loaded from disk.
- **`--config PATH`** (global flag, before the subcommand) / **`AVF_CONFIG`** env var select an
  alternate config file. Precedence: `--config` flag > `AVF_CONFIG` env var > default platform
  path. Unlike the default path (silently falls back to `Config.DEFAULTS` if missing), an
  explicitly-given path (flag or env var) is a **hard error** if it doesn't exist — no silent
  no-op.
- Config is read-once at `Config()` construction; `config.set()` marks dirty and `config.save()` writes YAML.
- A full, commented example covering every `Config.DEFAULTS` key lives at
  `docs/config.example.yaml`.

### Config cascade

`avf process` builds its effective config as an ordered stack of layers, each deep-merged
(`config.py`'s module-level `deep_merge()`, exposed as `Config.apply_layer(layer, source_label)`)
onto the previous one — a layer only clobbers the keys it actually specifies; keys it doesn't
mention keep whatever the earlier layers set. **List-valued keys are the one exception**: a list
(e.g. `pipeline.default_order`, a preset's `enable_stages`) is replaced **wholesale** by the last
layer that sets it, never merged element-by-element.

1. **`Config.DEFAULTS`** (`config.py`).
2. **The user config file** — `~/.config/auto-video-fixer/config.yaml` by default, or whatever
   the top-level `avf --config PATH` flag / `AVF_CONFIG` env var points at instead (unchanged from
   before this feature: this step decides *which file* fills the "user config" slot, not an
   additional layer on top of it). Unlike the default path (silently falls back to
   `Config.DEFAULTS` if missing), an explicitly-given path (flag or env var) is a **hard error** if
   it doesn't exist — no silent no-op. Precedence: `--config` flag > `AVF_CONFIG` env var > default
   platform path.
3. **Each `process --preset NAME_OR_PATH` / `process --config PATH` layer**, applied in the order
   the flags actually appear on the command line, interleaved (`--preset A --config B --preset C`
   applies A, then B, then C — not "both presets, then the config", which is what Click's own
   per-option parsing would otherwise give you). Both flags are repeatable. This `process --config
   PATH` is a *different* flag from the top-level one in step 2 — it adds a layer on top of
   whatever step 2 already loaded, rather than choosing which file step 2 itself uses; it's a hard
   error if the path doesn't exist or isn't a YAML mapping. `--preset` accepts either a registered
   preset name (`get_preset()`) or a path to a preset file (`load_preset()`, JSON — see
   `save_preset()`/`load_preset()` in `presets.py`); disambiguated by: contains a path separator,
   ends in `.yaml`/`.yml`, or the path exists on disk — otherwise treated as a name, and an unknown
   name is an error.
4. **Every other CLI option that maps to a config key** (`--threads`, `--ai`/`--no-ai`,
   `--fps`, `--resolution`, `--codec`/`--audio-codec`/`--crf`/`--encoder-preset`, `--hwaccel`,
   `--gpu-device`, `--scene-mode`, `--drop-non-content`, `--crop-limit`, `--zoom-coverage`,
   `--batch-size`/`--tile-batch-size`, `--enable-stage`/`--disable-stage`, and `--set
   KEY=VALUE`), folded into **one** final layer applied **last** — this layer always wins over
   every `--preset`/`--config` layer, *even if the CLI flag was typed before them on the command
   line* (e.g. `avf process in.mp4 --crf 99 --preset size_reduction` still ends up with `crf: 99`,
   not the preset's `crf: 28`). Within this final layer, when two different flags target the same
   key (e.g. `--crf` and `--set encoding.crf=...`), the one that appears **later on the command
   line** wins — recovered the same way as step 3's interleaving, via a `sys.argv` scan
   (`cli.py`'s `_scan_process_argv()`/`_repeatable_matches()`). Purely operational flags that don't
   map to a config key (`--output-name`, `--recursive`, `--dry-run`, `--list-presets`, `--stage`)
   are unaffected by this layering — they're consumed directly by `process()` as before.
   - **`--set KEY=VALUE`** (repeatable): `KEY` is dot-notation (e.g.
     `stages.upscale.ai_model=RealESRGAN_x2plus`), creating intermediate mappings as needed.
     `VALUE` is parsed as a YAML scalar (`yaml.safe_load`), so `true`/`16`/`null`/quoted strings/
     inline lists (`[a,b]`) all work. Malformed input (no `=`, or an empty key) is a
     `click.BadParameter` error. **Security**: a key containing `api_key`/`apikey`/`token`/
     `password`/`secret` (case-insensitive — the same `_looks_like_secret_key()` check
     `redact_secrets()` uses) is rejected with an error telling the user to put it in `config.yaml`
     instead — keeps secrets out of shell history.

Both the group-level `sys.argv` scan (step 3/4's ordering recovery) require `sys.argv` to actually
reflect the invocation being processed — true for a real CLI invocation, but **not** true for a
programmatic `CliRunner.invoke(main, [...])` call in a test, whose `sys.argv` is the test runner's
own. When the scan can't find `"process"` in `sys.argv`, or what it *did* find doesn't exactly
match what Click parsed (count and values), the whole scan is discarded and a fixed fallback order
is used instead: step 3 falls back to "all `--preset` values, then all `--config` values" (Click's
own per-option order, cross-option interleaving lost); step 4 falls back to a fixed declaration
order matching the option list above. Tests exercising true argv-order behavior must
`monkeypatch.setattr(sys, "argv", ["avf", "process", ...])` with the exact args passed to
`CliRunner.invoke()` — see `tests/unit/test_cli.py::TestConfigCascadeCli` for the pattern.

`Config.apply_layer(layer, source_label)` records `source_label` in `config.sources` (e.g.
`"defaults"`, `"user-config:/path"`, `"preset:1080p60"`, `"config:/path/extra.yaml"`,
`"cli-flags"`) and logs the layer stack plus the layer's own (redacted) contents at DEBUG on every
call — useful for tracing which layer set a given effective value. The GUI (`gui/main_window.py`'s
`_on_preset_changed()`) uses the identical `config.apply_layer(preset.to_config(),
f"preset:{name}")` call when a preset is selected from the dropdown, mutating `self.config` in
place (never reassigning it — `self.pipeline` holds the same object by reference).

## Stage/quality ffmpeg timeouts

Most `run_ffmpeg()` calls that used to hardcode `timeout=600`/`1800`/`3600` (a whole-video
pass's wall-clock cap) now resolve a configurable, null-means-unlimited timeout instead —
real-world long inputs legitimately exceed any fixed cap on a whole-video pass; a genuine hang
should surface as an external job runner/OS-level inactivity timeout, not an arbitrary
per-run number baked into this codebase.

- **`pipeline.stage_timeout`** (`config.py` DEFAULTS): the global default timeout (seconds) for
  a stage's MAIN ffmpeg processing/mux pass(es). `null` (default) = unlimited.
- **`stages.<name>.timeout`**: per-stage override, absent by default (falls back to
  `pipeline.stage_timeout`).
- **Resolution**: `BaseStage.stage_timeout()` (`core/stages/base.py`) resolves
  `stages.<name>.timeout` → `pipeline.stage_timeout` → `None`, via the shared
  `resolve_timeout(value, key)` helper in `config.py`. Semantics at every level: `null`/absent
  or `0` → `None` (unlimited); a positive number → seconds; negative/non-numeric → `ValueError`
  raised *at the point the stage resolves its timeout* ("at use time"), not eagerly at
  config-load time. Stages pass the result straight to `run_ffmpeg(..., timeout=...)`, which
  accepts `None` (`subprocess.Popen.wait(timeout=None)` blocks indefinitely — no special-cased
  branch needed in `run_ffmpeg()` itself).
- **Per-occurrence override**: no new machinery — a `pipeline.default_order` mapping entry's
  existing `config: {timeout: 1200}` deep-merges onto `stages.<name>` for that occurrence only
  (via `BaseStage.__init__`'s `overrides` param, same mechanism every other per-occurrence
  config key already uses), and `stage_timeout()` reads it from `self._stage_config` like any
  other key. See `docs/config.example.yaml`'s `pipeline.default_order` section for a worked
  example.
- **`quality.timeout`**: `core/quality.py`'s `estimate_quality_vmaf()`/`estimate_ssim_psnr()`
  (the whole-video VMAF/SSIM comparison pass behind `Pipeline.execute_job()`'s post-job quality
  gate) aren't stages, so they get their own config key with identical null-unlimited
  semantics, resolved via the same `resolve_timeout()` helper and passed as an explicit
  `timeout=` param (these are plain functions with no `Config` access of their own —
  `Pipeline.execute_job()` resolves `quality.timeout` and passes it through).
- **Scene mode** (`core/scenes.py`): the per-scene split/concat/mux helpers (whole-video-scale
  work, previously fixed at 300/600/1800s) take a `timeout` param; `run_scene_mode()` resolves
  `pipeline.stage_timeout` once and threads it through. Per-scene stabilize/interpolate already
  go through the stage classes and honor `stages.<name>.timeout` like any other run.
- **CLI**: nothing new — `avf process ... --set pipeline.stage_timeout=1200` (or
  `--set stages.upscale.timeout=1200`, `--set quality.timeout=900`) already works via the
  existing `--set KEY=VALUE` cascade layer (see "Config cascade" above).
- **Deliberately KEPT at small fixed timeouts** (not wired to the above — a hang here indicates
  real breakage, not a long input): probes (`ffmpeg_utils.probe()`, 120s), hwaccel detection
  (`detect_hardware_acceleration()`, 10s), `crop.py`'s VLM-check single-frame extracts and its
  `should_run()` quick cropdetect sample (60s), `stabilize.py`'s `_detect_scenes()` 1fps frame
  extraction for scene-change analysis (120s), `normalize_audio.py`'s loudnorm first-pass
  loudness *measurement* (300s — the `-vn -sn -dn -f null -` analysis pass, not the actual
  audio-encode pass, which IS configurable), and `core/analysis.py`'s `extract_clip()` (300s —
  bounded by one scene's duration, not the whole video). `core/analysis.py` has no other
  hardcoded whole-video-pass timeout to wire up (only that one `run_ffmpeg()` call site exists
  in that module; its VLM/HTTP `urllib.request.urlopen()` timeouts are unrelated network
  request budgets, not ffmpeg passes).
- **Wired to the resolved timeout** (every stage's main processing/mux pass, including
  multi-pass/chunked ones): `upscale`, `deblock`, `denoise_video` (traditional pass + AI mux),
  `interpolate` (traditional single/chunked passes, chunk concat, final mux, AI/RIFE mux),
  `encode`, `speed`, `hdr`, `remux`, `normalize_audio` (the actual normalization pass, not the
  measurement pass above), `crop` (the real whole-scan `cropdetect` in `execute()`, and the
  final crop encode), and `stabilize` (the `vidstabdetect` shake-detection pass, the
  no-stabilization-needed passthrough copy, and the decode/transform pipe's `pipe_timeout` —
  see `stages.stabilize.pipe_timeout` in `docs/config.example.yaml`, which now falls back to
  this stage's own resolved timeout instead of a bespoke hardcoded `1800` when not explicitly
  set).

## Gotchas

- **stabilize.py `_get_video_dimensions` / `_get_video_framerate`**: Must use `stdout=subprocess.PIPE` (not `subprocess.DEVNULL`). Using DEVNULL discards output and forces fallback to 1920×1080 / 30fps, stretching portrait video to landscape.
- **stabilize pipe sizing**: The decode subprocess **must** include `-s {width}x{height}` to match the transform's `-s` input. Without it, anamorphic or non-standard resolution videos produce corrupted output due to stride mismatch.
- **vid.stab B-frame corruption**: Known bug (github.com/georgmartius/vid.stab#144). The pipeline pipes raw YUV420P between decode and transform to avoid it. Use accuracy=15 (higher causes FFmpeg return code 222).
- **FFmpeg output paths**: Stages must use an explicit output path. Using `output_path or input_path` on failure produces an empty file.
- **Stage disabled check**: `should_run()` must check `self.is_enabled()` which reads from `DEFAULTS["stages"][name]["enabled"]`.
- **`NormalizeVolumeStage`** (name `"normalize_volume"`) and **`NormalizeAudioStage`** (name `"normalize_audio"`) are two separate classes in `normalize_audio.py`.
- **normalize_audio.py silence-skip**: inputs with no real audio track get a silent stereo track
  added earlier in the pipeline, so by normalize time there IS an audio stream, just silent (or
  near-silent with dithering). Two-pass loudnorm's first pass would measure `input_i = -inf` on
  pure silence and the second pass would blow up (infinite gain) feeding that back in. Both
  stages read `stages.<name>.silence_threshold_db` (default `-80.0` LUFS, shared implementation
  `NormalizeAudioStage._silence_skip_result()`/`_resolve_measured_i()`); when measured integrated
  loudness is `-inf`/`nan`/unparseable or `<=` the threshold, the stage returns
  `StageStatus.COMPLETED` with `skipped_reason` set and the input copied through unchanged
  (matching `UpscaleStage`'s "already at target" / `StabilizeStage`'s "no stabilization needed"
  mid-execute skip convention) instead of failing.
- **Preset stage enable**: Presets define `enable_stages` which controls which stages run. A stage not listed in a preset's `enable_stages` will not execute, even if it's in the default order.
- **Stage `execute()` kwargs that matter must be named parameters, not left to `**kwargs`**: a
  real incident -- `InterpolateStage.execute()` used to accept `parallel_chunks` only via
  `**kwargs` (never forwarded to `_execute_traditional()`), so scene mode's per-scene chunk-budget
  override (`_scene_worker_budget`) was silently dropped and every concurrently-processed scene
  fell back to its own independent auto-chunking -- 6 scenes x up to 8 auto-chunks peaked at ~12
  concurrent 4K `minterpolate` ffmpeg processes and OOM-killed a 56 GiB container. If a caller
  (scene mode, tests, another stage) passes a kwarg that changes behavior, it needs an explicit
  named parameter all the way down the call chain -- `**kwargs`'s tolerant-signature convention is
  for genuinely-ignorable extras only.
- **`scenes.drop_non_content` silently doing nothing is not "fail open enough"**: it must check
  `analysis.vlm.enabled` BEFORE attempting any VLM/network call, not just fail open after a failed
  call -- with VLM disabled and its endpoint unreachable, an unconditional VLM call still burns
  full HTTP timeouts (minutes) before "keeping all scenes anyway". `run_scene_pipeline()` checks
  `analysis.vlm.enabled` first and logs one WARNING + keeps all scenes with zero network calls
  when it's false.
- **Every new subprocess spawn must detach stdin**: an ffmpeg subprocess whose stdin is the
  caller's controlling terminal switches that tty to raw mode (to poll for interactive keys) and
  does **not** restore it if killed/crashed/backgrounded -- the classic symptom is a shell with no
  keystroke echo and a stray blank line per command after an interrupted `avf process` run. Any
  new ffmpeg spawn must pass `-nostdin` (early in the arg list, before `-i`) **and**
  `stdin=subprocess.DEVNULL` on the `Popen`/`run()` call (Rust: `.stdin(Stdio::null())` on
  `Command`) -- both, belt and suspenders. ffprobe has no `-nostdin` flag, so `stdin=DEVNULL`/
  `Stdio::null()` alone is the fix there. **Exception**: a process whose stdin is deliberately a
  pipe (frame-feeding writers like `StreamingVideoWriter`, `ai/frame_processor.py`'s two Popen
  writers, `avf_framepipe`'s `FrameWriter`, `stabilize.py`'s transform process reading
  decode_proc's stdout) already can't capture a tty and does not need `-nostdin` added -- don't
  touch those. `core/ffmpeg_utils.py`'s `run_ffmpeg()` is the central choke point most stages
  route through; a spawn that bypasses it (a direct `subprocess.run`/`Popen`/`Command::new`) needs
  the fix applied at its own call site. See CHANGELOG.md's "ffmpeg/ffprobe subprocess spawns could
  leave the user's terminal stuck in raw mode" entry for the full audit and fix list.
- **Console output of external data must go through `sanitize_console_text()`**: filenames from a
  directory scan, ffmpeg stderr excerpts, exception text, and VLM output can all contain raw
  control/escape bytes (C0 controls, DEL, the C1 range, ESC) that -- if echoed straight to a
  terminal -- can trigger the same kind of tty-mode corruption `-nostdin` fixes for ffmpeg itself
  (defense in depth, not the primary fix). `config.sanitize_console_text()` (next to
  `redact_secrets()`) strips/replaces those bytes with U+FFFD while leaving every other Unicode
  codepoint (CJK, RTL, combining marks, emoji) untouched -- never normalize or strip non-ASCII
  text here. The logging pipeline's console handler already sanitizes everything that goes through
  `logger.py` (`_SanitizingConsoleFormatter`, console-only -- file/log-file handlers keep raw
  text). Anything printed via a *direct* `console.print()`/`click.echo()` call that interpolates
  external data (not routed through the logger) needs its own call to
  `sanitize_console_text()`/`cli.py`'s `_safe()` wrapper at the interpolation site -- see `cli.py`
  for the pattern; do not sanitize the whole f-string blindly if it also contains deliberate Rich
  markup (`[red]...[/red]`) the code itself emits.

## Known bugs / pitfalls to avoid

The bugs formerly listed here (`cb` scoping crash in chunked AI paths, `upscaled`/`all_upscaled`
typo, missing `os`/`probe` imports in `deblock.py`/`denoise_video.py`, unused local `run_ffmpeg`
shadow import, F541 f-string lint errors, unused `fps_val`) are **fixed** — verified directly
against current `deblock.py`, `denoise_video.py`, `upscale.py`, `interpolate.py`: `cb` callbacks
are defined before their chunked loops in all four stages, `os` is imported at the top of
`deblock.py`, `denoise_video.py` imports `probe` at module level, and the AI upscale/deblock/
denoise chunked paths all correctly reference the frame lists they build. Kept here only as a
historical note in case of regression — do not assume these still need fixing.

Remaining lint/style notes (still current):
- **ruff I001** on `deblock.py` imports: blank line between `from __future__` and the stdlib imports is expected by the sort rule.
- **ruff E501** line length is 100. Several stage files (stabilize, interpolate, upscale, cli) exceed it with long f-strings; plan around this.
- **mypy `--ignore-missing-imports`** is required; `torch`/`cv2`/`rife` are optional and mypy will flag them without that flag.

## Scene mode

Opt-in, off by default (`scenes.enabled: false`). See `core/scenes.py`, `Config.DEFAULTS["scenes"]`
and `Config.DEFAULTS["analysis"]["llm"]`, and `Pipeline.execute_job()`'s scene-mode block (right
after `current_path = job.input_path`, before the main stage loop).

- **Trigger**: scene mode only engages when `scenes.enabled` is true AND the job's stage list
  includes `stabilize` and/or `interpolate`. Otherwise it's a strict no-op -- the whole-video path
  is completely unaffected.
- **Flow**: `run_scene_pipeline()` (1) runs the existing scene detector
  (`VideoAnalyzer.detect_events`, same `analysis.event_detection.*` config -- no separate
  scene-mode threshold), (2) optionally drops non-content scenes (see below), (3) re-encodes each
  kept scene as its own clip (`split_scene_video`/`split_scene_audio` -- re-encoded, not
  stream-copied, so cuts land exactly on the detected boundary instead of snapping to the nearest
  keyframe), (4) runs `stabilize` (tiered strength) and/or `interpolate` (never across a cut) on
  each scene clip independently, in parallel across scenes when there's more than one (bounded
  worker pool shared with per-scene interpolation chunking, see `_scene_worker_budget`), (5)
  concatenates the processed video clips and the original audio (cut at the same boundaries) back
  into one file. `Pipeline.execute_job()` then continues the normal per-file pipeline from that
  reassembled file, with `stabilize`/`interpolate` removed from the remaining stage list (they
  already ran) -- `upscale`/`denoise_video`/`deblock`/`normalize_*`/`encode` all run once on the
  whole reassembled video, not per-scene.
- **Resolution-aware scene worker budget** (`_scene_worker_budget()`): splitting only the CPU
  budget between "scenes run concurrently" and "chunks per scene's traditional interpolation" is
  memory-blind -- each concurrent whole-scene `minterpolate` process's footprint scales with
  resolution, so a 4K input could still run several concurrent multi-GB processes even with the
  chunk-forwarding fix above. `run_scene_pipeline()` passes the already-probed input's
  `(width, height)` (no extra probe) into `_scene_worker_budget()`, which scales the TOTAL budget
  down before splitting: `pixels = width * height`, `reference = 1920*1080`,
  `mem_scale = max(1.0, pixels / (2 * reference))` (inputs up to ~2x 1080p keep the full budget;
  4K halves it; 8K quarters it again), `total_budget = max(1, round(min(cpu, 8) / mem_scale))`.
  `scenes.max_workers` (config, default `null`) caps the resulting `scene_workers` explicitly when
  set to a positive int -- it does not change `total_budget` itself, so `per_scene_chunks` is
  still derived from the (uncapped) total budget split across the (capped) worker count.
- **Fewer than 2 scenes detected, or any internal failure** (split/stabilize/interpolate/concat
  error, unexpected exception): `run_scene_pipeline()` returns `None` and the pipeline silently
  falls back to normal whole-video processing for that job -- scene mode can never turn a job that
  would have succeeded without it into a failure.
- **Per-scene stabilize strength tiering** (`scenes.stabilize.*`): each scene clip gets its own
  `StabilizeStage.execute()` call, which does its own `vidstabdetect` pass and reports
  `avg_shake` in its result metadata. "skip" tier = `StabilizeStage`'s own `needs_stab` gate
  (`avg_shake < stages.stabilize.threshold`) says no stabilization needed. "normal" tier =
  stabilized once with the stage's default config. "aggressive" tier = `avg_shake >=
  scenes.stabilize.aggressive_shake_threshold` (default 8.0px) -- re-run with `smoothness`
  multiplied by `scenes.stabilize.aggressive_smoothness_multiplier` (default 2.0). This means a
  shaky scene can get run through `vidstabdetect` twice (once for the normal-tier attempt, once
  more for the aggressive re-run) -- accepted cost, bounded to shaky scenes only.
- **Never-interpolate-across-a-cut** (`InterpolateStage` invoked per scene clip, never on the
  whole file when scene mode is on): each scene's own framerate is checked against the target via
  `should_run()` before interpolating, so a scene already at/above target framerate passes through
  unchanged instead of being redundantly processed.
- **Scene-mode AI/traditional method selection** (`interpolate_scene_clip()`, `core/scenes.py`):
  `scenes.interpolate.use_ai` (tristate, default `null`) is a scene-mode-ONLY override, resolved
  separately from (but able to defer to) the standard interpolate stage's own resolution --
  `null` resolves the method via the exact same shared `BaseStage.resolve_ai_method()` call the
  standard `interpolate` stage uses (so both paths respond together to `stages.interpolate.use_ai`
  / `general.use_ai`); `true`/`false` forces AI/traditional for the scene path only, without
  touching whole-video `interpolate` behavior. Either way, the resolved method is passed to
  `InterpolateStage.execute()` as the explicit `method=` kwarg (step 1 of the shared precedence,
  see "AI/Traditional Method Selection" above), alongside the per-scene `parallel_chunks` override
  (from `_scene_worker_budget`, see below) when traditional -- `execute()` accepts
  `parallel_chunks` as a real named parameter (not swallowed into `**kwargs`) and forwards it to
  `_execute_traditional()`, so this override actually takes effect per scene instead of every
  concurrently-processed scene falling back to `stages.interpolate.parallel_chunks`'s own
  independent auto-chunking (a real OOM incident: 6 scenes x up to 8 auto-chunks each peaked at
  ~12 concurrent 4K minterpolate ffmpeg processes and OOM-killed a 56 GiB container).
- **GPU inference semaphore** (`gpu.max_concurrent_inferences`, default `1`): scene mode runs up
  to `scene_workers` scenes concurrently in a thread pool; when the resolved method for a scene is
  "ai", each thread would otherwise launch a full RIFE inference and contend for VRAM instead of
  parallelizing. `interpolate_scene_clip()` acquires a process-wide `threading.Semaphore` (lazily
  created from config by `ai.torch_utils.get_gpu_inference_semaphore()`) around the AI-inference
  `stage.execute()` call only -- the traditional/chunked path is NOT gated. A thread blocking on
  the semaphore logs at DEBUG. Whole-video (non-scene) runs execute stages serially already, so
  they never need this; it's also the single-GPU placeholder for future multi-GPU inference
  distribution (see `docs/ROADMAP.md`).
- **Drop non-content scenes** (`scenes.drop_non_content: false` default, `--drop-non-content`
  CLI flag): requires `analysis.vlm.enabled: true` -- if `drop_non_content` is true but VLM is
  disabled, `run_scene_pipeline()` logs one WARNING
  ("scenes.drop_non_content requires analysis.vlm.enabled; skipping VLM scene classification --
  keeping all N scenes") and keeps every scene without touching the network at all (a real
  incident: with VLM disabled and its endpoint down, the old unconditional VLM call still fired,
  costing ~8 minutes of 120s HTTP timeouts before "keeping all scenes anyway"). When VLM is
  enabled, per-scene VLM sampling (`VideoAnalyzer.run_vlm_analysis_for_scene` -- samples
  `min(max_sample_frames, ceil(scene_duration / sample_interval_sec))` frames from within just
  that scene's time range, via `_extract_sample_frames_range`) feeds a coordinating text-LLM pass
  (`run_scene_coordinator`, config `analysis.llm.*`: `provider`/`model`/`api_key`/`api_url`/
  `allow_http`, same HTTPS-required-for-non-loopback gate as `analysis.vlm.allow_http`). The
  coordinator returns which scene indices to drop and why; dropped scenes are logged at WARNING
  (index, time range, reason) and excluded from the reassembled output. **Fails open on ANY
  failure** -- unreachable endpoint, exception, unparseable JSON response, or a `drop` array
  containing a non-integer/out-of-range/negative index -- logging a WARNING and keeping every
  scene (`run_scene_coordinator` returns `{"drop": [], "failed": True, ...}`; `run_scene_pipeline`
  additionally catches any exception from the whole VLM+coordinator block and fails open the same
  way). If the coordinator response would drop literally every scene, that's also refused (treated
  as a bad response) and every scene is kept -- scene mode will never produce an empty output.
- **A/V sync**: audio is re-cut at the exact same scene boundaries as the video (both re-encoded,
  not stream-copied, from the original source) and concatenated separately, then muxed against
  the concatenated (processed) video track. Since neither stabilization nor interpolation changes
  a scene's time span (interpolation adds frames covering the *same* duration, it doesn't stretch
  it), this keeps sync without needing to re-derive timing from the processed video.
- **Known limitation**: concatenated output duration/frame-count is close to but not always
  bit-exact vs. the sum of scene durations -- `minterpolate`'s per-chunk (and per-scene) frame
  count is duration-based, not a fixed multiply, so independent per-scene/per-chunk retiming
  passes don't always sum to exactly what one continuous whole-file pass would have produced.
  Verified within ~1-2% on a synthetic multi-scene clip, with zero blend/ghosting artifacts at any
  scene boundary (the actual correctness-critical property) -- see CHANGELOG.

## Stabilization zoom coverage

`stages.stabilize.zoom_coverage` (float, default `1.0`) tunes how aggressively the stabilize
stage's zoom compensates for borders introduced by stabilization, once `zoom_enabled` /
`zoom_threshold`'s movement-extent gate (`apply_zoom` in `StabilizeStage.execute()`) has already
decided zoom applies at all -- `zoom_coverage` does NOT change that gate, only what zoom is used
once it fires:

- `1.0` (default, unchanged behavior): `vidstabtransform`'s own `optzoom=1` ("optimal static
  zoom"), sized to the single worst frame in the clip -- guaranteed no visible border on any
  frame, at the cost of being the least tight/most-cropped option.
- `0.0`: no zoom at all (`optzoom=0`, `zoom=0`) -- every border from camera motion stays visible,
  equivalent to `zoom_enabled=False`'s zoom behavior but without disabling the rest of the gate
  logic (movement is still analyzed, `apply_zoom` is still computed and reported in metadata).
- In between: `optzoom=0` plus a static `zoom=<pct>` computed by
  `StabilizeStage._compute_static_zoom_pct()` from the TRF vidstabdetect already parses
  (`_analyze_trf_file`/`_movement_extent`'s TRF-reading lineage) -- the `zoom_coverage`-th
  quantile of PER-FRAME required-zoom estimates, not the max. This lets a user trade "guaranteed
  no border, ever" for "less aggressive crop, with occasional brief borders on the most extreme
  motion" -- e.g. a violent-motion ending gets a border for a second or two while the rest of the
  clip stays zoomed less than the `1.0` case would force.

**Accuracy limitation (state honestly, this is not exact)**: what actually determines a frame's
visible border is `vidstabtransform`'s own internally-computed SMOOTHED camera path (a function of
`smoothing`/`maxshift`/`optalgo`/`interpol`), which this code has no access to -- it only sees the
raw per-block local-motion (LM) values in the TRF. `_compute_static_zoom_pct()` integrates those
into an approximate raw cumulative camera-path position, then applies a local moving-average
smoothing window (matching `stages.stabilize.smoothness`, the same window vidstabtransform itself
uses) to approximate the smoothed path, and measures each frame's deviation from that local
average as its "required zoom" (`zoom_pct = 200 * shift_px / dimension_px`, derived from
`vidstabtransform`'s `zoom=Z%` scaling the frame by `1+Z/100` around center). This is
**directionally correct and tunable** (verified: monotonically increasing with `coverage`,
verified via a synthetic shaky clip) but not an exact match to vidstabtransform's internal
computation -- empirically, on one synthetic test clip, this method's own `coverage=1.0` quantile
(the theoretical max) came out ~3x higher than vidstabtransform's own logged `optzoom=1` "Final
zoom" value, i.e. the approximation is conservative/over-corrects rather than under-corrects for
that clip. Individual frames' actual post-smoothing border requirements can come out higher or
lower than this estimate. This also does NOT account for rotation's contribution to border size
(same limitation as `_movement_extent()` -- no new rotation math was added).

**Verification methodology** (ffmpeg/CPU only, no GPU needed): a synthetic clip was generated with
`ffmpeg -f lavfi -i testsrc2=...` piped through a time-varying `crop=` filter simulating handheld
shake (sinusoidal jitter) plus one abrupt high-velocity displacement late in the clip (the
"violent-motion ending" scenario). Verified via whole-video-union `cropdetect=limit:round:reset=0`
(same methodology as the auto-crop stage) and per-frame `cropdetect=limit:round:reset=1` box
sampling: `zoom_coverage=1.0` produced zero border on any of 178 sampled frames (matches
`optzoom=1`'s guarantee); `zoom_coverage=0.0` reproduced the same borders as the fully-unzoomed
baseline (~43% of frames showing a small border, matching `crop_mode: black`'s border-fill
behavior); `zoom_coverage=0.6`'s computed static zoom quantile was measurably smaller than the
`coverage=1.0` quantile computed by the same function (10.4% vs. 25.5% on the test clip) while
still fully covering that particular clip's (small, ~2-4px) actual borders -- demonstrating the
dial responds correctly to `coverage` even though this specific synthetic clip's real border
requirement was too small to show a visible difference in cropdetect output at 0.6 vs. 1.0.

CLI: `--zoom-coverage FLOAT` (`stages.stabilize.zoom_coverage`), following the `--crop-limit`
precedent of a single per-stage override flag.

## Auto-crop

Opt-in, off by default (`stages.crop.enabled: false`). See `core/stages/crop.py`,
`Config.DEFAULTS["stages"]["crop"]`, `core/analysis.py::run_crop_vlm_check`/
`_parse_crop_vlm_response`, and `docs/REQUIREMENTS.md` feature 3.

- **What it does**: detects a video's true content bounds -- a transition-resilient union over
  the whole video (the furthest real content geometry ever reaches toward each edge, EXCLUDING
  isolated transient bursts), NOT a per-frame flickering crop -- and crops to that single window
  with a `crop=w:h:x:y` re-encode (`libx264 -crf 18`, `-c:a copy`).
- **Detection** (`_detect_crop_full()`, `execute()`'s authoritative pass): FFmpeg `cropdetect`
  runs with `reset=1` (recompute per analyzed frame, still subject to cropdetect's own default
  `skip=2` frame-skip) and `max_outliers=<N>` (see below), printing one `crop=w:h:x:y` / `t:
  <seconds>` line per frame. Every line is parsed (`_parse_crop_frames()`) and fed to
  `aggregate_crop_windows()` (pure, module-level, heavily unit-tested):
  1. Group consecutive (by timestamp) frames into runs whose windows all match the run's first
     window within `transition_tolerance_px` (per edge: left/top/right/bottom).
  2. A run is a TRANSITION (excluded) iff its duration <= `transition_max_run_sec` AND no other
     run within `transition_window_sec` before/after it has a similar window -- an isolated
     deviant burst (e.g. one bright full-frame flash) is excluded; the same window recurring
     nearby, or a deviant run outlasting `transition_max_run_sec` even in isolation, is kept as
     real content geometry (moving logo, letterbox change, etc.).
  3. Result = the union (max extent toward every edge) of the surviving runs' windows, rounded UP
     (never down, never cuts real content) to `round`.
  4. Edge cases: empty input -> `None`; every run classified transition (pathological) -> falls
     back to the union of everything (logged at DEBUG); single run -> its own window.

  This replaces an earlier `reset=0` "union that can only grow" design, where a single bright
  full-frame transition anywhere in the scanned range permanently widened the crop window (in the
  worst case, degrading the crop to the full frame) for the rest of the scan. `execute()` logs at
  INFO after aggregation: frames/runs analyzed, runs excluded as transitions (with time ranges,
  capped at ~5), and the final window.
- **`max_outliers` (overlay tolerance)**: `stages.crop.max_outlier_ratio` (default 0.2, range
  0..0.5, 0 disables) is converted to cropdetect's `max_outliers=<N>` via `N =
  round(max_outlier_ratio * min(probed_width, probed_height))` -- lets a border line contain up
  to N non-black pixels and still count as border, so a logo/overlay sitting in the letterbox area
  no longer widens the crop to "protect" it. `min(width, height)` is used because cropdetect
  applies one absolute `max_outliers` count to both the row-scan and column-scan axes. Live-
  validated against ffmpeg (1920x1080 clip, 1920x800 centered content, 140px black bars, a bright
  ~200x60 logo in the bottom bar): `max_outliers=0` -> `crop=1920:920:0:140` (logo included);
  `max_outliers=216` (`round(0.2*1080)`) -> `crop=1920:800:0:140` (true content box). Non-black
  borders are out of scope for `max_outliers`/cropdetect specifically -- see the `avf_borders` Rust
  detector below (`stages.crop.detector`) for arbitrary-color border detection, which was landed
  separately and IS in scope for those.
- **`should_run()` keeps the OLD `reset=0` single-pass behavior** for its cheap ~10s prefilter
  sample (`_detect_crop()`, unchanged) -- it only decides "worth attempting?", not the actual crop
  window, so it doesn't need max_outliers/aggregation precision; `execute()` always re-derives the
  real window via `_detect_crop_full()` (cropdetect) or `_detect_crop_full_rust()` (rust detector),
  whichever `stages.crop.detector` resolves to. This prefilter is skipped entirely (returns `True`
  unconditionally) when the resolved detector isn't `"cropdetect"` -- see the "Mixed Python/Rust"
  section's `avf_borders` writeup above for why (a luma-only prefilter would wrongly skip
  white/colored-bordered videos the rust pass could actually crop).
- **Stage order**: right after `stabilize`, before every other enhancement/AI stage -- see the
  "Pipeline Behavior" note above for why.
- **Config** (`stages.crop`): `limit` (cropdetect luma threshold, default 24), `round` (even-
  dimension rounding, default 2), `min_crop_px` (skip entirely if the detected crop would save
  fewer than this many pixels in BOTH width and height, default 8 -- avoids a pointless 2px crop
  from encoder rounding noise), `analyze_duration_sec` (0 = full-video scan (default); >0 = only
  scan the first N seconds, for very long inputs where a full scan is too slow -- cropdetect path
  only, the rust detector always scans the whole input), `max_outlier_ratio` (default 0.2, see
  above), `transition_max_run_sec` (default 2.0), `transition_window_sec` (default 4.0),
  `transition_tolerance_px` (default 16, see aggregation above), `vlm_check` (default false),
  `vlm_policy` (`"warn"` default | `"skip"`), `detector` (`"auto"` default | `"rust"` |
  `"cropdetect"`, see "Mixed Python/Rust" above), `border_tolerance` (max per-channel absolute
  color difference for the rust detector's color matching, default 24), `border_majority`
  (fraction of a line that must match for the rust detector's inward walk to continue, default
  0.80 -- tolerates overlays up to 20% of a line, matching max_outlier_ratio's 0.2 default), `border_solidity_min` (minimum strip-match fraction for the rust detector to treat an edge
  as having a solid border at all, default 0.60), `border_strip_px` (depth of the outer strip the
  rust detector samples for its dominant-color/solidity signal, default 4), `sample_fps` (rust
  detector only: 0 = every decoded frame (default), >0 samples at that rate via an ffmpeg `fps=`
  filter).
- **Rust detector INFO logging**: when `execute()` uses the rust detector, it logs (at INFO, same
  level as the frames/runs/final-window summary above) a per-edge color+solidity summary, e.g.
  `crop: [rust detector] borders: top 140px solid #000000 (99.8%), bottom 140px solid #000000
  (99.7%), left none, right none` -- both signals the user asked for (color, and whether the border
  is consistent/solid) are visible in every run, not just buried in stage metadata. The
  representative color/solidity per edge is taken from whichever surviving (non-transition) sampled
  frame's own detection reaches the final aggregated extent on that edge.
- **`should_run()` vs. `execute()`**: `should_run()` does a cheap ~10s cropdetect pre-filter
  against `input_info`'s filepath purely to skip obviously-nothing-to-do cases early; it is NOT
  authoritative. Per "Pipeline Behavior"'s per-stage re-probing, that filepath is the actual file
  this stage's `execute()` will receive (e.g. the post-stabilize intermediate, border and all) --
  not a stale original-input snapshot. `execute()` always re-runs a full cropdetect pass
  (respecting `analyze_duration_sec`) against the actual file it's handed and can independently
  return `SKIPPED` (e.g. "cropdetect produced no result", "saves only Nx Mpx, below
  min_crop_px", or an invalid/larger-than-input crop window) -- a false "proceed" from
  `should_run()` is harmless (`execute()` still catches it), a false "skip" (this quick sample's
  short window missing a border that only becomes visible later in the video) is the accepted
  risk of keeping the pre-filter cheap.
- **VLM assist** (`stages.crop.vlm_check`, requires `analysis.vlm.enabled: true`): a watermark/
  logo positioned relative to a letterboxed frame (i.e. sitting in the border area cropdetect
  would otherwise remove) can fool naive cropdetect either way -- brightness above `limit` widens
  the kept region to "protect" it, or a dim watermark below `limit` gets cropped away along with
  the border with no way for cropdetect alone to know that was meaningful content. When enabled,
  one frame is rendered twice (plain, and the same frame with the proposed crop box drawn via
  `drawbox`) and sent to the VLM with a fixed internal prompt asking whether the region OUTSIDE
  the box contains meaningful content (`core.analysis.run_crop_vlm_check`, its own system/user
  prompt pair -- `analysis.vlm.prompt_override`/`prompt_append` do NOT apply here). Parsed with a
  tolerant JSON-then-yes/no-text fallback (`_parse_crop_vlm_response`). If the VLM says yes:
  `vlm_policy: "warn"` (default) logs a WARNING and crops anyway; `"skip"` skips cropping this
  video entirely. **Fails open** on any VLM error (unreachable endpoint, exception, unparseable
  response) -- logs a WARNING and proceeds with the plain cropdetect result, never blocks on VLM
  availability. Note: the ORIGINAL intent (per the user) was for the VLM to SHRINK the detected
  window to exclude a non-content overlay sitting in the border area; `max_outlier_ratio`'s
  outlier-tolerant detector now handles that numerically (see above), so this VLM check is an
  optional secondary safety verification, not the primary overlay defense.
- **`"expand"` policy was considered and rejected**: growing the crop box to include only the
  flagged region isn't feasible because VLMs don't reliably return pixel coordinates for what they
  flag -- there's nothing to expand *to*. Only `"warn"`/`"skip"` are implemented.
- **CLI**: `--enable-stage crop` (or `stages.crop.enabled: true` in config/preset) opts in;
  `--crop-limit INT` overrides `stages.crop.limit` for the run (still requires the stage to be
  enabled some other way -- it's a threshold override, not an enable flag).
- **Aspect-ratio interaction**: crop always runs before `upscale`, so `quality.quality_target`
  dimensions apply to the already-cropped frame -- the target resolution is a post-crop target,
  which is almost always what's wanted (the user wants the *content* at the target resolution, not
  the pre-crop letterboxed frame).

## Rust rewrite candidates

Three CPU-hot-path targets are scoped for PyO3/`maturin`-based Rust rewrites, prioritized soon on
the roadmap. **Scene-detection frame differencing** (`_detect_scene_changes()`, `core/
analysis.py`) is **implemented** — see "Mixed Python/Rust" above and `rust/avf_scenes/`. Still
planned: perceptual-hash duplicate detection (`compute_video_hash`/`compute_video_dhash`, `core/
analysis.py`) and the chunked AI frame prefetch/writer threading (`ai/frame_processor.py`). See
`docs/REQUIREMENTS.md`'s "5. Rust rewrite candidates" section for full scope, interface
signatures, and verification bars per target, and `docs/ROADMAP.md`'s "Planned" section for
priority context. Two other pieces (FFmpeg subprocess orchestration, AI inference calls
themselves) were explicitly considered and rejected as Rust targets — see REQUIREMENTS.md for why
before proposing them again absent a new non-performance justification.

## CLI Flags

Global (before the subcommand):
- `--verbose, -v`: Enable DEBUG level logging (console and the always-on log file)
- `--log-level LEVEL`: Set console logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file PATH`: Write the run's log file here instead of the automatic timestamped file in
  the platform log dir (see "Configuration" above)
- `--file-log-level LEVEL`: Log level for the file log, if different from the console level
- `--config PATH` (env: `AVF_CONFIG`): Use this config file instead of the default platform path;
  must already exist (hard error if not)

`avf process`:
- `--preset, -p NAME_OR_PATH`: Apply a preset by registered name or a path to a preset file
  (can repeat — see "Config cascade" above for the interleaved `--preset`/`--config` layer order
  and name-vs-path disambiguation)
- `--config PATH`: Merge an additional config file on top of the base config as its own layer
  (can repeat, interleaved with `--preset` in command-line order — see "Config cascade" above;
  NOT the same as the top-level `avf --config PATH process ...` flag, which selects the base
  config file itself)
- `--set KEY=VALUE`: Set a config key directly via dot-notation (e.g.
  `stages.upscale.ai_model=RealESRGAN_x2plus`), value parsed as a YAML scalar (can repeat; see
  "Config cascade" above — always applied as part of the final CLI-flags layer, secret-looking
  keys rejected)
- `--output, -o DIR`: Output directory
- `--output-name NAME`: Explicit output filename (only valid with exactly one input file)
- `--recursive, -r`: Scan directories recursively
- `--stage NAME`: Run only these stage(s), replacing the preset/auto-determined list (can
  repeat); also forces `stages.<name>.enabled=false` to be bypassed for explicitly-requested
  stages (see `BaseStage._force_enabled` in `core/stages/base.py`)
- `--enable-stage NAME` / `--disable-stage NAME`: force-enable/disable a stage on top of the
  preset/default set (can repeat) — unlike `--stage`, this doesn't replace the whole stage list
- `--dry-run`: Show what would be done without processing
- `--list-presets`: List available presets
- `--ai` / `--no-ai`: Force AI or traditional methods (`general.use_ai`)
- `--ai-fallback` / `--no-ai-fallback`: Allow/forbid AI stages from silently falling back to
  traditional methods when the AI path can't run (`general.ai_fallback`; see "AI-fallback
  policy" above) — does not override a per-stage `stages.<name>.ai_fallback` in config
- `--overwrite` / `--no-overwrite`: Allow overwriting an existing output file
- `--fps FLOAT`: Target output framerate (`quality.quality_target.target_framerate`)
- `--resolution WIDTHxHEIGHT`: Target output resolution, e.g. `3840x2160`
  (`quality.quality_target.target_resolution`)
- `--codec` / `--audio-codec` / `--crf` / `--encoder-preset`: encode-stage overrides
  (`encoding.video_codec`/`audio_codec`/`crf`/`preset`) — `--encoder-preset` is NOT the same as
  `--preset` (which selects a named avf processing bundle)
- `--hwaccel {auto,none,cuda,vaapi,qsv,vulkan,videotoolbox}`: FFmpeg hwaccel for encode/decode
  (`ffmpeg.hwaccel`)
- `--gpu-device {auto,cpu,cuda,mps}`: PyTorch device for AI stages (`gpu.preferred_device`) —
  separate from `--hwaccel`
- `--threads N`: sets `general.max_concurrent_jobs`
- `--scene-mode` / `--no-scene-mode`: run `stabilize`/`interpolate` per-scene instead of
  whole-video (`scenes.enabled`) — see "Scene mode" above
- `--drop-non-content` / `--no-drop-non-content`: with `--scene-mode`, drop scenes flagged as
  non-content by the VLM+coordinator pass (`scenes.drop_non_content`) — fails open on any
  failure, see "Scene mode" above
- `--crop-limit INT`: cropdetect luma threshold override for the auto-crop stage
  (`stages.crop.limit`) — auto-crop itself is still opt-in, enable it via `--enable-stage crop`
  or `stages.crop.enabled: true` in config; see "Auto-crop" above
- `--zoom-coverage FLOAT`: fraction of frames (0.0-1.0) that should end up border-free once the
  stabilize stage's zoom gate decides zoom applies at all (`stages.stabilize.zoom_coverage`) —
  see "Stabilization zoom coverage" above
- `--batch-size INT` / `--tile-batch-size INT`: set `stages.{upscale,deblock,denoise_video}.
  batch_size`/`tile_batch_size` (all three at once) — see "Batched inference" above

`avf analyze PATHS...`: like `process`, accepts multiple files and/or directories (directories
scanned via `scan_directory`, hidden files skipped) and analyzes each in sequence; a per-file
failure is logged and skipped, remaining files still run, and the command exits non-zero if any
failed.
- `--vlm` / `--no-vlm`: Enable/disable VLM analysis
- `--events` / `--no-events`: Enable/disable event/scene detection
- `--classify`: Classify detected events with VLM
- `--clip DIR` / `--no-clip`: Extract detected scenes as clips to a directory
- `--recursive, -r`: Scan directories recursively (same as `process`)
- `--scene-threshold FLOAT`: Override `analysis.event_detection.scene_change_threshold` (default
  0.15) for this run -- mean fractional per-pixel luma change between consecutive downscaled
  frames, 0-1; lower is more sensitive (risks false positives from motion/noise), higher only
  catches hard cuts. See `_detect_scene_changes()` in `core/analysis.py` for the full metric
  writeup and calibration data (0.3, the previous default, under-detected real footage).
- `--min-scene-duration FLOAT`: Override `analysis.event_detection.min_scene_duration_sec`
  (default 2.0) for this run -- doesn't affect cut *detection*, only whether a cut close to the
  previous one gets its own scene vs. being merged into the next.
- `--full`: Print the complete, untruncated VLM summary per file in a Rich panel below the table
  (the table's own "VLM Summary" row is always a truncated preview with a "(use --full for full
  text)" hint). The full summary is also always written at INFO to the log (auto per-run DEBUG
  log file always has it; console has it at INFO too unless `--log-level` raises the threshold).
- `--csv PATH`: Write one row per analyzed video (filepath, filename, duration, resolution,
  framerate, codec, has_video/has_audio/hdr, scenes_detected, `scene_boundaries` (see below), and
  — when VLM ran — the FULL untruncated summary, `;`-joined tags/objects, and content_rating) to
  a UTF-8 CSV. Overwrites `PATH` if it exists; does not append across runs.
- `--prompt-append TEXT`: Extra text appended to the VLM user prompt for this run (overrides
  `analysis.vlm.prompt_append`) -- e.g. job-specific context like "these are trail-camera clips".
- `--prompt-override TEXT`: Replace the VLM user prompt entirely for this run (overrides
  `analysis.vlm.prompt_override`; `--prompt-append`/`prompt_append` still appends after an
  override). Changing the requested output format away from JSON degrades gracefully --
  `_parse_vlm_response()`'s fallback treats non-JSON text as the summary (empty tags/objects/
  rating) instead of erroring. There's no CLI flag for the system prompt -- use
  `analysis.vlm.system_prompt_override` in config.
- `--max-sample-frames INT`: Override `analysis.vlm.max_sample_frames` (default 8) for this run.
- `--sample-interval FLOAT`: Override `analysis.vlm.sample_interval_sec` (default 10.0) for this
  run. (Prior to this flag being added, `sample_interval_sec` in config was never actually read
  -- `run_vlm_analysis()` hardcoded 10.0 -- so this also fixes a dead config key.)
- `--vlm-model TEXT`: Override `analysis.vlm.model` (e.g. `llava`, `gpt-4o`) for this run.
- `--vlm-url TEXT`: Override `analysis.vlm.api_url` for this run. Still goes through the same
  HTTPS-required-for-non-loopback-hosts gate as the config value (`_call_custom_api`'s `urlparse`
  check) -- `analysis.vlm.allow_http` is what permits plain HTTP, and deliberately has no CLI
  flag, same as `analysis.vlm.api_key`; both stay config-file-only (secrets/security posture
  shouldn't be one flag away from a shell history entry).

**Scene-score visibility** (added after a real-world report of a flat scene count across
`--scene-threshold` 0.01-0.30 that turned out to need the raw scores to diagnose): every
`detect_events()`/`_detect_scene_changes()` call now logs at INFO, per file:
  - The effective `threshold`/`min_duration` actually used (config value or CLI override,
    whichever applied) -- confirms an override took effect.
  - A one-line cut-score summary: number of cuts found, min/median/max `diff_score` among them.
  - A "near-miss" line (only if any exist): up to the 10 highest diff_scores that fell *below*
    threshold but *above* threshold/4 -- i.e. plausible cuts a lower threshold would catch, with
    their timestamps, so "would lowering the threshold help, and where?" doesn't require a rerun.
`avf analyze --full` also prints each scene's boundary `diff_score` in the console scene listing
(`(score=0.XXX)`; the default view omits it to stay uncluttered), and `--csv`'s
`scene_boundaries` column has one `t=<seconds>s@<confidence>` entry per scene, `;`-joined.
`SceneEvent.confidence` is the `diff_score` of the cut that *ended* that scene (not started it);
the final scene per video has a fixed placeholder confidence of 0.5 (nothing ends it) -- see
`_scene_boundaries_str()`'s docstring in `cli/cli.py`.

Other subcommands: `avf find-duplicates REFERENCE DIRECTORY [--threshold FLOAT]`,
`avf presets-cmd` (lists presets), `avf gpu-info`, `avf model-info [--model NAME]`,
`avf model-download --model NAME [--url URL] [--force]`.
