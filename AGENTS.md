# Auto Video Fixer - Agent Instructions

## Standing rule: the agent never commits — prepare, the user signs

The user signs every commit with an SSH key that requires a passphrase typed by
hand, so **the agent must not run `git commit` (or `git push`)**. Instead, when
work is ready to land:

1. Stage the change with `git add -A` (or the specific paths).
2. Write the full commit message to a NEW file at
   `$CLAUDE_JOB_DIR/tmp/commit-msg-<short-slug>.txt` (one file per commit; see the
   existing `commit-msg-*.txt` there for the house style — concise subject line,
   blank line, wrapped body with bullet points, then the `Co-Authored-By:` and
   `Claude-Session:` trailers).
3. Give the user the path to that file. They review, commit with it, and push.

This is the standing protocol for the whole project — it applies to every session
regardless of what the task was. Do not assume a fresh session already knows it;
this section is here so it doesn't get missed again.

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

**Gotcha — `uv run` can silently serve a STALE cached wheel of a Rust crate you just edited.**
After changing Rust source, `maturin develop --release` installs the new build, but the next
`uv run <anything>` re-syncs and reinstalls a *cached* wheel built from the pre-edit source,
reverting your change with no warning. Observed concretely while landing the `retime` feature
(REQUIREMENTS.md § 12.6): the `avf_framepipe` frame reader returned the correct 48 frames when
invoked via `.venv/bin/python` straight after a rebuild, then 120 frames again through `uv run`.

Force a genuine rebuild instead:

```
uv sync --all-extras --reinstall-package avf_framepipe   # or avf_scenes / avf_hashing / avf_borders
```

Always re-verify Rust-side behavior **through `uv run`**, not just a direct `.venv/bin/python`
call — otherwise you can "confirm" a fix that the real entry points don't actually have.

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
`denoise_video`/`interpolate`'s stage loops never branch on backend. `frame_processor.py` itself is
**untouched** by this — it's still the fallback implementation and still directly usable by
anything that doesn't need transport-backend selection. Two fallback parity gaps worth knowing:
`write_queue`/`write_queue_depth` only actually varies the Rust backend's channel bound
(`AsyncVideoWriter`'s queue depth is a hardcoded constant `frame_processor.py` wasn't modified to
expose); and the Python fallback's `frames_written()` counts frames as they're *enqueued* rather
than as they're actually flushed to the ffmpeg pipe (both converge to the same final count once
`close()` returns). New config keys `stages.<name>.read_ahead` (default `2`) and
`stages.<name>.write_queue_depth` (default `4`) on `upscale`/`deblock`/`denoise_video`/`interpolate`
plumb into these factories; `chunk_size` (`25`) stays a call-site constant, not a config key.
`interpolate`'s AI/RIFE path used to be the one holdout still buffering its ENTIRE output in a
Python list before writing (see "AI stage temp-encode quality and the chunked streaming path"
below) — it now streams through `get_frame_reader()`/`get_frame_writer()` exactly like the other
three AI stages.

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

## Input file lists & pluggable parsers (`process --from-file`, REQUIREMENTS.md § 10)

`process` accepts input paths from either (or both) of two sources: the positional `PATHS`
argument, and `--from-file PATH` (repeatable) -- a list file whose *contents* are parsed into more
input paths (and, optionally, per-file output paths / recursive overrides). At least one input
must come from one of the two sources or `process` errors with "No video files found." At most
one of `--output-name`/a per-file `--from-file` output may apply -- combining them is an error
(ambiguous which output name wins).

- **Parser selection is stateful, mirroring `--preset`/`--config` interleaving**: `--from-file-parser
  NAME` (default `shlex`) sets the parser used for every `--from-file` that follows it on the
  command line, until the next `--from-file-parser`. True left-to-right argv order is recovered the
  same way as the `--preset`/`--config` cascade (see "Config cascade" below) -- via
  `_scan_process_argv()`/`_repeatable_matches()` -- with the same fallback when argv can't be
  trusted (e.g. a programmatic `CliRunner.invoke()` call): apply the *last* `--from-file-parser`
  value (or `shlex` if none given) to every `--from-file`, in `from_files`' own tuple order.
- **Inline override**: a `--from-file` value of the form `NAME:PATH` (e.g. `csv:manifest.csv`) uses
  parser `NAME` for just that file, regardless of the current stateful parser -- but only when
  `NAME` is an actually-registered parser name, so a real path/URL containing a colon (or a file
  literally named `x:y`) isn't misparsed; otherwise the whole string is the path. See
  `_split_inline_parser()` in `cli/cli.py`.
- An unknown parser name (stateful or inline) is a hard error (`sys.exit(1)`) before any file I/O
  or job creation.
- **Built-in parsers** (`core/input_parsers/`), each turning list-file text into a list of
  `InputSpec(input_path, output_path=None, recursive=None)`:
  - `shlex` (**the default**) -- `shlex.split(text, posix=True)`: whitespace- and newline-separated
    tokens, quotes and backslash escapes honored, `#` NOT treated as a comment (paths may contain
    it). This is deliberately the default because it's exactly what a file manager's
    drag-and-drop-onto-a-terminal paste produces (e.g. Dolphin onto Konsole): a run of
    whitespace/newline-separated, individually-quoted paths.
  - `lines` -- one path per line; blank lines and lines starting with `#` (after stripping) are
    skipped; one matched pair of surrounding quotes is stripped if present. The classic manifest
    style.
  - `csv` -- stdlib `csv` module. Positional columns `input[,output][,recursive]`; if the first
    row's first cell is exactly `input` (case-insensitive), it's treated as a header and columns
    are looked up by name in whatever order. `recursive` cell: `true/1/yes/y` → True,
    `false/0/no/n`/empty → False.
  - `json` -- accepts a JSON array of path strings, a JSON array of `{input, output?, recursive?}`
    objects, or an object `{"inputs": [...either form...]}`. A missing `input` key on an object
    entry is a hard error.
- **Relative-path resolution is the caller's job, not the parser's**: parsers return paths exactly
  as written and are otherwise I/O-free; `process` resolves a relative `input_path`/`output_path`
  against the list file's own directory (`base_dir`), and a relative per-entry `output_path`
  further against the configured output dir (`-o`/`general.output_dir`) if set, else the resolved
  input file's own directory.
- **Directory entries expand** the same as a positional directory path does: `scan_directory()`,
  recursive-ness given by the entry's own `recursive` (csv/json only) if set, else the run's
  `--recursive` flag.
- **How to add a new parser**: create `src/autovideofixer/core/input_parsers/<name>.py`, subclass
  `InputParser` (`core/input_parsers/base.py`), set a unique class-level `name`, implement
  `parse(self, text: str, base_dir: str) -> list[InputSpec]`, decorate the class with
  `@register_parser`. Then import the module in `core/input_parsers/__init__.py` so registration
  runs at package-import time -- the exact same self-registration pattern as
  `core/stages/__init__.py`'s `register_stage()`.

## Pipeline Behavior

- **Stage ordering, omission, and repetition are driven by config**:
  `Pipeline.resolve_stage_order()` (`core/pipeline.py`, called by both `optimize_stage_order()`
  and `execute_job()`) reads `config.get("pipeline", "default_order")` and resolves it into an
  ordered list of `StageOrderEntry` occurrences. The old hardcoded-and-ignored-config behavior is
  gone -- `DEFAULTS["pipeline"]["default_order"]` (and a matching `DEFAULT_STAGE_ORDER` constant
  in `core/pipeline.py`, used only as a fallback when the config key is missing/empty) now
  **is** the actual execution order. Default order:
  `detect, retime, crop, downscale, deblock, stabilize, denoise_video, upscale, interpolate,
  normalize_volume, normalize_audio, speed, hdr, encode`.
  - `retime` runs **immediately after `detect`** (REQUIREMENTS.md § 12): it recovers the input's
    TRUE content cadence (as opposed to the encoded/container framerate) via a single decode-only
    `mpdecimate`+`metadata=print` pass (`core/cadence.py`) and drops padding/duplicate frames
    while preserving genuine VFR timing, so every later stage sees the reduced, honest frame set.
    On by default; SKIPs (never fails) cheaply on an already-honest input. `InterpolateStage`
    prefers `input_info["true_framerate"]` (published by this stage) over the probed `framerate`
    -- this is the actual payoff: a 24-in-60 input targeting 60fps now correctly plans 24->60
    interpolation instead of reading 60->60 and skipping. See "Timestamp safety across stages"
    below for how the rest of the pipeline stays VFR-safe downstream of this stage.
  - `crop` now runs **first** among the enhancement stages, right after `retime` (previously it
    ran after `stabilize`):
    `stabilize`'s zoom is now a real percentile-based "borderless" zoom (not the old motion-guess
    zoom), so it no longer leaves a black border for a later crop to clean up -- there's nothing
    left for a post-stabilize crop to do. Running crop first also means `downscale` (which sits
    right after `crop`, opt-in, off by default) and every other downstream stage size off the
    already-cropped content instead of the full frame, so none of them (especially the AI-capable
    ones) spend compute on pixels about to be cropped away.
  - `deblock` still runs **before** `stabilize`: blocking artifacts come from the source video, so
    deblocking before stabilization's perspective warping keeps the deblock model's input accurate
    (no warped block edges), and gives the stabilizer cleaner detail to track motion against.
    `deblock` now also defaults to the compact, denoise-optimized `realesr-general-wdn-x4v3`
    checkpoint (previously `RealESRGAN_x4plus`) -- it was trained on real-world degradation
    including both compression/blocking AND noise, so this one early AI pass now covers both
    artifact classes. As a result, `denoise_video` now defaults to **disabled**
    (`stages.denoise_video.enabled: false`) -- it stays in `default_order` at its slot (after
    `stabilize`, before `upscale`; omission != disable) for users who want a separate denoise pass
    or set deblock back to an RRDB model. `max_quality` preset restores the old RRDB defaults
    (`RealESRGAN_x4plus` for `deblock`/`upscale`) and keeps `denoise_video` enabled.

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
## Config tooling: `avf config clean | dump | upgrade` (REQUIREMENTS.md § 6.8)

An `avf config` command group (`cli.py`, thin Click wiring only) over pure helpers in
`cli/config_tools.py` (all unit-testable without a `CliRunner`). Each subcommand writes to
`-o/--output` or, with no `-o`, to **stdout** — and stdout is a pure YAML data stream, so all
diagnostics (errors, warnings, backup notices) go to `err_console` (stderr) instead, or piping
to a file would corrupt it.

- **`clean INPUT`**: normalize a config to canonical bare YAML — comments stripped, only the
  keys actually present in INPUT (no DEFAULTS merged), key order preserved. A formatter, not a
  validator (unknown keys pass through); a non-mapping root is a hard error.
- **`dump`**: emit the EFFECTIVE config (DEFAULTS + user `config.yaml` + layers) as bare YAML.
  Accepts the same config-affecting layering `process` does — `--preset`/`--config`/`--set`,
  argv-order-interleaved via `_scan_config_dump_argv()` + `_repeatable_matches()` (the same
  recovery mechanism `process` uses; other `process` flags are NOT part of `dump`). Secrets
  redacted to `***` via `redact_secrets()` unless `--with-secrets`.
- **`upgrade INPUT`**: apply INPUT's leaf values onto a template (`--template`, default the
  packaged `config.example.yaml`) using **ruamel.yaml** round-trip mode so the template's
  comments/order/new-option defaults survive; only leaves INPUT specifies are overwritten
  in place. **Lists are atomic leaves** (e.g. `pipeline.default_order`) — replaced wholesale,
  never merged. Input keys ABSENT from the template (renamed/removed/moved, or a dict-vs-leaf
  type mismatch) → `ConfigUpgradeError` (refuse, exit 1) unless `--drop-unknown`, which omits
  them with a per-key stderr warning. Template keys that exist only inside a comment count as
  absent (documented limitation).
- **File safety** (all three, when `-o` names an existing file): refuse by default (exit 1),
  `--force` overwrites, `--backup` renames the existing file to the first unused `PATH.1`,
  `PATH.2`, … and prints the rename to stderr. `--force`+`--backup` together is an error.
  `config_tools.apply_file_safety()`.
- **Packaged template**: `docs/config.example.yaml` is the human-edited canonical source;
  `src/autovideofixer/data/config.example.yaml` is a byte-identical shipped copy (hatchling
  includes it automatically under the package dir; `default_template_text()` reads it via
  `importlib.resources`) so `upgrade` works without a repo checkout. A unit test asserts the
  two never drift — when you edit the docs file, copy it to `data/` in the same change.

## Log types: raw / clean / both / none (REQUIREMENTS.md § 6.7)

`general.log_type` (config) / `avf --log-type` (CLI flag, next to `--log-file`, on the
top-level `avf` group) controls what the automatic per-run log FILE contains — the CONSOLE is
always raw regardless. Four values: `"raw"` (default, today's behavior — unredacted paths/
endpoints/titles), `"clean"` (PII-substituted), `"both"` (two files), `"none"` (no file
logging at all). Invalid values are rejected at startup (clear message, non-zero exit) —
`config.VALID_LOG_TYPES`, checked both by `validate_output_handling_config()` (programmatic
`Config`/`Pipeline` users) and directly in `cli.py`'s group callback (which runs before any
`Pipeline` exists).

- **The cleaner**: `src/autovideofixer/logclean.py`'s `PIICleaner` (module singleton via
  `get_pii_cleaner()`/`reset_pii_cleaner()`) holds a per-run, thread-safe mapping of KNOWN real
  values → placeholders, substituted (not regex-guessed) in `clean()`: input files →
  `input_video_01.<ext>`, output files → `output_video_01.<ext>` (2-digit, numbered by first
  appearance, extension preserved — includes `Pipeline._rename_mismatched_output()`'s renamed
  outputs), directories → `/path/to/input/` / `/path/to/output/` / `/path/to/config/` /
  `/path/to/logs/` (role-based, not numbered; the logs role covers the run's own log-file
  directory, which would otherwise leak the user's home/state dir in the startup lines),
  VLM/LLM endpoints → `http://vlm-endpoint` (host/port only — path/
  query kept intact so the log still shows which route was hit; not numbered, all endpoints
  share the one placeholder), embedded video titles (`ProbeResult.title`, from
  `format.tags.title` falling back to the video stream's own `tags.title`) →
  `video_title_01`. Substitution composes full paths (directory placeholder + file placeholder)
  ahead of bare directories/basenames via longest-match-first ordering. `/tmp/avf_*` stage temp
  files are never registered by any call site (no PII by construction), so they're always left
  untouched. **v1 limitation**: this only catches values some code path explicitly registered —
  free-text PII embedded in e.g. a DEBUG-logged VLM response is NOT caught and is out of scope.
- **Registration call sites** (all idempotent — redundant registration from multiple call
  sites doesn't affect numbering): `cli.py`'s group callback registers the config directory
  (+ explicit `--config` path's directory) and does a best-effort `sys.argv` scan (existing-file
  tokens → input, `-o`/`--output` value → output directory) before logging the raw
  `"Invocation: ..."` line; `process()` registers VLM/LLM endpoints, `--config` layer
  directories, and the configured output dir before its own "Effective settings"/DEBUG "Full
  effective config" dumps, then each resolved input file + its parent directory right after
  they're collected; `Pipeline.add_job()` registers every job's input/output path (covers GUI/
  programmatic use); `Pipeline.execute_job()` registers the probed input's title right after the
  input probe. **Known v1 gap**: the group callback's `"Invocation:"` line prints raw `sys.argv`
  and fires before `process` has resolved anything beyond the best-effort argv scan above — a
  relative path that only resolves once `process` applies its own logic, or an output path that
  doesn't exist yet, stays raw on that ONE line; every other log line is covered.
- **Wiring**: `logger.py`'s `_CleaningFileFormatter` runs the normal plain
  `_PLAIN_FILE_FORMAT` formatting then `get_pii_cleaner().clean(...)` at emit time (so
  registration only has to happen before the log CALL, not before `setup_logging()`);
  `setup_logging()` grows a `log_type`/`log_suffix_raw`/`log_suffix_clean` param set and builds
  0/1/2 file handlers for `auto_log_file` accordingly, returning the actual path(s) opened (the
  separate, mostly-unused `log_file` param stays unconditionally raw for direct-caller backward
  compatibility — GUI's `setup_logging("INFO")` call is unaffected either way).
- **"both" mode**: `general.log_suffix_raw` (default `""`) / `general.log_suffix_clean`
  (default `"-clean"`) are inserted before the extension when one exists (`run.log` ->
  `run-clean.log`) or appended to the end for an extensionless name — applies to both a custom
  `--log-file` name and the auto-generated timestamped name. Two suffixes resolving to the
  SAME path (e.g. both left empty) is a config error at startup (`setup_logging()` raises
  `ValueError`, `cli.py` reports it and exits non-zero) — never a silent overwrite.
- **Config-cascade timing limitation**: `general.log_type`/`log_suffix_raw`/`log_suffix_clean`
  are only ever read from the BASE config (DEFAULTS + user `config.yaml`, or an explicit
  top-level `avf --config PATH`) — a `process`-level `--set`/`--config`/`--preset` layer does
  NOT affect logging, because `avf`'s group callback attaches log handlers before `process`
  builds its own cascade. Command-line users should use `avf --log-type clean process ...`
  instead (the CLI flag wins over config when both are given); config-file users set
  `general.log_type` in their user `config.yaml`.

## Reporting: per-video stage summary, media info + timing, JSON run report

`docs/REQUIREMENTS.md` § 6.4/6.5/6.6, implemented in `core/reporting.py` (pure, unit-tested
functions) plus new `JobResult` fields and `cli.py` display/write wiring:

- **`core/reporting.py`**: `classify_stage(StageResult) -> str` buckets a stage occurrence into
  `"ran-ai" | "ran-traditional" | "ran-traditional-fallback" | "failed" | "skipped"` — FAILED/
  SKIPPED status wins outright; `metadata["ai_fallback_used"]` (set once, at
  `BaseStage._ai_fallback_or_fail()`'s enabled-fallback branch — the ONLY seam AI→traditional
  fallback flows through) beats `metadata["method"] == "ai"`; everything else COMPLETED is
  `"ran-traditional"`, which also covers analysis-type stages (crop, detect) that report their
  own method string (detector name, `"ffprobe"`) instead of a literal `"ai"`/`"traditional"`.
  Every stage now sets `metadata["method"]` — audited and patched: `crop` (detector used),
  `detect` (`"ffprobe"`), `encode`/`remux`/`speed`/`stabilize`/`normalize_audio` (all
  `"traditional"`, no AI path). Note: `hdr`'s existing `method` kwarg is the tonemap algorithm
  name (`"bt2020"` etc.), an unrelated pre-existing overload of the same key — harmless for
  `classify_stage()` (anything ≠ `"ai"` classifies as traditional) but don't confuse the two.
  Also: `base_stage_name()` (strips a `pipeline.default_order` repeat's `"#N"` suffix),
  `stage_table_rows()`/`job_summary_line()` (per-job console+log content),
  `run_classification_aggregate()`/`run_outcome_aggregate()` (end-of-run counts),
  `format_media_info_lines()` (input-vs-output "field: in -> out" lines — resolution, framerate,
  duration, bitrate, codecs, filesize via `os.path.getsize`), `aggregate_stage_timing()` (totals/
  averages/failed-executions, computed at display time from `stage_results`, never stored —
  average divisor is videos that ran the stage SUCCESSFULLY, never the total job count; failed
  executions excluded and reported separately), `build_run_meta()`/`build_json_report()`/
  `write_json_report()` (§ 6.6 JSON, `default=str` fallback for non-serializable metadata).
- **New `JobResult` fields** (`core/pipeline.py`), populated on EVERY return path including every
  § 6.1-6.3 early terminal: `scene_stats` (`{"total", "kept", "dropped", "dropped_detail"}` or
  `None` when scene mode didn't run), `job_wall_ms` (from the job's turn starting — including
  input probing and the § 6.1/6.2 decision phase — to result finalization), `processing_ms`
  (stage-pipeline portion only; 0.0 for any SKIPPED/early-FAILED job), `reprocessed_mismatch`
  (`True` when § 6.2's rename/overwrite path reprocessed a verified mismatch — threaded out of
  `_decide_existing_output()` via its `_ExistingOutputDecision` return dataclass rather than a
  raw `JobResult | None`, since the caller needs both the terminal-or-None AND this fact).
  `total_duration` (seconds) keeps working as before; `job_wall_ms`/`processing_ms` are the new
  canonical millisecond numbers. `JobResult.output_info` is now actually populated (it existed as
  a field before but nothing ever filled it in) by reusing `input_info` at job-result-construction
  time — `input_info` is already kept fresh by the stage loop's per-stage re-probe, so this is
  free (no second ffprobe on the same file).
- **Console/log**: per-job Rich table (stage | classification | method/fallback | duration | skip
  reason/error) + an "at a glance" summary line, both ALSO logged as plain `logger.info` lines
  (`cli.py::_print_job_report()`) — Rich console output is line-wrapped and unfriendly to grep,
  the plain log file is what verification workflows should use (see `--log-file`). End-of-run:
  `_print_summary()` now also prints/logs the stage-classification aggregate and job-outcome
  aggregate. Per-stage wall-clock duration is logged at INFO right after each stage completes/
  fails in the stage loop, and input/output media info is logged at job start/finish.
- **`reporting.*` config section + CLI flags** (all OFF by default): `stage_timing_per_video` /
  `--stage-timing-per-video` (per-video per-stage duration table), `stage_timing_totals` /
  `--stage-timing-totals` (per-stage totals across the run), `stage_timing_averages` /
  `--stage-timing-averages` (per-stage average per video that ran it) — `cli.py::
  _print_run_stage_timing()`. `report_json` / `--report-json PATH` writes the § 6.6 JSON report
  once at the end of `process` (in a `finally` block around `execute_all()`, so partial-failure
  runs still get a report for whatever finished) — deliberately contains NO aggregates (derivable
  from per-job stage records; storing both invites picking the wrong one during analysis).
- **Feature 6 — live progress bars** (`cli/progress.py::ProgressReporter`, ON by default): two
  independent `rich.progress` bars in one live region sharing the module-level `console` — a BATCH
  bar (total = job count, completed = `jobs_done + current_job_progress` so it moves smoothly
  within a video, not just per-video) and a PER-FILE bar (total = 1.0, reset per job, tracking that
  job's own 0..1 progress). Each is independently gated by `reporting.progress_batch` /
  `reporting.progress_file` (`--progress-batch/--no-...`, `--progress-file/--no-...`, same
  bool-pair pattern as the stage-timing flags above) AND `console.is_terminal` — on-when-TTY:
  piped/redirected/non-TTY output silently gets no bars regardless of config, same as today's
  behavior. `Pipeline.execute_job`'s `progress_callback` fires on every per-stage progress update
  (not just once per stage), so the file bar moves at whatever granularity the currently-running
  stage itself reports. Per-job Rich report tables print via the same `console` DURING the live
  region (`rich.progress` renders `console.print` calls above the bars); the end-of-run summary/
  stage-timing/JSON-report tail prints AFTER the region stops (`with reporter:` wraps only the
  `execute_all()` call, not the `finally` reporting tail) so bars never overwrite it. Concurrency
  caveat: with `general.max_concurrent_jobs` > 1, `progress_callback` calls from multiple jobs
  interleave — there's one file bar, not one per job, so it just reflects whichever job most
  recently reported (switching jobs resets it to that job's own progress); the batch bar is
  unaffected since it always reflects the true completed-job count.
- **`reporting.color` tri-state + `--color/--no-color`** (default `"auto"`): overrides Rich
  colour on the CLI's module-level `console`/`err_console` (`cli.py`) and the logging
  `RichHandler`'s console (`logger.py::setup_logging`) — motivated by `avf process ... | tee
  run.log`, where piping stdout through `tee` makes it a non-TTY pipe and Rich silently drops
  colour even though a real terminal is still watching downstream of `tee`. `"always"` forces
  colour on (`Console(force_terminal=True)`) even off a TTY; `"never"` forces it off
  (`Console(no_color=True)`) even on a real TTY; built via `logger.py::build_console()`, shared
  by both the CLI globals and `setup_logging()` so all three consoles agree. Precedence: CLI flag
  (`process` only) > config file > `"auto"`. Since `--color` only lives on `process` and
  `setup_logging()` runs earlier in the group callback (before `process`'s own preset/config/CLI-
  flag cascade resolves), `main()` applies the base config's value up front and `process`
  retroactively re-applies the fully-resolved value via `logger.py::set_console_color()` (which
  just reassigns `RichHandler.console`, a public attribute) once its own cascade is done — see
  `cli.py::_apply_color_mode()`. **Independent of the live progress bars above**: forcing colour
  on makes `console.is_terminal` True even for piped output, but the bars' gate is `console.
  is_terminal AND sys.stdout.isatty()` — the `isatty()` half still catches it, so
  `reporting.color=always` deliberately does NOT re-enable bars for redirected/piped output.
  Never touches file handlers (`auto_log_file`/`--log-file` stay plain-text always, regardless of
  this setting).

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

### Hybrid RIFE + minterpolate for non-integer AI interpolation targets

RIFE (`ai/wrappers/interpolate.py`'s `interpolate_video(frames, factor, ...)`) is
timestep-conditioned and supports arbitrary INTEGER factors (each inserted frame at timestep
`j/factor`), but it can only exactly reach `current_fps * N` for integer `N` — it cannot hit a
fractional target on its own. minterpolate (`InterpolateStage._minterpolate_filter()`), by
contrast, retimes to ANY target fps exactly via its own `fps=` sub-option plus true
motion-compensated interpolation (`mi_mode=mci`), so a minterpolate pass adds no jitter even when
used purely to retime an already-smooth sequence to a nearby fractional rate.

Before this feature, `InterpolateStage._execute_ai()` computed
`factor = int(target_fps / current_fps)` and forced `factor = 2` whenever that floored to `<= 1`
— so 50→60 always overshot to 100fps (never retimed down to the labeled 60), and 24→60 landed at
48fps, never the requested target. The AI path now plans a "RIFE under, then minterpolate up"
strategy via `InterpolateStage._plan_ai_interpolation(current_fps, target_fps, hybrid_enabled)`
(pure function, unit-tested in `tests/unit/test_interpolate_streaming.py` /
`test_ai_method_selection.py`), gated by `stages.interpolate.hybrid_ai_minterpolate` (default
`True`) / CLI `--interpolate-hybrid` / `--no-interpolate-hybrid`:

- `rife_factor = floor(target_fps / current_fps)`.
- **`rife_factor >= 2`**: run RIFE at `rife_factor`. If `current_fps * rife_factor` still falls
  short of `target_fps` (e.g. 24→60: RIFE factor 2 → 48fps, short of 60), run ONE minterpolate
  finish pass over the RIFE output to retime the remainder to the exact target — only when hybrid
  is enabled. If `current_fps * rife_factor` already lands on `target_fps` (e.g. 30→60, 30→120),
  no finish pass runs.
- **`rife_factor <= 1`** (e.g. 50→60, 24→30, 60→75 — no integer RIFE factor helps without
  overshooting): with hybrid enabled, RIFE is skipped entirely and the AI path delegates straight
  to `_execute_traditional()` (minterpolate alone reaches the exact target) — this is the
  50→60 case. This is a deliberate strategy choice, not an AI-unavailable condition, so it does
  **not** route through `_ai_fallback_or_fail()`; the returned `StageResult.metadata["method"]`
  is accurately `"traditional"`. With hybrid disabled, the pre-hybrid legacy behavior is preserved
  exactly: `rife_factor` forced to 2, RIFE runs, no finish pass (the old overshoot output).

When RIFE does run, the finish pass (if any) is a second `.mkv` temp file (same crash-resilience
rationale as the RIFE temp — see the streaming-writer section above), muxed with the original
audio in place of the RIFE temp; both temps are cleaned up on every success/failure path. Returned
metadata gains `rife_factor`, `minterpolate_finish: bool`, and `fps_out` (the actual final
target) alongside the existing `method`/`model`/`factor`/`frames_in`/`frames_out`.

`target_approach` (`stages.interpolate.target_approach`, default `"under"`; CLI
`--interpolate-target-approach`) governs ONLY this uniform-factor strategy, via a fourth parameter
on `_plan_ai_interpolation(current_fps, target_fps, hybrid_enabled, target_approach)`: `"under"` is
the exact pre-existing logic above (bit-for-bit unchanged); `"over"` picks the smallest integer
factor at or above target (`ceil(target/current)` — always ≥ 2 since `target > current` is
guaranteed by `should_run()`, so unlike `"under"` it never delegates to minterpolate-alone) and
lets the finish pass retime DOWN to the exact target instead of up; `"nearest"` picks whichever of
`floor`/`ceil` lands closer (ties favor `floor`, matching `"under"`). Irrelevant whenever the
adaptive path below actually runs (it always reaches the target exactly and never calls this
function at all).

### Per-gap adaptive AI interpolation (timestamp-aware RIFE)

The hybrid strategy above still applies ONE global RIFE factor to the whole clip. That defeats two
things `retime` (§ 12) was built to fix, together documented as REQUIREMENTS.md § 16: (a) on an
irregular recovered cadence, `nominal_fps` is a mean, so `floor(target/nominal_fps)` routinely
floors to ≤ 1 and RIFE never runs at all — the phone-recorded-at-60fps-published-to-YouTube case
gets ZERO benefit from enabling AI interpolation; (b) even on a regular cadence, a uniform factor
generates the same number of intermediates for a long camera-stall gap as for a short one, so the
one place synthesized motion is most needed gets no extra frames.

**Fix**: resample the recovered timeline directly onto the target-fps grid instead of multiplying
frame count by a constant factor. `ai/wrappers/interpolate.py`'s `resample_plan(in_timestamps,
target_fps, *, eps, direct_synthesis_max, max_intermediates_per_gap, max_gap_sec, gap_fallback) ->
list[PlanEntry]` is a PURE, side-effect-free function (no torch/GPU/video file needed to test it —
see `tests/unit/test_adaptive_interpolation.py`): it builds the output grid `o[j] = t[0] + j/f` for
`j = 0..floor((t[-1]-t[0])*f)`, brackets each `o[j]` between real input frames `t[k] <= o[j] <=
t[k+1]`, computes `local = (o[j]-t[k])/(t[k+1]-t[k])`, and emits a `PlanEntry`: `op="emit"` (a real
frame, `local <= eps` or `>= 1-eps` — no inference) or `op="synthesize"` (RIFE — see below for what
it actually references). This is the grid formulation, not per-gap counting (`n_k =
round(gap_k*f)`) — the grid is **drift-free by construction**: every output timestamp derives from
the global grid, never accumulated gap-by-gap, so rounding cannot walk a long video's duration off
over time. It also hits the target framerate EXACTLY (no minterpolate finish pass needed on this
path — the output IS defined as the target grid) and produces genuinely CFR output, so neither
PyAV nor mkvtoolnix is needed here even though § 12.4b documents them as rejected write-side
options for arbitrary-PTS muxing elsewhere.

**Recursive subdivision for long gaps (REQUIREMENTS.md § 16.7, revised 2026-09-09).** The original
design capped a gap at `max_intermediates_per_gap` (default 8) and repeated the source frame past
it (`gap_fallback="hold"`) — but that made the *worst* inputs (heavily duplicated/VFR footage, the
whole reason this feature exists) look worst, since the longest holds are exactly where synthesized
motion is needed most. Holding reproduces the defect instead of repairing it. Now: a gap needing
more than `direct_synthesis_max` (default 3) intermediates is **recursively bisected** instead —
synthesize the midpoint from the gap's two real frames, then recurse into `[a, mid]`/`[mid, b]`
with each needed timestep's position renormalized into whichever half it falls in, treating `mid`
as a real reference frame for that half. This keeps every actual RIFE call short-range (its quality
degrades badly at large motion/extreme timesteps), at a depth of
`ceil(log2(n / direct_synthesis_max))` — a handful of levels even for a very long gap.

This lives in the plan, not the executor: `PlanEntry` gained `ref_a`/`ref_b`/`ref_local` (what a
`"synthesize"` call actually interpolates between/at — `ref_a`/`ref_b` are a real frame index
(>= 0) or a generated node id (< 0), defaulting to `(k, k+1, local)` via `__post_init__` so a
hand-built `PlanEntry` without them behaves exactly as before subdivision existed), `node_id` (a
`"synth_node"` entry's cache key), and `depth`. `_subdivide_bracket()` is the pure recursive
planner (own docstring in `ai/wrappers/interpolate.py`) — kept separate from `resample_plan()`'s
main loop and independently testable, since the property "`resample_plan()` stays a pure function
exhaustively testable without a GPU" had to survive this change. A `"synth_node"` entry is never
itself an output frame — only a later entry's `ref_a`/`ref_b` reference (see the executor below).
`k`/`local` on every entry still describe the ORIGINAL, unsubdivided bracket — reporting/tests never
need to know about subdivision to read those two fields.

`max_intermediates_per_gap` (default **0 = unlimited**, changed from the old default of 8) and
`max_gap_sec` (default **0.0 = unlimited**, new) are SAFETY VALVES for pathological input (a
corrupt timestamp implying an hours-long "gap"), not the normal path — left unlimited by default so
an ordinary long gap always subdivides. When either fires, `gap_fallback` (default **`"subdivide"`**,
changed from `"hold"`) decides the same as before: `"hold"` repeats the frame, `"blend"` crossfades
(`ai/wrappers/interpolate.py`'s `_linear_blend()`) — a firing valve always overrides subdivision
regardless of `gap_fallback`'s value, and degrades to `"hold"` if `gap_fallback` is still
`"subdivide"` (i.e. the valve wasn't paired with an explicit fallback choice), since "keep
subdividing" isn't a meaningful response to a gap just flagged as pathological. `plan_stats(plan)`
derives `frames_synthesized`/`frames_passed_through`/`gaps_capped` (now genuine safety-valve hits
only) plus two new counters: `gaps_subdivided` (distinct gaps that used subdivision) and
`max_subdivision_depth` (deepest recursion level used anywhere in the plan, 0 if none).

`execute_resample_plan(frame_source, plan, interpolate_fn) -> Iterator[frame]` is the streaming
executor: a two-pointer sliding window (`cur_frame`/`next_frame`) over the REAL frame source that
advances forward-only as plan entries reference increasing frame indices, holding AT MOST those two
real frames at any time, PLUS a small per-gap `node_cache` (cleared whenever `entry.k` changes) for
`"synth_node"` outputs a later entry's `ref_a`/`ref_b` needs — bounded by subdivision's shallow
depth, so this still holds only a handful of frames total. Reintroducing an accumulated output list
here would recreate the RIFE RAM blowup the chunked streaming path (`ai/frame_pipe.py`) was built
to fix in the first place. It degrades to repeating the last available frame (never raises) if the
frame source runs out early — should not happen when the timeline probe matched the reader's actual
output, but a streaming executor must not crash a job over a stale/mismatched timeline.

The timeline itself comes from `core/cadence.py`'s `probe_frame_timestamps(path, config)` — a
decode-only `ffmpeg -i IN -map 0:v:0 -an -sn -vf showinfo -f null -` pass (no `mpdecimate`; every
frame's timestamp is wanted here, not just mpdecimate's survivors), parsed with the same
`_parse_pts_times()` machinery `analyze_cadence()` (§ 12.1) already uses. Lives in `cadence.py`
(not `ffmpeg_utils.py`) purely to reuse that parsing/fail-open machinery. It probes the interpolate
stage's OWN input file (`InterpolateStage._get_adaptive_timeline()`, called with the same
`input_path` `_execute_ai`/`_execute_ai_adaptive` already operate on) rather than a timestamp list
plumbed through `input_info` — correct whether or not `retime` ran (native VFR input, which
`retime` deliberately SKIPs, still gets adaptive interpolation) and reflects retiming applied by
intermediate stages like `speed` (which scales PTS) that a list captured earlier wouldn't see.
Fails open exactly like `analyze_cadence()`: any probe failure, a missing/empty timeline, or (the
one check `probe_frame_timestamps()` itself does NOT do — its caller does) a timestamp count that
disagrees with the frame count `probe()` reports for the same file falls back to the existing
uniform-factor path, logged at INFO — a timeline miss must never fail a job.

`InterpolateStage._execute_ai()` tries the adaptive path FIRST (`stages.interpolate.
adaptive_cadence`, default `true`; CLI `--interpolate-adaptive`/`--no-interpolate-adaptive`),
before even computing the uniform-factor plan — a probe success routes straight to
`_execute_ai_adaptive()`, which duplicates the torch-availability/model-load/resolution-validation
preamble from `_execute_ai()` (it can't reuse that method's control flow directly since the
uniform path's early "skip RIFE, minterpolate alone reaches target" branch doesn't apply here) and
then streams via `execute_resample_plan()` instead of `RIFEInterpolator.interpolate_video()`'s
uniform `timestep = j/factor` loop, writing directly to `get_frame_writer()` at a carrier rate of
`target_fps` (exact, since the output is already CFR at that rate). Reported metadata:
`adaptive: true`, `frames_synthesized`, `frames_passed_through`, `gaps_capped`, `gaps_subdivided`,
`max_subdivision_depth`, `fps_out`, and `timeline_linearized: false` (timing is honoured exactly on this path, never linearized — contrast
with the uniform-factor raw-pipe writer's `nominal_fps` carrier rate, § 12.4b, which DOES linearize
a genuinely irregular cadence). A probe failure/mismatch falls through to the exact same
uniform-factor `_plan_ai_interpolation()` path documented above, unchanged.

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

- **RRDB (`RRDBNet`)** — `RealESRGAN_x4plus`, `RealESRGAN_x2plus`, `RealESRGAN_x4plus_anime_6B`.
  ~16.7M params, the highest-quality restoration. Still the default for `upscale` and
  `denoise_video`; `deblock` now defaults to the compact `realesr-general-wdn-x4v3` instead (see
  below) since it doubles as a denoise pass — set `stages.deblock.ai_model:
  RealESRGAN_x4plus` (the `max_quality` preset already does this) to restore the old RRDB
  behavior. `deblock`/`denoise_video` (both run Real-ESRGAN at scale<=2) and `upscale` (for
  scale<=2 requests) transparently substitute `RealESRGAN_x2plus` whenever the configured model is
  literally `"RealESRGAN_x4plus"` (a strict `==` string check, e.g. `core/stages/deblock.py`,
  `core/stages/denoise_video.py`) — this swap is model-name-string-based and does NOT trigger for
  any other model name, compact or RRDB (so it does not fire for deblock's new compact default).
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
internal temp file (`get_frame_writer()`/`StreamingVideoWriter`), then mux that temp file's video
track against the original audio to produce the stage's real output. Both the temp encode and the
mux pass take **explicit** `crf`/`preset` args instead of relying on implicit defaults:
- The temp encode passes `stages.<name>.temp_crf` (config key, default `16`) and the hardcoded
  preset `"medium"` (not itself a config key). Previously this write had NO `-crf`/`-preset` at
  all, silently landing on libx264's own default (CRF 23).
- The mux pass now uses `-c:v copy` (verified bit-identical to the temp file's video stream via
  stream-hash comparison), NOT a second `-crf 18` re-encode. Previously every AI stage muxed with
  `-c:v libx264 -crf 18`, meaning AI-processed frames were encoded TWICE at two different quality
  levels — a real quality bug (the temp's CRF-23 generation was baked in before the mux's CRF-18
  pass ever saw it) plus a wasted full x264 pass over the whole video for no benefit.
- `interpolate`'s AI/RIFE path had the identical double-encode pattern in its own internal temp +
  mux and got the same fix (also gained its own `stages.interpolate.temp_crf`, default 16).

All AI-capable videos (`upscale`/`deblock`/`denoise_video`/`interpolate`) now go through the
chunked streaming path (`ai/frame_pipe.get_frame_reader()`/`get_frame_writer()`, `chunk_size=25` —
see "Mixed Python/Rust" above for the Rust-backed `avf_framepipe` transport and its Python
fallback) regardless of frame count — the old `total_est > 1000`/`frame_count > 1000` frame-count
thresholds and their full-buffer `extract_frames()` → flat list → `frames_to_video()` routes are
gone from all four stages' `_execute_ai()`/`_run_single_ai_pass()`. That threshold was
resolution-blind: a short but large-resolution (e.g. 4K) clip could still fall under 1000 frames
while materializing its ENTIRE frame set in RAM (a 33s 4K clip is ~25GB uncompressed), with zero
decode/inference/write overlap. Streaming has no measurable downside for short clips either, so
there's no longer a reason to keep two code paths.

**`interpolate` was the last holdout, fixed separately (2026-07-20)**: unlike the other three
stages (which read N frames and write N frames, so a chunk's output size equals its input size),
interpolation's `_execute_ai` writes `factor`x MORE frames than it reads per chunk — the old
`all_interpolated.extend(chunk_interp)` accumulation (chunked path, `frame_count > 1000`) held
every OUTPUT frame in RAM for the whole clip, and the non-chunked path (`<= 1000` frames) called
`extract_frames()` to load the ENTIRE input up front. For a 61s 4K 30→60fps scene that's ~3667
output frames * ~24.9MB ≈ 91GB resident — enough to OOM/swap-thrash the host (confirmed real
incident: hit a 56GB LXC cap, swapped to ~76GB, froze the host until `pkill`), amplified further by
scene mode's up-to-4 concurrent scene workers each holding their own buffer. `_execute_ai` now
streams each chunk straight to `get_frame_writer()` and discards it immediately after
`write_batch()` — peak memory is bounded by `chunk_size * factor` (~50 frames for the default
`chunk_size=25`, factor 2), not `total_frames * factor`. The cross-chunk **carry-frame logic is
unchanged**: the previous chunk's last frame is still prepended (`extended = [carry_frame,
*chunk]`) so a real interpolated frame is generated across chunk boundaries instead of a hard
stutter every `chunk_size` source frames, and `chunk_interp[0]` (the re-emitted carried frame) is
still dropped before writing when a carry was prepended — this logic is interpolation-specific
(the other three AI stages have no equivalent) and is covered by
`tests/unit/test_interpolate_streaming.py`'s `TestStreamingNoAccumulation` (exact total-frame-count
assertions across multi-chunk boundaries, plus a peak-single-batch-size regression guard).

`interpolate`'s AI-path temp file is also now **Matroska (`.mkv`, H.264)**, not `.mp4` — the ONLY
one of the four AI stages using MKV for its internal temp (`upscale`/`deblock`/`denoise_video`
still use `.mp4`, since their fast N-in-N-out completion time makes a truncated temp far less
costly to just re-run). Matroska is written incrementally (clusters flushed as they go), so a
partial file stays playable/recoverable if the process is killed mid-write (OOM, host shutdown,
power loss) — default MP4 only writes its `moov` index at clean finalize, so a truncated MP4 is
unplayable. H.264-in-MKV remains stream-copyable (`-c:v copy`) into the stage's final MP4 output,
so this costs nothing at finalize time — the existing audio-mux finalize pass IS the MKV→MP4
remux, unchanged. Partial-output handling on failure:
- **Hard kill** (SIGKILL/OOM/power loss): the `finally`-based temp cleanup simply never runs, so
  the partial `.mkv` survives in place at its last flushed cluster — this is the primary benefit
  and falls out for free; nothing preemptively deletes it.
- **Graceful in-stage failure** (mux returns nonzero, a write failure, an inference exception):
  instead of silently unlinking the partial, `InterpolateStage._preserve_or_discard_partial_temp()`
  renames it to a visible sibling of the intended output, `<output_stem>_interp_partial.mkv`, and
  logs its path at WARNING ("partial interpolated output preserved for inspection: ...") so the
  user can inspect how far it got — falling back to leaving the temp in place if the rename itself
  fails. Covered by `tests/unit/test_interpolate_streaming.py`'s `TestPartialOutputPreservation`.
- **Success**: the temp is deleted in `finally` once the muxed output supersedes it (unchanged
  behavior).

`extract_frames()`/`frames_to_video()` themselves are untouched in `ai/frame_processor.py` — they
remain the fallback implementation for anything not using `ai/frame_pipe.py`'s transport
selection.

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
benchmark this was retroactively needed for. `interpolate`'s RIFE path streams through the same
`ai/frame_pipe.py` transport now (see "AI stage temp-encode quality and the chunked streaming
path" above) but is still NOT wired up to `StageTimer` — its per-chunk loop shape differs (carry-
frame prepend/drop, `factor`x output growth per chunk) enough that reusing the shared instrumentation
wasn't in scope for the streaming fix; a future pass could add it.

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

## Resolution fit modes & the `downscale` stage (REQUIREMENTS.md § 7)

`core/output_check.py:compute_fitted_dimensions()` is the shared dimension-fitting helper used
by both `UpscaleStage._calculate_target_dimensions()` and `DownscaleStage._target_dimensions()`
(`core/stages/downscale.py`) — one implementation of "fit an input into an orientation-aware
target box" instead of two, so an upscale and a downscale targeting the same box always agree on
the resulting dimensions. It builds on `effective_target_bounds()` (the same rotation-to-
orientation helper `UpscaleStage._effective_target_bounds()` already wrapped) and supports two
`quality.quality_target.resolution_fit_mode` values:

- **`preserve_aspect`** (default): scales the input to fit entirely within the (possibly
  rotated) target box at its exact aspect ratio, then rounds each dimension UP to
  `dimension_multiple` (default 2, i.e. `_round_to_even`'s "round up if odd" rule generalized).
  This is `UpscaleStage`'s original behavior, unchanged when `dimension_multiple == 2` — the
  truncate-then-round-up can land a few px short of a clean target (e.g. a 1920×1080 target on
  an off-aspect input coming out 1918×1080 or 1920×1078).
- **`snap_limiting`** ("snap-to-box-when-close"): the LIMITING axis (the one whose bound/input
  ratio is smaller — i.e. the axis that binds the preserve_aspect min-fit scale) always lands
  EXACTLY on its target bound value. The OTHER (derived) axis's exact aspect-preserving float
  value (`input_other * scale`) is at most its own bound; the gap between that value and the
  bound, as a fraction of the bound, is compared against `snap_tolerance` (default 0.01 = 1%).
  If the gap is `<= snap_tolerance`, the derived axis is ALSO snapped exactly onto its bound —
  both dimensions land on the full target box, accepting a sub-percent aspect-ratio shift for a
  clean standard resolution (e.g. a 1440×812 input against a `[1920, 1080]` target snaps to
  exactly 1920×1080 instead of preserve_aspect's ~1920×1078). Beyond that tolerance (a genuinely
  different aspect ratio, not just off-by-a-few-px), the derived axis is left at its
  aspect-preserving value instead, rounded to the NEAREST `dimension_multiple` — no over-eager
  snapping for inputs that really don't match the target's aspect ratio (e.g. a 3840×2106 input
  against `[1920, 1080]` — a 2.5% gap — still comes out 1920×1052, not snapped to 1920×1080).

`quality.quality_target.dimension_multiple` (default 2, required by H.264/yuv420p) controls the
rounding granularity for both fit modes and both stages. `quality.quality_target.snap_tolerance`
(default 0.01) only affects `snap_limiting`. All three are overridable per-run via
`avf process --resolution-fit-mode {preserve_aspect,snap_limiting}` / `--dimension-multiple N` /
`--snap-tolerance FLOAT`.

The **`downscale`** stage (`core/stages/downscale.py`, `stages.downscale.enabled`, default
`false` — strictly opt-in) is the mirror image of `upscale`: pure FFmpeg (lanczos), no AI path,
shrinks an OVERSIZED input down to the target resolution box. It sits in `pipeline.default_order`
immediately after `crop` and before `denoise_video`/`upscale`/etc. — running before the heavier
stages means they never spend compute on pixels that would just be discarded downstream anyway.

`downscale` and `upscale` are complementary, not redundant: for a given target box,
- an OVERSIZED input: `downscale` shrinks it to target; `upscale` then sees an input already at
  target and skips ("Already at target resolution").
- a SMALL/at-target input: `downscale.should_run()` skips it (`binding_ratio >= 1.0` → "Input
  already at or below target resolution", or within `SKIP_SCALE_THRESHOLD` tolerance → "Within
  downscale tolerance") — `downscale` never upscales; `upscale` (if enabled) runs as usual.

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
   `--gpu-device`, `--scene-mode`, `--drop-non-content`, `--crop-limit`, `--downscale`,
   `--retime`/`--no-retime`, `--retime-min-duplicate-ratio`, `--output-timing`,
   `--resolution-fit-mode`, `--dimension-multiple`, `--snap-tolerance`, `--zoom-coverage`,
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

- **Timestamp safety downstream of `retime` (REQUIREMENTS.md § 12.4)**: once an intermediate is
  VFR, any stage that silently re-conforms it to CFR re-inserts the duplicate frames `retime`
  just removed. Filter-only ffmpeg stages (`crop`/`downscale`/`denoise_video`/`deblock`
  traditional/`hdr`/`speed`) carry an explicit `-fps_mode passthrough` via the shared
  `ffmpeg_utils.timing_output_args()` helper. Raw-frame-pipe stages (`upscale`/`deblock` AI,
  `interpolate` AI/RIFE, `stabilize`'s manual decode/transform pipe) are the dangerous class: the
  **reader**'s ffmpeg args must also carry `-fps_mode passthrough` -- without it, ffmpeg silently
  re-expands a VFR input back to CFR by duplicating frames on the way into the rawvideo pipe
  (verified: 48 real frames -> 120 output frames), negating the whole feature at full AI-inference
  cost. `-fps_mode` is an OUTPUT option -- it must be placed after `-i`, never before (corrupts
  input parsing). Also do **not** re-probe for fps on a file downstream of `retime`:
  `avg_frame_rate` is unreliable on a VFR intermediate (empirically a 48-frame 2s VFR MKV still
  advertised `avg_frame_rate=60/1`) -- take the rate from `input_info["true_framerate"]` instead.
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

`stages.stabilize.zoom_coverage` (float, default `1.0`) is a **SIGNED dial** (REQUIREMENTS.md
§ 17) that tunes how the stabilize stage's zoom compensates for borders introduced by
stabilization, once `zoom_enabled` / `zoom_threshold`'s movement-extent gate (`apply_zoom` in
`StabilizeStage.execute()`) has already decided zoom applies at all -- `zoom_coverage` does NOT
change that gate, only what zoom is used once it fires (§ 17.1: `zoom_enabled: false` still means
no zoom of any kind, never redirected to zoom-out behaviour).

**BREAKING (2026-09)**: this used to mean "fraction of frames that end up border-free", with
`0.0` short-circuiting to "no zoom at all" (`optzoom=0, zoom=0`). That was a bug: no zoom is not
no cropping -- with `zoom=0` a stabilized frame is still translated/rotated by the smoothing path,
so it loses content off one edge while showing a border on the other. Every shaky frame ended up
both bordered *and* cropped, which is not what a user setting `0.0` (expecting "nothing is ever
cropped") wanted. `zoom_coverage` is now signed, spanning zoom-OUT as well as zoom-IN. The OLD
`0.0` behaviour (no zoom at all) now lives at `0.5`; anyone who set `0.0` expecting "leave it
alone" must change it to `0.5` -- `avf config upgrade` does NOT rewrite this for you, since it
cannot infer old intent (it also does not silently touch any existing `zoom_coverage: 0.0`).

Let `B_i` be the existing per-frame required-zoom-IN percentage that would just eliminate frame
`i`'s border (always `>= 0`; see `StabilizeStage._compute_required_zoom_percentages()`). The same
magnitude negated, `-B_i`, is the zoom-OUT needed to keep frame `i`'s content fully intact. The
dial (`StabilizeStage._signed_zoom_from_coverage()`):

| `zoom_coverage` | `zoom=`   | Meaning                                                          |
|------------------|-----------|-------------------------------------------------------------------|
| `0.00`           | `-max(B)` | 100% of frames keep ALL content; borders everywhere, nothing cropped |
| `0.25`           | `-p50(B)` | ~50% of frames fully preserved                                    |
| `0.50`           | `0`       | no zoom -- `vidstabtransform`'s own default, and the OLD `0.0` behaviour |
| `0.75`           | `+p50(B)` | ~50% of frames border-free                                        |
| `1.00`           | `+max(B)` | 100% border-free, delegated to `optzoom=1` exactly as before -- bit-for-bit unchanged, exact |

Formally: for `q >= 0.5`, `zoom = quantile(B, 2(q - 0.5))`; for `q < 0.5`,
`zoom = -quantile(B, 1 - 2q)`; `q == 0.5` is a hard `0.0` rather than the data-dependent quantile,
so the dial is continuous through zero and matches `vidstabtransform`'s own no-zoom default
exactly at the midpoint. `q = 1.0` keeps delegating to `optzoom=1` exactly as today (no
approximation involved, unlike every other point on the dial), so the default is unchanged. The
emitted percentage is clamped to `vidstabtransform`'s valid `zoom=` range, `[-100, 100]`.

The `q -> signed-zoom` mapping (`_signed_zoom_from_coverage()`/`_zoom_quantile()`) is pure and
exactly unit-testable given a synthetic list of B values -- see
`tests/unit/test_stabilize_zoom.py`. Producing the B values themselves from a TRF
(`_compute_required_zoom_percentages()`) is not exact; see the caveat below.

**Accuracy limitation (state honestly, this is not exact) -- applies to BOTH halves of the dial**:
what actually determines a frame's visible border is `vidstabtransform`'s own
internally-computed SMOOTHED camera path (a function of `smoothing`/`maxshift`/`optalgo`/
`interpol`), which this code has no access to -- it only sees the raw per-block local-motion (LM)
values in the TRF. `_compute_required_zoom_percentages()` integrates those into an approximate raw
cumulative camera-path position, then applies a local moving-average smoothing window (matching
`stages.stabilize.smoothness`, the same window vidstabtransform itself uses) to approximate the
smoothed path, and measures each frame's deviation from that local average as its "required zoom"
(`zoom_pct = 200 * shift_px / dimension_px`, derived from `vidstabtransform`'s `zoom=Z%` scaling
the frame by `1+Z/100` around center). This is **directionally correct and tunable** but not an
exact match to vidstabtransform's internal computation -- individual frames' actual post-smoothing
border requirements (and, by the same token, their actual post-smoothing content-preservation
needs on the zoom-out half) can come out higher or lower than this estimate. This also does NOT
account for rotation's contribution to border size (same limitation as `_movement_extent()` -- no
new rotation math was added). `1.0` remains exact because it delegates to `optzoom=1` rather than
to the estimate; every other point on the dial, including `0.0`, is "our best estimate", not a
mathematical guarantee -- `0.0` is "zoom out by our best estimate of the worst displacement", not
a guarantee that no pixel is ever lost.

**Verification methodology** (ffmpeg n9.0.1, CPU only, no GPU needed): measured the non-black
content box of a stabilized 560x400 frame via `cropdetect`. `optzoom=1` (the `1.0` endpoint)
produced a 560x400 content box, 100% non-black (zoomed in, no border, content cropped) --
confirms the `1.0` endpoint is unaffected by this change. `optzoom=0:zoom=0` (the `0.5` midpoint,
and the OLD, buggy `0.0` behaviour) produced 98.1% non-black -- confirming the bug this section
describes: already bordered *and* cropped simultaneously. `optzoom=0:zoom=-20` (a negative,
zoom-OUT value like the new `0.0` endpoint would compute) produced a 528x370 content box, 83.6%
non-black -- genuinely zoomed out, full frame content preserved with a border, demonstrating the
zoom-out half of the dial behaves as intended.

CLI: `--zoom-coverage FLOAT` (`stages.stabilize.zoom_coverage`), following the `--crop-limit`
precedent of a single per-stage override flag.

### Post-stabilize sharpening is configurable

The `unsharp` pass applied after stabilization (when it actually triggered and
`sharpen_enabled` is true) is no longer a hardcoded filter string. It's built by
`StabilizeStage._build_sharpen_suffix()` from four config keys: `stages.stabilize.sharpen_amount`
(unsharp `luma_amount`, float in [-2.0, 5.0], default **1.0** -- raised from the old hardcoded
`0.5`), `sharpen_luma_size` (`luma_msize_x`/`luma_msize_y`, odd int in [3, 63], default 3),
`sharpen_chroma_amount` (`chroma_amount`, float in [-2.0, 5.0], default 0.0), and
`sharpen_chroma_size` (`chroma_msize_x`/`chroma_msize_y`, odd int in [3, 63], default 3).
Out-of-range values raise a clear `ValueError` naming the offending key instead of failing
inside ffmpeg. CLI: `--sharpen`/`--no-sharpen` (`stages.stabilize.sharpen_enabled`) and
`--sharpen-amount FLOAT` (`stages.stabilize.sharpen_amount`); the matrix-size/chroma knobs are
`--set`-only.

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
- **Stage order**: right after `detect`, first in `default_order` -- see the "Pipeline Behavior"
  note above for why (stabilize's zoom is now borderless, so crop no longer needs to run after it).
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
- `--downscale` / `--no-downscale`: enable/disable the `downscale` stage (`stages.downscale.
  enabled`, off by default), which shrinks an oversized input to the target resolution box
  before the heavier stages run — see "Resolution fit modes & the `downscale` stage" above
- `--retime` / `--no-retime`: enable/disable the `retime` stage (`stages.retime.enabled`, ON by
  default), which recovers a video's TRUE source cadence and drops padding/duplicate frames —
  see REQUIREMENTS.md § 12 and "Pipeline Behavior" above
- `--retime-min-duplicate-ratio FLOAT`: minimum duplicate-frame fraction before `retime` actually
  re-encodes (`stages.retime.min_duplicate_ratio`, default 0.05) — below it the stage SKIPs
- `--output-timing {cfr,vfr,passthrough}`: timing mode for the FINAL `encode` stage only
  (`general.output_timing`, default `cfr`) — see REQUIREMENTS.md § 12.5
- `--resolution-fit-mode {preserve_aspect,snap_limiting}`: how upscale/downscale fit an input
  into the target resolution box (`quality.quality_target.resolution_fit_mode`) — see
  "Resolution fit modes & the `downscale` stage" above
- `--dimension-multiple INT`: rounding granularity for upscale/downscale output dimensions
  (`quality.quality_target.dimension_multiple`, default 2)
- `--snap-tolerance FLOAT`: with `--resolution-fit-mode snap_limiting`, how close (fractional,
  default 0.01 = 1%) the non-limiting axis must be to the target box before it's snapped exactly
  onto it (`quality.quality_target.snap_tolerance`) — see "Resolution fit modes & the
  `downscale` stage" above
- `--zoom-coverage FLOAT`: fraction of frames (0.0-1.0) that should end up border-free once the
  stabilize stage's zoom gate decides zoom applies at all (`stages.stabilize.zoom_coverage`) —
  see "Stabilization zoom coverage" above
- `--batch-size INT` / `--tile-batch-size INT`: set `stages.{upscale,deblock,denoise_video}.
  batch_size`/`tile_batch_size` (all three at once) — see "Batched inference" above
- `--stage-timing-per-video` / `--stage-timing-totals` / `--stage-timing-averages`: three
  independent, OFF-by-default stage-timing console/log views (`reporting.stage_timing_*`) — see
  "Reporting" above
- `--report-json PATH`: write the § 6.6 structured JSON run report once at the end of the run
  (`reporting.report_json`) — see "Reporting" above
- `--progress-batch/--no-progress-batch` / `--progress-file/--no-progress-file`: independent,
  ON-by-default (but only rendered when `console.is_terminal`) live progress bars
  (`reporting.progress_batch` / `reporting.progress_file`) — see "Feature 6 — live progress bars"
  above
- `--color/--no-color`: force Rich console colour on/off, e.g. to keep colour through `| tee
  run.log` (`reporting.color`, default `"auto"`) — independent of the progress bars above; see
  the `reporting.color` tri-state bullet above

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
