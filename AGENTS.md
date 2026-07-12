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

- **Stage ordering is hardcoded** in `Pipeline.optimize_stage_order()` (`core/pipeline.py`). Changing `DEFAULTS["pipeline"]["default_order"]` in config has **no effect**.
- **Stage name mismatch**: config `default_order` lists `"denoise"` but the registered name is `"denoise_video"`. Ignoring `default_order` avoids the bug.
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

## Output Path Resolution

Output path is resolved in this priority order:
1. `job.output_path` explicitly set (e.g., via `add_job(input, output)`)
2. `config.get("general", "output_dir")` + `{stem}_enhanced{ext}` — set via CLI `-o` or config.yaml
3. `{input_dir}/{stem}_enhanced{ext}` (fallback)

The CLI `-o` flag sets `general.output_dir` in config. The pipeline reads it when creating jobs.

## AI/Traditional Method Selection

Stages with AI alternatives (upscale, denoise_video, interpolate, deblock) check `config.get("general", "use_ai")`:
- `None` (default) — use preset or auto-detect
- `True` (`--ai`) — force AI methods
- `False` (`--no-ai`) — force traditional methods (much faster, no GPU needed)

CLI: `--ai` / `--no-ai` (mutually exclusive flag_value pattern). Without either flag, falls back to preset/config.

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

## Upscaling & Aspect Ratio

The upscale stage respects `quality.quality_target.keep_aspect_ratio` (default `True`):
- Rotates the preset bounding box to match input orientation (portrait input swaps w↔h)
- Scales to fit within the (possibly rotated) bounding box, preserving pixel count
- Square input uses the shorter preset edge for both dimensions
- Dimensions are rounded to even values (H.264 requirement)

Example: 1080p60 preset (1920×1080) + 9:16 portrait input → scales to ~1080×1920 (portrait).

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
- Preset merging is recursive — preset values override config, but config values not in preset are preserved.
- A full, commented example covering every `Config.DEFAULTS` key lives at
  `docs/config.example.yaml`.

## Gotchas

- **stabilize.py `_get_video_dimensions` / `_get_video_framerate`**: Must use `stdout=subprocess.PIPE` (not `subprocess.DEVNULL`). Using DEVNULL discards output and forces fallback to 1920×1080 / 30fps, stretching portrait video to landscape.
- **stabilize pipe sizing**: The decode subprocess **must** include `-s {width}x{height}` to match the transform's `-s` input. Without it, anamorphic or non-standard resolution videos produce corrupted output due to stride mismatch.
- **vid.stab B-frame corruption**: Known bug (github.com/georgmartius/vid.stab#144). The pipeline pipes raw YUV420P between decode and transform to avoid it. Use accuracy=15 (higher causes FFmpeg return code 222).
- **FFmpeg output paths**: Stages must use an explicit output path. Using `output_path or input_path` on failure produces an empty file.
- **Stage disabled check**: `should_run()` must check `self.is_enabled()` which reads from `DEFAULTS["stages"][name]["enabled"]`.
- **`NormalizeVolumeStage`** (name `"normalize_volume"`) and **`NormalizeAudioStage`** (name `"normalize_audio"`) are two separate classes in `normalize_audio.py`.
- **Preset stage enable**: Presets define `enable_stages` which controls which stages run. A stage not listed in a preset's `enable_stages` will not execute, even if it's in the default order.

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
- `--preset, -p NAME`: Apply preset (e.g., `1080p60`, `4k60`, `size_reduction`)
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
  framerate, codec, has_video/has_audio/hdr, scenes_detected, and — when VLM ran — the FULL
  untruncated summary, `;`-joined tags/objects, and content_rating) to a UTF-8 CSV. Overwrites
  `PATH` if it exists; does not append across runs.
- `--prompt-append TEXT`: Extra text appended to the VLM user prompt for this run (overrides
  `analysis.vlm.prompt_append`) -- e.g. job-specific context like "these are trail-camera clips".
- `--prompt-override TEXT`: Replace the VLM user prompt entirely for this run (overrides
  `analysis.vlm.prompt_override`; `--prompt-append`/`prompt_append` still appends after an
  override). Changing the requested output format away from JSON degrades gracefully --
  `_parse_vlm_response()`'s fallback treats non-JSON text as the summary (empty tags/objects/
  rating) instead of erroring. There's no CLI flag for the system prompt -- use
  `analysis.vlm.system_prompt_override` in config.

Other subcommands: `avf find-duplicates REFERENCE DIRECTORY [--threshold FLOAT]`,
`avf presets-cmd` (lists presets), `avf gpu-info`, `avf model-info [--model NAME]`,
`avf model-download --model NAME [--url URL] [--force]`.
