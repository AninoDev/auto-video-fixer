# Auto Video Fixer - Agent Instructions

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
│   ├── analysis.py         # VideoAnalyzer, duplicate detection
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

- **Stage ordering is hardcoded** in `Pipeline.optimize_stage_order()` (pipeline.py:238). Changing `DEFAULTS["pipeline"]["default_order"]` in config has **no effect**.
- **Stage name mismatch**: config `default_order` lists `"denoise"` but the registered name is `"denoise_video"`. Ignoring `default_order` avoids the bug.
- `remux` is **not** in the default pipeline. It is only added by `auto_determine_stages()` when the input is MKV.
- `detect` stage has `priority=1` (runs first) and `produces_output=False`.
- Default: `skip_stage_on_error: true` — pipeline continues on failure using the original input.
- Temp files: created in input directory, cleaned up after successful job.
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

## Upscaling & Aspect Ratio

The upscale stage respects `quality.quality_target.keep_aspect_ratio` (default `True`):
- Rotates the preset bounding box to match input orientation (portrait input swaps w↔h)
- Scales to fit within the (possibly rotated) bounding box, preserving pixel count
- Square input uses the shorter preset edge for both dimensions
- Dimensions are rounded to even values (H.264 requirement)

Example: 1080p60 preset (1920×1080) + 9:16 portrait input → scales to ~1080×1920 (portrait).

## Configuration

- Config file: `~/.config/auto-video-fixer/config.yaml` (Linux), macOS/Windows paths in `config.py:get_config_dir()`.
- Data dir (models, cache): `get_data_dir()`.
- Config is read-once at `Config()` construction; `config.set()` marks dirty and `config.save()` writes YAML.
- Preset merging is recursive — preset values override config, but config values not in preset are preserved.

## Gotchas

- **stabilize.py `_get_video_dimensions` / `_get_video_framerate`**: Must use `stdout=subprocess.PIPE` (not `subprocess.DEVNULL`). Using DEVNULL discards output and forces fallback to 1920×1080 / 30fps, stretching portrait video to landscape.
- **stabilize pipe sizing**: The decode subprocess **must** include `-s {width}x{height}` to match the transform's `-s` input. Without it, anamorphic or non-standard resolution videos produce corrupted output due to stride mismatch.
- **vid.stab B-frame corruption**: Known bug (github.com/georgmartius/vid.stab#144). The pipeline pipes raw YUV420P between decode and transform to avoid it. Use accuracy=15 (higher causes FFmpeg return code 222).
- **FFmpeg output paths**: Stages must use an explicit output path. Using `output_path or input_path` on failure produces an empty file.
- **Stage disabled check**: `should_run()` must check `self.is_enabled()` which reads from `DEFAULTS["stages"][name]["enabled"]`.
- **`NormalizeVolumeStage`** (name `"normalize_volume"`) and **`NormalizeAudioStage`** (name `"normalize_audio"`) are two separate classes in `normalize_audio.py`.
- **Preset stage enable**: Presets define `enable_stages` which controls which stages run. A stage not listed in a preset's `enable_stages` will not execute, even if it's in the default order.

## CLI Flags

- `--verbose, -v`: Enable DEBUG level logging
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file PATH`: Log to file (in addition to console)
- `--preset, -p NAME`: Apply preset (e.g., `1080p60`, `4k60`, `size_reduction`)
- `--stage NAME`: Run specific stage(s) (can repeat)
- `--output, -o DIR`: Output directory
- `--dry-run`: Show what would be done without processing
- `--list-presets`: List available presets
- `--ai` / `--no-ai`: Force AI or traditional methods
