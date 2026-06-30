# Auto Video Fixer - Agent Instructions

## Setup & Commands

```bash
uv sync --all-extras        # install core + dev deps
uv run pytest tests/unit/ -v   # unit tests (no FFmpeg needed by default)
uv run pytest tests/ -v -m "integration"  # integration tests (requires FFmpeg)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ --ignore-missing-imports
uv run avf process video.mp4 -p 4k60   # CLI entry
```

CI runs: `ruff check` -> `ruff format --check` -> `mypy` -> `pytest tests/unit/` -> `pytest -m integration` (3.12 only).

## Architecture

```
src/autovideofixer/
├── cli/cli.py              # Click CLI (`avf` entry, pyproject.scripts)
├── config.py               # Config + DEFAULTS
├── core/
│   ├── pipeline.py         # Pipeline orchestrator
│   ├── analysis.py         # VideoAnalyzer, duplicate detection
│   ├── ffmpeg_utils.py     # FFmpeg wrappers
│   ├── presets.py          # Preset definitions
│   └── stages/             # Processing stages
│       ├── base.py         # BaseStage, register_stage(), StageResult
│       ├── detect.py       # "detect" (analysis, priority=1, no output)
│       ├── stabilize.py    # "stabilize"
│       ├── deblock.py      # "deblock"
│       ├── denoise_video.py # "denoise_video"
│       ├── upscale.py      # "upscale"
│       ├── interpolate.py  # "interpolate"
│       ├── normalize_audio.py # "normalize_audio" (+ NormalizeVolumeStage="normalize_volume")
│       ├── encode.py       # "encode"
│       ├── remux.py        # "remux"
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
6. **Register** in `src/autovideofixer/core/stages/__init__.py`: import the class and call `register_stage(MyStage)` at module level. Can use either decorator syntax (`@register_stage`) or bare function call (`register_stage(MyStage)`).
7. Add config to `DEFAULTS["stages"][<name>]` in `src/autovideofixer/config.py`.

**CRITICAL - Stage ordering**: The pipeline does **not** use `DEFAULTS["pipeline"]["default_order"]` from config.py. The actual ordering is hardcoded in `Pipeline.optimize_stage_order()` in `pipeline.py:232`. Changing `DEFAULTS["pipeline"]["default_order"]` has **no effect** on execution order. To change order, modify `pipeline.py:optimize_stage_order()`.

**CRITICAL - Stage name mismatch bug**: The config `default_order` lists `"denoise"` but the actual registered stage name is `"denoise_video"`. This mismatch means the config-level order would skip the denoise stage if it were ever used. The pipeline's own hardcoded order uses the correct name.

**`remux` is not in the default pipeline**. It is only added by `auto_determine_stages()` when the input is MKV (detected by the `detect` stage).

## Pipeline Behavior

- Default: `skip_stage_on_error: true` - pipeline continues on failure using the original input.
- Temp files: created in input directory, cleaned up after successful job.
- `detect` stage: `produces_output = False` (analysis only). All other stages: `produces_output = True`.
- Stage outputs chain: each stage's `output_path` becomes the next stage's `input_path`.
- Final stage writes to `job.output_path`; intermediate stages write to temp files.

## Testing

- Unit tests: `tests/unit/test_*.py` - no FFmpeg required unless they use `tmp_video_file` fixture.
- Integration tests: `tests/integration/` + marked `@pytest.mark.integration` - require FFmpeg.
- Shared fixtures in `tests/conftest.py`.
- Run unit only: `uv run pytest tests/unit/ -m "not integration"`
- Run with coverage: `uv run pytest --cov=autovideofixer`

## Configuration

- Config file: `~/.config/auto-video-fixer/config.yaml` (Linux), macOS/Windows paths in `config.py:get_config_dir()`.
- Data dir (models, cache): platform-appropriate XDG/Home path via `get_data_dir()`.
- Config is read-once at `Config()` construction; `config.set()` marks dirty and `config.save()` writes YAML.

## Gotchas

- **FFmpeg output paths**: Stages must use an explicit output path. Using `output_path or input_path` on failure produces an empty file.
- **Deblock filter params**: FFmpeg's `deblock` filter parameters vary by FFmpeg version; test before committing.
- **Stage disabled check**: `should_run()` must check `self.is_enabled()` which reads from `DEFAULTS["stages"][name]["enabled"]`.
- **`NormalizeVolumeStage`** (name `"normalize_volume"`) and **`NormalizeAudioStage`** (name `"normalize_audio"`) are two separate classes in `normalize_audio.py`.
- **`detect` stage has priority=1** (runs first) and `produces_output=False`. It populates metadata used by later stages.
- **Preset merging**: When a preset is applied (`-p 1080p60`), it merges into config. The merge is recursive - preset values override config values, but config values not in the preset are preserved. Example: preset `stabilize.enabled=True` won't overwrite your `stabilize.smoothness=80`.
- **Stabilization threshold**: The `threshold` config parameter controls when stabilization activates. Test videos with little motion will skip stabilization (avg_value < threshold). Lower threshold to test stabilization effects.
- **vid.stab buffer corruption**: Known bug (github.com/georgmartius/vid.stab#144) affects B-frame sources. Use accuracy=15 (higher values cause FFmpeg return code 222). The pipeline pipes raw YUV420P video between decode and transform to avoid this.

## CLI Flags

- `--verbose, -v`: Enable DEBUG level logging
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file PATH`: Log to file (in addition to console)
- `--preset, -p NAME`: Apply preset (e.g., `1080p60`, `4k60`, `size_reduction`)
- `--stage NAME`: Run specific stage(s) (can repeat)
- `--output, -o DIR`: Output directory
- `--dry-run`: Show what would be done without processing
- `--list-presets`: List available presets
