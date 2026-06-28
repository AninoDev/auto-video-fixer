# Auto Video Fixer - Agent Instructions

## Quick Start

```bash
# Install deps
uv sync --all-extras

# Run tests
uv run pytest tests/unit/ -v
uv run pytest --cov=autovideofixer

# Run CLI
uv run avf process video.mp4 -p 4k60
```

## Architecture

**Core pipeline**: `src/autovideofixer/core/pipeline.py` orchestrates stages from `core/stages/`

**Stage registration**: Stages must be imported and registered in `core/stages/__init__.py`:
```python
from autovideofixer.core.stages.my_stage import MyStage
register_stage(MyStage)
```
Without this, stages won't be discovered by the pipeline.

**FFmpeg required**: All video processing depends on FFmpeg in PATH. Test with `avf gpu-info`.

**Config location**: `~/.config/auto-video-fixer/config.yaml` (Linux), `~/Library/Application Support/auto-video-fixer/config.yaml` (macOS), `%APPDATA%\auto-video-fixer\config.yaml` (Windows)

## Stage Development

1. Create file in `core/stages/`
2. Subclass `BaseStage` from `core/stages/base.py`
3. Implement `execute()` and `should_run()`
4. Add import and `register_stage()` call to `core/stages/__init__.py`
5. Add to `DEFAULTS["pipeline"]["default_order"]` in `config.py`
6. Add config to `DEFAULTS["stages"]` in `config.py`

## Testing

- Unit tests: `tests/unit/test_*.py` (no FFmpeg needed)
- Integration tests: marked with `@pytest.mark.integration` (requires FFmpeg)
- Run all: `uv run pytest tests/`
- Run unit only: `uv run pytest tests/unit/ -m "not integration"`

## Gotchas

- **Stage registration**: Forgetting to import and register a stage in `__init__.py` causes "Unknown stage" errors
- **FFmpeg output paths**: Always use explicit output path, never `output_path or input_path` (creates empty files on failure)
- **Stage failures**: Pipeline continues on failure by default (`pipeline.skip_stage_on_error: true`), uses original input for next stage
- **Debounce filter**: FFmpeg's `deblock` filter parameters vary by version; test before implementing
- **Temp files**: Generated as `.avf_<uuid>_<stage>.mp4` in input directory; cleaned up after successful job

## Dependencies

- **Required**: Python 3.10+, FFmpeg
- **Optional**: PyTorch (AI models), PySide6 (GUI)
- **Install**: `uv sync --all-extras` or `uv sync` (core only)

## CI/CD

- GitHub Actions: `.github/workflows/ci.yml`
- Runs on push/PR to main/develop
- Tests: Python 3.10, 3.11, 3.12
- Linting: ruff check/format
- Type checking: mypy

## Documentation

- **User guide**: `docs/USER_GUIDE.md`
- **Developer guide**: `docs/DEVELOPER.md`
- **API reference**: `docs/API.md`
- **Roadmap**: `docs/ROADMAP.md`
- **Implementation plan**: `docs/IMPLEMENTATION.md`

## Current Status (v0.1.0)

**Working**: CLI, pipeline, 11 stages (detect, stabilize, deblock, denoise, upscale, interpolate, normalize, encode, remux, speed, hdr), presets, video analysis, FFmpeg integration

**TODO**: AI models (Real-ESRGAN, RIFE), VLM integration, Qt GUI, hardware encoding optimization
