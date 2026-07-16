# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Detailed agent-facing instructions already live in `AGENTS.md` (commands, stage-development
checklist, output path resolution, AI/traditional method selection, known bugs/gotchas). Read
`AGENTS.md` first — this file adds context that sits above it.

## What this project is

Auto Video Fixer (`avf`) is a Python video enhancement pipeline: FFmpeg-based traditional
processing (stabilize, deblock, denoise, encode, HDR, speed, audio normalization) plus optional
PyTorch AI stages (Real-ESRGAN upscaling/denoising, RIFE frame interpolation), driven by a CLI
(Click) and an optional Qt GUI. Video analysis includes scene detection, VLM-based classification
(Ollama or OpenAI-compatible vision APIs), and perceptual-hash duplicate detection.

## Commands

See `AGENTS.md` → "Setup & Commands" for the canonical list (`uv sync`, `uv run pytest`,
`uv run ruff`, `uv run mypy`, `avf process ...`). CI order: ruff check → ruff format --check →
mypy → pytest unit → pytest integration (3.12 only).

Run a single test: `uv run pytest tests/unit/test_pipeline.py::TestClassName::test_name -v`

## Architecture

The package lives under `src/autovideofixer/` (installed via hatchling, `pyproject.toml` maps
`packages = ["src/autovideofixer"]`). Three top-level surfaces share the same `core/`:

- `cli/cli.py` — Click CLI, entry point `avf` (`project.scripts` in pyproject.toml)
- `gui/main_window.py` — Qt (PySide6) GUI, entry point `avf-gui`
- `core/pipeline.py` — the orchestrator both surfaces call into

Processing is a **stage pipeline**: each stage in `core/stages/` subclasses `BaseStage`
(`core/stages/base.py`), self-registers via `register_stage()` in `core/stages/__init__.py`, and
is looked up by `name` string, not by import order. `core/presets.py` defines named bundles
(`1080p60`, `4k60`, `size_reduction`, etc.) that set `enable_stages` and per-stage config —
a stage not in a preset's `enable_stages` won't run even if globally enabled. See AGENTS.md's
"Stage Development" and "Pipeline Behavior" sections before adding or reordering stages —
stage order, omission, and repetition are driven by `config.pipeline.default_order`, resolved by
`Pipeline.resolve_stage_order()`.

AI stages (upscale, denoise_video, interpolate) live under `ai/`:
- `ai/torch_utils.py` — device detection, tensor conversion, TTA, batched inference
- `ai/model_cache.py` — model registry, download + SHA256 verification, local cache
- `ai/frame_processor.py` — OpenCV frame extraction/conversion helpers
- `ai/wrappers/upscale.py` — Real-ESRGAN (RRDBNet)
- `ai/wrappers/interpolate.py` — RIFE (IFNet + EMD)

Each AI-capable stage falls back to a traditional FFmpeg implementation when PyTorch or a model
isn't available, and also honors the explicit `--ai`/`--no-ai` override (see AGENTS.md's
"AI/Traditional Method Selection").

`core/quality.py` estimates VMAF/SSIM/PSNR via FFmpeg to verify output quality against a target.
`core/analysis.py` covers scene detection (frame differencing), VLM classification, and duplicate
detection (ahash/dhash + Hamming distance) — this is what backs `avf analyze` and
`avf find-duplicates`.

Config is a single `Config` object (`config.py`) merging `DEFAULTS` with a YAML file at
`~/.config/auto-video-fixer/config.yaml` (platform-specific path via `get_config_dir()`), read
once at construction; presets merge recursively on top without clobbering user overrides not
present in the preset.

## Repo hygiene note

The repo root currently has stray dev/debug artifacts checked into the working tree (frame PNGs,
`session-ses_*.md` transcripts, test `.mp4`/`.trf` files, `dist/`). These aren't part of the
package or docs — don't treat them as reference material, and don't assume they should be edited
or cleaned up unless the user asks.
