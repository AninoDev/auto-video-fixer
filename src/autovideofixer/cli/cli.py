"""Auto Video Fixer - Command-line interface."""

from __future__ import annotations

import contextlib
import copy
import csv
import logging
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autovideofixer import __version__
from autovideofixer.config import (
    Config,
    diff_from_defaults,
    get_log_dir,
    prune_old_logs,
    redact_secrets,
)
from autovideofixer.core.analysis import is_video_file, scan_directory
from autovideofixer.core.pipeline import Pipeline
from autovideofixer.core.presets import get_preset, list_presets
from autovideofixer.logger import get_logger, setup_logging

if TYPE_CHECKING:
    from autovideofixer.core.analysis import VideoAnalysis

console = Console()

# Max automatic per-run log files retained under get_log_dir() (see
# config.prune_old_logs()). Oldest-by-mtime files beyond this count are
# deleted at startup, before the current run's log file is created.
MAX_RETAINED_LOGS = 50


@click.group()
@click.version_option(version=__version__, prog_name="avf")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (DEBUG level)")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Set logging level",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Write the run's log file to this path instead of the automatic "
    "timestamped file in the platform log directory",
)
@click.option(
    "--file-log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Log level for --log-file, if different from the console level "
    "(e.g. keep the console at INFO but capture DEBUG detail, including full "
    "ffmpeg commands, to the file)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    default=None,
    envvar="AVF_CONFIG",
    help="Use this config file instead of the default platform config path. "
    "Falls back to the AVF_CONFIG environment variable if not given. The file "
    "must already exist (this never silently creates or ignores a missing path).",
)
@click.pass_context
def main(
    ctx: click.Context,
    verbose: bool,
    log_level: str | None,
    log_file: str | None,
    file_log_level: str | None,
    config_path: str | None,
) -> None:
    """Auto Video Fixer - Automated video enhancement and processing.

    Process one or more video files with AI-powered upscaling,
    frame interpolation, denoising, and more.

    Logging:
      --verbose, -v          Enable DEBUG level logging (console and file)
      --log-level LEVEL      Set console logging level (DEBUG, INFO, WARNING, ERROR)
      --log-file PATH        Write the run's log file here instead of the
                             automatic location
      --file-log-level LEVEL Set file-only logging level, if different from console

    Every run logs to a file at DEBUG level independent of console verbosity:
    a timestamped file under the platform log directory by default (the exact
    path is logged at startup), or the --log-file path when given.
    """
    ctx.ensure_object(dict)

    console_level = "DEBUG" if verbose else (log_level or "INFO")

    # An explicit --log-file replaces the automatic state-dir log rather than
    # adding a second file: the run's canonical log lives wherever the user
    # pointed it, and it gets the same always-DEBUG treatment.
    if log_file:
        auto_log_file = log_file
    else:
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        prune_old_logs(log_dir, keep=MAX_RETAINED_LOGS)
        auto_log_file = str(log_dir / f"avf-{datetime.now():%Y%m%d-%H%M%S}.log")

    setup_logging(
        console_level, log_file=None, file_level=file_log_level, auto_log_file=auto_log_file
    )

    logger = get_logger("autovideofixer.cli")
    logger.info("avf %s -- log file: %s", __version__, auto_log_file)
    logger.info("Invocation: %s", " ".join(sys.argv))

    try:
        if config_path:
            ctx.obj["config"] = Config(config_path, require_exists=True)
        else:
            ctx.obj["config"] = Config()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    ctx.obj["config_path_source"] = config_path

    _log_effective_settings(logger, ctx.obj["config"], config_path)


def _log_effective_settings(
    logger: logging.Logger, config: Config, explicit_config_path: str | None
) -> None:
    """Log what config source/preset/settings this run is using.

    At INFO: which config file (default vs explicit), and a compact diff of
    effective settings vs Config.DEFAULTS (secrets redacted). At DEBUG: the
    full effective config dump (also redacted).
    """
    if explicit_config_path:
        logger.info("Config file: %s (explicit)", explicit_config_path)
    else:
        logger.info("Config file: %s (default)", config._path)

    diff = redact_secrets(diff_from_defaults(config.data, Config.DEFAULTS))
    if diff:
        logger.info("Effective settings (differ from defaults): %s", diff)
    else:
        logger.info("Effective settings: all defaults")

    logger.debug("Full effective config: %s", redact_secrets(config.data))


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--preset", "-p", default=None, help="Processing preset name")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option(
    "--output-name",
    default=None,
    help="Explicit output filename (only valid with exactly one input file)",
)
@click.option("--recursive", "-r", is_flag=True, help="Scan directories recursively")
@click.option("--dry-run", is_flag=True, help="Show what would be done without processing")
@click.option("--list-presets", "list_presets_flag", is_flag=True, help="List available presets")
@click.option(
    "--stage",
    "stages",
    multiple=True,
    help="Run only these stages, replacing the preset/auto-determined list (can repeat)",
)
@click.option(
    "--enable-stage",
    "enable_stages",
    multiple=True,
    help="Force-enable a stage on top of the preset/default set, e.g. --enable-stage hdr "
    "(can repeat)",
)
@click.option(
    "--disable-stage",
    "disable_stages",
    multiple=True,
    help="Force-disable a stage even if the preset/default set would run it (can repeat)",
)
@click.option("--threads", type=int, default=None, help="Number of processing threads")
@click.option("--ai", "use_ai", flag_value=True, default=None, help="Force AI-based processing")
@click.option(
    "--no-ai",
    "use_ai",
    flag_value=False,
    help="Disable AI (use traditional methods)",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=None,
    help="Allow overwriting an existing output file (overrides general.overwrite in config)",
)
@click.option(
    "--ai-fallback/--no-ai-fallback",
    "ai_fallback",
    default=None,
    help="Allow (default) or forbid AI-capable stages (upscale/interpolate/denoise_video/"
    "deblock) from silently falling back to their traditional FFmpeg method when the AI "
    "path can't run -- torch missing, model load failure, inference exception, or CUDA OOM "
    "after tiling retries. With --no-ai-fallback such a stage FAILS instead (overrides "
    "general.ai_fallback in config for this run; does not override a per-stage "
    "stages.<name>.ai_fallback set in config).",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Target output framerate (overrides quality.quality_target.target_framerate)",
)
@click.option(
    "--resolution",
    default=None,
    help="Target output resolution as WIDTHxHEIGHT, e.g. 3840x2160 "
    "(overrides quality.quality_target.target_resolution)",
)
@click.option(
    "--codec",
    default=None,
    help="Video codec, e.g. libx264, libx265, libvpx-vp9 (overrides encoding.video_codec)",
)
@click.option(
    "--audio-codec",
    default=None,
    help="Audio codec, e.g. aac, copy (overrides encoding.audio_codec)",
)
@click.option(
    "--crf", type=int, default=None, help="Constant rate factor / quality (overrides encoding.crf)"
)
@click.option(
    "--encoder-preset",
    default=None,
    help="Encoder speed/quality preset (e.g. medium, slow, veryfast) -- NOT the same as "
    "--preset, which selects a named avf processing bundle (overrides encoding.preset)",
)
@click.option(
    "--hwaccel",
    default=None,
    type=click.Choice(["auto", "none", "cuda", "vaapi", "qsv", "vulkan", "videotoolbox"]),
    help="FFmpeg hardware acceleration method for encode/decode stages "
    "(overrides ffmpeg.hwaccel). Run `avf gpu-info` to see what's available.",
)
@click.option(
    "--gpu-device",
    default=None,
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    help="PyTorch device for AI stages (upscale/interpolate/denoise), separate from "
    "--hwaccel (overrides gpu.preferred_device). Run `avf gpu-info` to check what "
    "PyTorch actually sees.",
)
@click.option(
    "--scene-mode/--no-scene-mode",
    "scene_mode",
    default=None,
    help="Process stabilize/interpolate per-scene instead of whole-video (overrides "
    "scenes.enabled): shaky scenes get stabilized more aggressively than calm ones, and "
    "frame interpolation never runs across a scene cut. Reuses existing scene detection "
    "(analysis.event_detection.*); off by default. Whole-video stages (upscale/denoise/"
    "deblock/normalize/encode) are unaffected and still run once on the reassembled video.",
)
@click.option(
    "--drop-non-content/--no-drop-non-content",
    "drop_non_content",
    default=None,
    help="With --scene-mode: run per-scene VLM analysis + a coordinating LLM pass "
    "(analysis.vlm / analysis.llm) that flags and removes scenes that aren't part of the "
    "video's main content (e.g. a 'like and subscribe' interstitial). Fails open (keeps "
    "every scene, logs a WARNING) if the VLM/coordinator is unavailable or its response "
    "can't be parsed -- never drops content on an LLM failure (overrides "
    "scenes.drop_non_content).",
)
@click.option(
    "--crop-limit",
    type=int,
    default=None,
    help="cropdetect luma threshold for the auto-crop stage (overrides stages.crop.limit); "
    "the crop stage itself still needs --enable-stage crop or stages.crop.enabled: true "
    "in config, since auto-crop is opt-in.",
)
@click.option(
    "--zoom-coverage",
    type=float,
    default=None,
    help="Fraction (0.0-1.0) of frames that should end up border-free once the stabilize "
    "stage's zoom gate decides zoom applies at all (overrides stages.stabilize.zoom_coverage). "
    "1.0 (default) = vidstabtransform's optzoom=1, guaranteed no border on any frame. 0.0 = no "
    "zoom (all borders visible). In between trades that guarantee for a less aggressive crop, "
    "with occasional brief borders on the most extreme motion -- see AGENTS.md's stabilize zoom "
    "section.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of frames batched into one Real-ESRGAN forward pass for the upscale/deblock/"
    "denoise_video AI stages (overrides stages.{upscale,deblock,denoise_video}.batch_size for "
    "all three; default 1 = one frame at a time, today's behavior). Only helps frames small "
    "enough to skip tiled inference -- see --tile-batch-size for large (e.g. 4K) frames, which "
    "always tile regardless of this setting.",
)
@click.option(
    "--tile-batch-size",
    type=int,
    default=None,
    help="Number of tiles batched into one Real-ESRGAN forward pass when a frame is large "
    "enough to trigger tiled inference (overrides stages.{upscale,deblock,denoise_video}."
    "tile_batch_size for all three; default 1 = one tile at a time, today's behavior). This is "
    "the batching knob that matters at e.g. 4K, where every frame always tiles.",
)
@click.pass_context
def process(
    ctx: click.Context,
    paths: tuple[str, ...],
    preset: str | None,
    output: str | None,
    output_name: str | None,
    recursive: bool,
    dry_run: bool,
    list_presets_flag: bool,
    stages: tuple[str, ...],
    enable_stages: tuple[str, ...],
    disable_stages: tuple[str, ...],
    threads: int | None,
    use_ai: bool | None,
    overwrite: bool | None,
    ai_fallback: bool | None,
    fps: float | None,
    resolution: str | None,
    codec: str | None,
    audio_codec: str | None,
    crf: int | None,
    encoder_preset: str | None,
    hwaccel: str | None,
    gpu_device: str | None,
    scene_mode: bool | None,
    drop_non_content: bool | None,
    crop_limit: int | None,
    zoom_coverage: float | None,
    batch_size: int | None,
    tile_batch_size: int | None,
) -> None:
    """Process video files with the specified settings."""
    if list_presets_flag:
        _list_presets()
        return

    logger = get_logger("autovideofixer.cli")
    logger.info("Preset: %s", preset or "(none -- auto-determined per-file)")

    config = ctx.obj["config"]
    if threads:
        config.set(threads, "general", "max_concurrent_jobs")

    # Apply preset if specified
    if preset:
        p = get_preset(preset)
        if p is None:
            console.print(f"[red]Unknown preset: {preset}[/red]")
            console.print(f"Available: {', '.join(list_presets())}")
            sys.exit(1)
        config_data = p.to_config()
        # Work on a deep copy to avoid persisting preset values to disk
        config = copy.deepcopy(config)
        _merge_config(config, config_data)

    # Resolve output directory
    if output:
        config.set(output, "general", "output_dir")

    # Resolve AI override
    if use_ai is not None:
        config.set(use_ai, "general", "use_ai")

    if overwrite is not None:
        config.set(overwrite, "general", "overwrite")

    if ai_fallback is not None:
        config.set(ai_fallback, "general", "ai_fallback")

    if fps is not None:
        config.set(fps, "quality", "quality_target", "target_framerate")

    if resolution is not None:
        try:
            w_str, h_str = resolution.lower().split("x")
            config.set([int(w_str), int(h_str)], "quality", "quality_target", "target_resolution")
        except ValueError:
            console.print(
                f"[red]Invalid --resolution {resolution!r}: expected WIDTHxHEIGHT[/red] "
                "(e.g. 3840x2160)"
            )
            sys.exit(1)

    # Encoder settings feed the same "encoding" config key that
    # Preset.to_config() writes, which Pipeline.execute_job() merges into
    # job.stage_overrides["encode"] -- explicit flags here are applied after
    # the preset merge above, so they take priority over the preset's values.
    encoding_overrides = {}
    if codec is not None:
        encoding_overrides["video_codec"] = codec
    if audio_codec is not None:
        encoding_overrides["audio_codec"] = audio_codec
    if crf is not None:
        encoding_overrides["crf"] = crf
    if encoder_preset is not None:
        encoding_overrides["preset"] = encoder_preset
    if encoding_overrides:
        merged = dict(config.get("encoding", default={}) or {})
        merged.update(encoding_overrides)
        config.set(merged, "encoding")

    if hwaccel is not None:
        config.set(hwaccel, "ffmpeg", "hwaccel")

    if gpu_device is not None:
        config.set(gpu_device, "gpu", "preferred_device")

    if scene_mode is not None:
        config.set(scene_mode, "scenes", "enabled")

    if drop_non_content is not None:
        config.set(drop_non_content, "scenes", "drop_non_content")

    if crop_limit is not None:
        config.set(crop_limit, "stages", "crop", "limit")

    if zoom_coverage is not None:
        config.set(zoom_coverage, "stages", "stabilize", "zoom_coverage")

    if batch_size is not None:
        for _stage_name in ("upscale", "deblock", "denoise_video"):
            config.set(batch_size, "stages", _stage_name, "batch_size")

    if tile_batch_size is not None:
        for _stage_name in ("upscale", "deblock", "denoise_video"):
            config.set(tile_batch_size, "stages", _stage_name, "tile_batch_size")

    for stage_name in enable_stages:
        config.set(True, "stages", stage_name, "enabled")
    for stage_name in disable_stages:
        config.set(False, "stages", stage_name, "enabled")

    # Re-log the effective settings now that the preset (if any) and every
    # CLI override above have been merged in -- the group-level log in
    # main() only reflects the config as loaded from disk, before any of
    # this command's own overrides.
    _log_effective_settings(logger, config, ctx.obj.get("config_path_source"))

    # Collect input files
    input_files = []
    for path in paths:
        if os.path.isdir(path):
            input_files.extend(scan_directory(path, recursive=recursive))
        elif is_video_file(path):
            input_files.append(path)
        else:
            console.print(f"[yellow]Skipping non-video file: {path}[/yellow]")

    if not input_files:
        console.print("[red]No video files found.[/red]")
        sys.exit(1)

    if output_name and len(input_files) != 1:
        console.print(
            f"[red]--output-name requires exactly one input file, got {len(input_files)}[/red]"
        )
        sys.exit(1)

    console.print(f"Found {len(input_files)} video file(s)")

    if dry_run:
        console.print("\n[bold]DRY RUN - No files will be processed:[/bold]")
        for f in input_files:
            console.print(f"  - {f}")
        return

    # Create pipeline and process
    pipeline = Pipeline(config)
    if output_name:
        output_dir = config.get("general", "output_dir", default=None) or os.path.dirname(
            input_files[0]
        )
        jobs = [pipeline.add_job(input_files[0], output_path=os.path.join(output_dir, output_name))]
    else:
        jobs = pipeline.add_files(input_files)

    # Override stages if specified
    if stages:
        for job in jobs:
            job.stages = list(stages)

    console.print(f"\nProcessing {len(jobs)} job(s)...")
    results = pipeline.execute_all(callback=_on_job_complete)

    # Summary
    failed = sum(1 for r in results if not r.success)
    _print_summary(results)
    if failed > 0:
        sys.exit(1)


# Preview length for the VLM summary in the default (non---full) table row.
# Kept generous (well beyond the old 120-char cutoff) since the table can wrap;
# --full prints the complete, untruncated text in a separate Rich Panel.
_VLM_SUMMARY_PREVIEW_LEN = 300

_ANALYZE_CSV_FIELDS = [
    "filepath",
    "filename",
    "duration_sec",
    "resolution",
    "framerate_fps",
    "video_codec",
    "has_video",
    "has_audio",
    "hdr",
    "scenes_detected",
    "scene_boundaries",
    "vlm_summary",
    "vlm_tags",
    "vlm_objects",
    "content_rating",
]


def _scene_boundaries_str(analysis: "VideoAnalysis") -> str:
    """Semicolon-joined "t=<seconds>s@<confidence>" per detected scene, for the
    CSV/--full output's "would a lower --scene-threshold find more cuts?"
    diagnostics. confidence is SceneEvent.confidence -- the frame-differencing
    diff_score of the cut that ended this scene (see _detect_scene_changes()'s
    docstring in core/analysis.py); the final scene's confidence is a fixed
    0.5 placeholder rather than a real cut score, since nothing ends it.
    """
    return ";".join(f"t={s.end_time:.1f}s@{s.confidence:.3f}" for s in analysis.scenes)


def _print_analysis_result(analysis: "VideoAnalysis", *, full_output: bool) -> None:
    """Print the Rich table (and, with --full, a full-text panel) for one analysis."""
    table = Table(title="Video Analysis")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Filename", analysis.filename)
    table.add_row("Duration", f"{analysis.duration:.1f}s")
    table.add_row("Resolution", f"{analysis.resolution[0]}x{analysis.resolution[1]}")
    table.add_row("Framerate", f"{analysis.framerate:.1f} fps")
    if analysis.video_codec:
        table.add_row("Video Codec", analysis.video_codec)
    table.add_row("Has Video", str(analysis.has_video))
    table.add_row("Has Audio", str(analysis.has_audio))
    table.add_row("HDR", str(analysis.is_hdr))
    table.add_row("Scenes Detected", str(analysis.total_scenes))

    if analysis.vlm_summary:
        summary = analysis.vlm_summary
        if len(summary) > _VLM_SUMMARY_PREVIEW_LEN:
            summary = (
                summary[:_VLM_SUMMARY_PREVIEW_LEN].rstrip() + " [...] (use --full for full text)"
            )
        table.add_row("VLM Summary", summary)
    if analysis.vlm_tags:
        table.add_row("Tags", ", ".join(analysis.vlm_tags))
    if analysis.vlm_objects:
        table.add_row("Objects", ", ".join(analysis.vlm_objects))
    if analysis.content_rating:
        table.add_row("Content Rating", analysis.content_rating)

    console.print(table)

    if full_output and analysis.vlm_summary:
        console.print(
            Panel(
                analysis.vlm_summary,
                title=f"Full VLM Summary: {analysis.filename}",
                expand=True,
            )
        )

    if analysis.scenes:
        console.print(f"\n[bold]Detected {analysis.total_scenes} Scene(s):[/bold]")
        for scene in analysis.scenes[:30]:
            desc = f" - {scene.description}" if scene.description else ""
            # Boundary confidence (the frame-differencing diff_score that ended
            # this scene) only in --full -- keeps the default view uncluttered;
            # see _scene_boundaries_str()'s docstring for what it means and the
            # last-scene caveat.
            score = f" [score={scene.confidence:.3f}]" if full_output else ""
            # markup=False: this line's own "[...]" segments (event_type, and
            # now score) are plain text, not Rich style tags -- with markup
            # parsing on (the console default), Rich silently swallows any
            # "[...]" that isn't a recognized style name instead of erroring,
            # which was quietly eating the "[scene_change]"/"[talking_head]"
            # prefix on every line here.
            console.print(
                f"  [{scene.event_type}] "
                f"{scene.start_time:.1f}s - {scene.end_time:.1f}s "
                f"({scene.duration:.1f}s){desc}{score}",
                markup=False,
            )


def _analysis_to_csv_row(analysis: "VideoAnalysis") -> dict[str, Any]:
    """Build one CSV row dict from a VideoAnalysis, matching the printed table's fields."""
    return {
        "filepath": analysis.filepath,
        "filename": analysis.filename,
        "duration_sec": f"{analysis.duration:.3f}",
        "resolution": f"{analysis.resolution[0]}x{analysis.resolution[1]}",
        "framerate_fps": f"{analysis.framerate:.3f}",
        "video_codec": analysis.video_codec,
        "has_video": analysis.has_video,
        "has_audio": analysis.has_audio,
        "hdr": analysis.is_hdr,
        "scenes_detected": analysis.total_scenes,
        "scene_boundaries": _scene_boundaries_str(analysis),
        "vlm_summary": analysis.vlm_summary or "",
        "vlm_tags": ";".join(analysis.vlm_tags),
        "vlm_objects": ";".join(analysis.vlm_objects),
        "content_rating": analysis.content_rating or "",
    }


def _write_analysis_csv(csv_path: str, rows: list[dict[str, Any]]) -> None:
    """Write analyze results to CSV. Overwrites csv_path if it already exists."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ANALYZE_CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def _make_progress_reporter(
    logger: logging.Logger,
    filepath: str,
    file_index: int,
    file_count: int,
    status: Any,
) -> Any:
    """Build a VideoAnalyzer progress_callback(phase, detail) for one file.

    Every phase transition is logged at INFO (so it always lands in the
    always-on per-run log file, and on the console unless --log-level raises
    the threshold), and additionally pushed to `status` (a Rich Status object)
    when the console is a TTY, so a long-running phase (scene detection, a
    slow VLM request) doesn't look hung. `status` may be None (non-TTY /
    plain fallback) -- the log lines alone then serve as the progress trail.
    """

    def _callback(phase: str, detail: str) -> None:
        message = f"{phase}: {detail}" if detail else phase
        # Rich's console handler renders log messages as markup, so avoid "[...]"
        # around an interpolated filepath -- an unrelated "[" in the path (or,
        # elsewhere, in the message) would otherwise be parsed as a markup tag
        # and raise MarkupError instead of just logging.
        logger.info("analyze progress (%s): %s", filepath, message)
        if status is not None:
            prefix = f"[{file_index}/{file_count}] " if file_count > 1 else ""
            status.update(f"{prefix}{os.path.basename(filepath)} -- {message}")

    return _callback


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--vlm", "vlm_flag", is_flag=True, default=None, help="Enable VLM analysis")
@click.option("--no-vlm", "vlm_flag", flag_value=False, help="Disable VLM analysis")
@click.option(
    "--events", "events_flag", is_flag=True, default=None, help="Enable event/scene detection"
)
@click.option("--no-events", "events_flag", flag_value=False, help="Disable event/scene detection")
@click.option(
    "--classify", "classify_events", is_flag=True, help="Classify detected events with VLM"
)
@click.option(
    "--clip",
    "clip_output",
    type=click.Path(),
    default=None,
    help="Extract scenes as clips to directory",
)
@click.option("--no-clip", "clip_output", flag_value="", help="Skip clip extraction")
@click.option("--recursive", "-r", is_flag=True, help="Scan directories recursively")
@click.option(
    "--scene-threshold",
    "scene_threshold",
    type=float,
    default=None,
    help="Scene-change sensitivity override for this run (overrides "
    "analysis.event_detection.scene_change_threshold in config; default 0.15). This is "
    "the mean fractional per-pixel luma change between consecutive downscaled frames, "
    "0-1: lower (e.g. 0.05-0.10) is more sensitive but risks false positives from "
    "camera motion/noise; higher (e.g. 0.3-0.5) only catches hard, high-contrast cuts "
    "and will under-detect soft/similar-toned shot changes. See _detect_scene_changes() "
    "in core/analysis.py for the full metric writeup and calibration data.",
)
@click.option(
    "--min-scene-duration",
    "min_scene_duration",
    type=float,
    default=None,
    help="Minimum scene duration override for this run, in seconds (overrides "
    "analysis.event_detection.min_scene_duration_sec in config; default 2.0). Does not "
    "affect whether a cut is detected -- only whether a cut that would produce a "
    "shorter scene gets to start its own scene vs. being merged into the next one. "
    "Lower this for fast-cut content (scenes under ~2s) or legitimate adjacent cuts "
    "will be silently merged away.",
)
@click.option(
    "--full",
    "full_output",
    is_flag=True,
    help="Print the complete, untruncated VLM summary for each file in a panel below "
    "the table (the table always shows a truncated preview)",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(),
    default=None,
    help="Write one row per analyzed video to this CSV file (UTF-8, full untruncated "
    "VLM summary included, plus a scene_boundaries column with ';'-joined "
    "t=<seconds>s@<confidence> entries per detected cut -- see --full). Overwrites "
    "the file if it already exists -- results are not appended across runs, since "
    "the header would drift as fields change.",
)
@click.option(
    "--prompt-append",
    "prompt_append",
    default=None,
    help="Extra text appended to the VLM user prompt for this run, e.g. job-specific "
    "context (overrides analysis.vlm.prompt_append in config)",
)
@click.option(
    "--prompt-override",
    "prompt_override",
    default=None,
    help="Replace the VLM user prompt entirely for this run (overrides "
    "analysis.vlm.prompt_override in config). Changing the requested output format "
    "away from JSON degrades gracefully into a plain-text summary -- see AGENTS.md.",
)
@click.option(
    "--max-sample-frames",
    "max_sample_frames",
    type=int,
    default=None,
    help="Max frames sampled from the video and sent to the VLM provider for this run "
    "(overrides analysis.vlm.max_sample_frames in config; default 8).",
)
@click.option(
    "--sample-interval",
    "sample_interval",
    type=float,
    default=None,
    help="Seconds between VLM sample frames for this run (overrides "
    "analysis.vlm.sample_interval_sec in config; default 10.0).",
)
@click.option(
    "--vlm-model",
    "vlm_model",
    default=None,
    help="VLM model name override for this run (overrides analysis.vlm.model in "
    "config; e.g. 'llava', 'gpt-4o').",
)
@click.option(
    "--vlm-url",
    "vlm_url",
    default=None,
    help="VLM provider URL override for this run (overrides analysis.vlm.api_url in "
    "config). Still subject to the same HTTPS-required-for-non-loopback gate as the "
    "config value (analysis.vlm.allow_http permits plain HTTP for e.g. a LAN box) -- "
    "there is no flag for allow_http or api_key; set those in config.yaml only.",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    paths: tuple[str, ...],
    vlm_flag: bool | None,
    events_flag: bool | None,
    classify_events: bool,
    clip_output: str | None,
    recursive: bool,
    scene_threshold: float | None,
    min_scene_duration: float | None,
    full_output: bool,
    csv_path: str | None,
    prompt_append: str | None,
    prompt_override: str | None,
    max_sample_frames: int | None,
    sample_interval: float | None,
    vlm_model: str | None,
    vlm_url: str | None,
) -> None:
    """Analyze video file(s) for properties, events, and content.

    Accepts one or more files and/or directories (directories are scanned for video
    files; pass --recursive to scan subdirectories). Each file is analyzed in
    sequence; a failure on one file is logged and skipped, and the remaining files
    are still processed -- the command exits non-zero if any file failed.
    """
    config = ctx.obj["config"]
    logger = get_logger("autovideofixer.cli")

    from autovideofixer.core.analysis import VideoAnalyzer

    analyzer = VideoAnalyzer(config)

    input_files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            input_files.extend(scan_directory(path, recursive=recursive))
        elif is_video_file(path):
            input_files.append(path)
        else:
            console.print(f"[yellow]Skipping non-video file: {path}[/yellow]")

    if not input_files:
        console.print("[red]No video files found.[/red]")
        sys.exit(1)

    console.print(f"Found {len(input_files)} video file(s)")

    csv_rows: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for idx, filepath in enumerate(input_files, start=1):
        if len(input_files) > 1:
            console.print(f"\n[bold]== [{idx}/{len(input_files)}] Analyzing: {filepath} ==[/bold]")
        else:
            console.print(f"Analyzing: {filepath}")

        # A live-updating status line on a real terminal; on a non-TTY (e.g.
        # captured output, CI, a pipe) Rich prints status updates as plain
        # scrolling lines instead, which combined with the INFO log lines
        # below is the "plain log lines" fallback.
        status_ctx = (
            console.status("Starting analysis...", spinner="dots")
            if console.is_terminal
            else contextlib.nullcontext()
        )

        try:
            with status_ctx as status:
                progress_cb = _make_progress_reporter(
                    logger, filepath, idx, len(input_files), status
                )
                analysis = analyzer.analyze(
                    filepath,
                    include_vlm=vlm_flag,
                    include_events=events_flag,
                    prompt_append=prompt_append,
                    prompt_override=prompt_override,
                    progress_callback=progress_cb,
                    scene_threshold=scene_threshold,
                    min_scene_duration=min_scene_duration,
                    max_sample_frames=max_sample_frames,
                    sample_interval_sec=sample_interval,
                    vlm_model=vlm_model,
                    vlm_api_url=vlm_url,
                )

                # analyze() doesn't take a classify_events param, so re-run event
                # detection directly with classification enabled when --classify
                # was requested.
                if classify_events and events_flag is not False:
                    analysis.scenes = analyzer.detect_events(
                        filepath,
                        classify_events=True,
                        progress_callback=progress_cb,
                        threshold=scene_threshold,
                        min_duration=min_scene_duration,
                    )
                    analysis.total_scenes = len(analysis.scenes)
        except Exception:
            logger.error("Analysis failed for %r", filepath, exc_info=True)
            console.print(f"[red]Analysis failed for {filepath} -- see log for details.[/red]")
            failed_files.append(filepath)
            continue

        # Full VLM result always lands in the log file (DEBUG-level auto log
        # captures everything regardless of console verbosity); INFO so it also
        # shows up in a --verbose/--log-level INFO console without needing --full.
        if analysis.vlm_summary or analysis.vlm_tags or analysis.vlm_objects:
            logger.info(
                "VLM analysis for %s -- summary: %s | tags: %s | objects: %s | rating: %s",
                filepath,
                analysis.vlm_summary or "",
                ", ".join(analysis.vlm_tags),
                ", ".join(analysis.vlm_objects),
                analysis.content_rating or "",
            )

        _print_analysis_result(analysis, full_output=full_output)

        if csv_path:
            csv_rows.append(_analysis_to_csv_row(analysis))

        # Extract clips if requested
        file_clip_output = clip_output
        if file_clip_output is not None and analysis.scenes:
            if file_clip_output == "":
                file_clip_output = None

            if file_clip_output:
                clips = analyzer.extract_scenes_as_clips(
                    filepath, analysis.scenes, output_dir=file_clip_output
                )
                if clips:
                    console.print(
                        f"\n[green]Extracted {len(clips)} clip(s) to: {file_clip_output}[/green]"
                    )
                    for clip in clips:
                        console.print(
                            f"  Clip {clip.scene_index}: "
                            f"{clip.start_time:.1f}s-{clip.end_time:.1f}s -> "
                            f"{os.path.basename(clip.output_path)}"
                        )
                else:
                    console.print("\n[yellow]No clips could be extracted.[/yellow]")

    if csv_path:
        _write_analysis_csv(csv_path, csv_rows)
        console.print(f"\n[green]Wrote {len(csv_rows)} row(s) to {csv_path}[/green]")

    if failed_files:
        console.print(
            f"\n[red]{len(failed_files)} of {len(input_files)} file(s) failed analysis.[/red]"
        )
        sys.exit(1)


@main.command()
@click.argument("reference")
@click.argument("directory")
@click.option("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
@click.pass_context
def find_duplicates(ctx: click.Context, reference: str, directory: str, threshold: float) -> None:
    """Find similar/duplicate videos in a directory."""
    config = ctx.obj["config"]
    from autovideofixer.core.analysis import VideoAnalyzer

    analyzer = VideoAnalyzer(config)

    candidates = scan_directory(directory)
    console.print(f"Comparing {reference} against {len(candidates)} candidate(s)...")

    results = analyzer.find_similar(reference, candidates, threshold)

    if results:
        console.print(f"\nFound {len(results)} similar video(s):")
        for path, sim in results:
            console.print(f"  [{sim * 100:.1f}%] {path}")
    else:
        console.print("No similar videos found.")


@main.command()
def presets_cmd() -> None:
    """List available processing presets."""
    _list_presets()


@main.command()
def gpu_info() -> None:
    """Show GPU and hardware acceleration information.

    Covers two independent GPU paths that are easy to conflate: FFmpeg
    hwaccel (used by the encode/decode stages) and PyTorch/CUDA (used by the
    AI upscale/interpolate/denoise stages). A system can have one without the
    other -- e.g. ffmpeg hwaccel working fine while PyTorch silently falls
    back to CPU for AI stages, which looks like "processing is just slow"
    with no other symptom.
    """
    from autovideofixer.core.ffmpeg_utils import detect_hardware_acceleration

    hwaccels = detect_hardware_acceleration()
    console.print("[bold]FFmpeg hardware acceleration[/bold] (encode/decode stages):")
    for hw in hwaccels:
        console.print(f"  - {hw}")
    if not hwaccels:
        console.print("  No hardware acceleration detected.")

    console.print("\n[bold]PyTorch / CUDA[/bold] (AI upscale/interpolate/denoise stages):")
    try:
        import torch
    except ImportError:
        console.print(
            "  [yellow]PyTorch not installed[/yellow] -- AI stages will fall back to "
            "traditional methods. Install the 'ai' extra: uv sync --extra ai"
        )
        return

    console.print(f"  torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    console.print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        console.print(f"  CUDA build version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            console.print(
                f"  [green]GPU {i}: {props.name} (sm_{props.major}{props.minor}, "
                f"{vram_gb:.1f} GB)[/green]"
            )
    else:
        console.print(
            "  [red]No CUDA GPU detected by PyTorch.[/red] AI stages will run on CPU, "
            "which is 1-2 orders of magnitude slower. Common causes: (1) a CPU-only "
            "torch build was installed (plain `pip install torch` on some platforms "
            "does not include CUDA support -- check https://pytorch.org for the "
            "correct install command for your CUDA version), or (2) the installed "
            "torch build predates support for this GPU's compute capability (very "
            "new GPUs need a recent-enough torch/CUDA release)."
        )

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    console.print(f"  MPS (Apple Silicon) available: {mps_available}")

    from autovideofixer.ai.torch_utils import get_device

    console.print(f"\n  get_device('auto') currently selects: {get_device('auto')}")


@main.command()
@click.option("--model", default=None, help="Specific model to show info for")
def model_info(model: str | None) -> None:
    """Show AI model information and download status."""
    from autovideofixer.ai.model_cache import (
        MODEL_REGISTRY,
        list_cached_models,
    )

    console.print("[bold]Available AI Models:[/bold]\n")

    table = Table(title="Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Description")
    table.add_column("Size")
    table.add_column("Cached")
    table.add_column("SHA256")

    cached = {m["name"] for m in list_cached_models()}

    for key, meta in MODEL_REGISTRY.items():
        if model and key != model:
            continue

        is_cached = key in cached
        sha = meta.get("sha256")
        if isinstance(sha, bytes):
            sha = sha.hex()
        sha_str = str(sha)[:12] + "..." if sha else "N/A"

        table.add_row(
            key,
            meta.get("type", "model"),
            meta.get("description", ""),
            f"{meta.get('size_mb', 0):.1f} MB",
            "[green]Yes[/green]" if is_cached else "[yellow]No[/yellow]",
            sha_str,
        )

    console.print(table)

    if cached:
        console.print(f"\n[blue]Cached models: {', '.join(cached)}[/blue]")
    else:
        console.print(
            "[red]No models cached. Download with: avf model-download --model <name>[/red]"
        )


@main.command()
@click.option("--model", required=True, help="Model name to download")
@click.option("--url", default=None, help="Custom download URL")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
def model_download(model: str, url: str | None, force: bool) -> None:
    """Download an AI model for processing."""
    from autovideofixer.ai.model_cache import (
        MODEL_REGISTRY,
        download_model,
        ensure_model_available,
    )
    from autovideofixer.ai.torch_utils import is_torch_available

    if not is_torch_available():
        console.print("[red]PyTorch is not installed. Install with: pip install torch[/red]")
        sys.exit(1)

    if url:
        try:
            ok, msg = download_model(model, url=url)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            sys.exit(1)
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]Download failed: {msg}[/red]")
            sys.exit(1)
    elif model in MODEL_REGISTRY:
        try:
            ok, msg = ensure_model_available(model, force_download=force)
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            sys.exit(1)
        if ok:
            meta = MODEL_REGISTRY[model]
            console.print("[green]Model downloaded:[/green]")
            console.print(f"  Name: {meta.get('description', model)}")
            console.print(f"  {msg}")
        else:
            console.print(f"[red]Download failed: {msg}[/red]")
            sys.exit(1)
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        console.print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)


# ─── Helpers ───────────────────────────────────────────────────────


def _list_presets() -> None:
    """Display available presets in a table."""
    table = Table(title="Available Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name")
    table.add_column("Description")

    for name, preset in list_presets().items():
        table.add_row(name, preset.display_name, preset.description)

    console.print(table)


def _merge_config(config: Config, data: dict) -> None:
    """Recursively merge data into config."""
    for key, value in data.items():
        if isinstance(value, dict):
            current = config.get(key, default={})
            if isinstance(current, dict):
                # Recursively merge nested dicts
                _merge_config_helper(current, value)
                config.set(current, key)
            else:
                config.set(value, key)
        else:
            config.set(value, key)


def _merge_config_helper(target: dict, source: dict) -> None:
    """Recursively merge source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _merge_config_helper(target[key], value)
        else:
            target[key] = value


def _on_job_complete(job, result) -> None:
    """Callback when a job completes."""
    status_icon = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
    console.print(f"  {status_icon} {os.path.basename(job.input_path)}")


def _print_summary(results) -> None:
    """Print processing summary."""
    total = len(results)
    success = sum(1 for r in results if r.success)
    failed = total - success

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total: {total}")
    console.print(f"  Success: {success}")
    console.print(f"  Failed: {failed}")

    if failed > 0:
        console.print("\n[red]Failed jobs:[/red]")
        for r in results:
            if not r.success:
                console.print(f"  - {r.input_path}: {'; '.join(r.errors)}")
