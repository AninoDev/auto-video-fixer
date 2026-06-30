"""Auto Video Fixer - Command-line interface."""

from __future__ import annotations

import copy
import logging
import os
import sys

import click
from rich.console import Console
from rich.table import Table

from autovideofixer import __version__
from autovideofixer.config import Config
from autovideofixer.core.analysis import is_video_file, scan_directory
from autovideofixer.core.pipeline import Pipeline
from autovideofixer.core.presets import get_preset, list_presets
from autovideofixer.logger import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="avf")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (DEBUG level)")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default=None, help="Set logging level")
@click.option("--log-file", type=click.Path(), default=None, help="Log to file in addition to console")
@click.pass_context
def main(ctx: click.Context, verbose: bool, log_level: str | None, log_file: str | None) -> None:
    """Auto Video Fixer - Automated video enhancement and processing.

    Process one or more video files with AI-powered upscaling,
    frame interpolation, denoising, and more.
    
    Logging:
      --verbose, -v          Enable DEBUG level logging
      --log-level LEVEL      Set logging level (DEBUG, INFO, WARNING, ERROR)
      --log-file PATH        Log to file (in addition to console)
    """
    ctx.ensure_object(dict)
    
    # Setup logging
    if verbose:
        setup_logging("DEBUG")
    elif log_level:
        setup_logging(log_level)
    else:
        setup_logging("INFO")
    
    # Add file handler if requested
    if log_file:
        from autovideofixer.logger import get_logger
        logger = get_logger("autovideofixer")
        import logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(file_handler)
    
    ctx.obj["config"] = Config()


@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--preset", "-p", default=None, help="Processing preset name")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--recursive", "-r", is_flag=True, help="Scan directories recursively")
@click.option("--dry-run", is_flag=True, help="Show what would be done without processing")
@click.option("--list-presets", "list_presets_flag", is_flag=True, help="List available presets")
@click.option("--stage", "stages", multiple=True, help="Specific stages to run (can repeat)")
@click.option("--threads", type=int, default=None, help="Number of processing threads")
@click.pass_context
def process(
    ctx: click.Context,
    paths: tuple[str, ...],
    preset: str | None,
    output: str | None,
    recursive: bool,
    dry_run: bool,
    list_presets_flag: bool,
    stages: tuple[str, ...],
    threads: int | None,
) -> None:
    """Process video files with the specified settings."""
    if list_presets_flag:
        _list_presets()
        return

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

    console.print(f"Found {len(input_files)} video file(s)")

    if dry_run:
        console.print("\n[bold]DRY RUN - No files will be processed:[/bold]")
        for f in input_files:
            console.print(f"  - {f}")
        return

    # Create pipeline and process
    pipeline = Pipeline(config)
    jobs = pipeline.add_files(input_files)

    # Override stages if specified
    if stages:
        for job in jobs:
            job.stages = list(stages)

    console.print(f"\nProcessing {len(jobs)} job(s)...")
    results = pipeline.execute_all(callback=_on_job_complete)

    # Summary
    _print_summary(results)


@main.command()
@click.argument("filepath")
@click.option("--vlm", is_flag=True, help="Run VLM analysis")
@click.pass_context
def analyze(ctx: click.Context, filepath: str, vlm: bool) -> None:
    """Analyze a video file for properties, events, and content."""
    config = ctx.obj["config"]

    from autovideofixer.core.analysis import VideoAnalyzer

    analyzer = VideoAnalyzer(config)

    console.print(f"Analyzing: {filepath}")
    analysis = analyzer.analyze(filepath, include_vlm=vlm)

    table = Table(title="Video Analysis")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Filename", analysis.filename)
    table.add_row("Duration", f"{analysis.duration:.1f}s")
    table.add_row("Resolution", f"{analysis.resolution[0]}x{analysis.resolution[1]}")
    table.add_row("Framerate", f"{analysis.framerate:.1f} fps")
    table.add_row("Has Video", str(analysis.has_video))
    table.add_row("Has Audio", str(analysis.has_audio))
    table.add_row("HDR", str(analysis.is_hdr))
    table.add_row("Scenes Detected", str(analysis.total_scenes))

    if analysis.vlm_summary:
        table.add_row("VLM Summary", analysis.vlm_summary[:100] + "...")
    if analysis.vlm_tags:
        table.add_row("Tags", ", ".join(analysis.vlm_tags))

    console.print(table)

    if analysis.scenes:
        console.print("\n[bold]Detected Scenes:[/bold]")
        for scene in analysis.scenes[:20]:  # Show first 20
            console.print(f"  {scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.event_type})")


@main.command()
@click.argument("reference")
@click.argument("directory")
@click.option("--threshold", type=float, default=0.95, help="Similarity threshold (0-1)")
def find_duplicates(reference: str, directory: str, threshold: float) -> None:
    """Find similar/duplicate videos in a directory."""
    config = Config()
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
    """Show GPU and hardware acceleration information."""
    from autovideofixer.core.ffmpeg_utils import detect_hardware_acceleration

    hwaccels = detect_hardware_acceleration()
    console.print("Available hardware accelerations:")
    for hw in hwaccels:
        console.print(f"  - {hw}")

    if not hwaccels:
        console.print("  No hardware acceleration detected.")


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

    cached = set(list_cached_models())

    for key, meta in MODEL_REGISTRY.items():
        if model and key != model:
            continue

        is_cached = key in cached
        sha = meta.get("sha256", "N/A")
        if isinstance(sha, bytes):
            sha = sha.hex()
        sha_str = str(sha)[:12] + "..."

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
        console.print(f"\n[Cyan]Cached models: {', '.join(cached)}[/cyan]")
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
        ensure_model_available,
    )
    from autovideofixer.ai.torch_utils import is_torch_available

    if not is_torch_available():
        console.print("[red]PyTorch is not installed. Install with: pip install torch[/red]")
        sys.exit(1)

    if url:
        try:
            path = ensure_model_available(model, url=url, force=force)
            console.print(f"[green]Model downloaded to: {path}[/green]")
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            sys.exit(1)
    elif model in MODEL_REGISTRY:
        try:
            path = ensure_model_available(model, force=force)
            meta = MODEL_REGISTRY[model]
            console.print("[green]Model downloaded:[/green]")
            console.print(f"  Name: {meta.get('description', model)}")
            console.print(f"  Path: {path}")
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
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
