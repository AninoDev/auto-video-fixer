"""Auto Video Fixer - Command-line interface."""

from __future__ import annotations

import copy
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

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="avf")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Auto Video Fixer - Automated video enhancement and processing.

    Process one or more video files with AI-powered upscaling,
    frame interpolation, denoising, and more.
    """
    ctx.ensure_object(dict)
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
                current.update(value)
                config.set(current, key)
            else:
                config.set(value, key)
        else:
            config.set(value, key)


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
