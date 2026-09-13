"""Auto Video Fixer - Command-line interface."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autovideofixer import __version__
from autovideofixer.cli import config_tools
from autovideofixer.cli.progress import ProgressReporter, resolve_show_progress
from autovideofixer.config import (
    VALID_COLOR_MODES,
    VALID_LOG_TYPES,
    Config,
    _looks_like_secret_key,
    diff_from_defaults,
    get_config_dir,
    get_log_dir,
    prune_old_logs,
    redact_secrets,
    sanitize_console_text,
)
from autovideofixer.core.analysis import is_video_file, scan_directory
from autovideofixer.core.input_parsers import get_parser, list_parsers
from autovideofixer.core.pipeline import Job, JobResult, Pipeline
from autovideofixer.core.presets import get_preset, list_presets, load_preset
from autovideofixer.core.reporting import (
    aggregate_stage_timing,
    build_json_report,
    build_run_meta,
    job_summary_line,
    run_classification_aggregate,
    run_outcome_aggregate,
    stage_table_rows,
    write_json_report,
)
from autovideofixer.logclean import get_pii_cleaner
from autovideofixer.logger import build_console, get_logger, set_console_color, setup_logging

if TYPE_CHECKING:
    from autovideofixer.core.analysis import VideoAnalysis

console = Console()
# Diagnostics stream for the `avf config` subcommands: their stdout is a data
# payload (clean/dumped/upgraded YAML), so every error, warning, and
# backup-rename notice must go to stderr instead -- otherwise
# `avf config upgrade in.yaml --drop-unknown > out.yaml` would prepend
# ANSI-coloured "Warning:" lines into the YAML file and corrupt it.
err_console = Console(stderr=True)


def _apply_color_mode(mode: str) -> None:
    """Rebuild `console`/`err_console` (this module) and the logging
    RichHandler's console (logger.py) for `mode`, so all three Rich
    consoles the CLI touches agree on the resolved reporting.color value.

    `console`/`err_console` are built at IMPORT time, before any config is
    read (see the module-level assignments above), so they always start out
    "auto" -- this is the retroactive fix-up, called once the effective
    reporting.color (config file, then `main()`'s early read; then
    `process`'s own --color/--no-color CLI-flag layer, which wins) is known.
    Every `console.print(...)`/`err_console.print(...)` call site in this
    module resolves the module global at call time, so reassigning these
    globals here is picked up by every later call without threading a
    `console` parameter through the whole file.

    Reassigns rather than mutates (new `Console` objects) -- Rich has no
    public API to flip an existing Console's colour mode in place, and
    `ProgressReporter`/anything else constructed AFTER this runs is handed
    the freshly rebuilt `console` global, never a stale pre-override one
    (both `process` call sites below apply this before constructing
    `ProgressReporter`). `set_console_color()` does the equivalent for
    logger.py's own RichHandler console -- see its docstring for why that
    one needs to be retargeted after the fact rather than just rebuilt here
    (it isn't a module global in THIS file).
    """
    global console, err_console
    console = build_console(stderr=False, color=mode)
    err_console = build_console(stderr=True, color=mode)
    set_console_color(mode)


def _safe(value: object) -> str:
    """Sanitize external data (filenames, exception text, subprocess stderr
    excerpts) before it's interpolated into a console.print() f-string.

    console.print() is Rich's own I/O path, not the logging pipeline that
    logger.py's _SanitizingConsoleFormatter covers -- direct console.print()
    calls in this module that embed user-supplied/filesystem/exception data
    need their own sanitization at the interpolation site. Deliberate Rich
    markup ("[red]...[/red]") the code itself writes is untouched since it's
    always plain ASCII brackets/letters written directly in the f-string,
    never passed through this helper.
    """
    return sanitize_console_text(value)


# Max automatic per-run log files retained under get_log_dir() (see
# config.prune_old_logs()). Oldest-by-mtime files beyond this count are
# deleted at startup, before the current run's log file is created.
MAX_RETAINED_LOGS = 50


# ─── Config cascade: --preset/--config layering + --set + CLI-flags layer ──
#
# See AGENTS.md's "Config cascade" section for the full model. In short:
#   1. DEFAULTS
#   2. the user config.yaml (or an explicit top-level --config/AVF_CONFIG path)
#   3. each `process --preset`/`process --config PATH` layer, in the order the
#      flags appear on the command line, interleaved
#   4. every non-config/non-preset CLI option (including --set), folded into
#      ONE final layer applied last, later-flag-beats-earlier-flag on conflict
#
# Click parses each `multiple=True` option into its own tuple, preserving
# that option's *own* order but losing cross-option interleaving. The scan
# below recovers the true left-to-right order from sys.argv; if sys.argv
# doesn't actually correspond to this invocation (e.g. a programmatic
# CliRunner.invoke() call in a test, whose sys.argv is the test runner's
# own), we can't trust it and fall back to a fixed, documented order instead.

# Flags (after the `process` subcommand token) that take a value, either as
# a following token or `--flag=value`. Includes both the repeatable
# cascade-layer flags (--preset/--config/--set/--enable-stage/--disable-stage)
# and every other value-taking option that maps to a config key, so their
# relative argv position can be recovered too (see _position() below).
_VALUE_FLAGS = {
    "--preset",
    "-p",
    "--config",
    "--set",
    "--enable-stage",
    "--disable-stage",
    "--output",
    "-o",
    "--threads",
    "--fps",
    "--resolution",
    "--codec",
    "--audio-codec",
    "--crf",
    "--encoder-preset",
    "--hwaccel",
    "--gpu-device",
    "--crop-limit",
    "--zoom-coverage",
    "--batch-size",
    "--tile-batch-size",
    "--resolution-fit-mode",
    "--dimension-multiple",
    "--snap-tolerance",
    "--from-file",
    "--from-file-parser",
    "--retime-min-duplicate-ratio",
    "--output-timing",
    "--interpolate-target-approach",
    "--interpolate-direct-synthesis-max",
    "--interpolate-gap-fallback",
    "--interpolate-max-intermediates-per-gap",
    "--interpolate-max-gap-sec",
}

# Boolean/flag-value options that map to a config key -- recorded with a
# None value (the flag's mere presence is the signal).
_BOOL_FLAGS = {
    "--ai",
    "--no-ai",
    "--overwrite",
    "--no-overwrite",
    "--ai-fallback",
    "--no-ai-fallback",
    "--scene-mode",
    "--no-scene-mode",
    "--drop-non-content",
    "--no-drop-non-content",
    "--downscale",
    "--no-downscale",
    "--retime",
    "--no-retime",
}


def _find_subcommand_start(argv: list[str], tokens: tuple[str, ...]) -> int | None:
    """Return the argv index right after the first occurrence of `tokens`
    appearing consecutively (e.g. ("config", "dump")), or None if not found.

    Shared by every subcommand-scoped argv scan below -- each subcommand
    (possibly nested, like `config dump`) scans only the flags that appear
    after its own token sequence, so a flag of the same name typed for a
    *different* subcommand (or before the subcommand at all) is never
    mistaken for this one's.
    """
    n = len(tokens)
    if n == 0:
        return None
    for i in range(len(argv) - n + 1):
        if tuple(argv[i : i + n]) == tokens:
            return i + n
    return None


def _scan_argv_flags(
    argv: list[str],
    start: int,
    value_flags: set[str],
    bool_flags: set[str],
) -> list[tuple[str, str | None, int]]:
    """Walk `argv` from `start` onward, recording every recognized flag.

    Returns an ordered list of (flag, value, argv_index) for occurrences of
    `value_flags` (takes a following token or `--flag=value`) and
    `bool_flags` (mere presence is the signal, value is always None).
    """
    out: list[tuple[str, str | None, int]] = []
    i = start
    n = len(argv)
    while i < n:
        tok = argv[i]
        matched = False
        for flag in value_flags:
            if tok == flag:
                val = argv[i + 1] if i + 1 < n else None
                out.append((flag, val, i))
                i += 1
                matched = True
                break
            if tok.startswith(flag + "="):
                out.append((flag, tok[len(flag) + 1 :], i))
                matched = True
                break
        if matched:
            i += 1
            continue
        if tok in bool_flags:
            out.append((tok, None, i))
        i += 1
    return out


def _scan_process_argv() -> list[tuple[str, str | None, int]]:
    """Walk sys.argv from the `process` subcommand token onward.

    Returns an ordered list of (flag, value, argv_index) for every
    recognized config-affecting flag occurrence (see _VALUE_FLAGS/
    _BOOL_FLAGS). Returns [] if "process" isn't found in sys.argv at all --
    callers must treat that (or a value mismatch against what Click actually
    parsed) as "argv unavailable" and fall back to a fixed order.
    """
    argv = sys.argv
    start = _find_subcommand_start(argv, ("process",))
    if start is None:
        return []
    return _scan_argv_flags(argv, start, _VALUE_FLAGS, _BOOL_FLAGS)


# `avf config dump` accepts only the layering flags (--preset/--config/--set)
# -- a subset of _VALUE_FLAGS -- and no boolean flags.
_DUMP_VALUE_FLAGS = {"--preset", "-p", "--config", "--set"}


def _scan_config_dump_argv() -> list[tuple[str, str | None, int]]:
    """Same as `_scan_process_argv`, scoped to `avf config dump`."""
    argv = sys.argv
    start = _find_subcommand_start(argv, ("config", "dump"))
    if start is None:
        return []
    return _scan_argv_flags(argv, start, _DUMP_VALUE_FLAGS, set())


def _repeatable_matches(
    occurrences: list[tuple[str, str | None, int]], flag_names: set[str], expected: tuple[str, ...]
) -> bool:
    """True iff the argv-scanned values for `flag_names` exactly match `expected`."""
    scanned = [val for flag, val, _idx in occurrences if flag in flag_names]
    return scanned == list(expected)


def _split_inline_parser(value: str, default: str) -> tuple[str, str]:
    """Split a --from-file value on an inline `NAME:PATH` prefix.

    Only treats the text before the first `:` as a parser-name override
    when it's actually a registered parser name -- otherwise the whole
    value is the path and `default` (the current stateful parser) is used.
    This keeps real paths/URLs containing a colon (or a file literally
    named e.g. "x:y") working via the normal stateful/default parser
    instead of being misparsed as `x:` + path `y`.
    """
    prefix, sep, rest = value.partition(":")
    if sep and prefix in list_parsers():
        return prefix, rest
    return default, value


def _resolve_from_file_specs(
    from_files: tuple[str, ...], from_file_parsers: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Resolve each --from-file occurrence to (list_file_path, parser_name).

    Prefers the true command-line order (recovered via `_scan_process_argv`)
    so `--from-file-parser` is genuinely stateful across interleaved
    `--from-file` occurrences. Falls back to applying the LAST
    `--from-file-parser` value (or "shlex" if none given) to every file, in
    `from_files`' own order, when argv can't be trusted (e.g. a
    programmatic CliRunner.invoke() call whose sys.argv doesn't match) --
    same fallback philosophy as the --preset/--config layering above.

    An inline `NAME:PATH` value on a --from-file occurrence overrides the
    current stateful parser for just that file (see `_split_inline_parser`),
    in both the argv and fallback paths.
    """
    occurrences = _scan_process_argv()
    argv_usable = _repeatable_matches(
        occurrences, {"--from-file"}, from_files
    ) and _repeatable_matches(occurrences, {"--from-file-parser"}, from_file_parsers)

    resolved: list[tuple[str, str]] = []
    if argv_usable and occurrences:
        current_parser = "shlex"
        for flag, val, _idx in occurrences:
            if flag == "--from-file-parser" and val is not None:
                current_parser = val
            elif flag == "--from-file" and val is not None:
                parser_name, path = _split_inline_parser(val, current_parser)
                resolved.append((path, parser_name))
    else:
        fallback_parser = from_file_parsers[-1] if from_file_parsers else "shlex"
        for val in from_files:
            parser_name, path = _split_inline_parser(val, fallback_parser)
            resolved.append((path, parser_name))
    return resolved


def _content_signature(path: str) -> str:
    """Return a blake2b hex digest identifying a file's content.

    Hashes the file size plus up to three sampled 1 MiB chunks (head, middle,
    tail) for large files, or the whole file for small ones (<= ~3 MiB) --
    cheap enough to distinguish content without reading entire large files.
    """
    chunk_size = 1024 * 1024
    size = os.stat(path).st_size
    digest = hashlib.blake2b()
    digest.update(str(size).encode("utf-8"))
    with open(path, "rb") as fh:
        if size <= chunk_size * 3:
            digest.update(fh.read())
        else:
            fh.seek(0)
            digest.update(fh.read(chunk_size))
            fh.seek(size // 2)
            digest.update(fh.read(chunk_size))
            fh.seek(max(0, size - chunk_size))
            digest.update(fh.read(chunk_size))
    return digest.hexdigest()


def _build_content_groups(realpaths: list[str]) -> dict[str, object]:
    """Assign each realpath a content-identity group key.

    Two different realpaths land in the same group iff they share both file
    size (cheap prefilter) and a `_content_signature` (blake2b over size +
    sampled chunks). Realpaths whose size is unique (or whose stat/read
    fails) get their own singleton group -- distinct size already proves
    distinct content, and I/O errors fail safe to "not a duplicate" rather
    than silently merging unrelated files.
    """
    sizes: dict[str, int] = {}
    for rp in realpaths:
        try:
            sizes[rp] = os.stat(rp).st_size
        except OSError:
            continue

    size_buckets: dict[int, list[str]] = {}
    for rp, size in sizes.items():
        size_buckets.setdefault(size, []).append(rp)
    hashable = {rp for bucket in size_buckets.values() if len(bucket) > 1 for rp in bucket}

    group_key: dict[str, object] = {}
    for rp in realpaths:
        if rp in hashable:
            try:
                group_key[rp] = (sizes[rp], _content_signature(rp))
                continue
            except OSError:
                pass
        group_key[rp] = ("__path__", rp)
    return group_key


def _plan_resolved_inputs(
    resolved_inputs: list[tuple[str, str | None]],
    pipeline: Pipeline,
    console: Console,
) -> tuple[list[tuple[str, str | None]], dict[str, list[str]]]:
    """Collapse content-duplicate inputs into one representative job each,
    with the rest recorded as fan-out copy destinations.

    Two entries are the same CONTENT (see `_build_content_groups`: realpath
    identity plus size+hash for distinct realpaths sharing content). Within
    a content group, walked in original order:
    - the first distinct resolved output becomes the REPRESENTATIVE -- it's
      kept in the returned to-process list and actually gets processed;
    - a later entry resolving to a different output is an EXTRA destination
      -- recorded in the returned fan-out dict (representative's resolved
      output -> list of extra resolved outputs) so the completion callback
      can copy the finished file there instead of reprocessing;
    - a later entry resolving to an output already seen in this group (same
      source, same destination) is a true no-op duplicate and is dropped.

    Outputs are resolved via `pipeline.resolve_output_path` up front so the
    fan-out dict key is the exact string `add_job`/`Job.output_path` will
    carry for the representative (override outputs are kept verbatim --
    only the in-group distinctness comparison uses realpath).
    """
    realpaths = list(dict.fromkeys(os.path.realpath(p) for p, _ in resolved_inputs))
    group_key_by_realpath = _build_content_groups(realpaths)

    # Per content group, the (output_key, resolved_output, input_path) tuples
    # seen so far, in first-seen order -- index 0 is the representative.
    group_seen: dict[object, list[tuple[str, str, str]]] = {}

    to_process: list[tuple[str, str | None]] = []
    fanout: dict[str, list[str]] = {}

    for entry in resolved_inputs:
        input_path, output_override = entry
        real = os.path.realpath(input_path)
        group_key = group_key_by_realpath[real]
        resolved_output = pipeline.resolve_output_path(input_path, output_override)
        output_key = os.path.realpath(resolved_output)

        seen_list = group_seen.setdefault(group_key, [])
        match = next((s for s in seen_list if s[0] == output_key), None)
        if match is not None:
            console.print(
                f"[yellow]Skipping duplicate input (same output as {_safe(match[2])}): "
                f"{_safe(input_path)}[/yellow]"
            )
            continue

        if not seen_list:
            seen_list.append((output_key, resolved_output, input_path))
            to_process.append(entry)
        else:
            rep_output = seen_list[0][1]
            rep_input = seen_list[0][2]
            seen_list.append((output_key, resolved_output, input_path))
            fanout.setdefault(rep_output, []).append(resolved_output)
            console.print(
                f"[yellow]Will copy result of {_safe(rep_input)} to additional destination: "
                f"{_safe(resolved_output)}[/yellow]"
            )

    return to_process, fanout


def _fanout_copy(
    job: Job, result: JobResult, extras: list[str], config: Config, console: Console
) -> None:
    """Copy a representative job's finished output to its extra destinations.

    Invoked from the job-completion callback for representatives that had
    other content-identical inputs collapsed onto them by
    `_plan_resolved_inputs` -- rather than reprocessing to produce a
    byte-identical file, the already-finished output is copied.
    """
    output_path = job.output_path
    overwrite = bool(config.get("general", "overwrite", default=False))
    source_ok = (
        result.outcome != "failed" and output_path is not None and os.path.exists(output_path)
    )

    for extra in extras:
        if not source_ok or not output_path:
            console.print(
                f"[yellow]Cannot copy to {_safe(extra)}: source output not produced[/yellow]"
            )
            continue
        try:
            if os.path.exists(extra) and not overwrite:
                console.print(
                    "[yellow]Output already exists, not overwriting "
                    f"(general.overwrite=false): {_safe(extra)}[/yellow]"
                )
                continue
            os.makedirs(os.path.dirname(extra) or ".", exist_ok=True)
            shutil.copy2(output_path, extra)
            console.print(f"[green]Copied output to {_safe(extra)}[/green]")
        except OSError as e:
            console.print(f"[red]Failed to copy output to {_safe(extra)}: {_safe(str(e))}[/red]")


def _is_preset_path(value: str) -> bool:
    """Disambiguate a --preset value: a filesystem path, or a registered name?"""
    return (
        os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or value.endswith((".yaml", ".yml"))
        or os.path.exists(value)
    )


def _resolve_preset_layer(value: str) -> dict[str, Any]:
    """Resolve one --preset value (name or path) to a config-layer dict."""
    if _is_preset_path(value):
        p = load_preset(value)
        if p is None:
            console.print(f"[red]Failed to load preset file: {value}[/red]")
            sys.exit(1)
        return p.to_config()
    p = get_preset(value)
    if p is None:
        console.print(f"[red]Unknown preset: {value}[/red]")
        console.print(f"Available: {', '.join(list_presets())}")
        sys.exit(1)
    return p.to_config()


def _resolve_config_layer(path: str) -> dict[str, Any]:
    """Resolve one --config PATH value to a config-layer dict.

    Unlike the default config path, an explicitly-given --config layer is a
    hard error if missing or malformed -- same "no silent no-op" policy as
    the top-level --config/AVF_CONFIG flag.
    """
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Config layer file not found: {path}[/red]")
        sys.exit(1)
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML in config layer {path}: {e}[/red]")
        sys.exit(1)
    if not isinstance(data, dict):
        console.print(f"[red]Config layer file must contain a YAML mapping: {path}[/red]")
        sys.exit(1)
    return data


def _parse_set_option(raw: str) -> tuple[list[str], Any]:
    """Parse one --set KEY=VALUE into (dot-notation key path, parsed value).

    VALUE is parsed as a YAML scalar, so `true`/`16`/`null`/quoted strings/
    inline lists (`[a,b]`) all work. Rejects secret-looking keys (api_key/
    token/password/etc.) -- those are config-file-only, to keep secrets out
    of shell history (see AGENTS.md).
    """
    if "=" not in raw:
        raise click.BadParameter(f"--set value must be KEY=VALUE, got {raw!r}")
    key, _, value_str = raw.partition("=")
    key = key.strip()
    if not key:
        raise click.BadParameter(f"--set KEY=VALUE: empty key in {raw!r}")
    key_path = key.split(".")
    if any(_looks_like_secret_key(part) for part in key_path):
        raise click.BadParameter(
            f"--set {key}=...: secret-looking keys (api_key/token/password/etc.) are "
            "config-file-only and must not be passed on the command line (shell history "
            "risk) -- set this in your config.yaml instead"
        )
    try:
        value = yaml.safe_load(value_str)
    except yaml.YAMLError as e:
        raise click.BadParameter(f"--set {key}=...: invalid YAML value: {e}") from e
    return key_path, value


def _set_nested(d: dict[str, Any], key_path: list[str], value: Any) -> None:
    """Assign `value` at `key_path` in `d`, creating intermediate dicts as needed."""
    node = d
    for k in key_path[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[key_path[-1]] = value


def _register_argv_known_paths() -> None:
    """Best-effort REQUIREMENTS.md § 6.7 registration pass over `sys.argv`.

    Runs in the group callback, before the "Invocation:" log line (which
    prints raw argv verbatim and fires before `process` has resolved
    anything). Not exhaustive -- covers only what's cheaply knowable from
    argv alone, without knowing which subcommand or option each token
    belongs to:
      - the value following `--config`/`-o`/`--output` gets registered as a
        directory (role "config"/"output" respectively) -- for `--config`,
        its PARENT directory (the token itself is a file);
      - the value following `--log-file` is skipped entirely -- by the time
        this scan runs, `setup_logging()` has already created that file, so
        it would otherwise look like an "existing file" below and get
        mis-registered as an input;
      - `argv[0]` (the `avf` executable's own path) is always skipped for
        the same reason -- it's a real file too, just never a video;
      - any OTHER token that `is_video_file()` recognizes (extension-based,
        see core/analysis.py) is registered as an input. Deliberately NOT
        "any existing file" -- that would also catch e.g. the log file
        itself, a --config YAML, or the `avf` binary, all real files that
        aren't video inputs.
    Anything not covered here (e.g. a relative path that only resolves once
    `process` applies its own cwd-relative logic, or an output path that
    doesn't exist yet) stays raw in the "Invocation:" line specifically --
    documented v1 limitation; `process`'s own registration (resolved input
    files, output dir, config/endpoint values) still covers every later log
    line.
    """
    cleaner = get_pii_cleaner()
    argv = sys.argv
    skip_next = False
    for i, tok in enumerate(argv):
        if i == 0:
            continue  # the avf executable's own path -- never an input
        if skip_next:
            skip_next = False
            continue
        if tok == "--config" and i + 1 < len(argv):
            cleaner.register_directory(os.path.dirname(os.path.abspath(argv[i + 1])), "config")
            skip_next = True
        elif tok in ("-o", "--output") and i + 1 < len(argv):
            cleaner.register_directory(argv[i + 1], "output")
            skip_next = True
        elif tok == "--log-file" and i + 1 < len(argv):
            skip_next = True  # avoid mis-registering the log file as input
        elif is_video_file(tok):
            cleaner.register_input(tok)


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
    "--log-type",
    type=click.Choice(list(VALID_LOG_TYPES), case_sensitive=False),
    default=None,
    help="Which log FILE variant(s) to write: raw (default -- today's behavior, "
    "unredacted), clean (PII-substituted via known input/output paths, directories, "
    "VLM/LLM endpoints, and embedded titles -- safe to share when asking for help), "
    "both (two files, see general.log_suffix_raw/log_suffix_clean), or none (no file "
    "logging at all). The CONSOLE always stays raw regardless of this setting. Wins "
    "over general.log_type in config when both are given; NOTE: `--set "
    "general.log_type=...`/`--config` passed to `process` does NOT affect logging -- "
    "handlers are attached here, before `process` builds its config cascade -- so "
    "config-file users must set this in their user config.yaml (or pass this flag).",
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
    log_type: str | None,
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
      --log-type TYPE        raw (default) | clean | both | none -- see AGENTS.md's
                             "Log types" note

    Every run logs to a file at DEBUG level independent of console verbosity:
    a timestamped file under the platform log directory by default (the exact
    path is logged at startup), or the --log-file path when given.
    """
    ctx.ensure_object(dict)

    console_level = "DEBUG" if verbose else (log_level or "INFO")

    # Config is constructed here (before logging is set up) rather than
    # after, so general.log_type/log_suffix_raw/log_suffix_clean can be read
    # from it when --log-type is absent -- this is the one config read cheap
    # enough (DEFAULTS + one YAML file) to justify doing before setup_logging()
    # (see AGENTS.md's "Log types" note for why this only ever reads the base
    # config, never a `process`-level --set/--config/--preset layer).
    try:
        if config_path:
            config = Config(config_path, require_exists=True)
        else:
            config = Config()
    except FileNotFoundError as e:
        console.print(f"[red]{_safe(e)}[/red]")
        sys.exit(1)

    ctx.obj["config"] = config
    ctx.obj["config_path_source"] = config_path

    resolved_log_type = (
        log_type or config.get("general", "log_type", default="raw") or "raw"
    ).lower()
    if resolved_log_type not in VALID_LOG_TYPES:
        console.print(
            f"[red]Invalid log_type: {resolved_log_type!r} "
            f"(must be one of {VALID_LOG_TYPES!r})[/red]"
        )
        sys.exit(1)
    log_suffix_raw = config.get("general", "log_suffix_raw", default="")
    log_suffix_clean = config.get("general", "log_suffix_clean", default="-clean")

    # reporting.color (§ "tri-state colour control"): no CLI flag at THIS
    # (group) level -- --color/--no-color lives on `process` only, since
    # that's the command people pipe through `tee` -- so this only ever
    # reflects the base config file, same "read before setup_logging()"
    # caveat as resolved_log_type above. `process` re-resolves and
    # retroactively applies the fully-layered value (including its own CLI
    # flag) later via `_apply_color_mode()`, once its preset/config/CLI-flag
    # cascade has finished.
    resolved_color = (config.get("reporting", "color", default="auto") or "auto").lower()
    if resolved_color not in VALID_COLOR_MODES:
        console.print(
            f"[red]Invalid reporting.color: {resolved_color!r} "
            f"(must be one of {VALID_COLOR_MODES!r})[/red]"
        )
        sys.exit(1)
    _apply_color_mode(resolved_color)

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

    try:
        opened_files = setup_logging(
            console_level,
            log_file=None,
            file_level=file_log_level,
            auto_log_file=auto_log_file,
            log_type=resolved_log_type,
            log_suffix_raw=log_suffix_raw,
            log_suffix_clean=log_suffix_clean,
            color=resolved_color,
        )
    except ValueError as e:
        console.print(f"[red]{_safe(e)}[/red]")
        sys.exit(1)

    logger = get_logger("autovideofixer.cli")

    # REQUIREMENTS.md § 6.7: register KNOWN values BEFORE the first log lines
    # that can echo them (substitution happens at emit time, so registration
    # only has to precede the log call): the log files' own directories (the
    # "avf ... log file(s)" line and Invocation's --log-file value would
    # otherwise leak the user's home/state dir), the config directory (and
    # any explicit --config path's directory), the VLM/LLM endpoints from the
    # base config (the effective-settings dumps below echo api_url), plus a
    # best-effort pass over the rest of sys.argv before "Invocation:" (which
    # prints raw argv verbatim and fires before `process` gets a chance to do
    # its own full registration -- resolved input files, output dir, layered
    # config endpoints). Not exhaustive -- see _register_argv_known_paths()'s
    # docstring and AGENTS.md's "Log types" note for what's NOT covered here.
    cleaner = get_pii_cleaner()
    for opened in opened_files:
        log_parent = os.path.dirname(os.path.abspath(opened))
        if log_parent:
            cleaner.register_directory(log_parent, "logs")
    cleaner.register_directory(str(get_config_dir()), "config")
    if config_path:
        cleaner.register_directory(os.path.dirname(os.path.abspath(config_path)), "config")
    for section in ("vlm", "llm"):
        cleaner.register_endpoint(config.get("analysis", section, "api_url", default=None))
    _register_argv_known_paths()

    if opened_files:
        logger.info("avf %s -- log file(s): %s", __version__, ", ".join(opened_files))
    else:
        logger.info("avf %s -- log file: none (general.log_type=none)", __version__)
    logger.info("Invocation: %s", " ".join(sys.argv))

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
@click.argument("paths", nargs=-1, required=False)
@click.option(
    "--from-file",
    "from_files",
    multiple=True,
    type=click.Path(),
    help="Read additional input paths from a list file, parsed by the current "
    "--from-file-parser (default: shlex). Repeatable -- may be combined with positional "
    "PATHS and/or other --from-file occurrences. A value of the form NAME:PATH (e.g. "
    "csv:manifest.csv) overrides the parser for just this one file, regardless of "
    "--from-file-parser, as long as NAME is a registered parser. Available parsers: "
    + ", ".join(sorted(list_parsers()))
    + ".",
)
@click.option(
    "--from-file-parser",
    "from_file_parsers",
    multiple=True,
    help="Select the parser used for every --from-file that follows it on the command "
    "line (stateful -- switches again at the next --from-file-parser). Default: shlex. "
    "See --from-file's help for the inline NAME:PATH override and the list of available "
    "parsers.",
)
@click.option(
    "--preset",
    "-p",
    "presets",
    multiple=True,
    help="Processing preset name, or a path to a preset YAML file (detected by a path "
    "separator, a .yaml/.yml extension, or the file existing on disk). Repeatable -- each "
    "one is applied as its own cascade layer, interleaved with --config in the order they "
    "appear on the command line (see AGENTS.md's 'Config cascade' section).",
)
@click.option(
    "--config",
    "config_layers",
    type=click.Path(),
    multiple=True,
    help="Additional config file to merge on top of the base config (DEFAULTS + the user "
    "config.yaml / top-level --config), as its own cascade layer. Repeatable -- interleaved "
    "with --preset in command-line order. Must already exist (hard error if not). Not the "
    "same as the top-level `avf --config PATH process ...` flag, which selects which file "
    "IS the base config; this one adds a layer on top.",
)
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    help="Set a config key directly: KEY=VALUE using dot-notation for nested keys (e.g. "
    "stages.upscale.ai_model=RealESRGAN_x2plus). VALUE is parsed as a YAML scalar, so "
    "true/16/null/quoted strings/inline lists ([a,b]) all work. Repeatable. Applied as part "
    "of the final CLI-flags layer (always last). Secret-looking keys (api_key/token/"
    "password/etc.) are rejected -- set those in config.yaml instead.",
)
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
    "--sharpen/--no-sharpen",
    "sharpen",
    default=None,
    help="Enable/disable post-stabilization sharpening (overrides "
    "stages.stabilize.sharpen_enabled). Only applies when stabilization actually runs.",
)
@click.option(
    "--sharpen-amount",
    type=float,
    default=None,
    help="Post-stabilization unsharp luma amount -- stronger = sharper (overrides "
    "stages.stabilize.sharpen_amount; ffmpeg's unsharp range is roughly -2.0..5.0). "
    "Finer knobs (matrix sizes, chroma amount) are available via "
    "--set stages.stabilize.sharpen_*.",
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
    "--interpolate-hybrid/--no-interpolate-hybrid",
    "interpolate_hybrid",
    default=None,
    help="With AI/RIFE interpolation: when the target framerate isn't an exact integer "
    "multiple of the source, run RIFE at the largest integer factor <= target then a "
    "minterpolate finish pass to reach the exact target, instead of overshooting and relying "
    "on a downstream frame-drop (overrides stages.interpolate.hybrid_ai_minterpolate; on by "
    "default). --no-interpolate-hybrid restores the pre-hybrid overshoot behavior.",
)
@click.option(
    "--interpolate-adaptive/--no-interpolate-adaptive",
    "interpolate_adaptive",
    default=None,
    help="With AI/RIFE interpolation: resample the recovered input timeline directly onto "
    "the target-fps grid (per-gap adaptive interpolation) instead of one uniform RIFE "
    "factor, so long gaps (camera stalls, an irregular-cadence download) get more "
    "synthesized frames and short gaps get fewer, reaching the target framerate exactly "
    "(overrides stages.interpolate.adaptive_cadence; on by default). Fails open to the "
    "uniform-factor path on a timeline probe failure. See docs/REQUIREMENTS.md § 16.",
)
@click.option(
    "--interpolate-target-approach",
    default=None,
    type=click.Choice(["under", "nearest", "over"]),
    help="Governs the UNIFORM-FACTOR AI interpolation path only (irrelevant when "
    "--interpolate-adaptive reaches the target exactly): 'under' (default) uses the "
    "largest integer RIFE factor at or below target then a minterpolate finish pass; "
    "'nearest' uses whichever integer factor lands closest to target; 'over' uses the "
    "smallest integer factor at or above target, then minterpolate finishes DOWN to the "
    "exact rate (overrides stages.interpolate.target_approach).",
)
@click.option(
    "--interpolate-direct-synthesis-max",
    type=int,
    default=None,
    help="With adaptive AI/RIFE interpolation: up to this many intermediates in one gap are "
    "generated directly from the gap's two real frames; more than this recursively "
    "subdivides (synthesize the midpoint, then reuse it as a new reference frame for two "
    "half-gap passes, and so on) so every RIFE call stays short-range (overrides "
    "stages.interpolate.direct_synthesis_max; default 3). See docs/REQUIREMENTS.md § 16.7.",
)
@click.option(
    "--interpolate-gap-fallback",
    default=None,
    type=click.Choice(["subdivide", "hold", "blend"]),
    help="With adaptive AI/RIFE interpolation: what a gap does once it exceeds "
    "--interpolate-direct-synthesis-max -- 'subdivide' (default) recursively bisects per "
    "§ 16.7; 'hold'/'blend' restore the pre-§16.7 behavior but ONLY take effect for a gap "
    "that also exceeds --interpolate-max-intermediates-per-gap or "
    "--interpolate-max-gap-sec (both unlimited by default, so this flag alone does "
    "nothing unless paired with one) (overrides stages.interpolate.gap_fallback).",
)
@click.option(
    "--interpolate-max-intermediates-per-gap",
    type=int,
    default=None,
    help="With adaptive AI/RIFE interpolation: a SAFETY VALVE for pathological input, not "
    "the normal path -- a gap needing more synthesized frames than this forces "
    "--interpolate-gap-fallback's hold/blend behavior instead of subdividing (overrides "
    "stages.interpolate.max_intermediates_per_gap; default 0 = unlimited, so an ordinary "
    "long gap always subdivides). See docs/REQUIREMENTS.md § 16.3.",
)
@click.option(
    "--interpolate-max-gap-sec",
    type=float,
    default=None,
    help="With adaptive AI/RIFE interpolation: a SAFETY VALVE, same idea as "
    "--interpolate-max-intermediates-per-gap but capping the gap's real DURATION in "
    "seconds instead of its intermediate count (overrides stages.interpolate.max_gap_sec; "
    "default 0.0 = unlimited).",
)
@click.option(
    "--downscale/--no-downscale",
    "downscale",
    default=None,
    help="Enable/disable the downscale stage, which shrinks an oversized input to the "
    "target resolution box before the heavier stages run (overrides stages.downscale."
    "enabled; off by default). Needs a target resolution (--resolution or "
    "quality.quality_target.target_resolution) to have any effect.",
)
@click.option(
    "--retime/--no-retime",
    "retime",
    default=None,
    help="Enable/disable the retime stage, which recovers a video's TRUE source "
    "cadence (as opposed to the encoded/container framerate) and drops padding/"
    "duplicate frames while preserving genuine VFR timing (overrides "
    "stages.retime.enabled; on by default). See docs/REQUIREMENTS.md § 12.",
)
@click.option(
    "--retime-min-duplicate-ratio",
    type=float,
    default=None,
    help="Minimum fraction of near-duplicate frames (1 - unique/total, from retime's "
    "cadence analysis) before the retime stage actually re-encodes -- below this the "
    "input is treated as already honest and the stage SKIPs (overrides "
    "stages.retime.min_duplicate_ratio; default 0.05).",
)
@click.option(
    "--audio-speed-method",
    default=None,
    type=click.Choice(["atempo", "rubberband", "asetrate"]),
    help="Audio algorithm for the speed stage (overrides stages.speed.audio_method; "
    "default atempo). 'atempo': pitch-preserving, chained for factors outside a single "
    "atempo's [0.5, 100] range -- audible quality loss at extreme slow-motion/speed-up "
    "factors. 'rubberband': pitch-preserving via ffmpeg's rubberband filter, one filter "
    "covers the whole range with none of atempo's chaining artifacts (needs an ffmpeg "
    "build with librubberband; falls back to atempo with a WARNING if missing). "
    "'asetrate': DELIBERATELY pitch-changing -- relabels the audio at a new sample rate "
    "instead of time-stretching it, which is the correct choice (not a compromise) for "
    "phone slow-motion footage recorded at an elevated mic sample rate and mapped down "
    "to normal speed: speeding it back up with asetrate restores the mic's true pitch "
    "(needs the input's probed audio sample rate; falls back to atempo with a WARNING "
    "if that can't be determined). Only takes effect when the speed stage runs "
    "(stages.speed.enabled/--enable-stage speed and a non-1.0 --set stages.speed.factor).",
)
@click.option(
    "--audio-sample-rate",
    type=int,
    default=None,
    help="Output audio sample rate in Hz (-ar) after the speed stage's audio filter "
    "(overrides stages.speed.audio_sample_rate; default 48000). 0 leaves the input's "
    "own sample rate alone (no -ar emitted). Also used as the 'asetrate' audio method's "
    "aresample target -- see --audio-speed-method.",
)
@click.option(
    "--audio-resampler",
    default=None,
    type=click.Choice(["soxr", "swr"]),
    help="Resampler used by the speed stage's 'asetrate' audio method (ignored by "
    "atempo/rubberband, which don't resample) -- overrides stages.speed.resampler; "
    "default soxr (libsoxr, higher quality; needs an ffmpeg build with --enable-libsoxr, "
    "which falls back to swr itself with its own warning if missing).",
)
@click.option(
    "--output-timing",
    default=None,
    type=click.Choice(["cfr", "vfr", "passthrough"]),
    help="Timing mode for the FINAL encode only (overrides general.output_timing; "
    "default cfr). 'cfr' keeps the deliverable constant-framerate at the target rate; "
    "'vfr' carries the recovered/genuine variable-framerate timeline all the way out; "
    "'passthrough' is the same idea without ffmpeg's timestamp-monotonicity "
    "normalization. Intermediates upstream of encode are always VFR-safe regardless.",
)
@click.option(
    "--resolution-fit-mode",
    default=None,
    type=click.Choice(["preserve_aspect", "snap_limiting"]),
    help="How upscale/downscale fit an input into the target resolution box (overrides "
    "quality.quality_target.resolution_fit_mode). preserve_aspect (default) keeps the "
    "input's exact aspect ratio; snap_limiting (snap-to-box-when-close) forces the limiting "
    "axis exactly onto the target, and also snaps the other axis onto the target when it "
    "would otherwise fall short by no more than --snap-tolerance (e.g. exactly 1920x1080 "
    "instead of ~1920x1078).",
)
@click.option(
    "--dimension-multiple",
    type=int,
    default=None,
    help="Round upscale/downscale output dimensions to a multiple of this (overrides "
    "quality.quality_target.dimension_multiple; default 2, required by H.264/yuv420p).",
)
@click.option(
    "--snap-tolerance",
    type=float,
    default=None,
    help="With --resolution-fit-mode snap_limiting: how close (as a fraction, e.g. 0.01 = 1%%) "
    "the non-limiting axis must be to the target box before it's snapped exactly onto it "
    "(overrides quality.quality_target.snap_tolerance; default 0.01). Ignored by "
    "preserve_aspect.",
)
@click.option(
    "--zoom-coverage",
    type=float,
    default=None,
    help="SIGNED zoom dial (0.0-1.0) once the stabilize stage's zoom gate decides zoom applies "
    "at all (overrides stages.stabilize.zoom_coverage; BREAKING as of 2026-09, see "
    "CHANGELOG.md). 1.0 (default) = vidstabtransform's optzoom=1, guaranteed no border on any "
    "frame (content cropped). 0.5 = no zoom at all (vidstabtransform's own default -- this is "
    "the OLD 0.0 behavior). 0.0 = zoom OUT far enough that every frame's full content survives "
    "(borders on all frames, nothing ever cropped). 0.25/0.75 = the zoom-out/zoom-in midpoints. "
    "See AGENTS.md's stabilize zoom section.",
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
@click.option(
    "--stage-timing-per-video/--no-stage-timing-per-video",
    "stage_timing_per_video",
    default=None,
    help="Print a per-video per-stage duration table (overrides "
    "reporting.stage_timing_per_video; off by default).",
)
@click.option(
    "--stage-timing-totals/--no-stage-timing-totals",
    "stage_timing_totals",
    default=None,
    help="Print per-stage total duration summed across the whole run at the end (overrides "
    "reporting.stage_timing_totals; off by default). Averaged/totaled by base stage name -- "
    "an occurrence label like 'upscale#2' rolls up into 'upscale'.",
)
@click.option(
    "--stage-timing-averages/--no-stage-timing-averages",
    "stage_timing_averages",
    default=None,
    help="Print per-stage average duration per video that actually ran it at the end "
    "(overrides reporting.stage_timing_averages; off by default). The divisor is videos "
    "that ran the stage successfully, never the total video count -- failed stage "
    "executions are excluded and shown separately.",
)
@click.option(
    "--report-json",
    "report_json",
    type=click.Path(),
    default=None,
    help="Write a single structured JSON run report to PATH once at the end of the run "
    "(also on partial failure -- whatever jobs finished) -- overrides reporting.report_json. "
    "Contains run metadata, per-job records (media info, outcome, decision trail, scene "
    "stats, timing), and per-stage records within each job (method/fallback provenance, "
    "duration, skip reason/error). No aggregates -- those are derivable from the per-job "
    "stage records.",
)
@click.option(
    "--progress-batch/--no-progress-batch",
    "progress_batch",
    default=None,
    help="Show a live batch progress bar (jobs completed, fractionally including the current "
    "job's own progress) while processing (overrides reporting.progress_batch; on by default, "
    "but only actually rendered when stdout is a real terminal).",
)
@click.option(
    "--progress-file/--no-progress-file",
    "progress_file",
    default=None,
    help="Show a live per-file progress bar tracking the current job's own 0..1 progress "
    "(overrides reporting.progress_file; on by default, but only actually rendered when "
    "stdout is a real terminal).",
)
@click.option(
    "--color/--no-color",
    "color",
    default=None,
    help="Force Rich console colour on/off (overrides reporting.color; default auto -- "
    "colour only when stdout/stderr is a real terminal). Useful for `avf process ... | tee "
    "run.log`: the pipe makes stdout non-interactive, so Rich normally drops colour even "
    "though you're still watching a real terminal downstream of `tee` -- pass --color to "
    "keep it. Independent of the progress bars: those stay gated on the real fd-level "
    "sys.stdout.isatty() check regardless of this flag (see --progress-batch/"
    "--progress-file) -- forcing colour on for a piped/redirected run does NOT re-enable "
    "them, since their redraw control codes would still corrupt the captured output.",
)
@click.pass_context
def process(
    ctx: click.Context,
    paths: tuple[str, ...],
    from_files: tuple[str, ...],
    from_file_parsers: tuple[str, ...],
    presets: tuple[str, ...],
    config_layers: tuple[str, ...],
    set_overrides: tuple[str, ...],
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
    sharpen: bool | None,
    sharpen_amount: float | None,
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
    interpolate_hybrid: bool | None,
    interpolate_adaptive: bool | None,
    interpolate_target_approach: str | None,
    interpolate_direct_synthesis_max: int | None,
    interpolate_gap_fallback: str | None,
    interpolate_max_intermediates_per_gap: int | None,
    interpolate_max_gap_sec: float | None,
    downscale: bool | None,
    retime: bool | None,
    retime_min_duplicate_ratio: float | None,
    audio_speed_method: str | None,
    audio_sample_rate: int | None,
    audio_resampler: str | None,
    output_timing: str | None,
    resolution_fit_mode: str | None,
    dimension_multiple: int | None,
    snap_tolerance: float | None,
    zoom_coverage: float | None,
    batch_size: int | None,
    tile_batch_size: int | None,
    stage_timing_per_video: bool | None,
    stage_timing_totals: bool | None,
    stage_timing_averages: bool | None,
    report_json: str | None,
    progress_batch: bool | None,
    progress_file: bool | None,
    color: bool | None,
) -> None:
    """Process video files with the specified settings."""
    if list_presets_flag:
        _list_presets()
        return

    logger = get_logger("autovideofixer.cli")
    logger.info(
        "Preset(s): %s", ", ".join(presets) if presets else "(none -- auto-determined per-file)"
    )

    # Work on a deep copy so preset/--config/--set/CLI-flag layers never
    # mutate (or get persisted from) the shared Config object main() built.
    config = copy.deepcopy(ctx.obj["config"])

    # --- Step 3: interleaved --preset/--config layers, in command-line order ---
    occurrences = _scan_process_argv()
    argv_usable = (
        _repeatable_matches(occurrences, {"--preset", "-p"}, presets)
        and _repeatable_matches(occurrences, {"--config"}, config_layers)
        and _repeatable_matches(occurrences, {"--set"}, set_overrides)
        and _repeatable_matches(occurrences, {"--enable-stage"}, enable_stages)
        and _repeatable_matches(occurrences, {"--disable-stage"}, disable_stages)
    )
    if argv_usable and occurrences:
        layer_seq = [
            (("preset" if flag in ("--preset", "-p") else "config"), val)
            for flag, val, _idx in occurrences
            if flag in ("--preset", "-p", "--config") and val is not None
        ]
    else:
        # Click's own order: each option's internal order is preserved, but
        # cross-option interleaving is lost -- all --preset values, then all
        # --config values.
        layer_seq = [("preset", v) for v in presets] + [("config", v) for v in config_layers]

    for kind, value in layer_seq:
        if kind == "preset":
            config.apply_layer(_resolve_preset_layer(value), f"preset:{value}")
        else:
            config.apply_layer(_resolve_config_layer(value), f"config:{value}")

    # --- Step 4: every non-config/non-preset CLI option, folded into ONE
    # final layer applied last (later-flag-beats-earlier-flag on conflict) ---
    # Each entry: (flag token(s) to look for in argv, key_path, value).
    cli_candidates: list[tuple[tuple[str, ...], list[str], Any]] = []
    if threads is not None:
        cli_candidates.append((("--threads",), ["general", "max_concurrent_jobs"], threads))
    if output is not None:
        cli_candidates.append((("--output", "-o"), ["general", "output_dir"], output))
    if use_ai is not None:
        cli_candidates.append((("--ai", "--no-ai"), ["general", "use_ai"], use_ai))
    if overwrite is not None:
        cli_candidates.append(
            (("--overwrite", "--no-overwrite"), ["general", "overwrite"], overwrite)
        )
    if ai_fallback is not None:
        cli_candidates.append(
            (("--ai-fallback", "--no-ai-fallback"), ["general", "ai_fallback"], ai_fallback)
        )
    if fps is not None:
        cli_candidates.append((("--fps",), ["quality", "quality_target", "target_framerate"], fps))
    if sharpen is not None:
        cli_candidates.append(
            (("--sharpen", "--no-sharpen"), ["stages", "stabilize", "sharpen_enabled"], sharpen)
        )
    if sharpen_amount is not None:
        cli_candidates.append(
            (("--sharpen-amount",), ["stages", "stabilize", "sharpen_amount"], sharpen_amount)
        )
    if resolution is not None:
        try:
            w_str, h_str = resolution.lower().split("x")
            resolved = [int(w_str), int(h_str)]
        except ValueError:
            console.print(
                f"[red]Invalid --resolution {resolution!r}: expected WIDTHxHEIGHT[/red] "
                "(e.g. 3840x2160)"
            )
            sys.exit(1)
        cli_candidates.append(
            (("--resolution",), ["quality", "quality_target", "target_resolution"], resolved)
        )
    # Encoder settings feed the same "encoding" config key that
    # Preset.to_config() writes, which Pipeline.execute_job() merges into
    # job.stage_overrides["encode"] -- each is its own leaf entry so deep_merge
    # folds it onto "encoding" without clobbering sibling keys.
    if codec is not None:
        cli_candidates.append((("--codec",), ["encoding", "video_codec"], codec))
    if audio_codec is not None:
        cli_candidates.append((("--audio-codec",), ["encoding", "audio_codec"], audio_codec))
    if crf is not None:
        cli_candidates.append((("--crf",), ["encoding", "crf"], crf))
    if encoder_preset is not None:
        cli_candidates.append((("--encoder-preset",), ["encoding", "preset"], encoder_preset))
    if hwaccel is not None:
        cli_candidates.append((("--hwaccel",), ["ffmpeg", "hwaccel"], hwaccel))
    if gpu_device is not None:
        cli_candidates.append((("--gpu-device",), ["gpu", "preferred_device"], gpu_device))
    if scene_mode is not None:
        cli_candidates.append(
            (("--scene-mode", "--no-scene-mode"), ["scenes", "enabled"], scene_mode)
        )
    if drop_non_content is not None:
        cli_candidates.append(
            (
                ("--drop-non-content", "--no-drop-non-content"),
                ["scenes", "drop_non_content"],
                drop_non_content,
            )
        )
    if crop_limit is not None:
        cli_candidates.append((("--crop-limit",), ["stages", "crop", "limit"], crop_limit))
    if interpolate_hybrid is not None:
        cli_candidates.append(
            (
                ("--interpolate-hybrid", "--no-interpolate-hybrid"),
                ["stages", "interpolate", "hybrid_ai_minterpolate"],
                interpolate_hybrid,
            )
        )
    if interpolate_adaptive is not None:
        cli_candidates.append(
            (
                ("--interpolate-adaptive", "--no-interpolate-adaptive"),
                ["stages", "interpolate", "adaptive_cadence"],
                interpolate_adaptive,
            )
        )
    if interpolate_target_approach is not None:
        cli_candidates.append(
            (
                ("--interpolate-target-approach",),
                ["stages", "interpolate", "target_approach"],
                interpolate_target_approach,
            )
        )
    if interpolate_direct_synthesis_max is not None:
        cli_candidates.append(
            (
                ("--interpolate-direct-synthesis-max",),
                ["stages", "interpolate", "direct_synthesis_max"],
                interpolate_direct_synthesis_max,
            )
        )
    if interpolate_gap_fallback is not None:
        cli_candidates.append(
            (
                ("--interpolate-gap-fallback",),
                ["stages", "interpolate", "gap_fallback"],
                interpolate_gap_fallback,
            )
        )
    if interpolate_max_intermediates_per_gap is not None:
        cli_candidates.append(
            (
                ("--interpolate-max-intermediates-per-gap",),
                ["stages", "interpolate", "max_intermediates_per_gap"],
                interpolate_max_intermediates_per_gap,
            )
        )
    if interpolate_max_gap_sec is not None:
        cli_candidates.append(
            (
                ("--interpolate-max-gap-sec",),
                ["stages", "interpolate", "max_gap_sec"],
                interpolate_max_gap_sec,
            )
        )
    if downscale is not None:
        cli_candidates.append(
            (("--downscale", "--no-downscale"), ["stages", "downscale", "enabled"], downscale)
        )
    if retime is not None:
        cli_candidates.append(
            (("--retime", "--no-retime"), ["stages", "retime", "enabled"], retime)
        )
    if retime_min_duplicate_ratio is not None:
        cli_candidates.append(
            (
                ("--retime-min-duplicate-ratio",),
                ["stages", "retime", "min_duplicate_ratio"],
                retime_min_duplicate_ratio,
            )
        )
    if audio_speed_method is not None:
        cli_candidates.append(
            (
                ("--audio-speed-method",),
                ["stages", "speed", "audio_method"],
                audio_speed_method,
            )
        )
    if audio_sample_rate is not None:
        cli_candidates.append(
            (
                ("--audio-sample-rate",),
                ["stages", "speed", "audio_sample_rate"],
                audio_sample_rate,
            )
        )
    if audio_resampler is not None:
        cli_candidates.append(
            (("--audio-resampler",), ["stages", "speed", "resampler"], audio_resampler)
        )
    if output_timing is not None:
        cli_candidates.append((("--output-timing",), ["general", "output_timing"], output_timing))
    if resolution_fit_mode is not None:
        cli_candidates.append(
            (
                ("--resolution-fit-mode",),
                ["quality", "quality_target", "resolution_fit_mode"],
                resolution_fit_mode,
            )
        )
    if dimension_multiple is not None:
        cli_candidates.append(
            (
                ("--dimension-multiple",),
                ["quality", "quality_target", "dimension_multiple"],
                dimension_multiple,
            )
        )
    if snap_tolerance is not None:
        cli_candidates.append(
            (
                ("--snap-tolerance",),
                ["quality", "quality_target", "snap_tolerance"],
                snap_tolerance,
            )
        )
    if zoom_coverage is not None:
        cli_candidates.append(
            (("--zoom-coverage",), ["stages", "stabilize", "zoom_coverage"], zoom_coverage)
        )
    if batch_size is not None:
        for _stage_name in ("upscale", "deblock", "denoise_video"):
            cli_candidates.append(
                (("--batch-size",), ["stages", _stage_name, "batch_size"], batch_size)
            )
    if tile_batch_size is not None:
        for _stage_name in ("upscale", "deblock", "denoise_video"):
            cli_candidates.append(
                (
                    ("--tile-batch-size",),
                    ["stages", _stage_name, "tile_batch_size"],
                    tile_batch_size,
                )
            )
    if stage_timing_per_video is not None:
        cli_candidates.append(
            (
                ("--stage-timing-per-video", "--no-stage-timing-per-video"),
                ["reporting", "stage_timing_per_video"],
                stage_timing_per_video,
            )
        )
    if stage_timing_totals is not None:
        cli_candidates.append(
            (
                ("--stage-timing-totals", "--no-stage-timing-totals"),
                ["reporting", "stage_timing_totals"],
                stage_timing_totals,
            )
        )
    if stage_timing_averages is not None:
        cli_candidates.append(
            (
                ("--stage-timing-averages", "--no-stage-timing-averages"),
                ["reporting", "stage_timing_averages"],
                stage_timing_averages,
            )
        )
    if report_json is not None:
        cli_candidates.append((("--report-json",), ["reporting", "report_json"], report_json))
    if progress_batch is not None:
        cli_candidates.append(
            (
                ("--progress-batch", "--no-progress-batch"),
                ["reporting", "progress_batch"],
                progress_batch,
            )
        )
    if progress_file is not None:
        cli_candidates.append(
            (
                ("--progress-file", "--no-progress-file"),
                ["reporting", "progress_file"],
                progress_file,
            )
        )
    if color is not None:
        cli_candidates.append(
            (
                ("--color", "--no-color"),
                ["reporting", "color"],
                "always" if color else "never",
            )
        )
    for stage_name in enable_stages:
        cli_candidates.append((("--enable-stage",), ["stages", stage_name, "enabled"], True))
    for stage_name in disable_stages:
        cli_candidates.append((("--disable-stage",), ["stages", stage_name, "enabled"], False))
    for raw in set_overrides:
        key_path, value = _parse_set_option(raw)
        cli_candidates.append((("--set",), key_path, value))

    # Order these entries. When the argv scan above is trustworthy, sort by
    # each entry's actual command-line position -- repeatable flags
    # (--enable-stage/--disable-stage/--set) are matched to their own
    # occurrences in order, so e.g. two --set flags on the same key resolve
    # later-wins correctly; otherwise fall back to the fixed declaration
    # order above (each option's own internal repeat order is still
    # preserved, just not interleaved with other flags).
    if argv_usable:
        _repeat_iters = {
            "--enable-stage": iter([idx for f, _v, idx in occurrences if f == "--enable-stage"]),
            "--disable-stage": iter([idx for f, _v, idx in occurrences if f == "--disable-stage"]),
            "--set": iter([idx for f, _v, idx in occurrences if f == "--set"]),
        }

        def _position(flag_names: tuple[str, ...], fallback_idx: int) -> int:
            if len(flag_names) == 1 and flag_names[0] in _repeat_iters:
                return next(_repeat_iters[flag_names[0]], fallback_idx)
            for flag, _v, idx in occurrences:
                if flag in flag_names:
                    return idx
            return fallback_idx

        positioned = [
            (_position(flags, 10_000 + i), key_path, value)
            for i, (flags, key_path, value) in enumerate(cli_candidates)
        ]
        positioned.sort(key=lambda t: t[0])
        ordered_entries = [(key_path, value) for _pos, key_path, value in positioned]
    else:
        ordered_entries = [(key_path, value) for _flags, key_path, value in cli_candidates]

    final_layer: dict[str, Any] = {}
    for key_path, value in ordered_entries:
        _set_nested(final_layer, key_path, value)
    if final_layer:
        config.apply_layer(final_layer, "cli-flags")

    # Re-resolve reporting.color now that the --color/--no-color CLI-flag
    # layer (if any) is folded in, and apply it retroactively to `console`/
    # `err_console` and the logging RichHandler console -- `main()` already
    # applied the base config's value earlier, before this command's own
    # cascade (preset/config/--set/CLI-flag layers) had been resolved, so
    # this is the point where a `process`-specific --color/--no-color (or a
    # preset/--set override of reporting.color) actually takes effect.
    resolved_color = (config.get("reporting", "color", default="auto") or "auto").lower()
    if resolved_color not in VALID_COLOR_MODES:
        console.print(
            f"[red]Invalid reporting.color: {resolved_color!r} "
            f"(must be one of {VALID_COLOR_MODES!r})[/red]"
        )
        sys.exit(1)
    _apply_color_mode(resolved_color)

    # REQUIREMENTS.md § 6.7: register what's known from the EFFECTIVE config
    # (VLM/LLM endpoints, this run's config layer files, the output dir if
    # set) before re-logging effective settings below -- that DEBUG dump
    # prints api_urls/directories verbatim, so registration must happen
    # before that log call, not merely before setup_logging(). Resolved
    # input files (below) aren't known yet at this point -- registered
    # right after they're collected instead.
    cleaner = get_pii_cleaner()
    cleaner.register_endpoint(config.get("analysis", "vlm", "api_url", default=""))
    cleaner.register_endpoint(config.get("analysis", "llm", "api_url", default=""))
    for layer_path in config_layers:
        cleaner.register_directory(os.path.dirname(os.path.abspath(layer_path)), "config")
    configured_output_dir = config.get("general", "output_dir", default=None)
    if configured_output_dir:
        cleaner.register_directory(configured_output_dir, "output")

    # Re-log the effective settings now that every preset/config/set/CLI
    # layer above has been folded in -- the group-level log in main() only
    # reflects the config as loaded from disk, before any of this command's
    # own layers.
    _log_effective_settings(logger, config, ctx.obj.get("config_path_source"))

    # --from-file resolution: (list_file_path, parser_name) pairs, in
    # command-line order (see _resolve_from_file_specs -- same argv-order-
    # recovery philosophy as the --preset/--config layering above). Validate
    # every selected parser name up front, before any file I/O or job
    # creation, so an unknown parser (stateful or inline) fails fast.
    from_file_specs = _resolve_from_file_specs(from_files, from_file_parsers)
    known_parsers = list_parsers()
    for _list_file, _parser_name in from_file_specs:
        if _parser_name not in known_parsers:
            console.print(
                f"[red]Unknown --from-file-parser {_parser_name!r}. Known parsers: "
                f"{', '.join(sorted(known_parsers))}[/red]"
            )
            sys.exit(1)

    # Collect input files: positional PATHS + every --from-file list file.
    # Each resolved entry is (input_path, output_override | None) --
    # output_override comes from a --from-file spec's per-entry output
    # (csv 'output' column / json "output" key) and is None for everything
    # else, falling through to the normal add_job()-computed output path.
    resolved_inputs: list[tuple[str, str | None]] = []
    for path in paths:
        if os.path.isdir(path):
            resolved_inputs.extend((f, None) for f in scan_directory(path, recursive=recursive))
        elif is_video_file(path):
            resolved_inputs.append((path, None))
        else:
            console.print(f"[yellow]Skipping non-video file: {_safe(path)}[/yellow]")

    configured_output_dir_for_specs = config.get("general", "output_dir", default=None)

    for list_file, parser_name in from_file_specs:
        try:
            with open(list_file, encoding="utf-8") as fh:
                list_text = fh.read()
        except OSError as e:
            console.print(
                f"[red]Failed to read --from-file {_safe(list_file)}: {_safe(str(e))}[/red]"
            )
            sys.exit(1)

        parser = get_parser(parser_name)
        base_dir = os.path.dirname(os.path.abspath(list_file))
        try:
            specs = parser.parse(list_text, base_dir=base_dir)
        except (ValueError, KeyError) as e:
            console.print(
                f"[red]Failed to parse --from-file {_safe(list_file)} with parser "
                f"{parser_name!r}: {_safe(str(e))}[/red]"
            )
            sys.exit(1)

        for spec in specs:
            input_path = spec.input_path
            if not os.path.isabs(input_path):
                input_path = os.path.join(base_dir, input_path)

            if os.path.isdir(input_path):
                entry_recursive = spec.recursive if spec.recursive is not None else recursive
                resolved_inputs.extend(
                    (f, None) for f in scan_directory(input_path, recursive=entry_recursive)
                )
                continue
            if not is_video_file(input_path):
                console.print(f"[yellow]Skipping non-video file: {_safe(input_path)}[/yellow]")
                continue

            output_override = spec.output_path
            if output_override and not os.path.isabs(output_override):
                out_base = configured_output_dir_for_specs or os.path.dirname(input_path)
                output_override = os.path.join(out_base, output_override)
            resolved_inputs.append((input_path, output_override))

    # Create the pipeline before planning -- the planner needs
    # `pipeline.resolve_output_path` to resolve each entry's actual output
    # up front (both to detect fan-out/duplicate destinations and so the
    # fan-out dict key matches the exact string `add_job` will later store
    # as the representative job's `output_path`).
    pipeline = Pipeline(config)
    resolved_inputs, fanout = _plan_resolved_inputs(resolved_inputs, pipeline, console)

    if not resolved_inputs:
        console.print("[red]No video files found.[/red]")
        sys.exit(1)

    input_files = [f for f, _ in resolved_inputs]

    # REQUIREMENTS.md § 6.7: register each resolved input file + its parent
    # directory (role "input") now that they're known -- before dry-run/job
    # creation, so even a --dry-run run's logging is covered. `Pipeline.
    # add_job()` also registers each job's input/output path (idempotent
    # no-op here for inputs already registered by this loop); this is what
    # actually numbers/registers each job's OUTPUT path. Fan-out extra
    # destinations are never passed to add_job (they're copy targets, not
    # jobs), so they're registered here explicitly to keep clean logs from
    # leaking them.
    for f in input_files:
        cleaner.register_input(f)
        cleaner.register_directory(os.path.dirname(os.path.abspath(f)), "input")
    for extras in fanout.values():
        for extra in extras:
            cleaner.register_output(extra)

    total_inputs = len(resolved_inputs)
    has_output_override = any(override is not None for _f, override in resolved_inputs)
    if output_name and total_inputs != 1:
        console.print(
            f"[red]--output-name requires exactly one input file, got {total_inputs}[/red]"
        )
        sys.exit(1)
    if output_name and has_output_override:
        console.print(
            "[red]--output-name cannot be combined with a per-file output from --from-file "
            "(ambiguous)[/red]"
        )
        sys.exit(1)

    console.print(f"Found {total_inputs} video file(s)")

    if dry_run:
        console.print("\n[bold]DRY RUN - No files will be processed:[/bold]")
        for f in input_files:
            console.print(f"  - {_safe(f)}")
        return

    # Process (pipeline was already constructed above, before planning).
    if output_name:
        output_dir = config.get("general", "output_dir", default=None) or os.path.dirname(
            input_files[0]
        )
        jobs = [pipeline.add_job(input_files[0], output_path=os.path.join(output_dir, output_name))]
    else:
        jobs = [
            pipeline.add_job(input_path, output_path=output_override)
            if output_override
            else pipeline.add_job(input_path)
            for input_path, output_override in resolved_inputs
        ]

    # Override stages if specified
    if stages:
        for job in jobs:
            job.stages = list(stages)

    console.print(f"\nProcessing {len(jobs)} job(s)...")

    # Feature 6: live nested progress bars. Each of the two bars is shown
    # only when its own config/CLI flag is truthy AND the console is
    # attached to a real terminal -- piped/redirected/non-TTY output gets no
    # bars, and the run behaves exactly as it did before this feature.
    batch_enabled = config.get("reporting", "progress_batch", default=True)
    file_enabled = config.get("reporting", "progress_file", default=True)
    # console.is_terminal alone can be True even when stdout isn't a real
    # POSIX terminal (e.g. some redirected/piped setups still report as a
    # terminal to Rich) -- also require the real fd-level isatty() signal so
    # a redirected/piped run never leaks bar-redraw control codes into the
    # captured output. Crucially, `reporting.color=always`/--color (see
    # `_apply_color_mode()` above) makes `console.is_terminal` True even for
    # a piped/redirected `console` (that's the whole point of forcing
    # colour) -- so `sys.stdout.isatty()` is the ONLY thing still preventing
    # bars from corrupting a `tee`d file once colour is forced on. Colour and
    # progress bars are deliberately independent controls: forcing colour on
    # must never re-enable bars for non-TTY output, and this AND keeps it
    # that way regardless of which reporting.color value was applied above.
    show_batch, show_file = resolve_show_progress(
        bool(batch_enabled), bool(file_enabled), console.is_terminal and sys.stdout.isatty()
    )
    show_progress = show_batch or show_file
    reporter = (
        ProgressReporter(console, len(jobs), show_batch, show_file) if show_progress else None
    )

    def _job_complete_cb(job: Job, result: JobResult) -> None:
        # Accumulate here rather than relying on execute_all()'s return value:
        # on Ctrl-C (KeyboardInterrupt propagating out of execute_all) the
        # assignment below never runs, so a cancelled run used to report an
        # EMPTY results list -- "Job outcomes aggregate: {}" and a zeroed
        # summary -- even though every finished job had already been reported
        # through this very callback. execute_all() invokes this for every
        # job it completes (both the sequential and the thread-pool path), so
        # appending here makes the `finally` reporting tail see exactly the
        # jobs that actually finished, on cancel and on unexpected-exception
        # paths alike.
        results.append(result)
        _on_job_complete(job, result)
        _print_job_report(logger, result, config)
        extras = fanout.get(job.output_path) if job.output_path else None
        if extras:
            _fanout_copy(job, result, extras, config, console)
        if reporter is not None:
            reporter.on_complete(job, result)

    def _progress_cb(job: Job, progress: float, message: str) -> None:
        if reporter is not None:
            reporter.on_progress(job, progress, message)

    # REQUIREMENTS.md § 6.6: the JSON report is written once at the end of
    # the run, including on partial failure -- whatever jobs finished by the
    # time something goes wrong. `results` starts empty and the whole
    # reporting tail (per-job/aggregate console+log output, JSON write) runs
    # in `finally` so an unexpected exception escaping execute_all() (per-job
    # exceptions are already caught inside it) -- or a KeyboardInterrupt from
    # the user cancelling the run -- still gets a report for whatever's in
    # `results` so far. `results` is populated incrementally by
    # `_job_complete_cb` above, NOT from execute_all()'s return value, which
    # is never assigned when the call is interrupted. The live-progress
    # region (if any) must
    # be stopped BEFORE that reporting tail prints, so the end-of-run summary
    # never gets overwritten/corrupted by the bars -- `with reporter:` (a
    # no-op contextlib stand-in when disabled) wraps only the execute_all()
    # call, not the finally block.
    results: list[JobResult] = []
    run_started_at = datetime.now(timezone.utc)
    try:
        if reporter is not None:
            with reporter:
                pipeline.execute_all(callback=_job_complete_cb, progress_callback=_progress_cb)
        else:
            pipeline.execute_all(callback=_job_complete_cb)
    finally:
        run_finished_at = datetime.now(timezone.utc)
        _print_summary(results)
        _print_run_stage_timing(console, logger, config, results)
        console.print(
            f"\nWhole run elapsed: {(run_finished_at - run_started_at).total_seconds():.3f}s"
        )
        logger.info(
            "Whole run elapsed: %.1fms",
            (run_finished_at - run_started_at).total_seconds() * 1000.0,
        )

        report_json_path = config.get("reporting", "report_json", default=None)
        if report_json_path:
            run_meta = build_run_meta(__version__, config, run_started_at, run_finished_at)
            report = build_json_report(run_meta, results)
            write_json_report(report_json_path, report)
            console.print(f"[green]Wrote JSON run report to {_safe(report_json_path)}[/green]")
            logger.info("Wrote JSON run report to %s", report_json_path)

    # Summary. A run whose jobs are all completed-or-skipped exits 0 -- only
    # true failures (outcome == "failed") make the run exit non-zero (see
    # docs/REQUIREMENTS.md § 6.1).
    failed = _count_failed(results)
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

    table.add_row("Filename", _safe(analysis.filename))
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
        table.add_row("VLM Summary", _safe(summary))
    if analysis.vlm_tags:
        table.add_row("Tags", _safe(", ".join(analysis.vlm_tags)))
    if analysis.vlm_objects:
        table.add_row("Objects", _safe(", ".join(analysis.vlm_objects)))
    if analysis.content_rating:
        table.add_row("Content Rating", _safe(analysis.content_rating))

    console.print(table)

    if full_output and analysis.vlm_summary:
        console.print(
            Panel(
                _safe(analysis.vlm_summary),
                title=f"Full VLM Summary: {_safe(analysis.filename)}",
                expand=True,
            )
        )

    if analysis.scenes:
        console.print(f"\n[bold]Detected {analysis.total_scenes} Scene(s):[/bold]")
        for scene in analysis.scenes[:30]:
            desc = f" - {_safe(scene.description)}" if scene.description else ""
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


# ─── `avf config clean | upgrade | dump` (docs/REQUIREMENTS.md § 6.8) ───────
#
# Pure logic (YAML normalization, ruamel-based template upgrade, file-safety
# rename/overwrite rules) lives in cli/config_tools.py; this is thin Click
# wiring only. See AGENTS.md's "Config tooling" section for the full model.


@main.group("config")
def config_group() -> None:
    """Config file tooling: clean, upgrade, and dump the effective config."""


def _write_config_output(text: str, output: str | None, *, force: bool, backup: bool) -> None:
    """Shared file-safety-aware output writer for all three config subcommands.

    stdout (output=None) never needs any safety handling. Otherwise: refuse
    an existing destination by default, `--force` overwrites, `--backup`
    renames the existing file aside first (see
    config_tools.apply_file_safety) and prints exactly what was renamed.
    """
    if output is None:
        click.echo(text, nl=False)
        return
    out_path = Path(output)
    try:
        message = config_tools.apply_file_safety(out_path, force=force, backup=backup)
    except (ValueError, FileExistsError) as e:
        err_console.print(f"[red]{e}[/red]")
        sys.exit(1)
    out_path.write_text(text)
    if message:
        err_console.print(message)


@config_group.command("clean")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Write to this path instead of stdout."
)
@click.option("--force", is_flag=True, help="Overwrite an existing --output file.")
@click.option(
    "--backup",
    is_flag=True,
    help="Rename an existing --output file aside (PATH.1, PATH.2, ...) before writing.",
)
def config_clean(input_path: str, output: str | None, force: bool, backup: bool) -> None:
    """Normalize INPUT config YAML: strip comments, canonical formatting.

    Emits ONLY the keys actually present in INPUT (no DEFAULTS merged in) --
    this is a formatter, not a validator. Unknown keys pass through
    untouched.
    """
    text = Path(input_path).read_text()
    try:
        cleaned = config_tools.clean_config_text(text)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        sys.exit(1)
    _write_config_output(cleaned, output, force=force, backup=backup)


@config_group.command("dump")
@click.option(
    "--preset",
    "-p",
    "presets",
    multiple=True,
    help="Same as `process --preset` -- applied as its own cascade layer, interleaved with "
    "--config in command-line order.",
)
@click.option(
    "--config",
    "config_layers",
    type=click.Path(),
    multiple=True,
    help="Same as `process --config` -- an additional layer merged on top, interleaved with "
    "--preset in command-line order. Must already exist.",
)
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    help="Same as `process --set KEY=VALUE` -- applied last, repeatable.",
)
@click.option(
    "--with-secrets",
    is_flag=True,
    help="Emit real secret values (api_key, etc.) instead of redacting them to '***'.",
)
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Write to this path instead of stdout."
)
@click.option("--force", is_flag=True, help="Overwrite an existing --output file.")
@click.option(
    "--backup",
    is_flag=True,
    help="Rename an existing --output file aside (PATH.1, PATH.2, ...) before writing.",
)
@click.pass_context
def config_dump(
    ctx: click.Context,
    presets: tuple[str, ...],
    config_layers: tuple[str, ...],
    set_overrides: tuple[str, ...],
    with_secrets: bool,
    output: str | None,
    force: bool,
    backup: bool,
) -> None:
    """Dump the EFFECTIVE config (DEFAULTS + user config.yaml + these layers).

    Accepts the same --preset/--config/--set layering `process` does (only
    these three config-affecting options -- other `process` flags aren't
    part of `dump`), so you can see exactly what a given invocation would
    run with. Secrets are redacted to "***" by default.
    """
    config = copy.deepcopy(ctx.obj["config"])

    occurrences = _scan_config_dump_argv()
    argv_usable = (
        _repeatable_matches(occurrences, {"--preset", "-p"}, presets)
        and _repeatable_matches(occurrences, {"--config"}, config_layers)
        and _repeatable_matches(occurrences, {"--set"}, set_overrides)
    )
    if argv_usable and occurrences:
        layer_seq = [
            (("preset" if flag in ("--preset", "-p") else "config"), val)
            for flag, val, _idx in occurrences
            if flag in ("--preset", "-p", "--config") and val is not None
        ]
    else:
        layer_seq = [("preset", v) for v in presets] + [("config", v) for v in config_layers]

    for kind, value in layer_seq:
        if kind == "preset":
            config.apply_layer(_resolve_preset_layer(value), f"preset:{value}")
        else:
            config.apply_layer(_resolve_config_layer(value), f"config:{value}")

    final_layer: dict[str, Any] = {}
    for raw in set_overrides:
        key_path, value = _parse_set_option(raw)
        _set_nested(final_layer, key_path, value)
    if final_layer:
        config.apply_layer(final_layer, "cli-flags")

    data = config.data if with_secrets else redact_secrets(config.data)
    text = config_tools.dump_config_text(data)
    _write_config_output(text, output, force=force, backup=backup)


@config_group.command("upgrade")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", type=click.Path(), default=None, help="Write to this path instead of stdout."
)
@click.option(
    "--template",
    "template_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Use this template instead of the packaged default (docs/config.example.yaml).",
)
@click.option(
    "--drop-unknown",
    is_flag=True,
    help="Omit INPUT keys absent from the template (with a per-key warning) instead of "
    "refusing to write.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing --output file.")
@click.option(
    "--backup",
    is_flag=True,
    help="Rename an existing --output file aside (PATH.1, PATH.2, ...) before writing.",
)
def config_upgrade(
    input_path: str,
    output: str | None,
    template_path: str | None,
    drop_unknown: bool,
    force: bool,
    backup: bool,
) -> None:
    """Apply INPUT's settings onto the config template, leaf by leaf.

    Output is the template's FULL text (comments, ordering, new options at
    their documented defaults) with each leaf value INPUT specifies replaced
    in place at the same tree path -- lists (e.g. pipeline.default_order)
    are atomic leaves, replaced wholesale, never merged element-wise. Input
    keys absent from the template (renamed/removed/moved options) are
    refused unless --drop-unknown.
    """
    input_text = Path(input_path).read_text()
    template_text = (
        Path(template_path).read_text() if template_path else config_tools.default_template_text()
    )
    try:
        upgraded, warnings = config_tools.upgrade_config_text(
            input_text, template_text, drop_unknown=drop_unknown
        )
    except config_tools.ConfigUpgradeError as e:
        err_console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        sys.exit(1)
    for key in warnings:
        err_console.print(
            f"[yellow]Warning: dropping unknown key not present in template: {key}[/yellow]"
        )
    zoom_note = config_tools.zoom_coverage_migration_note(input_text)
    if zoom_note:
        err_console.print(f"[yellow]Note: {zoom_note}[/yellow]")
    _write_config_output(upgraded, output, force=force, backup=backup)


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
            console.print(f"[yellow]Skipping non-video file: {_safe(path)}[/yellow]")

    if not input_files:
        console.print("[red]No video files found.[/red]")
        sys.exit(1)

    console.print(f"Found {len(input_files)} video file(s)")

    csv_rows: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for idx, filepath in enumerate(input_files, start=1):
        if len(input_files) > 1:
            console.print(
                f"\n[bold]== [{idx}/{len(input_files)}] Analyzing: {_safe(filepath)} ==[/bold]"
            )
        else:
            console.print(f"Analyzing: {_safe(filepath)}")

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
            console.print(
                f"[red]Analysis failed for {_safe(filepath)} -- see log for details.[/red]"
            )
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
                        f"\n[green]Extracted {len(clips)} clip(s) to: "
                        f"{_safe(file_clip_output)}[/green]"
                    )
                    for clip in clips:
                        console.print(
                            f"  Clip {clip.scene_index}: "
                            f"{clip.start_time:.1f}s-{clip.end_time:.1f}s -> "
                            f"{_safe(os.path.basename(clip.output_path))}"
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
    console.print(f"Comparing {_safe(reference)} against {len(candidates)} candidate(s)...")

    results = analyzer.find_similar(reference, candidates, threshold)

    if results:
        console.print(f"\nFound {len(results)} similar video(s):")
        for path, sim in results:
            console.print(f"  [{sim * 100:.1f}%] {_safe(path)}")
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


def _count_failed(results: list[JobResult]) -> int:
    """Count of jobs whose outcome is a true FAILED (REQUIREMENTS.md § 6.1).

    A run's exit code is driven by this, not ``JobResult.success`` --
    SKIPPED jobs have ``success=False`` too (see ``JobResult.__post_init__``)
    but must NOT make an otherwise-clean run exit non-zero. Extracted as its
    own function so the exit-code seam is directly unit-testable without a
    live pipeline run.
    """
    return sum(1 for r in results if r.outcome == "failed")


def _on_job_complete(job, result) -> None:
    """Callback when a job completes."""
    if result.outcome == "completed":
        status_icon = "[green]OK[/green]"
    elif result.outcome == "skipped":
        status_icon = "[yellow]SKIP[/yellow]"
    else:
        status_icon = "[red]FAIL[/red]"
    console.print(f"  {status_icon} {_safe(os.path.basename(job.input_path))}")


def _print_summary(results) -> None:
    """Print processing summary.

    SKIPPED jobs (REQUIREMENTS.md § 6.1 -- an existing output, or an
    unreadable input with general.skip_invalid_inputs) are counted
    separately from both success and failure -- they don't count as a
    "Failed" job, and don't drive the run's exit code (see the `process`
    command's exit-code logic).
    """
    total = len(results)
    success = sum(1 for r in results if r.outcome == "completed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    failed = total - success - skipped

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total: {total}")
    console.print(f"  Success: {success}")
    if skipped:
        console.print(f"  Skipped: {skipped}")
    console.print(f"  Failed: {failed}")

    if skipped:
        console.print("\n[yellow]Skipped jobs:[/yellow]")
        for r in results:
            if r.outcome == "skipped":
                reason = f" ({_safe(r.skip_reason)})" if r.skip_reason else ""
                console.print(f"  - {_safe(r.input_path)}{reason}")

    if failed > 0:
        console.print("\n[red]Failed jobs:[/red]")
        for r in results:
            if r.outcome == "failed":
                console.print(f"  - {_safe(r.input_path)}: {_safe('; '.join(r.errors))}")

    # REQUIREMENTS.md § 6.4 end-of-run aggregate: counts per stage
    # classification bucket (ran-ai/ran-traditional/ran-traditional-fallback/
    # failed/skipped), across every stage occurrence in every job.
    class_counts = run_classification_aggregate(results)
    if class_counts:
        console.print("\n[bold]Stage outcomes (all jobs):[/bold]")
        for cls, count in sorted(class_counts.items()):
            console.print(f"  {cls}: {count}")
        get_logger("autovideofixer.cli").info(
            "Stage outcomes aggregate: %s", dict(sorted(class_counts.items()))
        )

    outcome_counts = run_outcome_aggregate(results)
    get_logger("autovideofixer.cli").info(
        "Job outcomes aggregate: %s", dict(sorted(outcome_counts.items()))
    )


def _print_job_report(logger: logging.Logger, result: JobResult, config: Config) -> None:
    """§ 6.4 per-job stage summary: a Rich table (stage | classification |
    method/fallback | duration | skip reason/error), the video-level "at a
    glance" line (outcome, reprocessed note, scene stats, timing), and the
    same content logged as plain lines (grep-friendly -- Rich console output
    is line-wrapped, see AGENTS.md)."""
    rows = stage_table_rows(result.stage_results)
    if rows:
        table = Table(title=f"Stages: {_safe(os.path.basename(result.input_path))}")
        table.add_column("Stage")
        table.add_column("Outcome")
        table.add_column("Method/Fallback")
        table.add_column("Duration (s)")
        table.add_column("Skip reason / error")
        for row in rows:
            method_col = row["method"] or ""
            if row["ai_fallback_used"]:
                method_col = f"{method_col} (fallback: {row['ai_fallback_reason']})"
            detail = row["skipped_reason"] or row["error"] or ""
            table.add_row(
                _safe(row["label"]),
                _safe(row["classification"]),
                _safe(method_col),
                f"{row['duration_sec']:.2f}",
                _safe(detail),
            )
        console.print(table)
        logger.info(
            "Stage summary for %s: %s",
            os.path.basename(result.input_path),
            [
                f"{r['label']}={r['classification']}"
                + (f" ({r['method']})" if r["method"] else "")
                + f" {r['duration_sec']:.2f}s"
                for r in rows
            ],
        )

    summary_line = job_summary_line(result)
    console.print(f"[bold]{_safe(summary_line)}[/bold]")
    logger.info(summary_line)

    if config.get("reporting", "stage_timing_per_video", default=False) and rows:
        timing_table = Table(title=f"Stage timing: {_safe(os.path.basename(result.input_path))}")
        timing_table.add_column("Stage")
        timing_table.add_column("Duration (s)")
        for row in rows:
            timing_table.add_row(_safe(row["label"]), f"{row['duration_sec']:.2f}")
        console.print(timing_table)
        logger.info(
            "Per-video stage timing for %s: %s",
            os.path.basename(result.input_path),
            {r["label"]: round(r["duration_sec"], 3) for r in rows},
        )


def _print_run_stage_timing(
    console_: Console, logger: logging.Logger, config: Config, results: list[JobResult]
) -> None:
    """§ 6.5 stage-timing summary flags: per-stage totals/averages across the
    whole run, plus a separate "failed stage executions" section -- all OFF
    by default, computed at display time (never stored) from
    ``aggregate_stage_timing()``."""
    want_totals = config.get("reporting", "stage_timing_totals", default=False)
    want_averages = config.get("reporting", "stage_timing_averages", default=False)
    if not (want_totals or want_averages):
        return

    agg = aggregate_stage_timing(results)

    if want_totals and agg["totals"]:
        table = Table(title="Stage totals (whole run)")
        table.add_column("Stage")
        table.add_column("Total duration (s)")
        table.add_column("Successful runs")
        for name in sorted(agg["totals"]):
            table.add_row(
                name, f"{agg['totals'][name] / 1000.0:.2f}", str(agg["counts"].get(name, 0))
            )
        console_.print(table)
        logger.info(
            "Stage totals (ms) across run: %s",
            {name: round(v, 1) for name, v in sorted(agg["totals"].items())},
        )

    if want_averages and agg["averages"]:
        table = Table(title="Stage averages (per video that ran it)")
        table.add_column("Stage")
        table.add_column("Average duration (s)")
        table.add_column("Videos")
        for name in sorted(agg["averages"]):
            table.add_row(
                name,
                f"{agg['averages'][name] / 1000.0:.2f}",
                str(agg["counts"].get(name, 0)),
            )
        console_.print(table)
        logger.info(
            "Stage averages (ms) per video that ran it: %s",
            {name: round(v, 1) for name, v in sorted(agg["averages"].items())},
        )

    if agg["failed"]:
        table = Table(title="Failed stage executions")
        table.add_column("Stage")
        table.add_column("Video")
        table.add_column("Duration (s)")
        table.add_column("Error")
        for entry in agg["failed"]:
            table.add_row(
                _safe(entry["stage"]),
                _safe(os.path.basename(entry["video"])),
                f"{entry['duration_ms'] / 1000.0:.2f}",
                _safe(entry["error"] or ""),
            )
        console_.print(table)
        logger.info("Failed stage executions: %s", agg["failed"])
