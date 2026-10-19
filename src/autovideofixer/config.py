"""Auto Video Fixer - Configuration system."""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from autovideofixer.config_migrations import CURRENT_CONFIG_VERSION

# Set to 1/true/yes/on to stop the first-run bootstrap of the default config (§ 24.3).
BOOTSTRAP_OPT_OUT_ENV = "AVF_NO_CONFIG_BOOTSTRAP"


def get_config_dir() -> Path:
    """Return platform-appropriate config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "auto-video-fixer"


def get_data_dir() -> Path:
    """Return platform-appropriate data directory (models, cache)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "auto-video-fixer"


def get_state_dir() -> Path:
    """Return platform-appropriate state directory (run-time logs, etc.).

    Follows the same pattern as get_config_dir()/get_data_dir(): Linux uses
    XDG_STATE_HOME (falling back to ~/.local/state) per the XDG base
    directory spec; macOS/Windows don't have a separate "state" concept in
    their platform conventions, so they reuse the data dir's base (Windows
    LOCALAPPDATA, macOS Application Support) with a "logs" subdirectory
    layered on top by get_log_dir().
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "auto-video-fixer"


def get_log_dir() -> Path:
    """Return the directory automatic per-run log files are written to."""
    return get_state_dir() / "logs"


# Default for general.log_retention_runs. Kept bounded by default so a busy install
# can't fill its disk with logs; set the key to null/0 for unlimited retention.
DEFAULT_LOG_RETENTION_RUNS = 50

# One automatic per-run log = every file whose name starts with the same run stem:
# "avf-YYYYMMDD-HHMMSS" plus an optional ".N" collision counter (see new_run_log_path),
# followed by whatever general.log_type=both suffixes and extension setup_logging adds
# ("avf-20261018-105406-clean.log", "avf-20261018-105406.2-raw.log", ...).
_RUN_LOG_RE = re.compile(r"^(avf-\d{8}-\d{6}(?:\.\d+)?)(?=[-.])")


def resolve_log_retention(value: Any) -> int | None:
    """Validate ``general.log_retention_runs``: ``None``/``0`` -> unlimited (``None``),
    a positive int -> that many runs. Anything else raises ``ValueError``."""
    if value is None or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Invalid general.log_retention_runs: {value!r} "
            "(must be a positive integer, or null/0 for unlimited)"
        )
    return value


def prune_old_logs(log_dir: Path, keep_runs: int | None = DEFAULT_LOG_RETENTION_RUNS) -> None:
    """Delete automatic per-run logs beyond the ``keep_runs`` newest RUNS.

    Called once at CLI startup, before the current run's log is created, so it
    never counts (or deletes) the run about to start. ``keep_runs=None`` keeps
    everything.

    Counts runs, not files: since general.log_type=both, one run writes two files
    (raw + clean), so the old "newest 50 files" rule silently kept only ~25 runs
    (fewer still with extra files around). A run's age is its newest file's mtime.

    Only files whose names match the automatic log pattern (``_RUN_LOG_RE``) are
    ever considered -- anything else a user drops into the directory (notes,
    archived logs under other names) is never deleted. The old rule deleted
    every file in the directory.
    """
    if keep_runs is None or not log_dir.is_dir():
        return
    runs: dict[str, list[Path]] = {}
    try:
        for p in log_dir.iterdir():
            m = _RUN_LOG_RE.match(p.name)
            if m and p.is_file():
                runs.setdefault(m.group(1), []).append(p)
        newest = {stem: max(f.stat().st_mtime for f in files) for stem, files in runs.items()}
    except OSError:
        return
    ordered = sorted(runs, key=lambda stem: newest[stem], reverse=True)
    for stem in ordered[keep_runs:]:
        for stale in runs[stem]:
            try:
                stale.unlink()
            except OSError:
                pass


def new_run_log_path(log_dir: Path, now: datetime | None = None) -> Path:
    """Reserve and return this run's automatic log path: ``avf-YYYYMMDD-HHMMSS.log``,
    or ``avf-YYYYMMDD-HHMMSS.N.log`` when another run already used that second.

    Without the counter, two runs starting in the same second (back-to-back
    scripted runs, parallel invocations) wrote the same filename and the later
    one overwrote the earlier one's log. Any existing file with the same run stem
    counts as taken, including "both"-mode suffixed files.

    The returned file is CREATED (empty, ``O_EXCL``) so the reservation is atomic:
    a name check alone raced -- two concurrent runs both saw the name free before
    either opened its log (observed with two parallel ``avf`` invocations). The
    log handlers append to it; the caller removes it via
    ``release_unused_run_log()`` if no handler ended up writing there (e.g.
    ``log_type: none``, or a non-empty raw suffix in "both" mode).
    """
    stem = f"avf-{(now or datetime.now()):%Y%m%d-%H%M%S}"
    candidate, n = stem, 1
    while True:
        if not (any(log_dir.glob(f"{candidate}.*")) or any(log_dir.glob(f"{candidate}-*"))):
            path = log_dir / f"{candidate}.log"
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                return path
            except FileExistsError:
                pass  # lost the race to a concurrent run; try the next counter
        n += 1
        candidate = f"{stem}.{n}"


def release_unused_run_log(path: Path, opened: list[str]) -> None:
    """Remove the ``new_run_log_path()`` reservation if logging didn't use it."""
    if str(path) in opened:
        return
    try:
        if path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def get_config_path() -> Path:
    """Return path to the main config file."""
    return get_config_dir() / "config.yaml"


_SECRET_KEY_MARKERS = ("api_key", "apikey", "password", "secret")


def _looks_like_secret_key(key: str) -> bool:
    """Does a config key name look like it holds a credential?

    ``token`` is matched as a whole ``_``/``-``/``.``-separated segment or as the
    key's ending (``access_token``, ``hf_token``, ``authToken``), NOT as a
    substring: a substring match flagged ``max_tokens`` (masked as ``***`` in the
    startup config banner, and refused by ``--set``) and would flag the vision
    budget keys like ``tokens_per_tile``. The other markers stay substrings --
    nothing benign contains them.
    """
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
        return True
    return lowered.endswith("token") or "token" in re.split(r"[_.\-]", lowered)


def redact_secrets(data: Any) -> Any:
    """Return a deep copy of `data` with secret-looking values masked.

    A dict key "looks like a secret" if it contains api_key/apikey/token/
    password/secret (case-insensitive), e.g. "api_key", "API_KEY",
    "openai_api_key". Non-empty values for such keys are replaced with
    "***"; empty/falsy values are left as-is (nothing to leak).
    """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(k, str) and _looks_like_secret_key(k) and v:
                out[k] = "***"
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(data, list):
        return [redact_secrets(v) for v in data]
    return data


# C0 controls except \t (0x09) and \n (0x0A), DEL (0x7F), and the C1
# range (0x80-0x9F). ESC (0x1B) is already inside C0 but is the specific
# byte that kicks off terminal escape sequences (cursor moves, mode
# switches, OSC title-setting, etc.) a raw filename/stderr excerpt could
# smuggle into console output -- called out here for clarity, not handled
# separately.
_UNSAFE_CHARS_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_console_text(s: Any) -> str:
    """Neutralize control bytes that could corrupt a terminal's tty mode.

    Strips/replaces C0 control characters (except \\t and \\n, which are
    harmless and often meaningful in log output), DEL (U+007F), the C1
    control range (U+0080-U+009F), and ESC (U+001B, already covered by the
    C0 range) -- the bytes a malicious or merely exotic filename could use to
    trigger a terminal escape sequence (e.g. switching the tty to raw mode
    without restoring it, as ffmpeg itself is known to do when its stdin is
    the controlling terminal -- see AGENTS.md's "every new subprocess spawn
    must detach stdin" gotcha for the primary fix this is defense-in-depth
    for).

    Each offending character is replaced with U+FFFD (the standard Unicode
    replacement character) -- chosen over caret notation (``^[``) so the
    output length change is visually obvious without trying to look like a
    "real" representation of the stripped byte. All other Unicode -- CJK,
    RTL scripts, combining marks, emoji -- is passed through untouched; this
    must NOT normalize or strip any non-ASCII text, only the specific
    control/format byte ranges above.

    Accepts non-str input via ``str()`` for convenience at call sites that
    interpolate exceptions/paths of unknown type; ``None`` becomes ``""``.
    """
    if s is None:
        return ""
    text = s if isinstance(s, str) else str(s)
    return _UNSAFE_CHARS_RE.sub("�", text)


def diff_from_defaults(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of `data` whose values differ from `defaults`.

    Recurses into nested dicts so only the actually-changed leaf keys show
    up (unchanged sibling keys in a partially-overridden mapping are
    omitted), used to produce a compact "settings that differ from
    DEFAULTS" summary for startup logging.
    """
    diff: dict[str, Any] = {}
    for k, v in data.items():
        default_v = defaults.get(k, _MISSING)
        if isinstance(v, dict) and isinstance(default_v, dict):
            nested = diff_from_defaults(v, default_v)
            if nested:
                diff[k] = nested
        elif default_v is _MISSING or v != default_v:
            diff[k] = v
    return diff


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def deep_merge(base: dict, override: dict, _path: str = "") -> None:
    """Recursively merge `override` onto `base`, in place.

    Shared by ``Config._merge()`` (user config.yaml onto DEFAULTS) and
    ``BaseStage.__init__``'s per-occurrence ``overrides`` param
    (``pipeline.default_order`` entries' ``config:`` key onto the cascaded
    ``stages.<name>`` dict) -- same merge semantics both places: a mapping
    recurses, a scalar overwrites, and a scalar is refused if it would
    clobber a dict-valued default (logged and skipped, not raised) since
    callers unconditionally treat those keys as mappings.
    """
    from autovideofixer.logger import get_logger

    for k, v in override.items():
        key_path = f"{_path}.{k}" if _path else k
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v, key_path)
        elif k in base and isinstance(base[k], dict) and not isinstance(v, dict):
            # Refuse to clobber a default mapping (e.g. stages.upscale, quality.quality_target)
            # with a scalar override -- consumers unconditionally call .get()/.items() on
            # these and would crash with an unhandled AttributeError otherwise.
            get_logger("autovideofixer.config").warning(
                "Ignoring invalid config override for '%s': expected a mapping, got %s",
                key_path,
                type(v).__name__,
            )
        else:
            base[k] = v


def resolve_timeout(value: Any, key: str) -> float | None:
    """Resolve a raw config timeout value (seconds) to ``float | None``.

    Shared by ``BaseStage.stage_timeout()`` (``stages.<name>.timeout`` /
    ``pipeline.stage_timeout``) and ``core/quality.py`` (``quality.timeout``)
    -- both use identical null-unlimited semantics, so the validation lives
    here once instead of being duplicated per call site.

    - ``None`` (absent/explicit ``null``) means "no timeout" (unlimited).
    - ``0`` is accepted as an explicit alias for "unlimited" too, so
      ``--set pipeline.stage_timeout=0`` (which parses as an int, not a YAML
      null) also disables the timeout rather than making every ffmpeg call
      time out instantly.
    - A positive number is seconds.
    - Anything else (negative, non-numeric) is a hard config error, raised
      immediately here ("at use time", i.e. when a stage/quality check
      actually resolves its effective timeout) rather than silently clamped
      or coerced.
    """
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid timeout for {key!r}: {value!r} is not a number or null") from e
    if numeric < 0:
        raise ValueError(
            f"Invalid timeout for {key!r}: {value!r} must be >= 0 (0 or null means unlimited)"
        )
    if numeric == 0:
        return None
    return numeric


_VALID_EXISTING_OUTPUT = ("skip", "fail")
_VALID_EXISTING_MISMATCHED = ("rename", "overwrite")
# Also the canonical set logger.setup_logging()/cli.py validate CLI-flag/
# config values against -- kept here (not duplicated in logger.py) since
# config.py is the layer both cli.py and Pipeline construction already import
# from, and this tuple is otherwise a plain data constant with no logging-
# specific behavior attached.
VALID_LOG_TYPES = ("raw", "clean", "both", "none")
# Canonical set for reporting.color / --color/--no-color (see the DEFAULTS
# comment above for the tri-state's meaning). Kept here for the same reason
# as VALID_LOG_TYPES: config.py is the layer both cli.py and logger.py
# already import from.
VALID_COLOR_MODES = ("auto", "always", "never")
# Canonical set for stages.speed.audio_method -- validated at use-time by
# SpeedStage (same "use-time, not construction-time" posture as
# core/output_check.py's fit_mode check) rather than eagerly here, since an
# invalid value only matters if the speed stage actually runs. Kept here (not
# duplicated in speed.py) for the same reason as VALID_LOG_TYPES/
# VALID_COLOR_MODES: config.py is the layer stages already import from.
VALID_AUDIO_SPEED_METHODS = ("atempo", "rubberband", "asetrate")
# Canonical set for stages.speed.resampler -- the `aresample` resampler used
# by the "asetrate" audio_method (see SpeedStage._build_asetrate_chain).
VALID_AUDIO_RESAMPLERS = ("soxr", "swr")
# Canonical set for pipeline.duration_guard (REQUIREMENTS.md § 19.2).
_VALID_DURATION_GUARD = ("warn", "fail", "off")
# Canonical set for general.symlinks (REQUIREMENTS.md § 21) -- validated at
# use-time by core.analysis.scan_directory() itself (same "use-time, not
# construction-time" posture as VALID_AUDIO_SPEED_METHODS above) rather than
# eagerly here, since an invalid value only matters if a directory scan
# actually runs. Kept here (not duplicated in analysis.py/cli.py) for the
# same reason as every other VALID_* tuple in this module: config.py is the
# layer both cli.py and core/analysis.py already import from.
VALID_SYMLINKS_POLICIES = ("files_only", "follow", "skip")


def validate_output_handling_config(config: "Config") -> None:
    """Validate the § 6.1/6.2/6.7 ``general.*`` enum-valued keys eagerly (also
    ``pipeline.duration_guard``, § 19.2 -- same "fail fast at construction,
    not at first use" rationale, just a different config section).

    Called once per ``Pipeline`` construction so a typo'd value (e.g.
    ``general.existing_output: sikp``) is a clear startup error instead of a
    confusing failure the first time ``execute_job()``'s decision path reads
    it. Deliberately does NOT reject ``mismatched_max_renames == 0`` -- that's
    an intentional, allowed "renaming fully disabled" strict posture (see
    ``docs/REQUIREMENTS.md`` § 6.2's "0 = renaming fully disabled" note), not a
    misconfiguration.

    Also validates ``general.log_type`` (§ 6.7) for programmatic ``Config``/
    ``Pipeline`` users that bypass the CLI entirely -- the CLI's own group
    callback validates it separately (and earlier, before logging is even set
    up) since ``Pipeline`` isn't constructed until well after that point; see
    ``cli.py``'s ``main()``.
    """
    log_type = config.get("general", "log_type", default="raw")
    if log_type not in VALID_LOG_TYPES:
        raise ValueError(
            f"Invalid general.log_type: {log_type!r} (must be one of {VALID_LOG_TYPES!r})"
        )
    existing_output = config.get("general", "existing_output", default="skip")
    if existing_output not in _VALID_EXISTING_OUTPUT:
        raise ValueError(
            f"Invalid general.existing_output: {existing_output!r} "
            f"(must be one of {_VALID_EXISTING_OUTPUT!r})"
        )
    existing_mismatched = config.get("general", "existing_mismatched", default="rename")
    if existing_mismatched not in _VALID_EXISTING_MISMATCHED:
        raise ValueError(
            f"Invalid general.existing_mismatched: {existing_mismatched!r} "
            f"(must be one of {_VALID_EXISTING_MISMATCHED!r})"
        )
    duration_guard = config.get("pipeline", "duration_guard", default="warn")
    if duration_guard not in _VALID_DURATION_GUARD:
        raise ValueError(
            f"Invalid pipeline.duration_guard: {duration_guard!r} "
            f"(must be one of {_VALID_DURATION_GUARD!r})"
        )
    max_renames = config.get("general", "mismatched_max_renames", default=None)
    if max_renames is not None:
        try:
            max_renames = int(max_renames)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid general.mismatched_max_renames: {max_renames!r} "
                "(must be a non-negative integer or null)"
            ) from e
        if max_renames < 0:
            raise ValueError(
                f"Invalid general.mismatched_max_renames: {max_renames!r} "
                "(must be >= 0; 0 disables renaming, null means unlimited)"
            )


class Config:
    """Central configuration manager. Loads from disk and provides defaults."""

    DEFAULTS: dict[str, Any] = {
        # Schema version (REQUIREMENTS.md § 24.1) -- see config_migrations.py. A file without
        # this key is version 1; the loader migrates older files up to this value.
        "config_version": CURRENT_CONFIG_VERSION,
        "general": {
            "output_dir": None,  # None means same as input
            "output_container": "mp4",  # extension for auto-generated output filenames
            # ("mp4" or ".mp4"); None/empty keeps the input's own extension.
            "temp_dir": None,
            "max_concurrent_jobs": 1,
            "log_level": "INFO",
            "overwrite": False,
            "use_ai": None,  # None=auto (preset/config), True=force AI, False=force traditional
            # Global default for AI-capable stages (upscale, interpolate,
            # denoise_video, deblock) when the AI path can't run (torch
            # missing, model load failure, inference exception, CUDA OOM
            # after tiling retries are exhausted). True (default) = fall
            # back to the stage's traditional FFmpeg implementation with a
            # WARNING log. False = fail the stage instead of silently
            # producing traditional output. Per-stage
            # stages.<name>.ai_fallback overrides this when not null.
            "ai_fallback": True,
            # --- REQUIREMENTS.md § 6.1/6.2/6.3: existing-output handling and
            # input-probe policy (Pipeline.execute_job()'s decision path, right
            # after the input probe and before any stage runs) ---
            #
            # "skip" (default): an existing job.output_path with overwrite=False
            # is a new first-class SKIPPED outcome (INFO log), not a failure --
            # a video that's simply already done from a previous run shouldn't
            # read as "failed". "fail" restores the old behavior exactly: FAILED
            # + ERROR log.
            "existing_output": "skip",
            # When an output exists and is about to be SKIPPED, verify it
            # actually satisfies the CURRENT effective targets (container,
            # resolution, framerate, video/audio codec -- see
            # core/output_check.py) before trusting it. Default ON: ffprobe-ing
            # an existing file is cheap and the info is valuable (catches e.g.
            # a config change since the file was produced). False skips the
            # verification and always accepts an existing file at face value.
            "check_existing_target": True,
            # A verified mismatch (check_existing_target=true found the existing
            # file doesn't match) is SKIPPED by default -- set true to have that
            # one video actually reprocessed instead.
            "reprocess_mismatched": False,
            # What happens to the OLD mismatched file when reprocess_mismatched
            # is true: "rename" (default, never overwrite -- see
            # mismatched_rename_suffix/mismatched_max_renames below) or
            # "overwrite" (replace it directly).
            "existing_mismatched": "rename",
            # Rename pattern for a mismatched existing file being replaced:
            # "<stem><suffix><N><ext>", N from 1, first unused name wins.
            "mismatched_rename_suffix": "_mismatched-",
            # Cap on N above. null (default) = unlimited. A positive int fails
            # the job (explicit reason) once every N up to the cap is taken.
            # 0 = renaming fully disabled: a mismatch in rename mode then FAILS
            # rather than silently overwriting or silently skipping -- a
            # deliberate strict "never silently rename or overwrite" posture,
            # intentionally allowed here (not rejected as invalid config).
            "mismatched_max_renames": None,
            # An input ffprobe can't analyze at all (corrupt/unreadable file)
            # FAILS that job by default (ffprobe stderr surfaced in the log/
            # error). Set true for an opt-in "scavenge" batch mode: unanalyzable
            # inputs become SKIPPED (sub-reason "invalid-input") instead, so a
            # batch with known-bad members still completes; the failure is still
            # logged either way.
            "skip_invalid_inputs": False,
            # ffprobe can emit warnings on stderr even on a successful (rc=0)
            # probe (see core/ffmpeg_utils.probe()'s "-v error"). Non-empty
            # stderr on a successful probe is always surfaced as a WARNING log
            # by default; set true for an opt-in strict mode where that also
            # fails the job outright.
            "fail_on_probe_warnings": False,
            # --- REQUIREMENTS.md § 6.7: PII-clean log variant ---
            #
            # Which log FILE variant(s) get written: "raw" (default, today's
            # behavior -- unredacted paths/endpoints/titles), "clean"
            # (substitutes known real values -- input/output filenames,
            # input/output/config directories, VLM/LLM endpoint hosts, embedded
            # video titles -- with per-run-consistent placeholders, safe to
            # share when asking for help), "both" (two files, see
            # log_suffix_raw/log_suffix_clean below), or "none" (no file
            # logging at all). The CONSOLE is always raw regardless of this
            # setting -- only file logging is affected. Only takes effect for
            # command-line runs when the `avf --log-type` flag is absent (the
            # CLI flag wins when both are given) -- see AGENTS.md's "Log
            # types" note for why a `--set general.log_type=...`/`--config`
            # passed to `process` does NOT affect logging: the group callback
            # attaches log handlers before `process` builds its config
            # cascade, so this key is only read from the user config file (or
            # an explicit top-level `avf --config PATH`), never from a
            # `process`-level layer.
            "log_type": "raw",
            # Suffix inserted into the log filename for the RAW file in "both"
            # mode (before the extension when one exists, e.g. "run.log" ->
            # "run-raw.log"; appended to the end for an extensionless name).
            # Empty string (default) = no suffix -- the raw file keeps the
            # base name exactly (custom --log-file or the auto-generated
            # timestamped name).
            "log_suffix_raw": "",
            # Same as log_suffix_raw but for the CLEAN file in "both" mode.
            # If both suffixes would produce IDENTICAL paths (e.g. both left
            # empty), that's a config error at startup (clear message,
            # non-zero exit) -- never a silent overwrite of one file by the
            # other.
            "log_suffix_clean": "-clean",
            # How many RUNS of automatic per-run logs (under the platform state
            # dir's logs/) to keep; older runs are deleted at startup. A run's
            # raw + clean files count as ONE run. Only files named like AVF's own
            # automatic logs are ever deleted. null or 0 = keep everything. Bounded
            # by default so a busy install can't fill its disk. Like log_type, this
            # is read from the base config file only (logging is set up before
            # `process` layers are applied). Has no effect with --log-file.
            "log_retention_runs": DEFAULT_LOG_RETENTION_RUNS,
            # REQUIREMENTS.md § 12.5: governs ONLY the final `encode` stage's
            # ffmpeg `-fps_mode`. "cfr" (default) keeps the deliverable
            # constant-framerate at the target rate -- the same posture as
            # before this feature, now explicit rather than relying on
            # ffmpeg's own muxer-dependent implicit default. "vfr" carries
            # the recovered/genuine variable-framerate timeline all the way
            # out to the final file. "passthrough" is the same idea without
            # ffmpeg's timestamp-monotonicity normalization. Intermediates
            # produced by every stage BEFORE encode are always VFR-safe
            # (``-fps_mode passthrough``) regardless of this key -- it only
            # affects the last encode.
            "output_timing": "cfr",
            # REQUIREMENTS.md § 21: how core.analysis.scan_directory() (and
            # therefore `process`/`analyze`/`find-duplicates`/the GUI's
            # "Add Directory" scan -- see AGENTS.md for the full caller
            # list) treats symlinks encountered while scanning a directory
            # for video files.
            #
            # "files_only" (default): today's original behavior, now
            # documented rather than accidental -- a symlinked FILE is
            # followed/picked up normally, but a symlinked DIRECTORY is
            # never descended into (matches os.walk's own followlinks=False
            # default). This asymmetry is exactly why a user couldn't tell
            # whether symlinks were being ignored: a symlink to a single
            # video was processed, but a symlink to a folder of videos was
            # silently invisible.
            # "follow": also descend into symlinked directories. Guards
            # against symlink LOOPS (a -> b -> a) by tracking visited real
            # directories (os.path.realpath) and refusing to re-enter one --
            # plain os.walk(followlinks=True) has no such guard and would
            # hang forever on a cycle. A file reachable via both the symlink
            # and its real path can surface twice; the existing per-run
            # input dedup (keyed on realpath first) already collapses that.
            # "skip": ignore every symlink, file or directory -- tested via
            # os.path.islink() on the entry itself, not the resolved target
            # (a symlink to a genuinely valid video is still skipped).
            "symlinks": "files_only",
        },
        "gpu": {
            "auto_detect": True,
            "preferred_device": "auto",  # auto, cpu, cuda, metal
            "memory_limit_gb": None,
            # Which Vulkan physical device index the ncnn backend uses (stages.<name>.backend:
            # ncnn only -- irrelevant to the torch backend, which uses gpu.preferred_device
            # instead). 0 = ncnn's own default device. On a multi-GPU machine this is NOT
            # necessarily your fastest/discrete GPU -- e.g. an iGPU can enumerate before a
            # passed-through dGPU. Check `avf gpu-info` (or `ncnn.get_gpu_count()`/
            # `ncnn.get_gpu_info(i).device_name()`) to find the right index.
            "vulkan_device": 0,
            # Process-wide cap on concurrent GPU AI inferences -- currently only gates
            # scene-mode's AI/RIFE interpolation path (see
            # ai/torch_utils.get_gpu_inference_semaphore / core/scenes.py's
            # interpolate_scene_clip), which otherwise runs up to scene_workers threads
            # each launching a full RIFE inference and contending for VRAM instead of
            # parallelizing. Whole-video (non-scene) runs execute stages serially
            # already, so this never gates them. 1 (default) = fully serialize GPU
            # inference across scene threads -- the safe default on a single GPU.
            # Raise only if you know your GPU has VRAM headroom for concurrent
            # inferences. This is also the single-GPU placeholder for future
            # multi-GPU inference distribution -- see docs/ROADMAP.md.
            "max_concurrent_inferences": 1,
        },
        "ffmpeg": {
            "binary": None,  # None = auto-detect in PATH
            "hwaccel": "auto",  # auto, cuda, vulkan, qsv, vaapi, none
            "threads": 0,  # 0 = auto
        },
        "quality": {
            "vmaf_model": "vmaf_v0.6.1",
            "vmaf_features": "psnr,ssim,ms_ssim,fast",
            # Timeout (seconds) for the quality gate's whole-video VMAF/SSIM
            # ffmpeg comparison pass (core/quality.py) -- not a stage, so it
            # gets its own key rather than living under stages.<name>. None
            # (default) = unlimited: a fixed wall-clock cap on a whole-video
            # comparison pass is wrong-shaped for long inputs, same reasoning
            # as pipeline.stage_timeout below. 0 is accepted as an alias for
            # null. A positive number is seconds; negative/non-numeric raises
            # a config error when the quality gate actually runs. See
            # resolve_timeout() in this module.
            "timeout": None,
            "quality_target": {
                "mode": "none",  # none, min, avg, max
                "target": 95.0,
                "max_loss_pct": 5.0,
                "target_resolution": None,  # (width, height) or None
                "target_framerate": None,
                "keep_aspect_ratio": True,  # preserve original video aspect ratio
                # How UpscaleStage/DownscaleStage fit an input into
                # target_resolution's (orientation-aware) box -- see
                # core.output_check.compute_fitted_dimensions() and
                # docs/REQUIREMENTS.md § 7. "preserve_aspect" (default) keeps
                # the input's exact aspect ratio, fits it entirely within the
                # target box, and rounds up to dimension_multiple.
                # "snap_limiting" forces the LIMITING axis exactly onto the
                # target and derives+rounds the other axis, allowing a
                # sub-percent aspect shift in exchange for clean standard
                # dimensions (e.g. exactly 1920x1080 instead of 1920x1078).
                "resolution_fit_mode": "preserve_aspect",
                # Output dimensions are rounded to a multiple of this. 2 is
                # required by H.264/yuv420p (chroma subsampling needs even
                # dimensions); raise for stricter codecs that need e.g. a
                # multiple of 4/8/16.
                "dimension_multiple": 2,
                # Only used by resolution_fit_mode: "snap_limiting" --
                # "snap-to-box-when-close". The limiting axis (the one that
                # binds the aspect-preserving fit) always lands exactly on
                # its target bound; the OTHER (derived) axis is snapped
                # exactly onto ITS bound too -- both dimensions land on the
                # full target box -- whenever it would otherwise fall short
                # by no more than this fraction (e.g. 0.01 = 1%), accepting
                # a sub-percent aspect-ratio shift for a clean standard
                # resolution (e.g. 1440x812 -> exactly 1920x1080 instead of
                # ~1920x1078). Beyond this tolerance (a genuinely different
                # aspect ratio, not just off-by-a-few-px), the derived axis
                # is left at its aspect-preserving value instead.
                "snap_tolerance": 0.01,
            },
        },
        "pipeline": {
            # Drives actual execution order/omission/repetition --
            # Pipeline.resolve_stage_order() (called by optimize_stage_order()/
            # execute_job()) reads this list; the hardcoded DEFAULT_STAGE_ORDER
            # fallback in core/pipeline.py only kicks in if this key is
            # missing/empty. Entries are either a plain stage name (string --
            # behaves exactly as before: runs iff the stage is in the
            # requested/auto-determined set) or a mapping
            # ``{stage, enabled, config}`` for explicit per-occurrence control
            # (force-run/force-skip regardless of global gating, and/or
            # per-occurrence config overrides) and repetition (the same stage
            # name can appear more than once; each occurrence resolves and
            # runs independently). See docs/config.example.yaml for the full
            # mapping-entry syntax and AGENTS.md's "Pipeline Behavior" section
            # for semantics. DEFAULTS itself stays plain strings so
            # out-of-box behavior is unaffected by this feature.
            #
            # crop now runs FIRST (right after detect), before stabilize:
            # stabilize's zoom is now a real percentile-based "borderless"
            # zoom (not the old motion-guess zoom), so it no longer leaves a
            # black border for a later crop to clean up -- there's nothing
            # left for a post-stabilize crop to do. Running crop first also
            # means downscale (see below) and every downstream stage size
            # off the already-cropped content instead of the full frame.
            #
            # deblock now defaults to the compact, denoise-optimized
            # realesr-general-wdn-x4v3 model (see stages.deblock.ai_model
            # below) instead of the RRDB RealESRGAN_x4plus checkpoint. That
            # model was trained on real-world degradation including both
            # compression/blocking AND noise, so one early AI pass now
            # handles both artifact classes -- which is why
            # stages.denoise_video.enabled defaults to False (see that key's
            # own comment). deblock still runs before stabilize: blocking
            # artifacts come from the source video, so deblocking before
            # stabilization's perspective warping keeps the deblock model's
            # input accurate (no warped block edges) and gives the
            # stabilizer cleaner detail to track motion against.
            #
            # downscale sits right after crop: it's opt-in and off by
            # default (stages.downscale.enabled), and shrinks an oversized
            # input down to the target resolution box BEFORE the heavier
            # deblock/denoise_video/upscale/interpolate stages run, so they
            # never spend compute on pixels a downscale would shrink away
            # anyway. See core/stages/downscale.py.
            #
            # retime runs immediately after detect, before every other
            # stage (REQUIREMENTS.md § 12): it recovers the input's TRUE
            # content cadence (as opposed to the encoded/container
            # framerate) and drops padding/duplicate frames, so the reduced,
            # honest frame set -- and the compute saving -- carries through
            # every later stage. Enabled by default; SKIPs (never fails)
            # cheaply on an already-honest input. See core/stages/retime.py
            # and core/cadence.py.
            #
            # normalize_codecs runs right after retime, before every heavy stage
            # (REQUIREMENTS.md § 22.2): it rewrites only the streams whose codec
            # is "unusual" (VP9/AV1 video, Opus/Vorbis/FLAC/PCM audio -- typical
            # YouTube .webm) into H.264/AAC so no later stage ever stream-copies
            # a codec its container (or ffmpeg build) can't hold. A no-op SKIP
            # when everything is already H.264/AAC. See
            # core/stages/normalize_codecs.py.
            #
            # deinterlace runs right after retime(+normalize_codecs), before crop (REQUIREMENTS.md
            # § 20.1): every downstream stage (crop bounds, scene detection,
            # RIFE interpolation) benefits from seeing progressive frames, and
            # comb artefacts actively corrupt motion-based analysis. Off by
            # default -- see stages.deinterlace.enabled below for why. See
            # core/stages/deinterlace.py.
            "default_order": [
                "detect",
                "retime",
                "normalize_codecs",
                "deinterlace",
                "crop",
                "downscale",
                "deblock",
                "stabilize",
                "denoise_video",
                "upscale",
                "interpolate",
                "normalize_volume",
                "normalize_audio",
                "speed",
                "hdr",
                "encode",
            ],
            # Must stay >= len(default_order) above (16) since a full default run
            # legitimately uses every stage; this only guards against pathological
            # --stage/default_order repetition, not normal preset/auto-determined
            # pipelines. Counts resolved OCCURRENCES, not unique stage names -- a
            # stage repeated via default_order counts once per repetition. Kept at
            # len(default_order) + 2 (two repeats of headroom): bump it whenever a
            # stage is added to default_order.
            "max_stages": 18,
            "skip_stage_on_error": True,
            # Default timeout (seconds) for a stage's MAIN ffmpeg processing/mux
            # pass(es) -- see BaseStage.stage_timeout() in core/stages/base.py.
            # None (default) = unlimited. Real-world long videos legitimately
            # take longer than any fixed wall-clock cap on a whole-video pass
            # (the old hardcoded timeout=600/1800/3600 sprinkled across
            # core/stages/*.py died with "timeout reached" on inputs that were
            # simply long, not stuck) -- a hang is instead something the
            # caller/OS-level job runner should notice from external
            # inactivity, not something this pipeline should second-guess with
            # an arbitrary per-run cap. Set a positive number of seconds here
            # to restore a global cap (e.g. for a batch job runner that wants
            # to bound worst-case wall time), or per-stage via
            # stages.<name>.timeout (falls back to this key when absent), or
            # per-occurrence via a pipeline.default_order mapping entry's
            # ``config: {timeout: ...}`` (see that key's own docs above). 0 is
            # accepted as an alias for null/unlimited. Negative/non-numeric
            # values raise a config error when a stage actually resolves its
            # effective timeout (BaseStage.stage_timeout() / resolve_timeout()
            # in config.py) -- not validated eagerly at config-load time.
            # Short, genuinely-bounded helper calls (probes, hwaccel
            # detection, single-frame extraction, cropdetect quick samples,
            # loudness measurement) intentionally keep their own small fixed
            # timeouts regardless of this key -- a hang there indicates real
            # breakage, not a long input.
            "stage_timeout": None,
            # REQUIREMENTS.md § 19.2: duration invariant check -- the pipeline-
            # level catch-all half of the duration/cadence guard (§ 19.1's
            # per-stage frame-count check, in core/cadence_guard.py and the
            # raw-pipe stages, is the other half). After a stage that declares
            # `preserves_duration = True` (BaseStage's default -- see that
            # class attribute's own comment; `speed` and `retime` are the only
            # stages that override it to False) produces a NEW output file,
            # Pipeline.execute_job() compares that file's duration to its
            # input's.
            #
            # This exists because of a real incident: a stale `avf_framepipe`
            # build (see stages.retime's § 12.6 build-gotcha reference) made a
            # raw-pipe stage re-expand a 235-frame retimed VFR intermediate
            # back to 714 CFR frames, then tag those 714 frames at the
            # recovered 9.864fps cadence -- correct only for the 235-frame
            # stream -- producing a 72.4s output from a 23.8s input. Every
            # stage reported COMPLETED, the run exited 0, and nothing checked.
            # This guard is the tripwire: it would have caught that 3x
            # disagreement even with no per-stage cadence awareness at all.
            #
            # "warn" (default): log loudly (WARNING) and record the
            # discrepancy on the job result (JobResult.duration_guard_warnings
            # in core/pipeline.py) -- the job still completes. "fail": mark
            # the offending stage FAILED instead, so
            # pipeline.skip_stage_on_error=false stops the run rather than
            # building further stages on a broken intermediate (with the
            # default skip_stage_on_error=true, a "fail" verdict here behaves
            # like any other stage failure: logged as an error, pipeline
            # continues from the pre-stage file). "off" disables the check
            # entirely. Defaults to "warn" rather than "fail": duration
            # estimates from container metadata are not always exact (VFR
            # files especially -- see stages.retime's § 12.3 reference on
            # `avg_frame_rate` being unreliable there), and a guard that fails
            # good jobs on false alarms is worse than one that flags bad ones
            # loudly. A duration that's unknown/zero on either side (common
            # for VFR containers, and for the mocked probes many pipeline
            # tests use) skips the check rather than dividing by zero or
            # firing a false alarm.
            "duration_guard": "warn",
            # Fraction (0.05 = 5%) an output's duration may differ from its
            # input's before pipeline.duration_guard fires, for a stage that
            # declares preserves_duration = True.
            "duration_guard_tolerance": 0.05,
        },
        "stages": {
            "upscale": {
                "enabled": True,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "superres",
                "scale_factor": 4,
                "tta_mode": 0,
                # 0 = auto: tile only when a frame's resolution risks CUDA OOM
                # (see RealESRGANUpscaler.AUTO_TILE_THRESHOLD_PX) or on a
                # caught OOM. Set > 0 to always tile at that pixel size.
                "tile_size": 0,
                # Frames batched into one forward pass in RealESRGANUpscaler.
                # upscale_video() (whole-*frame* batching -- N different
                # frames per call). 1 (default) = today's one-frame-at-a-time
                # behavior; opt-in until field-tested. Only takes effect for
                # frames small enough to skip tiling -- see tile_batch_size
                # below for the batching that matters at e.g. 4K, where every
                # frame always tiles regardless of this setting.
                "batch_size": 1,
                # Tiles (sharing the same padded input shape) batched into one
                # forward pass inside run_tiled_inference(), when a frame's
                # resolution triggers tiled inference (see tile_size/
                # AUTO_TILE_THRESHOLD_PX above). 1 (default) = today's
                # one-tile-at-a-time behavior. This is the batching knob that
                # matters for large frames -- a 4K frame at the default
                # tile_size=512 needs a 5x8=40-tile grid, so raising this
                # collapses many small sequential forward passes into a
                # handful of larger ones. Independent of batch_size above
                # (different batching axis: N tiles of ONE frame, vs N whole
                # frames) -- opt-in for the same reason.
                "tile_batch_size": 1,
                # None = inherit general.ai_fallback; True/False overrides it
                # for this stage only.
                "ai_fallback": None,
                # None (default) = auto: use AI when the needed scale exceeds
                # _SKIP_SCALE_THRESHOLD, else traditional lanczos scaling.
                # True/False forces AI/traditional for this stage only,
                # overriding the auto scale-threshold logic but NOT an
                # explicit method= kwarg from a caller. See
                # BaseStage.resolve_ai_method / AGENTS.md's "AI/Traditional
                # Method Selection" for the full precedence (explicit
                # method= kwarg > this key > general.use_ai > auto default).
                "use_ai": None,
                # Inference backend: "torch" (PyTorch/CUDA) or "ncnn" (Vulkan via
                # the ncnn Python package -- portable to AMD/Intel/iGPUs). Falls
                # through the ai_fallback policy if the backend is unavailable.
                "backend": "torch",
                # CRF for the AI stage's internal temp encode (StreamingVideoWriter/
                # frames_to_video), NOT the final output. Default 16 (near-visually-
                # lossless) rather than libx264's own default (23) -- the temp file
                # used to be encoded at CRF 23 and then RE-encoded by the stage's mux
                # pass at CRF 18, a double lossy generation plus a wasted full x264
                # pass; the mux pass now stream-copies (`-c:v copy`) instead, so this
                # is the ONLY encode the AI-processed frames actually go through.
                "temp_crf": 16,
                # Frame-transport backpressure knobs (ai/frame_pipe.py) --
                # only meaningfully vary the Rust (avf_framepipe) backend's
                # bounded channels; the pure-Python fallback's writer queue
                # depth is a fixed constant in frame_processor.py (see
                # ai/frame_pipe.py's "fallback parity gaps" docstring note).
                # read_ahead: max decoded chunks buffered ahead of inference.
                "read_ahead": 2,
                # write_queue_depth: max encoded chunks buffered ahead of the
                # ffmpeg encoder pipe.
                "write_queue_depth": 4,
            },
            "retime": {
                # REQUIREMENTS.md § 12: recovers the input's TRUE content
                # cadence (as opposed to the encoded/container framerate)
                # via a single decode-only mpdecimate + metadata=print pass
                # (core/cadence.py's analyze_cadence()), and -- when the
                # input is actually padded -- drops the duplicate/padding
                # frames while preserving genuine VFR timing
                # (`-fps_mode vfr`). On by default: the analysis pass is
                # cheap (decode-only, no encode) relative to any real
                # processing stage, and SKIPs (never fails, never re-encodes)
                # on an input that's already honest. See core/stages/
                # retime.py.
                "enabled": True,
                # mpdecimate hi/lo/frac thresholds (encoder-noise tolerance
                # for "these two frames are near-duplicates"). hi/lo are
                # 8x8-pixel-block noise-metric thresholds (mpdecimate's own
                # units); frac is the fraction of blocks that must be below
                # `lo` OR the fraction below `hi` overall, whichever
                # `mpdecimate` triggers on. See ffmpeg's mpdecimate filter
                # docs for the precise semantics; these defaults (768=64*12,
                # 320=64*5, 0.33) are ffmpeg's own filter defaults, chosen
                # for its documented (and empirically validated -- see
                # REQUIREMENTS.md § 12.6) real-world padding-detection
                # behavior.
                "hi": 768,
                "lo": 320,
                "frac": 0.33,
                # Below this duplicate_ratio (= 1 - unique_frames/total_frames,
                # from the analysis pass), the input is treated as already
                # honest and the stage SKIPs without ever running the real
                # mpdecimate re-encode pass. 0.05 = at least 5% of frames
                # must be near-duplicates before retime bothers.
                "min_duplicate_ratio": 0.05,
                # Analysis pass is limited to the first N seconds of content
                # (cheaper decision -- the actual retime re-encode pass, once
                # triggered, always runs mpdecimate over the WHOLE input
                # regardless of this). 0 = analyse the whole file.
                "analysis_sample_sec": 120,
                # nominal_fps snapping window: detected_fps snaps to the
                # nearest standard rate (23.976/24/25/29.97/30/48/50/59.94/
                # 60/100/120) when within this RELATIVE tolerance (e.g.
                # 0.02 = 2%), else stays at the raw detected value.
                "snap_tolerance": 0.02,
                # is_regular classification: the coefficient of variation
                # (std/mean) of the recovered timeline's inter-frame gaps
                # must be under this to count as a uniform (not genuinely
                # variable-framerate) cadence. Also reused as the tolerance
                # for grid_rate's "does every gap divide evenly onto the
                # encoded grid" check (see core/cadence.py).
                "regularity_tolerance": 0.15,
                # CRF for the retime stage's own internal re-encode pass
                # (NOT the final output -- this is an intermediate .mkv temp
                # every later stage reads from). Default 16 (near-visually-
                # lossless), matching the AI stages' own temp_crf default.
                "temp_crf": 16,
            },
            "normalize_codecs": {
                # REQUIREMENTS.md § 22.2. Makes every stream the pipeline keeps
                # one of a small universally-compatible set of codecs, touching a
                # stream ONLY if it has to (SKIPs with no new file when nothing
                # needs changing). Why it exists: stages stream-copy source
                # streams (`-c:a copy`, `-c copy`) into containers that may not
                # hold them (Vorbis in MP4, ...), killing the job; normalizing
                # once, up front, makes every later copy path safe without each
                # stage reasoning about codecs. Video is re-encoded even though a
                # later stage often re-encodes again, because this temp is
                # visually lossless (temp_crf) and the alternative is per-stage
                # codec special-casing. Disable only if you know every input is
                # already H.264/AAC (then it SKIPs for free anyway).
                "enabled": True,
                # Video codecs (ffprobe codec_name) kept with `-c:v copy`.
                # Anything else is re-encoded with `video_encoder`.
                "video_codecs_allowed": ["h264"],
                # Audio codecs kept with `-c:a copy`, per audio stream. Anything
                # else is re-encoded with `audio_codec` at `audio_bitrate`.
                "audio_codecs_allowed": ["aac"],
                "video_encoder": "libx264",
                # Temp quality for a video transcode -- matches the AI stages'
                # and retime's temp_crf (near-visually-lossless intermediate).
                "temp_crf": 16,
                "preset": "medium",
                "audio_codec": "aac",
                "audio_bitrate": "256k",
                # Per-stage timeout (seconds); None -> pipeline.stage_timeout.
                "timeout": None,
            },
            "deinterlace": {
                # REQUIREMENTS.md § 20: converts interlaced (fielded) content
                # to progressive frames. Runs right after retime, before crop
                # -- see pipeline.default_order above.
                #
                # MUST default False. idet (the detector below) is a
                # heuristic and it false-positives on detailed progressive
                # motion: measured on the reference build (ffmpeg n9.0.1), a
                # progressive testsrc2 clip reported TFF: 31, Progressive: 0
                # (i.e. it looked "genuinely interlaced" by count alone),
                # genuinely interlaced content reported TFF: 50,
                # Progressive: 0, and a static progressive clip reported
                # Undetermined: 50. Deinterlacing progressive content is
                # destructive -- it halves either resolution (bob) or frame
                # rate (frame mode) for no reason, or duplicates rows/motion
                # that was never actually fielded -- so this must never
                # auto-enable off a probe. Opt in explicitly once you know
                # the source is interlaced.
                "enabled": False,
                # Output rate mode:
                #   "frame" -- one progressive frame per input FRAME (half
                #     the field rate, e.g. 50i -> 25p). Lowest cost, discards
                #     half the temporal information. <filter>=mode=0.
                #   "field" (default once enabled) -- one progressive frame
                #     per FIELD, preserving full motion (e.g. 50i -> 50p).
                #     <filter>=mode=1. This DOUBLES the frame count, so the
                #     stage reports fps_out in its metadata (REQUIREMENTS.md
                #     § 12.5) -- without that, encode's CFR path would pin
                #     the pre-deinterlace rate and duplicate frames to reach
                #     it.
                #   "bob" -- each field line-doubled to full height at the
                #     full FIELD rate (also doubles frame count, same
                #     fps_out reporting as "field"). Cheapest, visibly
                #     softer than a real motion-adaptive deinterlacer.
                #     None of the four filters below expose a literal
                #     line-doubling mode (checked against ffmpeg n9.0.1's
                #     own `-h filter=<name>` for all four -- each only
                #     supports mode=0/frame and mode=1/field, nothing else),
                #     so "bob" is built from a separate, filter-INDEPENDENT
                #     chain instead (setfield+separatefields+scale) --
                #     stages.deinterlace.filter is ignored when mode is
                #     "bob". See core/stages/deinterlace.py.
                "mode": "field",
                # Which deinterlacing filter to use for "frame"/"field" mode
                # (ignored for "bob", see above). "bwdif" (default -- best
                # quality of the motion-adaptive set), "yadif" (fastest, most
                # familiar), "w3fdif", "estdif". All four are present in the
                # reference build (ffmpeg n9.0.1) and -- verified against
                # that build -- all four support the same mode=0/1 (frame/
                # field) option identically; there is no per-filter mode
                # restriction to map around in v1. CUDA variants
                # (yadif_cuda, bwdif_cuda) exist but are out of scope: this
                # stage is a plain ffmpeg filter pass, not part of the
                # raw-frame AI pipe.
                "filter": "bwdif",
                # Field order: "auto" (default, decided by idet -- see
                # min_interlaced_ratio below), "tff" (top-field-first),
                # "bff" (bottom-field-first). Getting this backwards produces
                # juddering worse than leaving the content interlaced, so an
                # explicit tff/bff is always honored outright and SKIPS idet
                # detection entirely.
                "field_order": "auto",
                # Only used when field_order is "auto". idet classifies each
                # analysed frame as TFF/BFF/Progressive/Undetermined; this is
                # the minimum fraction of CLASSIFIED frames (TFF+BFF+
                # Progressive -- Undetermined is excluded from the
                # denominator) that must be interlaced (TFF+BFF) before the
                # stage actually deinterlaces. Below this ratio, the stage
                # SKIPS with the raw counts in the skip reason rather than
                # guessing. 0.7 = a strong majority is required, precisely
                # because idet false-positives on detailed progressive
                # motion (see the "MUST default False" note above).
                "min_interlaced_ratio": 0.7,
                # Analysis pass (idet) is limited to the first N seconds of
                # content -- cheaper decision, matching stages.retime.
                # analysis_sample_sec's convention. 0 = analyse the whole
                # file. The actual deinterlace pass, once triggered, always
                # runs over the WHOLE input regardless of this.
                "analysis_sample_sec": 120,
            },
            "downscale": {
                # Shrinks an oversized input down to the target resolution
                # box BEFORE the heavier deblock/denoise_video/upscale/
                # interpolate stages run, so they never spend compute on
                # pixels that would just be discarded downstream anyway.
                # Off by default -- strictly opt-in (see
                # docs/REQUIREMENTS.md § 7). Traditional/FFmpeg only, no AI
                # path. Uses the same quality.quality_target.
                # resolution_fit_mode/dimension_multiple as upscale (via the
                # shared core.output_check.compute_fitted_dimensions()), so
                # a downscale and a subsequent upscale on the same target
                # box always agree on dimensions.
                "enabled": False,
            },
            "interpolate": {
                "enabled": True,
                "ai_model": "rife_v4.6",
                "traditional_method": "minterpolate",
                "ai_fallback": None,  # see "upscale".ai_fallback above
                # None (default) = auto (traditional minterpolate -- deliberate,
                # fast with decent quality). True/False forces AI/traditional for
                # this stage only. See "upscale".use_ai above for the full
                # precedence; scene mode has its OWN scenes.interpolate.use_ai
                # override (see "scenes" section below) that supersedes this key
                # for the per-scene interpolation path only.
                "use_ai": None,
                # "torch" | "ncnn" -- see "upscale".backend. RIFE's ncnn backend
                # uses the separate "rife-ncnn-vulkan-python" package (the
                # generic ncnn Python bindings lack RIFE's custom rife.Warp
                # layer); requires the "ncnn" extra installed, otherwise falls
                # back per ai_fallback policy.
                "backend": "torch",
                # Traditional (minterpolate) path only -- the AI/RIFE path stays serial
                # (GPU-bound; concurrent GPU jobs contend rather than speed things up).
                # minterpolate is single-threaded per ffmpeg process, so a long clip is
                # split into N time-chunks (1-frame overlap trimmed on concat) and run as
                # N parallel ffmpeg processes, then concatenated back together.
                # 0 = auto (min(os.cpu_count(), 8)); 1 = disable chunking (previous serial
                # behavior, one ffmpeg process for the whole input).
                "parallel_chunks": 0,
                # Minimum chunk length in seconds -- below this, chunking isn't worth the
                # per-chunk ffmpeg startup/concat overhead and the input runs as one chunk.
                "min_chunk_duration_sec": 5.0,
                "temp_crf": 16,  # see "upscale".temp_crf above; only used by the AI/RIFE path
                "read_ahead": 2,  # see "upscale".read_ahead above; AI/RIFE path only
                "write_queue_depth": 4,  # see "upscale".write_queue_depth above; AI/RIFE path only
                # When the AI/RIFE path is used and the target framerate isn't
                # an exact integer multiple of the source, run RIFE at the
                # largest integer factor <= target (never overshoot) then a
                # minterpolate finish pass to reach the exact target, instead
                # of overshooting (e.g. 50->60 forced to factor 2 -> 100fps)
                # and relying on a downstream frame-drop to get back to 60.
                # False restores the pre-hybrid overshoot behavior. See
                # InterpolateStage._plan_ai_interpolation() in
                # core/stages/interpolate.py.
                "hybrid_ai_minterpolate": True,
                # REQUIREMENTS.md § 16 -- per-gap adaptive AI interpolation
                # (timestamp-aware RIFE). Resamples the recovered INPUT
                # timeline (a probe of the interpolate stage's own input
                # file, see core/cadence.py's probe_frame_timestamps())
                # directly onto the target-fps grid instead of applying one
                # global RIFE factor -- long gaps (camera stalls, an
                # irregular-cadence download) get MORE synthesized frames,
                # short gaps get fewer, and grid landings that coincide with
                # a real input frame are emitted verbatim (no inference).
                # This is what makes AI interpolation actually DO something
                # useful on an irregular recovered cadence (a uniform factor
                # collapses to run_rife=False for those inputs -- see § 16's
                # problem statement) and reaches target_fps EXACTLY, so no
                # minterpolate finish pass is needed on this path. Tried
                # FIRST, ahead of the uniform-factor "hybrid_ai_minterpolate"
                # strategy above; fails open (falls back to the
                # uniform-factor path, logged at INFO, job never fails) on a
                # probe failure or a timestamp count that disagrees with the
                # frames the reader actually produces. False always uses the
                # uniform-factor path.
                "adaptive_cadence": True,
                # § 16.7 -- up to this many intermediates in one gap are
                # generated directly from the gap's two real bracketing
                # frames, one RIFE call each at the exact needed timestep.
                # More than this RECURSIVELY SUBDIVIDES instead of a single
                # long-range RIFE call (bad quality at extreme timesteps) or
                # holding (reproduces the very defect this feature exists to
                # fix -- see gap_fallback below): the midpoint is
                # synthesized from the two real frames, then reused as a new
                # reference frame for two half-gap recursive passes, and so
                # on, so every actual RIFE call stays short-range. Depth is
                # ceil(log2(n / direct_synthesis_max)).
                "direct_synthesis_max": 3,
                # § 16.3 SAFETY VALVES for pathological input (e.g. a
                # corrupt timestamp implying an hours-long "gap"), NOT the
                # normal path -- 0 / 0.0 (both default) mean unlimited, so
                # an ordinary long gap always subdivides instead of
                # degrading to gap_fallback. max_intermediates_per_gap caps
                # the intermediate COUNT a single gap may need before
                # gap_fallback applies instead of subdividing;
                # max_gap_sec (seconds) caps the gap's real DURATION the
                # same way. Either firing always overrides subdivision,
                # regardless of gap_fallback's value (see gap_fallback
                # below) -- pair one of these with an explicit
                # gap_fallback: hold to restore the pre-§16.7 default
                # behavior for extreme gaps only.
                "max_intermediates_per_gap": 0,
                "max_gap_sec": 0.0,
                # What to do instead of synthesizing/subdividing, once a
                # safety valve above actually fires for a gap: "subdivide"
                # (default) recursively bisects per § 16.7 -- since a
                # firing valve always overrides subdivision (see above),
                # this default effectively DEGRADES TO "hold" only when a
                # valve fires (subdividing is not a meaningful response to
                # a gap just flagged as pathological); set explicitly to
                # "hold" (repeat the frame before the gap, exactly what the
                # source did) or "blend" (linear crossfade across the gap)
                # to choose which. NOTE: with both safety valves left at
                # their unlimited defaults, gap_fallback is never consulted
                # at all -- every gap subdivides regardless of this value.
                "gap_fallback": "subdivide",
                # Governs the UNIFORM-FACTOR AI path only (REQUIREMENTS.md
                # § 16.4) -- irrelevant when adaptive_cadence reaches the
                # target exactly, and only consulted with
                # hybrid_ai_minterpolate: true (hybrid disabled keeps the
                # pre-existing forced-factor-2 overshoot regardless of this
                # key). "under" (default -- bit-for-bit unchanged existing
                # behavior): largest integer RIFE factor at or below target,
                # then a minterpolate finish pass for the fractional
                # remainder. "nearest": whichever integer factor (over or
                # under) lands closest to target. "over": smallest integer
                # factor at or above target, then minterpolate finishes DOWN
                # to the exact rate.
                "target_approach": "under",
            },
            "denoise_video": {
                # Off by default: deblock's default ai_model
                # (realesr-general-wdn-x4v3) is trained on real-world
                # degradation including noise, not just blocking, so its
                # one early AI pass now covers denoising too and the other
                # default stages don't introduce new noise. Stays in
                # pipeline.default_order (omission != disable) so users who
                # want a separate denoise pass -- or who set a different,
                # denoise-focused deblock model -- can just flip this back
                # on (or use a preset like max_quality that already does).
                "enabled": False,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "hqdn3d",
                "tile_size": 0,  # see "upscale".tile_size above
                "batch_size": 1,  # see "upscale".batch_size above
                "tile_batch_size": 1,  # see "upscale".tile_batch_size above
                "ai_fallback": None,  # see "upscale".ai_fallback above
                # None (default) = auto (traditional hqdn3d -- deliberate,
                # denoising benefits less from Real-ESRGAN than deblocking does
                # and hqdn3d needs no GPU/model). True/False forces AI/traditional
                # for this stage only. See "upscale".use_ai above for the full
                # precedence.
                "use_ai": None,
                "temp_crf": 16,  # see "upscale".temp_crf above
                "read_ahead": 2,  # see "upscale".read_ahead above
                "write_queue_depth": 4,  # see "upscale".write_queue_depth above
            },
            "denoise_audio": {
                "enabled": True,
                "ai_model": "demucs",
                "traditional_method": "afftdn",
            },
            "deblock": {
                "enabled": True,
                "strength": "medium",  # low, medium, high
                # Compact SRVGG general-v3 checkpoint trained on real-world
                # degradation including compression/blocking artifacts AND
                # noise -- ~3x faster than the RRDB RealESRGAN_x4plus model
                # and doubles as a denoise pass (see stages.denoise_video's
                # "enabled": False comment). Set back to "RealESRGAN_x4plus"
                # for the old higher-fidelity RRDB behavior (max_quality
                # preset does this).
                "ai_model": "realesr-general-wdn-x4v3",
                "tile_size": 0,  # see "upscale".tile_size above
                "batch_size": 1,  # see "upscale".batch_size above
                "tile_batch_size": 1,  # see "upscale".tile_batch_size above
                "ai_fallback": None,  # see "upscale".ai_fallback above
                # None (default) = auto (AI -- deliberate, better quality than
                # the traditional unsharp filter). True/False forces
                # AI/traditional for this stage only. See "upscale".use_ai
                # above for the full precedence.
                "use_ai": None,
                "temp_crf": 16,  # see "upscale".temp_crf above
                "read_ahead": 2,  # see "upscale".read_ahead above
                "write_queue_depth": 4,  # see "upscale".write_queue_depth above
            },
            "stabilize": {
                "enabled": True,
                "threshold": 2.0,  # stabilize if shake > this value
                "smoothness": 40,  # frames for lowpass filtering (higher = smoother)
                "maxshift": 20,  # max pixels to shift per frame (limits overcorrection)
                "optalgo": "gauss",  # optimization algorithm (opt, gauss, avg)
                "shakiness": 10,  # motion detection sensitivity (1-10, higher = more sensitive)
                "zoom_enabled": True,  # auto zoom-out for very shaky video
                "zoom_threshold": 50.0,  # min movement (px) to trigger zoom
                "zoom_mode": "black",  # black or keep
                "sharpen_enabled": True,  # auto sharpen after stabilization
                # Post-stabilize unsharp filter tuning (only applied when
                # sharpen_enabled and stabilization actually triggered). See
                # StabilizeStage._build_sharpen_suffix() for validation.
                "sharpen_amount": 1.0,  # unsharp luma_amount, float in [-2.0, 5.0]
                "sharpen_luma_size": 3,  # unsharp luma_msize_x/y, odd int in [3, 63]
                "sharpen_chroma_amount": 0.0,  # unsharp chroma_amount, float in [-2.0, 5.0]
                "sharpen_chroma_size": 3,  # unsharp chroma_msize_x/y, odd int in [3, 63]
                # SIGNED zoom dial (0.0-1.0) once zoom_enabled's gate decides zoom
                # applies at all -- REQUIREMENTS.md § 17. BREAKING (2026-09): this
                # used to mean "fraction of frames border-free" with 0.0 = "no zoom"
                # (which was NOT "no cropping" -- a bug, see below); it is now a
                # signed dial spanning zoom-OUT (preserve content) to zoom-IN
                # (eliminate borders):
                #   0.00 -> zoom=-max(B)  : every frame's content fully preserved,
                #                           borders on all frames, nothing ever cropped
                #   0.25 -> zoom=-p50(B)  : ~50% of frames fully preserved
                #   0.50 -> zoom=0        : no zoom at all (vidstabtransform's own
                #                           default -- this is the OLD 0.0 behavior;
                #                           anyone who set 0.0 to mean "leave it
                #                           alone" must change it to 0.5)
                #   0.75 -> zoom=+p50(B)  : ~50% of frames border-free
                #   1.00 -> zoom=+max(B)  : every frame border-free (today's/old 1.0
                #                           behavior, delegated to vidstabtransform's
                #                           own optzoom=1 -- bit-for-bit unchanged,
                #                           exact, no approximation)
                # B_i is the per-frame required-zoom-IN estimate (see
                # StabilizeStage._compute_required_zoom_percentages);
                # -B_i is the mirror-image zoom-OUT needed to keep that frame's
                # content fully intact. See StabilizeStage._signed_zoom_from_coverage
                # for the exact q -> signed-zoom formula, and AGENTS.md's stabilize
                # zoom section for the accuracy caveat (both halves approximate
                # vidstabtransform's own smoothed camera path from raw per-frame
                # local-motion values; only 1.0 is exact).
                "zoom_coverage": 1.0,
            },
            "normalize_volume": {
                "enabled": True,
                "target_db": -23.0,  # EBU R128
                "true_peak_db": -2.0,
                # Silence-skip: below this measured integrated loudness (LUFS),
                # normalize_volume/normalize_audio skip normalization entirely
                # (pass the input through unchanged) instead of feeding loudnorm's
                # second pass an unusable -inf/near-inf gain. -80.0 dBFS sits just
                # above 2-3 LSBs of 16-bit dither noise (20*log10(3/32768) ~=
                # -80.8 dBFS) -- see normalize_audio.py's DEFAULT_SILENCE_THRESHOLD_DB.
                "silence_threshold_db": -80.0,
            },
            "normalize_audio": {
                "enabled": True,
                "target_db": -23.0,  # EBU R128
                "true_peak_db": -2.0,
                # See normalize_volume.silence_threshold_db above -- same key,
                # same default, own config section (normalize_audio and
                # normalize_volume are two separately-addressable stage names
                # running the identical loudnorm algorithm).
                "silence_threshold_db": -80.0,
            },
            "hdr_to_sdr": {
                "enabled": False,
                "method": "bt2020",
            },
            "speed": {
                "enabled": False,
                "factor": 1.0,
                # Which audio speed-change algorithm to use -- one of
                # VALID_AUDIO_SPEED_METHODS above. "atempo" (default): the
                # classic pitch-preserving time-stretch, chained
                # (SpeedStage._build_atempo_chain) for factors outside a
                # single atempo's [0.5, 100] range -- e.g. a 0.25x (4x
                # slow-motion) chain is three stacked atempo=0.5 stages,
                # which has real, audible quality loss (robotic timbre,
                # artifacting) at that extreme. Behavior here is unchanged
                # from before this key existed. "rubberband": ffmpeg's
                # `rubberband` filter (needs an ffmpeg build with
                # --enable-librubberband) -- also pitch-preserving, but one
                # filter covers the entire factor range with none of
                # atempo's chaining artifacts, so it's the better choice at
                # extreme factors when librubberband is available. Falls
                # back to "atempo" (WARNING logged) if this ffmpeg build
                # lacks librubberband -- never fails the stage over it.
                # "asetrate": resample-based and DELIBERATELY
                # pitch-changing -- relabels the audio at a new sample rate
                # (input sample rate * factor) instead of time-stretching
                # it, so slowing down also drops the pitch and speeding up
                # raises it. This isn't a lesser fallback: it's the CORRECT
                # choice for phone slow-motion footage, which is captured at
                # an elevated mic sample rate and mapped down to normal
                # speed in the container -- speeding that audio back up
                # with asetrate restores the mic's true original pitch,
                # which a pitch-preserving method (atempo/rubberband) would
                # leave artificially low/wrong. Needs the INPUT stream's
                # real audio sample rate, probed automatically
                # (ffmpeg_utils.probe/StreamInfo.sample_rate); falls back to
                # "atempo" (WARNING logged) if that can't be determined (no
                # audio stream, or the probe fails).
                "audio_method": "atempo",
                # Output audio sample rate in Hz, applied via `-ar` on the
                # muxed output regardless of audio_method. 0 = leave the
                # input's own sample rate alone (no `-ar` emitted). Also
                # used as the "asetrate" method's `aresample` target (the
                # step that renormalizes the sample rate asetrate itself
                # relabeled) -- when this is 0, that renormalization targets
                # the ORIGINAL input rate instead.
                "audio_sample_rate": 48000,
                # Resampler used by the "asetrate" method's `aresample` step
                # (ignored by "atempo"/"rubberband", which don't resample) --
                # one of VALID_AUDIO_RESAMPLERS above. "soxr" (default): the
                # libsoxr resampler, higher quality than ffmpeg's built-in
                # "swr" (swresample), at extra CPU cost. Requires an ffmpeg
                # build with --enable-libsoxr; ffmpeg itself falls back to
                # swr with its own warning if soxr isn't compiled in (this
                # stage does not additionally detect/fall back for this
                # key).
                "resampler": "soxr",
            },
            "crop": {
                # Auto-crop: detect and remove black borders (letterbox/pillarbox),
                # including residual borders stabilize can introduce. On by
                # default since REQUIREMENTS.md § 22.1 (it was opt-in before):
                # should_run() SKIPs cheaply (one cropdetect sample, no
                # re-encode) on a borderless input, and cropping first saves
                # every downstream stage compute. Existing user configs that
                # spell out `enabled: false` keep it until the P3 config
                # migration (§ 24) flips it. See AGENTS.md "Auto-crop".
                "enabled": True,
                # cropdetect luma threshold (0-255, ffmpeg default is 24) -- pixels
                # darker than this are treated as "black" border.
                "limit": 24,
                # cropdetect round=2 keeps cropped width/height even (H.264
                # requirement), matching UpscaleStage._round_to_even's convention.
                "round": 2,
                # Skip cropping entirely if the detected crop would save fewer than
                # this many pixels in BOTH width and height -- avoids a pointless
                # 2px crop from encoder rounding noise.
                "min_crop_px": 8,
                # 0 = scan the whole video. >0 = seconds to sample from the start
                # of the video instead, for very long inputs where a full scan is
                # too slow. The real (execute()) detection pass runs cropdetect
                # with reset=1 (one crop=w:h:x:y line per analyzed frame) and
                # aggregates those per-frame windows with transition-exclusion
                # logic -- see aggregate_crop_windows() in core/stages/crop.py and
                # AGENTS.md's Auto-crop section. should_run()'s cheap 10s prefilter
                # sample keeps the old single-pass reset=0 union behavior (it only
                # decides "worth attempting", not the actual crop window, so it
                # doesn't need max_outliers/aggregation precision).
                "analyze_duration_sec": 0,
                # cropdetect's max_outliers option (int, ffmpeg default 0): a
                # border line may contain up to N pixels above `limit` and still
                # count as black/border. Without this, a single logo/overlay
                # sitting in the letterbox area breaks the all-pixels-black line
                # test and permanently widens the detected crop to include it.
                # We expose a RATIO (not a raw pixel count) because an overlay
                # typically covers a minority of the border line it sits on
                # regardless of resolution, and a ratio scales sanely across
                # resolutions where a fixed pixel count wouldn't. At detection
                # time this is converted to max_outliers = round(max_outlier_ratio
                # * min(width, height)) -- min(w, h) is used (rather than the
                # dimension of each scanned axis individually) because cropdetect
                # applies a single absolute max_outliers count to both the
                # row-scan (height-driven) and column-scan (width-driven) axes, so
                # basing it on the smaller dimension keeps the tolerance
                # conservative on both axes. Range 0..0.5; 0 disables (strict
                # cropdetect behavior, matching pre-max_outliers versions of this
                # stage). Live-validated against ffmpeg: on a 1920x1080 clip with
                # 1920x800 centered content (140px black bars) and a bright
                # ~200x60 logo drawn in the bottom bar, max_outliers=0 detected
                # crop=1920:920:0:140 (the logo widened the window); with
                # max_outliers=round(0.2*1080)=216, cropdetect correctly returned
                # crop=1920:800:0:140 (the true content box).
                "max_outlier_ratio": 0.2,
                # Transition-exclusion tuning for the per-frame aggregation that
                # replaces the old reset=0 "can only grow" union (which let a
                # single bright full-frame transition permanently widen the crop
                # to the full frame). A run of consecutive similar per-frame
                # windows lasting <= transition_max_run_sec, with no similar
                # window recurring within transition_window_sec before/after it,
                # is treated as a transition and excluded from the final union.
                # The same window recurring nearby (or lasting longer than
                # transition_max_run_sec even in isolation) is kept -- that's real
                # content geometry (e.g. a moving logo/letterbox change), not a
                # transient flash. See aggregate_crop_windows() in
                # core/stages/crop.py.
                "transition_max_run_sec": 2.0,
                "transition_window_sec": 4.0,
                # Per-edge (left/top/right/bottom) pixel tolerance for treating two
                # per-frame crop windows as "the same" window when grouping frames
                # into runs and comparing runs to each other.
                "transition_tolerance_px": 16,
                # Optional VLM-assisted disambiguation: a watermark/logo sitting
                # outside the true content area (e.g. positioned relative to a
                # letterboxed frame) can fool naive cropdetect into "protecting" it
                # as non-black content. When enabled AND analysis.vlm.enabled is
                # true, one frame is rendered twice (plain + the proposed crop box
                # drawn via drawbox) and sent to the VLM with a narrow fixed
                # question. Off by default; fails open (WARNING + proceed with the
                # plain cropdetect result) on any VLM error. The original intent
                # here was for the VLM to SHRINK the detected window to exclude a
                # non-content overlay sitting in the border area; the
                # outlier-tolerant detector above now handles that numerically, so
                # this check remains an optional safety verification layer (warn/
                # skip policies below), not the primary defense against overlays.
                "vlm_check": False,
                # "warn" (default): log the VLM's objection but still crop.
                # "skip": don't crop this video at all if the VLM flags content
                # outside the box. "expand" (growing the crop box to include the
                # flagged region) was considered and rejected -- VLMs don't return
                # reliable pixel coordinates, so there's nothing to expand *to*.
                "vlm_policy": "warn",
                # Which per-frame border detector execute() uses. "cropdetect"
                # (the FFmpeg filter above) is luma-threshold-only -- white/gray/
                # colored borders and any non-dark padding are invisible to it.
                # "rust" is the avf_borders extension (see rust/avf_borders/ and
                # AGENTS.md's Auto-crop section): per edge, per sampled frame, it
                # detects the DOMINANT color of the outer border strip (any
                # color, not just black) plus a "solidity" percentage, then walks
                # inward while a majority of each line still matches -- both
                # signals (color + consistency) are logged at INFO. "auto"
                # (default) uses "rust" when the avf_borders extension is
                # importable, else falls back to "cropdetect" (logged at DEBUG,
                # matching the avf_scenes/avf_hashing fallback convention). The
                # rust detector's per-frame windows feed the SAME
                # aggregate_crop_windows() aggregation as cropdetect -- only the
                # per-frame detection step changes.
                "detector": "auto",
                # avf_borders: max per-channel absolute color difference (0-255)
                # for two colors to "match" -- used both for a border strip's
                # solidity and for the inward line-by-line walk.
                "border_tolerance": 24,
                # avf_borders: fraction of a line's pixels that must match the
                # border's dominant color for the inward walk to continue past
                # it. Lets a logo/overlay occupying a MINORITY of a border line
                # (e.g. a small watermark sitting in the letterbox area) pass
                # through without stopping the walk early. 0.80 tolerates
                # overlays covering up to 20% of a line -- deliberately
                # consistent with max_outlier_ratio's 0.2 default so the rust
                # and cropdetect detectors agree on the same borderline logo.
                "border_majority": 0.80,
                # avf_borders: minimum fraction of an edge's outer strip that
                # must match its own dominant color for that edge to be treated
                # as having a solid border at all (below this, border_px = 0 for
                # that edge). Protects blurred-video-background pillarboxing (a
                # real but non-uniform edge) from being cropped away in v1 --
                # see AGENTS.md's Auto-crop section.
                "border_solidity_min": 0.60,
                # avf_borders: depth (px) of the outer strip sampled per edge to
                # compute the dominant color + solidity signal, before the
                # line-by-line inward walk begins.
                "border_strip_px": 4,
                # avf_borders: frames-per-second to sample at (0 = every decoded
                # frame, matching cropdetect's own reset=1 per-frame behavior
                # modulo its default skip=2). >0 inserts an ffmpeg `fps=` filter
                # before detection, trading temporal resolution for speed on
                # long inputs.
                "sample_fps": 0,
            },
        },
        "scenes": {
            # Master switch for scene-based processing (see docs/REQUIREMENTS.md features
            # 1-2, AGENTS.md "Scene mode"). Off by default -- zero behavior change to the
            # existing whole-video pipeline when disabled. When enabled, stabilize and
            # interpolate (if requested for the job) run per-scene instead of on the whole
            # file, then scenes are concatenated back together before the remaining
            # whole-video stages (upscale/denoise/deblock/normalize/encode) run.
            "enabled": False,
            # Reuses analysis.event_detection.scene_change_threshold/min_scene_duration_sec
            # for boundary detection -- no separate scene-mode threshold.
            "drop_non_content": False,  # see analysis.llm below; requires VLM + coordinator
            # Re-encode codec/crf used for the per-scene split/concat intermediate passes
            # (segments get re-encoded again by the final whole-video "encode" stage, so
            # this only needs to be visually lossless-ish, not archival quality).
            "intermediate_crf": 14,
            "intermediate_preset": "veryfast",
            # None (default) = auto: cpu-and-resolution-based heuristic (see
            # core/scenes.py's _scene_worker_budget) picks how many scenes
            # process concurrently. A positive int caps scene_workers
            # explicitly (the per-scene traditional-interpolation chunk
            # split still derives from the same underlying total budget --
            # this only bounds the outer scene-level pool).
            "max_workers": None,
            "interpolate": {
                # None (default) = resolve the AI/traditional method for
                # scene-mode interpolation with the EXACT same precedence as
                # the standard "interpolate" stage (stages.interpolate.use_ai
                # / general.use_ai / the stage's traditional auto default),
                # so both paths respond together to the same config. True/
                # False forces AI/traditional for the scene-mode path ONLY,
                # without touching whole-video interpolate behavior. See
                # AGENTS.md's "Scene mode" section.
                "use_ai": None,
            },
            "stabilize": {
                # Whether per-scene stabilize-strength tiering runs at all when scene mode
                # is active and the "stabilize" stage was requested for the job. If false,
                # stabilize still runs per-scene (never across a cut) but with the stage's
                # normal (non-tiered) config for every scene.
                "enabled": True,
                # avg_shake (px, from StabilizeStage's own TRF analysis) above which a
                # scene is escalated from "normal" to "aggressive" tier (re-run with
                # smoothness multiplied by aggressive_smoothness_multiplier). Below the
                # stage's own stabilize.threshold, a scene is already skipped entirely
                # ("skip" tier) by StabilizeStage's existing needs_stab logic.
                "aggressive_shake_threshold": 8.0,
                "aggressive_smoothness_multiplier": 2.0,
            },
        },
        # Network/TLS policy for EVERY outbound HTTPS request (VLM/LLM providers, model and
        # archive downloads) -- docs/REQUIREMENTS.md § 23, autovideofixer/net.py. Exists
        # because corporate TLS-inspection firewalls re-sign traffic with a private CA that
        # Python does not trust, and self-hosted inference servers often use self-signed certs;
        # before this, both failed outright with no override.
        "network": {
            "tls": {
                # verify       -- normal verification (system trust store + ca_files). Default.
                # untrusted_ca -- accept a chain that does not verify (self-signed / unknown CA)
                #                 but STILL reject an expired / not-yet-valid certificate and a
                #                 hostname mismatch. Convenient for a home server; weaker than
                #                 verify because any self-made cert for the right name is
                #                 accepted (a man-in-the-middle could present one).
                # insecure     -- no verification at all (still encrypted). Last resort.
                # Prefer `verify` + ca_files whenever you can get hold of the certificate.
                "mode": "verify",
                # Extra PEM files trusted IN ADDITION to the system store (never replacing it)
                # in `verify` mode. Each may be a CA certificate (corporate TLS inspection) or
                # a self-signed server certificate (home server) -- one key covers both.
                # Ignored by the other modes.
                "ca_files": [],
            },
            # Optional override of `tls` for MODEL DOWNLOADS only (same shape), e.g. relax TLS
            # for the corporate proxy that intercepts huggingface/github while keeping your
            # VLM endpoint strict. null = downloads use `tls`. Downloaded models are still
            # SHA256-verified against the registry in every mode, so even `insecure` cannot
            # substitute tampered weights for registry models.
            "download_tls": None,
        },
        "analysis": {
            # Named, fully independent model connections (REQUIREMENTS.md § 24.4). Every
            # consumer (whole-video + per-scene VLM, crop VLM check, scene coordinator)
            # resolves one by name via endpoints.resolve_endpoint(). Add as many as you like
            # and select them with analysis.vlm.endpoint / analysis.llm.endpoint (or
            # `avf analyze --vlm-endpoint NAME`).
            "endpoints": {
                "default_vlm": {
                    "provider": "ollama",  # ollama | openai | api | llamacpp
                    "model": "llava",
                    "api_url": "",
                    "api_key": "",
                    # Plain-HTTP api_url endpoints are refused for non-loopback hosts unless
                    # this is true (frames + api_key travel unencrypted; only enable for a
                    # server on a network you control, e.g. a LAN inference box).
                    "allow_http": False,
                    # Request timeout in seconds (resolve_timeout() semantics: null/0 =
                    # unlimited). Slow local models on big prompts need far more than 120.
                    "timeout_sec": 120,
                    # Response token budget (max_tokens / Ollama num_predict) per call.
                    # Raise for reasoning VLMs whose thinking tokens count against it.
                    "max_tokens": 1024,
                    # Overrides network.tls for THIS endpoint only (same shape; § 23).
                    "tls": {},
                    # Request-shape knobs for OpenAI-shaped providers (openai/api/llamacpp):
                    # system_content (string|parts), response_format
                    # (json_object|json_schema|none), extra_body (deep-merged into payload).
                    "compat": {},
                    # Model image limits (P4, § 25.3); {} = none.
                    "vision": {},
                },
                # Coordinating text-LLM used only when scenes.drop_non_content is true.
                # Reviews ALL per-scene VLM summaries together and returns which scene
                # indices (if any) are not part of the video's main content and should be
                # dropped. Can be a cheaper/faster text-only model than the VLM.
                "default_llm": {
                    "provider": "ollama",
                    "model": "llama3",
                    "api_url": "",
                    "api_key": "",
                    "allow_http": False,
                    "timeout_sec": 120,
                    # Reasoning models spend "thinking" tokens from this same budget before
                    # emitting their JSON -- keep it generous or the response is truncated
                    # mid-thought (which fails open: no scenes dropped).
                    "max_tokens": 4096,
                    "tls": {},
                    "compat": {},
                    "vision": {},
                },
            },
            "llm": {"endpoint": "default_llm"},
            # Print a per-call table (time, tokens, tokens/s, images, status) and run totals
            # in `avf analyze`. The same figures are always logged (INFO per call) and
            # always written by `avf analyze --json` / the process JSON report; this only
            # controls the console table (§ 26.2).
            "report_call_stats": True,
            "vlm": {
                "enabled": False,
                "endpoint": "default_vlm",
                # Which structured fields the default prompt asks the model for, and the
                # json_schema response_format is built from (§ 26.1). summary/tags/objects/
                # rating are always requested. rating_reasoning is emitted LAST, after
                # rating, so a non-thinking model has seen the content before it justifies.
                "outputs": {"media_type": True, "rating_reasoning": False},
                # Allowed media_type values; a reply outside the list is stored as "other"
                # (the original wording is kept in vlm_extra.media_type_raw).
                "media_type_choices": [
                    "real_footage",
                    "3d_rendered",
                    "2d_digital",
                    "hand_drawn",
                    "ai_generated",
                    "screen_capture",
                    "mixed",
                    "other",
                ],
                # How much the default prompt asks the summary to cover:
                # brief | standard | detailed | exhaustive. Higher levels cost output tokens
                # and time -- raise the endpoint's max_tokens with them.
                "detail": "brief",
                # Frame sampling for every VLM call (whole video and per scene);
                # REQUIREMENTS.md § 25.1/25.2. Validated at use time
                # (vlm_sampling.SamplingConfig.from_mapping).
                "sampling": {
                    # Floor / ceiling on frames per call. The count is ceil(duration /
                    # interval_sec) clamped to [min_frames, max_frames], then to the
                    # endpoint's vision.max_frames_per_call and to the number of real
                    # frames (a 3-frame clip sends 3, never padded duplicates).
                    "min_frames": 4,
                    "max_frames": 8,
                    "interval_sec": 10.0,
                    # Drop a sampled frame that matches the previous KEPT one (VFR /
                    # retimed / static content) and refill from the widest time gap.
                    "dedupe_consecutive": True,
                    # Mean abs luma difference on a 64-px grayscale thumbnail (0-1);
                    # at or below this = duplicate. See § 25.5 for the calibration.
                    "dedupe_threshold": 0.002,
                    # Frames-vs-resolution trade under vision.max_tiles_per_call:
                    # auto | temporal | spatial | fixed.
                    "balance": "auto",
                    # Resolution is only reduced below this long side to keep
                    # min_frames; absolute floor 64 px.
                    "min_long_side_px": 336,
                },
                # Extra text appended to the default (or overridden) user prompt when
                # non-empty, e.g. job-specific context like "these are wildlife clips
                # from a trail camera". Also settable per-run via `avf analyze
                # --prompt-append`.
                "prompt_append": "",
                # Replaces the default user prompt entirely when non-empty.
                # `prompt_append` (if set) is still appended after this. Also settable
                # per-run via `avf analyze --prompt-override`. WARNING: the default
                # prompt instructs the model to return JSON with summary/tags/objects/
                # rating (+ media_type / rating_reasoning when enabled in `outputs`) -- an
                # override REPLACES the generated prompt, so it must ask for every field
                # itself; one that changes the output format breaks the JSON parsing (see
                # _parse_vlm_response's fallback below).
                "prompt_override": "",
                # Replaces the default system prompt entirely when non-empty. No CLI
                # flag; config-only.
                "system_prompt_override": "",
            },
            # Multi-VLM cross-check (REQUIREMENTS.md § 27): several VLM endpoints look at the
            # same frames (each at its own vision limits) and their answers are combined.
            # Applies to whole-video and per-scene analysis (not the crop VLM check). Off by
            # default: costs one extra model call per listed endpoint (plus judge/arbiter).
            # `avf analyze --vlm-endpoint/--vlm-model/--vlm-url` replace the FIRST (primary)
            # endpoint only. Validated at use time (core/cross_check.py).
            "cross_check": {
                "enabled": False,
                # refine: each next endpoint verifies/corrects the previous one's JSON (the last
                #   successful output wins; sequential).
                # judge: every endpoint describes independently, then a text LLM compares them
                #   and merges (see judge_endpoint / agreement_threshold / arbiter_endpoint).
                "mode": "refine",
                # Ordered VLM endpoint names (analysis.endpoints.<name>); the first is the
                # primary. At least two when enabled.
                "endpoints": [],
                # judge mode: the text LLM that compares the outputs. null = the endpoint
                # analysis.llm.endpoint selects.
                "judge_endpoint": None,
                # judge mode: optional VLM (typically a larger one) that sees the frames, all
                # descriptions and the conflicts and writes the final when agreement is below
                # the threshold. null = keep the primary's output, flagged `disagreement`.
                "arbiter_endpoint": None,
                # judge mode: agreement (0-1, as scored by the judge) at or above this accepts
                # the judge's merged output.
                "agreement_threshold": 0.75,
                # judge mode: describe legs that may run at once. 1 = one at a time (a single
                # local GPU serializes requests anyway); refine is always sequential.
                "max_parallel": 1,
            },
            "event_detection": {
                "enabled": True,
                # Frame-differencing sensitivity (mean fractional luma change between
                # consecutive downscaled frames, 0-1; see _detect_scene_changes()'s
                # docstring in core/analysis.py for the full metric writeup and
                # calibration data). 0.3 (the old default) under-detected real cuts;
                # 0.15 was calibrated against a synthetic ground-truth clip to catch
                # every hard cut with zero false positives. Also settable per-run via
                # `avf analyze --scene-threshold`.
                "scene_change_threshold": 0.15,
                # Cuts closer together than this are merged into the following scene
                # rather than starting a new one -- doesn't affect cut *detection*,
                # only whether a short segment gets its own SceneEvent. Also settable
                # per-run via `avf analyze --min-scene-duration`.
                "min_scene_duration_sec": 2.0,
                "classify_events": False,
            },
            "duplicate_detection": {
                "enabled": True,
                "similarity_threshold": 0.85,
                # `hash_type` was removed (R5.2): the old ahash/dhash/combined choice
                # no longer applies now that compute_video_phash() always uses a
                # single pHash (DCT-based) algorithm -- see core/analysis.py and
                # rust/avf_hashing/ for why pHash replaced both.
            },
        },
        # --- REQUIREMENTS.md § 6.4/6.5/6.6: per-video stage/mode summary,
        # media info + timing instrumentation, structured JSON run report ---
        "reporting": {
            # Three independent stage-timing console/log views (core/reporting.py's
            # aggregate_stage_timing()), all OFF by default -- the per-job
            # classification table (§ 6.4) and per-stage duration logging (§ 6.5)
            # always happen regardless of these flags; these three only add extra
            # cross-video views for correlating stage cost with video characteristics.
            "stage_timing_per_video": False,  # per-video per-stage duration table
            "stage_timing_totals": False,  # per-stage totals summed across the run
            "stage_timing_averages": False,  # per-stage average per video that ran it
            # Optional path to write a single § 6.6 structured JSON run report to
            # (one document per run, written once at the end -- also on partial
            # failure). null (default) = don't write one. See --report-json.
            "report_json": None,
            # Feature 6: live nested progress bars (rich.progress) during `avf
            # process` -- a BATCH bar (jobs completed + current job's fraction)
            # and a PER-FILE bar (current job's own 0..1 progress). Each is
            # independently enabled here but only ever actually shown when the
            # console is attached to a real terminal (see
            # cli/progress.py:resolve_show_progress) -- piped/redirected/non-TTY
            # output silently gets no bars, same as before this feature existed.
            # See --progress-batch/--no-progress-batch and
            # --progress-file/--no-progress-file.
            "progress_batch": True,
            "progress_file": True,
            # Tri-state override for Rich console colour on the CLI's stdout/
            # stderr consoles AND the logging RichHandler's console (see
            # logger.py:build_console/set_console_color). "auto" (default)
            # keeps today's behavior -- Rich autodetects a real terminal and
            # enables colour only then. The motivating case for the other two
            # is `avf process ... | tee run.log`: piping stdout through `tee`
            # makes it a pipe, not a TTY, so Rich silently drops to plain text
            # even though the user is still watching a real terminal (just
            # downstream of `tee`) -- "always" forces colour back on
            # (force_terminal=True) for that case. "never" is the inverse,
            # forcing colour off (no_color=True) even when attached to a real
            # terminal (e.g. a terminal that mishandles ANSI, or scripting
            # `avf` interactively but wanting clean output to eyeball). See
            # --color/--no-color on `avf process` (CLI flag wins over this
            # config value; absent flag leaves this value in effect).
            # Deliberately independent of the progress bars above: those stay
            # gated on the real fd-level `sys.stdout.isatty()` check
            # regardless of this setting (see cli.py's `process` command) --
            # forcing colour on for a `tee`d run must NOT re-enable bars,
            # since their redraw control codes would still corrupt the piped
            # file. Also never affects file logs (auto_log_file/--log-file),
            # which stay plain-text always -- only the two Rich consoles are
            # ever touched. Must be one of VALID_COLOR_MODES below.
            "color": "auto",
        },
    }

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        config_path: str | Path | None = None,
        require_exists: bool = False,
        defer_notices: bool = False,
    ):
        """Create a Config.

        Args:
            path: Explicit config file path (positional, pre-existing API).
            config_path: Same as `path`, accepted as a keyword alias so
                callers that pass an explicit path (e.g. the CLI's
                `--config`/`AVF_CONFIG`) can be self-documenting about intent.
                `path` wins if both are given.
            require_exists: If True, the resolved path must already exist on
                disk -- raises FileNotFoundError instead of silently falling
                back to DEFAULTS. Used for an explicitly-requested config
                path (CLI flag / env var), where a typo'd path should be a
                hard error, not a silent no-op. The default platform config
                path (neither `path` nor `config_path` given) is never
                required to exist.
            defer_notices: Load-time events (first-run bootstrap, schema migration
                warnings -- REQUIREMENTS.md § 24) are normally logged at the end of
                construction. The CLI constructs Config before logging is set up, so it
                passes True and calls `emit_notices()` once the handlers (and the PII
                cleaner's registrations) exist.
        """
        resolved = path or config_path
        self._path = Path(resolved) if resolved is not None else get_config_path()
        self._explicit = resolved is not None
        self._require_exists = require_exists and resolved is not None
        self._notices: list[tuple[int, str]] = []
        if self._require_exists and not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        self._data = self._merge()
        self._save_pending = False
        # Ordered record of every layer folded into `self._data` so far, for
        # DEBUG logging / diagnostics -- see `apply_layer()`. The base two
        # layers (DEFAULTS, and the user config.yaml if it was found on disk)
        # are recorded here; every subsequent cascade layer (--preset,
        # --config, --set, other CLI flags) is appended via apply_layer().
        self._sources: list[str] = ["defaults"]
        if self._path.exists():
            self._sources.append(f"user-config:{self._path}")
        if not defer_notices:
            self.emit_notices()

    # ---- load-time events (bootstrap / migration), REQUIREMENTS.md § 24 ----

    def _notice(self, level: int, message: str) -> None:
        self._notices.append((level, message))

    def emit_notices(self) -> None:
        """Log (and clear) the events recorded while loading the config file."""
        from autovideofixer.logger import get_logger

        log = get_logger("autovideofixer.config")
        pending, self._notices = self._notices, []
        for level, message in pending:
            log.log(level, "%s", message)

    @property
    def notices(self) -> list[tuple[int, str]]:
        """Events recorded at load time and not yet emitted (see `emit_notices`)."""
        return list(self._notices)

    def _bootstrap_default_config(self) -> None:
        """First run: copy the packaged example to the default path (§ 24.3).

        Never raises: a read-only filesystem etc. is a WARNING and the run continues on
        DEFAULTS. Opt out with ``AVF_NO_CONFIG_BOOTSTRAP=1``.
        """
        if os.environ.get(BOOTSTRAP_OPT_OUT_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
            return
        try:
            from importlib import resources

            text = (
                resources.files("autovideofixer").joinpath("data/config.example.yaml").read_text()
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create so two processes starting together cannot clobber each
            # other (or a file that appeared since the exists() check).
            with open(self._path, "x", encoding="utf-8") as f:
                f.write(text)
        except FileExistsError:
            return
        except Exception as e:
            self._notice(
                logging.WARNING,
                f"Could not create default config at {self._path}: {e}. "
                "Continuing with built-in defaults.",
            )
            return
        self._notice(
            logging.INFO,
            f"First run: created a default config at {self._path} (a commented copy of the "
            f"packaged example). Edit it to customise AVF; set {BOOTSTRAP_OPT_OUT_ENV}=1 to "
            "disable this.",
        )

    def _migrate_user_config(self, text: str, user: dict[str, Any]) -> dict[str, Any]:
        """Apply schema migrations to a freshly loaded user config (§ 24.1).

        Default-path file older than current: timestamped backup, migrate, rewrite in
        place. Explicit file (--config / AVF_CONFIG): in memory only. Newer than known:
        warn, load as-is. An empty file has nothing to migrate and is left alone.
        """
        from autovideofixer.config_migrations import (
            get_config_version,
            migrate,
            migrate_text,
            timestamped_backup_path,
        )

        if not user:
            return user
        version = get_config_version(user)
        if version > CURRENT_CONFIG_VERSION:
            self._notice(
                logging.WARNING,
                f"Config {self._path} has config_version {version}, newer than this AVF "
                f"understands ({CURRENT_CONFIG_VERSION}); loading it as-is (best effort).",
            )
            return user
        if version == CURRENT_CONFIG_VERSION:
            return user

        in_memory = copy.deepcopy(user)
        notes = migrate(in_memory)
        listing = "".join(f"\n  - {n}" for n in notes) or "\n  - (version stamp only)"

        if not self._explicit:
            try:
                new_text, old_version, _ = migrate_text(text)
                rewritten = yaml.safe_load(new_text) or {}
                if rewritten != in_memory:
                    # The comment-preserving rewrite must agree with the plain-dict
                    # migration; if ruamel round-tripped something unexpectedly, do NOT
                    # touch the user's file.
                    raise RuntimeError("round-trip rewrite disagreed with in-memory migration")
                backup = timestamped_backup_path(self._path, old_version)
                shutil.copy2(self._path, backup)
                tmp = self._path.with_name(self._path.name + ".avf-tmp")
                tmp.write_text(new_text, encoding="utf-8")
                shutil.copymode(self._path, tmp)
                os.replace(tmp, self._path)
            except Exception as e:
                self._notice(
                    logging.WARNING,
                    f"Config {self._path} is schema v{version} (current v{CURRENT_CONFIG_VERSION}) "
                    f"but could not be rewritten in place ({e}); migrated in memory only. "
                    f"Run `avf config migrate {self._path}` to rewrite it. Changes:{listing}",
                )
                return in_memory
            self._notice(
                logging.WARNING,
                f"Migrated config {self._path} from schema v{version} to "
                f"v{CURRENT_CONFIG_VERSION}; the original was saved to {backup}. "
                f"Changes:{listing}",
            )
            return rewritten
        self._notice(
            logging.WARNING,
            f"Config {self._path} is schema v{version} (current v{CURRENT_CONFIG_VERSION}); "
            f"migrated in memory only -- the file was not modified. Run "
            f"`avf config migrate {self._path}` to update it. Changes:{listing}",
        )
        return in_memory

    def _merge(self) -> dict[str, Any]:
        merged = self._deep_copy(self.DEFAULTS)
        if not self._explicit and not self._path.exists():
            self._bootstrap_default_config()
        if self._path.exists():
            try:
                with open(self._path) as f:
                    text = f.read()
                user = yaml.safe_load(text) or {}
                if isinstance(user, dict):
                    user = self._migrate_user_config(text, user)
                self._deep_update(merged, user)
            except Exception as e:
                from autovideofixer.logger import get_logger

                get_logger("autovideofixer.config").warning(
                    "Failed to load config from %s: %s. Using defaults.", self._path, e
                )
        return merged

    @staticmethod
    def _deep_copy(d: dict) -> dict:
        import copy

        return copy.deepcopy(d)

    @staticmethod
    def _deep_update(base: dict, override: dict, _path: str = "") -> None:
        """Recursively merge `override` onto `base`, in place.

        Thin backward-compatible wrapper -- the actual merge logic is the
        module-level `deep_merge()`, reused by BaseStage.__init__ for
        per-occurrence stage config overrides (see pipeline.default_order's
        `config:` key).
        """
        deep_merge(base, override, _path)

    def apply_layer(self, layer: dict[str, Any], source_label: str) -> None:
        """Deep-merge an additional cascade layer onto the current config data.

        Used by CLI/GUI callers to fold in each `--preset`/`--config PATH` layer
        (in the order they should apply -- see AGENTS.md's "Config cascade"
        section) and the final CLI-flags layer, on top of the DEFAULTS + user
        config.yaml base already loaded by `__init__`. A layer only clobbers the
        keys it actually specifies (`deep_merge()` semantics) -- list-valued
        keys (e.g. `pipeline.default_order`) are replaced wholesale, not
        element-wise merged.

        `source_label` is a short human-readable tag for this layer (e.g.
        `"preset:1080p60"`, `"config:/path/to/extra.yaml"`, `"cli-flags"`) --
        recorded in `self.sources` and logged at DEBUG (redacted) so a run's
        effective config can be traced back to the layer that set each value.
        """
        from autovideofixer.config_migrations import migrate_layer
        from autovideofixer.logger import get_logger

        if isinstance(layer, dict) and ("analysis" in layer or "config_version" in layer):
            # Old-shape analysis keys (or a declared old config_version) arriving in a layer
            # are migrated in memory rather than silently ignored (§ 24.1). The selected
            # endpoints are looked up on the current data so a layer's old-style
            # `analysis.vlm.model` lands on the endpoint actually in use.
            layer = copy.deepcopy(layer)
            layer_notes = migrate_layer(
                layer,
                vlm_endpoint=str(self.get("analysis", "vlm", "endpoint", default="default_vlm")),
                llm_endpoint=str(self.get("analysis", "llm", "endpoint", default="default_llm")),
            )
            if layer_notes:
                get_logger("autovideofixer.config").warning(
                    "Config layer %r uses the old (schema v1) analysis shape; migrated in "
                    "memory (update the source). Changes:%s",
                    source_label,
                    "".join(f"\n  - {n}" for n in layer_notes),
                )

        deep_merge(self._data, layer)
        self._sources.append(source_label)
        get_logger("autovideofixer.config").debug(
            "Applied config layer %r (stack so far: %s): %s",
            source_label,
            self._sources,
            redact_secrets(layer),
        )

    @property
    def sources(self) -> list[str]:
        """Ordered labels of every layer folded into this Config so far."""
        return list(self._sources)

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    def set(self, value: Any, *keys: str) -> None:
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            elif not isinstance(node[k], dict):
                raise TypeError(
                    f"Cannot set config key {'.'.join(keys)!r}: "
                    f"{'.'.join(keys[: keys.index(k) + 1])!r} is a {type(node[k]).__name__}, "
                    "not a mapping"
                )
            node = node[k]
        node[keys[-1]] = value
        self._save_pending = True

    def save(self) -> None:
        if not self._save_pending:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        self._save_pending = False

    @property
    def data(self) -> dict[str, Any]:
        return self._data
