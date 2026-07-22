"""Auto Video Fixer - Configuration system."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


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


def prune_old_logs(log_dir: Path, keep: int = 50) -> None:
    """Delete the oldest files in `log_dir` (by mtime) beyond `keep` newest.

    Simple retention for the automatic per-run log file: called once at CLI
    startup, before the current run's log file is created, so it never
    counts (or deletes) the file about to be written.
    """
    if not log_dir.is_dir():
        return
    try:
        files = sorted(
            (p for p in log_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def get_config_path() -> Path:
    """Return path to the main config file."""
    return get_config_dir() / "config.yaml"


_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "password", "secret")


def _looks_like_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


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


def validate_output_handling_config(config: "Config") -> None:
    """Validate the § 6.1/6.2/6.7 ``general.*`` enum-valued keys eagerly.

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
            # deblock now runs BEFORE stabilize (previously the reverse):
            # blocking artifacts come from the source video, so deblocking
            # before stabilization's perspective warping keeps the deblock
            # model's input accurate (no warped block edges) and gives the
            # stabilizer cleaner detail to track motion against.
            "default_order": [
                "detect",
                "deblock",
                "stabilize",
                "crop",
                "denoise_video",
                "upscale",
                "interpolate",
                "normalize_volume",
                "normalize_audio",
                "speed",
                "hdr",
                "encode",
            ],
            # Must stay >= len(default_order) above (12) since a full default run
            # legitimately uses every stage; this only guards against pathological
            # --stage/default_order repetition, not normal preset/auto-determined
            # pipelines. Counts resolved OCCURRENCES, not unique stage names -- a
            # stage repeated via default_order counts once per repetition.
            "max_stages": 15,
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
            },
            "denoise_video": {
                "enabled": True,
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
                # How much of the clip should end up border-free once zoom_enabled's
                # gate decides zoom applies at all (0.0-1.0). 1.0 (default) = today's
                # behavior: vidstabtransform's own optzoom=1 ("optimal static zoom"),
                # sized to the single worst frame so NO frame ever shows a border.
                # 0.0 = no zoom at all (every border from camera motion stays visible).
                # In between: a static zoom= percentage is computed from the
                # zoom_coverage-quantile of per-frame required-zoom estimates (see
                # StabilizeStage._compute_static_zoom_pct) instead of the max --
                # trades "guaranteed no border, ever" for "less aggressive crop,
                # occasional brief borders on the most extreme motion". See AGENTS.md's
                # stabilize zoom section for the accuracy caveat (approximates the
                # smoothed camera path from raw per-frame local-motion values).
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
            },
            "crop": {
                # Auto-crop: detect and remove black borders (letterbox/pillarbox),
                # including residual borders stabilize can introduce. Strictly
                # opt-in -- see docs/REQUIREMENTS.md feature 3, AGENTS.md stage list.
                "enabled": False,
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
        "analysis": {
            "llm": {
                # Coordinating text-LLM used only when scenes.drop_non_content is true.
                # Reviews ALL per-scene VLM summaries together and returns which scene
                # indices (if any) are not part of the video's main content and should be
                # dropped (e.g. a "like and subscribe" interstitial, a channel bumper, an
                # unrelated promotional insert). Can be a cheaper/faster text-only model
                # than analysis.vlm's vision model -- separate provider/model/url/key.
                # Same provider set and the same allow_http/HTTPS gate as analysis.vlm.
                "provider": "ollama",  # ollama, openai, api
                "model": "llama3",
                "api_key": "",
                "api_url": "",
                "allow_http": False,
                # Response token budget (max_tokens / Ollama num_predict) for the
                # coordinator call. Reasoning models spend "thinking" tokens from
                # this same budget before emitting their JSON -- keep this
                # generous or the response gets truncated mid-thought (which
                # fails open: no scenes dropped).
                "max_tokens": 4096,
            },
            "vlm": {
                "enabled": False,
                "provider": "local",  # local, api, ollama, openai
                "model": "llava",
                "api_key": "",
                "api_url": "",
                # Plain-HTTP api_url endpoints are refused for non-loopback
                # hosts unless this is true (frames + api_key travel
                # unencrypted; only enable for a server on a network you
                # control, e.g. a LAN inference box).
                "allow_http": False,
                # Response token budget (max_tokens / Ollama num_predict) per VLM
                # call. Raise if using a reasoning VLM whose thinking tokens
                # count against the response budget.
                "max_tokens": 1024,
                "max_sample_frames": 8,
                "sample_interval_sec": 10.0,
                # Extra text appended to the default (or overridden) user prompt when
                # non-empty, e.g. job-specific context like "these are wildlife clips
                # from a trail camera". Also settable per-run via `avf analyze
                # --prompt-append`.
                "prompt_append": "",
                # Replaces the default user prompt entirely when non-empty.
                # `prompt_append` (if set) is still appended after this. Also settable
                # per-run via `avf analyze --prompt-override`. WARNING: the default
                # prompt instructs the model to return JSON with summary/tags/objects/
                # rating -- an override that changes the requested output format will
                # break that JSON parsing (see _parse_vlm_response's fallback below).
                "prompt_override": "",
                # Replaces the default system prompt entirely when non-empty. No CLI
                # flag; config-only.
                "system_prompt_override": "",
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
        },
    }

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        config_path: str | Path | None = None,
        require_exists: bool = False,
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
        """
        resolved = path or config_path
        self._path = Path(resolved) if resolved is not None else get_config_path()
        self._require_exists = require_exists and resolved is not None
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

    def _merge(self) -> dict[str, Any]:
        merged = self._deep_copy(self.DEFAULTS)
        if self._path.exists():
            try:
                with open(self._path) as f:
                    user = yaml.safe_load(f) or {}
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
        from autovideofixer.logger import get_logger

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
