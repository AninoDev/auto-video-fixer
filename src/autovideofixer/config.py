"""Auto Video Fixer - Configuration system."""

from __future__ import annotations

import os
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
        },
        "gpu": {
            "auto_detect": True,
            "preferred_device": "auto",  # auto, cpu, cuda, metal
            "memory_limit_gb": None,
        },
        "ffmpeg": {
            "binary": None,  # None = auto-detect in PATH
            "hwaccel": "auto",  # auto, cuda, vulkan, qsv, vaapi, none
            "threads": 0,  # 0 = auto
        },
        "quality": {
            "vmaf_model": "vmaf_v0.6.1",
            "vmaf_features": "psnr,ssim,ms_ssim,fast",
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
            # Informational only -- Pipeline.optimize_stage_order() is the actual
            # source of truth for execution order and does not read this list.
            "default_order": [
                "detect",
                "stabilize",
                "deblock",
                "denoise_video",
                "upscale",
                "interpolate",
                "normalize_volume",
                "normalize_audio",
                "speed",
                "hdr",
                "encode",
            ],
            # Must stay >= len(default_order) above (11) since a full default run
            # legitimately uses every stage; this only guards against pathological
            # --stage repetition, not normal preset/auto-determined pipelines.
            "max_stages": 15,
            "skip_stage_on_error": True,
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
                # None = inherit general.ai_fallback; True/False overrides it
                # for this stage only.
                "ai_fallback": None,
            },
            "interpolate": {
                "enabled": True,
                "ai_model": "rife_v4.6",
                "traditional_method": "minterpolate",
                "ai_fallback": None,  # see "upscale".ai_fallback above
            },
            "denoise_video": {
                "enabled": True,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "hqdn3d",
                "tile_size": 0,  # see "upscale".tile_size above
                "ai_fallback": None,  # see "upscale".ai_fallback above
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
                "ai_fallback": None,  # see "upscale".ai_fallback above
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
            },
            "normalize_volume": {
                "enabled": True,
                "target_db": -23.0,  # EBU R128
                "true_peak_db": -2.0,
            },
            "hdr_to_sdr": {
                "enabled": False,
                "method": "bt2020",
            },
            "speed": {
                "enabled": False,
                "factor": 1.0,
            },
        },
        "analysis": {
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
                "similarity_threshold": 0.95,
                "hash_type": "perceptual",  # perceptual (ahash), dhash, combined
            },
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
        from autovideofixer.logger import get_logger

        for k, v in override.items():
            key_path = f"{_path}.{k}" if _path else k
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                Config._deep_update(base[k], v, key_path)
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
