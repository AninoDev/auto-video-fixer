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


def get_config_path() -> Path:
    """Return path to the main config file."""
    return get_config_dir() / "config.yaml"


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
            },
            "interpolate": {
                "enabled": True,
                "ai_model": "rife_v4.6",
                "traditional_method": "minterpolate",
            },
            "denoise_video": {
                "enabled": True,
                "ai_model": "RealESRGAN_x4plus",
                "traditional_method": "hqdn3d",
            },
            "denoise_audio": {
                "enabled": True,
                "ai_model": "demucs",
                "traditional_method": "afftdn",
            },
            "deblock": {
                "enabled": True,
                "strength": "medium",  # low, medium, high
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
                "max_sample_frames": 8,
                "sample_interval_sec": 10.0,
            },
            "event_detection": {
                "enabled": True,
                "scene_change_threshold": 0.3,
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

    def __init__(self, path: Path | None = None):
        self._path = path or get_config_path()
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
