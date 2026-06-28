"""Auto Video Fixer - Configuration presets.

Provides predefined processing profiles for common use cases.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Preset:
    """A processing preset that defines target output properties."""

    name: str
    display_name: str
    description: str = ""
    target_resolution: tuple[int, int] | None = None
    target_framerate: float | None = None
    target_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18
    preset: str = "medium"
    quality_target: dict[str, Any] = field(default_factory=dict)
    enable_stages: dict[str, bool] = field(default_factory=dict)
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True

    def to_config(self) -> dict[str, Any]:
        """Convert preset to configuration dict."""
        config = {}

        if self.target_resolution:
            config["quality"] = {
                "quality_target": {
                    "mode": "target",
                    "target": self.crf,
                    "target_resolution": list(self.target_resolution),
                    "target_framerate": self.target_framerate,
                }
            }

        stages = {}
        for name, enabled in self.enable_stages.items():
            stages[name] = {"enabled": enabled}
        config["stages"] = stages

        if self.stage_overrides:
            config["stages"] = config.get("stages", {})
            config["stages"].update(self.stage_overrides)

        # Encoding settings
        config["encoding"] = {
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "crf": self.crf,
            "preset": self.preset,
        }

        return config


# ─── Built-in presets ──────────────────────────────────────────────

PRESETS: dict[str, Preset] = {
    "max_quality": Preset(
        name="max_quality",
        display_name="Maximum Quality",
        description="Process for maximum quality output. Slowest processing.",
        target_resolution=(3840, 2160),  # 4K
        target_framerate=60.0,
        video_codec="libx264",
        audio_codec="aac",
        crf=12,
        preset="slower",
        enable_stages={
            "detect": True,
            "stabilize": True,
            "deblock": True,
            "denoise_video": True,
            "upscale": True,
            "interpolate": True,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
        stage_overrides={
            "upscale": {"method": "ai", "scale_factor": 2.0},
            "interpolate": {"method": "ai"},
            "denoise_video": {"method": "ai"},
        },
    ),
    "4k60": Preset(
        name="4k60",
        display_name="4K 60fps",
        description="Upscale to 4K at 60fps",
        target_resolution=(3840, 2160),
        target_framerate=60.0,
        video_codec="libx264",
        audio_codec="aac",
        crf=18,
        preset="medium",
        enable_stages={
            "detect": True,
            "stabilize": True,
            "deblock": True,
            "denoise_video": True,
            "upscale": True,
            "interpolate": True,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
    ),
    "4k30": Preset(
        name="4k30",
        display_name="4K 30fps",
        description="Upscale to 4K at 30fps",
        target_resolution=(3840, 2160),
        target_framerate=30.0,
        video_codec="libx264",
        audio_codec="aac",
        crf=18,
        preset="medium",
        enable_stages={
            "detect": True,
            "stabilize": True,
            "deblock": True,
            "denoise_video": True,
            "upscale": True,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
    ),
    "1080p60": Preset(
        name="1080p60",
        display_name="1080p 60fps",
        description="Smooth 1080p60 output",
        target_resolution=(1920, 1080),
        target_framerate=60.0,
        video_codec="libx264",
        audio_codec="aac",
        crf=20,
        preset="medium",
        enable_stages={
            "detect": True,
            "stabilize": True,
            "denoise_video": True,
            "interpolate": True,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
    ),
    "size_reduction": Preset(
        name="size_reduction",
        display_name="Size Reduction",
        description="Reduce file size with acceptable quality loss",
        video_codec="libx264",
        audio_codec="aac",
        crf=28,
        preset="fast",
        enable_stages={
            "detect": True,
            "stabilize": False,
            "deblock": False,
            "denoise_video": False,
            "upscale": False,
            "interpolate": False,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
        quality_target={
            "mode": "max_loss_pct",
            "target": 28,
            "max_loss_pct": 10.0,
        },
    ),
    "remux_only": Preset(
        name="remux_only",
        display_name="Remux Only",
        description="Just change container format, no encoding",
        video_codec="copy",
        audio_codec="copy",
        enable_stages={
            "detect": True,
            "remux": True,
            "encode": False,
        },
    ),
    "hdr_enhance": Preset(
        name="hdr_enhance",
        display_name="HDR Enhancement",
        description="Convert and enhance HDR content",
        video_codec="libx265",
        audio_codec="aac",
        crf=22,
        preset="medium",
        enable_stages={
            "detect": True,
            "hdr": True,
            "stabilize": True,
            "denoise_video": True,
            "normalize_volume": True,
            "normalize_audio": True,
            "encode": True,
        },
    ),
}


def get_preset(name: str) -> Preset | None:
    """Get a preset by name."""
    return PRESETS.get(name)


def list_presets() -> dict[str, Preset]:
    """List all available presets."""
    return dict(PRESETS)


def save_preset(preset: Preset, path: str | None = None) -> str:
    """Save a custom preset to disk."""
    presets_dir = Path(__file__).parent.parent / "config" / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    if path is None:
        path = str(presets_dir / f"{preset.name}.json")

    with open(path, "w") as f:
        data = asdict(preset)
        # Convert tuple to list for JSON
        data["target_resolution"] = (
            list(preset.target_resolution) if preset.target_resolution else None
        )
        json.dump(data, f, indent=2)

    return path


def load_preset(path: str) -> Preset | None:
    """Load a preset from disk."""
    try:
        with open(path) as f:
            data = json.load(f)
        data["target_resolution"] = (
            tuple(data["target_resolution"]) if data.get("target_resolution") else None
        )
        return Preset(**data)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None
