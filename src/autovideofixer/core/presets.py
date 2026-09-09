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
        config: dict[str, Any] = {}

        quality_target: dict[str, Any] = {}
        if self.target_resolution:
            quality_target.update(
                {
                    "mode": "target",
                    "target": self.crf,
                    "target_resolution": list(self.target_resolution),
                    "target_framerate": self.target_framerate,
                }
            )
        # Explicit quality_target (e.g. size_reduction's max_loss_pct bound) always
        # merges in, and can override the target-resolution-derived defaults above.
        if self.quality_target:
            quality_target.update(self.quality_target)
        if quality_target:
            config["quality"] = {"quality_target": quality_target}

        stages = {}
        for name, enabled in self.enable_stages.items():
            stages[name] = {"enabled": enabled}

        if self.stage_overrides:
            # Must merge per-key, not `stages.update(self.stage_overrides)` -- a
            # plain dict.update() replaces the whole per-stage value, silently
            # dropping "enabled" whenever a stage appears in BOTH enable_stages
            # and stage_overrides (true for upscale/interpolate/denoise_video in
            # max_quality). Downstream, Config._deep_update() then merges this
            # preset dict into the user's persisted config -- since "enabled" is
            # missing here, the user's existing (possibly disabled) value for
            # that stage survives untouched, so a preset that means to force AI
            # upscaling/interpolation/denoising back on silently fails to if the
            # user had previously disabled that stage in config.yaml or the GUI.
            for name, overrides in self.stage_overrides.items():
                stages[name] = {**stages.get(name, {}), **overrides}

        config["stages"] = stages

        # Encoding settings -- consumed by Pipeline.execute_job(), which merges
        # this into job.stage_overrides["encode"] (see pipeline.py).
        config["encoding"] = {
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "crf": self.crf,
            "preset": self.preset,
        }

        # Target container format -- consumed by Pipeline.execute_job(), which
        # injects it into input_info so RemuxStage.should_run() can see it.
        config["general"] = {"target_format": self.target_format}

        return config


# ─── Built-in presets ──────────────────────────────────────────────
#
# Note (docs/REQUIREMENTS.md § 12): none of the presets below mention
# "retime" in enable_stages/stage_overrides -- this is deliberate, not an
# omission. Cadence recovery is universally beneficial (it only ever removes
# duplicate/padding frames the source itself doesn't need, or SKIPs cheaply
# on an already-honest input) and is on by default (stages.retime.enabled:
# true in Config.DEFAULTS), so every preset -- including remux_only, where
# it still pays off before a mere container change -- simply inherits the
# global default rather than each preset re-asserting it.

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
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
            "encode": True,
        },
        stage_overrides={
            # Pin the highest-quality RRDB models -- this preset is the
            # explicit "maximum quality, slowest" option, so it should not
            # inherit the faster/lower-fidelity compact realesr-general-
            # wdn-x4v3 default (see DEFAULTS["stages"]["deblock"] in
            # config.py).
            "upscale": {"method": "ai", "scale_factor": 2.0, "ai_model": "RealESRGAN_x4plus"},
            "interpolate": {"method": "ai"},
            "deblock": {"ai_model": "RealESRGAN_x4plus"},
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
            # deblock's default ai_model (realesr-general-wdn-x4v3) already
            # doubles as a denoise pass -- see DEFAULTS["stages"].
            "denoise_video": False,
            "upscale": True,
            "interpolate": True,
            "normalize_volume": True,
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
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
            # deblock's default ai_model (realesr-general-wdn-x4v3) already
            # doubles as a denoise pass -- see DEFAULTS["stages"].
            "denoise_video": False,
            "upscale": True,
            "normalize_volume": True,
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
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
            "deblock": True,
            # deblock's default ai_model (realesr-general-wdn-x4v3) already
            # doubles as a denoise pass -- see DEFAULTS["stages"].
            "denoise_video": False,
            "upscale": True,
            "interpolate": True,
            "normalize_volume": True,
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
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
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
            "encode": True,
        },
        quality_target={
            # "max_loss_pct" is not a recognized QualityMode (only
            # none/min/avg/max/target exist) and `target` was previously set to
            # 28 -- the CRF value, not a quality score -- so this never gated
            # anything. "target" mode with an SSIM*100-scale threshold is what
            # Pipeline.execute_job()'s quality gate actually checks against.
            "mode": "target",
            "target": 85.0,
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
            "normalize_audio": False,  # duplicate of normalize_volume, see that stage's docstring
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
    except FileNotFoundError, json.JSONDecodeError, TypeError:
        return None
