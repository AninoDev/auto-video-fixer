"""Tests for preset system."""

import os

from autovideofixer.core.presets import (
    Preset,
    get_preset,
    list_presets,
    load_preset,
    save_preset,
)


class TestPresets:
    """Test preset management."""

    def test_list_presets(self):
        """Test listing all presets."""
        presets = list_presets()

        assert isinstance(presets, dict)
        assert len(presets) > 0

        # Check for required presets
        required = ["4k60", "4k30", "max_quality", "size_reduction", "remux_only"]
        for name in required:
            assert name in presets, f"Missing required preset: {name}"

    def test_get_preset(self):
        """Test getting a specific preset."""
        preset = get_preset("4k60")

        assert preset is not None
        assert preset.name == "4k60"
        assert preset.display_name == "4K 60fps"
        assert preset.target_resolution == (3840, 2160)
        assert preset.target_framerate == 60.0

    def test_get_nonexistent_preset(self):
        """Test getting a nonexistent preset."""
        preset = get_preset("nonexistent_preset")
        assert preset is None

    def test_preset_to_config(self):
        """Test converting preset to config dictionary."""
        preset = get_preset("4k60")
        config = preset.to_config()

        assert isinstance(config, dict)
        assert "quality" in config
        assert "stages" in config

        # Check quality target
        assert config["quality"]["quality_target"]["target_resolution"] == [3840, 2160]
        assert config["quality"]["quality_target"]["target_framerate"] == 60.0

    def test_preset_enable_stages(self):
        """Test preset stage enablement."""
        preset = get_preset("max_quality")

        assert preset.enable_stages["upscale"] is True
        assert preset.enable_stages["interpolate"] is True
        assert preset.enable_stages["denoise_video"] is True

    def test_max_quality_pins_rrdb_models(self):
        """max_quality is the explicit "maximum quality, slowest" preset --
        it must not inherit deblock's faster/lower-fidelity compact
        realesr-general-wdn-x4v3 default, so it pins the RRDB models."""
        preset = get_preset("max_quality")

        assert preset.stage_overrides["deblock"]["ai_model"] == "RealESRGAN_x4plus"
        assert preset.stage_overrides["upscale"]["ai_model"] == "RealESRGAN_x4plus"
        # Existing upscale overrides are preserved, not clobbered.
        assert preset.stage_overrides["upscale"]["method"] == "ai"
        assert preset.stage_overrides["upscale"]["scale_factor"] == 2.0

    def test_1080p60_4k60_4k30_denoise_video_disabled(self):
        """1080p60/4k60/4k30 previously enabled denoise_video; they now leave
        it disabled and inherit deblock's default denoise-optimized model
        (realesr-general-wdn-x4v3) instead."""
        for name in ("1080p60", "4k60", "4k30"):
            preset = get_preset(name)
            assert preset.enable_stages["denoise_video"] is False, name
            assert preset.enable_stages["deblock"] is True, name

    def test_hdr_enhance_denoise_video_still_enabled(self):
        """hdr_enhance doesn't enable deblock -- denoise_video is its only
        artifact-removal stage, so it stays on there deliberately."""
        preset = get_preset("hdr_enhance")
        assert preset.enable_stages["denoise_video"] is True
        assert "deblock" not in preset.enable_stages

    def test_preset_size_reduction(self):
        """Test size reduction preset configuration."""
        preset = get_preset("size_reduction")

        assert preset.crf == 28  # Higher CRF = smaller file
        assert preset.enable_stages["upscale"] is False
        assert preset.enable_stages["interpolate"] is False

    def test_preset_remux_only(self):
        """Test remux-only preset configuration."""
        preset = get_preset("remux_only")

        assert preset.video_codec == "copy"
        assert preset.audio_codec == "copy"
        assert preset.enable_stages["encode"] is False

    def test_save_and_load_preset(self, tmp_path):
        """Test saving and loading a custom preset."""
        custom_preset = Preset(
            name="custom_test",
            display_name="Custom Test",
            description="Test preset",
            target_resolution=(1920, 1080),
            crf=20,
        )

        # Save
        preset_path = str(tmp_path / "custom_test.json")
        save_preset(custom_preset, preset_path)

        assert os.path.exists(preset_path)

        # Load
        loaded = load_preset(preset_path)
        assert loaded is not None
        assert loaded.name == "custom_test"
        assert loaded.display_name == "Custom Test"
        assert loaded.target_resolution == (1920, 1080)
        assert loaded.crf == 20

    def test_load_nonexistent_preset(self):
        """Test loading a nonexistent preset."""
        loaded = load_preset("/nonexistent/preset.json")
        assert loaded is None
