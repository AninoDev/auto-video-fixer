"""Tests for configuration system."""

from pathlib import Path

import pytest
import yaml

from autovideofixer.config import Config, validate_output_handling_config


class TestConfig:
    """Test configuration management."""

    def test_default_config_creation(self, tmp_path):
        """Test creating config with defaults."""
        config = Config(tmp_path / "nonexistent.yaml")
        assert config is not None
        assert config.get("general", "max_concurrent_jobs") == 1
        assert config.get("quality", "vmaf_model") == "vmaf_v0.6.1"

    def test_config_with_custom_path(self, tmp_path):
        """Test creating config with custom path."""
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        assert config is not None
        assert config._path == config_path

    def test_get_nested_value(self, tmp_path):
        """Test getting nested configuration values."""
        config = Config(tmp_path / "config.yaml")
        assert config.get("stages", "upscale", "enabled") is True
        assert config.get("stages", "interpolate", "ai_model") == "rife_v4.6"

    def test_deblock_defaults_to_compact_denoise_optimized_model(self, tmp_path):
        """deblock's default ai_model is the compact, denoise-optimized
        realesr-general-wdn-x4v3 (not the RRDB RealESRGAN_x4plus) -- it
        doubles as both deblock and denoise, which is why denoise_video
        defaults to disabled (see the next test)."""
        config = Config(tmp_path / "nonexistent.yaml")
        assert config.get("stages", "deblock", "ai_model") == "realesr-general-wdn-x4v3"

    def test_denoise_video_disabled_by_default(self, tmp_path):
        """denoise_video defaults to disabled -- deblock's default ai_model
        already covers denoising. It stays in pipeline.default_order at its
        slot (omission != disable)."""
        config = Config(tmp_path / "nonexistent.yaml")
        assert config.get("stages", "denoise_video", "enabled") is False
        assert "denoise_video" in config.get("pipeline", "default_order")

    def test_get_with_default(self, tmp_path):
        """Test getting values with defaults."""
        config = Config(tmp_path / "nonexistent.yaml")
        assert config.get("nonexistent", "key", default="fallback") == "fallback"
        assert config.get("general", "nonexistent_key", default="/tmp") == "/tmp"

    def test_set_value(self, tmp_path):
        """Test setting configuration values."""
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        config.set(2, "general", "max_concurrent_jobs")
        assert config.get("general", "max_concurrent_jobs") == 2

    def test_set_nested_value(self, tmp_path):
        """Test setting nested configuration values."""
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)

        config.set(24, "stages", "normalize_volume", "target_db")
        assert config.get("stages", "normalize_volume", "target_db") == 24

    def test_save_and_reload(self, tmp_path):
        """Test saving and reloading configuration."""
        config_path = tmp_path / "test_config.yaml"

        # Save
        config1 = Config(config_path)
        config1.set(3, "general", "max_concurrent_jobs")
        config1.save()

        # Reload
        config2 = Config(config_path)
        assert config2.get("general", "max_concurrent_jobs") == 3

    def test_config_persistence(self, tmp_path):
        """Test that config persists across instances."""
        config_path = tmp_path / "persistent_config.yaml"

        config1 = Config(config_path)
        config1.set("test_value", "general", "test_key")
        config1.save()

        config2 = Config(config_path)
        assert config2.get("general", "test_key") == "test_value"

    def test_merge_user_config(self, tmp_path):
        """Test merging user configuration with defaults."""
        config_path = tmp_path / "merge_config.yaml"

        # Create initial config
        config = Config(config_path)
        config.set(5, "general", "max_concurrent_jobs")

        # Create user config file
        user_config = {"general": {"max_concurrent_jobs": 10, "new_key": "new_value"}}
        with open(config_path, "w") as f:
            yaml.dump(user_config, f)

        # Reload and verify merge
        config2 = Config(config_path)
        assert config2.get("general", "max_concurrent_jobs") == 10
        assert config2.get("general", "new_key") == "new_value"

    def test_config_data_property(self, tmp_path):
        """Test accessing raw config data."""
        config = Config(tmp_path / "nonexistent.yaml")
        data = config.data
        assert isinstance(data, dict)
        assert "general" in data
        assert "stages" in data

    def test_platform_config_dirs(self):
        """Test platform-specific config directory detection."""
        from autovideofixer.config import get_config_dir, get_data_dir

        config_dir = get_config_dir()
        data_dir = get_data_dir()

        assert isinstance(config_dir, Path)
        assert isinstance(data_dir, Path)
        assert "auto-video-fixer" in str(config_dir)
        assert "auto-video-fixer" in str(data_dir)

    def test_state_and_log_dirs(self):
        """Test platform-specific state/log directory detection."""
        from autovideofixer.config import get_log_dir, get_state_dir

        state_dir = get_state_dir()
        log_dir = get_log_dir()

        assert isinstance(state_dir, Path)
        assert isinstance(log_dir, Path)
        assert "auto-video-fixer" in str(state_dir)
        assert log_dir == state_dir / "logs"


class TestConfigExplicitPath:
    """Tests for Config's explicit-path / --config support (Feature 3)."""

    def test_explicit_path_loads_file_values(self, tmp_path):
        """An explicit config_path loads that file's values."""
        config_path = tmp_path / "custom.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"general": {"max_concurrent_jobs": 7}}, f)

        config = Config(config_path=config_path)
        assert config.get("general", "max_concurrent_jobs") == 7
        assert config._path == config_path

    def test_positional_path_still_works(self, tmp_path):
        """Existing positional `path` API keeps working unchanged."""
        config_path = tmp_path / "custom.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"general": {"max_concurrent_jobs": 3}}, f)

        config = Config(config_path)
        assert config.get("general", "max_concurrent_jobs") == 3

    def test_require_exists_missing_path_errors(self, tmp_path):
        """require_exists=True with a missing path fails loudly instead of
        silently falling back to defaults."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            Config(missing, require_exists=True)

    def test_require_exists_existing_path_ok(self, tmp_path):
        config_path = tmp_path / "custom.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"general": {"max_concurrent_jobs": 2}}, f)

        config = Config(config_path, require_exists=True)
        assert config.get("general", "max_concurrent_jobs") == 2

    def test_default_path_not_required_to_exist(self, tmp_path, monkeypatch):
        """Without an explicit path, require_exists is a no-op -- the default
        platform config path is never mandatory."""
        monkeypatch.setattr("autovideofixer.config.get_config_path", lambda: tmp_path / "nope.yaml")
        config = Config(require_exists=True)
        assert config is not None


class TestRedactSecrets:
    def test_redacts_api_key(self):
        from autovideofixer.config import redact_secrets

        data = {"analysis": {"vlm": {"api_key": "sk-supersecret", "provider": "openai"}}}
        redacted = redact_secrets(data)
        assert redacted["analysis"]["vlm"]["api_key"] == "***"
        assert redacted["analysis"]["vlm"]["provider"] == "openai"

    def test_leaves_empty_secret_values_alone(self):
        from autovideofixer.config import redact_secrets

        data = {"api_key": ""}
        assert redact_secrets(data)["api_key"] == ""

    def test_leaves_non_secret_keys_alone(self):
        from autovideofixer.config import redact_secrets

        data = {"general": {"output_dir": "/tmp/out"}}
        assert redact_secrets(data) == data


class TestSanitizeConsoleText:
    """sanitize_console_text() must neutralize control/escape bytes a
    filename or ffmpeg stderr excerpt could use to corrupt a terminal's tty
    mode, while never touching legitimate printable Unicode.
    """

    def test_strips_esc_sequence(self):
        from autovideofixer.config import sanitize_console_text

        # OSC "set title" sequence terminated by BEL -- a classic
        # terminal-corrupting payload smuggled via a crafted filename.
        raw = "evil\x1b]0;pwned\x07name.mp4"
        out = sanitize_console_text(raw)
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "evil" in out and "name.mp4" in out

    def test_strips_c1_range(self):
        from autovideofixer.config import sanitize_console_text

        raw = "c1\x9b31mtest"
        out = sanitize_console_text(raw)
        assert "\x9b" not in out
        assert "c1" in out and "test" in out

    def test_strips_del(self):
        from autovideofixer.config import sanitize_console_text

        out = sanitize_console_text("a\x7fb")
        assert "\x7f" not in out

    def test_keeps_tab_and_newline(self):
        from autovideofixer.config import sanitize_console_text

        raw = "line1\tcolumn\nline2"
        assert sanitize_console_text(raw) == raw

    def test_preserves_cjk_rtl_emoji_and_combining_marks(self):
        from autovideofixer.config import sanitize_console_text

        raw = "中文测试 \U0001f600 اختبار é"
        assert sanitize_console_text(raw) == raw

    def test_does_not_normalize_unicode(self):
        from autovideofixer.config import sanitize_console_text

        # Combining acute accent (U+0301) kept as a separate codepoint, not
        # collapsed/NFC-normalized into a precomposed form.
        raw = "café"
        out = sanitize_console_text(raw)
        assert out == raw
        assert len(out) == len(raw) == 5

    def test_none_is_safe(self):
        from autovideofixer.config import sanitize_console_text

        assert sanitize_console_text(None) == ""

    def test_empty_string_is_safe(self):
        from autovideofixer.config import sanitize_console_text

        assert sanitize_console_text("") == ""

    def test_accepts_non_str_via_str_conversion(self):
        from autovideofixer.config import sanitize_console_text

        assert sanitize_console_text(42) == "42"

    def test_every_c0_except_tab_newline_is_replaced(self):
        from autovideofixer.config import sanitize_console_text

        for code in range(0x00, 0x20):
            if code in (0x09, 0x0A):
                continue
            out = sanitize_console_text(f"x{chr(code)}y")
            assert chr(code) not in out, f"C0 byte 0x{code:02x} leaked through"


class TestDiffFromDefaults:
    def test_no_changes_yields_empty_diff(self):
        from autovideofixer.config import diff_from_defaults

        assert diff_from_defaults(Config.DEFAULTS, Config.DEFAULTS) == {}

    def test_changed_leaf_surfaces_in_diff(self):
        from autovideofixer.config import diff_from_defaults

        data = Config.DEFAULTS.copy()
        data["general"] = {**data["general"], "max_concurrent_jobs": 8}
        diff = diff_from_defaults(data, Config.DEFAULTS)
        assert diff == {"general": {"max_concurrent_jobs": 8}}


class TestConfigCascadeLayers:
    """Tests for Config.apply_layer() / the cascade-layer mechanism (spec:
    DEFAULTS < user config.yaml < explicit --config/--preset layers, in
    argv order < the final CLI-flags layer)."""

    def test_apply_layer_folds_onto_existing_data(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.apply_layer({"general": {"max_concurrent_jobs": 5}}, "layer1")
        assert config.get("general", "max_concurrent_jobs") == 5
        # Sibling keys under "general" are untouched (deep-merge, not replace).
        assert config.get("general", "output_container") == "mp4"

    def test_layer_fold_order_defaults_lt_user_lt_layer1_lt_layer2(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"general": {"max_concurrent_jobs": 2}}, f)

        config = Config(config_path)
        assert config.get("general", "max_concurrent_jobs") == 2  # user yaml beat DEFAULTS (1)

        config.apply_layer({"general": {"max_concurrent_jobs": 3}}, "layer1")
        assert config.get("general", "max_concurrent_jobs") == 3  # layer1 beat user yaml

        config.apply_layer({"general": {"max_concurrent_jobs": 4}}, "layer2")
        assert config.get("general", "max_concurrent_jobs") == 4  # layer2 beat layer1

    def test_layer_only_clobbers_keys_it_specifies(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.apply_layer({"stages": {"upscale": {"scale_factor": 8}}}, "layer1")
        assert config.get("stages", "upscale", "scale_factor") == 8
        # Untouched sibling key under stages.upscale survives.
        assert config.get("stages", "upscale", "ai_model") == "RealESRGAN_x4plus"

    def test_wholesale_list_replacement(self, tmp_path):
        """A later layer's list-valued key fully replaces the earlier one's --
        no element-wise merging."""
        config = Config(tmp_path / "nonexistent.yaml")
        config.apply_layer({"pipeline": {"default_order": ["detect", "encode"]}}, "layer1")
        assert config.get("pipeline", "default_order") == ["detect", "encode"]

        config.apply_layer({"pipeline": {"default_order": ["encode"]}}, "layer2")
        assert config.get("pipeline", "default_order") == ["encode"]

    def test_sources_records_each_layer_label(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"general": {"max_concurrent_jobs": 2}}, f)

        config = Config(config_path)
        assert config.sources == ["defaults", f"user-config:{config_path}"]

        config.apply_layer({}, "preset:1080p60")
        config.apply_layer({}, "config:/tmp/extra.yaml")
        assert config.sources == [
            "defaults",
            f"user-config:{config_path}",
            "preset:1080p60",
            "config:/tmp/extra.yaml",
        ]

    def test_sources_omits_user_config_when_absent(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        assert config.sources == ["defaults"]

    def test_apply_layer_can_set_arbitrary_top_level_keys(self, tmp_path):
        """A layer (e.g. a preset) may set ANY config key, not just stage keys."""
        config = Config(tmp_path / "nonexistent.yaml")
        config.apply_layer({"encoding": {"video_codec": "libx265"}}, "layer1")
        assert config.get("encoding", "video_codec") == "libx265"

    def test_apply_layer_logs_at_debug_with_secrets_redacted(self, tmp_path, caplog):
        import logging

        config = Config(tmp_path / "nonexistent.yaml")
        with caplog.at_level(logging.DEBUG, logger="autovideofixer.config"):
            config.apply_layer(
                {"analysis": {"vlm": {"api_key": "sk-supersecret"}}}, "config:/tmp/extra.yaml"
            )
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "sk-supersecret" not in joined
        assert "***" in joined


class TestValidateOutputHandlingConfig:
    """REQUIREMENTS.md § 6.1/6.2 general.* enum validation -- a clear startup
    error on a typo'd value, but mismatched_max_renames=0 is allowed (it's a
    deliberate "renaming disabled" posture, not a misconfiguration)."""

    def test_defaults_pass(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        validate_output_handling_config(config)  # must not raise

    def test_invalid_existing_output_raises(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set("sikp", "general", "existing_output")
        with pytest.raises(ValueError, match="existing_output"):
            validate_output_handling_config(config)

    def test_invalid_existing_mismatched_raises(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set("delete", "general", "existing_mismatched")
        with pytest.raises(ValueError, match="existing_mismatched"):
            validate_output_handling_config(config)

    def test_negative_max_renames_raises(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(-1, "general", "mismatched_max_renames")
        with pytest.raises(ValueError, match="mismatched_max_renames"):
            validate_output_handling_config(config)

    def test_max_renames_zero_is_allowed(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(0, "general", "mismatched_max_renames")
        validate_output_handling_config(config)  # must not raise

    def test_max_renames_null_is_allowed(self, tmp_path):
        config = Config(tmp_path / "nonexistent.yaml")
        config.set(None, "general", "mismatched_max_renames")
        validate_output_handling_config(config)  # must not raise

    def test_invalid_log_type_raises(self, tmp_path):
        """REQUIREMENTS.md § 6.7: general.log_type is validated the same way
        as the § 6.1/6.2 enum keys above -- a clear startup error, not a
        confusing failure the first time logging tries to read it."""
        config = Config(tmp_path / "nonexistent.yaml")
        config.set("verbose", "general", "log_type")
        with pytest.raises(ValueError, match="log_type"):
            validate_output_handling_config(config)

    def test_valid_log_types_pass(self, tmp_path):
        for value in ("raw", "clean", "both", "none"):
            config = Config(tmp_path / "nonexistent.yaml")
            config.set(value, "general", "log_type")
            validate_output_handling_config(config)  # must not raise
