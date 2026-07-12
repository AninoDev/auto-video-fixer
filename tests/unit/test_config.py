"""Tests for configuration system."""

from pathlib import Path

import pytest
import yaml

from autovideofixer.config import Config


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
