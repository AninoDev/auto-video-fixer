"""Tests for configuration system."""

from pathlib import Path

import yaml

from autovideofixer.config import Config


class TestConfig:
    """Test configuration management."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = Config()
        assert config is not None
        assert config.get("general", "max_concurrent_jobs") == 1
        assert config.get("quality", "vmaf_model") == "vmaf_v0.6.1"

    def test_config_with_custom_path(self, tmp_path):
        """Test creating config with custom path."""
        config_path = tmp_path / "test_config.yaml"
        config = Config(config_path)
        assert config is not None
        assert config._path == config_path

    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = Config()
        assert config.get("stages", "upscale", "enabled") is True
        assert config.get("stages", "interpolate", "ai_model") == "rife_v4.6"

    def test_get_with_default(self):
        """Test getting values with defaults."""
        config = Config()
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

    def test_config_data_property(self):
        """Test accessing raw config data."""
        config = Config()
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
