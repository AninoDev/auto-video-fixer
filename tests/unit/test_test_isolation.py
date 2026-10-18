"""Regression guard: the test suite must never be able to reach the developer's real config.

Since REQUIREMENTS.md § 24 a default-path `Config()` may bootstrap or schema-migrate (rewrite +
back up) the real `~/.config/auto-video-fixer/config.yaml`. tests/conftest.py's autouse fixture
redirects XDG_CONFIG_HOME/XDG_STATE_HOME; these tests fail loudly if that ever regresses.
"""

import os
from pathlib import Path

from autovideofixer.config import Config, get_config_dir, get_config_path, get_log_dir


def test_default_config_path_is_not_the_real_home_config():
    real = Path(os.path.expanduser("~")) / ".config" / "auto-video-fixer"
    assert not str(get_config_dir()).startswith(str(real))
    assert get_config_path().parent != real
    assert "userdirs" in str(get_config_path())


def test_log_dir_is_isolated():
    assert "userdirs" in str(get_log_dir())


def test_bootstrap_disabled_by_default_in_tests():
    cfg = Config()
    assert not get_config_path().exists()
    assert cfg.get("config_version") == 2
