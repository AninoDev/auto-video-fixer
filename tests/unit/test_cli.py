"""Tests for CLI interface."""

import re

from click.testing import CliRunner

from autovideofixer.cli.cli import main

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI escapes and Rich's line-wrapping so substring checks on
    long paths/dicts in logged output aren't broken by terminal-width
    wrapping or color codes."""
    return _ANSI_RE.sub("", output).replace("\n", "").replace(" ", "")


class TestCLI:
    """Test command-line interface."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_cli_version(self):
        """Test version command."""
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_help(self):
        """Test help command."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Auto Video Fixer" in result.output
        assert "process" in result.output
        assert "analyze" in result.output

    def test_process_help(self):
        """Test process command help."""
        result = self.runner.invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "--preset" in result.output
        assert "--output" in result.output
        assert "--dry-run" in result.output

    def test_presets_command(self):
        """Test presets listing command."""
        result = self.runner.invoke(main, ["presets"])
        assert result.exit_code == 0
        assert "4k60" in result.output
        assert "max_quality" in result.output

    def test_process_dry_run(self, tmp_path):
        """Test dry run mode."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(main, ["process", str(test_file), "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "test.mp4" in result.output

    def test_process_no_files(self):
        """Test processing with no video files."""
        result = self.runner.invoke(main, ["process", "/nonexistent/path"])

        assert result.exit_code != 0
        assert "No video files found" in result.output

    def test_process_invalid_preset(self, tmp_path):
        """Test processing with invalid preset."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(main, ["process", str(test_file), "-p", "invalid_preset"])

        assert result.exit_code != 0
        assert "Unknown preset" in result.output

    def test_gpu_info_command(self):
        """Test GPU info command."""
        result = self.runner.invoke(main, ["gpu-info"])
        assert result.exit_code == 0
        assert "hardware" in result.output.lower() or "No hardware" in result.output

    def test_find_duplicates_help(self):
        """Test find-duplicates help."""
        result = self.runner.invoke(main, ["find-duplicates", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output

    def test_model_info_command(self):
        """Test model-info command."""
        result = self.runner.invoke(main, ["model-info"])
        assert result.exit_code == 0, f"CLI failed: {result.output}\nException: {result.exception}"
        assert "RealESRGAN_x4plus" in result.output
        assert "rife_v4.6" in result.output

    def test_model_download_unknown_model(self):
        """Test model-download with unknown model."""
        result = self.runner.invoke(main, ["model-download", "--model", "nonexistent_model"])
        assert result.exit_code != 0
        assert "Unknown model" in result.output

    def test_model_download_help(self):
        """Test model-download help."""
        result = self.runner.invoke(main, ["model-download", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output


class TestConfigFlag:
    """Tests for the global --config flag / AVF_CONFIG env var (Feature 3)."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_explicit_flag_loads_custom_settings(self, tmp_path):
        """--config PATH loads that file's values into the run."""
        config_path = tmp_path / "custom.yaml"
        config_path.write_text("general:\n  max_concurrent_jobs: 9\n")
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(
            main, ["--config", str(config_path), "process", str(test_file), "--dry-run"]
        )

        assert result.exit_code == 0
        plain = _plain(result.output)
        assert str(config_path).replace(" ", "") in plain
        assert "max_concurrent_jobs':9" in plain

    def test_missing_explicit_path_errors(self, tmp_path):
        """A --config path that doesn't exist is a hard error, not a silent
        fallback to defaults."""
        missing = tmp_path / "does_not_exist.yaml"
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(
            main, ["--config", str(missing), "process", str(test_file), "--dry-run"]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_env_var_used_when_no_flag(self, tmp_path, monkeypatch):
        """AVF_CONFIG is used as a fallback when --config isn't passed."""
        config_path = tmp_path / "env_custom.yaml"
        config_path.write_text("general:\n  max_concurrent_jobs: 4\n")
        monkeypatch.setenv("AVF_CONFIG", str(config_path))
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(main, ["process", str(test_file), "--dry-run"])

        assert result.exit_code == 0
        plain = _plain(result.output)
        assert str(config_path).replace(" ", "") in plain
        assert "max_concurrent_jobs':4" in plain

    def test_flag_beats_env_var(self, tmp_path, monkeypatch):
        """An explicit --config flag wins over AVF_CONFIG."""
        env_config = tmp_path / "env.yaml"
        env_config.write_text("general:\n  max_concurrent_jobs: 4\n")
        flag_config = tmp_path / "flag.yaml"
        flag_config.write_text("general:\n  max_concurrent_jobs: 11\n")
        monkeypatch.setenv("AVF_CONFIG", str(env_config))
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake video")

        result = self.runner.invoke(
            main, ["--config", str(flag_config), "process", str(test_file), "--dry-run"]
        )

        assert result.exit_code == 0
        plain = _plain(result.output)
        assert str(flag_config).replace(" ", "") in plain
        assert str(env_config).replace(" ", "") not in plain
        assert "max_concurrent_jobs':11" in plain
