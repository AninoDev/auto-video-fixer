"""Tests for CLI interface."""

from click.testing import CliRunner

from autovideofixer.cli.cli import main


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
        assert result.exit_code == 0
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
