"""Tests for CLI interface."""

import os
import re
from unittest.mock import patch

import pytest
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

    def test_analyze_help(self):
        """Test analyze command help lists the new flags."""
        result = self.runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--recursive" in result.output
        assert "--full" in result.output
        assert "--csv" in result.output
        assert "--prompt-append" in result.output
        assert "--prompt-override" in result.output
        assert "--scene-threshold" in result.output
        assert "--min-scene-duration" in result.output


class TestAnalyzeCommand:
    """Tests for `avf analyze`: multi-path collection, --csv, --full, prompt
    overrides, per-file failure handling, and progress-callback wiring."""

    def setup_method(self):
        self.runner = CliRunner()

    @staticmethod
    def _touch_video(path) -> str:
        """Create an empty file with a video extension.

        is_video_file()/scan_directory() only look at the extension (and that
        the path exists), so these are enough for tests that stub out
        VideoAnalyzer.analyze() and never actually probe/decode the file.
        """
        path.write_bytes(b"")
        return str(path)

    @staticmethod
    def _fake_analysis(filepath: str, **overrides):
        from autovideofixer.core.analysis import VideoAnalysis

        defaults = dict(
            filepath=filepath,
            filename=os.path.basename(filepath),
            duration=3.0,
            resolution=(320, 240),
            framerate=10.0,
            has_video=True,
            has_audio=False,
            is_hdr=False,
            video_codec="h264",
            total_scenes=1,
        )
        defaults.update(overrides)
        return VideoAnalysis(**defaults)

    def test_multiple_paths_files_and_directory(self, tmp_path):
        """analyze accepts multiple files/directories, mirroring `process`."""
        d = tmp_path / "dir"
        d.mkdir()
        self._touch_video(d / "a.mp4")
        self._touch_video(d / "b.mp4")
        single = self._touch_video(tmp_path / "c.mp4")

        with patch(
            "autovideofixer.core.analysis.VideoAnalyzer.analyze",
            side_effect=lambda fp, **kw: self._fake_analysis(fp),
        ):
            result = self.runner.invoke(main, ["analyze", str(d), single, "--no-vlm"])

        assert result.exit_code == 0, result.output
        assert _plain("Found 3 video file(s)") in _plain(result.output)
        assert "a.mp4" in result.output
        assert "b.mp4" in result.output
        assert "c.mp4" in result.output

    def test_recursive_flag_scans_subdirectories(self, tmp_path):
        """--recursive/-r mirrors `process`'s directory scanning."""
        d = tmp_path / "dir"
        (d / "sub").mkdir(parents=True)
        self._touch_video(d / "top.mp4")
        self._touch_video(d / "sub" / "nested.mp4")

        with patch(
            "autovideofixer.core.analysis.VideoAnalyzer.analyze",
            side_effect=lambda fp, **kw: self._fake_analysis(fp),
        ):
            non_recursive = self.runner.invoke(main, ["analyze", str(d), "--no-vlm"])
            recursive = self.runner.invoke(main, ["analyze", str(d), "-r", "--no-vlm"])

        assert _plain("Found 1 video file(s)") in _plain(non_recursive.output)
        assert _plain("Found 2 video file(s)") in _plain(recursive.output)

    def test_csv_export_handles_commas_and_newlines(self, tmp_path):
        """--csv writes a parseable UTF-8 CSV with the FULL (untruncated) summary."""
        video = self._touch_video(tmp_path / "clip.mp4")
        csv_path = tmp_path / "out.csv"
        summary = "A dog, a cat, and a bird.\nThey all get along surprisingly well."

        fake = self._fake_analysis(
            video,
            vlm_summary=summary,
            vlm_tags=["animals", "cute"],
            vlm_objects=["dog", "cat", "bird"],
            content_rating="G",
        )

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", return_value=fake):
            result = self.runner.invoke(main, ["analyze", video, "--csv", str(csv_path), "--vlm"])

        assert result.exit_code == 0, result.output
        assert _plain(f"Wrote 1 row(s) to {csv_path}") in _plain(result.output)

        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["vlm_summary"] == summary
        assert row["vlm_tags"] == "animals;cute"
        assert row["vlm_objects"] == "dog;cat;bird"
        assert row["content_rating"] == "G"
        assert row["filename"] == "clip.mp4"

    def test_csv_overwrites_existing_file(self, tmp_path):
        """--csv overwrites rather than appending across runs."""
        video = self._touch_video(tmp_path / "clip.mp4")
        csv_path = tmp_path / "out.csv"
        csv_path.write_text("stale,header,row\n1,2,3\n", encoding="utf-8")

        with patch(
            "autovideofixer.core.analysis.VideoAnalyzer.analyze",
            side_effect=lambda fp, **kw: self._fake_analysis(fp),
        ):
            result = self.runner.invoke(
                main, ["analyze", video, "--csv", str(csv_path), "--no-vlm"]
            )

        assert result.exit_code == 0, result.output

        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "stale" not in csv_path.read_text(encoding="utf-8")

    def test_full_flag_prints_untruncated_summary(self, tmp_path):
        """--full prints the complete VLM summary; the default table row is truncated."""
        video = self._touch_video(tmp_path / "clip.mp4")
        long_summary = "Lorem ipsum dolor sit amet. " * 20  # well past the table preview cutoff
        fake = self._fake_analysis(video, vlm_summary=long_summary)

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", return_value=fake):
            result = self.runner.invoke(main, ["analyze", video, "--full", "--vlm"])

        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert _plain(long_summary) in plain
        assert "usefull" not in plain  # sanity: not matching on a substring accident

    def test_default_view_truncates_long_summary(self, tmp_path):
        """Without --full, the table row is truncated with a hint to use --full.

        Note: the FULL summary still lands in the console via the always-on INFO
        log line (feature 1's requirement) -- this test checks the *table's*
        preview specifically, not overall output, since the full text legitimately
        appears elsewhere (the log line) even without --full.
        """
        video = self._touch_video(tmp_path / "clip.mp4")
        long_summary = "Lorem ipsum dolor sit amet. " * 20
        fake = self._fake_analysis(video, vlm_summary=long_summary)

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", return_value=fake):
            result = self.runner.invoke(main, ["analyze", video, "--vlm"])

        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert _plain("use --full for full text") in plain
        # No Rich Panel (the --full-only full-text display) was printed.
        assert "Full VLM Summary" not in result.output

    def test_per_file_failure_continues_and_exits_nonzero(self, tmp_path):
        """A failure analyzing one file logs and continues; overall exit code is non-zero."""
        good = self._touch_video(tmp_path / "good.mp4")
        bad = self._touch_video(tmp_path / "bad.mp4")

        def side_effect(fp, **kw):
            if fp == bad:
                raise RuntimeError("ffprobe boom")
            return self._fake_analysis(fp)

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", side_effect=side_effect):
            result = self.runner.invoke(main, ["analyze", bad, good, "--no-vlm"])

        assert result.exit_code == 1
        assert "Analysis failed for" in result.output
        assert _plain("1 of 2 file(s) failed analysis") in _plain(result.output)
        # The good file was still analyzed despite the earlier failure.
        assert "good.mp4" in result.output

    def test_prompt_flags_forwarded_to_analyzer(self, tmp_path):
        """--prompt-append/--prompt-override are threaded through to VideoAnalyzer.analyze()."""
        video = self._touch_video(tmp_path / "clip.mp4")
        captured = {}

        def side_effect(fp, **kw):
            captured.update(kw)
            return self._fake_analysis(fp)

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", side_effect=side_effect):
            result = self.runner.invoke(
                main,
                [
                    "analyze",
                    video,
                    "--no-vlm",
                    "--prompt-append",
                    "extra context",
                    "--prompt-override",
                    "custom prompt",
                ],
            )

        assert result.exit_code == 0, result.output
        assert captured.get("prompt_append") == "extra context"
        assert captured.get("prompt_override") == "custom prompt"

    def test_scene_flags_forwarded_to_analyzer(self, tmp_path):
        """--scene-threshold/--min-scene-duration are threaded through to analyze()."""
        video = self._touch_video(tmp_path / "clip.mp4")
        captured = {}

        def side_effect(fp, **kw):
            captured.update(kw)
            return self._fake_analysis(fp)

        with patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", side_effect=side_effect):
            result = self.runner.invoke(
                main,
                [
                    "analyze",
                    video,
                    "--no-vlm",
                    "--scene-threshold",
                    "0.08",
                    "--min-scene-duration",
                    "0.5",
                ],
            )

        assert result.exit_code == 0, result.output
        assert captured.get("scene_threshold") == pytest.approx(0.08)
        assert captured.get("min_scene_duration") == pytest.approx(0.5)

    def test_no_video_files_found(self, tmp_path):
        """analyze on a path with no video files exits non-zero with a clear message."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = self.runner.invoke(main, ["analyze", str(empty_dir)])
        assert result.exit_code != 0
        assert "No video files found" in result.output

    @pytest.mark.integration
    def test_real_video_progress_wiring_smoke(self, tmp_video_file):
        """A real (non-mocked) analyze run completes and shows progress-callback
        log lines under CliRunner's non-TTY output capture."""
        result = self.runner.invoke(main, ["analyze", tmp_video_file, "--no-vlm"])

        assert result.exit_code == 0, result.output
        assert "Video Analysis" in result.output
        assert "analyze progress" in result.output


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
