"""Tests for `process --from-file`/`--from-file-parser`: the argv-order-
recovery helper (_resolve_from_file_specs), the inline NAME:PATH override
(_split_inline_parser), and end-to-end --dry-run resolution through the
four built-in parsers."""

import re
import sys

from click.testing import CliRunner

from autovideofixer.cli.cli import _resolve_from_file_specs, _split_inline_parser, main

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return _ANSI_RE.sub("", output)


def _video(tmp_path, name="test.mp4"):
    f = tmp_path / name
    f.write_text("fake video")
    return f


class TestSplitInlineParser:
    def test_registered_prefix_is_treated_as_override(self):
        assert _split_inline_parser("csv:manifest.csv", "shlex") == ("csv", "manifest.csv")

    def test_unregistered_prefix_falls_through_to_default(self):
        # "http" isn't a registered parser name, so the whole value is the
        # path and the current stateful parser is used.
        assert _split_inline_parser("http://example.com/x.mp4", "shlex") == (
            "shlex",
            "http://example.com/x.mp4",
        )

    def test_no_colon_uses_default(self):
        assert _split_inline_parser("plain/path.txt", "lines") == ("lines", "plain/path.txt")

    def test_literal_filename_with_colon_not_a_known_parser(self):
        assert _split_inline_parser("x:y.txt", "shlex") == ("shlex", "x:y.txt")


class TestResolveFromFileSpecsArgvOrder:
    def test_stateful_ordering_across_from_file_parser_switches(self, monkeypatch):
        argv = [
            "avf",
            "process",
            "--from-file-parser",
            "csv",
            "--from-file",
            "a.csv",
            "--from-file",
            "b.csv",
            "--from-file-parser",
            "json",
            "--from-file",
            "c.json",
            "dummy_positional.mp4",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        resolved = _resolve_from_file_specs(
            from_files=("a.csv", "b.csv", "c.json"),
            from_file_parsers=("csv", "json"),
        )
        assert resolved == [
            ("a.csv", "csv"),
            ("b.csv", "csv"),
            ("c.json", "json"),
        ]

    def test_default_parser_is_shlex_before_any_from_file_parser(self, monkeypatch):
        argv = ["avf", "process", "--from-file", "a.txt"]
        monkeypatch.setattr(sys, "argv", argv)
        resolved = _resolve_from_file_specs(from_files=("a.txt",), from_file_parsers=())
        assert resolved == [("a.txt", "shlex")]

    def test_inline_override_wins_regardless_of_stateful_parser(self, monkeypatch):
        argv = [
            "avf",
            "process",
            "--from-file-parser",
            "csv",
            "--from-file",
            "json:special.json",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        resolved = _resolve_from_file_specs(
            from_files=("json:special.json",), from_file_parsers=("csv",)
        )
        assert resolved == [("special.json", "json")]

    def test_argv_mismatch_falls_back_to_last_parser_for_all_files(self, monkeypatch):
        # sys.argv doesn't correspond to this invocation at all (e.g. a
        # programmatic call whose real argv is the test runner's own) --
        # the fallback applies the LAST --from-file-parser to every file,
        # preserving from_files' own order.
        monkeypatch.setattr(sys, "argv", ["pytest", "unrelated", "args"])
        resolved = _resolve_from_file_specs(
            from_files=("a.txt", "b.txt"), from_file_parsers=("lines", "csv")
        )
        assert resolved == [("a.txt", "csv"), ("b.txt", "csv")]

    def test_argv_mismatch_fallback_defaults_to_shlex_with_no_parser_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pytest"])
        resolved = _resolve_from_file_specs(from_files=("a.txt",), from_file_parsers=())
        assert resolved == [("a.txt", "shlex")]

    def test_argv_mismatch_fallback_honors_inline_override(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pytest"])
        resolved = _resolve_from_file_specs(
            from_files=("a.txt", "json:b.json"), from_file_parsers=("lines",)
        )
        assert resolved == [("a.txt", "lines"), ("b.json", "json")]


class TestFromFileDryRunIntegration:
    def setup_method(self):
        self.runner = CliRunner()

    def test_unknown_parser_errors_before_processing(self, tmp_path):
        list_file = tmp_path / "list.txt"
        list_file.write_text("a.mp4\n")
        result = self.runner.invoke(
            main,
            [
                "process",
                "--from-file-parser",
                "nope",
                "--from-file",
                str(list_file),
                "--dry-run",
            ],
        )
        assert result.exit_code == 1
        assert "unknown" in result.output.lower()

    def test_shlex_default_whitespace_and_quotes(self, tmp_path):
        a = _video(tmp_path, "a.mp4")
        b = _video(tmp_path, "has space.mp4")
        list_file = tmp_path / "list.txt"
        list_file.write_text(f'{a}\n"{b}"\n')

        result = self.runner.invoke(main, ["process", "--from-file", str(list_file), "--dry-run"])
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(a) in out
        assert str(b) in out
        assert "Found 2 video file(s)" in out

    def test_lines_parser_comments_and_blanks(self, tmp_path):
        a = _video(tmp_path, "a.mp4")
        b = _video(tmp_path, "b.mp4")
        list_file = tmp_path / "list.txt"
        list_file.write_text(f"# a comment\n\n{a}\n{b}\n")

        result = self.runner.invoke(
            main,
            [
                "process",
                "--from-file-parser",
                "lines",
                "--from-file",
                str(list_file),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(a) in out
        assert str(b) in out
        assert "Found 2 video file(s)" in out

    def test_csv_inline_parser_with_output_column(self, tmp_path):
        a = _video(tmp_path, "a.mp4")
        out_path = tmp_path / "renamed.mp4"
        list_file = tmp_path / "manifest.csv"
        list_file.write_text(f"input,output\n{a},{out_path}\n")

        result = self.runner.invoke(
            main, ["process", "--from-file", f"csv:{list_file}", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(a) in out
        assert "Found 1 video file(s)" in out

    def test_switching_parsers_mid_list(self, tmp_path, monkeypatch):
        a = _video(tmp_path, "a.mp4")
        b = _video(tmp_path, "b.mp4")
        lines_file = tmp_path / "lines.txt"
        lines_file.write_text(f"{a}\n")
        json_file = tmp_path / "list.json"
        json_file.write_text(f'["{b}"]')

        argv = [
            "process",
            "--from-file-parser",
            "lines",
            "--from-file",
            str(lines_file),
            "--from-file-parser",
            "json",
            "--from-file",
            str(json_file),
            "--dry-run",
        ]
        # CliRunner.invoke() doesn't update sys.argv to match its programmatic
        # args, so the stateful argv-order recovery needs a real sys.argv to
        # work from here -- same pattern as
        # test_cli.py's test_interleaved_preset_config_order_from_argv.
        monkeypatch.setattr(sys, "argv", ["avf", *argv])
        result = self.runner.invoke(main, argv)
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(a) in out
        assert str(b) in out
        assert "Found 2 video file(s)" in out

    def test_directory_entry_expands(self, tmp_path):
        d = tmp_path / "videos"
        d.mkdir()
        v1 = _video(d, "v1.mp4")
        v2 = _video(d, "v2.mp4")
        list_file = tmp_path / "list.txt"
        list_file.write_text(str(d) + "\n")

        result = self.runner.invoke(main, ["process", "--from-file", str(list_file), "--dry-run"])
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(v1) in out
        assert str(v2) in out
        assert "Found 2 video file(s)" in out

    def test_combined_positional_and_from_file(self, tmp_path):
        positional = _video(tmp_path, "positional.mp4")
        listed = _video(tmp_path, "listed.mp4")
        list_file = tmp_path / "list.txt"
        list_file.write_text(f"{listed}\n")

        result = self.runner.invoke(
            main,
            ["process", str(positional), "--from-file", str(list_file), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        out = _plain(result.output)
        assert str(positional) in out
        assert str(listed) in out
        assert "Found 2 video file(s)" in out

    def test_no_inputs_from_either_source_errors(self):
        result = self.runner.invoke(main, ["process", "--dry-run"])
        assert result.exit_code == 1
        assert "no video files found" in result.output.lower()

    def test_output_name_combined_with_per_file_output_errors(self, tmp_path):
        a = _video(tmp_path, "a.mp4")
        out_path = tmp_path / "renamed.mp4"
        list_file = tmp_path / "manifest.csv"
        list_file.write_text(f"input,output\n{a},{out_path}\n")

        result = self.runner.invoke(
            main,
            [
                "process",
                "--from-file",
                f"csv:{list_file}",
                "--output-name",
                "final.mp4",
                "--dry-run",
            ],
        )
        assert result.exit_code == 1
        assert "output-name" in result.output.lower()
