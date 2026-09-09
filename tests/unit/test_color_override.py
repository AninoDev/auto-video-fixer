"""Tests for the reporting.color tri-state ("auto"/"always"/"never") and its
`--color`/`--no-color` CLI override on `avf process`.

The feature exists so `avf process ... | tee run.log` doesn't lose colour:
piping stdout through `tee` makes it a non-TTY pipe, so Rich's own
autodetection silently disables colour even though the user is still
watching a real terminal downstream of `tee`. Covers: the config default and
round-trip, CLI-flag precedence over the config file, the two hard
non-regressions called out in the task (progress bars must stay suppressed
on non-TTY output even with colour forced on; file logs must never gain ANSI
codes), and rejection of an invalid value.
"""

from __future__ import annotations

import io
import logging
import re
import sys

import pytest
from click.testing import CliRunner
from rich.console import Console

from autovideofixer.cli.cli import main
from autovideofixer.cli.progress import resolve_show_progress
from autovideofixer.config import VALID_COLOR_MODES, Config
from autovideofixer.logger import build_console, get_logger, set_console_color, setup_logging

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI escapes and Rich's line-wrapping so substring checks on
    long dict-repr output aren't broken by terminal-width wrapping or color
    codes -- same helper as tests/unit/test_cli.py."""
    return _ANSI_RE.sub("", output).replace("\n", "").replace(" ", "")


class TestConfigDefault:
    def test_default_is_auto(self):
        assert Config().get("reporting", "color", default=None) == "auto"

    def test_always_round_trips_through_apply_layer(self):
        config = Config()
        config.apply_layer({"reporting": {"color": "always"}}, "test")
        assert config.get("reporting", "color") == "always"

    def test_never_round_trips_through_apply_layer(self):
        config = Config()
        config.apply_layer({"reporting": {"color": "never"}}, "test")
        assert config.get("reporting", "color") == "never"


class TestBuildConsole:
    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="color"):
            build_console(color="purple")

    def test_always_forces_ansi_even_on_non_tty_stream(self):
        buf = io.StringIO()
        console = build_console(color="always", file=buf)
        console.print("[red]hello[/red]")
        assert "\x1b[" in buf.getvalue()

    def test_never_suppresses_ansi_even_on_forced_terminal(self):
        # Sanity check first: force_terminal alone (simulating an already
        # forced/real terminal) DOES emit ANSI when color isn't overridden.
        sanity_buf = io.StringIO()
        Console(file=sanity_buf, force_terminal=True).print("[red]hello[/red]")
        assert "\x1b[" in sanity_buf.getvalue()

        # "never" must suppress colour even on that same forced-terminal
        # console (Rich's no_color wins over force_terminal).
        buf = io.StringIO()
        never_console = Console(file=buf, force_terminal=True, no_color=True)
        never_console.print("[red]hello[/red]")
        assert "\x1b[" not in buf.getvalue()

        # build_console("never") itself must produce the same no_color
        # suppression.
        buf2 = io.StringIO()
        build_console(color="never", file=buf2).print("[red]hello[/red]")
        assert "\x1b[" not in buf2.getvalue()

    def test_auto_leaves_rich_autodetection_in_place(self, monkeypatch):
        # Rich's own autodetection also honors FORCE_COLOR/NO_COLOR/
        # CLICOLOR_FORCE env vars -- clear them so this test exercises pure
        # TTY autodetection (a plain io.StringIO() isn't a terminal), not
        # whatever happens to be set in the ambient environment.
        for var in ("FORCE_COLOR", "NO_COLOR", "CLICOLOR_FORCE"):
            monkeypatch.delenv(var, raising=False)
        buf = io.StringIO()
        console = build_console(color="auto", file=buf)
        console.print("[red]hello[/red]")
        # A plain io.StringIO() isn't a terminal, so "auto" must NOT force
        # colour on -- this is the behavior --color exists to override.
        assert "\x1b[" not in buf.getvalue()


class TestSetupLoggingColor:
    def teardown_method(self):
        setup_logging(level="INFO")  # reset to a clean default state

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="color"):
            setup_logging(level="INFO", color="purple")

    def test_color_always_forces_ansi_in_console_output(self, capsys):
        setup_logging(level="INFO", color="always")
        logger = get_logger("autovideofixer.test_color_override_console")
        logger.info("hello")
        for h in logging.getLogger("autovideofixer").handlers:
            h.flush()
        captured = capsys.readouterr()
        assert "\x1b[" in captured.err

    def test_set_console_color_retargets_existing_richhandler(self, capsys):
        """set_console_color() lets `process` apply a value resolved AFTER
        setup_logging() already ran (its own CLI-flag cascade finishes
        later than the group callback's initial setup_logging() call)."""
        setup_logging(level="INFO", color="auto")
        set_console_color("always")
        logger = get_logger("autovideofixer.test_set_console_color")
        logger.info("hello")
        for h in logging.getLogger("autovideofixer").handlers:
            h.flush()
        captured = capsys.readouterr()
        assert "\x1b[" in captured.err

    def test_color_always_does_not_leak_ansi_into_file_log(self, tmp_path):
        """Regression: forcing colour on must never affect the plain-text
        auto_log_file handler -- only the RichHandler console."""
        log_path = tmp_path / "run.log"
        setup_logging(level="INFO", auto_log_file=str(log_path), color="always")
        logger = get_logger("autovideofixer.test_color_file_regression")
        logger.info("hello world, this should stay plain text")
        for h in logging.getLogger("autovideofixer").handlers:
            h.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "\x1b[" not in content
        assert "hello world, this should stay plain text" in content


class TestProgressBarIndependence:
    """Regression: reporting.color=always must NOT re-enable progress bars
    for non-TTY output -- cli.py's `resolve_show_progress` gate is AND'd
    with the real fd-level sys.stdout.isatty(), which forcing colour cannot
    fake (unlike console.is_terminal, which force_terminal=True *does*
    flip)."""

    def test_forced_color_console_reports_terminal_true(self):
        # Confirm the trap is real: force_terminal=True does make
        # console.is_terminal True even though nothing about the real
        # stdout changed.
        console = build_console(color="always")
        assert console.is_terminal is True

    def test_forced_color_plus_non_tty_stdout_keeps_bars_disabled(self):
        console = build_console(color="always")
        # sys.stdout is not a real terminal under the test runner, so this
        # mirrors cli.py's exact `console.is_terminal and sys.stdout.isatty()`
        # expression.
        is_real_terminal = console.is_terminal and sys.stdout.isatty()
        show_batch, show_file = resolve_show_progress(True, True, is_real_terminal)
        assert show_batch is False
        assert show_file is False


class TestProcessCliFlag:
    def setup_method(self):
        self.runner = CliRunner()

    @staticmethod
    def _video(tmp_path, name="test.mp4"):
        f = tmp_path / name
        f.write_text("fake video")
        return f

    def test_color_flag_sets_config_key_to_always(self, tmp_path):
        test_file = self._video(tmp_path)
        result = self.runner.invoke(main, ["process", str(test_file), "--color", "--dry-run"])
        assert result.exit_code == 0
        assert "'color':'always'" in _plain(result.output)

    def test_no_color_flag_sets_config_key_to_never(self, tmp_path):
        test_file = self._video(tmp_path)
        result = self.runner.invoke(main, ["process", str(test_file), "--no-color", "--dry-run"])
        assert result.exit_code == 0
        assert "'color':'never'" in _plain(result.output)

    def test_flag_absent_leaves_config_file_value_untouched(self, tmp_path, monkeypatch):
        """No --color/--no-color on the command line must leave whatever
        reporting.color the config file set alone -- CLI flag > config file
        > "auto", and an absent flag must not reset it to "auto". No other
        CLI flag is passed either, so the "cli-flags" config layer never
        even gets applied (see process()'s `if final_layer:` guard) --
        proof the config-file value survived untouched all the way through,
        not just that some CLI layer happened to match it."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("reporting:\n  color: always\n")
        monkeypatch.setenv("AVF_CONFIG", str(config_path))

        test_file = self._video(tmp_path)
        result = self.runner.invoke(main, ["process", str(test_file), "--dry-run"])

        assert result.exit_code == 0
        assert "'color':'always'" in _plain(result.output)

    def test_cli_flag_overrides_config_file(self, tmp_path, monkeypatch):
        """--no-color must beat a config file that set reporting.color: always."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("reporting:\n  color: always\n")
        monkeypatch.setenv("AVF_CONFIG", str(config_path))

        test_file = self._video(tmp_path)
        result = self.runner.invoke(main, ["process", str(test_file), "--no-color", "--dry-run"])

        assert result.exit_code == 0
        assert "'color':'never'" in _plain(result.output)

    def test_invalid_color_value_from_set_is_rejected(self, tmp_path):
        """--set reporting.color=purple (an invalid enum value, same class of
        error as an invalid general.log_type) must be rejected with a
        non-zero exit rather than silently accepted."""
        test_file = self._video(tmp_path)
        result = self.runner.invoke(
            main,
            ["process", str(test_file), "--set", "reporting.color=purple", "--dry-run"],
        )
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()
        assert "reporting.color" in result.output.lower() or "color" in result.output.lower()


class TestValidColorModesConstant:
    def test_contains_exactly_the_three_documented_values(self):
        assert set(VALID_COLOR_MODES) == {"auto", "always", "never"}
