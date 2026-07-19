"""Tests for the logging setup's console-output sanitization.

Regression coverage for: a filename (or ffmpeg stderr excerpt) containing
raw ESC/C1 control bytes must never reach the console handler's rendered
output -- those bytes can toggle a terminal's tty mode (see
config.sanitize_console_text's docstring and AGENTS.md's "every new
subprocess spawn must detach stdin" gotcha for the primary fix this is
defense-in-depth for).
"""

from __future__ import annotations

import logging

import pytest

from autovideofixer.logclean import get_pii_cleaner, reset_pii_cleaner
from autovideofixer.logger import (
    _CleaningFileFormatter,
    _insert_log_suffix,
    _SanitizingConsoleFormatter,
    get_logger,
    setup_logging,
)


def _make_record(msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="autovideofixer.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestSanitizingConsoleFormatter:
    def test_strips_esc_and_c1_from_formatted_message(self):
        formatter = _SanitizingConsoleFormatter("%(message)s")
        record = _make_record('Processing "evil\x1b]0;evil\x07file.mp4" with C1 \x9b31m marker')
        out = formatter.format(record)
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "\x9b" not in out
        # The surrounding legitimate text survives.
        assert "Processing" in out
        assert "file.mp4" in out

    def test_percent_style_interpolation_still_sanitized(self):
        """Args interpolated via %-style logging (the common `logger.info("...%s", x)`
        pattern) go through record.getMessage() inside super().format() before
        sanitization runs, so an external value passed as an arg is covered too."""
        formatter = _SanitizingConsoleFormatter("%(message)s")
        record = _make_record("Analyzing: %s", "clip\x1b]0;pwn\x07.mp4")
        out = formatter.format(record)
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "clip" in out and ".mp4" in out

    def test_preserves_rich_markup_tags(self):
        """Rich's [style]...[/style] bbcode-like markup is plain ASCII
        brackets/letters, not raw control bytes -- sanitizing must not
        mangle deliberate markup the code itself emits."""
        formatter = _SanitizingConsoleFormatter("%(message)s")
        record = _make_record("[red]Failed[/red]: normal text")
        out = formatter.format(record)
        assert out == "[red]Failed[/red]: normal text"

    def test_preserves_non_ascii(self):
        formatter = _SanitizingConsoleFormatter("%(message)s")
        record = _make_record("视频处理完成 emoji \U0001f600")
        out = formatter.format(record)
        assert out == "视频处理完成 emoji \U0001f600"


class TestSetupLoggingWiresSanitizer:
    def test_console_handler_uses_sanitizing_formatter(self):
        """setup_logging() must attach _SanitizingConsoleFormatter to the
        console handler specifically (not file handlers, which keep raw
        text for debugging)."""
        setup_logging(level="INFO")
        try:
            root = logging.getLogger("autovideofixer")
            console_handlers = [
                h for h in root.handlers if isinstance(h.formatter, _SanitizingConsoleFormatter)
            ]
            assert len(console_handlers) == 1
        finally:
            setup_logging(level="INFO")  # reset to a clean default state

    def test_file_handler_keeps_raw_formatter(self, tmp_path):
        """The always-on auto log file must NOT run through the console
        sanitizer -- raw bytes matter for debugging and a log file isn't a
        tty that can be corrupted."""
        log_path = tmp_path / "run.log"
        setup_logging(level="INFO", auto_log_file=str(log_path))
        try:
            logger = get_logger("autovideofixer.test_logger_file")
            logger.info("evil\x1b]0;pwn\x07file.mp4")
            for h in logging.getLogger("autovideofixer").handlers:
                h.flush()
            content = log_path.read_text(encoding="utf-8", errors="replace")
            # The raw ESC byte IS present in the file -- sanitization is
            # console-only, by design.
            assert "\x1b" in content
        finally:
            setup_logging(level="INFO")

    def test_end_to_end_console_output_has_no_raw_esc_or_c1(self, capsys):
        """Full regression: a filename containing '\\x1b]0;evil\\x07' and
        '\\x9b31m' (the exact payloads called out in the spec) must not
        leave those raw ESC/C1 sequences reaching what actually gets
        written to the console handler's stream -- exercises the real
        RichHandler/Console pipeline, not just the formatter in isolation.

        Rich itself legitimately emits its own ESC-based ANSI styling (e.g.
        coloring the timestamp/level tag) as part of rendering this same
        log line -- that's deliberate formatting, not smuggled data, so the
        assertion targets the specific injected byte *sequences* rather
        than "no ESC byte anywhere in the line".
        """
        setup_logging(level="INFO")
        try:
            logger = get_logger("autovideofixer.test_logger_e2e")
            logger.info('evil filename: "clip\x1b]0;evil\x07.mp4" c1=\x9b31m')
            for h in logging.getLogger("autovideofixer").handlers:
                h.flush()
        finally:
            setup_logging(level="INFO")

        captured = capsys.readouterr()
        assert "\x1b]0;evil\x07" not in captured.err
        assert "\x9b31m" not in captured.err
        # The replacement marker did appear, proving sanitization actually ran
        # (rather than the assertions above passing because nothing matched).
        assert "�" in captured.err


class TestInsertLogSuffix:
    """REQUIREMENTS.md § 6.7 "both" mode suffix insertion."""

    def test_extension_suffix_inserted_before_extension(self):
        assert _insert_log_suffix("run.log", "-clean") == "run-clean.log"

    def test_extensionless_suffix_appended_to_end(self):
        assert _insert_log_suffix("run", "-clean") == "run-clean"

    def test_empty_suffix_returns_path_unchanged(self):
        assert _insert_log_suffix("run.log", "") == "run.log"

    def test_custom_suffix(self):
        assert _insert_log_suffix("run.log", "_RAW") == "run_RAW.log"

    def test_path_with_directory_and_extension(self):
        assert _insert_log_suffix("/var/log/avf/run.log", "-clean") == "/var/log/avf/run-clean.log"


class TestLogTypeValidation:
    def test_invalid_log_type_raises(self):
        with pytest.raises(ValueError, match="log_type"):
            setup_logging(level="INFO", log_type="verbose")
        setup_logging(level="INFO")  # reset to a clean default state


class TestLogTypeNone:
    def test_none_attaches_no_file_handler(self, tmp_path):
        log_path = tmp_path / "run.log"
        opened = setup_logging(level="INFO", auto_log_file=str(log_path), log_type="none")
        try:
            assert opened == []
            assert not log_path.exists()
            for h in logging.getLogger("autovideofixer").handlers:
                assert not isinstance(h, logging.FileHandler)
        finally:
            setup_logging(level="INFO")


class TestLogTypeRaw:
    def test_raw_returns_single_path_and_is_unredacted(self, tmp_path):
        log_path = tmp_path / "run.log"
        opened = setup_logging(level="INFO", auto_log_file=str(log_path), log_type="raw")
        try:
            assert opened == [str(log_path)]
            logger = get_logger("autovideofixer.test_logtype_raw")
            logger.info("processing /data/in/secret_video.mkv")
            for h in logging.getLogger("autovideofixer").handlers:
                h.flush()
            content = log_path.read_text()
            assert "/data/in/secret_video.mkv" in content
        finally:
            setup_logging(level="INFO")


class TestLogTypeClean:
    def test_clean_file_gets_cleaned_content(self, tmp_path):
        """The clean file's content is redacted while a raw console/handler
        stays raw -- captured here via a temp log file (clean) + an
        in-memory handler attached separately (kept raw)."""
        reset_pii_cleaner()
        get_pii_cleaner().register_input("/data/in/secret_video.mkv")

        log_path = tmp_path / "run.log"
        opened = setup_logging(level="INFO", auto_log_file=str(log_path), log_type="clean")
        try:
            assert opened == [str(log_path)]

            # A separate, always-raw in-memory handler attached directly to
            # the root logger (simulating some other raw sink) must NOT be
            # affected by the clean file formatter.
            import io

            raw_stream = io.StringIO()
            raw_handler = logging.StreamHandler(raw_stream)
            raw_handler.setFormatter(logging.Formatter("%(message)s"))
            root = logging.getLogger("autovideofixer")
            root.addHandler(raw_handler)

            logger = get_logger("autovideofixer.test_logtype_clean")
            logger.info("processing /data/in/secret_video.mkv")
            for h in root.handlers:
                h.flush()

            file_content = log_path.read_text()
            assert "secret_video" not in file_content
            assert "input_video_01.mkv" in file_content

            raw_content = raw_stream.getvalue()
            assert "/data/in/secret_video.mkv" in raw_content

            root.removeHandler(raw_handler)
        finally:
            setup_logging(level="INFO")
            reset_pii_cleaner()

    def test_clean_file_handler_uses_cleaning_formatter(self, tmp_path):
        log_path = tmp_path / "run.log"
        setup_logging(level="INFO", auto_log_file=str(log_path), log_type="clean")
        try:
            root = logging.getLogger("autovideofixer")
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            assert isinstance(file_handlers[0].formatter, _CleaningFileFormatter)
        finally:
            setup_logging(level="INFO")


class TestLogTypeBoth:
    def test_both_mode_writes_two_files(self, tmp_path):
        log_path = tmp_path / "run.log"
        opened = setup_logging(
            level="INFO",
            auto_log_file=str(log_path),
            log_type="both",
            log_suffix_raw="",
            log_suffix_clean="-clean",
        )
        try:
            assert set(opened) == {str(log_path), str(tmp_path / "run-clean.log")}
            assert (tmp_path / "run-clean.log").exists()
            assert log_path.exists()
        finally:
            setup_logging(level="INFO")

    def test_both_mode_raw_file_raw_clean_file_clean(self, tmp_path):
        reset_pii_cleaner()
        get_pii_cleaner().register_input("/data/in/secret_video.mkv")
        log_path = tmp_path / "run.log"
        setup_logging(
            level="INFO",
            auto_log_file=str(log_path),
            log_type="both",
            log_suffix_raw="",
            log_suffix_clean="-clean",
        )
        try:
            logger = get_logger("autovideofixer.test_logtype_both")
            logger.info("processing /data/in/secret_video.mkv")
            for h in logging.getLogger("autovideofixer").handlers:
                h.flush()

            raw_content = log_path.read_text()
            clean_content = (tmp_path / "run-clean.log").read_text()
            assert "secret_video" in raw_content
            assert "secret_video" not in clean_content
            assert "input_video_01.mkv" in clean_content
        finally:
            setup_logging(level="INFO")
            reset_pii_cleaner()

    def test_both_mode_identical_paths_is_a_startup_error(self, tmp_path):
        """If both suffixes are empty (or otherwise equal), that's a config
        error at startup -- never a silent overwrite of one file by the
        other."""
        log_path = tmp_path / "run.log"
        with pytest.raises(ValueError, match="log_type=both"):
            setup_logging(
                level="INFO",
                auto_log_file=str(log_path),
                log_type="both",
                log_suffix_raw="",
                log_suffix_clean="",
            )
        setup_logging(level="INFO")  # reset to a clean default state

    def test_both_mode_custom_equal_suffixes_is_also_an_error(self, tmp_path):
        log_path = tmp_path / "run.log"
        with pytest.raises(ValueError, match="log_type=both"):
            setup_logging(
                level="INFO",
                auto_log_file=str(log_path),
                log_type="both",
                log_suffix_raw="-x",
                log_suffix_clean="-x",
            )
        setup_logging(level="INFO")
