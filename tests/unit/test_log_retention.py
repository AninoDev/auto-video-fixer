"""Log retention by run, same-second log names, secret-key matching, relative temp paths."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from autovideofixer.cli.cli import main
from autovideofixer.config import (
    _looks_like_secret_key,
    get_log_dir,
    new_run_log_path,
    prune_old_logs,
    redact_secrets,
    release_unused_run_log,
    resolve_log_retention,
)
from autovideofixer.core.ffmpeg_utils import generate_temp_path


def _run(log_dir: Path, stamp: str, *suffixes: str, mtime: float) -> list[Path]:
    """Create one run's log files (one per suffix) with the given mtime."""
    paths = []
    for suffix in suffixes or ("",):
        p = log_dir / f"avf-{stamp}{suffix}.log"
        p.write_text("x")
        os.utime(p, (mtime, mtime))
        paths.append(p)
    return paths


class TestPruneOldLogs:
    def test_counts_runs_not_files(self, tmp_path):
        # "both" mode: two files per run. Keeping 2 runs must keep 4 files.
        old = _run(tmp_path, "20260101-000000", "-raw", "-clean", mtime=100)
        mid = _run(tmp_path, "20260102-000000", "-raw", "-clean", mtime=200)
        new = _run(tmp_path, "20260103-000000", "-raw", "-clean", mtime=300)
        prune_old_logs(tmp_path, keep_runs=2)
        assert all(not p.exists() for p in old)
        assert all(p.exists() for p in mid + new)

    def test_never_deletes_files_it_did_not_name(self, tmp_path):
        keep = [tmp_path / "notes.txt", tmp_path / "real-run-backup.log", tmp_path / "avf.log"]
        for p in keep:
            p.write_text("mine")
            os.utime(p, (1, 1))
        _run(tmp_path, "20260101-000000", mtime=100)
        _run(tmp_path, "20260102-000000", mtime=200)
        prune_old_logs(tmp_path, keep_runs=1)
        assert all(p.exists() for p in keep)
        assert sorted(p.name for p in tmp_path.glob("avf-*.log")) == ["avf-20260102-000000.log"]

    def test_none_keeps_everything(self, tmp_path):
        runs = [_run(tmp_path, f"2026010{i}-000000", mtime=i) for i in range(1, 8)]
        prune_old_logs(tmp_path, keep_runs=None)
        assert all(p.exists() for run in runs for p in run)

    def test_run_age_is_its_newest_file(self, tmp_path):
        # A run whose clean file was touched late counts as recent.
        a = _run(tmp_path, "20260101-000000", "-raw", mtime=100) + _run(
            tmp_path, "20260101-000000", "-clean", mtime=500
        )
        b = _run(tmp_path, "20260102-000000", mtime=300)
        prune_old_logs(tmp_path, keep_runs=1)
        assert all(p.exists() for p in a)
        assert all(not p.exists() for p in b)

    def test_collision_counter_runs_are_separate_runs(self, tmp_path):
        first = _run(tmp_path, "20260101-000000", mtime=100)
        second = _run(tmp_path, "20260101-000000.2", "-raw", "-clean", mtime=200)
        prune_old_logs(tmp_path, keep_runs=1)
        assert all(not p.exists() for p in first)
        assert all(p.exists() for p in second)

    def test_missing_dir_is_fine(self, tmp_path):
        prune_old_logs(tmp_path / "nope", keep_runs=1)


class TestNewRunLogPath:
    NOW = datetime(2026, 10, 18, 10, 54, 6)

    def test_plain_name_when_free(self, tmp_path):
        assert new_run_log_path(tmp_path, self.NOW).name == "avf-20261018-105406.log"

    @pytest.mark.parametrize(
        "existing", ["avf-20261018-105406.log", "avf-20261018-105406-clean.log"]
    )
    def test_same_second_gets_a_counter(self, tmp_path, existing):
        (tmp_path / existing).write_text("earlier run")
        assert new_run_log_path(tmp_path, self.NOW).name == "avf-20261018-105406.2.log"

    def test_counter_increments(self, tmp_path):
        (tmp_path / "avf-20261018-105406.log").write_text("1")
        (tmp_path / "avf-20261018-105406.2-raw.log").write_text("2")
        assert new_run_log_path(tmp_path, self.NOW).name == "avf-20261018-105406.3.log"


class TestResolveLogRetention:
    @pytest.mark.parametrize("value", [None, 0])
    def test_unlimited(self, value):
        assert resolve_log_retention(value) is None

    def test_positive(self):
        assert resolve_log_retention(7) == 7

    @pytest.mark.parametrize("value", [-1, 2.5, "50", True])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="log_retention_runs"):
            resolve_log_retention(value)


class TestCliRetention:
    def _invoke(self, tmp_path, retention_yaml: str):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(f"config_version: 2\ngeneral:\n  log_retention_runs: {retention_yaml}\n")
        return CliRunner().invoke(main, ["--config", str(cfg), "presets"])

    def test_unlimited_config_keeps_old_runs(self, tmp_path):
        log_dir = get_log_dir()  # isolated per test by conftest
        log_dir.mkdir(parents=True, exist_ok=True)
        old = [_run(log_dir, f"2020010{i}-000000", mtime=i) for i in range(1, 6)]
        result = self._invoke(tmp_path, "null")
        assert result.exit_code == 0, result.output
        assert all(p.exists() for run in old for p in run)

    def test_limit_applies(self, tmp_path):
        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        old = [_run(log_dir, f"2020010{i}-000000", mtime=i) for i in range(1, 6)]
        result = self._invoke(tmp_path, "2")
        assert result.exit_code == 0, result.output
        survivors = [p for run in old for p in run if p.exists()]
        assert [p.name for p in survivors] == [
            "avf-20200104-000000.log",
            "avf-20200105-000000.log",
        ]

    def test_invalid_value_exits_nonzero(self, tmp_path):
        result = self._invoke(tmp_path, "-3")
        assert result.exit_code == 1
        assert "log_retention_runs" in result.output


class TestSecretKeyMatching:
    @pytest.mark.parametrize(
        "key",
        ["api_key", "API_KEY", "apikey", "password", "client_secret", "token", "access_token",
         "hf_token", "authToken", "auth-token", "refresh.token"],
    )  # fmt: skip
    def test_secrets(self, key):
        assert _looks_like_secret_key(key)

    @pytest.mark.parametrize(
        "key", ["max_tokens", "tokens_per_tile", "max_tiles_per_call", "tokenizer", "timeout_sec"]
    )
    def test_not_secrets(self, key):
        assert not _looks_like_secret_key(key)

    def test_max_tokens_not_redacted(self):
        out = redact_secrets({"endpoints": {"x": {"max_tokens": 2048, "api_key": "sk-1"}}})
        assert out["endpoints"]["x"] == {"max_tokens": 2048, "api_key": "***"}


def test_generate_temp_path_bare_relative_filename(tmp_path, monkeypatch):
    # `avf process clip.mp4` from the clip's own directory: dirname("clip.mp4") == "".
    monkeypatch.chdir(tmp_path)
    path = generate_temp_path("", "clip.mp4")
    assert os.path.dirname(path) == str(tmp_path)


class TestRunLogReservation:
    def test_reservation_creates_the_file(self, tmp_path):
        path = new_run_log_path(tmp_path, TestNewRunLogPath.NOW)
        assert path.exists() and path.stat().st_size == 0
        # A second caller in the same second sees it taken.
        assert new_run_log_path(tmp_path, TestNewRunLogPath.NOW).name.endswith(".2.log")

    def test_concurrent_reservations_never_share_a_name(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as pool:
            names = list(
                pool.map(
                    lambda _: new_run_log_path(tmp_path, TestNewRunLogPath.NOW).name, range(16)
                )
            )
        assert len(set(names)) == 16

    def test_release_removes_unused_empty_reservation(self, tmp_path):
        path = new_run_log_path(tmp_path, TestNewRunLogPath.NOW)
        release_unused_run_log(path, [str(tmp_path / "avf-20261018-105406-raw.log")])
        assert not path.exists()

    def test_release_keeps_used_reservation(self, tmp_path):
        path = new_run_log_path(tmp_path, TestNewRunLogPath.NOW)
        release_unused_run_log(path, [str(path)])
        assert path.exists()
