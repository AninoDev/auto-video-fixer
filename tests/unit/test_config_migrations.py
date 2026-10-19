"""Config schema versioning, migrations, load policy and bootstrap (REQUIREMENTS.md § 24.1-24.3)."""

import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from autovideofixer import config_migrations as cm
from autovideofixer.cli.cli import main
from autovideofixer.config import BOOTSTRAP_OPT_OUT_ENV, Config, get_config_path

V1_COMMENTED = """\
# my header comment
general:
  output_container: mkv   # keep mkv

pipeline:
  # my order
  default_order:
    - detect
    - retime
    - stage: deblock
      enabled: true
    - encode

stages:
  crop:
    # I do not want crop
    enabled: false

analysis:
  llm:
    provider: local
    model: llama3
    api_url: "http://127.0.0.1:1"
  vlm:
    enabled: true
    # connection below
    provider: local
    model: llava
    api_key: "sk-secret"
    api_url: "http://127.0.0.1:2"
    allow_http: true
    max_tokens: 777
    max_sample_frames: 3
    sample_interval_sec: 4.5
    prompt_append: "hello"
"""


class TestMigrationItems:
    def test_crop_flipped(self):
        doc = {"stages": {"crop": {"enabled": False}}}
        notes = cm.migrate(doc)
        assert doc["stages"]["crop"]["enabled"] is True
        assert any("crop" in n for n in notes)

    def test_crop_true_or_absent_untouched(self):
        doc = {"stages": {"crop": {"limit": 5}}}
        cm.migrate(doc)
        assert "enabled" not in doc["stages"]["crop"]

    @pytest.mark.parametrize(
        "order,expected",
        [
            (["detect", "retime", "encode"], ["detect", "retime", "normalize_codecs", "encode"]),
            (["detect", "encode"], ["detect", "normalize_codecs", "encode"]),
            (["encode"], ["normalize_codecs", "encode"]),
            (
                ["detect", {"stage": "retime", "enabled": True}, "encode"],
                ["detect", {"stage": "retime", "enabled": True}, "normalize_codecs", "encode"],
            ),
            (
                ["detect", {"stage": "normalize_codecs"}, "encode"],
                ["detect", {"stage": "normalize_codecs"}, "encode"],
            ),
        ],
    )
    def test_default_order_insert(self, order, expected):
        doc = {"pipeline": {"default_order": list(order)}}
        cm.migrate(doc, target=2)  # the v1 -> v2 item; v2 -> v3 is tested separately
        assert doc["pipeline"]["default_order"] == expected

    def test_absent_default_order_stays_absent(self):
        doc = {"general": {"output_dir": None}}
        cm.migrate(doc)
        assert "pipeline" not in doc

    def test_connection_fields_moved(self):
        doc = yaml.safe_load(V1_COMMENTED)
        notes = cm.migrate(doc)
        ep = doc["analysis"]["endpoints"]["default_vlm"]
        assert ep == {
            "provider": "ollama",  # local normalised
            "model": "llava",
            "api_key": "sk-secret",
            "api_url": "http://127.0.0.1:2",
            "allow_http": True,
            "max_tokens": 777,
        }
        assert doc["analysis"]["endpoints"]["default_llm"]["provider"] == "ollama"
        assert doc["analysis"]["vlm"]["endpoint"] == "default_vlm"
        assert doc["analysis"]["llm"]["endpoint"] == "default_llm"
        for k in cm.ENDPOINT_FIELDS:
            assert k not in doc["analysis"]["vlm"]
        assert doc["analysis"]["vlm"]["prompt_append"] == "hello"
        assert any("local" in n for n in notes)

    def test_sampling_renamed(self):
        doc = yaml.safe_load(V1_COMMENTED)
        cm.migrate(doc)
        vlm = doc["analysis"]["vlm"]
        assert vlm["sampling"] == {"max_frames": 3, "interval_sec": 4.5}
        assert "max_sample_frames" not in vlm and "sample_interval_sec" not in vlm

    def test_only_present_keys_invented(self):
        doc = {"analysis": {"vlm": {"model": "x"}}}
        cm.migrate(doc)
        assert doc["analysis"]["endpoints"] == {"default_vlm": {"model": "x"}}
        assert "default_llm" not in doc["analysis"]["endpoints"]
        assert "sampling" not in doc["analysis"]["vlm"]
        assert "llm" not in doc["analysis"]

    def test_idempotent(self):
        doc = yaml.safe_load(V1_COMMENTED)
        cm.migrate(doc)
        snapshot = yaml.safe_dump(doc)
        assert cm.migrate(doc) == []
        assert yaml.safe_dump(doc) == snapshot
        assert doc["config_version"] == 3

    def test_newer_version_untouched(self):
        doc = {"config_version": 9, "stages": {"crop": {"enabled": False}}}
        assert cm.migrate(doc) == []
        assert doc["stages"]["crop"]["enabled"] is False


class TestTextRoundTrip:
    def test_comments_survive_and_content_matches(self):
        new_text, old, notes = cm.migrate_text(V1_COMMENTED)
        assert old == 1 and notes
        for comment in ("# my header comment", "# keep mkv", "# my order", "# I do not want crop"):
            assert comment in new_text
        data = yaml.safe_load(new_text)
        plain = yaml.safe_load(V1_COMMENTED)
        cm.migrate(plain)
        assert data == plain
        assert data["config_version"] == 3
        assert data["pipeline"]["default_order"][2] == "normalize_codecs"

    def test_second_pass_is_noop(self):
        once, _, _ = cm.migrate_text(V1_COMMENTED)
        twice, old, notes = cm.migrate_text(once)
        assert (twice, old, notes) == (once, 3, [])

    def test_empty_text_untouched(self):
        assert cm.migrate_text("# only a comment\n") == ("# only a comment\n", 1, [])


class TestLayerMigration:
    def test_layer_old_shape_targets_selected_endpoint_without_selector(self):
        layer = {"analysis": {"vlm": {"model": "m2", "max_sample_frames": 2}}}
        notes = cm.migrate_layer(layer, vlm_endpoint="mine")
        assert layer["analysis"]["endpoints"]["mine"] == {"model": "m2"}
        assert "endpoint" not in layer["analysis"]["vlm"]
        assert layer["analysis"]["vlm"]["sampling"]["max_frames"] == 2
        assert notes

    def test_layer_does_not_flip_crop(self):
        layer = {"stages": {"crop": {"enabled": False}}}
        assert cm.migrate_layer(layer) == []
        assert layer["stages"]["crop"]["enabled"] is False

    def test_layer_declaring_v1_gets_full_migration(self):
        layer = {"config_version": 1, "stages": {"crop": {"enabled": False}}}
        cm.migrate_layer(layer)
        assert layer["stages"]["crop"]["enabled"] is True
        assert "config_version" not in layer

    def test_apply_layer_migrates_and_warns(self, tmp_path, caplog):
        cfg = Config(tmp_path / "none.yaml")
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg.apply_layer({"analysis": {"vlm": {"model": "layer-model"}}}, "cli-flags")
        assert cfg.get("analysis", "endpoints", "default_vlm", "model") == "layer-model"
        assert any("old (schema v1)" in r.message for r in caplog.records)


class TestLoadPolicy:
    def _write(self, d: Path, text=V1_COMMENTED) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        p = d / "config.yaml"
        p.write_text(text)
        return p

    def test_default_path_rewritten_in_place_with_backup(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        p = self._write(get_config_path().parent)
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg = Config()
        backups = list(p.parent.glob("config.yaml.bak-v1-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == V1_COMMENTED
        assert yaml.safe_load(p.read_text())["config_version"] == 3
        assert "# keep mkv" in p.read_text()
        assert cfg.get("analysis", "endpoints", "default_vlm", "api_url") == "http://127.0.0.1:2"
        assert cfg.get("stages", "crop", "enabled") is True
        msg = " ".join(r.message for r in caplog.records)
        assert str(backups[0]) in msg and "crop" in msg
        # second load: no further migration or backup
        Config()
        assert len(list(p.parent.glob("config.yaml.bak-*"))) == 1

    def test_explicit_config_in_memory_only(self, tmp_path, caplog):
        p = self._write(tmp_path / "x")
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg = Config(p, require_exists=True)
        assert p.read_text() == V1_COMMENTED
        assert not list(p.parent.glob("*.bak-*"))
        assert cfg.get("analysis", "vlm", "endpoint") == "default_vlm"
        assert any("avf config migrate" in r.message for r in caplog.records)

    def test_newer_version_warns_and_loads(self, tmp_path, caplog):
        p = self._write(tmp_path, "config_version: 99\ngeneral:\n  overwrite: true\n")
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg = Config(p)
        assert cfg.get("general", "overwrite") is True
        assert any("newer than this AVF" in r.message for r in caplog.records)
        assert p.read_text().startswith("config_version: 99")

    def test_deferred_notices_emitted_later(self, tmp_path, caplog):
        p = self._write(tmp_path)
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg = Config(p, defer_notices=True)
            assert not caplog.records
            cfg.emit_notices()
        assert caplog.records

    def test_unwritable_default_falls_back_in_memory(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        p = self._write(get_config_path().parent)
        p.parent.chmod(0o500)
        try:
            with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
                cfg = Config()
        finally:
            p.parent.chmod(0o700)
        assert cfg.get("stages", "crop", "enabled") is True
        assert p.read_text() == V1_COMMENTED


class TestBootstrap:
    def test_bootstrap_creates_default(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv(BOOTSTRAP_OPT_OUT_ENV)
        with caplog.at_level(logging.INFO, logger="autovideofixer.config"):
            cfg = Config()
        p = get_config_path()
        assert p.exists()
        assert yaml.safe_load(p.read_text())["config_version"] == 3
        assert cfg.get("config_version") == 3
        assert any("First run" in r.message for r in caplog.records)
        assert not list(p.parent.glob("*.bak-*"))
        mtime = p.stat().st_mtime_ns
        Config()
        assert p.stat().st_mtime_ns == mtime

    def test_bootstrapped_file_equals_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv(BOOTSTRAP_OPT_OUT_ENV)
        assert Config().data == Config(tmp_path / "absent.yaml").data

    def test_opt_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv(BOOTSTRAP_OPT_OUT_ENV, "1")
        Config()
        assert not get_config_path().exists()

    def test_explicit_path_never_bootstrapped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
        monkeypatch.delenv(BOOTSTRAP_OPT_OUT_ENV)
        Config(tmp_path / "other.yaml")
        assert not (tmp_path / "other.yaml").exists()
        assert not get_config_path().exists()

    def test_failure_is_warning_and_continues(self, tmp_path, monkeypatch, caplog):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(blocker))  # parent is a file -> mkdir fails
        monkeypatch.delenv(BOOTSTRAP_OPT_OUT_ENV)
        with caplog.at_level(logging.WARNING, logger="autovideofixer.config"):
            cfg = Config()
        assert cfg.get("general", "output_container") == "mp4"
        assert any("Could not create default config" in r.message for r in caplog.records)

    def test_example_matches_defaults(self):
        from autovideofixer.cli import config_tools

        example = yaml.safe_load(config_tools.default_template_text())
        assert example == Config.DEFAULTS


class TestConfigMigrateCli:
    def test_in_place_with_backup(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(V1_COMMENTED)
        r = CliRunner().invoke(main, ["config", "migrate", str(p)])
        assert r.exit_code == 0, r.output
        assert len(list(tmp_path.glob("c.yaml.bak-v1-*"))) == 1
        assert yaml.safe_load(p.read_text())["config_version"] == 3
        assert "# keep mkv" in p.read_text()
        r2 = CliRunner().invoke(main, ["config", "migrate", str(p)])
        assert r2.exit_code == 0
        assert len(list(tmp_path.glob("c.yaml.bak-*"))) == 1

    def test_no_backup(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(V1_COMMENTED)
        r = CliRunner().invoke(main, ["config", "migrate", str(p), "--no-backup"])
        assert r.exit_code == 0
        assert not list(tmp_path.glob("*.bak-*"))

    def test_output_leaves_input(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(V1_COMMENTED)
        out = tmp_path / "out.yaml"
        r = CliRunner().invoke(main, ["config", "migrate", str(p), "-o", str(out)])
        assert r.exit_code == 0
        assert p.read_text() == V1_COMMENTED
        assert yaml.safe_load(out.read_text())["config_version"] == 3
        r2 = CliRunner().invoke(main, ["config", "migrate", str(p), "-o", str(out)])
        assert r2.exit_code == 1  # refuses to clobber without --force

    def test_upgrade_runs_migrations_first(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(V1_COMMENTED)
        r = CliRunner().invoke(main, ["config", "upgrade", str(p)])
        assert r.exit_code == 0, r.output
        data = yaml.safe_load(r.stdout)
        assert data["analysis"]["endpoints"]["default_vlm"]["model"] == "llava"
        assert data["analysis"]["vlm"]["sampling"]["max_frames"] == 3
        assert data["config_version"] == 3

    def test_upgrade_keeps_custom_endpoints(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(
            "config_version: 2\nanalysis:\n  endpoints:\n    mine:\n      provider: llamacpp\n"
            "      compat:\n        extra_body:\n          cache_prompt: true\n"
            "      tls:\n        mode: insecure\n"
        )
        r = CliRunner().invoke(main, ["config", "upgrade", str(p)])
        assert r.exit_code == 0, r.output
        mine = yaml.safe_load(r.stdout)["analysis"]["endpoints"]["mine"]
        assert mine["provider"] == "llamacpp"
        assert mine["compat"]["extra_body"] == {"cache_prompt": True}
        assert mine["tls"] == {"mode": "insecure"}

    def test_dump_has_version(self):
        r = CliRunner().invoke(main, ["config", "dump"])
        assert r.exit_code == 0
        assert yaml.safe_load(r.stdout)["config_version"] == 3
