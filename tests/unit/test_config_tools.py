"""Tests for `avf config clean|dump|upgrade` (REQUIREMENTS.md § 6.8).

Most coverage is at the pure-helper level in ``cli/config_tools.py`` (fast and
deterministic); a few ``CliRunner`` tests exercise the Click wiring and the
stdout-is-data / stderr-is-diagnostics contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from autovideofixer.cli import config_tools
from autovideofixer.cli.cli import main
from autovideofixer.config import Config

# --- clean -----------------------------------------------------------------


class TestCleanConfigText:
    def test_strips_comments_and_keeps_only_present_keys(self):
        src = """
# a header comment
general:
  overwrite: true   # inline comment
  log_type: clean
"""
        out = config_tools.clean_config_text(src)
        assert "#" not in out  # comments gone
        data = yaml.safe_load(out)
        assert data == {"general": {"overwrite": True, "log_type": "clean"}}
        # No DEFAULTS merged in -- only the two keys the input had.
        assert set(data["general"]) == {"overwrite", "log_type"}

    def test_preserves_key_order(self):
        src = "z_key: 1\na_key: 2\nm_key: 3\n"
        out = config_tools.clean_config_text(src)
        assert list(yaml.safe_load(out)) == ["z_key", "a_key", "m_key"]

    def test_unknown_keys_pass_through(self):
        # clean is a formatter, not a validator.
        out = config_tools.clean_config_text("totally_made_up_key: 42\n")
        assert yaml.safe_load(out) == {"totally_made_up_key": 42}

    def test_non_mapping_root_rejected(self):
        with pytest.raises(ValueError):
            config_tools.clean_config_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError):
            config_tools.clean_config_text("42\n")


# --- dump ------------------------------------------------------------------


class TestDumpConfigText:
    def test_emits_valid_yaml(self):
        out = config_tools.dump_config_text({"general": {"overwrite": True}})
        assert yaml.safe_load(out) == {"general": {"overwrite": True}}


# --- upgrade ---------------------------------------------------------------

_TEMPLATE = """\
# Top-of-file comment (must survive)
general:
  # keep this option comment
  overwrite: false
  log_type: "raw"
pipeline:
  default_order:
  - detect
  - encode
"""


class TestUpgradeConfigText:
    def test_replaces_leaves_in_place_with_comments_intact(self):
        src = "general:\n  overwrite: true\n"
        out, warnings = config_tools.upgrade_config_text(src, _TEMPLATE)
        assert warnings == []
        assert "# Top-of-file comment (must survive)" in out
        assert "# keep this option comment" in out
        data = yaml.safe_load(out)
        assert data["general"]["overwrite"] is True
        # untouched template leaves keep their defaults
        assert data["general"]["log_type"] == "raw"

    def test_list_replaced_wholesale(self):
        src = "pipeline:\n  default_order: [crop, hdr, encode]\n"
        out, _ = config_tools.upgrade_config_text(src, _TEMPLATE)
        assert yaml.safe_load(out)["pipeline"]["default_order"] == ["crop", "hdr", "encode"]

    def test_unknown_key_refused(self):
        src = "general:\n  renamed_option: 5\n"
        with pytest.raises(config_tools.ConfigUpgradeError) as exc:
            config_tools.upgrade_config_text(src, _TEMPLATE)
        assert exc.value.unknown_keys == ["general.renamed_option"]

    def test_unknown_section_refused(self):
        src = "made_up_section:\n  foo: bar\n"
        with pytest.raises(config_tools.ConfigUpgradeError) as exc:
            config_tools.upgrade_config_text(src, _TEMPLATE)
        assert exc.value.unknown_keys == ["made_up_section.foo"]

    def test_drop_unknown_proceeds_with_warnings(self):
        src = "general:\n  overwrite: true\n  renamed_option: 5\n"
        out, warnings = config_tools.upgrade_config_text(src, _TEMPLATE, drop_unknown=True)
        assert warnings == ["general.renamed_option"]
        # the known leaf still got applied; the unknown one is absent
        data = yaml.safe_load(out)
        assert data["general"]["overwrite"] is True
        assert "renamed_option" not in data["general"]

    def test_type_mismatch_dict_vs_leaf_refused(self):
        # input makes `overwrite` a mapping, but the template has it as a leaf
        src = "general:\n  overwrite:\n    nested: 1\n"
        with pytest.raises(config_tools.ConfigUpgradeError) as exc:
            config_tools.upgrade_config_text(src, _TEMPLATE)
        assert exc.value.unknown_keys == ["general.overwrite.nested"]


# --- packaged template vs. canonical source --------------------------------


class TestPackagedTemplate:
    def test_packaged_template_matches_repo_docs(self):
        # The shipped copy must not drift from the human-edited canonical file.
        repo_docs = (
            Path(__file__).resolve().parents[2] / "docs" / "config.example.yaml"
        ).read_text()
        assert config_tools.default_template_text() == repo_docs

    def test_default_template_is_upgradeable_against_real_config(self):
        # A real DEFAULTS-shaped config's leaves must all exist in the template
        # (otherwise a from-scratch upgrade would spuriously refuse).
        defaults_yaml = yaml.safe_dump(Config.DEFAULTS)
        # Should not raise -- every DEFAULTS leaf path exists in the template.
        config_tools.upgrade_config_text(defaults_yaml, config_tools.default_template_text())


# --- file safety -----------------------------------------------------------


class TestFileSafety:
    def test_new_destination_no_message(self, tmp_path):
        assert (
            config_tools.apply_file_safety(tmp_path / "new.yaml", force=False, backup=False) is None
        )

    def test_existing_refused_by_default(self, tmp_path):
        p = tmp_path / "e.yaml"
        p.write_text("x")
        with pytest.raises(FileExistsError):
            config_tools.apply_file_safety(p, force=False, backup=False)

    def test_force_allows_overwrite(self, tmp_path):
        p = tmp_path / "e.yaml"
        p.write_text("x")
        assert config_tools.apply_file_safety(p, force=True, backup=False) is None

    def test_backup_renames_incrementally(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("first")
        msg1 = config_tools.apply_file_safety(p, force=False, backup=True)
        assert (tmp_path / "c.yaml.1").read_text() == "first"
        assert "c.yaml.1" in msg1
        p.write_text("second")
        msg2 = config_tools.apply_file_safety(p, force=False, backup=True)
        assert (tmp_path / "c.yaml.2").read_text() == "second"
        assert "c.yaml.2" in msg2

    def test_force_and_backup_conflict(self, tmp_path):
        with pytest.raises(ValueError):
            config_tools.apply_file_safety(tmp_path / "x.yaml", force=True, backup=True)


# --- CLI wiring ------------------------------------------------------------


class TestConfigCliWiring:
    def setup_method(self):
        # Click >= 8.2 always captures stderr separately (result.stderr); the
        # old mix_stderr kwarg was removed.
        self.runner = CliRunner()

    def test_clean_stdout_is_pure_yaml(self, tmp_path):
        src = tmp_path / "in.yaml"
        src.write_text("# comment\ngeneral:\n  overwrite: true\n")
        result = self.runner.invoke(main, ["config", "clean", str(src)])
        assert result.exit_code == 0
        assert yaml.safe_load(result.stdout) == {"general": {"overwrite": True}}

    def test_upgrade_drop_unknown_warnings_go_to_stderr_not_stdout(self, tmp_path):
        # The stdout stream must stay clean YAML so `... > out.yaml` isn't
        # corrupted by warning text.
        src = tmp_path / "in.yaml"
        src.write_text("general:\n  overwrite: true\n  bogus_key: 1\n")
        result = self.runner.invoke(main, ["config", "upgrade", str(src), "--drop-unknown"])
        assert result.exit_code == 0
        assert "Warning" not in result.stdout
        assert "Warning" in result.stderr
        assert yaml.safe_load(result.stdout)["general"]["overwrite"] is True

    def test_upgrade_unknown_key_refused_exit_1(self, tmp_path):
        src = tmp_path / "in.yaml"
        src.write_text("general:\n  bogus_key: 1\n")
        result = self.runner.invoke(main, ["config", "upgrade", str(src)])
        assert result.exit_code == 1
        assert "bogus_key" in result.stderr

    def test_existing_output_refused_exit_1(self, tmp_path):
        src = tmp_path / "in.yaml"
        src.write_text("general:\n  overwrite: true\n")
        dest = tmp_path / "out.yaml"
        dest.write_text("existing")
        result = self.runner.invoke(main, ["config", "clean", str(src), "-o", str(dest)])
        assert result.exit_code == 1
        assert dest.read_text() == "existing"  # untouched
