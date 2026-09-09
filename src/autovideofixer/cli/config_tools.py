"""Pure helpers backing `avf config clean|upgrade|dump` (docs/REQUIREMENTS.md § 6.8).

Click wiring for the three subcommands lives in `cli.py` (kept lean per the
project convention -- see AGENTS.md); everything here is plain functions
operating on strings/paths/dicts so it's directly unit-testable without a
CliRunner.
"""

from __future__ import annotations

import io
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


class ConfigUpgradeError(Exception):
    """Raised when an input config has keys absent from the upgrade template.

    `unknown_keys` is the sorted, dot-notation list of offending input leaf
    paths -- either genuinely absent from the template, or present but with
    a type mismatch (dict vs. leaf on either side), which is treated
    identically per REQUIREMENTS.md § 6.8 ("counts as absent/mismatched").
    """

    def __init__(self, unknown_keys: list[str]):
        self.unknown_keys = unknown_keys
        super().__init__(
            "Input config has keys absent from the upgrade template: "
            + ", ".join(unknown_keys)
            + " (pass --drop-unknown to omit them with a warning instead of refusing)"
        )


def clean_config_text(text: str) -> str:
    """Normalize an input config YAML to canonical bare YAML.

    Comments and formatting are stripped; ONLY the keys actually present in
    `text` are emitted (no defaults merged in) and their order is preserved
    (`yaml.safe_load` already preserves mapping key order; `sort_keys=False`
    keeps it on the way back out). Unknown keys pass through untouched --
    this is a formatter, not a validator. Raises `ValueError` if the parsed
    root isn't a mapping.
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        got = type(data).__name__ if data is not None else "empty/None"
        raise ValueError(f"Config root must be a YAML mapping, got {got}")
    # yaml.safe_dump is stubbed as returning Any; it always returns str here
    # (no `stream=` argument), so the annotation is exact.
    return str(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def dump_config_text(data: dict[str, Any]) -> str:
    """Emit an effective-config dict as canonical bare YAML (no key filtering)."""
    return str(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def default_template_text() -> str:
    """Return the packaged `avf config upgrade` default template's text.

    Kept identical to `docs/config.example.yaml` (enforced by a unit test) --
    that repo file is the canonical, human-edited source; this packaged copy
    under `data/` is what ships with the installed package so `upgrade` works
    without a repo checkout present.
    """
    return resources.files("autovideofixer").joinpath("data/config.example.yaml").read_text()


_MISSING = object()


def _walk_leaves(
    d: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten `d` into (path, value) pairs for every non-dict leaf.

    Lists are atomic leaves (never recursed into) per § 6.8's "lists are
    atomic leaves, replaced wholesale" rule.
    """
    out: list[tuple[tuple[str, ...], Any]] = []
    for k, v in d.items():
        path = (*prefix, str(k))
        if isinstance(v, dict):
            out.extend(_walk_leaves(v, path))
        else:
            out.append((path, v))
    return out


def zoom_coverage_migration_note(input_text: str) -> str | None:
    """Return a migration note if `input_text` sets `stages.stabilize.zoom_coverage`,
    or None otherwise.

    REQUIREMENTS.md § 17.2: `zoom_coverage` changed meaning in a BREAKING way
    (old `0.0` = "no zoom" now means "zoom out to preserve everything"; the
    old `0.0` behaviour moved to `0.5`) and `upgrade_config_text()` cannot
    infer intent, so it passes the value through UNCHANGED rather than
    rewriting it -- this note exists purely to surface that fact to the user
    at `avf config upgrade` time, since a silent passthrough of a
    now-differently-meaning value is easy to miss.
    """
    try:
        data = yaml.safe_load(input_text) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    stages = data.get("stages")
    stabilize = stages.get("stabilize") if isinstance(stages, dict) else None
    if not isinstance(stabilize, dict) or "zoom_coverage" not in stabilize:
        return None
    value = stabilize["zoom_coverage"]
    return (
        f"stages.stabilize.zoom_coverage={value!r} carried over UNCHANGED. Its meaning changed "
        'in a breaking way (REQUIREMENTS.md § 17): 0.0 used to mean "no zoom" and now means '
        '"zoom out to preserve everything" -- the old 0.0 behaviour is now 0.5. If this value '
        "was set under the old meaning, update it by hand; upgrade cannot infer your intent."
    )


def upgrade_config_text(
    input_text: str, template_text: str, *, drop_unknown: bool = False
) -> tuple[str, list[str]]:
    """Apply `input_text`'s leaf values onto `template_text`, leaf by leaf.

    Returns (upgraded_yaml_text, warnings) where `warnings` lists the
    dot-notation paths of input keys that don't exist (or type-mismatch) in
    the template -- only ever non-empty when `drop_unknown=True` (otherwise
    those keys raise `ConfigUpgradeError` instead of being returned).

    The template is parsed with ruamel.yaml's round-trip mode so comments,
    key ordering, and any commented-out example entries survive untouched;
    only leaf VALUES the input actually specifies are overwritten in place,
    at the same tree path. An input leaf whose template counterpart is a
    dict (or a template path that doesn't exist as a real key -- e.g. it
    only appears inside a comment) is "absent" and refused/warned exactly
    like a genuinely-missing key.
    """
    input_data = yaml.safe_load(input_text) or {}
    if not isinstance(input_data, dict):
        raise ValueError(
            f"Input config root must be a YAML mapping, got {type(input_data).__name__}"
        )

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    template = yaml_rt.load(template_text)
    if template is None:
        template = CommentedMap()

    unknown: list[str] = []
    for path, value in _walk_leaves(input_data):
        *parents, last = path
        node: Any = template
        ok = True
        for p in parents:
            if isinstance(node, dict) and p in node and isinstance(node[p], dict):
                node = node[p]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and last in node and not isinstance(node[last], dict):
            node[last] = value
        else:
            unknown.append(".".join(path))

    if unknown and not drop_unknown:
        raise ConfigUpgradeError(sorted(unknown))

    out = io.StringIO()
    yaml_rt.dump(template, out)
    return out.getvalue(), sorted(unknown)


def find_first_unused_backup_path(path: Path) -> Path:
    """Return the first unused `PATH.1`, `PATH.2`, ... sibling of `path`."""
    n = 1
    while True:
        candidate = path.with_name(f"{path.name}.{n}")
        if not candidate.exists():
            return candidate
        n += 1


def apply_file_safety(output_path: Path, *, force: bool, backup: bool) -> str | None:
    """Enforce the § 6.8 "never overwrite by default" file-safety rule.

    Called before actually writing `output_path`. Returns a human-readable
    message describing a backup rename that was just performed (the caller
    should print it), or None if nothing needed to happen. Raises
    `ValueError` if `force` and `backup` are both set, or `FileExistsError`
    if `output_path` exists and neither flag was given.
    """
    if force and backup:
        raise ValueError("--force and --backup cannot be used together")
    if output_path.exists():
        if backup:
            renamed = find_first_unused_backup_path(output_path)
            output_path.rename(renamed)
            return f"Renamed existing {output_path} to {renamed}"
        if not force:
            raise FileExistsError(
                f"Output file already exists: {output_path} "
                "(use --force to overwrite or --backup to keep a copy)"
            )
    return None
