"""Config schema versioning and migrations (docs/REQUIREMENTS.md § 24.1/24.2).

A config file carries a top-level ``config_version`` integer; a file without one is
version 1 (every file written before this module existed). ``CURRENT_CONFIG_VERSION`` is
what this AVF release writes and understands.

Why migrations exist at all: the bootstrapped/example config *pins every key*, so once a
user has such a file, a changed ``Config.DEFAULTS`` value never reaches them (the
``denoise_video`` and ``retime`` default flips already hit this). **Standing rule: any
future default change that should reach existing users needs a migration here that
rewrites values equal to the old default.** (Also stated in AGENTS.md.)

Each migration is ``fn(doc, notes) -> None`` operating IN PLACE on either a ruamel
``CommentedMap`` (the file path -- comments, key order and quoting survive) or a plain
``dict`` (in-memory path for explicit configs and cascade layers). It appends a
human-readable string to ``notes`` for every change it makes, and must be idempotent and
must not invent keys the user did not have beyond what the spec says -- an absent key
stays absent so ``Config.DEFAULTS`` keeps applying.

This module is pure (no logging, no Config import) so it is trivially unit-testable;
the load-time policy (backup / in-place rewrite / in-memory only / warnings) lives in
``config.py``, and ``avf config migrate`` in ``cli/cli.py`` + ``cli/config_tools.py``.
"""

from __future__ import annotations

import io
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

CURRENT_CONFIG_VERSION = 2

# v1 stored the connection fields directly on analysis.vlm / analysis.llm; v2 moves them
# into analysis.endpoints.<name> (REQUIREMENTS.md § 24.4).
ENDPOINT_FIELDS = ("provider", "model", "api_key", "api_url", "allow_http", "max_tokens")

Migration = Callable[[MutableMapping[str, Any], list[str]], None]


def get_config_version(doc: Any) -> int:
    """Return the schema version of a parsed config (absent -> 1).

    A non-integer ``config_version`` is treated as 1 (it cannot be trusted, and
    re-running the idempotent migrations on a file that is really newer is harmless
    compared with refusing to start).
    """
    if not isinstance(doc, MutableMapping):
        return 1
    value = doc.get("config_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _new_map(parent: Any) -> Any:
    """A new empty mapping of the same flavour as ``parent`` (CommentedMap or dict)."""
    return CommentedMap() if isinstance(parent, CommentedMap) else {}


def _get_map(parent: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any] | None:
    child = parent.get(key)
    return child if isinstance(child, MutableMapping) else None


def _ensure_map(parent: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    child = parent.get(key)
    if isinstance(child, MutableMapping):
        return child
    created: MutableMapping[str, Any] = _new_map(parent)
    parent[key] = created
    return created


# --------------------------------------------------------------------------------------
# v1 -> v2 items
# --------------------------------------------------------------------------------------


def _flip_crop_enabled(doc: MutableMapping[str, Any], notes: list[str]) -> None:
    """§ 24.2 item 1: ``stages.crop.enabled: false`` -> ``true`` (the new default, § 22.1)."""
    stages = _get_map(doc, "stages")
    crop = _get_map(stages, "crop") if stages is not None else None
    if crop is not None and crop.get("enabled") is False:
        crop["enabled"] = True
        notes.append(
            "stages.crop.enabled: false -> true (crop is now on by default; a deliberate "
            "false is flipped once too -- set it back to false after migration if you really "
            "want it off)"
        )


def _entry_stage_name(entry: Any) -> str | None:
    """Stage name of a ``pipeline.default_order`` entry (string, or ``{stage: name, ...}``)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, MutableMapping):
        name = entry.get("stage")
        return name if isinstance(name, str) else None
    return None


def _insert_normalize_codecs(doc: MutableMapping[str, Any], notes: list[str]) -> None:
    """§ 24.2 item 2: insert ``normalize_codecs`` into a pinned ``default_order`` lacking it.

    Placed after ``retime`` (its position in the built-in order, § 22.2), else after
    ``detect``, else at the front. Mapping-form entries (``- stage: deblock``) are
    recognised both when checking for presence and when locating the anchor.
    """
    pipeline = _get_map(doc, "pipeline")
    order = pipeline.get("default_order") if pipeline is not None else None
    if not isinstance(order, list):
        return
    names = [_entry_stage_name(e) for e in order]
    if "normalize_codecs" in names:
        return
    for anchor in ("retime", "detect"):
        if anchor in names:
            # Last occurrence: a custom order might repeat a stage; heavy stages come
            # after the final one, which is where the normalised stream must exist.
            idx = len(names) - 1 - names[::-1].index(anchor) + 1
            where = f"after {anchor!r}"
            break
    else:
        idx = 0
        where = "at the front"
    order.insert(idx, "normalize_codecs")
    notes.append(f"pipeline.default_order: inserted 'normalize_codecs' {where}")


def _move_endpoint(
    analysis: MutableMapping[str, Any],
    section: str,
    default_name: str,
    notes: list[str],
    *,
    layer: bool,
) -> None:
    """Move one of ``analysis.vlm`` / ``analysis.llm``'s connection fields into an endpoint.

    ``layer=True`` (cascade layers): target the endpoint the layer itself names, else
    ``default_name`` -- which the caller sets to the endpoint currently *selected* by the base
    config, so a ``--set analysis.vlm.model=x`` lands on the endpoint actually in use -- and
    never write the ``endpoint`` selector (that would silently re-point the run).
    """
    sec = _get_map(analysis, section)
    if sec is None:
        return
    present = [k for k in ENDPOINT_FIELDS if k in sec]
    if not present and (layer or "endpoint" in sec):
        # Nothing connection-related to move -> leave the section (and thus the
        # DEFAULTS' endpoint reference) alone.
        return
    chosen = sec.get("endpoint")
    name = chosen if isinstance(chosen, str) and chosen else default_name
    endpoints = _ensure_map(analysis, "endpoints")
    ep = _ensure_map(endpoints, name)
    moved: list[str] = []
    for key in present:
        value = sec[key]
        if key == "provider" and value == "local":
            value = "ollama"
            notes.append(
                f"analysis.{section}.provider 'local' normalised to 'ollama' (they were aliases)"
            )
        if key not in ep:
            ep[key] = value
        # else: an endpoint entry already carries it (hand-edited half-migrated file) --
        # it wins; the stale legacy key is still dropped below.
        del sec[key]
        moved.append(key)
    if not layer and "endpoint" not in sec:
        sec["endpoint"] = name
    if moved:
        notes.append(
            f"analysis.{section}.{{{', '.join(moved)}}} moved to analysis.endpoints.{name}"
            + ("" if layer else f"; analysis.{section}.endpoint = {name}")
        )
    else:
        notes.append(f"analysis.{section}.endpoint set to {name}")


def _move_connection_fields(
    doc: MutableMapping[str, Any],
    notes: list[str],
    *,
    layer: bool = False,
    vlm_endpoint: str = "default_vlm",
    llm_endpoint: str = "default_llm",
) -> None:
    """§ 24.2 item 3."""
    analysis = _get_map(doc, "analysis")
    if analysis is None:
        return
    _move_endpoint(analysis, "vlm", vlm_endpoint, notes, layer=layer)
    _move_endpoint(analysis, "llm", llm_endpoint, notes, layer=layer)


def _rename_sampling_keys(doc: MutableMapping[str, Any], notes: list[str]) -> None:
    """§ 24.2 item 4: ``max_sample_frames`` / ``sample_interval_sec`` -> ``sampling.*``."""
    analysis = _get_map(doc, "analysis")
    vlm = _get_map(analysis, "vlm") if analysis is not None else None
    if vlm is None:
        return
    for old, new in (("max_sample_frames", "max_frames"), ("sample_interval_sec", "interval_sec")):
        if old not in vlm:
            continue
        sampling = _ensure_map(vlm, "sampling")
        if new not in sampling:
            sampling[new] = vlm[old]
        del vlm[old]
        notes.append(f"analysis.vlm.{old} renamed to analysis.vlm.sampling.{new}")


def shape_migrate_analysis(
    doc: MutableMapping[str, Any],
    notes: list[str],
    *,
    layer: bool = False,
    vlm_endpoint: str = "default_vlm",
    llm_endpoint: str = "default_llm",
) -> None:
    """Items 3 and 4 only: the *shape* changes that make an old-style ``analysis`` block work.

    Used on cascade layers (``process --config`` without a ``config_version``, presets,
    ``--set``) -- those are partial overlays, so the default-flipping items (1, 2) must NOT
    run on them: a layer saying ``crop.enabled: false`` is a deliberate per-run choice, not a
    stale pinned default.
    """
    _move_connection_fields(
        doc, notes, layer=layer, vlm_endpoint=vlm_endpoint, llm_endpoint=llm_endpoint
    )
    _rename_sampling_keys(doc, notes)


def _migrate_v1_to_v2(doc: MutableMapping[str, Any], notes: list[str]) -> None:
    _flip_crop_enabled(doc, notes)
    _insert_normalize_codecs(doc, notes)
    shape_migrate_analysis(doc, notes)


# Ordered registry: key = the version a migration upgrades FROM.
MIGRATIONS: dict[int, Migration] = {1: _migrate_v1_to_v2}


def _stamp_version(doc: MutableMapping[str, Any], version: int) -> None:
    if "config_version" in doc:
        doc["config_version"] = version
    elif isinstance(doc, CommentedMap):
        # Front of the file (after the header comment, which ruamel keeps on the map).
        doc.insert(0, "config_version", version)
    else:
        doc["config_version"] = version


def migrate(doc: MutableMapping[str, Any], *, target: int = CURRENT_CONFIG_VERSION) -> list[str]:
    """Run every needed migration on ``doc`` in place; return the change notes.

    A document already at (or newer than) ``target`` is left untouched and yields no
    notes. Idempotent: running it again on its own output is a no-op.
    """
    version = get_config_version(doc)
    notes: list[str] = []
    if version >= target:
        return notes
    while version < target:
        step = MIGRATIONS.get(version)
        if step is None:  # pragma: no cover - registry gap is a programming error
            raise RuntimeError(f"No config migration registered from version {version}")
        step(doc, notes)
        version += 1
        _stamp_version(doc, version)
    return notes


def migrate_layer(
    layer: dict[str, Any],
    *,
    vlm_endpoint: str = "default_vlm",
    llm_endpoint: str = "default_llm",
) -> list[str]:
    """Bring a cascade layer (plain dict) up to date, in place; return change notes.

    A layer that declares ``config_version`` below current is treated as a full old config
    and gets the complete migration; otherwise only the shape items (3, 4) apply. The
    ``config_version`` key is removed from the layer either way (the base config owns it).
    ``vlm_endpoint`` / ``llm_endpoint`` are the endpoints the base config currently selects
    (old-style connection keys in a layer are applied to those).
    """
    declared = layer.pop("config_version", None)
    if isinstance(declared, int) and not isinstance(declared, bool):
        if declared < CURRENT_CONFIG_VERSION:
            layer["config_version"] = declared
            full_notes = migrate(layer)
            layer.pop("config_version", None)
            return full_notes
        return []  # newer than we know: pass through as-is
    notes: list[str] = []
    shape_migrate_analysis(
        layer, notes, layer=True, vlm_endpoint=vlm_endpoint, llm_endpoint=llm_endpoint
    )
    return notes


# --------------------------------------------------------------------------------------
# Text-level helpers (comment-preserving)
# --------------------------------------------------------------------------------------


def _round_trip_yaml() -> YAML:
    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    # Keep long comment/scalar lines on one line (the example config has very long ones).
    yaml_rt.width = 4096
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    return yaml_rt


def parse_round_trip(text: str) -> Any:
    """Parse ``text`` with ruamel round-trip mode (comments preserved); None if empty."""
    return _round_trip_yaml().load(text)


def dump_round_trip(doc: Any) -> str:
    out = io.StringIO()
    _round_trip_yaml().dump(doc, out)
    return out.getvalue()


def timestamped_backup_path(path: Path, old_version: int, *, now: datetime | None = None) -> Path:
    """``config.yaml.bak-v<old>-<YYYYMMDD-HHMMSS>`` next to ``path`` (never an existing file)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = path.with_name(f"{path.name}.bak-v{old_version}-{stamp}")
    candidate, n = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}.{n}")
        n += 1
    return candidate


def migrate_text(text: str) -> tuple[str, int, list[str]]:
    """Migrate a config file's text; return ``(new_text, old_version, notes)``.

    Comments, key order and quoting are preserved (ruamel round-trip). Text that is
    empty/comment-only, not a mapping, or already current is returned unchanged with no
    notes. Raises whatever ruamel raises on malformed YAML -- callers decide the policy.
    """
    doc = parse_round_trip(text)
    if not isinstance(doc, MutableMapping):
        return text, 1, []
    old = get_config_version(doc)
    notes = migrate(doc)
    if old >= CURRENT_CONFIG_VERSION:
        return text, old, []
    return dump_round_trip(doc), old, notes


__all__ = [
    "CURRENT_CONFIG_VERSION",
    "ENDPOINT_FIELDS",
    "MIGRATIONS",
    "dump_round_trip",
    "get_config_version",
    "migrate",
    "migrate_layer",
    "migrate_text",
    "parse_round_trip",
    "timestamped_backup_path",
    "shape_migrate_analysis",
]
