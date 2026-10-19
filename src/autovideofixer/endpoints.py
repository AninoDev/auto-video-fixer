"""Endpoint registry: named, fully independent model-connection entries (REQUIREMENTS.md § 24.4).

``analysis.endpoints.<name>`` holds everything needed to reach one model server --
provider, model, URL, key, ``allow_http``, timeout, token budget, per-endpoint TLS override,
request-shape ``compat`` knobs and (P4) ``vision`` limits. ``analysis.vlm.endpoint`` and
``analysis.llm.endpoint`` pick which entry the whole-video/per-scene VLM, the crop VLM check
and the scene coordinator use. **Every consumer resolves through ``resolve_endpoint``** so
there is exactly one place that validates an entry and one place to add a field.

Resolution is *use-time*: an unknown name, bad provider or bad ``compat`` value raises
``ValueError`` when something is about to call the endpoint (same posture as
``config.resolve_timeout`` and ``net.resolve_tls_policy``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from autovideofixer.config import resolve_timeout

# "local" is the legacy alias for "ollama" (the v1->v2 migration rewrites it; accepted here
# so a stale layer or programmatic config keeps working).
PROVIDERS = ("ollama", "openai", "api", "llamacpp")
_PROVIDER_ALIASES = {"local": "ollama"}
OPENAI_SHAPED = ("openai", "api", "llamacpp")

SYSTEM_CONTENT_MODES = ("string", "parts")
RESPONSE_FORMAT_MODES = ("json_object", "json_schema", "none")

# § 24.5 provider defaults. openai/api keep today's request shape; llamacpp mirrors the
# user's working hand-patched setup (system content as parts, no response_format).
_COMPAT_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {},
    "openai": {"system_content": "string", "response_format": "json_object"},
    "api": {"system_content": "string", "response_format": "json_object"},
    "llamacpp": {"system_content": "parts", "response_format": "none"},
}

_FIELD_DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "model": "",
    "api_url": "",
    "api_key": "",
    "allow_http": False,
    "timeout_sec": 120,
    "max_tokens": 1024,
}


@dataclass(frozen=True)
class Endpoint:
    """A fully resolved, validated endpoint entry.

    ``timeout_sec`` is already resolved (``None`` = unlimited). ``compat`` is the
    *effective* mapping (provider defaults overlaid with the entry's own ``compat``) with
    ``system_content`` / ``response_format`` always set for OpenAI-shaped providers and
    ``extra_body`` always a dict. ``tls`` is the raw per-endpoint override mapping; the caller
    turns it into a policy with ``net.resolve_tls_policy(config, endpoint.tls, purpose=...)``.
    """

    name: str
    provider: str
    model: str
    api_url: str
    api_key: str
    allow_http: bool
    timeout_sec: float | None
    max_tokens: int
    tls: dict[str, Any] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)

    @property
    def tls_purpose(self) -> str:
        return f"endpoint:{self.name}"

    def with_overrides(self, *, model: str | None = None, api_url: str | None = None) -> Endpoint:
        """Per-run override of model / api_url (``avf analyze --vlm-model/--vlm-url``)."""
        changes: dict[str, Any] = {}
        if model is not None:
            changes["model"] = model
        if api_url is not None:
            changes["api_url"] = api_url
        return replace(self, **changes) if changes else self


def normalize_provider(provider: Any) -> str:
    value = str(provider).strip().lower() if provider is not None else ""
    return _PROVIDER_ALIASES.get(value, value)


def _as_mapping(value: Any, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid {what}: expected a mapping, got {type(value).__name__}")
    return copy.deepcopy(dict(value))


def _cfg_get(config: Any, *keys: str) -> Any:
    if config is None:
        return None
    if hasattr(config, "get") and not isinstance(config, Mapping):
        return config.get(*keys)
    node: Any = config
    for k in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(k)
    return node


def resolve_compat(provider: str, raw: Any, where: str) -> dict[str, Any]:
    """Effective ``compat`` for ``provider``: its defaults overlaid with the entry's own."""
    entry = _as_mapping(raw, f"{where}.compat")
    compat: dict[str, Any] = dict(_COMPAT_DEFAULTS.get(provider, {}))
    for key in ("system_content", "response_format"):
        if entry.get(key) is not None:
            compat[key] = entry[key]
    if provider in OPENAI_SHAPED:
        if compat["system_content"] not in SYSTEM_CONTENT_MODES:
            raise ValueError(
                f"Invalid {where}.compat.system_content {compat['system_content']!r}: "
                f"must be one of {SYSTEM_CONTENT_MODES}"
            )
        if compat["response_format"] not in RESPONSE_FORMAT_MODES:
            raise ValueError(
                f"Invalid {where}.compat.response_format {compat['response_format']!r}: "
                f"must be one of {RESPONSE_FORMAT_MODES}"
            )
        compat["extra_body"] = _as_mapping(entry.get("extra_body"), f"{where}.compat.extra_body")
    return compat


VISION_DEFAULTS: dict[str, Any] = {
    "max_long_side_px": None,
    "max_megapixels": None,
    "tile_width": None,
    "tile_height": None,
    "tokens_per_tile": 1,
    "max_tiles_per_frame": None,
    "max_tiles_per_call": None,
    "max_frames_per_call": None,
    "jpeg_quality": 90,
}
_VISION_INT_KEYS = (
    "max_long_side_px",
    "tile_width",
    "tile_height",
    "max_tiles_per_frame",
    "max_tiles_per_call",
    "max_frames_per_call",
)


def resolve_vision(raw: Any, where: str) -> dict[str, Any]:
    """Validated ``vision`` block (§ 25.3) with every key present (``None`` = no limit).

    Raises ``ValueError`` naming the endpoint (``where``) and key. Integers must be positive
    ints, ``max_megapixels``/``tokens_per_tile`` positive numbers, ``jpeg_quality`` 1-100. A
    tile size needs both ``tile_width`` and ``tile_height``, and the tile budgets
    (``max_tiles_per_*``) need a tile size.
    """
    entry = _as_mapping(raw, f"{where}.vision")
    unknown = sorted(set(entry) - set(VISION_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {where}.vision: {', '.join(map(str, unknown))} "
            f"(valid: {', '.join(VISION_DEFAULTS)})"
        )
    out = dict(VISION_DEFAULTS)
    for key, value in entry.items():
        if value is None:
            continue
        label = f"{where}.vision.{key}"
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key in _VISION_INT_KEYS:
            if not (isinstance(value, int) and not isinstance(value, bool)) or value <= 0:
                raise ValueError(f"Invalid {label} {value!r}: must be a positive integer")
        elif key in ("max_megapixels", "tokens_per_tile"):
            if not is_num or not (value > 0) or value == float("inf"):
                raise ValueError(f"Invalid {label} {value!r}: must be a positive number")
        elif key == "jpeg_quality" and (
            not (isinstance(value, int) and not isinstance(value, bool)) or not 1 <= value <= 100
        ):
            raise ValueError(f"Invalid {label} {value!r}: must be an integer from 1 to 100")
        out[key] = value
    if (out["tile_width"] is None) != (out["tile_height"] is None):
        raise ValueError(f"{where}.vision: tile_width and tile_height must be set together")
    for key in ("max_tiles_per_frame", "max_tiles_per_call"):
        if out[key] is not None and out["tile_width"] is None:
            raise ValueError(
                f"{where}.vision.{key} needs tile_width and tile_height (tiles are counted "
                "in units of that size)"
            )
    return out


def resolve_endpoint(config: Any, name: str) -> Endpoint:
    """Resolve ``analysis.endpoints.<name>`` to a validated ``Endpoint``.

    ``config`` may be a ``Config`` or a plain nested dict. Raises ``ValueError`` with a
    message naming the available endpoints when ``name`` is unknown, and on an invalid
    provider / timeout / ``compat`` / ``vision`` value.
    """
    endpoints = _cfg_get(config, "analysis", "endpoints")
    if not isinstance(endpoints, Mapping):
        endpoints = {}
    raw = endpoints.get(name)
    if not isinstance(raw, Mapping):
        available = ", ".join(sorted(str(k) for k in endpoints)) or "(none defined)"
        raise ValueError(
            f"Unknown analysis endpoint {name!r} (defined under analysis.endpoints: {available})"
        )
    where = f"analysis.endpoints.{name}"
    merged = {**_FIELD_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}
    # timeout_sec: an explicit null means unlimited, so look at the raw key, not `merged`.
    timeout_raw = raw["timeout_sec"] if "timeout_sec" in raw else _FIELD_DEFAULTS["timeout_sec"]

    provider = normalize_provider(merged["provider"])
    if provider not in PROVIDERS:
        raise ValueError(
            f"Invalid {where}.provider {merged['provider']!r}: must be one of {PROVIDERS}"
        )
    try:
        max_tokens = int(merged["max_tokens"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {where}.max_tokens {merged['max_tokens']!r}") from e

    return Endpoint(
        name=name,
        provider=provider,
        model=str(merged["model"]),
        api_url=str(merged["api_url"] or ""),
        api_key=str(merged["api_key"] or ""),
        allow_http=bool(merged["allow_http"]),
        timeout_sec=resolve_timeout(timeout_raw, f"{where}.timeout_sec"),
        max_tokens=max_tokens,
        tls=_as_mapping(raw.get("tls"), f"{where}.tls"),
        compat=resolve_compat(provider, raw.get("compat"), where),
        vision=resolve_vision(raw.get("vision"), where),
    )


def role_endpoint_name(config: Any, role: str) -> str:
    """Name selected by ``analysis.<role>.endpoint`` (``role`` is ``vlm`` or ``llm``)."""
    default = {"vlm": "default_vlm", "llm": "default_llm"}[role]
    value = _cfg_get(config, "analysis", role, "endpoint")
    return str(value) if value else default


def resolve_role_endpoint(config: Any, role: str, name: str | None = None) -> Endpoint:
    """Endpoint for the ``vlm`` / ``llm`` role; ``name`` overrides the configured choice."""
    return resolve_endpoint(config, name or role_endpoint_name(config, role))


def all_endpoint_urls(config: Any) -> list[str]:
    """Every configured endpoint ``api_url`` (for the PII log cleaner)."""
    endpoints = _cfg_get(config, "analysis", "endpoints")
    if not isinstance(endpoints, Mapping):
        return []
    return [
        str(ep["api_url"])
        for ep in endpoints.values()
        if isinstance(ep, Mapping) and ep.get("api_url")
    ]
