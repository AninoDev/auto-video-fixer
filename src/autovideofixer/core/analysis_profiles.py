"""Analysis profiles (docs/REQUIREMENTS.md § 28.1).

A profile is a named, *partial* ``analysis`` config. ``avf analyze --profile NAME`` and
``stages.analyze.profile`` deep-merge it onto the ``analysis`` section with the same layering
as a preset (``Config.with_layer`` -> ``deep_merge``: a mapping recurses, a scalar or list
replaces). Explicit CLI flags are call arguments, so they still win over the profile.

Built-ins: ``quick``, ``standard``, ``thorough``. A user profile in ``analysis.profiles`` with a
built-in's name is merged ON TOP of the built-in (set one key on ``quick`` and keep the rest);
to replace it wholesale, give it a new name. Profiles shape *how* the analysis runs, not
*whether* the VLM runs: none of them sets ``analysis.vlm.enabled`` (use ``--vlm`` / the config).
"""

from __future__ import annotations

import copy
from typing import Any

from autovideofixer.config import Config, deep_merge
from autovideofixer.logger import get_logger

logger = get_logger(__name__)

BUILTIN_PROFILE_NAMES = ("quick", "standard", "thorough")

# quick: cheapest useful run -- brief text from 2-4 frames per call.
_QUICK: dict[str, Any] = {
    "vlm": {"detail": "brief", "sampling": {"min_frames": 2, "max_frames": 4}},
}

# standard: exactly today's defaults, i.e. an empty overlay.
_STANDARD: dict[str, Any] = {}

# thorough: a whole-video overview AND a description per scene (`both`, so the table's summary
# row and the per-scene results are both filled), `detailed` text (`exhaustive` roughly doubles
# the output tokens again and truncates at the default max_tokens), media type and the rating's
# reasoning. Cross-check is added by `_thorough()` only when it would validate.
_THOROUGH: dict[str, Any] = {
    "vlm": {
        "detail": "detailed",
        "scope": "both",
        "outputs": {"media_type": True, "rating_reasoning": True},
        "sampling": {"min_frames": 6, "max_frames": 12},
    },
}


class UnknownProfileError(ValueError):
    """``--profile`` / ``stages.analyze.profile`` names no built-in or configured profile."""


def user_profiles(config: Config) -> dict[str, Any]:
    """The ``analysis.profiles`` mapping (``ValueError`` if it is not one)."""
    profiles = config.get("analysis", "profiles", default={}) or {}
    if not isinstance(profiles, dict):
        raise ValueError("Invalid analysis.profiles: must be a mapping of name -> profile")
    return profiles


def available_profiles(config: Config) -> list[str]:
    """Built-in names first, then user-defined ones, each once."""
    names = list(BUILTIN_PROFILE_NAMES)
    names.extend(n for n in user_profiles(config) if n not in names)
    return names


def _thorough(config: Config) -> dict[str, Any]:
    """``thorough``, plus ``cross_check.enabled`` iff the configured cross-check validates.

    Only turned on when ``analysis.cross_check.endpoints`` lists at least two names AND the
    resulting config passes the real cross-check validation (names resolve, judge/arbiter
    endpoints exist...). Anything else leaves it off, with an INFO line: a profile must never
    make a run fail that would have worked without it.
    """
    profile = copy.deepcopy(_THOROUGH)
    endpoints = config.get("analysis", "cross_check", "endpoints", default=[])
    if not isinstance(endpoints, list) or len(endpoints) < 2:
        return profile
    from autovideofixer.core.cross_check import CrossCheckConfig

    trial = config.with_layer({"analysis": {"cross_check": {"enabled": True}}}, "profile-trial")
    try:
        CrossCheckConfig.from_config(trial)
    except ValueError as e:
        logger.info("Profile 'thorough': cross-check left off (%s)", e)
        return profile
    profile["cross_check"] = {"enabled": True}
    return profile


def resolve_profile(config: Config, name: str) -> dict[str, Any]:
    """The effective overlay for profile ``name`` (built-in, then the user's on top).

    Raises ``UnknownProfileError`` (listing the available names) for an unknown name and
    ``ValueError`` for a malformed profile.
    """
    users = user_profiles(config)
    if name not in BUILTIN_PROFILE_NAMES and name not in users:
        raise UnknownProfileError(
            f"Unknown analysis profile {name!r}. Available: {', '.join(available_profiles(config))}"
        )
    if name == "quick":
        profile = copy.deepcopy(_QUICK)
    elif name == "thorough":
        profile = _thorough(config)
    else:
        profile = copy.deepcopy(_STANDARD)
    mine = users.get(name)
    if mine is not None:
        if not isinstance(mine, dict):
            raise ValueError(f"Invalid analysis.profiles.{name}: must be a mapping")
        _check_profile_keys(name, mine, config)
        deep_merge(profile, mine)
    return profile


def _check_profile_keys(name: str, profile: dict[str, Any], config: Config) -> None:
    """Top-level keys must be real ``analysis`` keys (a typo would silently do nothing)."""
    known = set(Config.DEFAULTS["analysis"]) - {"profiles"}
    for key in profile:
        if key not in known:
            raise ValueError(
                f"Invalid analysis.profiles.{name}: unknown key {key!r} "
                f"(allowed: {', '.join(sorted(known))})"
            )


def apply_profile(config: Config, name: str | None) -> Config:
    """``config`` with profile ``name`` layered onto ``analysis`` (a copy); ``None`` -> as is."""
    if name is None:
        return config
    overlay = resolve_profile(config, name)
    if not overlay:
        return config.with_layer({}, f"analysis-profile:{name}")
    return config.with_layer({"analysis": overlay}, f"analysis-profile:{name}")
