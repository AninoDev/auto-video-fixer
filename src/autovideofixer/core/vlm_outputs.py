"""Structured VLM output fields: which fields to ask for, the prompt and schema for them, and
parsing the reply (REQUIREMENTS.md § 26.1).

One ``OutputsConfig`` (from ``analysis.vlm.outputs`` / ``media_type_choices`` / ``detail``) drives
three things from the same field list so they cannot drift apart: the default system prompt, the
default user prompt and the ``json_schema`` ``response_format`` schema. ``parse_fields`` is the
matching reader. ``prompt_override`` / ``system_prompt_override`` still replace the generated
prompts wholesale -- an override has to ask for the fields itself (the schema, when
``compat.response_format`` is ``json_schema``, is still built from the enabled outputs).

Field order is deliberate and fixed: ``summary, tags, objects, media_type, rating,
rating_reasoning``. ``rating_reasoning`` is last so a non-thinking model has already seen the
content (and the rating it is justifying) when it writes the justification; generation order is
the only "reasoning" such a model gets. A thinking model reasons first anyway.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DETAIL_LEVELS = ("brief", "standard", "detailed", "exhaustive")
DEFAULT_MEDIA_TYPE_CHOICES: tuple[str, ...] = (
    "real_footage",
    "3d_rendered",
    "2d_digital",
    "hand_drawn",
    "ai_generated",
    "screen_capture",
    "mixed",
    "other",
)
OTHER_MEDIA_TYPE = "other"
RATING_CHOICES = ("G", "PG", "PG-13", "R")
OUTPUT_KEYS = ("media_type", "rating_reasoning")
DEFAULT_OUTPUTS: dict[str, bool] = {"media_type": True, "rating_reasoning": False}

# Keys the parser maps onto VideoAnalysis fields; anything else the model returned is "extra".
KNOWN_FIELDS = ("summary", "tags", "objects", "media_type", "rating", "rating_reasoning")


@dataclass(frozen=True)
class OutputsConfig:
    media_type: bool = True
    rating_reasoning: bool = False
    detail: str = "brief"
    media_type_choices: tuple[str, ...] = DEFAULT_MEDIA_TYPE_CHOICES

    @property
    def keys(self) -> tuple[str, ...]:
        """Enabled JSON keys in generation order (``rating_reasoning`` always last)."""
        out = ["summary", "tags", "objects"]
        if self.media_type:
            out.append("media_type")
        out.append("rating")
        if self.rating_reasoning:
            out.append("rating_reasoning")
        return tuple(out)

    @classmethod
    def from_config(cls, vlm_config: Mapping[str, Any] | None) -> OutputsConfig:
        """Validated settings from the ``analysis.vlm`` mapping; ``ValueError`` names the key."""
        vlm_config = vlm_config or {}
        outputs = vlm_config.get("outputs")
        flags = dict(DEFAULT_OUTPUTS)
        if outputs is not None:
            if not isinstance(outputs, Mapping):
                raise ValueError(
                    f"Invalid analysis.vlm.outputs {outputs!r}: expected a mapping "
                    f"with keys {', '.join(OUTPUT_KEYS)}"
                )
            unknown = sorted(str(k) for k in outputs if k not in DEFAULT_OUTPUTS)
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in analysis.vlm.outputs: {', '.join(unknown)} "
                    f"(valid: {', '.join(OUTPUT_KEYS)})"
                )
            for key, value in outputs.items():
                if value is None:
                    continue
                if not isinstance(value, bool):
                    raise ValueError(
                        f"Invalid analysis.vlm.outputs.{key} {value!r}: must be true or false"
                    )
                flags[key] = value

        detail = vlm_config.get("detail")
        detail = "brief" if detail is None else detail
        if not isinstance(detail, str) or detail.strip().lower() not in DETAIL_LEVELS:
            raise ValueError(
                f"Invalid analysis.vlm.detail {detail!r}: must be one of {', '.join(DETAIL_LEVELS)}"
            )

        raw_choices = vlm_config.get("media_type_choices")
        if raw_choices is None:
            choices = DEFAULT_MEDIA_TYPE_CHOICES
        else:
            if (
                not isinstance(raw_choices, (list, tuple))
                or not raw_choices
                or not all(isinstance(c, str) and c.strip() for c in raw_choices)
            ):
                raise ValueError(
                    f"Invalid analysis.vlm.media_type_choices {raw_choices!r}: must be a "
                    "non-empty list of non-empty strings"
                )
            seen: dict[str, str] = {}
            for c in raw_choices:
                norm = normalize_token(c)
                if not norm:
                    raise ValueError(
                        f"Invalid analysis.vlm.media_type_choices entry {c!r}: needs at least "
                        "one letter or digit"
                    )
                seen.setdefault(norm, c.strip())
            choices = tuple(seen.values())
        return cls(
            media_type=flags["media_type"],
            rating_reasoning=flags["rating_reasoning"],
            detail=detail.strip().lower(),
            media_type_choices=tuple(choices),
        )

    @property
    def schema_media_types(self) -> list[str]:
        """Enum for the schema: the choices, plus ``other`` (the fallback) if not listed."""
        values = list(self.media_type_choices)
        if OTHER_MEDIA_TYPE not in {normalize_token(v) for v in values}:
            values.append(OTHER_MEDIA_TYPE)
        return values


def normalize_token(value: str) -> str:
    """Lower-case, non-alphanumerics collapsed to ``_``: '3D Rendered' -> '3d_rendered'."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_media_type(
    value: Any, choices: tuple[str, ...] = DEFAULT_MEDIA_TYPE_CHOICES
) -> tuple[str | None, str | None]:
    """``(normalised, raw)``. ``normalised`` is a listed choice, else ``other``; ``None`` when
    the model returned nothing. ``raw`` is the original text, returned only when it differs from
    the normalised value (so the caller can keep it without cluttering the common case)."""
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return None, None
    wanted = normalize_token(raw)
    for choice in choices:
        if normalize_token(choice) == wanted:
            return choice, (raw if raw != choice else None)
    return OTHER_MEDIA_TYPE, raw


# ─── Prompt generation ─────────────────────────────────────────────

_SUMMARY_BY_DETAIL = {
    "brief": "a brief summary of the content (1-2 sentences)",
    "standard": (
        "a summary of the content (3-5 sentences) covering the main subjects, what happens "
        "and where"
    ),
    "detailed": (
        "a detailed description of the content (one full paragraph) covering the characters "
        "or subjects, their actions, the setting, camera work and the visual or rendering style"
    ),
    "exhaustive": (
        "an exhaustive description of the content (several paragraphs) covering every "
        "character or subject, their actions in order, the setting and environment, camera "
        "work and movement, lighting, colour, any on-screen text, and the visual or rendering "
        "style"
    ),
}
_TAGS_BY_DETAIL = {
    "brief": "relevant tags (comma-separated keywords)",
    "standard": "relevant tags (comma-separated keywords)",
    "detailed": "descriptive tags (comma-separated keywords, including genre and style)",
    "exhaustive": "descriptive tags (comma-separated keywords, including genre, style and mood)",
}
_OBJECTS_BY_DETAIL = {
    "brief": "objects detected (comma-separated)",
    "standard": "objects detected (comma-separated)",
    "detailed": "objects and characters detected (comma-separated)",
    "exhaustive": "every distinct object and character detected (comma-separated)",
}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def build_system_prompt(outputs: OutputsConfig) -> str:
    """Default system prompt for the enabled outputs and detail level.

    With ``brief`` detail and no optional outputs this is byte-identical to the pre-§ 26 prompt.
    """
    d = outputs.detail
    parts = [_SUMMARY_BY_DETAIL[d], _TAGS_BY_DETAIL[d], _OBJECTS_BY_DETAIL[d]]
    if outputs.media_type:
        parts.append(
            "the media type (exactly one of: "
            + ", ".join(outputs.schema_media_types)
            + "; use other if none fits)"
        )
    parts.append("an appropriate content rating (" + ", ".join(RATING_CHOICES) + ")")
    if outputs.rating_reasoning:
        parts.append("a short justification for the rating (1-2 sentences naming what drove it)")
    keys = ", ".join(f'"{k}"' for k in outputs.keys)
    text = (
        "You are a video analysis assistant. Analyze the provided frames from a video and "
        f"provide: {_join(parts)}. Respond in valid JSON format with keys: {keys}."
    )
    if outputs.rating_reasoning:
        text += (
            ' Consider the content first, then give "rating" followed by "rating_reasoning" '
            "as the last two keys."
        )
    return text


def build_user_prompt(outputs: OutputsConfig) -> str:
    """Default user prompt (byte-identical to the pre-§ 26 one with nothing optional enabled)."""
    items = ["a summary", "tags", "detected objects"]
    if outputs.media_type:
        items.append("the media type")
    items.append("content rating")
    if outputs.rating_reasoning:
        items.append("the reasoning for the rating")
    return f"Analyze these video frames and describe the content. Provide {_join(items)}."


def build_response_schema(outputs: OutputsConfig) -> dict[str, Any]:
    """JSON Schema for ``compat.response_format: json_schema``; key order = generation order."""
    str_or_list = {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}
    props: dict[str, Any] = {
        "summary": {"type": "string"},
        "tags": dict(str_or_list),
        "objects": dict(str_or_list),
    }
    if outputs.media_type:
        props["media_type"] = {"type": "string", "enum": outputs.schema_media_types}
    props["rating"] = {"type": "string", "enum": list(RATING_CHOICES)}
    if outputs.rating_reasoning:
        props["rating_reasoning"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}


# ─── Parsing ───────────────────────────────────────────────────────


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, list):
        return value
    return []


def parse_fields(
    data: Mapping[str, Any], choices: tuple[str, ...] = DEFAULT_MEDIA_TYPE_CHOICES
) -> dict[str, Any]:
    """Map a decoded model JSON object onto the result dict.

    ``media_type`` is normalised to one of ``choices`` (else ``other``; the original wording is
    kept in ``extra['media_type_raw']`` when it differed). Any key the model returned that is not
    a known field lands in ``extra`` unchanged.
    """
    media_type, raw = normalize_media_type(data.get("media_type"), choices)
    reasoning = data.get("rating_reasoning")
    extra = {k: v for k, v in data.items() if k not in KNOWN_FIELDS}
    if raw is not None:
        extra["media_type_raw"] = raw
    return {
        "summary": data.get("summary", ""),
        "tags": _as_list(data.get("tags", [])),
        "objects": _as_list(data.get("objects", [])),
        "rating": data.get("rating"),
        "media_type": media_type,
        "rating_reasoning": None if reasoning is None else str(reasoning),
        "extra": extra,
    }
