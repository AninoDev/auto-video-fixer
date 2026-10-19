"""Multi-VLM cross-check: config, prompts, verdict parsing and the result block (§ 27).

``analysis.cross_check`` makes several VLM endpoints look at the same range of a video and
combines their answers, in one of two modes:

* ``refine`` -- the first endpoint describes; each next endpoint is shown its own frames plus the
  previous JSON and must verify/correct it (the reply is the normal schema plus ``changes``).
  The last successful output wins. Inherently sequential.
* ``judge`` -- every endpoint describes independently; a text-only judge LLM receives all the
  JSON outputs and returns ``{agreement, conflicts, merged}``. ``agreement >= agreement_threshold``
  accepts ``merged``; below it an optional arbiter VLM (frames + all descriptions + the conflicts)
  writes the final, and with no arbiter the primary's output is kept and flagged
  ``disagreement``.

This module is the pure half: validation, prompts/schemas generated from the same field list as
the normal describe prompt (``vlm_outputs``) so they cannot drift, verdict parsing and the
``cross_check`` result block. The orchestration (frames, provider calls, threads) is in
``core/analysis.py`` next to the provider plumbing it reuses.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from autovideofixer.core.vlm_outputs import (
    OutputsConfig,
    build_response_schema,
    build_system_prompt,
    parse_fields,
    result_fields,
)
from autovideofixer.endpoints import resolve_endpoint, role_endpoint_name

MODES = ("refine", "judge")
DEFAULT_MODE = "refine"
DEFAULT_THRESHOLD = 0.75
DEFAULT_MAX_PARALLEL = 1
CONFIG_KEYS = (
    "enabled",
    "mode",
    "endpoints",
    "judge_endpoint",
    "arbiter_endpoint",
    "agreement_threshold",
    "max_parallel",
)

# CallStats.purpose of each non-primary leg. The primary leg keeps ``video`` / ``scene``.
PURPOSE_DESCRIBE = "cross_check:describe"
PURPOSE_REFINE = "cross_check:refine"
PURPOSE_JUDGE = "cross_check:judge"
PURPOSE_ARBITER = "cross_check:arbiter"

# Leg roles as written to the ``legs`` list.
ROLE_PRIMARY = "primary"
ROLE_DESCRIBE = "describe"
ROLE_REFINE = "refine"
ROLE_JUDGE = "judge"
ROLE_ARBITER = "arbiter"


@dataclass(frozen=True)
class CrossCheckConfig:
    enabled: bool = False
    mode: str = DEFAULT_MODE
    endpoints: tuple[str, ...] = ()
    judge_endpoint: str | None = None
    arbiter_endpoint: str | None = None
    agreement_threshold: float = DEFAULT_THRESHOLD
    max_parallel: int = DEFAULT_MAX_PARALLEL

    @classmethod
    def from_config(cls, config: Any) -> CrossCheckConfig:
        """Validated ``analysis.cross_check``; ``ValueError`` names the offending key.

        The shape is always checked. Endpoint names are only resolved when ``enabled`` (a
        disabled block must never be able to break an analysis), where ``enabled: true`` with
        fewer than two endpoints is an error too.
        """
        raw = config.get("analysis", "cross_check", default=None) if config is not None else None
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Invalid analysis.cross_check {raw!r}: expected a mapping with keys "
                f"{', '.join(CONFIG_KEYS)}"
            )
        unknown = sorted(str(k) for k in raw if k not in CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown key(s) in analysis.cross_check: {', '.join(unknown)} "
                f"(valid: {', '.join(CONFIG_KEYS)})"
            )

        def get(key: str, default: Any) -> Any:
            value = raw.get(key)
            return default if value is None else value

        enabled = get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Invalid analysis.cross_check.enabled {enabled!r}: must be true or false"
            )
        mode = get("mode", DEFAULT_MODE)
        if not isinstance(mode, str) or mode.strip().lower() not in MODES:
            raise ValueError(
                f"Invalid analysis.cross_check.mode {mode!r}: must be one of {', '.join(MODES)}"
            )
        names = get("endpoints", [])
        if not isinstance(names, (list, tuple)) or not all(
            isinstance(n, str) and n.strip() for n in names
        ):
            raise ValueError(
                f"Invalid analysis.cross_check.endpoints {names!r}: must be a list of "
                "endpoint names (analysis.endpoints.<name>)"
            )
        threshold = get("agreement_threshold", DEFAULT_THRESHOLD)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                f"Invalid analysis.cross_check.agreement_threshold {threshold!r}: "
                "must be a number from 0 to 1"
            )
        parallel = get("max_parallel", DEFAULT_MAX_PARALLEL)
        if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
            raise ValueError(
                f"Invalid analysis.cross_check.max_parallel {parallel!r}: must be an integer >= 1"
            )
        optional: dict[str, str | None] = {}
        for key in ("judge_endpoint", "arbiter_endpoint"):
            value = raw.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"Invalid analysis.cross_check.{key} {value!r}: must be an endpoint name "
                    "or null"
                )
            optional[key] = value.strip() if isinstance(value, str) else None

        cfg = cls(
            enabled=enabled,
            mode=mode.strip().lower(),
            endpoints=tuple(n.strip() for n in names),
            judge_endpoint=optional["judge_endpoint"],
            arbiter_endpoint=optional["arbiter_endpoint"],
            agreement_threshold=float(threshold),
            max_parallel=parallel,
        )
        if enabled:
            if len(cfg.endpoints) < 2:
                raise ValueError(
                    "analysis.cross_check.enabled is true but analysis.cross_check.endpoints "
                    f"lists {len(cfg.endpoints)} endpoint(s); a cross-check needs at least 2 "
                    "(the first is the primary)"
                )
            for i, name in enumerate(cfg.endpoints):
                _require_endpoint(config, name, f"analysis.cross_check.endpoints[{i}]")
            if cfg.mode == "judge":
                _require_endpoint(
                    config, cfg.judge_name(config), "analysis.cross_check.judge_endpoint"
                )
            if cfg.arbiter_endpoint is not None:
                _require_endpoint(
                    config, cfg.arbiter_endpoint, "analysis.cross_check.arbiter_endpoint"
                )
        return cfg

    def judge_name(self, config: Any) -> str:
        """The judge LLM's endpoint name: ``judge_endpoint``, else ``analysis.llm.endpoint``."""
        return self.judge_endpoint or role_endpoint_name(config, "llm")


def _require_endpoint(config: Any, name: str, where: str) -> None:
    try:
        resolve_endpoint(config, name)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e


# ─── Judge / arbiter prompts ───────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You compare several independent analyses of the same video, each written by a different "
    "model that looked at its own frames. You cannot see the video. Decide how well the analyses "
    "agree on what the video shows (subjects, actions, setting, media type, rating), list the "
    "concrete points on which they conflict, and write one merged analysis that keeps what they "
    "agree on and resolves small differences sensibly. Respond in valid JSON with keys: "
    '"agreement" (a number from 0 for complete disagreement to 1 for full agreement), '
    '"conflicts" (a list of short strings, one per point of disagreement; empty if none) and '
    '"merged" (the merged analysis, an object with the same keys as the analyses).'
)

ARBITER_NOTE = (
    "Other models analysed this video and disagreed. You are shown frames from the video, "
    "their analyses and the points on which they conflict. Settle each conflict from what you "
    "actually see in the frames, and write the final analysis."
)


def build_judge_schema(outputs: OutputsConfig) -> dict[str, Any]:
    """JSON Schema of the judge reply; ``merged`` is the normal outputs schema."""
    props: dict[str, Any] = {
        "agreement": {"type": "number", "minimum": 0, "maximum": 1},
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "merged": build_response_schema(outputs),
    }
    return {"type": "object", "properties": props, "required": list(props)}


def _analyses_text(
    analyses: Sequence[tuple[str, Mapping[str, Any]]], outputs: OutputsConfig
) -> str:
    blocks = []
    for i, (name, result) in enumerate(analyses, start=1):
        body = json.dumps(result_fields(result, outputs), ensure_ascii=False, indent=2)
        blocks.append(f"Analysis {i} (from {name}):\n{body}")
    return "\n\n".join(blocks)


def build_judge_user_prompt(
    analyses: Sequence[tuple[str, Mapping[str, Any]]], outputs: OutputsConfig
) -> str:
    """User prompt for the judge: every successful describe leg's JSON, in endpoint order."""
    return (
        f"Here are {len(analyses)} analyses of the same video.\n\n"
        f"{_analyses_text(analyses, outputs)}\n\n"
        "Compare them and respond with the JSON object described."
    )


def build_arbiter_system_prompt(outputs: OutputsConfig) -> str:
    """The normal describe prompt (same fields/detail) preceded by the arbiter's brief."""
    return f"{ARBITER_NOTE} {build_system_prompt(outputs)}"


def build_arbiter_user_prompt(
    analyses: Sequence[tuple[str, Mapping[str, Any]]],
    conflicts: list[str],
    outputs: OutputsConfig,
) -> str:
    listed = "\n".join(f"- {c}" for c in conflicts) or "- (none listed)"
    return (
        f"{_analyses_text(analyses, outputs)}\n\n"
        f"Points of conflict:\n{listed}\n\n"
        "Look at the frames and write the final analysis as JSON."
    )


# ─── Judge verdict ─────────────────────────────────────────────────


@dataclass
class JudgeVerdict:
    agreement: float
    conflicts: list[str] = field(default_factory=list)
    merged: dict[str, Any] | None = None  # parse_fields() shape, None when absent/unusable

    def to_dict(self, threshold: float) -> dict[str, Any]:
        return {
            "agreement": self.agreement,
            "conflicts": list(self.conflicts),
            "merged": self.merged,
            "threshold": threshold,
            "accepted": self.agreement >= threshold,
        }


def parse_judge_verdict(
    data: Mapping[str, Any] | None, choices: tuple[str, ...]
) -> JudgeVerdict | None:
    """Validate a decoded judge reply; ``None`` (a failed judge leg) when it has no usable
    ``agreement``. The number is clamped to [0, 1] (a judge that answers ``85`` gets 1.0, it is
    not guessed to mean a percentage). ``conflicts`` tolerates a bare string; ``merged`` is
    parsed like a model reply, and is ``None`` unless it is an object with some content."""
    if not isinstance(data, Mapping):
        return None
    raw = data.get("agreement")
    if isinstance(raw, bool):
        return None
    try:
        agreement = float(raw)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    if not math.isfinite(agreement):
        return None
    agreement = min(1.0, max(0.0, agreement))

    conflicts_raw = data.get("conflicts")
    if isinstance(conflicts_raw, str):
        conflicts_raw = [conflicts_raw]
    conflicts: list[str] = []
    if isinstance(conflicts_raw, (list, tuple)):
        for item in conflicts_raw:
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            if text.strip():
                conflicts.append(text.strip())

    merged: dict[str, Any] | None = None
    merged_raw = data.get("merged")
    if isinstance(merged_raw, Mapping):
        parsed = parse_fields(merged_raw, choices)
        if is_usable(parsed):
            merged = parsed
    return JudgeVerdict(agreement=agreement, conflicts=conflicts, merged=merged)


def is_usable(result: Mapping[str, Any] | None) -> bool:
    """A parsed result with at least one non-empty field (an all-empty one is a failed leg)."""
    if not result:
        return False
    return bool(
        result.get("summary")
        or result.get("tags")
        or result.get("objects")
        or result.get("rating")
        or result.get("media_type")
    )


# ─── Result block ──────────────────────────────────────────────────

_RESULT_KEYS = ("summary", "tags", "objects", "media_type", "rating", "rating_reasoning", "extra")


def strip_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A leg's parsed fields without the bookkeeping keys (``call``, ``sampling``)."""
    if result is None:
        return None
    return {k: result[k] for k in _RESULT_KEYS if k in result}


@dataclass
class Leg:
    """One model call of a cross-check, successful or not."""

    endpoint: str
    role: str
    success: bool = False
    result: dict[str, Any] | None = None
    call: dict[str, Any] | None = None
    changes: list[str] | None = None  # refine legs only
    sampling: dict[str, Any] | None = (
        None  # the frames this leg was shown (describe/refine/arbiter)
    )
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "endpoint": self.endpoint,
            "role": self.role,
            "success": self.success,
            "result": strip_result(self.result),
            "call": self.call,
        }
        if self.changes is not None:
            out["changes"] = list(self.changes)
        if self.sampling is not None:
            out["sampling"] = self.sampling
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class CrossCheckOutcome:
    """Everything the ``cross_check`` result block reports."""

    mode: str
    endpoints: list[str]
    legs: list[Leg]
    # Endpoint that wrote the final (a judge's ``merged`` counts as the judge's) and the index
    # of its leg; both None when every call failed.
    final_source: str | None = None
    final_leg: int | None = None
    agreement: float | None = None
    conflicts: list[str] = field(default_factory=list)
    arbiter_ran: bool = False
    disagreement: bool = False
    verdict: dict[str, Any] | None = None

    def calls(self) -> list[dict[str, Any]]:
        """Every leg's CallStats dict, in leg order (what ``vlm_calls`` lists)."""
        return [leg.call for leg in self.legs if leg.call]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "endpoints": list(self.endpoints),
            "final_source": self.final_source,
            "final_leg": self.final_leg,
            "agreement": self.agreement,
            "conflicts": list(self.conflicts),
            "arbiter_ran": self.arbiter_ran,
            "disagreement": self.disagreement,
            "legs": [leg.to_dict() for leg in self.legs],
            "verdict": self.verdict,
        }
