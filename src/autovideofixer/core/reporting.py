"""Auto Video Fixer - stage/job reporting helpers (REQUIREMENTS.md § 6.4-6.6).

Pure, unit-testable functions that turn ``StageResult``/``JobResult`` data
into human-readable classifications, media-info comparisons, timing
aggregates, and a JSON-serializable run report. No I/O beyond
``write_json_report`` and no Rich rendering here -- callers (``cli.py``) own
console/log presentation; this module only computes *what* to show.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from autovideofixer.config import redact_secrets
from autovideofixer.core.stages.base import StageResult, StageStatus

if TYPE_CHECKING:
    from autovideofixer.config import Config
    from autovideofixer.core.pipeline import JobResult


# --- § 6.4 per-stage classification ----------------------------------------


def classify_stage(sr: StageResult) -> str:
    """Classify one stage's outcome into one of five buckets (§ 6.4):
    ``"ran-ai"``, ``"ran-traditional"``, ``"ran-traditional-fallback"``,
    ``"failed"``, ``"skipped"``.

    Precedence: ``status`` FAILED/SKIPPED wins outright regardless of
    ``metadata`` contents. For a COMPLETED result, an explicit
    ``ai_fallback_used`` marker (set once, at the
    ``BaseStage._ai_fallback_or_fail()`` seam) takes precedence over
    ``metadata["method"] == "ai"`` -- a completed AI-path result never sets
    ``ai_fallback_used`` in practice, but this ordering documents which one
    would win if it ever did. Anything else COMPLETED (no "ai" method, no
    fallback marker) is ``"ran-traditional"`` -- this covers both stages that
    literally set ``method: "traditional"`` and analysis-type stages that
    report their own method string (e.g. crop's detector name, detect's
    "ffprobe").
    """
    if sr.status == StageStatus.FAILED:
        return "failed"
    if sr.status == StageStatus.SKIPPED:
        return "skipped"
    if sr.metadata.get("ai_fallback_used"):
        return "ran-traditional-fallback"
    if sr.metadata.get("method") == "ai":
        return "ran-ai"
    return "ran-traditional"


def base_stage_name(label: str) -> str:
    """Map an occurrence label (``"upscale"``, ``"upscale#2"``) to its base
    registered stage name, per ``Pipeline.resolve_stage_order``'s
    ``f"{name}#{idx}"`` labeling convention (``idx == 1`` uses the bare
    name)."""
    return label.split("#", 1)[0]


def stage_table_rows(stage_results: dict[str, StageResult]) -> list[dict[str, Any]]:
    """Flatten a job's ``stage_results`` into plain dicts for rendering
    (Rich table columns and/or plain log lines), in insertion order."""
    rows = []
    for label, sr in stage_results.items():
        rows.append(
            {
                "label": label,
                "name": base_stage_name(label),
                "classification": classify_stage(sr),
                "method": sr.metadata.get("method"),
                "ai_fallback_used": bool(sr.metadata.get("ai_fallback_used")),
                "ai_fallback_reason": sr.metadata.get("ai_fallback_reason"),
                "duration_sec": sr.duration_sec,
                "skipped_reason": sr.skipped_reason,
                "error": sr.error,
            }
        )
    return rows


def job_summary_line(result: "JobResult") -> str:
    """One human-readable "at a glance" line for a completed job (§ 6.4):
    outcome (+ sub-reason), reprocessed note, scene stats when present."""
    parts = [f"outcome={result.outcome}"]
    if result.skip_reason:
        parts.append(f"sub-reason={result.skip_reason}")
    if result.reprocessed_mismatch:
        parts.append("reprocessed (was mismatched)")
    if result.scene_stats:
        ss = result.scene_stats
        parts.append(
            f"scenes={ss.get('kept')}/{ss.get('total')} kept ({ss.get('dropped')} dropped)"
        )
    parts.append(f"job_wall_ms={result.job_wall_ms:.1f}")
    parts.append(f"processing_ms={result.processing_ms:.1f}")
    return f"{os.path.basename(result.input_path)}: " + ", ".join(parts)


def run_classification_aggregate(job_results: list["JobResult"]) -> dict[str, int]:
    """§ 6.4 end-of-run aggregate: counts per stage classification bucket,
    across every stage occurrence in every job."""
    counts: dict[str, int] = {}
    for jr in job_results:
        for sr in jr.stage_results.values():
            cls = classify_stage(sr)
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def run_outcome_aggregate(job_results: list["JobResult"]) -> dict[str, int]:
    """§ 6.4 end-of-run aggregate: counts per job outcome (completed/failed/
    skipped)."""
    counts: dict[str, int] = {}
    for jr in job_results:
        counts[jr.outcome] = counts.get(jr.outcome, 0) + 1
    return counts


# --- § 6.5 media info + timing ----------------------------------------------

_MEDIA_INFO_FIELDS: list[tuple[str, str]] = [
    ("resolution", "resolution"),
    ("framerate", "framerate"),
    ("duration", "duration"),
    ("bit_rate", "bitrate"),
    ("video_codec", "video codec"),
    ("audio_codecs", "audio codec"),
]


def _format_field(info: dict[str, Any], key: str) -> str:
    if key not in info:
        return "?"
    value = info.get(key)
    if key == "resolution" and isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]}x{value[1]}"
    if key == "audio_codecs" and isinstance(value, list):
        return ",".join(str(v) for v in value) if value else "none"
    if key == "framerate" and isinstance(value, (int, float)):
        return f"{value:.3f}"
    if key == "duration" and isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return str(value)


def _filesize(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def format_media_info_lines(
    input_info: dict[str, Any],
    output_info: dict[str, Any] | None,
    input_path: str,
    output_path: str | None,
) -> list[str]:
    """§ 6.5 media info: one "field: in -> out" line per field (or just
    "field: in" when there's no output side yet, e.g. at job start). Pure
    formatting -- callers decide whether these become a Rich table or plain
    log lines."""
    lines = []
    for key, label in _MEDIA_INFO_FIELDS:
        in_val = _format_field(input_info, key)
        if output_info is not None:
            out_val = _format_field(output_info, key)
            lines.append(f"{label}: {in_val} -> {out_val}")
        else:
            lines.append(f"{label}: {in_val}")

    in_size = _filesize(input_path)
    if output_info is not None:
        out_size = _filesize(output_path)
        lines.append(
            f"filesize: {in_size if in_size is not None else '?'} -> "
            f"{out_size if out_size is not None else '?'}"
        )
    else:
        lines.append(f"filesize: {in_size if in_size is not None else '?'}")
    return lines


def aggregate_stage_timing(job_results: list["JobResult"]) -> dict[str, Any]:
    """§ 6.5 stage-timing summary flags: per-(base)stage totals/averages
    across the run, computed at display time from ``JobResult.stage_results``
    -- never stored (§ 6.6's non-redundancy requirement covers this too).

    Average divisor is the number of videos that actually RAN that stage
    successfully -- i.e. classification in {"ran-ai", "ran-traditional",
    "ran-traditional-fallback"} -- never the total job count; a video that
    failed at an earlier stage contributes nothing to later stages, and a
    skipped stage contributes nothing anywhere. Failed stage executions are
    excluded from totals/averages entirely and reported separately.

    Returns
    -------
    {
        "totals": {stage_name: total_ms},
        "counts": {stage_name: successful_run_count},
        "averages": {stage_name: avg_ms},
        "failed": [{"stage", "label", "video", "duration_ms", "error"}, ...],
    }
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    failed: list[dict[str, Any]] = []
    for jr in job_results:
        for label, sr in jr.stage_results.items():
            name = base_stage_name(label)
            cls = classify_stage(sr)
            if cls == "failed":
                failed.append(
                    {
                        "stage": name,
                        "label": label,
                        "video": jr.input_path,
                        "duration_ms": sr.duration_sec * 1000.0,
                        "error": sr.error,
                    }
                )
                continue
            if cls == "skipped":
                continue
            totals[name] = totals.get(name, 0.0) + sr.duration_sec * 1000.0
            counts[name] = counts.get(name, 0) + 1
    averages = {name: totals[name] / counts[name] for name in totals if counts.get(name)}
    return {"totals": totals, "counts": counts, "averages": averages, "failed": failed}


# --- § 6.6 structured JSON run report ---------------------------------------


def build_run_meta(
    avf_version: str,
    config: "Config",
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    """Run-level metadata for the § 6.6 JSON report: version, ISO timestamps,
    elapsed ms, and the effective non-default settings (reusing the same
    diff-from-DEFAULTS + redact_secrets mechanism as
    ``cli._log_effective_settings``)."""
    from autovideofixer.config import Config, diff_from_defaults

    diff = redact_secrets(diff_from_defaults(config.data, Config.DEFAULTS))
    return {
        "avf_version": avf_version,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_ms": (finished_at - started_at).total_seconds() * 1000.0,
        "effective_settings": diff,
    }


def _stage_to_json(label: str, sr: StageResult) -> dict[str, Any]:
    return {
        "label": label,
        "name": base_stage_name(label),
        "classification": classify_stage(sr),
        "method": sr.metadata.get("method"),
        "ai_fallback_used": bool(sr.metadata.get("ai_fallback_used")),
        "ai_fallback_reason": sr.metadata.get("ai_fallback_reason"),
        "duration_ms": sr.duration_sec * 1000.0,
        "skipped_reason": sr.skipped_reason,
        "error": sr.error,
        "metadata": redact_secrets(sr.metadata),
    }


def job_to_json(jr: "JobResult") -> dict[str, Any]:
    """One § 6.6 per-job JSON record. Deliberately excludes any aggregate
    (totals/averages) -- those are derivable from ``stages`` and must not be
    duplicated here (non-redundancy is a hard requirement)."""
    return {
        "input_path": jr.input_path,
        "output_path": jr.output_path,
        "outcome": jr.outcome,
        "skip_reason": jr.skip_reason,
        "reprocessed_mismatch": jr.reprocessed_mismatch,
        "decision_log": list(jr.decision_log),
        "errors": list(jr.errors),
        "input_info": redact_secrets(jr.input_info),
        "output_info": redact_secrets(jr.output_info),
        "scene_stats": jr.scene_stats,
        "job_wall_ms": jr.job_wall_ms,
        "processing_ms": jr.processing_ms,
        "quality_score": jr.quality_score,
        "quality_meets_target": jr.quality_meets_target,
        "stages": [_stage_to_json(label, sr) for label, sr in jr.stage_results.items()],
    }


def build_json_report(run_meta: dict[str, Any], job_results: list["JobResult"]) -> dict[str, Any]:
    """Assemble the full § 6.6 JSON run report document."""
    return {
        "run": run_meta,
        "jobs": [job_to_json(jr) for jr in job_results],
    }


def write_json_report(path: str, report: dict[str, Any]) -> None:
    """Serialize and write the § 6.6 JSON report.

    ``default=str`` is a deliberate robustness fallback: stage ``metadata``
    dicts are free-form and may contain non-JSON-serializable values (e.g. a
    tuple already handled by ``json`` as a list, but also things like
    ``Path`` objects or numpy scalars a stage might stash) -- rather than the
    whole report failing to write, such a value survives as its ``str()``.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
