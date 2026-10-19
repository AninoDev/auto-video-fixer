"""Auto Video Fixer - Analysis sidecar stage (docs/REQUIREMENTS.md § 28.5).

Writes the analysis as JSON beside the FINAL output (``<output><suffix>``, default
``clip_enhanced.mp4.avf-analysis.json``). Reference implementation of a stage that consumes
the analysis -- see docs/DEVELOPER.md "Writing a stage that uses analysis".

It runs just before ``encode`` (the encode stage is always last and the intermediates it is
handed are temp files), and takes the destination from ``input_info["final_output_path"]``,
which the pipeline injects. Times in the sidecar are in the analysed file's timeline; each
scene also carries ``output_start_time`` / ``output_end_time`` mapped onto the output's
(``analysis_service.map_time``), which differ when ``speed`` runs after ``analyze``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from autovideofixer import __version__
from autovideofixer.config import Config
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

SIDECAR_SCHEMA = 1


class AnalysisSidecarStage(BaseStage):
    """Write ``<final output><suffix>`` containing the analysis JSON."""

    name = "analysis_sidecar"
    display_name = "Analysis Sidecar"
    description = "Write the content analysis as a JSON file beside the final output"
    category = "output"
    priority = 85
    requires_input = True
    produces_output = False
    can_parallelize = False

    def __init__(self, config: Config, overrides: dict[str, Any] | None = None) -> None:
        super().__init__(config, overrides)

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback: Any = None,
        input_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        start = time.time()
        final = (input_info or {}).get("final_output_path")
        if not final:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="final output path unknown",
                duration_sec=time.time() - start,
            )
        target = f"{final}{self._stage_config.get('suffix', '.avf-analysis.json')}"
        if os.path.exists(target) and not self.config.get("general", "overwrite", default=False):
            self.logger.warning(
                "analysis_sidecar: %s already exists and general.overwrite is false; not "
                "replacing it",
                os.path.basename(target),
            )
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="sidecar exists (general.overwrite is false)",
                duration_sec=time.time() - start,
            )

        from autovideofixer.core.analysis_service import obtain_analysis

        got = obtain_analysis(
            self.config,
            self.name,
            self._stage_config,
            input_path,
            input_info,
            self.analysis_cache,
        )
        if got is None:
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="no analysis available",
                duration_sec=time.time() - start,
            )
        analysis, scale = got
        try:
            doc = build_sidecar(analysis, scale, os.path.basename(final))
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp = f"{target}.avf-tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            os.replace(tmp, target)
        except OSError as e:
            self.logger.warning("analysis_sidecar: could not write %s: %s", target, e)
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=f"write failed: {e.__class__.__name__}",
                duration_sec=time.time() - start,
            )
        self.logger.info("analysis_sidecar: wrote %s", os.path.basename(target))
        return StageResult(
            status=StageStatus.COMPLETED,
            metadata={"method": "sidecar", "sidecar": target},
            duration_sec=time.time() - start,
        )


def build_sidecar(analysis: dict[str, Any], time_scale: float, output_name: str) -> dict[str, Any]:
    """The sidecar document for ``analysis`` (an ``AnalysisResult.to_dict()``).

    Paths of the analysed intermediate are dropped (they are internal temp files). Each scene
    gains ``output_start_time`` / ``output_end_time``; ``timeline.analysis_time_scale`` is the
    factor used (output seconds per analysed second).
    """
    body = {k: v for k, v in analysis.items() if k not in ("path", "filepath", "filename")}
    scenes = []
    for sc in body.get("scenes", []):
        sc = dict(sc)
        sc["output_start_time"] = sc["start_time"] * time_scale
        sc["output_end_time"] = sc["end_time"] * time_scale
        scenes.append(sc)
    body["scenes"] = scenes
    return {
        "avf_sidecar_version": SIDECAR_SCHEMA,
        "avf_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": output_name,
        "timeline": {
            "analysis_time_scale": time_scale,
            "note": "start_time/end_time are in the analysed file's timeline; "
            "output_start_time/output_end_time are mapped onto the output.",
        },
        "analysis": body,
    }
