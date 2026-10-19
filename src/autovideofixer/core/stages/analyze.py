"""Auto Video Fixer - Analysis stage (docs/REQUIREMENTS.md § 28.4).

Runs the reusable analysis module (``core/analysis_service.py``) on the stage's input and
publishes the result for later stages and the job report:

- ``input_info["analysis"]`` -- ``AnalysisResult.to_dict()`` (times in THIS file's timeline);
- ``input_info["analysis_time_scale"]`` -- 1.0 here; the pipeline multiplies it by the
  ``time_scale`` of every later stage that rescales time (``speed``), so a consumer can map
  analysis times onto the output with ``analysis_service.map_time()``.

Off by default. Produces no file. Fail-open: an analysis that raises is one WARNING and a
SKIPPED stage; a VLM that is down just leaves the VLM fields empty (its failed calls are in
``jobs[].vlm_calls`` like every other call).
"""

from __future__ import annotations

import time
from typing import Any

from autovideofixer.config import Config
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus


class AnalyzeStage(BaseStage):
    """Analyse the current intermediate (scenes, optional VLM) and publish the result."""

    name = "analyze"
    display_name = "Content Analysis"
    description = "Scene detection and VLM description, published for later stages and reports"
    category = "analysis"
    priority = 5
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
        self._report_progress(0.0, "Analysing content...", progress_callback)
        from autovideofixer.core.analysis_service import analyze_video

        profile = self._stage_config.get("profile")
        scope = self._stage_config.get("scope")
        planned = list((input_info or {}).get("planned_stages") or [])
        if (
            scope is None
            and self.name in planned
            and "scene_subtitles" in planned[planned.index(self.name) + 1 :]
        ):
            # A later scene_subtitles needs per-scene results. Getting them here (before speed,
            # mapped through analysis_time_scale) beats it re-analysing a sped-up intermediate,
            # where shorter scenes fall under min_scene_duration_sec and merge.
            from autovideofixer.core.analysis_service import resolve_analysis_config

            try:
                eff_scope = resolve_analysis_config(self.config, profile).get(
                    "analysis", "vlm", "scope", default="video"
                )
            except Exception:
                eff_scope = None  # e.g. an unknown profile: analyze_video below reports it
            if eff_scope == "video":
                scope = "both"
                self.logger.info(
                    "analyze: scope 'both' (scene_subtitles runs later and needs per-scene "
                    "results; set stages.analyze.scope to override)"
                )

        def _progress(phase: str, detail: str) -> None:
            self.logger.debug("analyze: %s %s", phase, detail)

        try:
            result = analyze_video(
                input_path,
                self.config,
                profile=profile,
                scope=scope,
                cache=self.analysis_cache,
                progress_callback=_progress,
            )
        except Exception as e:
            self.logger.warning("analyze: analysis failed (%s: %s); skipping", type(e).__name__, e)
            self.logger.debug("analyze failure detail", exc_info=True)
            return StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason=f"analysis failed: {type(e).__name__}",
                duration_sec=time.time() - start,
            )

        doc = result.to_dict()
        if input_info is not None:
            input_info["analysis"] = doc
            input_info["analysis_time_scale"] = 1.0
        calls = doc.get("vlm_calls", [])
        self._report_progress(1.0, "Analysis complete", progress_callback)
        return StageResult(
            status=StageStatus.COMPLETED,
            metadata={
                "method": "analysis",
                "profile": result.profile,
                "scope": result.scope,
                "scenes": doc.get("scenes_detected", 0),
                "vlm_calls": len(calls),
                "vlm_failed_calls": sum(1 for c in calls if not c.get("success", True)),
                "cached": result.cached,
            },
            duration_sec=time.time() - start,
        )
