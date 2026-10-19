"""Auto Video Fixer - Scene subtitles stage (docs/REQUIREMENTS.md § 28.5).

Turns the per-scene VLM descriptions into an SRT and delivers it in one of three ways
(``stages.scene_subtitles.mode``):

- ``sidecar`` -- ``<final output stem><suffix>`` (default ``.srt``) next to the deliverable.
- ``soft`` -- a text track muxed into the file: ``mov_text`` for MP4/M4V/MOV, ``srt`` for
  MKV. Done with stream copy (no extra generation); ``encode`` then re-muxes it with the
  matching ``-c:s`` (``input_info["soft_subtitle_codec"]``). Other containers (WebM, ...)
  degrade to ``sidecar`` with a WARNING.
- ``burn`` -- drawn onto the picture with the ``subtitles`` (libass) filter. This is a real
  re-encode (libx264 at ``crf``, ``-fps_mode passthrough`` so a VFR intermediate stays VFR) and
  therefore one extra lossy generation before ``encode``; it degrades to ``soft`` with a
  WARNING when this ffmpeg build has no ``subtitles`` filter.

Placement: after ``speed`` and ``hdr`` (the text is not sped up and is drawn on tone-mapped
pixels), before ``encode``. It needs per-scene descriptions (analysis scope ``scenes``/``both``
and the VLM enabled); without them it SKIPs with a WARNING. Cue times come from the analysis
mapped through ``input_info["analysis_time_scale"]`` (speed) and are clamped to the current
duration.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any

from autovideofixer.config import Config
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus

_SOFT_CODECS = {".mp4": "mov_text", ".m4v": "mov_text", ".mov": "mov_text", ".mkv": "srt"}


@dataclass
class Cue:
    start: float
    end: float
    text: str


def srt_time(seconds: float) -> str:
    """``HH:MM:SS,mmm`` (rounded to the millisecond, never negative)."""
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _clean_text(text: str, max_chars: int, wrap_chars: int) -> str:
    # Braces and angle brackets are override/markup syntax for libass and SRT players.
    for ch in "{}<>":
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,;:.") + "..."
    if wrap_chars > 0:
        return "\n".join(textwrap.wrap(text, wrap_chars)) or text
    return text


def build_cues(
    analysis: dict[str, Any],
    time_scale: float = 1.0,
    duration: float | None = None,
    *,
    max_chars: int = 200,
    wrap_chars: int = 42,
    min_cue_sec: float = 0.5,
) -> list[Cue]:
    """Cues from the per-scene summaries of ``analysis`` (an ``AnalysisResult.to_dict()``).

    Scene times are multiplied by ``time_scale`` (output seconds per analysed second) and the
    end is clamped to ``duration`` (the current file's, when known). Scenes without a
    description, or shorter than ``min_cue_sec`` after mapping, get no cue.
    """
    cues: list[Cue] = []
    for sc in analysis.get("scenes", []):
        summary = ((sc.get("vlm") or {}).get("summary") or "").strip()
        if not summary:
            continue
        start = float(sc["start_time"]) * time_scale
        end = float(sc["end_time"]) * time_scale
        if duration is not None and duration > 0:
            end = min(end, duration)
        if end - start < min_cue_sec:
            continue
        text = _clean_text(summary, max_chars, wrap_chars)
        if text:
            cues.append(Cue(start, end, text))
    return cues


def format_srt(cues: list[Cue]) -> str:
    return "".join(
        f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n\n"
        for i, c in enumerate(cues, start=1)
    )


def escape_filter_value(value: str) -> str:
    """Escape ``value`` for use as an option value inside an ffmpeg ``-vf`` string.

    Two escaping levels apply (ffmpeg-filters "Notes on filtergraph escaping"): first the
    filter's own option syntax (``\\`` ``'`` ``:`` ``=``), then the filtergraph syntax that
    contains it (``\\`` ``'`` ``[`` ``]`` ``,`` ``;``). Applied in that order so the second
    pass also escapes the backslashes the first one added.
    """
    for ch in ("\\", "'", ":", "="):
        value = value.replace(ch, "\\" + ch)
    for ch in ("\\", "'", "[", "]", ",", ";"):
        value = value.replace(ch, "\\" + ch)
    return value


def subtitles_filter(srt_path: str, style: str = "") -> str:
    """The ``subtitles=...`` video filter for ``srt_path`` (absolute path, fully escaped)."""
    flt = f"subtitles=filename={escape_filter_value(os.path.abspath(srt_path))}"
    if style:
        flt += f":force_style={escape_filter_value(style)}"
    return flt


class SceneSubtitlesStage(BaseStage):
    """SRT of per-scene descriptions: sidecar, soft track or burned in."""

    name = "scene_subtitles"
    display_name = "Scene Subtitles"
    description = "Subtitles from the per-scene content descriptions"
    category = "output"
    priority = 84
    requires_input = True
    produces_output = True
    can_parallelize = False

    def __init__(self, config: Config, overrides: dict[str, Any] | None = None) -> None:
        super().__init__(config, overrides)

    def _skip(self, start: float, reason: str) -> StageResult:
        return StageResult(
            status=StageStatus.SKIPPED, skipped_reason=reason, duration_sec=time.time() - start
        )

    def execute(
        self,
        input_path: str,
        output_path: str | None = None,
        progress_callback: Any = None,
        input_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        start = time.time()
        info = input_info or {}
        mode = str(self._stage_config.get("mode", "sidecar"))
        if mode not in ("sidecar", "soft", "burn"):
            self.logger.warning(
                "scene_subtitles: invalid mode %r (sidecar | soft | burn); skipping", mode
            )
            return self._skip(start, f"invalid mode {mode!r}")
        final = info.get("final_output_path")
        if not final:
            return self._skip(start, "final output path unknown")

        from autovideofixer.core.analysis_service import obtain_analysis

        got = obtain_analysis(
            self.config,
            self.name,
            self._stage_config,
            input_path,
            info,
            self.analysis_cache,
            need_scenes=True,
        )
        if got is None:
            return self._skip(start, "no analysis available")
        analysis, scale = got
        cues = build_cues(
            analysis,
            scale,
            float(info.get("duration") or 0) or None,
            max_chars=int(self._stage_config.get("max_chars", 200)),
            wrap_chars=int(self._stage_config.get("wrap_chars", 42)),
            min_cue_sec=float(self._stage_config.get("min_cue_sec", 0.5)),
        )
        if not cues:
            self.logger.warning(
                "scene_subtitles: the analysis has no per-scene descriptions (needs "
                "analysis.vlm.enabled and a scope of 'scenes' or 'both', e.g. "
                "stages.analyze.scope: scenes); skipping"
            )
            return self._skip(start, "no per-scene descriptions")

        mode = self._degrade(mode, final)
        srt = format_srt(cues)
        if mode == "sidecar":
            return self._sidecar(start, srt, final, input_path, len(cues))
        if not output_path:
            return StageResult(
                status=StageStatus.FAILED,
                error="no output_path provided to scene_subtitles stage",
                duration_sec=time.time() - start,
            )
        return self._mux_or_burn(
            start, mode, srt, input_path, output_path, final, len(cues), info, progress_callback
        )

    def _degrade(self, mode: str, final: str) -> str:
        """burn -> soft (no libass) -> sidecar (no suitable container), each with a WARNING."""
        if mode == "burn":
            from autovideofixer.core.ffmpeg_utils import has_filter

            if not has_filter("subtitles", self.config):
                self.logger.warning(
                    "scene_subtitles: mode=burn needs ffmpeg's 'subtitles' filter (libass), "
                    "which this build lacks; falling back to mode=soft"
                )
                mode = "soft"
        if mode == "soft" and os.path.splitext(final)[1].lower() not in _SOFT_CODECS:
            self.logger.warning(
                "scene_subtitles: mode=soft cannot be muxed into a %s output (supported: "
                "MP4/M4V/MOV, MKV); falling back to mode=sidecar",
                os.path.splitext(final)[1] or "extension-less",
            )
            mode = "sidecar"
        return mode

    def _sidecar(
        self, start: float, srt: str, final: str, input_path: str, cues: int
    ) -> StageResult:
        target = os.path.splitext(final)[0] + str(self._stage_config.get("suffix", ".srt"))
        if os.path.exists(target) and not self.config.get("general", "overwrite", default=False):
            self.logger.warning(
                "scene_subtitles: %s already exists and general.overwrite is false; not "
                "replacing it",
                os.path.basename(target),
            )
            return self._skip(start, "subtitle file exists (general.overwrite is false)")
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp = f"{target}.avf-tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(srt)
            os.replace(tmp, target)
        except OSError as e:
            self.logger.warning("scene_subtitles: could not write %s: %s", target, e)
            return self._skip(start, f"write failed: {e.__class__.__name__}")
        self.logger.info("scene_subtitles: wrote %s (%d cue(s))", os.path.basename(target), cues)
        # No new file: the next stage keeps reading input_path. `produces_output` is a class
        # attribute (True for soft/burn); this instance produced none, so the pipeline must
        # not treat "COMPLETED without an output_path" as a failure. (Returning input_path as
        # the output instead would get the job's SOURCE deleted by the temp-file cleanup.)
        self.produces_output = False
        return StageResult(
            status=StageStatus.COMPLETED,
            metadata={"method": "sidecar", "cues": cues, "sidecar": target},
            duration_sec=time.time() - start,
        )

    def _mux_or_burn(
        self,
        start: float,
        mode: str,
        srt: str,
        input_path: str,
        output_path: str,
        final: str,
        cues: int,
        info: dict[str, Any],
        progress_callback: Any,
    ) -> StageResult:
        from autovideofixer.core.ffmpeg_utils import run_ffmpeg, timing_output_args

        temp_dir = self.config.get("general", "temp_dir", default=None)
        work = tempfile.mkdtemp(prefix="avf_subs_", dir=temp_dir or os.path.dirname(output_path))
        try:
            srt_path = os.path.join(work, "scenes.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt)
            codec: str | None = None
            if mode == "burn":
                args = [
                    "-i",
                    input_path,
                    "-vf",
                    subtitles_filter(srt_path, str(self._stage_config.get("style", ""))),
                    "-c:v",
                    "libx264",
                    "-preset",
                    str(self._stage_config.get("preset", "medium")),
                    "-crf",
                    str(self._stage_config.get("crf", 16)),
                    "-c:a",
                    "copy",
                    "-sn",
                    # Same VFR-safety rule as every filter stage (AGENTS.md "Timestamp safety").
                    *timing_output_args(),
                ]
            else:
                codec = _SOFT_CODECS[os.path.splitext(final)[1].lower()]
                # Stream copy: no re-encode, nothing for -fps_mode to govern. Existing
                # subtitle streams are not carried (encode's default selection would pick
                # only one anyway, and ours must be it).
                args = [
                    "-i",
                    input_path,
                    "-i",
                    srt_path,
                    "-map",
                    "0:v",
                    "-map",
                    "0:a?",
                    "-map",
                    "1:0",
                    "-c",
                    "copy",
                    # The intermediate is always .mkv (generate_temp_path), whose text codec is
                    # srt; encode re-muxes with the final container's codec.
                    "-c:s",
                    "srt",
                    "-metadata:s:s:0",
                    f"language={self._stage_config.get('language', 'eng')}",
                    "-metadata:s:s:0",
                    "title=Scene descriptions",
                ]
            args += ["-y", output_path]
            result = run_ffmpeg(
                args,
                progress_callback=lambda p, m: self._report_progress(p, m, progress_callback),
                timeout=self.stage_timeout(),
            )
            if result.returncode != 0:
                return StageResult(
                    status=StageStatus.FAILED,
                    error=f"scene_subtitles ({mode}) failed: {result.stderr[:300]}",
                    duration_sec=time.time() - start,
                )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        metadata: dict[str, Any] = {"method": mode, "cues": cues}
        if codec is not None:
            info["soft_subtitle_codec"] = codec
            metadata["subtitle_codec"] = codec
        return StageResult(
            status=StageStatus.COMPLETED,
            output_path=output_path,
            metadata=metadata,
            duration_sec=time.time() - start,
        )
