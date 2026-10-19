"""§ 28: the ``analyze`` stage and the two example stages (``analysis_sidecar``,
``scene_subtitles``), the v2 -> v3 config migration and the stage-order plumbing.

ffmpeg is mocked here (argument lists are asserted); the real-ffmpeg runs -- including speed
!= 1 and a path full of filter-graph special characters -- are in
``tests/integration/test_analysis_stages_e2e.py``.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import yaml

from autovideofixer import config_migrations as cm
from autovideofixer.config import Config
from autovideofixer.core.analysis_service import AnalysisCache
from autovideofixer.core.pipeline import DEFAULT_STAGE_ORDER, Job, Pipeline
from autovideofixer.core.stages.analysis_sidecar import AnalysisSidecarStage, build_sidecar
from autovideofixer.core.stages.analyze import AnalyzeStage
from autovideofixer.core.stages.base import StageStatus, get_stage
from autovideofixer.core.stages.scene_subtitles import (
    SceneSubtitlesStage,
    build_cues,
    escape_filter_value,
    format_srt,
    srt_time,
    subtitles_filter,
)


def _analysis(summaries=("a cat", "a dog", "")):
    scenes = []
    for i, (s, text) in enumerate(zip([0.0, 4.0, 10.0], summaries)):
        end = [4.0, 10.0, 20.0][i]
        scenes.append(
            {
                "start_time": s,
                "end_time": end,
                "event_type": "scene_change",
                "confidence": 0.3,
                "description": None,
                "vlm": {"summary": text} if text else None,
            }
        )
    return {"scope": "scenes", "profile": None, "scenes": scenes, "scenes_detected": 3,
            "path": "/tmp/avf_internal.mkv", "filename": "avf_internal.mkv",
            "vlm_calls": []}  # fmt: skip


def _cfg(tmp_path, **extra) -> Config:
    cfg = Config(tmp_path / "none.yaml")
    for key, value in extra.items():
        cfg.set(value, *key.split("."))
    return cfg


# ─── cues / SRT / escaping ─────────────────────────────────────────


class TestCues:
    def test_srt_time(self):
        assert srt_time(0) == "00:00:00,000"
        assert srt_time(3661.0075) == "01:01:01,008"
        assert srt_time(-4) == "00:00:00,000"

    def test_cues_skip_scenes_without_a_description(self):
        cues = build_cues(_analysis())
        assert [(c.start, c.end, c.text) for c in cues] == [
            (0.0, 4.0, "a cat"),
            (4.0, 10.0, "a dog"),
        ]

    def test_times_are_mapped_through_the_scale_and_clamped(self):
        cues = build_cues(_analysis(("x", "y", "z")), 0.5, duration=7.0)
        assert [(c.start, c.end) for c in cues] == [(0.0, 2.0), (2.0, 5.0), (5.0, 7.0)]

    def test_short_cues_dropped_after_mapping(self):
        assert build_cues(_analysis(("x", "y", "z")), 0.1, min_cue_sec=0.5) == build_cues(
            _analysis(("x", "y", "z")), 0.1, min_cue_sec=0.5
        )
        cues = build_cues(_analysis(("x", "y", "z")), 0.1, min_cue_sec=0.5)
        assert [c.text for c in cues] == ["y", "z"]  # the 0.4 s one is gone

    def test_text_is_cleaned_wrapped_and_truncated(self):
        long = "word " * 80
        (cue,) = build_cues(
            {"scenes": [{"start_time": 0, "end_time": 5,
                         "vlm": {"summary": "{\\an8}<b>hi</b> " + long}}]},
            max_chars=60,
            wrap_chars=20,
        )  # fmt: skip
        assert "{" not in cue.text and "<" not in cue.text
        assert cue.text.endswith("...") and len(cue.text.replace("\n", " ")) <= 60
        assert all(len(line) <= 20 for line in cue.text.split("\n"))

    def test_format_srt(self):
        text = format_srt(build_cues(_analysis()))
        assert text.startswith("1\n00:00:00,000 --> 00:00:04,000\na cat\n\n2\n00:00:04,000 --> ")


class TestEscaping:
    def test_each_special_character_is_escaped_at_both_levels(self):
        assert escape_filter_value("a b") == "a b"
        assert escape_filter_value("a:b") == r"a\\:b"
        assert escape_filter_value("a=b") == r"a\\=b"
        assert escape_filter_value("a,b") == r"a\,b"
        assert escape_filter_value("a'b") == r"a\\\'b"
        assert escape_filter_value("[x];y") == r"\[x\]\;y"
        assert escape_filter_value("a\\b") == r"a\\\\b"

    def test_filter(self):
        flt = subtitles_filter("/t/a b/c:d.srt", "Fontsize=16,Outline=1")
        assert flt == (
            r"subtitles=filename=/t/a b/c\\:d.srt"
            r":force_style=Fontsize\\=16\,Outline\\=1"
        )


# ─── analyze stage ─────────────────────────────────────────────────


def _stage(cls, cfg, cache=None):
    stage = cls(cfg)
    stage._force_enabled = True
    stage.analysis_cache = cache
    return stage


class TestAnalyzeStage:
    def test_registered_and_off_by_default(self, tmp_path):
        assert get_stage("analyze") is AnalyzeStage and AnalyzeStage.produces_output is False
        assert AnalyzeStage(_cfg(tmp_path)).should_run({}) == (
            False,
            "Stage disabled in configuration",
        )

    def test_publishes_the_analysis_and_scale(self, tmp_path):
        info: dict = {}
        fake = type("R", (), {"profile": "quick", "scope": "video", "cached": False,
                              "to_dict": lambda self: {"scenes_detected": 2, "vlm_calls": [
                                  {"success": True}, {"success": False}]}})()  # fmt: skip
        with patch("autovideofixer.core.analysis_service.analyze_video", return_value=fake) as av:
            res = _stage(
                AnalyzeStage, _cfg(tmp_path, **{"stages.analyze.profile": "quick"})
            ).execute("/x.mp4", None, input_info=info)
        assert res.status == StageStatus.COMPLETED and res.output_path is None
        assert info["analysis"]["scenes_detected"] == 2 and info["analysis_time_scale"] == 1.0
        assert res.metadata["vlm_calls"] == 2 and res.metadata["vlm_failed_calls"] == 1
        assert av.call_args.kwargs["profile"] == "quick"

    def test_fail_open_is_one_warning_without_a_traceback(self, tmp_path, caplog):
        with (
            patch(
                "autovideofixer.core.analysis_service.analyze_video", side_effect=RuntimeError("no")
            ),
            caplog.at_level("WARNING"),
        ):
            res = _stage(AnalyzeStage, _cfg(tmp_path)).execute("/x.mp4", None, input_info={})
        assert res.status == StageStatus.SKIPPED and "RuntimeError" in res.skipped_reason
        warnings = [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]
        assert len(warnings) == 1 and warnings[0].exc_info is None

    def test_unknown_profile_is_fail_open_too(self, tmp_path, caplog):
        cfg = _cfg(tmp_path, **{"stages.analyze.profile": "nope"})
        for info in ({}, {"planned_stages": ["analyze", "scene_subtitles"]}):
            with caplog.at_level("WARNING"):
                res = _stage(AnalyzeStage, cfg).execute("/x.mp4", None, input_info=info)
            assert res.status == StageStatus.SKIPPED and "Unknown analysis profile" in caplog.text

    @pytest.mark.parametrize(
        ("planned", "extra", "expected"),
        [
            (["crop", "analyze", "speed", "scene_subtitles", "encode"], {}, "both"),
            (["analyze", "encode"], {}, None),  # no scene consumer: unchanged
            (["scene_subtitles", "analyze"], {}, None),  # consumer runs before: unchanged
            (["analyze", "scene_subtitles"], {"stages.analyze.scope": "video"}, "video"),
            (["analyze", "scene_subtitles"], {"analysis.vlm.scope": "scenes"}, None),
        ],
    )
    def test_scope_includes_scenes_when_scene_subtitles_runs_later(
        self, tmp_path, planned, extra, expected
    ):
        fake = type("R", (), {"profile": None, "scope": "x", "cached": False,
                              "to_dict": lambda self: {"vlm_calls": []}})()  # fmt: skip
        with patch("autovideofixer.core.analysis_service.analyze_video", return_value=fake) as av:
            _stage(AnalyzeStage, _cfg(tmp_path, **extra)).execute(
                "/x.mp4", None, input_info={"planned_stages": planned}
            )
        assert av.call_args.kwargs["scope"] == expected


# ─── sidecar ───────────────────────────────────────────────────────


class TestSidecar:
    def _run(self, tmp_path, scale=1.0, **cfg):
        final = tmp_path / "out" / "clip_enhanced.mp4"
        final.parent.mkdir()
        info = {
            "final_output_path": str(final),
            "analysis": _analysis(),
            "analysis_time_scale": scale,
        }
        stage = _stage(AnalysisSidecarStage, _cfg(tmp_path, **cfg))
        return stage.execute("/tmp/intermediate.mkv", None, input_info=info), final

    def test_written_beside_the_final_output_with_mapped_times(self, tmp_path):
        res, final = self._run(tmp_path, scale=0.5)
        side = final.parent / "clip_enhanced.mp4.avf-analysis.json"
        assert res.status == StageStatus.COMPLETED and side.exists()
        doc = json.loads(side.read_text())
        assert (
            doc["output"] == "clip_enhanced.mp4" and doc["timeline"]["analysis_time_scale"] == 0.5
        )
        sc = doc["analysis"]["scenes"][1]
        assert (sc["start_time"], sc["end_time"]) == (4.0, 10.0)
        assert (sc["output_start_time"], sc["output_end_time"]) == (2.0, 5.0)
        # internal temp paths never leak into a deliverable
        assert not {"path", "filepath", "filename"} & set(doc["analysis"])
        assert not list(final.parent.glob("*.avf-tmp"))

    def test_existing_sidecar_is_not_clobbered_unless_overwrite(self, tmp_path, caplog):
        _, final = self._run(tmp_path)
        side = final.parent / "clip_enhanced.mp4.avf-analysis.json"
        side.write_text("mine")
        with caplog.at_level("WARNING"):
            res, _ = self._run(tmp_path) if False else (None, None)
        info = {"final_output_path": str(final), "analysis": _analysis()}
        with caplog.at_level("WARNING"):
            res = _stage(AnalysisSidecarStage, _cfg(tmp_path)).execute(
                "/t.mkv", None, input_info=info
            )
        assert res.status == StageStatus.SKIPPED and side.read_text() == "mine"
        assert "general.overwrite is false" in caplog.text
        res = _stage(AnalysisSidecarStage, _cfg(tmp_path, **{"general.overwrite": True})).execute(
            "/t.mkv", None, input_info=info
        )
        assert res.status == StageStatus.COMPLETED and json.loads(side.read_text())["analysis"]

    def test_custom_suffix(self, tmp_path):
        res, final = self._run(tmp_path, **{"stages.analysis_sidecar.suffix": ".json"})
        assert (
            final.parent / "clip_enhanced.mp4.json"
        ).exists() and res.status == StageStatus.COMPLETED

    def test_no_analysis_and_not_allowed_skips_with_a_warning(self, tmp_path, caplog):
        info = {"final_output_path": str(tmp_path / "o.mp4")}
        cfg = _cfg(tmp_path, **{"stages.analysis_sidecar.analyze_if_missing": False})
        with caplog.at_level("WARNING"):
            res = _stage(AnalysisSidecarStage, cfg).execute("/t.mkv", None, input_info=info)
        assert res.status == StageStatus.SKIPPED and "no analysis available" in caplog.text
        assert not (tmp_path / "o.mp4.avf-analysis.json").exists()

    def test_runs_the_module_itself_through_the_job_cache(self, tmp_path):
        final = tmp_path / "o.mp4"
        calls = []

        def fake(path, config, **kw):
            calls.append(kw["cache"])
            return type("R", (), {"to_dict": lambda self: _analysis()})()

        cache = AnalysisCache()
        with patch("autovideofixer.core.analysis_service.analyze_video", fake):
            res = _stage(AnalysisSidecarStage, _cfg(tmp_path), cache).execute(
                "/t.mkv", None, input_info={"final_output_path": str(final)}
            )
        assert res.status == StageStatus.COMPLETED and calls == [cache]

    def test_unknown_final_path_skips(self, tmp_path):
        res = _stage(AnalysisSidecarStage, _cfg(tmp_path)).execute("/t.mkv", None, input_info={})
        assert res.status == StageStatus.SKIPPED

    def test_build_sidecar_is_pure(self):
        a = _analysis()
        build_sidecar(a, 2.0, "o.mp4")
        assert "output_start_time" not in a["scenes"][0] and "path" in a


# ─── scene subtitles ───────────────────────────────────────────────


class _Run:
    returncode = 0
    stderr = ""


class TestSceneSubtitles:
    def _stage(self, tmp_path, **cfg):
        return _stage(SceneSubtitlesStage, _cfg(tmp_path, **cfg))

    def _info(self, tmp_path, name="o.mp4", **extra):
        return {"final_output_path": str(tmp_path / name), "analysis": _analysis(), **extra}

    def test_sidecar_mode_writes_the_srt_and_passes_the_file_through(self, tmp_path):
        info = self._info(tmp_path)
        res = self._stage(tmp_path).execute("/t.mkv", "/t2.mkv", input_info=info)
        srt = tmp_path / "o.srt"
        assert res.status == StageStatus.COMPLETED and res.output_path is None
        assert srt.read_text().startswith("1\n00:00:00,000 --> 00:00:04,000\na cat")
        assert res.metadata["cues"] == 2

    def test_sidecar_mode_is_not_an_output_producing_run(self, tmp_path):
        stage = self._stage(tmp_path)
        assert stage.produces_output is True  # soft / burn
        stage.execute("/t.mkv", "/t2.mkv", input_info=self._info(tmp_path))
        assert stage.produces_output is False  # this instance wrote no media file

    def test_cleanup_never_removes_the_jobs_source(self, tmp_path):
        from autovideofixer.core.stages.base import StageResult

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        job = Job(input_path=str(src), output_path=str(tmp_path / "o.mp4"))
        results = {"s": StageResult(status=StageStatus.COMPLETED, output_path=str(src))}
        Pipeline(_cfg(tmp_path))._cleanup_temp_files(["s"], results, job, keep=set())
        assert src.exists()

    def test_cleanup_never_removes_the_source_by_another_spelling(self, tmp_path):
        from autovideofixer.core.stages.base import StageResult

        src = tmp_path / "src.mp4"
        src.write_bytes(b"x")
        link = tmp_path / "link.mp4"
        link.symlink_to(src)
        job = Job(input_path=str(src), output_path=str(tmp_path / "o.mp4"))
        for spelling in (str(link), str(tmp_path / "." / "src.mp4")):
            results = {"s": StageResult(status=StageStatus.COMPLETED, output_path=spelling)}
            Pipeline(_cfg(tmp_path))._cleanup_temp_files(["s"], results, job, keep=set())
            assert src.exists() and os.path.lexists(spelling)

    def test_sidecar_is_not_clobbered(self, tmp_path, caplog):
        (tmp_path / "o.srt").write_text("mine")
        with caplog.at_level("WARNING"):
            res = self._stage(tmp_path).execute(
                "/t.mkv", "/t2.mkv", input_info=self._info(tmp_path)
            )
        assert res.status == StageStatus.SKIPPED and (tmp_path / "o.srt").read_text() == "mine"

    def test_speed_scale_changes_the_cue_times(self, tmp_path):
        info = self._info(tmp_path, analysis_time_scale=0.5, duration=10.0)
        self._stage(tmp_path).execute("/t.mkv", "/t2.mkv", input_info=info)
        text = (tmp_path / "o.srt").read_text()
        assert "00:00:00,000 --> 00:00:02,000" in text and "00:00:02,000 --> 00:00:05,000" in text

    def test_no_descriptions_skips_with_a_hint(self, tmp_path, caplog):
        info = self._info(tmp_path)
        info["analysis"] = _analysis(("", "", ""))
        with caplog.at_level("WARNING"):
            res = self._stage(tmp_path).execute("/t.mkv", "/t2.mkv", input_info=info)
        assert res.status == StageStatus.SKIPPED and "scope" in caplog.text

    def test_invalid_mode_skips(self, tmp_path):
        res = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "tattoo"}).execute(
            "/t.mkv", "/t2.mkv", input_info=self._info(tmp_path)
        )
        assert res.status == StageStatus.SKIPPED

    @pytest.mark.parametrize(
        "name,codec",
        [("o.mp4", "mov_text"), ("o.m4v", "mov_text"), ("o.mov", "mov_text"), ("o.mkv", "srt")],
    )
    def test_soft_is_a_stream_copy_with_the_containers_codec_hint(self, tmp_path, name, codec):
        info = self._info(tmp_path, name)
        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "soft"})
        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=_Run()) as rf:
            res = stage.execute("/t.mkv", str(tmp_path / "temp.mkv"), input_info=info)
        args = rf.call_args.args[0]
        assert res.status == StageStatus.COMPLETED and info["soft_subtitle_codec"] == codec
        assert args[args.index("-c") + 1] == "copy" and "libx264" not in args
        assert "-fps_mode" not in args  # nothing is re-timed: no encoder to govern
        assert args[args.index("-c:s") + 1] == "srt"  # the intermediate is MKV
        assert not [p for p in os.listdir(tmp_path) if p.startswith("avf_subs_")]

    def test_soft_in_webm_degrades_to_sidecar_with_a_warning(self, tmp_path, caplog):
        info = self._info(tmp_path, "o.webm")
        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "soft"})
        with caplog.at_level("WARNING"), patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg") as rf:
            res = stage.execute("/t.mkv", str(tmp_path / "temp.mkv"), input_info=info)
        rf.assert_not_called()
        assert (tmp_path / "o.srt").exists() and res.metadata["method"] == "sidecar"
        assert "falling back to mode=sidecar" in caplog.text
        assert "soft_subtitle_codec" not in info

    def test_burn_reencodes_with_the_subtitles_filter_and_keeps_vfr(self, tmp_path):
        info = self._info(tmp_path)
        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "burn"})
        with (
            patch("autovideofixer.core.ffmpeg_utils.has_filter", return_value=True),
            patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=_Run()) as rf,
        ):
            res = stage.execute("/t.mkv", str(tmp_path / "temp.mkv"), input_info=info)
        args = rf.call_args.args[0]
        vf = args[args.index("-vf") + 1]
        assert res.metadata["method"] == "burn" and vf.startswith("subtitles=filename=")
        assert args[args.index("-fps_mode") + 1] == "passthrough"
        assert args[args.index("-c:v") + 1] == "libx264" and args[args.index("-c:a") + 1] == "copy"
        assert "soft_subtitle_codec" not in info

    def test_burn_without_libass_degrades_to_soft_with_a_warning(self, tmp_path, caplog):
        info = self._info(tmp_path)
        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "burn"})
        with (
            caplog.at_level("WARNING"),
            patch("autovideofixer.core.ffmpeg_utils.has_filter", return_value=False),
            patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=_Run()) as rf,
        ):
            res = stage.execute("/t.mkv", str(tmp_path / "temp.mkv"), input_info=info)
        args = rf.call_args.args[0]
        assert res.metadata["method"] == "soft" and "-vf" not in args
        assert "lacks" in caplog.text and "falling back to mode=soft" in caplog.text
        assert info["soft_subtitle_codec"] == "mov_text"

    def test_burn_without_libass_into_webm_ends_as_a_sidecar(self, tmp_path):
        info = self._info(tmp_path, "o.webm")
        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "burn"})
        with patch("autovideofixer.core.ffmpeg_utils.has_filter", return_value=False):
            res = stage.execute("/t.mkv", str(tmp_path / "temp.mkv"), input_info=info)
        assert res.metadata["method"] == "sidecar" and (tmp_path / "o.srt").exists()

    def test_ffmpeg_failure_fails_the_stage_and_cleans_up(self, tmp_path):
        class Bad(_Run):
            returncode = 1
            stderr = "boom"

        stage = self._stage(tmp_path, **{"stages.scene_subtitles.mode": "soft"})
        with patch("autovideofixer.core.ffmpeg_utils.run_ffmpeg", return_value=Bad()):
            res = stage.execute("/t.mkv", str(tmp_path / "t.mkv"), input_info=self._info(tmp_path))
        assert res.status == StageStatus.FAILED and "boom" in res.error
        assert not [p for p in os.listdir(tmp_path) if p.startswith("avf_subs_")]

    def test_needs_scene_scope_when_it_runs_the_analysis_itself(self, tmp_path):
        seen = {}

        def fake(path, config, **kw):
            seen.update(kw)
            return type("R", (), {"to_dict": lambda self: _analysis()})()

        info = {"final_output_path": str(tmp_path / "o.mp4")}
        with patch("autovideofixer.core.analysis_service.analyze_video", fake):
            self._stage(tmp_path).execute("/t.mkv", "/t2.mkv", input_info=info)
        assert seen["scope"] == "scenes"


# ─── stage order + migration ───────────────────────────────────────


class TestOrderAndConfig:
    def test_default_order_has_the_new_stages_in_place(self):
        order = Config.DEFAULTS["pipeline"]["default_order"]
        assert order == DEFAULT_STAGE_ORDER
        assert order.index("analyze") == order.index("crop") + 1
        assert order.index("downscale") == order.index("analyze") + 1
        assert order[-3:] == ["scene_subtitles", "analysis_sidecar", "encode"]
        assert order.index("hdr") < order.index("scene_subtitles")
        assert Config.DEFAULTS["pipeline"]["max_stages"] >= len(order) + 2

    def test_new_stages_are_off_by_default_and_not_planned(self, tmp_path):
        cfg = _cfg(tmp_path)
        for name in ("analyze", "analysis_sidecar", "scene_subtitles"):
            assert cfg.get("stages", name, "enabled") is False
        job = Job(input_path="x.mp4")
        with patch("autovideofixer.core.pipeline.get_video_info", return_value={"framerate": 24}):
            stages = Pipeline(cfg).auto_determine_stages(job)
        assert not {"analyze", "analysis_sidecar", "scene_subtitles"} & set(stages)

    def test_enabled_stages_are_planned_and_ordered_by_default_order(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            **{
                f"stages.{n}.enabled": True
                for n in ("analyze", "analysis_sidecar", "scene_subtitles")
            },
        )
        pipe = Pipeline(cfg)
        with patch("autovideofixer.core.pipeline.get_video_info", return_value={"framerate": 24}):
            stages = pipe.auto_determine_stages(Job(input_path="x.mp4"))
        order = pipe.optimize_stage_order(stages)
        assert (
            order.index("crop") < order.index("analyze") < order.index("downscale")
            if "downscale" in order
            else True
        )
        assert order[-3:] == ["scene_subtitles", "analysis_sidecar", "encode"]


V2_PINNED = """\
# my header
config_version: 2
pipeline:
  # the order
  default_order:
    - detect
    - retime
    - normalize_codecs
    - deinterlace
    - crop      # keep
    - downscale
    - deblock
    - stabilize
    - denoise_video
    - upscale
    - interpolate
    - normalize_volume
    - normalize_audio
    - speed
    - hdr
    - encode
  max_stages: 18  # cap
general:
  overwrite: false
"""


class TestMigrationV2ToV3:
    def test_pinned_default_order_gets_the_stages_at_their_positions(self):
        doc = yaml.safe_load(V2_PINNED)
        notes = cm.migrate(doc)
        assert doc["config_version"] == 3
        assert doc["pipeline"]["default_order"] == Config.DEFAULTS["pipeline"]["default_order"]
        assert doc["pipeline"]["max_stages"] == 21
        assert any("'analyze' after 'crop'" in n for n in notes) and any(
            "max_stages" in n for n in notes
        )

    def test_custom_order_is_never_reordered(self):
        order = ["detect", "encode", "crop", "speed", "encode"]
        doc = {"config_version": 2, "pipeline": {"default_order": list(order), "max_stages": 3}}
        cm.migrate(doc)
        got = doc["pipeline"]["default_order"]
        assert [
            n for n in got if n not in ("analyze", "scene_subtitles", "analysis_sidecar")
        ] == order
        assert got.index("analyze") == got.index("crop") + 1  # only inserted after the anchor
        assert got[-3:] == [
            "scene_subtitles",
            "analysis_sidecar",
            "encode",
        ]  # before the LAST encode
        assert doc["pipeline"]["max_stages"] == len(got)  # a too-small custom cap is raised

    def test_custom_max_stages_big_enough_is_untouched(self):
        doc = {
            "config_version": 2,
            "pipeline": {"default_order": ["crop", "encode"], "max_stages": 40},
        }
        cm.migrate(doc)
        assert doc["pipeline"]["max_stages"] == 40

    def test_missing_anchor_means_no_insert_and_a_note(self):
        doc = {"config_version": 2, "pipeline": {"default_order": ["detect"]}}
        notes = cm.migrate(doc)
        assert doc["pipeline"]["default_order"] == ["detect"]
        assert any("no 'crop'" in n for n in notes) and any("no 'encode'" in n for n in notes)

    def test_mapping_form_entries_count_as_anchors_and_stay(self):
        doc = {"config_version": 2, "pipeline": {"default_order": [
            {"stage": "crop", "enabled": True}, {"stage": "encode"}]}}  # fmt: skip
        cm.migrate(doc)
        assert doc["pipeline"]["default_order"] == [
            {"stage": "crop", "enabled": True}, "analyze", "scene_subtitles",
            "analysis_sidecar", {"stage": "encode"},
        ]  # fmt: skip

    def test_stage_already_present_is_left_alone_and_absent_order_stays_absent(self):
        doc = {"config_version": 2, "pipeline": {"default_order": ["crop", "analyze", "encode"]}}
        cm.migrate(doc)
        assert doc["pipeline"]["default_order"].count("analyze") == 1
        doc = {"config_version": 2, "general": {"overwrite": True}}
        cm.migrate(doc)
        assert "pipeline" not in doc

    def test_idempotent(self):
        doc = yaml.safe_load(V2_PINNED)
        cm.migrate(doc)
        snap = yaml.safe_dump(doc)
        assert cm.migrate(doc) == [] and yaml.safe_dump(doc) == snap

    def test_text_rewrite_keeps_comments(self):
        new, old, notes = cm.migrate_text(V2_PINNED)
        assert old == 2 and notes
        assert "# my header" in new and "# the order" in new and "# keep" in new and "# cap" in new
        data = yaml.safe_load(new)
        assert data["config_version"] == 3 and data["pipeline"]["max_stages"] == 21
        assert data["pipeline"]["default_order"] == Config.DEFAULTS["pipeline"]["default_order"]
        again, old2, notes2 = cm.migrate_text(new)
        assert again == new and old2 == 3 and notes2 == []

    def test_default_path_load_rewrites_in_place_with_a_backup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        from autovideofixer.config import get_config_path

        path = get_config_path()
        path.parent.mkdir(parents=True)
        path.write_text(V2_PINNED)
        cfg = Config()
        assert cfg.get("pipeline", "default_order") == Config.DEFAULTS["pipeline"]["default_order"]
        assert yaml.safe_load(path.read_text())["config_version"] == 3
        assert "# keep" in path.read_text()
        assert list(path.parent.glob("config.yaml.bak-v2-*"))
        assert cfg.get("config_version") == 3
