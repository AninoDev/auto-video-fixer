"""§ 28: analysis profiles, ``analysis.vlm.scope``, the ``analyze_video`` facade and its cache.

Scenes, probe and the VLM calls are faked (``_detect_scene_changes``, ``ffmpeg_utils.probe``,
``VideoAnalyzer.run_vlm_analysis[_for_scene]``); the real-ffmpeg / real-socket runs are the E2E
notes in REQUIREMENTS.md § 28.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from autovideofixer.cli import cli as cli_mod
from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.analysis_profiles import (
    UnknownProfileError,
    apply_profile,
    available_profiles,
    resolve_profile,
)
from autovideofixer.core.analysis_service import (
    AnalysisCache,
    analyze_video,
    map_time,
    obtain_analysis,
)

SCENES = [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]


def _fake_probe(path, *a, **k):
    return SimpleNamespace(
        filename=os.path.basename(path),
        duration=20.0,
        resolution=(320, 240),
        framerate=24.0,
        has_video=True,
        has_audio=False,
        is_hdr=False,
        video_stream=SimpleNamespace(codec_name="h264"),
    )


def _scene_list(*a, **k):
    return [an.SceneEvent(s, e, "scene_change", 0.3) for s, e in SCENES]


def _vlm_for_scene(self, filepath, scene, **kw):
    return {
        "summary": f"scene {scene.start_time:g}",
        "tags": ["t"],
        "objects": [],
        "rating": "G",
        "media_type": "real_footage",
        "rating_reasoning": None,
        "extra": {},
        "call": {"purpose": "scene", "detail": f"scene[{scene.start_time}]", "success": True},
        "sampling": {"frames": 2},
    }


def _vlm_video(self, filepath, **kw):
    return {
        "summary": "whole video",
        "tags": ["v"],
        "objects": [],
        "rating": "PG",
        "media_type": "mixed",
        "rating_reasoning": None,
        "extra": {},
        "call": {"purpose": "video", "success": True},
        "sampling": {"frames": 4},
    }


class Fakes:
    """Patches the three seams and counts calls."""

    def __init__(self):
        self.video_calls = 0
        self.scene_calls = 0
        self.configs: list[Config] = []

    def __enter__(self):
        outer = self

        def video(self, *a, **k):
            outer.video_calls += 1
            outer.configs.append(self.config)
            return _vlm_video(self, *a, **k)

        def scene(self, *a, **k):
            outer.scene_calls += 1
            return _vlm_for_scene(self, *a, **k)

        self._p = [
            patch("autovideofixer.core.ffmpeg_utils.probe", _fake_probe),
            patch("autovideofixer.core.analysis._detect_scene_changes", _scene_list),
            patch.object(an.VideoAnalyzer, "run_vlm_analysis", video),
            patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", scene),
        ]
        for p in self._p:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._p:
            p.stop()


def _cfg(tmp_path, vlm=True, **extra) -> Config:
    cfg = Config(tmp_path / "none.yaml")
    if vlm:
        cfg.set(True, "analysis", "vlm", "enabled")
    for key, value in extra.items():
        cfg.set(value, *key.split("."))
    return cfg


def _video(tmp_path, name="clip.mp4"):
    p = tmp_path / name
    p.write_bytes(b"x")
    return str(p)


# ─── profiles ──────────────────────────────────────────────────────


class TestProfiles:
    def test_builtin_names_and_user_names(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.profiles": {"mine": {"vlm": {"detail": "standard"}}}})
        assert available_profiles(cfg) == ["quick", "standard", "thorough", "mine"]

    def test_standard_is_a_no_op(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert resolve_profile(cfg, "standard") == {}
        eff = apply_profile(cfg, "standard")
        assert eff.get("analysis") == cfg.get("analysis")

    def test_quick(self, tmp_path):
        eff = apply_profile(_cfg(tmp_path), "quick")
        assert eff.get("analysis", "vlm", "detail") == "brief"
        assert eff.get("analysis", "vlm", "sampling", "min_frames") == 2
        assert eff.get("analysis", "vlm", "sampling", "max_frames") == 4
        # untouched siblings keep their values (deep merge, not replacement)
        assert eff.get("analysis", "vlm", "sampling", "dedupe_consecutive") is True

    def test_thorough_without_cross_check_endpoints(self, tmp_path):
        eff = apply_profile(_cfg(tmp_path), "thorough")
        assert eff.get("analysis", "vlm", "detail") == "detailed"
        assert eff.get("analysis", "vlm", "scope") == "both"
        assert eff.get("analysis", "vlm", "outputs") == {
            "media_type": True,
            "rating_reasoning": True,
        }
        assert eff.get("analysis", "cross_check", "enabled") is False

    def _two_endpoints(self, tmp_path, names=("one", "two")):
        cfg = _cfg(tmp_path)
        for n in ("one", "two"):
            cfg.set("llamacpp", "analysis", "endpoints", n, "provider")
            cfg.set("http://127.0.0.1:1/x", "analysis", "endpoints", n, "api_url")
        cfg.set(list(names), "analysis", "cross_check", "endpoints")
        return cfg

    def test_thorough_enables_a_valid_cross_check(self, tmp_path):
        eff = apply_profile(self._two_endpoints(tmp_path), "thorough")
        assert eff.get("analysis", "cross_check", "enabled") is True

    def test_thorough_never_enables_a_cross_check_that_would_fail(self, tmp_path):
        cfg = self._two_endpoints(tmp_path, names=("one", "ghost"))
        eff = apply_profile(cfg, "thorough")
        assert eff.get("analysis", "cross_check", "enabled") is False
        # the run still validates and works
        with Fakes():
            analyze_video(_video(tmp_path), cfg, profile="thorough")

    def test_thorough_with_a_single_endpoint_leaves_it_off(self, tmp_path):
        cfg = self._two_endpoints(tmp_path, names=("one",))
        assert apply_profile(cfg, "thorough").get("analysis", "cross_check", "enabled") is False

    def test_user_profile_merges_onto_the_builtin_of_the_same_name(self, tmp_path):
        cfg = _cfg(
            tmp_path, **{"analysis.profiles": {"quick": {"vlm": {"sampling": {"max_frames": 3}}}}}
        )
        eff = apply_profile(cfg, "quick")
        assert eff.get("analysis", "vlm", "sampling", "max_frames") == 3  # user's
        assert eff.get("analysis", "vlm", "sampling", "min_frames") == 2  # built-in's

    def test_user_profile_new_name_lists_replace(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            **{"analysis.profiles": {"mine": {"vlm": {"media_type_choices": ["a", "other"]}}}},
        )
        eff = apply_profile(cfg, "mine")
        assert eff.get("analysis", "vlm", "media_type_choices") == ["a", "other"]

    def test_profile_layers_like_a_preset_and_leaves_the_original_alone(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.apply_layer({"analysis": {"vlm": {"detail": "standard"}}}, "preset:x")
        eff = apply_profile(cfg, "quick")
        # profile wins over earlier layers on the keys it sets...
        assert eff.get("analysis", "vlm", "detail") == "brief"
        # ...and keeps everything it does not mention
        assert eff.get("analysis", "vlm", "enabled") is True
        assert eff.sources[-1] == "analysis-profile:quick"
        assert cfg.get("analysis", "vlm", "detail") == "standard"
        assert cfg.sources[-1] == "preset:x"
        assert eff.get("analysis", "vlm", "sampling", "max_frames") == 4
        assert cfg.get("analysis", "vlm", "sampling", "max_frames") == 8

    def test_unknown_profile_lists_the_available_names(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.profiles": {"mine": {}}})
        with pytest.raises(UnknownProfileError) as ei:
            apply_profile(cfg, "nope")
        assert "quick, standard, thorough, mine" in str(ei.value)

    def test_unknown_key_in_a_profile_is_an_error(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.profiles": {"bad": {"vlmm": {}}}})
        with pytest.raises(ValueError, match="analysis.profiles.bad: unknown key 'vlmm'"):
            apply_profile(cfg, "bad")

    def test_profile_contents_go_through_the_normal_validation(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.profiles": {"bad": {"vlm": {"detail": "wordy"}}}})
        with pytest.raises(ValueError, match="analysis.vlm.detail"):
            analyze_video(_video(tmp_path), cfg, profile="bad")
        cfg = _cfg(tmp_path, **{"analysis.profiles": {"bad": {"vlm": {"scope": "frames"}}}})
        with pytest.raises(ValueError, match="analysis.vlm.scope"):
            analyze_video(_video(tmp_path), cfg, profile="bad")

    def test_profile_reaches_the_analyzer_and_overrides_win(self, tmp_path):
        with Fakes() as f:
            analyze_video(_video(tmp_path), _cfg(tmp_path), profile="quick")
        assert f.configs[0].get("analysis", "vlm", "sampling", "max_frames") == 4

    def test_the_config_hash_depends_on_the_profile(self, tmp_path):
        with Fakes():
            a = analyze_video(_video(tmp_path), _cfg(tmp_path), profile="quick")
            b = analyze_video(_video(tmp_path), _cfg(tmp_path), profile="thorough")
            c = analyze_video(_video(tmp_path), _cfg(tmp_path), profile="quick")
        assert a.config_hash != b.config_hash and a.config_hash == c.config_hash


# ─── scope ─────────────────────────────────────────────────────────


class TestScope:
    def test_default_scope_is_video_and_never_describes_scenes(self, tmp_path):
        with Fakes() as f:
            res = analyze_video(_video(tmp_path), _cfg(tmp_path))
        assert (f.video_calls, f.scene_calls) == (1, 0)
        assert res.scope == "video" and res.analysis.scene_vlm == []
        assert Config.DEFAULTS["analysis"]["vlm"]["scope"] == "video"

    def test_video_scope_json_and_csv_are_what_earlier_releases_wrote(self, tmp_path):
        with Fakes():
            analysis = an.VideoAnalyzer(_cfg(tmp_path)).analyze(_video(tmp_path))
        doc = cli_mod._analysis_to_json(analysis)
        assert set(doc) == {
            "filepath", "filename", "duration_sec", "resolution", "framerate_fps",
            "video_codec", "has_video", "has_audio", "hdr", "scenes_detected", "scenes",
            "vlm", "vlm_sampling", "vlm_calls",
        }  # fmt: skip
        assert all(
            set(s) == {"start_time", "end_time", "event_type", "confidence", "description"}
            for s in doc["scenes"]
        )
        row = cli_mod._analysis_to_csv_row(analysis)
        assert row["analysis_scope"] == "video" and row["scene_vlm"] == ""
        assert row["vlm_calls"] == 1

    def test_scenes_scope_describes_each_scene_only(self, tmp_path):
        with Fakes() as f:
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="scenes")
        assert (f.video_calls, f.scene_calls) == (0, 3)
        va = res.analysis
        assert [r["summary"] for r in va.scene_vlm] == ["scene 0", "scene 5", "scene 12"]
        assert [s.description for s in va.scenes] == ["scene 0", "scene 5", "scene 12"]
        assert va.vlm_summary is None
        assert [c["purpose"] for c in va.vlm_calls] == ["scene"] * 3

    def test_both_scope_is_video_plus_scenes_and_calls_are_in_order(self, tmp_path):
        with Fakes() as f:
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="both")
        assert (f.video_calls, f.scene_calls) == (1, 3)
        assert res.analysis.vlm_summary == "whole video"
        assert [c["purpose"] for c in res.analysis.vlm_calls] == ["video"] + ["scene"] * 3

    def test_scope_from_config_and_argument_wins(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.vlm.scope": "scenes"})
        with Fakes() as f:
            analyze_video(_video(tmp_path), cfg)
            assert (f.video_calls, f.scene_calls) == (0, 3)
            analyze_video(_video(tmp_path), cfg, scope="video")
        assert (f.video_calls, f.scene_calls) == (1, 3)

    def test_scene_scope_without_a_vlm_warns_and_calls_nothing(self, tmp_path, caplog):
        with Fakes() as f, caplog.at_level("WARNING"):
            analyze_video(_video(tmp_path), _cfg(tmp_path, vlm=False), scope="scenes")
        assert (f.video_calls, f.scene_calls) == (0, 0)
        assert "VLM is disabled" in caplog.text

    def test_scene_scope_with_events_disabled_warns(self, tmp_path, caplog):
        with Fakes() as f, caplog.at_level("WARNING"):
            res = analyze_video(
                _video(tmp_path),
                _cfg(tmp_path),
                scope="scenes",
                overrides={"include_events": False},
            )
        assert f.scene_calls == 0 and res.analysis.scenes == []
        assert "needs scene detection" in caplog.text

    def test_scene_scope_runs_detection_even_when_the_config_disables_events(self, tmp_path):
        cfg = _cfg(tmp_path, **{"analysis.event_detection.enabled": False})
        with Fakes() as f:
            analyze_video(_video(tmp_path), cfg, scope="scenes")
        assert f.scene_calls == 3

    def test_cross_check_applies_per_scene_and_legs_reach_the_call_list(self, tmp_path):
        block = {
            "mode": "refine",
            "legs": [{"call": {"purpose": "scene"}}, {"call": {"purpose": "cross_check:refine"}}],
        }

        def scene(self, filepath, sc, **kw):
            r = _vlm_for_scene(self, filepath, sc)
            r["cross_check"] = block
            return r

        with Fakes(), patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", scene):
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="scenes")
        assert len(res.analysis.vlm_calls) == 6
        assert res.analysis.scene_vlm[0]["cross_check"] == block

    def test_an_unreachable_server_stops_the_scene_loop_after_one_warning(self, tmp_path, caplog):
        def down(self, filepath, sc, **kw):
            return {"summary": "", "call": {"success": False, "error_class": "connection"}}

        with (
            Fakes() as f,
            patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", down),
            caplog.at_level("WARNING"),
        ):
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="scenes")
        assert len(res.analysis.scene_vlm) == 1 and f.scene_calls == 0
        assert caplog.text.count("skipping the remaining 2 scene(s)") == 1

    def test_a_failed_whole_video_call_skips_the_scene_calls(self, tmp_path, caplog):
        def down(self, filepath, **kw):
            return {"summary": "", "call": {"success": False, "error_class": "timeout"}}

        with (
            Fakes() as f,
            patch.object(an.VideoAnalyzer, "run_vlm_analysis", down),
            caplog.at_level("WARNING"),
        ):
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="both")
        assert f.scene_calls == 0 and res.analysis.scene_vlm == []
        assert "skipping the 3 per-scene call(s)" in caplog.text

    def test_other_failures_do_not_stop_the_loop(self, tmp_path):
        def bad(self, filepath, sc, **kw):
            return {"summary": "", "call": {"success": False, "error_class": "bad_json"}}

        with Fakes(), patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", bad):
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="scenes")
        assert len(res.analysis.scene_vlm) == 3

    def test_invalid_scope_value(self, tmp_path):
        with pytest.raises(ValueError, match="analysis.vlm.scope"):
            analyze_video(_video(tmp_path), _cfg(tmp_path), scope="frames")

    def test_scene_calls_share_the_run_overrides(self, tmp_path):
        seen = {}

        def scene(self, filepath, sc, **kw):
            seen.update(kw)
            return _vlm_for_scene(self, filepath, sc)

        with Fakes(), patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", scene):
            analyze_video(
                _video(tmp_path),
                _cfg(tmp_path),
                scope="scenes",
                overrides={"max_sample_frames": 3, "prompt_append": "hi"},
            )
        assert seen["max_sample_frames"] == 3 and seen["prompt_append"] == "hi"


# ─── facade + cache ────────────────────────────────────────────────


class TestFacadeAndCache:
    def test_result_to_dict_is_json_serialisable(self, tmp_path):
        with Fakes():
            res = analyze_video(_video(tmp_path), _cfg(tmp_path), scope="both", profile="quick")
        doc = json.loads(json.dumps(res.to_dict()))
        assert doc["scope"] == "both" and doc["profile"] == "quick" and doc["config_hash"]
        assert doc["scenes_detected"] == 3 and doc["scenes"][1]["vlm"]["summary"] == "scene 5"
        assert doc["vlm"]["summary"] == "whole video"

    def test_unknown_override_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown analysis override"):
            analyze_video(_video(tmp_path), _cfg(tmp_path), overrides={"bogus": 1})

    def test_cache_hit_on_same_key(self, tmp_path):
        cache, cfg, vid = AnalysisCache(), _cfg(tmp_path), _video(tmp_path)
        with Fakes() as f:
            a = analyze_video(vid, cfg, cache=cache)
            b = analyze_video(vid, cfg, cache=cache)
        assert f.video_calls == 1 and not a.cached and b.cached and len(cache) == 1
        assert b.analysis is a.analysis

    def test_cache_miss_on_config_scope_profile_and_mtime(self, tmp_path):
        cache, cfg, vid = AnalysisCache(), _cfg(tmp_path), _video(tmp_path)
        with Fakes() as f:
            analyze_video(vid, cfg, cache=cache)
            analyze_video(vid, cfg, cache=cache, scope="scenes")
            analyze_video(vid, cfg, cache=cache, profile="quick")
            cfg.set("standard", "analysis", "vlm", "detail")
            analyze_video(vid, cfg, cache=cache)
            analyze_video(vid, cfg, cache=cache, overrides={"max_sample_frames": 2})
            n = f.video_calls
            analyze_video(vid, cfg, cache=cache)  # hit again
            assert f.video_calls == n
            st = os.stat(vid)
            os.utime(vid, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
            analyze_video(vid, cfg, cache=cache)  # mtime changed
            assert f.video_calls == n + 1
            with open(vid, "ab") as fh:
                fh.write(b"more")
            analyze_video(vid, cfg, cache=cache)  # size changed
            assert f.video_calls == n + 2

    def test_cache_keys_on_the_realpath(self, tmp_path):
        cache, cfg, vid = AnalysisCache(), _cfg(tmp_path), _video(tmp_path)
        link = tmp_path / "link.mp4"
        link.symlink_to(vid)
        with Fakes() as f:
            analyze_video(vid, cfg, cache=cache)
            analyze_video(str(link), cfg, cache=cache)
        assert f.video_calls == 1

    def test_no_cache_means_no_sharing(self, tmp_path):
        with Fakes() as f:
            analyze_video(_video(tmp_path), _cfg(tmp_path))
            analyze_video(_video(tmp_path), _cfg(tmp_path))
        assert f.video_calls == 2

    def test_failures_are_not_cached(self, tmp_path):
        cache, cfg, vid = AnalysisCache(), _cfg(tmp_path), _video(tmp_path)
        boom = {"n": 0}

        def flaky(*a, **k):
            boom["n"] += 1
            if boom["n"] == 1:
                raise RuntimeError("probe failed")
            return _fake_probe(a[0])

        with Fakes(), patch("autovideofixer.core.ffmpeg_utils.probe", flaky):
            with pytest.raises(RuntimeError):
                analyze_video(vid, cfg, cache=cache)
            assert len(cache) == 0
            analyze_video(vid, cfg, cache=cache)
        assert len(cache) == 1

    def test_concurrent_callers_compute_once(self, tmp_path):
        cache, cfg, vid = AnalysisCache(), _cfg(tmp_path), _video(tmp_path)
        results, started = [], threading.Barrier(6)

        def slow_video(self, *a, **k):
            time.sleep(0.2)
            return _vlm_video(self, *a, **k)

        def worker():
            started.wait()
            results.append(analyze_video(vid, cfg, cache=cache))

        with Fakes() as f, patch.object(an.VideoAnalyzer, "run_vlm_analysis", slow_video):
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert len(results) == 6 and len(cache) == 1
        assert sum(1 for r in results if not r.cached) == 1
        assert len({id(r.analysis) for r in results}) == 1
        assert f.scene_calls == 0

    def test_map_time(self):
        assert map_time(10.0, None) == 10.0
        assert map_time(10.0, {"analysis_time_scale": 0.5}) == 5.0


class TestObtainAnalysis:
    def test_published_analysis_wins_and_carries_the_scale(self, tmp_path):
        info = {"analysis": {"scenes": []}, "analysis_time_scale": 0.5}
        got = obtain_analysis(_cfg(tmp_path), "x", {}, "p", info, None)
        assert got == ({"scenes": []}, 0.5)

    def test_missing_and_not_allowed_skips_with_one_warning(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            got = obtain_analysis(_cfg(tmp_path), "x", {"analyze_if_missing": False}, "p", {}, None)
        assert got is None and caplog.text.count("no analysis available") == 1

    def test_missing_runs_the_module_through_the_cache(self, tmp_path):
        cache, vid = AnalysisCache(), _video(tmp_path)
        with Fakes() as f:
            a = obtain_analysis(_cfg(tmp_path), "x", {}, vid, {}, cache, need_scenes=True)
            b = obtain_analysis(_cfg(tmp_path), "y", {}, vid, {}, cache, need_scenes=True)
        assert a is not None and b is not None and a[1] == 1.0
        assert f.scene_calls == 3  # video scope was upgraded to scenes; second call hit the cache
        assert a[0]["scope"] == "scenes"

    def test_video_scope_publication_does_not_starve_a_scene_consumer(self, tmp_path):
        # `analyze` ran with the default scope (video): a scene consumer analyses per scene
        # itself (the current file, so scale 1.0) instead of skipping.
        info = {"analysis": {"scope": "video", "scenes": []}, "analysis_time_scale": 0.5}
        with Fakes() as f:
            got = obtain_analysis(
                _cfg(tmp_path), "x", {}, _video(tmp_path), info, None, need_scenes=True
            )
        assert got is not None and got[0]["scope"] == "scenes" and got[1] == 1.0
        assert f.scene_calls == 3

    def test_scene_scope_publication_is_used_by_a_scene_consumer(self, tmp_path):
        info = {"analysis": {"scope": "both", "scenes": []}, "analysis_time_scale": 0.5}
        with Fakes() as f:
            got = obtain_analysis(_cfg(tmp_path), "x", {}, "p", info, None, need_scenes=True)
        assert got == (info["analysis"], 0.5) and f.scene_calls == 0

    def test_stage_scope_and_profile_fall_back_to_stages_analyze(self, tmp_path):
        cfg = _cfg(tmp_path, **{"stages.analyze.profile": "quick", "stages.analyze.scope": "both"})
        with Fakes():
            got = obtain_analysis(cfg, "x", {}, _video(tmp_path), {}, None)
        assert got is not None and got[0]["profile"] == "quick" and got[0]["scope"] == "both"

    def test_an_analysis_failure_is_one_warning_and_none(self, tmp_path, caplog):
        with (
            patch("autovideofixer.core.ffmpeg_utils.probe", side_effect=RuntimeError("bad file")),
            caplog.at_level("WARNING"),
        ):
            got = obtain_analysis(_cfg(tmp_path), "x", {}, _video(tmp_path), {}, None)
        assert got is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and warnings[0].exc_info is None


# ─── CLI ───────────────────────────────────────────────────────────


class TestAnalyzeCli:
    def _run(self, tmp_path, *args, cfg_text="config_version: 3\n"):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text(cfg_text)
        vid = _video(tmp_path)
        out = {"json": tmp_path / "o.json", "csv": tmp_path / "o.csv"}
        with (
            Fakes() as f,
            patch("autovideofixer.cli.cli.is_video_file", return_value=True),
        ):
            r = CliRunner().invoke(
                cli_mod.main,
                ["--config", str(cfgp), "analyze", vid, "--json", str(out["json"]),
                 "--csv", str(out["csv"]), *args],
            )  # fmt: skip
        return r, f, out

    def test_unknown_profile_exits_2_and_lists_names(self, tmp_path):
        r, f, _ = self._run(tmp_path, "--vlm", "--profile", "nope")
        assert r.exit_code == 2
        assert "Unknown analysis profile 'nope'" in r.output and "thorough" in r.output
        assert f.video_calls == 0

    def test_profile_and_flag_precedence(self, tmp_path):
        r, f, _ = self._run(tmp_path, "--vlm", "--profile", "quick", "--no-events")
        assert r.exit_code == 0, r.output
        assert f.configs[0].get("analysis", "vlm", "sampling", "max_frames") == 4

        captured = {}

        def analyze(self, filepath, **kw):
            captured.update(kw)
            captured["cfg_max"] = self.config.get("analysis", "vlm", "sampling", "max_frames")
            return an.VideoAnalysis(filepath=filepath, filename="clip.mp4")

        cfgp = tmp_path / "c2.yaml"
        cfgp.write_text("config_version: 3\n")
        with (
            patch.object(an.VideoAnalyzer, "analyze", analyze),
            patch("autovideofixer.cli.cli.is_video_file", return_value=True),
        ):
            r = CliRunner().invoke(
                cli_mod.main,
                ["--config", str(cfgp), "analyze", _video(tmp_path), "--profile", "quick",
                 "--max-sample-frames", "7", "--scope", "scenes"],
            )  # fmt: skip
        assert r.exit_code == 0, r.output
        # the profile set the config value; the explicit flag is passed through and wins
        assert captured["cfg_max"] == 4 and captured["max_sample_frames"] == 7
        assert captured["scope"] == "scenes"

    def test_scenes_scope_output_everywhere(self, tmp_path):
        r, f, out = self._run(tmp_path, "--vlm", "--scope", "both", "--full")
        assert r.exit_code == 0, r.output
        assert (f.video_calls, f.scene_calls) == (1, 3)
        assert "Scene 2 VLM" in r.output and "scene 5" in r.output  # --full panel
        assert "VLM Scope" in r.output
        assert "VLM call totals: 4 call(s)" in r.output
        doc = json.loads(out["json"].read_text())["files"][0]
        assert doc["scope"] == "both" and doc["scenes"][2]["vlm"]["summary"] == "scene 12"
        assert [c["purpose"] for c in doc["vlm_calls"]] == ["video", "scene", "scene", "scene"]
        row = next(csv.DictReader(out["csv"].open()))
        assert row["analysis_scope"] == "both" and row["vlm_calls"] == "4"
        assert json.loads(row["scene_vlm"])[0]["summary"] == "scene 0"

    def test_config_scope_and_profile_column(self, tmp_path):
        r, f, out = self._run(
            tmp_path, "--vlm", "--profile", "standard",
            cfg_text="config_version: 3\nanalysis:\n  vlm:\n    scope: scenes\n",
        )  # fmt: skip
        assert r.exit_code == 0 and (f.video_calls, f.scene_calls) == (0, 3)
        row = next(csv.DictReader(out["csv"].open()))
        assert row["analysis_profile"] == "standard" and row["analysis_scope"] == "scenes"

    def test_bad_scope_in_config_exits_2(self, tmp_path):
        r, _, _ = self._run(
            tmp_path, "--vlm", cfg_text="config_version: 3\nanalysis:\n  vlm:\n    scope: nope\n"
        )
        assert r.exit_code == 2 and "analysis.vlm.scope" in r.output
