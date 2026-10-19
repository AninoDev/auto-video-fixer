"""§ 27 multi-VLM cross-check: config validation, refine and judge flows, every fail-open
permutation, per-leg purposes through the call collector (also across threads), the shared
decode cache and the analyze console/CSV/JSON output.

Provider calls are scripted at ``_call_provider_raw`` (the single choke point that records the
call stats), frame sampling at ``sample_vlm_frames_multi``; the shared decode cache is tested
against the real sampler with a synthetic ``FFmpegFrameSource._run``. Real sockets, real ffmpeg
and the CLI end to end are covered by the E2E notes in REQUIREMENTS.md § 27.5.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from autovideofixer.cli import cli as cli_mod
from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core import vlm_sampling as vs
from autovideofixer.core.cross_check import (
    CrossCheckConfig,
    JudgeVerdict,
    build_judge_schema,
    parse_judge_verdict,
)
from autovideofixer.core.vlm_calls import (
    CallStats,
    ProviderResponse,
    collect_call_stats,
    record_call,
)
from autovideofixer.core.vlm_outputs import (
    OutputsConfig,
    build_refine_schema,
    build_refine_system_prompt,
    build_response_schema,
)

# ─── scaffolding ───────────────────────────────────────────────────

ENDPOINTS = ("vlm_a", "vlm_b", "vlm_c")


def _cfg(tmp_path, **cross_check) -> Config:
    cfg = Config(tmp_path / "none.yaml")
    for name in (*ENDPOINTS, "judge_llm", "arb"):
        cfg.set("llamacpp", "analysis", "endpoints", name, "provider")
        cfg.set(f"model-{name}", "analysis", "endpoints", name, "model")
        cfg.set("http://127.0.0.1:1/x", "analysis", "endpoints", name, "api_url")
    base = {"enabled": True, "mode": "refine", "endpoints": list(ENDPOINTS[:2])}
    for key, value in {**base, **cross_check}.items():
        cfg.set(value, "analysis", "cross_check", key)
    return cfg


def _json(summary="a cat", tags=("cat",), objects=("cat",), rating="G", media="real_footage", **kw):
    return json.dumps(
        {
            "summary": summary,
            "tags": list(tags),
            "objects": list(objects),
            "media_type": media,
            "rating": rating,
            **kw,
        }
    )


def _verdict(agreement, merged=None, conflicts=()):
    return json.dumps(
        {
            "agreement": agreement,
            "conflicts": list(conflicts),
            "merged": merged if merged is not None else json.loads(_json("merged")),
        }
    )


class Fake:
    """Scripted ``_call_provider_raw``: ``script[(endpoint, purpose)]`` is a reply text, an
    ``Exception`` (raised), or ``"FAIL"`` (a failed call). Every call is recorded like the real
    one (``record_call``) and kept in ``calls`` with the request it carried."""

    def __init__(self, script):
        self.script = dict(script)
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, provider, **kw):
        key = (kw["endpoint_name"], kw["purpose"])
        reply = self.script.get(key, self.script.get(kw["endpoint_name"], "FAIL"))
        stats = CallStats(
            provider=provider,
            model=kw["model"],
            images=len(kw["image_b64_list"]),
            input_tokens=10,
            output_tokens=5,
        )
        stats.endpoint, stats.purpose, stats.detail = (
            kw["endpoint_name"],
            kw["purpose"],
            kw["detail"],
        )
        with self.lock:
            self.calls.append({"key": key, "kw": kw, "thread": threading.get_ident()})
        if isinstance(reply, Exception):
            raise reply
        if reply == "FAIL":
            stats.success, stats.error_class, stats.error = False, "http_500", "HTTP 500"
            record_call(stats)
            return ProviderResponse("", stats)
        record_call(stats)
        return ProviderResponse(reply, stats)

    def for_key(self, endpoint, purpose):
        return [c for c in self.calls if c["key"] == (endpoint, purpose)]

    @property
    def keys(self):
        return [c["key"] for c in self.calls]


def _fake_probe(path):
    from types import SimpleNamespace

    return SimpleNamespace(
        filename="clip.mp4", duration=4.0, resolution=(64, 36), framerate=25.0,
        has_video=True, has_audio=False, is_hdr=False, video_stream=None,
    )  # fmt: skip


def _frames_for(tmp_path, visions, n=3):
    out = []
    for i, v in enumerate(visions):
        d = tmp_path / f"s{i}"
        d.mkdir(exist_ok=True)
        paths = []
        for j in range(n + i):  # distinct frame counts so a leg's images identify it
            p = d / f"f{j}.jpg"
            p.write_bytes(b"\xff\xd8\xff")
            paths.append(str(p))
        rep = vs.SamplingReport(frames=len(paths), width=100 * (i + 1), height=50 * (i + 1))
        out.append(vs.SampledFrames(paths=paths, tmp_dir=str(d), report=rep))
    return out


@pytest.fixture
def run(tmp_path):
    """``run(script, mode=..., **cc)`` -> (result, fake); also exposes ``run.visions``."""

    def _run(script, *, scene=None, analyzer_kwargs=None, no_frames=(), **cc):
        cfg = _cfg(tmp_path, **cc)
        fake = Fake(script)

        def multi(filepath, sampling, visions, start=0.0, end=None, **kw):
            _run.visions = list(visions)
            _run.range = (start, end)
            frames = _frames_for(tmp_path, visions)
            for i in no_frames:
                frames[i] = vs.SampledFrames(paths=[], tmp_dir="", report=vs.SamplingReport())
            return frames

        with (
            patch("autovideofixer.core.analysis._call_provider_raw", fake),
            patch("autovideofixer.core.analysis.sample_vlm_frames_multi", multi),
        ):
            analyzer = an.VideoAnalyzer(cfg)
            if scene is not None:
                result = analyzer.run_vlm_analysis_for_scene(
                    "v.mp4", scene, **(analyzer_kwargs or {})
                )
            else:
                result = analyzer.run_vlm_analysis("v.mp4", **(analyzer_kwargs or {}))
        return result, fake

    return _run


def _legs(result):
    return result["cross_check"]["legs"]


# ─── config ────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults_are_disabled(self, tmp_path):
        cfg = CrossCheckConfig.from_config(Config(tmp_path / "n.yaml"))
        assert cfg == CrossCheckConfig()
        assert (cfg.enabled, cfg.mode, cfg.endpoints) == (False, "refine", ())
        assert (cfg.judge_endpoint, cfg.arbiter_endpoint) == (None, None)
        assert (cfg.agreement_threshold, cfg.max_parallel) == (0.75, 1)

    def test_valid_enabled(self, tmp_path):
        cfg = CrossCheckConfig.from_config(
            _cfg(
                tmp_path,
                mode="Judge",
                arbiter_endpoint="arb",
                agreement_threshold=1,
                max_parallel=3,
            )
        )
        assert cfg.enabled and cfg.mode == "judge" and cfg.arbiter_endpoint == "arb"
        assert cfg.agreement_threshold == 1.0 and cfg.max_parallel == 3
        assert cfg.endpoints == ("vlm_a", "vlm_b")

    @pytest.mark.parametrize(
        "key,value,match",
        [
            ("mode", "vote", "analysis.cross_check.mode"),
            ("mode", 3, "analysis.cross_check.mode"),
            ("enabled", "yes", "enabled"),
            ("agreement_threshold", 1.5, "agreement_threshold"),
            ("agreement_threshold", -0.1, "agreement_threshold"),
            ("agreement_threshold", True, "agreement_threshold"),
            ("agreement_threshold", "0.5", "agreement_threshold"),
            ("agreement_threshold", float("nan"), "agreement_threshold"),
            ("max_parallel", 0, "max_parallel"),
            ("max_parallel", 1.5, "max_parallel"),
            ("max_parallel", True, "max_parallel"),
            ("endpoints", "vlm_a", "endpoints"),
            ("endpoints", ["vlm_a", 3], "endpoints"),
            ("judge_endpoint", 5, "judge_endpoint"),
            ("arbiter_endpoint", "", "arbiter_endpoint"),
        ],
    )
    def test_shape_errors(self, tmp_path, key, value, match):
        with pytest.raises(ValueError, match=match):
            CrossCheckConfig.from_config(_cfg(tmp_path, **{key: value}))

    def test_unknown_key_and_non_mapping(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown key.*bogus"):
            CrossCheckConfig.from_config(_cfg(tmp_path, bogus=1))
        cfg = Config(tmp_path / "n.yaml")
        cfg.set([1], "analysis", "cross_check")
        with pytest.raises(ValueError, match="expected a mapping"):
            CrossCheckConfig.from_config(cfg)

    @pytest.mark.parametrize("endpoints", [[], ["vlm_a"]])
    def test_enabled_needs_two_endpoints(self, tmp_path, endpoints):
        with pytest.raises(ValueError, match="needs at least 2"):
            CrossCheckConfig.from_config(_cfg(tmp_path, endpoints=endpoints))

    def test_unknown_names_are_clear_errors(self, tmp_path):
        with pytest.raises(ValueError, match=r"endpoints\[1\].*Unknown analysis endpoint 'nope'"):
            CrossCheckConfig.from_config(_cfg(tmp_path, endpoints=["vlm_a", "nope"]))
        with pytest.raises(ValueError, match="arbiter_endpoint.*'ghost'"):
            CrossCheckConfig.from_config(_cfg(tmp_path, arbiter_endpoint="ghost"))
        with pytest.raises(ValueError, match="judge_endpoint.*'ghost'"):
            CrossCheckConfig.from_config(_cfg(tmp_path, mode="judge", judge_endpoint="ghost"))
        # the judge is not used by refine, so a bad name there is not an error
        CrossCheckConfig.from_config(_cfg(tmp_path, mode="refine", judge_endpoint="ghost"))

    def test_disabled_block_never_resolves_names(self, tmp_path):
        cfg = CrossCheckConfig.from_config(
            _cfg(tmp_path, enabled=False, endpoints=["ghost"], arbiter_endpoint="ghost")
        )
        assert not cfg.enabled  # but the shape was still checked
        with pytest.raises(ValueError, match="mode"):
            CrossCheckConfig.from_config(_cfg(tmp_path, enabled=False, mode="bad"))

    def test_judge_falls_back_to_llm_endpoint(self, tmp_path):
        cfg = _cfg(tmp_path, mode="judge")
        assert CrossCheckConfig.from_config(cfg).judge_name(cfg) == "default_llm"
        cfg.set("judge_llm", "analysis", "llm", "endpoint")
        assert CrossCheckConfig.from_config(cfg).judge_name(cfg) == "judge_llm"
        cfg.set("arb", "analysis", "cross_check", "judge_endpoint")
        assert CrossCheckConfig.from_config(cfg).judge_name(cfg) == "arb"

    def test_defaults_in_config_and_examples_agree(self, tmp_path):
        from pathlib import Path

        from ruamel.yaml import YAML

        expected = {
            "enabled": False,
            "mode": "refine",
            "endpoints": [],
            "judge_endpoint": None,
            "arbiter_endpoint": None,
            "agreement_threshold": 0.75,
            "max_parallel": 1,
        }
        assert Config(tmp_path / "n.yaml").get("analysis", "cross_check") == expected
        root = Path(__file__).resolve().parents[2]
        for example in (
            root / "docs" / "config.example.yaml",
            root / "src" / "autovideofixer" / "data" / "config.example.yaml",
        ):
            doc = YAML(typ="safe").load(example.read_text(encoding="utf-8"))
            assert doc["analysis"]["cross_check"] == expected


# ─── disabled == today ─────────────────────────────────────────────


class TestDisabled:
    def test_disabled_makes_the_same_call_and_adds_no_keys(self, tmp_path):
        cfg = _cfg(tmp_path, enabled=False)
        cfg.set("vlm_a", "analysis", "vlm", "endpoint")
        sampled = _frames_for(tmp_path, [vs.VisionLimits()], n=2)[0]
        fake = Fake({"vlm_a": _json("solo")})
        with (
            patch("autovideofixer.core.analysis._call_provider_raw", fake),
            patch("autovideofixer.core.analysis.sample_vlm_frames", return_value=sampled) as one,
            patch("autovideofixer.core.analysis.sample_vlm_frames_multi") as multi,
        ):
            result = an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        multi.assert_not_called()
        one.assert_called_once()
        assert fake.keys == [("vlm_a", "video")]
        assert set(result) == {
            "summary",
            "tags",
            "objects",
            "rating",
            "media_type",
            "rating_reasoning",
            "extra",
            "call",
            "sampling",
        }
        assert "cross_check" not in result and result["summary"] == "solo"
        assert result["call"]["purpose"] == "video"

    def test_absent_block_is_disabled(self, tmp_path):
        cfg = Config(tmp_path / "n.yaml")
        assert not CrossCheckConfig.from_config(cfg).enabled
        a = an.VideoAnalysis(filepath="/a", filename="a")
        assert a.cross_check is None
        assert "cross_check" not in cli_mod._analysis_to_json(a)
        assert cli_mod._analysis_to_csv_row(a)["cross_check_mode"] == ""


# ─── refine ────────────────────────────────────────────────────────

REFINE = "cross_check:refine"
DESCRIBE = "cross_check:describe"


class TestRefine:
    def test_chain_of_three(self, run):
        result, fake = run(
            {
                ("vlm_a", "video"): _json("first"),
                ("vlm_b", REFINE): _json("second", changes=["fixed colour"]),
                ("vlm_c", REFINE): _json("third", changes=[]),
            },
            endpoints=list(ENDPOINTS),
        )
        assert fake.keys == [("vlm_a", "video"), ("vlm_b", REFINE), ("vlm_c", REFINE)]
        assert result["summary"] == "third"
        cc = result["cross_check"]
        assert (cc["mode"], cc["final_source"], cc["final_leg"]) == ("refine", "vlm_c", 2)
        assert [leg["role"] for leg in cc["legs"]] == ["primary", "refine", "refine"]
        assert [leg["changes"] for leg in cc["legs"][1:]] == [["fixed colour"], []]
        assert "changes" not in result["extra"]  # bookkeeping, not a model extra
        assert result["call"]["purpose"] == "video"  # the primary's call

    def test_each_leg_gets_its_own_frames_and_the_previous_json(self, run):
        result, fake = run(
            {("vlm_a", "video"): _json("alpha seen"), ("vlm_b", REFINE): _json("beta")},
        )
        assert [v for v in run.visions] == [vs.VisionLimits(), vs.VisionLimits()]
        a, b = fake.calls
        assert len(a["kw"]["image_b64_list"]) == 3 and len(b["kw"]["image_b64_list"]) == 4
        user = b["kw"]["user_prompt"]
        assert "alpha seen" in user and '"rating": "G"' in user and "vlm_a" in user
        assert "alpha seen" not in a["kw"]["user_prompt"]
        # generated from the same field list as the describe prompt / schema
        outputs = OutputsConfig()
        assert b["kw"]["system_prompt"] == build_refine_system_prompt(outputs)
        assert b["kw"]["response_schema"] == build_refine_schema(outputs)
        assert "changes" in b["kw"]["system_prompt"]
        props = b["kw"]["response_schema"]["properties"]
        assert list(props) == [*build_response_schema(outputs)["properties"], "changes"]
        assert a["kw"]["response_schema"] == build_response_schema(outputs)

    def test_prompt_append_reaches_refine_legs(self, run):
        _, fake = run(
            {("vlm_a", "video"): _json(), ("vlm_b", REFINE): _json()},
            analyzer_kwargs={"prompt_append": "trail camera"},
        )
        assert all("trail camera" in c["kw"]["user_prompt"] for c in fake.calls)

    def test_primary_fails_second_describes_third_refines(self, run):
        result, fake = run(
            {
                ("vlm_a", "video"): "FAIL",
                ("vlm_b", DESCRIBE): _json("from b"),
                ("vlm_c", REFINE): _json("from c"),
            },
            endpoints=list(ENDPOINTS),
        )
        assert fake.keys == [("vlm_a", "video"), ("vlm_b", DESCRIBE), ("vlm_c", REFINE)]
        assert [leg["role"] for leg in _legs(result)] == ["primary", "describe", "refine"]
        assert "from b" in fake.calls[2]["kw"]["user_prompt"]
        assert result["summary"] == "from c"
        assert [leg["success"] for leg in _legs(result)] == [False, True, True]

    def test_middle_leg_fails_next_refines_the_last_good(self, run):
        result, fake = run(
            {
                ("vlm_a", "video"): _json("from a"),
                ("vlm_b", REFINE): "FAIL",
                ("vlm_c", REFINE): _json("from c"),
            },
            endpoints=list(ENDPOINTS),
        )
        assert "from a" in fake.calls[2]["kw"]["user_prompt"]
        assert result["summary"] == "from c" and result["cross_check"]["final_source"] == "vlm_c"

    def test_last_leg_fails_keeps_earlier_output(self, run):
        result, _ = run({("vlm_a", "video"): _json("from a"), ("vlm_b", REFINE): "FAIL"})
        assert result["summary"] == "from a"
        assert result["cross_check"]["final_source"] == "vlm_a"
        assert [leg["success"] for leg in _legs(result)] == [True, False]
        assert _legs(result)[1]["error"] == "http_500"

    @pytest.mark.parametrize("reply", ["I think it is a cat.", "{}", '{"summary": ""}', "[1]"])
    def test_refine_reply_that_is_not_usable_json_cannot_overwrite(self, run, reply, caplog):
        with caplog.at_level("WARNING"):
            result, _ = run({("vlm_a", "video"): _json("from a"), ("vlm_b", REFINE): reply})
        assert result["summary"] == "from a"
        leg = _legs(result)[1]
        assert not leg["success"] and leg["result"] is None
        assert any("refine leg" in r.message and "vlm_b" in r.message for r in caplog.records)
        assert all(r.exc_info is None for r in caplog.records)

    def test_provider_exception_is_a_one_line_warning(self, run, caplog):
        with caplog.at_level("DEBUG"):
            result, _ = run(
                {("vlm_a", "video"): _json("from a"), ("vlm_b", REFINE): RuntimeError("boom")}
            )
        assert result["summary"] == "from a"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("boom" in r.message for r in warnings)
        assert all(r.exc_info is None for r in warnings)
        assert any(r.levelname == "DEBUG" and r.exc_info for r in caplog.records)

    def test_all_fail_is_todays_empty_result_plus_the_block(self, run):
        result, _ = run({})
        assert result["summary"] == "" and result["tags"] == [] and result["rating"] is None
        cc = result["cross_check"]
        assert cc["final_source"] is None and cc["final_leg"] is None
        assert [leg["success"] for leg in cc["legs"]] == [False, False]

    def test_garbage_primary_reply_still_counts_as_a_text_summary(self, run):
        # same fallback as the single-endpoint path: prose becomes the summary
        result, _ = run({("vlm_a", "video"): "just prose", ("vlm_b", REFINE): "FAIL"})
        assert result["summary"] == "just prose"

    def test_no_frames_for_one_endpoint_fails_only_that_leg(self, run):
        result, fake = run(
            {("vlm_a", "video"): _json("from a"), ("vlm_b", REFINE): _json("b")},
            no_frames=(1,),
        )
        assert fake.keys == [("vlm_a", "video")]
        assert result["summary"] == "from a" and not _legs(result)[1]["success"]

    def test_no_frames_at_all_is_the_plain_empty_result(self, run):
        result, fake = run({}, no_frames=(0, 1))
        assert fake.calls == [] and "cross_check" not in result and result["summary"] == ""

    def test_per_endpoint_vision_limits_are_planned_separately(self, run, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.set({"max_long_side_px": 336}, "analysis", "endpoints", "vlm_b", "vision")
        with (
            patch("autovideofixer.core.analysis._call_provider_raw", Fake({})),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ) as multi,
        ):
            an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        visions = multi.call_args[0][2]
        assert [v.max_long_side_px for v in visions] == [None, 336]


# ─── judge ─────────────────────────────────────────────────────────

JUDGE = "cross_check:judge"
ARBITER = "cross_check:arbiter"
J = {"mode": "judge", "judge_endpoint": "judge_llm"}


def _describe_script(extra=None):
    return {
        ("vlm_a", "video"): _json("a says cat", rating="G"),
        ("vlm_b", DESCRIBE): _json("b says dog", rating="PG"),
        **(extra or {}),
    }


class TestJudge:
    def test_agreement_accepts_merged(self, run):
        result, fake = run(
            _describe_script({("judge_llm", JUDGE): _verdict(0.9, conflicts=["pet type"])}),
            **J,
        )
        assert fake.keys == [("vlm_a", "video"), ("vlm_b", DESCRIBE), ("judge_llm", JUDGE)]
        assert result["summary"] == "merged"
        cc = result["cross_check"]
        assert (cc["mode"], cc["agreement"], cc["conflicts"]) == ("judge", 0.9, ["pet type"])
        assert (cc["final_source"], cc["final_leg"], cc["arbiter_ran"]) == ("judge_llm", 2, False)
        assert cc["disagreement"] is False
        assert cc["verdict"]["accepted"] and cc["verdict"]["threshold"] == 0.75
        assert cc["verdict"]["merged"]["summary"] == "merged"
        assert [leg["role"] for leg in cc["legs"]] == ["primary", "describe", "judge"]
        assert cc["legs"][2]["result"]["summary"] == "merged"

    def test_describers_are_independent_and_judge_sees_text_only(self, run):
        _, fake = run(_describe_script({("judge_llm", JUDGE): _verdict(1.0)}), **J)
        a, b, judge = fake.calls
        assert "a says cat" not in b["kw"]["user_prompt"]  # no previous JSON in judge mode
        assert a["kw"]["system_prompt"] == b["kw"]["system_prompt"]
        assert judge["kw"]["image_b64_list"] == []
        text = judge["kw"]["user_prompt"]
        assert "a says cat" in text and "b says dog" in text and "vlm_a" in text and "vlm_b" in text
        assert judge["kw"]["response_schema"] == build_judge_schema(OutputsConfig())
        assert judge["kw"]["response_schema"]["required"] == ["agreement", "conflicts", "merged"]
        assert judge["kw"]["model"] == "model-judge_llm"

    def test_threshold_boundary_is_inclusive(self, run):
        script = _describe_script({("judge_llm", JUDGE): _verdict(0.75)})
        result, fake = run(script, agreement_threshold=0.75, arbiter_endpoint="arb", **J)
        assert result["summary"] == "merged" and not result["cross_check"]["arbiter_ran"]
        assert ("arb", ARBITER) not in fake.keys
        script = _describe_script({("judge_llm", JUDGE): _verdict(0.7499)})
        result, _ = run(script, agreement_threshold=0.75, **J)
        assert result["cross_check"]["disagreement"] is True

    def test_below_threshold_arbiter_decides(self, run):
        script = _describe_script(
            {
                ("judge_llm", JUDGE): _verdict(0.2, conflicts=["cat vs dog", "rating"]),
                ("arb", ARBITER): _json("arbiter says fox"),
            }
        )
        result, fake = run(script, arbiter_endpoint="arb", **J)
        assert fake.keys[-1] == ("arb", ARBITER)
        assert result["summary"] == "arbiter says fox"
        cc = result["cross_check"]
        assert (cc["arbiter_ran"], cc["disagreement"], cc["final_source"]) == (True, False, "arb")
        assert cc["legs"][-1]["role"] == "arbiter" and cc["agreement"] == 0.2
        arb = fake.calls[-1]["kw"]
        text = arb["user_prompt"]
        assert "a says cat" in text and "b says dog" in text
        assert "cat vs dog" in text and "rating" in text
        assert len(arb["image_b64_list"]) == 5  # the arbiter's own frames (3 + index 2)
        assert arb["response_schema"] == build_response_schema(OutputsConfig())
        assert arb["model"] == "model-arb"

    def test_below_threshold_without_arbiter_keeps_primary_flagged(self, run):
        script = _describe_script({("judge_llm", JUDGE): _verdict(0.1, conflicts=["x"])})
        result, fake = run(script, **J)
        assert result["summary"] == "a says cat"
        cc = result["cross_check"]
        assert (cc["disagreement"], cc["arbiter_ran"], cc["final_source"]) == (True, False, "vlm_a")
        assert cc["conflicts"] == ["x"] and cc["agreement"] == 0.1
        assert all(k[0] != "arb" for k in fake.keys)

    def test_arbiter_failure_falls_back_to_primary_flagged(self, run):
        for arbiter_reply in ("FAIL", "no json here", "{}"):
            script = _describe_script(
                {("judge_llm", JUDGE): _verdict(0.1), ("arb", ARBITER): arbiter_reply}
            )
            result, _ = run(script, arbiter_endpoint="arb", **J)
            cc = result["cross_check"]
            assert result["summary"] == "a says cat"
            assert (cc["arbiter_ran"], cc["disagreement"], cc["final_source"]) == (
                True,
                True,
                "vlm_a",
            )
            assert not cc["legs"][-1]["success"]

    @pytest.mark.parametrize(
        "reply",
        [
            "FAIL",
            "not json at all",
            "{}",
            '{"agreement": "high", "conflicts": [], "merged": {}}',
            '{"agreement": null}',
            '{"agreement": true}',
            '{"agreement": NaN}',
            "[0.9]",
            RuntimeError("boom"),
        ],
    )
    def test_judge_failure_or_garbage_degrades_to_primary(self, run, reply):
        script = _describe_script({("judge_llm", JUDGE): reply})
        result, fake = run(script, arbiter_endpoint="arb", **J)
        cc = result["cross_check"]
        assert result["summary"] == "a says cat" and cc["final_source"] == "vlm_a"
        assert cc["legs"][-1]["role"] == "judge" and not cc["legs"][-1]["success"]
        assert cc["agreement"] is None and cc["verdict"] is None
        assert cc["disagreement"] is False and cc["arbiter_ran"] is False
        assert ("arb", ARBITER) not in fake.keys

    @pytest.mark.parametrize("given,clamped", [(1.7, 1.0), (-3, 0.0), ("0.8", 0.8), (85, 1.0)])
    def test_agreement_is_clamped(self, given, clamped):
        v = parse_judge_verdict({"agreement": given, "merged": {"summary": "m"}}, ("other",))
        assert isinstance(v, JudgeVerdict) and v.agreement == clamped

    def test_accepted_but_no_merged_falls_back_to_primary(self, run):
        script = _describe_script(
            {("judge_llm", JUDGE): json.dumps({"agreement": 0.95, "conflicts": []})}
        )
        result, _ = run(script, **J)
        cc = result["cross_check"]
        assert result["summary"] == "a says cat" and cc["final_source"] == "vlm_a"
        assert cc["legs"][-1]["success"] is False and cc["agreement"] == 0.95

    def test_primary_fails_fallback_is_the_next_endpoint(self, run):
        script = {
            ("vlm_a", "video"): "FAIL",
            ("vlm_b", DESCRIBE): _json("b alone"),
            ("vlm_c", DESCRIBE): _json("c alone"),
            ("judge_llm", JUDGE): "FAIL",
        }
        result, _ = run(script, endpoints=list(ENDPOINTS), **J)
        assert result["summary"] == "b alone" and result["cross_check"]["final_source"] == "vlm_b"

    def test_only_one_describer_succeeds_skips_the_judge(self, run):
        result, fake = run({("vlm_a", "video"): "FAIL", ("vlm_b", DESCRIBE): _json("b")}, **J)
        assert result["summary"] == "b" and all(k[0] != "judge_llm" for k in fake.keys)
        assert result["cross_check"]["verdict"] is None

    def test_all_describers_fail(self, run):
        result, fake = run({}, arbiter_endpoint="arb", **J)
        assert result["summary"] == "" and result["cross_check"]["final_source"] is None
        assert [k[1] for k in fake.keys] == ["video", DESCRIBE]  # no judge, no arbiter

    def test_judge_defaults_to_the_llm_endpoint(self, run, tmp_path):
        cfg = _cfg(tmp_path, mode="judge")
        cfg.set("judge_llm", "analysis", "llm", "endpoint")
        fake = Fake(_describe_script({("judge_llm", JUDGE): _verdict(0.9)}))
        with (
            patch("autovideofixer.core.analysis._call_provider_raw", fake),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ),
        ):
            result = an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        assert result["cross_check"]["final_source"] == "judge_llm"

    def test_arbiter_frames_are_planned_with_the_arbiter_vision(self, run, tmp_path):
        cfg = _cfg(tmp_path, mode="judge", arbiter_endpoint="arb", judge_endpoint="judge_llm")
        cfg.set({"max_long_side_px": 2000}, "analysis", "endpoints", "arb", "vision")
        with (
            patch("autovideofixer.core.analysis._call_provider_raw", Fake({})),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ) as multi,
        ):
            an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        visions = multi.call_args[0][2]
        assert len(visions) == 3 and visions[2].max_long_side_px == 2000


# ─── purposes, collector, threads, scenes, CLI overrides ──────────


class TestStatsAndThreads:
    def test_purposes_and_collector(self, run):
        script = {
            **_describe_script({("judge_llm", JUDGE): _verdict(0.1), ("arb", ARBITER): _json("z")}),
        }
        with collect_call_stats() as sink:
            result, _ = run(script, arbiter_endpoint="arb", **J)
        assert [c.purpose for c in sink] == ["video", DESCRIBE, JUDGE, ARBITER]
        assert [c.endpoint for c in sink] == ["vlm_a", "vlm_b", "judge_llm", "arb"]
        assert [leg["call"]["purpose"] for leg in _legs(result)] == [c.purpose for c in sink]

    def test_refine_purposes_and_scene_detail(self, run):
        scene = an.SceneEvent(start_time=1.0, end_time=3.0, event_type="x")
        with collect_call_stats() as sink:
            result, _ = run(
                {("vlm_a", "scene"): _json("a"), ("vlm_b", REFINE): _json("b")}, scene=scene
            )
        assert [c.purpose for c in sink] == ["scene", REFINE]
        assert {c.detail for c in sink} == {"scene[1.0-3.0]"}
        assert run.range == (1.0, 3.0)
        assert result["summary"] == "b" and result["call"]["purpose"] == "scene"

    def test_scene_with_zero_duration_makes_no_calls(self, run):
        scene = an.SceneEvent(start_time=2.0, end_time=2.0, event_type="x")
        result, fake = run({}, scene=scene)
        assert fake.calls == [] and "cross_check" not in result

    def test_failed_legs_are_recorded_too(self, run):
        with collect_call_stats() as sink:
            run({("vlm_a", "video"): "FAIL", ("vlm_b", DESCRIBE): "FAIL"}, **J)
        assert [c.success for c in sink] == [False, False]

    def test_parallel_describe_legs_overlap_and_keep_endpoint_order(self, tmp_path):
        """max_parallel=2: legs 0 and 1 must be in flight together (a 2-party barrier fails if
        they are not), the collector must still receive every call from the worker threads, and
        results stay in endpoint order even when a later leg finishes first."""
        barrier = threading.Barrier(2, timeout=5)
        fake = Fake(
            {
                ("vlm_a", "video"): _json("a"),
                ("vlm_b", DESCRIBE): _json("b"),
                ("vlm_c", DESCRIBE): _json("c"),
                ("judge_llm", JUDGE): _verdict(0.9),
            }
        )
        inner = fake.__call__

        def gated(provider, **kw):
            if kw["endpoint_name"] in ("vlm_a", "vlm_b"):
                barrier.wait()
            if kw["endpoint_name"] == "vlm_a":
                time.sleep(0.2)  # finishes after vlm_c
            return inner(provider, **kw)

        cfg = _cfg(tmp_path, endpoints=list(ENDPOINTS), max_parallel=2, **J)
        main_thread = threading.get_ident()
        with (
            patch("autovideofixer.core.analysis._call_provider_raw", gated),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ),
            collect_call_stats() as sink,
        ):
            result = an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        assert not barrier.broken
        describe = [c for c in fake.calls if c["key"][1] != JUDGE]
        assert main_thread not in {c["thread"] for c in describe}
        assert len({c["thread"] for c in describe}) >= 2
        assert sorted(c.endpoint for c in sink) == ["judge_llm", "vlm_a", "vlm_b", "vlm_c"]
        assert [leg["endpoint"] for leg in _legs(result)] == [
            "vlm_a",
            "vlm_b",
            "vlm_c",
            "judge_llm",
        ]
        assert [leg["success"] for leg in _legs(result)] == [True] * 4
        assert result["cross_check"]["final_source"] == "judge_llm"

    def test_max_parallel_one_stays_on_the_calling_thread(self, run):
        _, fake = run(_describe_script(), **J)
        assert {c["thread"] for c in fake.calls} == {threading.get_ident()}

    def test_refine_ignores_max_parallel(self, run):
        _, fake = run(
            {("vlm_a", "video"): _json("a"), ("vlm_b", REFINE): _json("b")}, max_parallel=4
        )
        assert {c["thread"] for c in fake.calls} == {threading.get_ident()}


class TestCliOverrides:
    def test_overrides_apply_to_the_primary_only(self, run):
        _, fake = run(
            {("vlm_c", "video"): _json("p"), ("vlm_b", REFINE): _json("r")},
            analyzer_kwargs={
                "endpoint": "vlm_c",
                "model": "forced",
                "api_url": "http://127.0.0.1:9/z",
            },
        )
        first, second = fake.calls
        assert first["key"] == ("vlm_c", "video") and first["kw"]["model"] == "forced"
        assert first["kw"]["api_url"] == "http://127.0.0.1:9/z"
        assert second["key"] == ("vlm_b", REFINE) and second["kw"]["model"] == "model-vlm_b"
        assert second["kw"]["api_url"] == "http://127.0.0.1:1/x"

    def test_unknown_override_endpoint_is_an_error_before_sampling(self, tmp_path):
        with patch("autovideofixer.core.analysis.sample_vlm_frames_multi") as multi:
            with pytest.raises(ValueError, match="Unknown analysis endpoint 'nope'"):
                an.VideoAnalyzer(_cfg(tmp_path)).run_vlm_analysis("v.mp4", endpoint="nope")
        multi.assert_not_called()

    def test_invalid_config_raises_before_sampling(self, tmp_path):
        with patch("autovideofixer.core.analysis.sample_vlm_frames_multi") as multi:
            with pytest.raises(ValueError, match="needs at least 2"):
                an.VideoAnalyzer(_cfg(tmp_path, endpoints=["vlm_a"])).run_vlm_analysis("v.mp4")
        multi.assert_not_called()


# ─── shared decode cache (real sampler, synthetic ffmpeg) ──────────


class _Spy(vs.FFmpegFrameSource):
    instances: list = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.runs: list[tuple[float, int, int]] = []
        _Spy.instances.append(self)

    def _run(self, seek, w, h, *, fast):
        self.decodes += 1
        self.runs.append((round(seek, 6), w, h))
        rng = np.random.default_rng(int(seek * 1000))
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        return frame.tobytes(), seek


@pytest.fixture
def synthetic(tmp_path):
    """A 1920x1080 / 30 fps / 10 s 'video' decoded by a synthetic ``_run``."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    tl = vs.Timeline(width=1920, height=1080, times=[i / 30 for i in range(300)], duration=10.0)
    _Spy.instances = []
    with (
        patch.object(vs, "probe_timeline", return_value=tl),
        patch.object(vs, "FFmpegFrameSource", _Spy),
        patch("autovideofixer.core.ffmpeg_utils.get_ffmpeg_path", return_value="ffmpeg"),
        patch("autovideofixer.core.ffmpeg_utils.get_ffprobe_path", return_value="ffprobe"),
    ):
        yield str(video)


def _jpeg_size(path):
    h, w = cv2.imread(path).shape[:2]
    return w, h


class TestSharedDecodeCache:
    CFG = vs.SamplingConfig(min_frames=4, max_frames=4, interval_sec=1.0)

    def test_each_frame_decoded_once_across_endpoints(self, synthetic):
        small = vs.VisionLimits(max_long_side_px=480, jpeg_quality=60)
        big = vs.VisionLimits(max_long_side_px=960, jpeg_quality=95)
        out = vs.sample_vlm_frames_multi(synthetic, self.CFG, [small, big, small])
        try:
            (source,) = _Spy.instances
            seeks = [s for s, _, _ in source.runs]
            assert len(seeks) == len(set(seeks)) == source.decodes  # one decode per frame index
            assert {(w, h) for _, w, h in source.runs} == {(960, 540)}  # decoded at the largest
            sizes = [{_jpeg_size(p) for p in s.paths} for s in out]
            assert sizes == [{(480, 270)}, {(960, 540)}, {(480, 270)}]
            assert [s.report.frame_size for s in out] == ["480x270", "960x540", "480x270"]
            assert [len(s.paths) for s in out] == [4, 4, 4]
            # same frames, so the smaller ones are downscaled copies of the big one
            assert len(source.runs) == 4
            # JPEG quality is each endpoint's own: the q60 file is smaller than a q95 one would be
            big_q95 = cv2.imread(out[1].paths[0])
            _, buf95 = cv2.imencode(".jpg", cv2.resize(big_q95, (480, 270)), [1, 95])
            assert __import__("os").path.getsize(out[0].paths[0]) < len(buf95)
        finally:
            import shutil

            for s in out:
                shutil.rmtree(s.tmp_dir, ignore_errors=True)

    def test_different_frame_counts_still_decode_each_index_once(self, synthetic):
        few = vs.VisionLimits(max_frames_per_call=2)
        many = vs.VisionLimits()
        out = vs.sample_vlm_frames_multi(synthetic, self.CFG, [many, few])
        (source,) = _Spy.instances
        seeks = [s for s, _, _ in source.runs]
        assert len(seeks) == len(set(seeks))
        assert [len(s.paths) for s in out] == [4, 2]
        import shutil

        for s in out:
            shutil.rmtree(s.tmp_dir, ignore_errors=True)

    def test_single_endpoint_is_the_original_path(self, synthetic):
        out = vs.sample_vlm_frames_multi(
            synthetic, self.CFG, [vs.VisionLimits(max_long_side_px=480)]
        )
        (source,) = _Spy.instances
        assert source.cache_budget_bytes is None  # rejected frames are freed in place
        assert len(out) == 1 and {_jpeg_size(p) for p in out[0].paths} == {(480, 270)}
        assert {(w, h) for _, w, h in source.runs} == {(480, 270)}  # decoded at its own size
        single = vs.sample_vlm_frames(synthetic, self.CFG, vs.VisionLimits(max_long_side_px=480))
        assert [_jpeg_size(p) for p in single.paths] == [_jpeg_size(p) for p in out[0].paths]
        import shutil

        for s in (*out, single):
            shutil.rmtree(s.tmp_dir, ignore_errors=True)

    def test_cache_budget_bounds_memory_and_redecodes_instead_of_failing(self, synthetic):
        with patch.object(vs, "SHARED_CACHE_BUDGET_BYTES", 1):
            out = vs.sample_vlm_frames_multi(
                synthetic, self.CFG, [vs.VisionLimits(max_long_side_px=480)] * 2
            )
        (source,) = _Spy.instances
        assert source.decodes > 4  # evicted images were decoded again...
        assert [len(s.paths) for s in out] == [4, 4]  # ...and nothing was lost
        assert all(
            c.image is None for c in source._cache.values()
        )  # a 1-byte budget keeps no image beyond the newest
        import shutil

        for s in out:
            shutil.rmtree(s.tmp_dir, ignore_errors=True)

    def test_images_are_released_after_writing(self, synthetic):
        out = vs.sample_vlm_frames_multi(synthetic, self.CFG, [vs.VisionLimits()] * 2)
        (source,) = _Spy.instances
        assert source._cache == {}  # source.clear() ran
        import shutil

        for s in out:
            shutil.rmtree(s.tmp_dir, ignore_errors=True)

    def test_empty_visions_and_unprobeable_video(self, synthetic, tmp_path):
        assert vs.sample_vlm_frames_multi(synthetic, self.CFG, []) == []
        with patch.object(vs, "probe_timeline", return_value=None):
            res = vs.sample_vlm_frames_multi(synthetic, self.CFG, [vs.VisionLimits()] * 2)
        assert [r.paths for r in res] == [[], []]
        assert res[0].report is not res[1].report


# ─── output: console, CSV, JSON, process report ────────────────────


def _analysis_with_block(**cc):
    block = {
        "mode": "judge",
        "endpoints": ["vlm_a", "vlm_b"],
        "final_source": "arb",
        "final_leg": 3,
        "agreement": 0.4,
        "conflicts": ["cat vs dog"],
        "arbiter_ran": True,
        "disagreement": False,
        "legs": [
            {
                "endpoint": "vlm_a", "role": "primary", "success": True, "result": {},
                "call": {"a": 1},
            },
            {
                "endpoint": "vlm_b", "role": "describe", "success": False, "result": None,
                "call": None, "error": "timeout", "changes": ["fixed tags"],
            },
        ],
        "verdict": {
            "agreement": 0.4, "threshold": 0.75, "accepted": False, "conflicts": [],
            "merged": None,
        },
        **cc,
    }  # fmt: skip
    return an.VideoAnalysis(
        filepath="/x/clip.mp4", filename="clip.mp4", vlm_summary="s", cross_check=block,
        vlm_calls=[{"purpose": "video", "endpoint": "vlm_a", "success": True}],
    )  # fmt: skip


class TestOutput:
    PRIOR_COLUMNS = [
        "filepath", "filename", "duration_sec", "resolution", "framerate_fps", "video_codec",
        "has_video", "has_audio", "hdr", "scenes_detected", "scene_boundaries", "vlm_summary",
        "vlm_tags", "vlm_objects", "content_rating", "media_type", "rating_reasoning",
        "vlm_extra", "vlm_calls", "vlm_wall_sec", "vlm_input_tokens", "vlm_output_tokens",
        "vlm_frames", "vlm_dropped_duplicates", "vlm_frame_size", "vlm_est_vision_tokens",
    ]  # fmt: skip

    def test_csv_columns_are_appended_and_order_is_stable(self):
        cols = cli_mod._ANALYZE_CSV_FIELDS
        assert cols[: len(self.PRIOR_COLUMNS)] == self.PRIOR_COLUMNS
        assert cols[len(self.PRIOR_COLUMNS) :] == [
            "cross_check_mode",
            "cross_check_agreement",
            "cross_check_arbiter_ran",
            "cross_check_final_source",
            "cross_check_disagreement",
            # § 28 columns, appended after these (never reordering anything before them).
            "analysis_profile",
            "analysis_scope",
            "scene_vlm",
        ]

    def test_csv_row_values(self, tmp_path):
        row = cli_mod._analysis_to_csv_row(_analysis_with_block())
        assert (row["cross_check_mode"], row["cross_check_agreement"]) == ("judge", 0.4)
        assert (row["cross_check_arbiter_ran"], row["cross_check_final_source"]) == (True, "arb")
        assert row["cross_check_disagreement"] is False
        path = tmp_path / "o.csv"
        cli_mod._write_analysis_csv(str(path), [row])
        got = next(csv.DictReader(path.open()))
        assert got["cross_check_mode"] == "judge" and got["cross_check_final_source"] == "arb"
        # refine has no agreement: blank, not "None"
        refine = cli_mod._analysis_to_csv_row(
            _analysis_with_block(mode="refine", agreement=None, arbiter_ran=False)
        )
        assert refine["cross_check_agreement"] == ""

    def test_json_carries_the_whole_block(self):
        doc = cli_mod._analysis_to_json(_analysis_with_block())
        assert doc["cross_check"]["legs"][1]["error"] == "timeout"
        assert doc["cross_check"]["verdict"]["threshold"] == 0.75
        json.dumps(doc)

    @pytest.mark.parametrize("width", [60, 200])
    def test_console_shows_agreement_arbiter_and_legs(self, capsys, width):
        cli_mod.console.width = width
        cli_mod._print_analysis_result(_analysis_with_block(), full_output=True)
        out = capsys.readouterr().out
        for needle in ("judge", "0.40", "threshold 0.75", "Arbiter Ran", "yes", "arb"):
            assert needle in out
        assert "FAILED (timeout)" in out
        assert "cat vs dog" in out and "fixed tags" in out  # --full panels
        cli_mod.console.width = 200
        cli_mod._print_analysis_result(_analysis_with_block(), full_output=False)
        out = capsys.readouterr().out
        assert "Cross-Check Agreement" in out and "cat vs dog" not in out

    def test_console_flags_disagreement_and_total_failure(self, capsys):
        cli_mod.console.width = 200
        a = _analysis_with_block(disagreement=True, arbiter_ran=False, final_source="vlm_a")
        cli_mod._print_analysis_result(a, full_output=False)
        assert "DISAGREEMENT" in capsys.readouterr().out
        a = _analysis_with_block(final_source=None, agreement=None, mode="refine")
        cli_mod._print_analysis_result(a, full_output=False)
        out = capsys.readouterr().out
        assert "every leg failed" in out and "Cross-Check Agreement" not in out

    def test_no_block_no_rows(self, capsys):
        cli_mod.console.width = 200
        cli_mod._print_analysis_result(
            an.VideoAnalysis(filepath="/a", filename="a"), full_output=True
        )
        assert "Cross-Check" not in capsys.readouterr().out

    def test_analyze_uses_every_leg_call_and_block(self, tmp_path):
        cfg = _cfg(tmp_path)
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        fake = Fake({("vlm_a", "video"): _json("a"), ("vlm_b", REFINE): _json("b")})
        with (
            patch("autovideofixer.core.ffmpeg_utils.probe", _fake_probe),
            patch("autovideofixer.core.analysis._call_provider_raw", fake),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ),
        ):
            analysis = an.VideoAnalyzer(cfg).analyze(
                str(vid), include_vlm=True, include_events=False
            )
        assert [c["purpose"] for c in analysis.vlm_calls] == ["video", REFINE]
        assert analysis.cross_check and analysis.cross_check["final_source"] == "vlm_b"
        assert analysis.vlm_summary == "b"

    def test_cli_rejects_a_bad_cross_check_up_front(self, tmp_path):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text(
            "config_version: 2\nanalysis:\n  cross_check:\n    enabled: true\n"
            "    endpoints: [default_vlm]\n"
        )
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        with patch("autovideofixer.cli.cli.is_video_file", return_value=True):
            r = CliRunner().invoke(
                cli_mod.main, ["--config", str(cfgp), "analyze", str(vid), "--vlm"]
            )
        assert r.exit_code == 2 and "needs at least 2" in r.output

    def test_cli_end_to_end_with_fakes(self, tmp_path):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text(
            "config_version: 2\nanalysis:\n  endpoints:\n"
            "    one: {provider: llamacpp, model: m1, api_url: 'http://127.0.0.1:1/x'}\n"
            "    two: {provider: llamacpp, model: m2, api_url: 'http://127.0.0.1:1/x'}\n"
            "  cross_check: {enabled: true, mode: refine, endpoints: [one, two]}\n"
        )
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        fake = Fake({("one", "video"): _json("first"), ("two", REFINE): _json("second")})
        out_json, out_csv = tmp_path / "o.json", tmp_path / "o.csv"
        with (
            patch("autovideofixer.cli.cli.is_video_file", return_value=True),
            patch("autovideofixer.core.ffmpeg_utils.probe", _fake_probe),
            patch("autovideofixer.core.analysis._call_provider_raw", fake),
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames_multi",
                side_effect=lambda *a, **k: _frames_for(tmp_path, a[2]),
            ),
        ):
            r = CliRunner().invoke(
                cli_mod.main,
                ["--config", str(cfgp), "analyze", str(vid), "--vlm", "--no-events", "--full",
                 "--json", str(out_json), "--csv", str(out_csv)],
            )  # fmt: skip
        assert r.exit_code == 0, r.output
        assert "Cross-Check Mode" in r.output and "second" in r.output
        doc = json.loads(out_json.read_text())["files"][0]
        assert doc["cross_check"]["final_source"] == "two"
        assert [c["purpose"] for c in doc["vlm_calls"]] == ["video", REFINE]
        row = next(csv.DictReader(out_csv.open()))
        assert row["cross_check_mode"] == "refine" and row["cross_check_final_source"] == "two"
        assert row["vlm_calls"] == "2"

    def test_scene_summaries_and_report_carry_the_block(self, tmp_path):
        from autovideofixer.core import scenes as sc

        scene = an.SceneEvent(start_time=0.0, end_time=5.0, event_type="x")
        block = {"mode": "refine", "legs": []}
        with patch.object(
            an.VideoAnalyzer,
            "run_vlm_analysis_for_scene",
            return_value={"summary": "s", "cross_check": block},
        ):
            sm = sc._collect_scene_vlm_summaries("v.mp4", [scene], Config(tmp_path / "n.yaml"))
        assert sm[0]["cross_check"] == block
        off = {"summary": "s"}
        with patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", return_value=off):
            sm = sc._collect_scene_vlm_summaries("v.mp4", [scene], Config(tmp_path / "n.yaml"))
        assert sm[0]["cross_check"] is None


class TestVerdictAndSchemas:
    def test_conflicts_normalisation(self):
        v = parse_judge_verdict(
            {"agreement": 0.5, "conflicts": [" a ", "", {"k": 1}], "merged": "nope"}, ("other",)
        )
        assert v is not None and v.conflicts == ["a", '{"k": 1}'] and v.merged is None
        v = parse_judge_verdict({"agreement": 0.5, "conflicts": "one problem"}, ("other",))
        assert v is not None and v.conflicts == ["one problem"]

    def test_merged_is_parsed_like_a_model_reply(self):
        v = parse_judge_verdict(
            {
                "agreement": 1,
                "merged": {"summary": "m", "tags": "x, y", "media_type": "3D Rendered"},
            },
            ("real_footage", "3d_rendered", "other"),
        )
        assert v is not None and v.merged is not None
        assert v.merged["tags"] == ["x", "y"] and v.merged["media_type"] == "3d_rendered"

    def test_schemas_follow_the_outputs_config(self):
        outputs = OutputsConfig(media_type=False, rating_reasoning=True)
        assert list(build_refine_schema(outputs)["properties"]) == [
            "summary", "tags", "objects", "rating", "rating_reasoning", "changes",
        ]  # fmt: skip
        merged = build_judge_schema(outputs)["properties"]["merged"]
        assert merged == build_response_schema(outputs)
