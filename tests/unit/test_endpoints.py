"""Endpoint registry, compat payload shapes, response cleaning and a real-network stub
(REQUIREMENTS.md § 24.4/24.5)."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from autovideofixer.cli.cli import main
from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.vlm_sampling import SampledFrames, SamplingReport
from autovideofixer.endpoints import all_endpoint_urls, resolve_endpoint, resolve_role_endpoint


def _cfg(tmp_path, **endpoints) -> Config:
    cfg = Config(tmp_path / "none.yaml")
    for name, ep in endpoints.items():
        for k, v in ep.items():
            cfg.set(v, "analysis", "endpoints", name, k)
    return cfg


class TestResolver:
    def test_defaults(self, tmp_path):
        ep = resolve_endpoint(_cfg(tmp_path), "default_vlm")
        assert (ep.provider, ep.model, ep.timeout_sec, ep.max_tokens) == (
            "ollama",
            "llava",
            120.0,
            1024,
        )
        assert resolve_endpoint(_cfg(tmp_path), "default_llm").max_tokens == 4096

    def test_unknown_name_clear_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown analysis endpoint 'nope'.*default_vlm"):
            resolve_endpoint(_cfg(tmp_path), "nope")

    def test_role_selector_and_override(self, tmp_path):
        cfg = _cfg(tmp_path, mine={"provider": "llamacpp", "model": "m"})
        cfg.set("mine", "analysis", "vlm", "endpoint")
        assert resolve_role_endpoint(cfg, "vlm").name == "mine"
        assert resolve_role_endpoint(cfg, "vlm", "default_vlm").name == "default_vlm"

    @pytest.mark.parametrize("raw,expected", [(None, None), (0, None), (600, 600.0), ("30", 30.0)])
    def test_timeout_semantics(self, tmp_path, raw, expected):
        cfg = _cfg(tmp_path, t={"provider": "api", "timeout_sec": raw})
        assert resolve_endpoint(cfg, "t").timeout_sec == expected

    def test_bad_values(self, tmp_path):
        with pytest.raises(ValueError, match="timeout"):
            resolve_endpoint(_cfg(tmp_path, t={"timeout_sec": -1}), "t")
        with pytest.raises(ValueError, match="provider"):
            resolve_endpoint(_cfg(tmp_path, t={"provider": "bogus"}), "t")
        with pytest.raises(ValueError, match="response_format"):
            resolve_endpoint(
                _cfg(tmp_path, t={"provider": "api", "compat": {"response_format": "x"}}), "t"
            )

    def test_local_alias(self, tmp_path):
        assert resolve_endpoint(_cfg(tmp_path, t={"provider": "local"}), "t").provider == "ollama"

    def test_compat_defaults_per_provider(self, tmp_path):
        cfg = _cfg(tmp_path, a={"provider": "api"}, l={"provider": "llamacpp"})
        a, ll = resolve_endpoint(cfg, "a").compat, resolve_endpoint(cfg, "l").compat
        assert (a["system_content"], a["response_format"]) == ("string", "json_object")
        assert (ll["system_content"], ll["response_format"]) == ("parts", "none")

    def test_all_urls(self, tmp_path):
        cfg = _cfg(tmp_path, a={"api_url": "http://a"}, b={"api_url": "http://b"})
        assert set(all_endpoint_urls(cfg)) == {"http://a", "http://b"}


class TestPayloadShape:
    def _payload(self, compat, schema=None):
        return an._build_openai_payload("m", "SYS", "USR", ["QUJD"], 50, compat, schema)

    def test_default_shape_unchanged(self):
        p = self._payload(None)
        assert p["messages"][0] == {"role": "system", "content": "SYS"}
        assert p["response_format"] == {"type": "json_object"}

    def test_llamacpp_shape(self, tmp_path):
        compat = resolve_endpoint(_cfg(tmp_path, l={"provider": "llamacpp"}), "l").compat
        p = self._payload(compat)
        assert p["messages"][0]["content"] == [{"type": "text", "text": "SYS"}]
        assert "response_format" not in p

    def test_json_schema_and_fallback(self):
        compat = {"system_content": "string", "response_format": "json_schema", "extra_body": {}}
        p = self._payload(compat, an._VLM_RESPONSE_SCHEMA)
        assert p["response_format"]["type"] == "json_schema"
        assert p["response_format"]["json_schema"]["schema"]["properties"]["summary"]
        assert self._payload(compat, None)["response_format"] == {"type": "json_object"}

    def test_extra_body_deep_merged(self):
        compat = {
            "system_content": "string",
            "response_format": "none",
            "extra_body": {"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": True}},
        }
        p = self._payload(compat)
        assert p["cache_prompt"] is True
        assert p["chat_template_kwargs"] == {"enable_thinking": True}
        assert p["model"] == "m"


class TestResponseParsing:
    def test_think_block_and_prose(self):
        text = (
            '<think>hmm {"summary": "wrong"}</think>\nSure! Here:\n'
            '{"summary": "ok", "tags": "a, b", "objects": [], "rating": "G"}\nThanks'
        )
        r = an._parse_vlm_response(text)
        assert r["summary"] == "ok" and r["tags"] == ["a", "b"] and r["rating"] == "G"

    def test_unterminated_think_gives_empty_summary(self):
        assert an._parse_vlm_response("<think>I am still reasoning")["summary"] == ""

    def test_stray_closer(self):
        r = an._parse_vlm_response('reasoning...</think>{"summary": "x"}')
        assert r["summary"] == "x"

    def test_braces_inside_strings(self):
        r = an._parse_vlm_response('pre {"summary": "a } b { c", "tags": []} post')
        assert r["summary"] == "a } b { c"

    def test_nested_and_first_object_wins(self):
        r = an._parse_vlm_response('{"summary": "s", "x": {"y": {"z": 1}}} {"summary": "other"}')
        assert r["summary"] == "s"

    def test_skips_non_json_braces(self):
        r = an._parse_vlm_response('use {curly} words then {"summary": "real"}')
        assert r["summary"] == "real"

    def test_fenced(self):
        assert an._parse_vlm_response('```json\n{"summary": "f"}\n```')["summary"] == "f"

    def test_truncated_falls_back_to_text(self):
        r = an._parse_vlm_response('{"summary": "abc", "tags": [')
        assert r["summary"].startswith('{"summary"') and r["tags"] == []

    def test_plain_text(self):
        assert an._parse_vlm_response("just words")["summary"] == "just words"

    def test_coordinator_parser_uses_extraction(self):
        text = (
            '<think>x</think> Result: {"main_content_summary": "m", "drop": [1], '
            '"reasons": {"1": "ad"}} done'
        )
        assert an._parse_coordinator_response(text, 3)["drop"] == [1]
        assert an._parse_coordinator_response("<think>cut off", 3) is None

    def test_crop_parser_uses_extraction(self):
        text = '<think>maybe yes</think>Answer: {"content_outside": false, "reason": "black"}'
        assert an._parse_crop_vlm_response(text) == {"content_outside": False, "reason": "black"}


class _Stub:
    """Local HTTP server recording requests; ``handler(body)`` -> (delay_s, response_obj)."""

    def __init__(self, reply, delay=0.0):
        self.requests: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                time.sleep(delay)
                data = json.dumps(reply).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _sampled(paths):
    """A ``SampledFrames`` as the sampler would return it (files stay put: tmp_dir is empty)."""
    return SampledFrames(
        paths=[str(p) for p in paths],
        tmp_dir="",
        report=SamplingReport(frames=len(paths), width=1, height=1),
    )


@pytest.fixture
def frame(tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8fakejpeg")
    return str(p)


LLAMA_REPLY = {
    "choices": [
        {
            "message": {
                "content": "<think>let me look {braces}</think>Here you go:\n"
                '{"summary": "a cat", "tags": "cat, pet", "objects": ["cat"], "rating": "G"}\nBye'
            }
        }
    ]
}


class TestRealNetworkStub:
    def test_llamacpp_payload_and_parse(self, tmp_path, frame):
        stub = _Stub(LLAMA_REPLY)
        try:
            cfg = _cfg(
                tmp_path,
                lc={
                    "provider": "llamacpp",
                    "model": "qwen",
                    "api_url": stub.url,
                    "timeout_sec": 30,
                    "compat": {"extra_body": {"cache_prompt": True}},
                },
            )
            ep = resolve_endpoint(cfg, "lc")
            out = an._dispatch_vlm(
                [frame],
                ep.provider,
                ep.model,
                ep.api_url,
                ep.api_key,
                "SYS",
                "USR",
                allow_http=ep.allow_http,
                max_tokens=ep.max_tokens,
                timeout=ep.timeout_sec,
                compat=ep.compat,
            )
        finally:
            stub.close()
        body = stub.requests[0]["body"]
        assert body["messages"][0]["content"] == [{"type": "text", "text": "SYS"}]
        assert "response_format" not in body
        assert body["cache_prompt"] is True and body["model"] == "qwen"
        assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")
        assert out == {
            "summary": "a cat",
            "tags": ["cat", "pet"],
            "objects": ["cat"],
            "rating": "G",
        }

    def test_per_endpoint_timeout_enforced(self, tmp_path, frame, caplog):
        stub = _Stub(LLAMA_REPLY, delay=2.0)
        try:
            cfg = _cfg(
                tmp_path, slow={"provider": "llamacpp", "api_url": stub.url, "timeout_sec": 0.3}
            )
            ep = resolve_endpoint(cfg, "slow")
            t0 = time.monotonic()
            out = an._dispatch_vlm(
                [frame], ep.provider, ep.model, ep.api_url, "", "S", "U", timeout=ep.timeout_sec
            )
            elapsed = time.monotonic() - t0
        finally:
            stub.close()
        assert elapsed < 1.5 and out["summary"] == ""

    def test_unlimited_timeout_waits(self, tmp_path, frame):
        stub = _Stub(LLAMA_REPLY, delay=0.5)
        try:
            cfg = _cfg(
                tmp_path, u={"provider": "llamacpp", "api_url": stub.url, "timeout_sec": None}
            )
            ep = resolve_endpoint(cfg, "u")
            out = an._dispatch_vlm(
                [frame], ep.provider, ep.model, ep.api_url, "", "S", "U", timeout=ep.timeout_sec
            )
        finally:
            stub.close()
        assert out["summary"] == "a cat"

    def test_coordinator_end_to_end(self, tmp_path):
        reply = {
            "choices": [
                {
                    "message": {
                        "content": '<think>x</think>{"main_content_summary": "m", "drop": [0], '
                        '"reasons": {"0": "ad"}}'
                    }
                }
            ]
        }
        stub = _Stub(reply)
        try:
            cfg = _cfg(tmp_path, co={"provider": "llamacpp", "api_url": stub.url})
            cfg.set("co", "analysis", "llm", "endpoint")
            res = an.run_scene_coordinator(
                [{"index": 0, "start_time": 0, "end_time": 1, "duration": 1, "summary": "s"}], cfg
            )
        finally:
            stub.close()
        assert res["drop"] == [0] and not res["failed"]
        assert "response_format" not in stub.requests[0]["body"]


class TestConsumersUseEndpoint:
    def _analyzer_cfg(self, tmp_path, **ep):
        cfg = _cfg(tmp_path, mine={"provider": "llamacpp", **ep})
        cfg.set("mine", "analysis", "vlm", "endpoint")
        return cfg

    @patch("autovideofixer.core.analysis.sample_vlm_frames")
    @patch("autovideofixer.core.analysis._dispatch_vlm")
    def test_whole_video(self, disp, extract, tmp_path, frame):
        extract.return_value = _sampled([frame])
        disp.return_value = {}
        cfg = self._analyzer_cfg(
            tmp_path, model="mm", timeout_sec=77, api_url="http://127.0.0.1:1/x"
        )
        an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        args, kw = disp.call_args
        assert args[1:4] == ("llamacpp", "mm", "http://127.0.0.1:1/x")
        assert kw["timeout"] == 77.0 and kw["compat"]["system_content"] == "parts"

    @patch("autovideofixer.core.analysis.sample_vlm_frames")
    @patch("autovideofixer.core.analysis._dispatch_vlm")
    def test_override_and_explicit_endpoint(self, disp, extract, tmp_path, frame):
        extract.return_value = _sampled([frame])
        disp.return_value = {}
        cfg = self._analyzer_cfg(tmp_path, model="mm")
        an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4", endpoint="default_vlm", model="zzz")
        assert disp.call_args[0][1:3] == ("ollama", "zzz")

    def test_unknown_endpoint_raises_before_extraction(self, tmp_path):
        with patch("autovideofixer.core.analysis.sample_vlm_frames") as ex:
            with pytest.raises(ValueError, match="Unknown analysis endpoint"):
                an.VideoAnalyzer(Config(tmp_path / "n.yaml")).run_vlm_analysis(
                    "v.mp4", endpoint="x"
                )
            ex.assert_not_called()

    @patch("autovideofixer.core.analysis.sample_vlm_frames")
    @patch("autovideofixer.core.analysis._dispatch_vlm")
    def test_per_scene(self, disp, extract, tmp_path, frame):
        extract.return_value = _sampled([frame])
        disp.return_value = {}
        cfg = self._analyzer_cfg(tmp_path, timeout_sec=5)
        scene = an.SceneEvent(start_time=0.0, end_time=4.0, event_type="x")
        an.VideoAnalyzer(cfg).run_vlm_analysis_for_scene("v.mp4", scene)
        assert disp.call_args[1]["timeout"] == 5.0

    def test_crop_check_and_coordinator_use_timeout_and_tls(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            c={
                "provider": "llamacpp",
                "timeout_sec": 9,
                "tls": {"mode": "insecure"},
                "api_url": "http://127.0.0.1:1/x",
            },
        )
        cfg.set("c", "analysis", "vlm", "endpoint")
        cfg.set("c", "analysis", "llm", "endpoint")
        f = tmp_path / "a.jpg"
        f.write_bytes(b"x")
        with patch(
            "autovideofixer.core.analysis._call_provider_raw",
            return_value='{"content_outside": true, "reason": "r"}',
        ) as call:
            an.run_crop_vlm_check(str(f), str(f), cfg)
            kw = call.call_args[1]
            assert kw["timeout"] == 9.0 and kw["tls"].mode == "insecure"
            assert kw["tls"].label == "crop-vlm:c"
            call.return_value = '{"main_content_summary": "", "drop": [], "reasons": {}}'
            an.run_scene_coordinator(
                [{"index": 0, "start_time": 0, "end_time": 1, "duration": 1}], cfg
            )
            kw = call.call_args[1]
            assert kw["timeout"] == 9.0 and kw["tls"].label == "endpoint:c"

    def test_endpoint_tls_reaches_provider_call_in_whole_video(self, tmp_path, frame):
        cfg = self._analyzer_cfg(tmp_path)
        cfg.set({"mode": "untrusted_ca"}, "analysis", "endpoints", "mine", "tls")
        with (
            patch("autovideofixer.core.analysis.sample_vlm_frames", return_value=_sampled([frame])),
            patch("autovideofixer.core.analysis._dispatch_vlm", return_value={}) as disp,
        ):
            an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        assert disp.call_args[1]["tls"].mode == "untrusted_ca"

    def test_coordinator_unknown_endpoint_fails_open(self, tmp_path):
        cfg = Config(tmp_path / "n.yaml")
        cfg.set("ghost", "analysis", "llm", "endpoint")
        res = an.run_scene_coordinator(
            [{"index": 0, "start_time": 0, "end_time": 1, "duration": 1}], cfg
        )
        assert res["failed"] and res["drop"] == []


class TestCli:
    def _analyze(self, tmp_path, args, cfg_text=""):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text(cfg_text or "config_version: 2\n")
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        captured = {}

        def fake(self, filepath, **kw):
            captured.update(kw)
            raise RuntimeError("stop")

        with (
            patch("autovideofixer.cli.cli.is_video_file", return_value=True),
            patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", fake),
        ):
            r = CliRunner().invoke(main, ["--config", str(cfgp), "analyze", str(vid), *args])
        return r, captured

    def test_endpoint_flag_forwarded(self, tmp_path):
        r, cap = self._analyze(
            tmp_path,
            ["--vlm", "--vlm-endpoint", "mine", "--vlm-model", "m", "--vlm-url", "http://u"],
            "config_version: 2\nanalysis:\n  endpoints:\n    mine:\n      provider: api\n",
        )
        assert cap["vlm_endpoint"] == "mine" and cap["vlm_model"] == "m"
        assert cap["vlm_api_url"] == "http://u"

    def test_unknown_endpoint_is_usage_error(self, tmp_path):
        r, cap = self._analyze(tmp_path, ["--vlm", "--vlm-endpoint", "ghost"])
        assert r.exit_code == 2 and "Unknown analysis endpoint" in r.output and not cap
