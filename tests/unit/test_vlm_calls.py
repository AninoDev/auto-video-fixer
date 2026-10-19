"""§ 26.2: call-stat extraction, error classes, recording, console/CSV/JSON reporting, and the
timeout traceback fix. HTTP is faked at ``open_url`` or served by a real local socket."""

from __future__ import annotations

import csv
import json
import logging
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from autovideofixer.cli import cli as cli_mod
from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.vlm_calls import (
    CallStats,
    apply_ollama_usage,
    apply_openai_usage,
    classify_exception,
    collect_call_stats,
    failure_message,
    format_call_line,
    record_call,
    summarize_calls,
)

OLLAMA_BODY = {
    "model": "llava",
    "message": {"role": "assistant", "content": "{}"},
    "done": True,
    "total_duration": 5_000_000_000,
    "load_duration": 100_000_000,
    "prompt_eval_count": 1234,
    "prompt_eval_duration": 2_000_000_000,
    "eval_count": 87,
    "eval_duration": 2_900_000_000,
}
LLAMA_BODY = {
    "choices": [{"message": {"content": "{}"}}],
    "usage": {"prompt_tokens": 2048, "completion_tokens": 120, "total_tokens": 2168},
    "timings": {
        "prompt_n": 2040, "prompt_ms": 400.0, "prompt_per_second": 5100.0,
        "predicted_n": 120, "predicted_ms": 1200.0, "predicted_per_second": 100.0,
    },
}  # fmt: skip
OPENAI_BODY = {
    "choices": [{"message": {"content": "{}"}}],
    "usage": {
        "prompt_tokens": 900,
        "completion_tokens": 300,
        "total_tokens": 1200,
        "completion_tokens_details": {"reasoning_tokens": 200},
    },
}


class TestExtraction:
    def test_ollama(self):
        s = CallStats()
        apply_ollama_usage(s, OLLAMA_BODY, wall=5.2)
        assert (s.input_tokens, s.output_tokens, s.total_tokens) == (1234, 87, 1321)
        assert s.prompt_tps == pytest.approx(617.0)
        assert s.gen_tps == pytest.approx(87 / 2.9)
        assert not s.tps_estimated and s.usage_source == "ollama"
        assert s.raw["eval_count"] == 87

    def test_ollama_partial_and_missing(self):
        s = CallStats()
        apply_ollama_usage(s, {"eval_count": 10}, wall=1.0)
        assert s.input_tokens is None and s.output_tokens == 10 and s.total_tokens == 10
        assert s.gen_tps is None and s.prompt_tps is None  # no durations -> no rate
        s = CallStats()
        apply_ollama_usage(s, {}, wall=1.0)
        assert s.total_tokens is None and s.usage_source == "none"

    def test_ollama_zero_duration_gives_no_rate(self):
        s = CallStats()
        apply_ollama_usage(s, {"eval_count": 5, "eval_duration": 0}, wall=1.0)
        assert s.gen_tps is None

    def test_llamacpp_usage_and_timings(self):
        s = CallStats()
        apply_openai_usage(s, LLAMA_BODY, wall=1.7)
        assert (s.input_tokens, s.output_tokens, s.total_tokens) == (2048, 120, 2168)
        assert s.prompt_tps == 5100.0 and s.gen_tps == 100.0
        assert not s.tps_estimated and s.usage_source == "usage+timings"

    def test_llamacpp_timings_only(self):
        body = {"choices": [], "timings": LLAMA_BODY["timings"]}
        s = CallStats()
        apply_openai_usage(s, body, wall=1.0)
        assert (s.input_tokens, s.output_tokens, s.total_tokens) == (2040, 120, 2160)
        assert s.usage_source == "timings"

    def test_llamacpp_rate_from_n_and_ms_when_per_second_missing(self):
        body = {
            "timings": {
                "prompt_n": 100,
                "prompt_ms": 50.0,
                "predicted_n": 10,
                "predicted_ms": 200.0,
            }
        }
        s = CallStats()
        apply_openai_usage(s, body, wall=1.0)
        assert s.prompt_tps == pytest.approx(2000.0) and s.gen_tps == pytest.approx(50.0)
        assert not s.tps_estimated

    def test_openai_usage_with_reasoning_is_estimated(self):
        s = CallStats()
        apply_openai_usage(s, OPENAI_BODY, wall=4.0)
        assert (s.input_tokens, s.output_tokens, s.total_tokens) == (900, 300, 1200)
        assert s.reasoning_tokens == 200
        assert s.gen_tps == pytest.approx(75.0) and s.tps_estimated
        assert s.prompt_tps is None and s.usage_source == "usage"

    def test_openai_missing_usage(self):
        s = CallStats()
        apply_openai_usage(s, {"choices": []}, wall=2.0)
        assert s.input_tokens is None and s.gen_tps is None and not s.tps_estimated
        assert s.usage_source == "none" and s.raw == {}

    def test_openai_partial_usage_and_garbage_types(self):
        s = CallStats()
        apply_openai_usage(s, {"usage": {"prompt_tokens": 7, "completion_tokens": "x"}}, wall=1.0)
        assert s.input_tokens == 7 and s.output_tokens is None and s.total_tokens == 7
        s = CallStats()
        apply_openai_usage(s, {"usage": "nope", "timings": [1]}, wall=1.0)
        assert s.input_tokens is None

    def test_to_dict_drops_raw_and_rounds(self):
        s = CallStats(wall_sec=1.23456, gen_tps=3.14159, raw={"a": 1})
        d = s.to_dict()
        assert "raw" not in d and d["wall_sec"] == 1.235 and d["gen_tps"] == 3.142
        json.dumps(d)


class TestErrorClasses:
    @pytest.mark.parametrize(
        "exc,cls",
        [
            (TimeoutError("timed out"), "timeout"),
            (socket.timeout(), "timeout"),
            (urllib.error.URLError(TimeoutError("t")), "timeout"),
            (
                urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
                "connection",
            ),
            (ConnectionResetError("reset"), "connection"),
            (urllib.error.HTTPError("u", 500, "Internal Server Error", {}, None), "http_500"),
            (urllib.error.HTTPError("u", 401, "Unauthorized", {}, None), "http_401"),
            (json.JSONDecodeError("x", "y", 0), "bad_json"),
            (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"), "bad_json"),
            (KeyError("choices"), "bad_response"),
            (RuntimeError("boom"), "RuntimeError"),
        ],
    )
    def test_classify(self, exc, cls):
        assert classify_exception(exc)[0] == cls

    def test_messages_have_no_url(self):
        _, msg = classify_exception(urllib.error.URLError("[Errno 111] Connection refused"))
        assert "http" not in msg

    def test_timeout_message_names_endpoint_timeout_and_hint(self):
        s = CallStats(endpoint="slow", provider="llamacpp", model="m", purpose="video")
        s.success, s.error_class, s.error = False, "timeout", "request timed out"
        msg = failure_message(s, 2.0)
        assert "'slow'" in msg and "2s" in msg and "analysis.endpoints.slow.timeout_sec" in msg
        assert "\n" not in msg and "Traceback" not in msg


class TestCollector:
    def test_nested_collectors_all_receive(self):
        with collect_call_stats() as outer:
            record_call(CallStats(purpose="a"))
            with collect_call_stats() as inner:
                record_call(CallStats(purpose="b"))
            record_call(CallStats(purpose="c"))
        assert [c.purpose for c in outer] == ["a", "b", "c"]
        assert [c.purpose for c in inner] == ["b"]

    def test_no_collector_is_a_noop(self):
        record_call(CallStats())  # must not raise

    def test_totals(self):
        calls = [
            CallStats(wall_sec=1.0, input_tokens=10, output_tokens=5, total_tokens=15, images=2),
            CallStats(wall_sec=2.5, success=False, error_class="timeout", images=1),
        ]
        t = summarize_calls(calls)
        assert t["calls"] == 2 and t["failed"] == 1 and t["succeeded"] == 1
        assert t["wall_sec"] == 3.5 and t["input_tokens"] == 10 and t["images"] == 3
        assert summarize_calls([c.to_dict() for c in calls]) == t

    def test_format_line(self):
        s = CallStats(endpoint="e", provider="openai", model="m", purpose="video", wall_sec=4.0)
        apply_openai_usage(s, OPENAI_BODY, 4.0)
        line = format_call_line(s)
        assert "in=900" in line and "out=300 (reasoning 200)" in line and "75.0~" in line
        assert "(estimated)" in line
        s.success, s.error_class = False, "timeout"
        assert "FAILED (timeout)" in format_call_line(s)


def _mock_resp(body: dict | bytes):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = MagicMock()
    r.read.return_value = raw
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    return r


class TestProviderFunctions:
    @patch("autovideofixer.core.analysis.open_url")
    def test_ollama_response_carries_stats(self, m):
        m.return_value = _mock_resp(OLLAMA_BODY)
        r = an._call_ollama("http://localhost:11434", "llava", "s", "u", ["a", "b"])
        assert r.stats.input_tokens == 1234 and r.stats.images == 2
        assert r.stats.success is True and r.text == "{}"

    @patch("autovideofixer.core.analysis.open_url")
    def test_llamacpp_stats_and_provider_name(self, m):
        m.return_value = _mock_resp(LLAMA_BODY)
        r = an._call_custom_api(
            "", "http://127.0.0.1:1/v1/chat", "m", "s", "u", ["a"], provider="llamacpp"
        )
        assert r.stats.provider == "llamacpp" and r.stats.gen_tps == 100.0

    @pytest.mark.parametrize(
        "side_effect,cls",
        [
            (TimeoutError("timed out"), "timeout"),
            (urllib.error.URLError(ConnectionRefusedError(111, "refused")), "connection"),
            (urllib.error.HTTPError("u", 500, "Server Error", {}, None), "http_500"),
        ],
    )
    @patch("autovideofixer.core.analysis.open_url")
    def test_failures_become_failed_stats(self, m, side_effect, cls):
        m.side_effect = side_effect
        r = an._call_custom_api("", "http://127.0.0.1:1/v1/chat", "m", "s", "u", ["a"])
        assert r.text == "" and not r.stats.success and r.stats.error_class == cls
        assert r.stats.wall_sec >= 0

    @patch("autovideofixer.core.analysis.open_url")
    def test_malformed_json_body(self, m):
        m.return_value = _mock_resp(b"<html>502 Bad Gateway</html>")
        r = an._call_custom_api("", "http://127.0.0.1:1/v1/chat", "m", "s", "u", ["a"])
        assert r.stats.error_class == "bad_json" and r.text == ""

    @patch("autovideofixer.core.analysis.open_url")
    def test_unexpected_json_shape(self, m):
        m.return_value = _mock_resp({"choices": "nope"})
        r = an._call_custom_api("", "http://127.0.0.1:1/v1/chat", "m", "s", "u", ["a"])
        assert not r.stats.success and r.stats.error_class in ("bad_response", "empty_response")

    @patch("autovideofixer.core.analysis.open_url")
    def test_json_array_body(self, m):
        m.return_value = _mock_resp(b"[1, 2]")
        r = an._call_openai("k", "m", "s", "u", [])
        assert r.stats.error_class == "bad_response"

    @patch("autovideofixer.core.analysis.open_url")
    def test_empty_content_is_failed_call(self, m):
        m.return_value = _mock_resp(
            {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 3}}
        )
        r = an._call_openai("k", "m", "s", "u", [])
        assert r.stats.error_class == "empty_response" and r.stats.input_tokens == 3


class TestProviderRawRecordingAndLogging:
    KW = dict(
        model="m", api_url="http://127.0.0.1:1/v1/chat", api_key="", system_prompt="s",
        user_prompt="u", image_b64_list=["a", "b"],
    )  # fmt: skip

    @patch("autovideofixer.core.analysis.open_url")
    def test_success_is_stamped_recorded_and_logged(self, m, caplog):
        m.return_value = _mock_resp(LLAMA_BODY)
        with caplog.at_level(logging.DEBUG, logger="autovideofixer"), collect_call_stats() as calls:
            r = an._call_provider_raw(
                "llamacpp", endpoint_name="ep", purpose="video", est_vision_tokens=99, **self.KW
            )
        (c,) = calls
        assert c is r.stats
        assert (c.endpoint, c.purpose, c.est_vision_tokens, c.images) == ("ep", "video", 99, 2)
        info = [x for x in caplog.records if x.levelno == logging.INFO]
        assert any("VLM call video endpoint=ep" in x.getMessage() for x in info)
        dbg = [
            x
            for x in caplog.records
            if x.levelno == logging.DEBUG and "raw usage" in x.getMessage()
        ]
        assert dbg and "5100" in dbg[0].getMessage()

    @patch("autovideofixer.core.analysis.open_url")
    def test_timeout_one_line_warning_traceback_only_at_debug(self, m, caplog):
        m.side_effect = TimeoutError("timed out")
        with caplog.at_level(logging.DEBUG, logger="autovideofixer"), collect_call_stats() as calls:
            r = an._call_provider_raw(
                "llamacpp", endpoint_name="slow", purpose="video", timeout=2.0, **self.KW
            )
        assert calls[0].error_class == "timeout" and not r.stats.success
        warnings = [x for x in caplog.records if x.levelno >= logging.WARNING]
        assert len(warnings) == 1
        w = warnings[0]
        assert w.exc_info is None and "'slow'" in w.getMessage() and "timeout_sec" in w.getMessage()
        assert "2s" in w.getMessage() and "\n" not in w.getMessage()
        assert any(x.levelno == logging.DEBUG and x.exc_info for x in caplog.records)

    def test_unknown_provider_records_failed_stats(self):
        with collect_call_stats() as calls:
            assert (
                an._call_provider_raw("nope", endpoint_name="e", purpose="crop_check", **self.KW)
                is None
            )
        assert calls[0].error_class == "unknown_provider" and calls[0].purpose == "crop_check"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = self.server.mode  # type: ignore[attr-defined]
        if mode == "slow":
            time.sleep(1.5)
        if mode == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        out = b"<html>not json" if mode == "badjson" else json.dumps(LLAMA_BODY).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.mode = "ok"  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


class TestRealSocketFailuresAreOneLiners:
    """Each network failure: _dispatch_vlm survives, stats carry the class, and the console-level
    (WARNING+) records never carry a traceback."""

    @pytest.fixture
    def frame(self, tmp_path):
        p = tmp_path / "f.jpg"
        p.write_bytes(b"\xff\xd8fake")
        return str(p)

    def _run(self, srv, frame, caplog, timeout=5.0, port=None):
        url = f"http://127.0.0.1:{port or srv.server_address[1]}/v1/chat/completions"
        with caplog.at_level(logging.DEBUG, logger="autovideofixer"), collect_call_stats() as calls:
            out = an._dispatch_vlm(
                [frame], "llamacpp", "m", url, "", "S", "U", timeout=timeout,
                endpoint_name="ep", purpose="video",
            )  # fmt: skip
        assert out["summary"] == "" and len(calls) == 1
        loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert loud and all(r.exc_info is None for r in loud)
        assert all("Traceback" not in r.getMessage() for r in loud)
        return calls[0], loud

    def test_timeout(self, stub, frame, caplog):
        stub.mode = "slow"
        c, loud = self._run(stub, frame, caplog, timeout=0.3)
        assert c.error_class == "timeout" and c.wall_sec < 1.2
        assert "timeout_sec" in loud[0].getMessage() and "'ep'" in loud[0].getMessage()

    def test_connection_refused(self, frame, caplog):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        c, loud = self._run(None, frame, caplog, port=port)
        assert c.error_class == "connection" and "could not connect" in loud[0].getMessage()

    def test_http_500(self, stub, frame, caplog):
        stub.mode = "500"
        c, loud = self._run(stub, frame, caplog)
        assert c.error_class == "http_500" and "HTTP 500" in loud[0].getMessage()

    def test_malformed_json(self, stub, frame, caplog):
        stub.mode = "badjson"
        c, loud = self._run(stub, frame, caplog)
        assert c.error_class == "bad_json" and "not valid JSON" in loud[0].getMessage()

    def test_success_roundtrip_has_numbers(self, stub, frame):
        with collect_call_stats() as calls:
            an._dispatch_vlm(
                [frame], "llamacpp", "m", f"http://127.0.0.1:{stub.server_address[1]}/x", "",
                "S", "U", endpoint_name="ep",
            )  # fmt: skip
        c = calls[0]
        assert c.success and c.input_tokens == 2048 and c.gen_tps == 100.0


class TestCoordinatorAndCropRecord:
    def test_coordinator_failure_records_stats_and_fails_open(self, tmp_path):
        cfg = Config(tmp_path / "n.yaml")
        cfg.set("llamacpp", "analysis", "endpoints", "default_llm", "provider")
        cfg.set("http://127.0.0.1:9/x", "analysis", "endpoints", "default_llm", "api_url")
        with collect_call_stats() as calls:
            res = an.run_scene_coordinator(
                [{"index": 0, "start_time": 0, "end_time": 1, "duration": 1}], cfg
            )
        assert res["failed"] and res["drop"] == []
        assert calls[0].purpose == "coordinator" and calls[0].endpoint == "default_llm"
        assert not calls[0].success

    def test_crop_check_records_purpose(self, tmp_path):
        cfg = Config(tmp_path / "n.yaml")
        cfg.set("llamacpp", "analysis", "endpoints", "default_vlm", "provider")
        cfg.set("http://127.0.0.1:9/x", "analysis", "endpoints", "default_vlm", "api_url")
        f = tmp_path / "a.jpg"
        f.write_bytes(b"x")
        with collect_call_stats() as calls:
            res = an.run_crop_vlm_check(str(f), str(f), cfg)
        assert res["failed"] and calls[0].purpose == "crop_check"

    @patch("autovideofixer.core.analysis.open_url")
    def test_coordinator_success_has_usage(self, m, tmp_path):
        reply = {
            "choices": [
                {"message": {"content": '{"main_content_summary":"m","drop":[],"reasons":{}}'}}
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58},
        }
        m.return_value = _mock_resp(reply)
        cfg = Config(tmp_path / "n.yaml")
        cfg.set("openai", "analysis", "endpoints", "default_llm", "provider")
        with collect_call_stats() as calls:
            res = an.run_scene_coordinator(
                [{"index": 0, "start_time": 0, "end_time": 1, "duration": 1}], cfg
            )
        assert not res["failed"]
        assert (
            calls[0].purpose == "coordinator"
            and calls[0].input_tokens == 50
            and calls[0].images == 0
        )


class TestPipelineJobReport:
    def test_job_json_carries_calls(self):
        from autovideofixer.core.pipeline import JobResult
        from autovideofixer.core.reporting import job_to_json

        jr = JobResult(input_path="/x.mp4", success=True)
        jr.vlm_calls = [CallStats(purpose="scene", endpoint="e").to_dict()]
        out = job_to_json(jr)
        assert out["vlm_calls"][0]["purpose"] == "scene"
        assert job_to_json(JobResult(input_path="/y.mp4"))["vlm_calls"] == []

    def test_execute_job_collects_calls_on_every_path(self, tmp_path):
        from autovideofixer.core.pipeline import Job, JobResult, Pipeline

        p = Pipeline(Config(tmp_path / "n.yaml"))

        def fake(self, job, progress_callback=None):
            record_call(CallStats(purpose="crop_check"))
            return JobResult(input_path=job.input_path)

        with patch.object(Pipeline, "_execute_job", fake):
            res = p.execute_job(Job(input_path="/x.mp4"))
        assert [c["purpose"] for c in res.vlm_calls] == ["crop_check"]


class TestAnalyzeReporting:
    def _analysis(self):
        calls = [
            CallStats(
                endpoint="e", provider="llamacpp", model="m", purpose="video", wall_sec=2.0,
                input_tokens=2048, output_tokens=120, total_tokens=2168, prompt_tps=5100.0,
                gen_tps=100.0, images=4, est_vision_tokens=64, usage_source="usage+timings",
            ).to_dict()
        ]  # fmt: skip
        return an.VideoAnalysis(
            filepath="/x/v.mp4", filename="v.mp4", duration=30.0, resolution=(1920, 1080),
            framerate=30.0, vlm_summary="s", media_type="3d_rendered", content_rating="G",
            rating_reasoning="no violence", vlm_extra={"style": "cel [shaded]"},
            vlm_calls=calls, vlm_sampling={"frames": 4},
        )  # fmt: skip

    def test_csv_row(self):
        row = cli_mod._analysis_to_csv_row(self._analysis())
        assert row["media_type"] == "3d_rendered" and row["rating_reasoning"] == "no violence"
        assert json.loads(row["vlm_extra"]) == {"style": "cel [shaded]"}
        assert (row["vlm_calls"], row["vlm_input_tokens"], row["vlm_output_tokens"]) == (
            1,
            2048,
            120,
        )
        assert set(row) == set(cli_mod._ANALYZE_CSV_FIELDS)
        blank = cli_mod._analysis_to_csv_row(an.VideoAnalysis(filepath="/a", filename="a"))
        assert blank["media_type"] == "" and blank["vlm_calls"] == "" and blank["vlm_extra"] == ""

    def test_csv_file_roundtrip(self, tmp_path):
        path = tmp_path / "o.csv"
        cli_mod._write_analysis_csv(str(path), [cli_mod._analysis_to_csv_row(self._analysis())])
        row = next(csv.DictReader(path.open(encoding="utf-8")))
        assert row["media_type"] == "3d_rendered" and row["vlm_output_tokens"] == "120"

    def test_json_document(self, tmp_path):
        path = tmp_path / "o.json"
        cli_mod._write_analysis_json(str(path), [self._analysis()], ["/bad.mp4"])
        doc = json.loads(path.read_text(encoding="utf-8"))
        f = doc["files"][0]
        assert doc["failed_files"] == ["/bad.mp4"]
        assert f["vlm"]["media_type"] == "3d_rendered" and f["vlm"]["extra"] == {
            "style": "cel [shaded]"
        }
        assert f["vlm"]["rating_reasoning"] == "no violence"
        assert f["vlm_calls"][0]["input_tokens"] == 2048 and f["vlm_calls"][0]["gen_tps"] == 100.0
        assert f["vlm_sampling"] == {"frames": 4}
        novlm = cli_mod._analysis_to_json(an.VideoAnalysis(filepath="/a", filename="a"))
        assert novlm["vlm"] is None and novlm["vlm_calls"] == []

    @pytest.mark.parametrize("width", [60, 80, 200])
    def test_table_and_full_output_at_any_width(self, capsys, width):
        cli_mod.console.width = width
        a = self._analysis()
        cli_mod._print_analysis_result(a, full_output=True)
        cli_mod._print_call_stats(a.vlm_calls, title="VLM Calls")
        cli_mod._print_call_totals(a.vlm_calls)
        out = capsys.readouterr().out
        assert "3d_rendered" in out and "no violence" in out
        assert "cel [shaded]" in out  # brackets not eaten as Rich markup
        assert "2048" in out and "VLM call totals" in out
        assert ("Prompt t/s" in out) == (width >= cli_mod._CALL_TABLE_MIN_WIDTH)

    def test_raw_media_type_shown(self, capsys):
        cli_mod.console.width = 200
        a = self._analysis()
        a.vlm_extra = {"media_type_raw": "claymation"}
        a.media_type = "other"
        cli_mod._print_analysis_result(a, full_output=False)
        assert "model said: claymation" in capsys.readouterr().out

    def test_failed_call_row(self, capsys):
        cli_mod.console.width = 200
        a = self._analysis()
        a.vlm_calls = [CallStats(endpoint="slow", success=False, error_class="timeout").to_dict()]
        cli_mod._print_analysis_result(a, full_output=False)
        cli_mod._print_call_stats(a.vlm_calls, title="VLM Calls")
        out = capsys.readouterr().out
        assert "FAILED (timeout)" in out and "FAILED: timeout" in out

    def test_no_calls_prints_nothing(self, capsys):
        cli_mod._print_call_stats([], title="x")
        cli_mod._print_call_totals([])
        assert capsys.readouterr().out == ""


class TestAnalyzeCommand:
    def _run(self, tmp_path, extra_cfg="", args=()):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("config_version: 2\n" + extra_cfg)
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        analysis = an.VideoAnalysis(
            filepath=str(vid), filename="clip.mp4", duration=1.0, resolution=(2, 2), framerate=1.0,
            vlm_summary="s", media_type="real_footage",
            vlm_calls=[CallStats(endpoint="e", model="m", purpose="video", wall_sec=1.0).to_dict()],
        )  # fmt: skip
        with patch.object(an.VideoAnalyzer, "analyze", return_value=analysis):
            return CliRunner().invoke(
                cli_mod.main, ["--config", str(cfg), "analyze", str(vid), "--no-events", *args]
            )

    def test_json_flag_writes_file(self, tmp_path):
        out = tmp_path / "sub" / "r.json"
        res = self._run(tmp_path, args=("--json", str(out)))
        assert res.exit_code == 0, res.output
        assert json.loads(out.read_text())["files"][0]["vlm"]["media_type"] == "real_footage"

    def test_report_call_stats_default_shows_totals(self, tmp_path):
        assert "VLM call totals" in self._run(tmp_path).output

    def test_report_call_stats_false_hides_console_table_only(self, tmp_path):
        out = tmp_path / "r.json"
        res = self._run(tmp_path, "analysis:\n  report_call_stats: false\n", ("--json", str(out)))
        assert "VLM call totals" not in res.output and "VLM Calls" not in res.output
        assert json.loads(out.read_text())["files"][0]["vlm_calls"]  # still in the JSON

    def test_bad_report_call_stats_is_usage_error(self, tmp_path):
        res = self._run(tmp_path, "analysis:\n  report_call_stats: maybe\n")
        assert res.exit_code == 2 and "report_call_stats" in res.output

    def test_bad_detail_is_usage_error(self, tmp_path):
        res = self._run(tmp_path, "analysis:\n  vlm:\n    enabled: true\n    detail: wordy\n")
        assert res.exit_code == 2 and "analysis.vlm.detail" in res.output


class TestSceneSummariesCarryNewFields:
    def test_collect_scene_vlm_summaries(self, tmp_path):
        from autovideofixer.core import scenes as scenes_mod

        result = {
            "summary": "s",
            "tags": ["t"],
            "objects": ["o"],
            "media_type": "hand_drawn",
            "rating": "PG",
            "rating_reasoning": "cartoon peril",
            "extra": {"k": 1},
            "call": {"purpose": "scene"},
            "sampling": {"frames": 2},
        }
        with patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", return_value=result):
            out = scenes_mod._collect_scene_vlm_summaries(
                "v.mp4", [an.SceneEvent(0.0, 4.0)], Config(tmp_path / "n.yaml")
            )
        (sm,) = out
        assert sm["media_type"] == "hand_drawn" and sm["rating_reasoning"] == "cartoon peril"
        assert sm["extra"] == {"k": 1} and sm["call"] == {"purpose": "scene"}
        assert sm["objects"] == ["o"] and sm["rating"] == "PG"


class TestTruncatedReplies:
    """A reply cut off at max_tokens (finish_reason/done_reason "length") still succeeds, but
    is flagged, warned about with a max_tokens hint, and keeps the fields that arrived whole."""

    FULL = (
        '{"summary": "A drone shot over hills.", "tags": ["aerial", "nature"], '
        '"objects": ["hill"], "media_type": "real_footage", "rating": "G"}'
    )

    def test_openai_finish_reason_length_flags_truncation(self):
        s = CallStats()
        apply_openai_usage(s, {"choices": [{"finish_reason": "length"}]}, 1.0)
        assert (s.finish_reason, s.truncated) == ("length", True)
        s = CallStats()
        apply_openai_usage(s, {"choices": [{"finish_reason": "stop"}]}, 1.0)
        assert (s.finish_reason, s.truncated) == ("stop", False)

    def test_ollama_done_reason_length_flags_truncation(self):
        s = CallStats()
        apply_ollama_usage(s, {**OLLAMA_BODY, "done_reason": "length"}, 1.0)
        assert s.truncated

    def test_call_line_and_dict_show_truncation(self):
        s = CallStats(endpoint="ep", purpose="video", truncated=True, finish_reason="length")
        assert "TRUNCATED" in format_call_line(s)
        assert s.to_dict()["truncated"] is True

    @patch("autovideofixer.core.analysis.open_url")
    def test_truncated_reply_warns_with_max_tokens_hint(self, m, caplog):
        body = {**LLAMA_BODY, "choices": [{"message": {"content": "{"}, "finish_reason": "length"}]}
        m.return_value = _mock_resp(body)
        with caplog.at_level(logging.INFO, logger="autovideofixer"), collect_call_stats() as calls:
            r = an._call_provider_raw(
                "llamacpp", endpoint_name="ep", purpose="video", max_tokens=256,
                **TestProviderRawRecordingAndLogging.KW,
            )  # fmt: skip
        assert r.stats.success and calls[0].truncated
        (w,) = [x for x in caplog.records if x.levelno == logging.WARNING]
        msg = w.getMessage()
        assert "analysis.endpoints.ep.max_tokens" in msg and "256" in msg and "\n" not in msg

    @pytest.mark.parametrize("cut", [70, 95, 120, len(FULL) - 1])
    def test_salvage_keeps_complete_fields_only(self, cut):
        text = self.FULL[:cut]
        out = an._parse_vlm_response(text)
        assert out["summary"] == "A drone shot over hills."
        assert not out["summary"].startswith("{")
        # a list or string cut mid-way is dropped, never half-kept
        assert out["tags"] in ([], ["aerial", "nature"])
        assert out["rating"] in (None, "G")

    def test_salvage_through_think_and_fence(self):
        text = "<think>hmm</think>\n```json\n" + self.FULL[:95]
        assert an._parse_vlm_response(text)["summary"] == "A drone shot over hills."

    def test_plain_text_still_becomes_the_summary(self):
        assert an._parse_vlm_response("Just a sentence.")["summary"] == "Just a sentence."

    def test_fragment_with_nothing_complete_falls_back_to_text(self):
        assert an._parse_vlm_response('{"summ')["summary"] == '{"summ'
