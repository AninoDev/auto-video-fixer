"""Real-TLS accept/reject matrix for autovideofixer.net (REQUIREMENTS.md § 23).

Nothing in the TLS stack is mocked: genuine certificates (private CA, CA-signed, self-signed,
expired, not-yet-valid, wrong-hostname) are served by threaded `ssl`-wrapped http.servers on
127.0.0.1 and exercised through the real `open_url`, the real model-download path, the real
`requests`/`curl` fallbacks and a real VLM provider call.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import urllib.error

import pytest

from autovideofixer import net
from autovideofixer.ai import model_cache
from autovideofixer.core import analysis
from autovideofixer.net import (
    FallbackUnsupported,
    TLSPolicy,
    curl_tls_args,
    open_url,
    requests_verify,
)
from tests.unit import _tls_certs as tc

HAVE_CURL = shutil.which("curl") is not None


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    return tc.generate_all(tmp_path_factory.mktemp("tlscerts"))


@pytest.fixture(scope="module")
def ports(certs):
    """One live TLS server per certificate, keyed by cert name."""
    import contextlib

    with contextlib.ExitStack() as stack:
        yield {name: stack.enter_context(tc.serve(c)) for name, c in certs.items()}


@pytest.fixture(autouse=True)
def _fresh_warnings():
    net._reset_warnings_for_tests()
    yield
    net._reset_warnings_for_tests()


def url(port: int, path: str = "/file", host: str = "127.0.0.1") -> str:
    return f"https://{host}:{port}{path}"


def fetch(policy: TLSPolicy, port: int, host: str = "127.0.0.1") -> bytes:
    with open_url(url(port, host=host), timeout=10, tls=policy) as resp:
        return resp.read()


def P(mode, certs=None, *names):
    return TLSPolicy(mode, tuple(str(certs[n].cert_path) for n in names))


# (policy-builder, {server cert name: accepted?}) -- every server cert appears in every row.
ALL = [
    "ca_signed",
    "sloppy_signed",
    "self_signed",
    "expired",
    "expired_ca_signed",
    "not_yet_valid",
    "wrong_host",
    "wrong_host_ca_signed",
]


def matrix(accepted: set[str]) -> dict[str, bool]:
    return {n: n in accepted for n in ALL}


class TestOpenUrlMatrix:
    def _check(self, policy, expected, ports):
        for name, ok in expected.items():
            if ok:
                assert fetch(policy, ports[name]) == tc.PAYLOAD, f"{name} should be accepted"
            else:
                with pytest.raises((urllib.error.URLError, OSError)):
                    fetch(policy, ports[name])

    def test_verify_default_rejects_all_private_certs(self, ports):
        self._check(TLSPolicy("verify"), matrix(set()), ports)

    def test_verify_with_private_ca(self, certs, ports):
        # Only certs chained to the CA, valid, and with a matching name.
        self._check(P("verify", certs, "ca"), matrix({"ca_signed"}), ports)

    def test_verify_with_ca_lacking_key_identifiers(self, certs, ports):
        """Corporate inspection CAs often fail Python 3.13+'s VERIFY_X509_STRICT; we clear it."""
        self._check(P("verify", certs, "ca_no_ids"), matrix({"sloppy_signed"}), ports)

    @pytest.mark.parametrize("leaf", ["self_signed", "expired", "not_yet_valid", "wrong_host"])
    def test_verify_with_self_signed_leaf_as_anchor(self, certs, ports, leaf):
        # Trusting a specific server cert accepts exactly that server, and only while it is
        # valid and named correctly: trust never overrides expiry or hostname.
        accepted = {"self_signed"} if leaf == "self_signed" else set()
        self._check(P("verify", certs, leaf), matrix(accepted), ports)

    def test_verify_ca_files_are_additive_not_replacing(self, certs, ports):
        # Adding a CA must not break normal verification of public hosts; asserted structurally:
        # the context still contains the system roots (cert_store_stats counts both).
        base = net.build_ssl_context(TLSPolicy("verify")).cert_store_stats()["x509_ca"]
        extra = net.build_ssl_context(P("verify", certs, "ca")).cert_store_stats()["x509_ca"]
        assert extra == base + 1

    def test_untrusted_ca_accepts_unknown_chains_but_not_bad_names_or_dates(self, ports):
        self._check(
            TLSPolicy("untrusted_ca"),
            matrix({"ca_signed", "sloppy_signed", "self_signed"}),
            ports,
        )

    def test_untrusted_ca_error_names_the_reason(self, ports):
        with pytest.raises(urllib.error.URLError) as e:
            fetch(TLSPolicy("untrusted_ca"), ports["expired"])
        assert "expired" in str(e.value)
        with pytest.raises(urllib.error.URLError) as e:
            fetch(TLSPolicy("untrusted_ca"), ports["not_yet_valid"])
        assert "not yet valid" in str(e.value)
        with pytest.raises(urllib.error.URLError) as e:
            fetch(TLSPolicy("untrusted_ca"), ports["wrong_host"])
        assert "does not match" in str(e.value)

    def test_untrusted_ca_hostname_checked_against_requested_name(self, ports):
        # self_signed has SAN localhost + 127.0.0.1 so both spellings pass ...
        assert fetch(TLSPolicy("untrusted_ca"), ports["self_signed"], host="localhost")
        # ... wrong_host names only other.example.com, so neither does.
        with pytest.raises(urllib.error.URLError):
            fetch(TLSPolicy("untrusted_ca"), ports["wrong_host"], host="localhost")

    def test_insecure_accepts_everything_still_over_tls(self, ports):
        self._check(TLSPolicy("insecure"), matrix(set(ALL)), ports)

    def test_untrusted_ca_checks_every_redirect_hop(self, certs):
        """A redirect from an acceptable server to an expired one must be refused."""
        import contextlib

        with contextlib.ExitStack() as stack:
            bad = stack.enter_context(tc.serve(certs["expired"]))
            good = stack.enter_context(
                tc.serve(certs["self_signed"], redirect_to=url(bad, "/file"))
            )
            with pytest.raises(urllib.error.URLError, match="expired"):
                with open_url(url(good, "/redirect"), timeout=10, tls=TLSPolicy("untrusted_ca")):
                    pass
            # control: redirect to another acceptable server succeeds
            ok = stack.enter_context(tc.serve(certs["ca_signed"]))
            hop = stack.enter_context(tc.serve(certs["self_signed"], redirect_to=url(ok, "/file")))
            with open_url(url(hop, "/redirect"), timeout=10, tls=TLSPolicy("untrusted_ca")) as r:
                assert r.read() == tc.PAYLOAD

    def test_none_policy_is_secure_default(self, ports):
        with pytest.raises(urllib.error.URLError):
            with open_url(url(ports["self_signed"]), timeout=10):
                pass

    def test_weak_modes_warn_once_through_open_url(self, ports, caplog):
        with caplog.at_level(logging.WARNING, logger="autovideofixer.net"):
            for _ in range(3):
                fetch(TLSPolicy("insecure", (), "download"), ports["expired"])
        assert len([r for r in caplog.records if "insecure" in r.getMessage()]) == 1


# ─── requests / curl fallbacks, individually, against the same servers ─────────────────


def _requests_ok(policy, port) -> bool:
    import requests

    try:
        with requests_verify(policy) as verify:
            return requests.get(url(port), timeout=10, verify=verify).content == tc.PAYLOAD
    except requests.exceptions.SSLError:
        return False


def _curl_ok(policy, port) -> bool:
    with curl_tls_args(policy) as args:
        r = subprocess.run(
            ["curl", "-fsS", "--max-time", "10", *args, url(port)],
            capture_output=True,
            timeout=30,
        )
    return r.returncode == 0 and r.stdout == tc.PAYLOAD


FALLBACK_ROWS = [
    ("verify-default", lambda c: TLSPolicy("verify"), set()),
    ("verify-ca", lambda c: P("verify", c, "ca"), {"ca_signed"}),
    ("verify-leaf", lambda c: P("verify", c, "self_signed"), {"self_signed"}),
    ("verify-two", lambda c: P("verify", c, "ca", "self_signed"), {"ca_signed", "self_signed"}),
    ("insecure", lambda c: TLSPolicy("insecure"), set(ALL)),
]


@pytest.mark.filterwarnings("ignore:Unverified HTTPS request")
class TestFallbacksNeverWeaker:
    @pytest.mark.parametrize(("label", "build", "accepted"), FALLBACK_ROWS)
    def test_requests(self, certs, ports, label, build, accepted):
        pytest.importorskip("requests")
        for name in ALL:
            assert _requests_ok(build(certs), ports[name]) is (name in accepted), (label, name)

    @pytest.mark.skipif(not HAVE_CURL, reason="curl not installed")
    @pytest.mark.parametrize(("label", "build", "accepted"), FALLBACK_ROWS)
    def test_curl(self, certs, ports, label, build, accepted):
        for name in ALL:
            assert _curl_ok(build(certs), ports[name]) is (name in accepted), (label, name)

    def test_untrusted_ca_skips_both(self):
        for ctx in (requests_verify, curl_tls_args):
            with pytest.raises(FallbackUnsupported):
                with ctx(TLSPolicy("untrusted_ca")):
                    pass


# ─── Real model download path ─────────────────────────────────────────────────────────


@pytest.fixture
def no_urllib(monkeypatch):
    """Force the urllib attempt to fail so the requests/curl fallbacks are what's tested.

    Only the urllib *entry point* is replaced; the fallbacks and the servers are real.
    """

    def boom(*a, **k):
        raise OSError("urllib disabled for fallback test")

    monkeypatch.setattr(model_cache, "open_url", boom)


SHA = hashlib.sha256(tc.PAYLOAD).hexdigest()


@pytest.mark.filterwarnings("ignore:Unverified HTTPS request")
class TestModelDownloadPath:
    def _dl(self, tmp_path, policy, port, sha=SHA):
        dest = tmp_path / "model.bin"
        model_cache._download_file(url(port), str(dest), expected_sha256=sha, tls=policy)
        return dest

    def test_default_policy_rejects_self_signed(self, tmp_path, ports):
        with pytest.raises(RuntimeError, match="All download methods failed"):
            self._dl(tmp_path, None, ports["self_signed"])
        assert not (tmp_path / "model.bin").exists()

    @pytest.mark.parametrize(
        ("name", "mode", "cas"),
        [
            ("ca_signed", "verify", ("ca",)),
            ("self_signed", "verify", ("self_signed",)),
            ("self_signed", "untrusted_ca", ()),
            ("ca_signed", "untrusted_ca", ()),
            ("expired", "insecure", ()),
            ("wrong_host", "insecure", ()),
        ],
    )
    def test_accepted(self, tmp_path, certs, ports, name, mode, cas):
        dest = self._dl(tmp_path, P(mode, certs, *cas), ports[name])
        assert dest.read_bytes() == tc.PAYLOAD
        assert not (tmp_path / "model.bin.part").exists()

    @pytest.mark.parametrize(
        ("name", "mode", "cas"),
        [
            ("self_signed", "verify", ()),
            ("expired", "verify", ("expired",)),
            ("expired_ca_signed", "verify", ("ca",)),
            ("expired", "untrusted_ca", ()),
            ("wrong_host", "untrusted_ca", ()),
            ("not_yet_valid", "untrusted_ca", ()),
        ],
    )
    def test_rejected_by_every_method(self, tmp_path, certs, ports, name, mode, cas):
        """Even with all three fallbacks in play none may succeed where the policy rejects."""
        with pytest.raises(RuntimeError, match="All download methods failed"):
            self._dl(tmp_path, P(mode, certs, *cas), ports[name])
        assert not (tmp_path / "model.bin").exists()

    def test_untrusted_ca_failure_reports_fallbacks_skipped(self, tmp_path, certs, ports, caplog):
        caplog.set_level(logging.DEBUG, logger="autovideofixer.ai.model_cache")
        with pytest.raises(RuntimeError) as e:
            self._dl(tmp_path, P("untrusted_ca", certs), ports["expired"])
        msg = str(e.value)
        assert "requests: skipped" in msg and "curl: skipped" in msg
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Skipping requests fallback" in m for m in debug)
        assert any("Skipping curl fallback" in m for m in debug)

    def test_hash_still_enforced_in_insecure_mode(self, tmp_path, ports):
        with pytest.raises(RuntimeError, match="hash mismatch"):
            self._dl(tmp_path, TLSPolicy("insecure"), ports["self_signed"], sha="0" * 64)
        assert not (tmp_path / "model.bin").exists()

    def test_untrusted_ca_never_downgrades_to_fallback_when_urllib_fails(
        self, tmp_path, certs, ports, no_urllib
    ):
        # A perfectly acceptable server, but urllib "fails": requests/curl must NOT be run
        # in untrusted_ca mode (they could not do the hostname/expiry-only check).
        with pytest.raises(RuntimeError, match="skipped"):
            self._dl(tmp_path, P("untrusted_ca", certs), ports["self_signed"])

    @pytest.mark.parametrize(
        ("name", "mode", "cas", "ok"),
        [
            ("ca_signed", "verify", ("ca",), True),
            ("self_signed", "verify", ("self_signed",), True),
            ("self_signed", "verify", (), False),
            ("expired_ca_signed", "verify", ("ca",), False),
            ("wrong_host_ca_signed", "verify", ("ca",), False),
            ("expired", "insecure", (), True),
        ],
    )
    def test_fallback_chain_honours_policy(
        self, tmp_path, certs, ports, no_urllib, name, mode, cas, ok
    ):
        policy = P(mode, certs, *cas)
        if ok:
            assert self._dl(tmp_path, policy, ports[name]).read_bytes() == tc.PAYLOAD
        else:
            with pytest.raises(RuntimeError, match="All download methods failed"):
                self._dl(tmp_path, policy, ports[name])

    def test_download_model_uses_download_tls_over_tls(self, tmp_path, certs, ports, monkeypatch):
        """Public entry point: config -> download_tls override -> policy -> real download."""
        from autovideofixer.config import Config

        monkeypatch.setattr(model_cache, "get_model_dir", lambda: tmp_path)
        cfg = Config()
        # network.tls stays strict; only downloads are relaxed.
        cfg.apply_layer({"network": {"download_tls": {"mode": "insecure"}}}, "test")
        ok, msg = model_cache.download_model(
            "custom", url=url(ports["expired"]), custom_path="custom.pth", config=cfg
        )
        assert ok, msg
        assert (tmp_path / "custom.pth").read_bytes() == tc.PAYLOAD
        # and with no config (secure default) the same call is refused
        (tmp_path / "custom.pth").unlink()
        ok, msg = model_cache.download_model(
            "custom", url=url(ports["expired"]), custom_path="custom.pth"
        )
        assert not ok and "Download failed" in msg

    def test_download_model_bad_policy_is_clean_failure(self, tmp_path, ports, monkeypatch):
        from autovideofixer.config import Config

        monkeypatch.setattr(model_cache, "get_model_dir", lambda: tmp_path)
        cfg = Config()
        cfg.apply_layer({"network": {"tls": {"mode": "bogus"}}}, "test")
        ok, msg = model_cache.download_model(
            "custom", url=url(ports["self_signed"]), custom_path="c.pth", config=cfg
        )
        assert not ok and "bogus" in msg


# ─── VLM provider call ────────────────────────────────────────────────────────────────


class TestProviderCall:
    def _call(self, port, policy):
        return analysis._call_custom_api(
            "key",
            url(port, "/v1/chat/completions"),
            "m",
            "sys",
            "user",
            ["aGVsbG8="],
            tls=policy,
        ).text

    def test_default_fails_closed(self, ports):
        assert self._call(ports["self_signed"], None) == ""

    def test_verify_with_leaf_ca_file(self, certs, ports):
        assert "tls-ok" in self._call(ports["self_signed"], P("verify", certs, "self_signed"))

    def test_untrusted_ca(self, ports):
        assert "tls-ok" in self._call(ports["self_signed"], TLSPolicy("untrusted_ca"))
        assert self._call(ports["expired"], TLSPolicy("untrusted_ca")) == ""
        assert self._call(ports["wrong_host"], TLSPolicy("untrusted_ca")) == ""

    def test_insecure(self, ports):
        assert "tls-ok" in self._call(ports["expired"], TLSPolicy("insecure"))

    def test_run_api_vlm_end_to_end_parse(self, certs, ports, tmp_path):
        import cv2
        import numpy as np

        frame = tmp_path / "f.jpg"
        cv2.imwrite(str(frame), np.zeros((16, 16, 3), dtype=np.uint8))
        out = analysis._run_api_vlm(
            [str(frame)],
            "k",
            url(ports["self_signed"], "/v1/chat/completions"),
            "m",
            tls=TLSPolicy("untrusted_ca"),
        )
        assert out["summary"] == "tls-ok"

    def test_coordinator_and_crop_resolve_policy_from_config(self, ports, tmp_path):
        """The config-driven callers pick up network.tls (analysis.vlm/llm provider=api)."""
        from autovideofixer.config import Config

        cfg = Config()
        endpoint = url(ports["self_signed"], "/v1/chat/completions")
        for section in ("vlm", "llm"):
            cfg.apply_layer(
                {"analysis": {section: {"provider": "api", "api_url": endpoint}}}, "test"
            )
        scenes = [
            {"index": 0, "start_time": 0, "end_time": 5, "duration": 5, "summary": "a", "tags": []}
        ]
        strict = analysis.run_scene_coordinator(scenes, cfg)
        assert strict["failed"] is True
        cfg.apply_layer({"network": {"tls": {"mode": "untrusted_ca"}}}, "test")
        relaxed = analysis.run_scene_coordinator(scenes, cfg)
        # The stub answers with valid JSON lacking a "drop" list: once the request clears TLS
        # the coordinator parses it as "nothing to drop" (failed False). A transport failure
        # would instead report failed True (the strict case above).
        assert relaxed["failed"] is False
        assert relaxed["drop"] == []

    def test_bad_tls_config_is_loud_for_vlm_and_fail_open_for_coordinator(self):
        from autovideofixer.config import Config

        cfg = Config()
        cfg.apply_layer({"network": {"tls": {"mode": "bogus"}}}, "test")
        with pytest.raises(ValueError, match="bogus"):
            analysis.VideoAnalyzer(cfg).run_vlm_analysis("/nonexistent.mp4")
        res = analysis.run_scene_coordinator(
            [
                {
                    "index": 0,
                    "start_time": 0,
                    "end_time": 1,
                    "duration": 1,
                    "summary": "",
                    "tags": [],
                }
            ],
            cfg,
        )
        assert res["failed"] is True and res["drop"] == []
