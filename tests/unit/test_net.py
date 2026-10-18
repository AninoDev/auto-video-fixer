"""Unit tests for autovideofixer.net: policy resolution, hostname matcher, fallback args."""

from __future__ import annotations

import ipaddress
import logging
import os
from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509

from autovideofixer import net
from autovideofixer.config import Config
from autovideofixer.net import (
    FallbackUnsupported,
    TLSPolicy,
    TLSValidationError,
    curl_tls_args,
    match_certificate_hostname,
    match_dns_name,
    requests_verify,
    resolve_download_policy,
    resolve_tls_policy,
    validate_peer_certificate,
)
from tests.unit import _tls_certs as tc


@pytest.fixture(autouse=True)
def _fresh_warnings():
    net._reset_warnings_for_tests()
    yield
    net._reset_warnings_for_tests()


@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    return tc.generate_all(tmp_path_factory.mktemp("certs"))


class TestPolicyResolution:
    def test_defaults_are_verify_with_no_extra_cas(self):
        p = resolve_tls_policy(Config())
        assert (p.mode, p.ca_files) == ("verify", ())

    def test_config_defaults_declare_network_block(self):
        assert Config.DEFAULTS["network"]["tls"] == {"mode": "verify", "ca_files": []}
        assert Config.DEFAULTS["network"]["download_tls"] is None

    def test_none_and_plain_dict_config(self):
        assert resolve_tls_policy(None).mode == "verify"
        assert resolve_tls_policy({"network": {"tls": {"mode": "insecure"}}}).mode == "insecure"

    def test_override_deep_merges_over_base(self, tmp_path):
        ca = tmp_path / "a.pem"
        ca.write_text("x")
        cfg = {"network": {"tls": {"mode": "verify", "ca_files": [str(ca)]}}}
        # An override that only sets mode keeps the base ca_files.
        p = resolve_tls_policy(cfg, {"mode": "untrusted_ca"})
        assert p.mode == "untrusted_ca" and p.ca_files == (str(ca),)

    def test_override_lists_replace_wholesale(self, tmp_path):
        a, b = tmp_path / "a.pem", tmp_path / "b.pem"
        a.write_text("x")
        b.write_text("x")
        cfg = {"network": {"tls": {"ca_files": [str(a)]}}}
        p = resolve_tls_policy(cfg, {"ca_files": [str(b)]})
        assert p.ca_files == (str(b),)

    def test_download_tls_overrides_for_downloads_only(self):
        cfg = {"network": {"tls": {"mode": "verify"}, "download_tls": {"mode": "insecure"}}}
        assert resolve_download_policy(cfg).mode == "insecure"
        assert resolve_tls_policy(cfg).mode == "verify"

    def test_download_tls_null_falls_back_to_tls(self):
        cfg = {"network": {"tls": {"mode": "untrusted_ca"}, "download_tls": None}}
        assert resolve_download_policy(cfg).mode == "untrusted_ca"

    def test_download_tls_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="download_tls"):
            resolve_download_policy({"network": {"download_tls": "insecure"}})

    def test_unknown_mode_is_clear_error(self):
        with pytest.raises(ValueError, match=r"mode 'bogus'.*verify, untrusted_ca, insecure"):
            resolve_tls_policy({"network": {"tls": {"mode": "bogus"}}})

    def test_unknown_mode_in_override(self):
        with pytest.raises(ValueError, match="mode"):
            resolve_tls_policy(Config(), {"mode": "nope"})

    def test_missing_ca_file_error_in_verify_mode(self, tmp_path):
        cfg = {"network": {"tls": {"ca_files": [str(tmp_path / "missing.pem")]}}}
        with pytest.raises(ValueError, match="not a readable file"):
            resolve_tls_policy(cfg)

    def test_ca_files_must_be_list_of_strings(self):
        with pytest.raises(ValueError, match="list of file paths"):
            resolve_tls_policy({"network": {"tls": {"ca_files": "/a.pem"}}})

    def test_ca_files_not_checked_when_unused(self, tmp_path):
        cfg = {"network": {"tls": {"mode": "insecure", "ca_files": [str(tmp_path / "gone.pem")]}}}
        assert resolve_tls_policy(cfg).mode == "insecure"

    def test_ca_file_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "ca.pem").write_text("x")
        p = resolve_tls_policy({"network": {"tls": {"ca_files": ["~/ca.pem"]}}})
        assert p.ca_files == (str(tmp_path / "ca.pem"),)

    def test_invalid_pem_is_clear_error_at_context_build(self, tmp_path):
        bad = tmp_path / "bad.pem"
        bad.write_text("not a pem")
        with pytest.raises(ValueError, match="cannot load"):
            net.build_ssl_context(TLSPolicy("verify", (str(bad),)))

    def test_set_cascade_shape_reaches_policy(self):
        """`--set network.tls.mode=insecure` builds exactly this nested layer."""
        cfg = Config()
        cfg.apply_layer({"network": {"tls": {"mode": "insecure"}}}, "cli-flags")
        assert resolve_tls_policy(cfg).mode == "insecure"
        cfg.apply_layer({"network": {"download_tls": {"mode": "untrusted_ca"}}}, "cli-flags")
        assert resolve_download_policy(cfg).mode == "untrusted_ca"

    def test_label_is_not_part_of_equality(self):
        assert TLSPolicy("verify", (), "a") == TLSPolicy("verify", (), "b")


class TestWeakModeWarning:
    def test_warns_once_per_label_and_mode(self, caplog):
        with caplog.at_level(logging.WARNING, logger="autovideofixer.net"):
            for _ in range(3):
                with pytest.raises(FallbackUnsupported):
                    with requests_verify(TLSPolicy("untrusted_ca", (), "vlm")):
                        pass
            with pytest.raises(FallbackUnsupported):
                with requests_verify(TLSPolicy("untrusted_ca", (), "llm")):
                    pass
            with requests_verify(TLSPolicy("insecure", (), "vlm")):
                pass
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(msgs) == 3
        assert any("'untrusted_ca'" in m and "vlm" in m for m in msgs)
        assert any("'insecure'" in m for m in msgs)

    def test_verify_never_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="autovideofixer.net"):
            with requests_verify(TLSPolicy("verify")):
                pass
        assert not caplog.records


class TestDnsMatcher:
    @pytest.mark.parametrize(
        ("pattern", "host", "expected"),
        [
            ("example.com", "example.com", True),
            ("Example.COM", "example.com", True),
            ("example.com", "EXAMPLE.com.", True),
            ("example.com", "www.example.com", False),
            ("*.example.com", "www.example.com", True),
            ("*.example.com", "WWW.Example.Com", True),
            ("*.example.com", "example.com", False),
            ("*.example.com", "a.b.example.com", False),
            ("*.example.com", ".example.com", False),
            ("*.com", "example.com", False),
            ("*", "localhost", False),
            ("w*.example.com", "www.example.com", False),
            ("*w.example.com", "www.example.com", False),
            ("www.*.com", "www.example.com", False),
            ("*.*.example.com", "a.b.example.com", False),
            ("*.example.com", "evil-example.com", False),
            ("", "example.com", False),
            # IDNA: U-label host vs A-label cert name (and the reverse).
            ("xn--bcher-kva.example", "bücher.example", True),
            ("bücher.example", "xn--bcher-kva.example", True),
            ("*.bücher.example", "www.bücher.example", True),
        ],
    )
    def test_cases(self, pattern, host, expected):
        assert match_dns_name(pattern, host) is expected


def _cert(cn="x", sans=None, **kw):
    return tc.make_cert(cn, sans=sans, **kw)[0]


class TestCertificateHostname:
    def test_san_dns_match_and_no_match(self):
        c = _cert(sans=[x509.DNSName("a.example.com"), x509.DNSName("*.b.example.com")])
        assert match_certificate_hostname(c, "a.example.com")
        assert match_certificate_hostname(c, "x.b.example.com")
        assert not match_certificate_hostname(c, "b.example.com")
        assert not match_certificate_hostname(c, "c.example.com")

    def test_san_ip_match_ipv4_and_ipv6(self):
        c = _cert(
            sans=[
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address("::1")),
            ]
        )
        assert match_certificate_hostname(c, "127.0.0.1")
        assert match_certificate_hostname(c, "::1")
        assert match_certificate_hostname(c, "[::1]")
        assert not match_certificate_hostname(c, "127.0.0.2")

    def test_ip_never_matches_dns_san_or_wildcard(self):
        c = _cert(sans=[x509.DNSName("127.0.0.1"), x509.DNSName("*.0.0.1")])
        assert not match_certificate_hostname(c, "127.0.0.1")

    def test_dns_host_does_not_match_ip_san(self):
        c = _cert(sans=[x509.IPAddress(ipaddress.ip_address("127.0.0.1"))])
        assert not match_certificate_hostname(c, "localhost")

    def test_cn_fallback_only_without_san_extension(self):
        legacy = _cert(cn="legacy.example.com", sans=None)
        assert match_certificate_hostname(legacy, "legacy.example.com")
        assert match_certificate_hostname(legacy, "LEGACY.example.com")
        # SAN present (even with an unrelated name) => CN is ignored.
        modern = _cert(cn="legacy.example.com", sans=[x509.DNSName("other.example.com")])
        assert not match_certificate_hostname(modern, "legacy.example.com")

    def test_cn_fallback_not_used_when_san_has_only_ip_entries(self):
        c = _cert(cn="name.example.com", sans=[x509.IPAddress(ipaddress.ip_address("10.0.0.1"))])
        assert not match_certificate_hostname(c, "name.example.com")

    def test_cn_wildcard_and_ip_literal_cn(self):
        assert match_certificate_hostname(_cert(cn="*.example.com", sans=None), "a.example.com")
        assert match_certificate_hostname(_cert(cn="10.1.2.3", sans=None), "10.1.2.3")
        assert not match_certificate_hostname(_cert(cn="*.example.com", sans=None), "10.1.2.3")


class TestValidatePeerCertificate:
    def _der(self, cert):
        from cryptography.hazmat.primitives import serialization

        return cert.public_bytes(serialization.Encoding.DER)

    def test_valid_cert_passes(self):
        validate_peer_certificate(self._der(_cert(sans=tc.LOCAL_SANS)), "127.0.0.1")

    def test_expired_rejected(self):
        now = datetime.now(UTC)
        c = _cert(
            sans=tc.LOCAL_SANS, not_before=now - timedelta(days=5), not_after=now - timedelta(1)
        )
        with pytest.raises(TLSValidationError, match="expired"):
            validate_peer_certificate(self._der(c), "127.0.0.1")

    def test_not_yet_valid_rejected(self):
        now = datetime.now(UTC)
        c = _cert(
            sans=tc.LOCAL_SANS, not_before=now + timedelta(days=1), not_after=now + timedelta(5)
        )
        with pytest.raises(TLSValidationError, match="not yet valid"):
            validate_peer_certificate(self._der(c), "127.0.0.1")

    def test_hostname_mismatch_rejected(self):
        with pytest.raises(TLSValidationError, match="does not match"):
            validate_peer_certificate(self._der(_cert(sans=tc.OTHER_SANS)), "127.0.0.1")

    def test_explicit_now_boundary(self):
        c = _cert(sans=tc.LOCAL_SANS)
        later = c.not_valid_after_utc + timedelta(seconds=1)
        with pytest.raises(TLSValidationError):
            validate_peer_certificate(self._der(c), "localhost", now=later)

    def test_garbage_rejected(self):
        with pytest.raises(TLSValidationError, match="unparseable"):
            validate_peer_certificate(b"garbage", "localhost")

    def test_is_an_ssl_cert_verification_error(self):
        import ssl

        assert issubclass(TLSValidationError, ssl.SSLCertVerificationError)


class TestFallbackArgs:
    def test_insecure(self):
        with requests_verify(TLSPolicy("insecure")) as v:
            assert v is False
        with curl_tls_args(TLSPolicy("insecure")) as a:
            assert a == ["-k"]

    def test_verify_without_cas(self):
        with requests_verify(TLSPolicy("verify")) as v:
            assert v is True
        with curl_tls_args(TLSPolicy("verify")) as a:
            assert a == []

    def test_untrusted_ca_is_skipped_for_both(self):
        with pytest.raises(FallbackUnsupported, match="untrusted_ca"):
            with requests_verify(TLSPolicy("untrusted_ca")):
                pass
        with pytest.raises(FallbackUnsupported, match="untrusted_ca"):
            with curl_tls_args(TLSPolicy("untrusted_ca")):
                pass

    def test_verify_with_ca_files_builds_union_bundle_and_cleans_up(self, certs):
        policy = TLSPolicy(
            "verify", (str(certs["ca"].cert_path), str(certs["self_signed"].cert_path))
        )
        with requests_verify(policy) as path:
            assert isinstance(path, str) and os.path.isfile(path)
            text = open(path).read()
            base = open(net._base_ca_bundle()).read()
            assert text.startswith(base)  # system/certifi bundle is kept, not replaced
            assert certs["ca"].cert_path.read_text() in text
            assert certs["self_signed"].cert_path.read_text() in text
            bundle_dir = os.path.dirname(path)
        assert not os.path.exists(bundle_dir)

    def test_curl_uses_same_union_bundle(self, certs):
        with curl_tls_args(TLSPolicy("verify", (str(certs["ca"].cert_path),))) as a:
            assert a[0] == "--cacert" and os.path.isfile(a[1])
            path = a[1]
        assert not os.path.exists(path)

    def test_no_default_bundle_means_skip_not_weaker(self, certs, monkeypatch):
        monkeypatch.setattr(net, "_base_ca_bundle", lambda: None)
        policy = TLSPolicy("verify", (str(certs["ca"].cert_path),))
        with pytest.raises(FallbackUnsupported):
            with requests_verify(policy):
                pass
        with pytest.raises(FallbackUnsupported):
            with curl_tls_args(policy):
                pass
