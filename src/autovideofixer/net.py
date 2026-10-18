"""Auto Video Fixer - central network/TLS policy (docs/REQUIREMENTS.md § 23).

Every outbound HTTP(S) request AVF makes (VLM/LLM providers in ``core/analysis.py``, model and
archive downloads in ``ai/model_cache.py``) goes through this module so a single, documented TLS
policy applies everywhere. Python's TLS settings are not central: ``urllib``, ``requests`` and the
``curl`` subprocess each need the policy expressed in their own terms, and a fallback that cannot
express it must be *skipped*, never run with weaker checking (see ``FallbackUnsupported``).

Modes (``network.tls.mode``):

* ``verify`` (default): normal verification against the system trust store **plus** every PEM in
  ``ca_files`` (added, never replacing). A private/corporate CA cert *or* a self-signed server
  (leaf) cert both work as an ``ca_files`` entry -- OpenSSL accepts either as a trust anchor.
* ``untrusted_ca``: accept a chain that does not verify (self-signed, unknown CA) but still reject
  an expired / not-yet-valid certificate and a hostname mismatch. Implemented as a ``CERT_NONE``
  handshake followed by our own inspection of the leaf certificate (``validate_peer_certificate``).
* ``insecure``: no verification at all (``CERT_NONE``); the connection is still encrypted.

Revocation (OCSP/CRL) is not checked in any mode.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import logging
import os
import shutil
import ssl
import tempfile
import threading
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

VALID_TLS_MODES = ("verify", "untrusted_ca", "insecure")

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        from autovideofixer.logger import get_logger

        _logger = get_logger("autovideofixer.net")
    return _logger


class TLSValidationError(ssl.SSLCertVerificationError):
    """Raised by ``untrusted_ca`` mode when the leaf certificate fails our own checks.

    Subclasses ``ssl.SSLCertVerificationError`` (an ``OSError``) so ``urllib`` wraps it in
    ``URLError`` exactly like a native verification failure, and callers that already handle TLS
    errors need no new except clause.
    """


class FallbackUnsupported(Exception):
    """A download fallback (``requests``/``curl``) cannot faithfully express the TLS policy.

    The caller must skip that fallback (DEBUG log naming ``str(exc)``) rather than run it with
    weaker checking than configured (REQUIREMENTS.md § 23.2).
    """


# ─── Policy ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TLSPolicy:
    """A resolved, validated TLS policy.

    ``ca_files`` is a tuple (frozen dataclass); ``resolve_tls_policy`` accepts a list. ``label``
    names the purpose/endpoint for the once-per-process weak-mode WARNING and is not part of
    policy equality.
    """

    mode: str = "verify"
    ca_files: tuple[str, ...] = ()
    label: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TLSPolicy):
            return NotImplemented
        return (self.mode, self.ca_files) == (other.mode, other.ca_files)

    def __hash__(self) -> int:
        return hash((self.mode, self.ca_files))

    @property
    def is_weak(self) -> bool:
        return self.mode != "verify"


DEFAULT_POLICY = TLSPolicy()


def _merge_tls_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto ``base`` (mappings recurse; lists/scalars replace wholesale,
    matching the config cascade's rule that list-valued keys are never merged element-wise)."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_tls_mappings(out[key], value)
        else:
            out[key] = value
    return out


def _validate_policy_mapping(raw: Mapping[str, Any], where: str) -> TLSPolicy:
    mode = raw.get("mode", "verify")
    if mode not in VALID_TLS_MODES:
        raise ValueError(
            f"Invalid {where}.mode {mode!r}: must be one of {', '.join(VALID_TLS_MODES)}"
        )
    ca_raw = raw.get("ca_files")
    if ca_raw is None:
        ca_raw = []
    if not isinstance(ca_raw, (list, tuple)) or not all(isinstance(p, str) for p in ca_raw):
        raise ValueError(f"Invalid {where}.ca_files {ca_raw!r}: must be a list of file paths")
    ca_files: list[str] = []
    for path in ca_raw:
        expanded = os.path.expanduser(path)
        # Only `verify` consumes ca_files; checking them in the other modes would make a typo in
        # an unused list block the very workaround (untrusted_ca/insecure) the user switched to
        # because verify was not working.
        if mode == "verify":
            if not os.path.isfile(expanded):
                raise ValueError(f"{where}.ca_files: {path!r} is not a readable file")
            try:
                with open(expanded, "rb"):
                    pass
            except OSError as e:
                raise ValueError(f"{where}.ca_files: cannot read {path!r}: {e}") from e
        ca_files.append(expanded)
    return TLSPolicy(mode=mode, ca_files=tuple(ca_files))


def _cfg_get(config: Any, *keys: str) -> Any:
    """Read ``keys`` from a ``Config`` or a plain nested mapping; None if absent."""
    if config is None:
        return None
    if hasattr(config, "get") and not isinstance(config, Mapping):
        return config.get(*keys, default=None)
    node: Any = config
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def resolve_tls_policy(
    config: Any,
    override: Mapping[str, Any] | None = None,
    *,
    purpose: str = "",
) -> TLSPolicy:
    """Resolve the effective TLS policy: ``network.tls`` with ``override`` deep-merged on top.

    ``override`` is a mapping of the same shape (``{mode:, ca_files:}``) -- P3's per-endpoint
    ``tls:`` block, or ``network.download_tls`` for model downloads (see
    ``resolve_download_policy``). ``config`` may be a ``Config``, a plain nested dict, or None
    (then the secure default). Raises ``ValueError`` at *use time* (when something is about to
    make a request) on an unknown mode or unreadable ``ca_files`` entry, mirroring
    ``config.resolve_timeout``.
    """
    got = _cfg_get(config, "network", "tls")
    base: Mapping[str, Any] = got if isinstance(got, Mapping) else {}
    merged = _merge_tls_mappings(base, override) if override else dict(base)
    policy = _validate_policy_mapping(merged, "network.tls")
    return TLSPolicy(policy.mode, policy.ca_files, purpose)


def resolve_download_policy(config: Any) -> TLSPolicy:
    """Policy for model/archive downloads: ``network.download_tls`` over ``network.tls``."""
    override = None
    got = _cfg_get(config, "network", "download_tls")
    if isinstance(got, Mapping):
        override = got
    elif got is not None:
        raise ValueError(f"Invalid network.download_tls {got!r}: must be a mapping or null")
    return resolve_tls_policy(config, override, purpose="model download")


_warned: set[tuple[str, str]] = set()
_warned_lock = threading.Lock()


def _warn_if_weak(policy: TLSPolicy) -> None:
    """WARNING once per process per (purpose/endpoint, mode) when verification is relaxed."""
    if not policy.is_weak:
        return
    key = (policy.label or "default", policy.mode)
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    if policy.mode == "insecure":
        detail = (
            "NO certificate verification (traffic is encrypted but the server is not authenticated)"
        )
    else:
        detail = (
            "the certificate chain is NOT verified (hostname and expiry are still checked); "
            "a machine-in-the-middle with a self-made certificate would be accepted"
        )
    _get_logger().warning(
        "TLS mode %r in effect for %s: %s", policy.mode, policy.label or "default", detail
    )


def _reset_warnings_for_tests() -> None:
    with _warned_lock:
        _warned.clear()


# ─── SSL context ───────────────────────────────────────────────────


def build_ssl_context(policy: TLSPolicy) -> ssl.SSLContext:
    """Build the ``ssl.SSLContext`` for ``policy``.

    ``untrusted_ca`` returns a ``CERT_NONE`` context; the leaf checks that make that mode safe
    run in ``_ValidatingHTTPSConnection.connect`` (they need the live socket).
    """
    if policy.mode == "verify":
        ctx = ssl.create_default_context()
        for path in policy.ca_files:
            try:
                ctx.load_verify_locations(cafile=path)
            except (OSError, ssl.SSLError) as e:
                raise ValueError(f"network.tls.ca_files: cannot load {path!r}: {e}") from e
        if policy.ca_files:
            # Python 3.13+ turns on VERIFY_X509_STRICT by default, which rejects many real-world
            # corporate TLS-inspection CAs (missing Authority/Subject Key Identifier, non-critical
            # basicConstraints). The user explicitly chose to trust these files, and the chain
            # and hostname are still fully verified; only the extra RFC 5280 pedantry is dropped.
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return ctx
    if policy.mode in ("untrusted_ca", "insecure"):
        ctx = ssl.create_default_context()
        # Order matters: check_hostname must be off before verify_mode can be CERT_NONE.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    raise ValueError(f"Invalid network.tls.mode {policy.mode!r}")


# ─── Leaf validation for untrusted_ca ──────────────────────────────


def _normalize_dns(name: str) -> str:
    """Lowercase, strip one trailing dot, and IDNA-encode (A-label) a hostname for comparison."""
    name = name.strip().rstrip(".").lower()
    try:
        return name.encode("idna").decode("ascii")
    except UnicodeError:
        return name


def match_dns_name(pattern: str, host: str) -> bool:
    """RFC 6125 §6.4 DNS-ID match of a certificate name against a (non-IP) host.

    Case-insensitive; IDN compared as A-labels. A wildcard is honoured only as the *entire*
    left-most label (``*.example.com``); it matches exactly one label (not ``example.com``, not
    ``a.b.example.com``), and ``*.tld``-style patterns (fewer than two labels after the
    wildcard's own) are refused. Partial wildcards (``f*.example.com``) are not supported.
    """
    pat = _normalize_dns(pattern)
    hst = _normalize_dns(host)
    if not pat or not hst:
        return False
    if "*" not in pat:
        return pat == hst
    labels = pat.split(".")
    if labels[0] != "*" or any("*" in label for label in labels[1:]):
        return False
    rest = labels[1:]
    if len(rest) < 2 or not all(rest):
        return False
    host_labels = hst.split(".")
    if len(host_labels) != len(labels) or not host_labels[0]:
        return False
    return host_labels[1:] == rest


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def match_certificate_hostname(cert: Any, host: str) -> bool:
    """Does ``cert`` (a ``cryptography.x509.Certificate``) identify ``host``?

    SAN DNS names (host is a name) or SAN IP addresses (host is an IP literal) are authoritative
    when a SubjectAlternativeName extension exists -- even one holding only other name types.
    The subject CN is consulted only when there is *no* SAN extension at all (legacy certs), per
    RFC 6125 §6.4.4. An IP literal never matches a DNS name or wildcard.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    ip = _parse_ip(host)
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        san = None

    if san is not None:
        if ip is not None:
            return any(ip == candidate for candidate in san.get_values_for_type(x509.IPAddress))
        return any(match_dns_name(n, host) for n in san.get_values_for_type(x509.DNSName))

    for attr in cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME):
        value = attr.value if isinstance(attr.value, str) else attr.value.decode("utf-8", "replace")
        if ip is not None:
            if _parse_ip(value) == ip:
                return True
        elif match_dns_name(value, host):
            return True
    return False


def validate_peer_certificate(der: bytes, host: str, *, now: datetime | None = None) -> None:
    """``untrusted_ca`` leaf checks: validity window and hostname. Raises ``TLSValidationError``.

    The chain/issuer is deliberately NOT examined (that is the point of the mode).
    """
    from cryptography import x509

    try:
        cert = x509.load_der_x509_certificate(der)
    except ValueError as e:
        raise TLSValidationError(1, f"untrusted_ca: unparseable peer certificate: {e}") from e
    current = now or datetime.now(UTC)
    if current < cert.not_valid_before_utc:
        raise TLSValidationError(
            1, f"untrusted_ca: certificate not yet valid (valid from {cert.not_valid_before_utc})"
        )
    if current > cert.not_valid_after_utc:
        raise TLSValidationError(
            1, f"untrusted_ca: certificate has expired (expired {cert.not_valid_after_utc})"
        )
    if not match_certificate_hostname(cert, host):
        raise TLSValidationError(1, f"untrusted_ca: certificate does not match hostname {host!r}")


class _ValidatingHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that runs ``validate_peer_certificate`` right after the handshake.

    Doing the check inside ``connect()`` (rather than after the response) means it runs for every
    connection urllib opens, including each hop of a redirect chain, and before any request bytes
    (credentials, frames) are sent.
    """

    def connect(self) -> None:
        super().connect()
        sock = self.sock
        assert isinstance(sock, ssl.SSLSocket)
        # With a CONNECT tunnel `self.host` is the proxy; the certificate must match the target.
        target = getattr(self, "_tunnel_host", None) or self.host
        try:
            der = sock.getpeercert(binary_form=True)
            if not der:
                raise TLSValidationError(1, "untrusted_ca: server presented no certificate")
            validate_peer_certificate(der, target)
        except BaseException:
            sock.close()
            self.sock = None
            raise


class _ValidatingHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_ValidatingHTTPSConnection, req, context=self._context)  # type: ignore[attr-defined]


def build_opener(policy: TLSPolicy) -> urllib.request.OpenerDirector:
    ctx = build_ssl_context(policy)
    handler: urllib.request.HTTPSHandler
    if policy.mode == "untrusted_ca":
        handler = _ValidatingHTTPSHandler(context=ctx)
    else:
        handler = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(handler)


def open_url(
    request_or_url: urllib.request.Request | str,
    *,
    timeout: float | None,
    tls: TLSPolicy | None = None,
) -> Any:
    """``urlopen`` with the TLS policy applied. Use as a context manager like ``urlopen``.

    ``tls=None`` means the secure default (``verify``, system store only).
    """
    policy = tls if tls is not None else DEFAULT_POLICY
    _warn_if_weak(policy)
    opener = build_opener(policy)
    if timeout is None:
        return opener.open(request_or_url)
    return opener.open(request_or_url, timeout=timeout)


# ─── requests / curl fallbacks ─────────────────────────────────────


def _base_ca_bundle() -> str | None:
    """The default CA bundle `requests` would use (certifi), else the system cafile."""
    try:
        import certifi

        path = certifi.where()
        if os.path.isfile(path):
            return path
    except ImportError:
        pass
    cafile = ssl.get_default_verify_paths().cafile
    if cafile and os.path.isfile(cafile):
        return cafile
    return None


@contextlib.contextmanager
def _combined_bundle(policy: TLSPolicy) -> Iterator[str | None]:
    """Yield a path to (default bundle + ca_files) as a temp PEM, or None if no ca_files.

    curl's ``--cacert`` and requests' ``verify=<path>`` *replace* the default trust store, but the
    policy says ca_files are *added*; so build the union. The temp file is removed on exit.
    """
    if not policy.ca_files:
        yield None
        return
    base = _base_ca_bundle()
    if base is None:
        raise FallbackUnsupported(
            "no default CA bundle (certifi/system) found to extend with network.tls.ca_files"
        )
    tmpdir = tempfile.mkdtemp(prefix="avf_ca_")
    try:
        bundle = os.path.join(tmpdir, "bundle.pem")
        with open(bundle, "wb") as out:
            with open(base, "rb") as f:
                shutil.copyfileobj(f, out)
            for extra in policy.ca_files:
                out.write(b"\n")
                with open(extra, "rb") as f:
                    shutil.copyfileobj(f, out)
        yield bundle
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def requests_verify(policy: TLSPolicy) -> Iterator[bool | str]:
    """The ``verify=`` argument for ``requests`` honouring ``policy``.

    ``insecure`` -> False; ``verify`` -> True or a combined bundle path (cleaned up on exit).
    Raises ``FallbackUnsupported`` for ``untrusted_ca`` (requests cannot run our leaf checks).
    """
    _warn_if_weak(policy)
    if policy.mode == "untrusted_ca":
        raise FallbackUnsupported(
            "network.tls.mode=untrusted_ca cannot be expressed to requests "
            "(hostname/expiry-only checking)"
        )
    if policy.mode == "insecure":
        yield False
        return
    with _combined_bundle(policy) as bundle:
        yield bundle if bundle is not None else True


@contextlib.contextmanager
def curl_tls_args(policy: TLSPolicy) -> Iterator[list[str]]:
    """Extra ``curl`` arguments honouring ``policy`` (``-k`` / ``--cacert <bundle>`` / none).

    Raises ``FallbackUnsupported`` for ``untrusted_ca``.
    """
    _warn_if_weak(policy)
    if policy.mode == "untrusted_ca":
        raise FallbackUnsupported(
            "network.tls.mode=untrusted_ca cannot be expressed to curl "
            "(hostname/expiry-only checking)"
        )
    if policy.mode == "insecure":
        yield ["-k"]
        return
    with _combined_bundle(policy) as bundle:
        yield ["--cacert", bundle] if bundle is not None else []


__all__ = [
    "DEFAULT_POLICY",
    "VALID_TLS_MODES",
    "FallbackUnsupported",
    "TLSPolicy",
    "TLSValidationError",
    "build_opener",
    "build_ssl_context",
    "curl_tls_args",
    "match_certificate_hostname",
    "match_dns_name",
    "open_url",
    "requests_verify",
    "resolve_download_policy",
    "resolve_tls_policy",
    "validate_peer_certificate",
]
