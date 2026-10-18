"""Real-TLS test scaffolding: generate certificates with `cryptography` and serve them.

No part of the TLS stack is mocked -- servers are genuine `ssl`-wrapped `http.server`s on
127.0.0.1:0, so the accept/reject matrices in test_net_tls.py exercise the real handshake.
Also imported by the P2 end-to-end scripts.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import ssl
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

PAYLOAD = b"avf-test-payload-" * 512  # small file served at /file


@dataclass
class Cert:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey
    cert_path: Path
    key_path: Path


def make_cert(
    cn: str,
    *,
    sans: list[x509.GeneralName] | None,
    issuer: tuple[x509.Certificate, ec.EllipticCurvePrivateKey] | None = None,
    is_ca: bool = False,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
    key_ids: bool = True,
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    now = datetime.now(UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    issuer_name = issuer[0].subject if issuer else name
    signing_key = issuer[1] if issuer else key
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(days=1))
        .not_valid_after(not_after or now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None), critical=True)
    )
    if is_ca:
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    if key_ids:
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        if issuer:
            builder = builder.add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer[0].public_key()),
                critical=False,
            )
    if sans is not None:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return builder.sign(signing_key, hashes.SHA256()), key


LOCAL_SANS: list[x509.GeneralName] = [
    x509.DNSName("localhost"),
    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
]
OTHER_SANS: list[x509.GeneralName] = [x509.DNSName("other.example.com")]


def _write(dirpath: Path, name: str, cert: x509.Certificate, key: ec.EllipticCurvePrivateKey):
    cert_path = dirpath / f"{name}.pem"
    key_path = dirpath / f"{name}.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return Cert(cert, key, cert_path, key_path)


def generate_all(dirpath: Path) -> dict[str, Cert]:
    """Generate every certificate the matrices need, written under `dirpath`."""
    now = datetime.now(UTC)
    out: dict[str, Cert] = {}

    def add(name: str, pair: tuple[x509.Certificate, ec.EllipticCurvePrivateKey]) -> None:
        out[name] = _write(dirpath, name, *pair)

    ca = make_cert("AVF Test Private CA", sans=None, is_ca=True)
    add("ca", ca)
    # A CA with no Subject/Authority Key Identifier, like many corporate inspection CAs.
    ca_no_ids = make_cert("AVF Sloppy Corporate CA", sans=None, is_ca=True, key_ids=False)
    add("ca_no_ids", ca_no_ids)

    add("ca_signed", make_cert("localhost", sans=LOCAL_SANS, issuer=ca))
    add("sloppy_signed", make_cert("localhost", sans=LOCAL_SANS, issuer=ca_no_ids, key_ids=False))
    add("self_signed", make_cert("localhost", sans=LOCAL_SANS))
    add(
        "expired",
        make_cert(
            "localhost",
            sans=LOCAL_SANS,
            not_before=now - timedelta(days=10),
            not_after=now - timedelta(days=1),
        ),
    )
    add(
        "expired_ca_signed",
        make_cert(
            "localhost",
            sans=LOCAL_SANS,
            issuer=ca,
            not_before=now - timedelta(days=10),
            not_after=now - timedelta(days=1),
        ),
    )
    add(
        "not_yet_valid",
        make_cert(
            "localhost",
            sans=LOCAL_SANS,
            not_before=now + timedelta(days=1),
            not_after=now + timedelta(days=10),
        ),
    )
    add("wrong_host", make_cert("other.example.com", sans=OTHER_SANS))
    add("wrong_host_ca_signed", make_cert("other.example.com", sans=OTHER_SANS, issuer=ca))
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    redirect_to: str = ""

    def log_message(self, *args, **kwargs) -> None:  # noqa: D401 - silence
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/octet-stream") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/file":
            self._send(200, PAYLOAD)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", self.server.redirect_to)  # type: ignore[attr-defined]
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send(404, b"nope")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {"choices": [{"message": {"content": '{"summary": "tls-ok", "tags": []}'}}]}
        ).encode()
        self._send(200, body, "application/json")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    redirect_to = ""

    def handle_error(self, request, client_address) -> None:  # failed handshakes are expected
        pass


@contextlib.contextmanager
def serve(cert: Cert, redirect_to: str = "") -> Iterator[int]:
    """Serve /file, /redirect and an OpenAI-shaped POST over TLS; yield the port."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert.cert_path), str(cert.key_path))
    server = _Server(("127.0.0.1", 0), _Handler)
    server.redirect_to = redirect_to
    # Handshake lazily in the handler thread so a rejected client never blocks accept().
    server.socket = ctx.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
