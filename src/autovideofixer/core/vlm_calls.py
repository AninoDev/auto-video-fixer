"""Per-call timing / token statistics for every VLM and LLM request (REQUIREMENTS.md § 26.2).

Every model call AVF makes (whole-video VLM, per-scene VLM, crop check, scene coordinator)
produces one ``CallStats``. Provider functions in ``core/analysis.py`` return a
``ProviderResponse(text, stats)`` instead of a bare string; ``_call_provider_raw`` stamps the
caller-side facts (endpoint name, purpose, estimated vision tokens), logs the call and records it
with ``record_call`` so any enclosing ``collect_call_stats()`` block (one ``avf analyze`` file, one
``avf process`` job) receives it without the call sites having to thread a list around.

Token/time sources, by what the server actually reports:

* **Ollama** ``/api/chat``: ``prompt_eval_count`` / ``eval_count`` and the ``*_duration`` fields
  (nanoseconds) give exact tokens and exact tokens/s.
* **llama.cpp** (OpenAI-shaped): ``usage`` for tokens plus a ``timings`` object whose
  ``prompt_per_second`` / ``predicted_per_second`` are exact tokens/s.
* **OpenAI and other OpenAI-shaped servers**: ``usage`` only. Generation tokens/s is then
  ``output_tokens / wall`` and flagged ``tps_estimated`` (it includes prompt processing and network
  time, so it under-reads); prompt tokens/s is unknown.

Without streaming there is no time-to-first-token; that needs ``stream: true`` plus SSE parsing
(listed under "revisit" in § 26.3).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import socket
import threading
import urllib.error
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

# Error classes recorded in CallStats.error_class. Anything else is the exception class name.
ERR_TIMEOUT = "timeout"
ERR_CONNECTION = "connection"
ERR_BAD_JSON = "bad_json"
ERR_BAD_RESPONSE = "bad_response"
ERR_EMPTY = "empty_response"
ERR_URL_REFUSED = "url_refused"
ERR_UNKNOWN_PROVIDER = "unknown_provider"


@dataclass
class CallStats:
    """Timing and token figures for one model call. ``None`` = the server did not report it."""

    endpoint: str = ""
    provider: str = ""
    model: str = ""
    purpose: str = ""  # video | scene | crop_check | coordinator | ...
    detail: str = ""  # e.g. "scene[5.0-9.0]"; never contains the file path
    wall_sec: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    prompt_tps: float | None = None
    gen_tps: float | None = None
    tps_estimated: bool = False  # True when gen_tps was derived from output_tokens / wall_sec
    images: int = 0
    est_vision_tokens: int | None = None
    success: bool = True
    error_class: str | None = None
    error: str | None = None  # short, one-line, no URLs
    usage_source: str = "none"  # ollama | usage | usage+timings | none
    # Why generation stopped, as the server reported it ("stop", "length", ...). "length" means
    # the reply hit max_tokens and was cut off -- the call still succeeded, but the JSON is
    # usually incomplete, so it is flagged and warned about (see ``truncation_message``).
    finish_reason: str | None = None
    truncated: bool = False
    # Raw usage/timings the server sent, for the DEBUG log only (not serialised).
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        for key in ("wall_sec", "prompt_tps", "gen_tps"):
            if isinstance(d[key], float):
                d[key] = round(d[key], 3)
        return d


@dataclass
class ProviderResponse:
    """What a provider function returns: the model's text plus the call's stats."""

    text: str
    stats: CallStats


# ─── Error classification ──────────────────────────────────────────


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """``(error_class, one_line_message)`` for an exception raised by a provider request.

    The message never contains the request URL (it can embed hosts/paths the PII-clean log must
    not show) -- callers name the *endpoint* instead.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}", f"HTTP {exc.code} {exc.reason}".strip()
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return ERR_TIMEOUT, "request timed out"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return ERR_TIMEOUT, "connection timed out"
        text = str(reason).replace("\n", " ")[:160]
        return ERR_CONNECTION, text or "connection failed"
    if isinstance(exc, OSError):
        return ERR_CONNECTION, (str(exc).replace("\n", " ")[:160] or type(exc).__name__)
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return ERR_BAD_JSON, "response body was not valid JSON"
    if isinstance(exc, (KeyError, IndexError, TypeError, AttributeError)):
        return ERR_BAD_RESPONSE, "response JSON did not have the expected shape"
    return type(exc).__name__, (str(exc).replace("\n", " ")[:160] or type(exc).__name__)


def failure_message(stats: CallStats, timeout: float | None) -> str:
    """Concise console text for a failed call: names the endpoint, the cause and a hint."""
    who = f"endpoint '{stats.endpoint}'" if stats.endpoint else f"{stats.provider} endpoint"
    head = f"{stats.purpose or 'VLM'} call to {who} (model {stats.model or '?'}) failed"
    cls = stats.error_class or "error"
    if cls == ERR_TIMEOUT:
        limit = f"{timeout:g}s" if timeout else "its limit"
        name = stats.endpoint or "<name>"
        return (
            f"{head}: timed out after {limit}. Hint: raise "
            f"analysis.endpoints.{name}.timeout_sec (currently "
            f"{f'{timeout:g}' if timeout else 'unset'}) if the model is just slow, or check "
            "that the server is not overloaded."
        )
    if cls == ERR_CONNECTION:
        return (
            f"{head}: could not connect ({stats.error}). Hint: check the server is running "
            "and that api_url / allow_http are right."
        )
    if cls in ("http_401", "http_403"):
        return f"{head}: {stats.error}. Hint: check the endpoint's api_key."
    if cls.startswith("http_"):
        return f"{head}: {stats.error}. Hint: see the model server's own log for the cause."
    if cls == ERR_BAD_JSON:
        return (
            f"{head}: the server's reply was not valid JSON. Hint: api_url may point at the "
            "wrong path (a web page or proxy error) or the server is mid-restart."
        )
    if cls == ERR_URL_REFUSED:
        return f"{head}: {stats.error}"
    return f"{head}: {stats.error or cls}"


# ─── Stats extraction from response bodies ─────────────────────────


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> int | None:
    n = _num(value)
    return None if n is None else int(n)


def _positive_rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds


def _set_finish_reason(stats: CallStats, reason: Any) -> None:
    if isinstance(reason, str) and reason:
        stats.finish_reason = reason
        stats.truncated = reason == "length"


def truncation_message(stats: CallStats, max_tokens: int | None) -> str:
    """Console warning for a reply cut off at ``max_tokens`` (``finish_reason: length``)."""
    name = stats.endpoint or "<name>"
    limit = f"{max_tokens}" if max_tokens else "unset"
    return (
        f"{stats.purpose or 'VLM'} call to endpoint '{name}' (model {stats.model or '?'}): the "
        f"reply was cut off at the token limit, so some fields may be missing. Hint: raise "
        f"analysis.endpoints.{name}.max_tokens (currently {limit}), especially with a higher "
        "analysis.vlm.detail level or a reasoning model."
    )


def apply_ollama_usage(stats: CallStats, body: Mapping[str, Any], wall: float) -> None:
    """Fill ``stats`` from an Ollama ``/api/chat`` response (durations are nanoseconds)."""
    stats.input_tokens = _int(body.get("prompt_eval_count"))
    stats.output_tokens = _int(body.get("eval_count"))
    if stats.input_tokens is not None or stats.output_tokens is not None:
        stats.total_tokens = (stats.input_tokens or 0) + (stats.output_tokens or 0)
        stats.usage_source = "ollama"
    pe_ns, ev_ns = _num(body.get("prompt_eval_duration")), _num(body.get("eval_duration"))
    stats.prompt_tps = _positive_rate(stats.input_tokens, pe_ns / 1e9 if pe_ns else None)
    stats.gen_tps = _positive_rate(stats.output_tokens, ev_ns / 1e9 if ev_ns else None)
    _set_finish_reason(stats, body.get("done_reason"))
    stats.raw = {
        k: body[k]
        for k in (
            "prompt_eval_count",
            "eval_count",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
        if k in body
    }


def apply_openai_usage(stats: CallStats, body: Mapping[str, Any], wall: float) -> None:
    """Fill ``stats`` from an OpenAI-shaped response (``usage``, plus llama.cpp ``timings``)."""
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        _set_finish_reason(stats, choices[0].get("finish_reason"))
    usage = body.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    timings = body.get("timings")
    timings = timings if isinstance(timings, Mapping) else {}

    stats.input_tokens = _int(usage.get("prompt_tokens"))
    stats.output_tokens = _int(usage.get("completion_tokens"))
    stats.total_tokens = _int(usage.get("total_tokens"))
    details = usage.get("completion_tokens_details")
    if isinstance(details, Mapping):
        stats.reasoning_tokens = _int(details.get("reasoning_tokens"))
    # llama.cpp may omit `usage` on some builds but always has timings.*_n; prompt_n excludes
    # tokens served from the prompt cache, so it is only a fallback.
    if stats.input_tokens is None:
        stats.input_tokens = _int(timings.get("prompt_n"))
    if stats.output_tokens is None:
        stats.output_tokens = _int(timings.get("predicted_n"))
    if stats.total_tokens is None and (
        stats.input_tokens is not None or stats.output_tokens is not None
    ):
        stats.total_tokens = (stats.input_tokens or 0) + (stats.output_tokens or 0)

    stats.prompt_tps = _num(timings.get("prompt_per_second"))
    stats.gen_tps = _num(timings.get("predicted_per_second"))
    if stats.prompt_tps is None:
        stats.prompt_tps = _positive_rate(
            _int(timings.get("prompt_n")),
            (_num(timings.get("prompt_ms")) or 0) / 1000.0 or None,
        )
    if stats.gen_tps is None:
        stats.gen_tps = _positive_rate(
            _int(timings.get("predicted_n")),
            (_num(timings.get("predicted_ms")) or 0) / 1000.0 or None,
        )
    if stats.gen_tps is None and stats.output_tokens is not None and wall > 0:
        stats.gen_tps = stats.output_tokens / wall
        stats.tps_estimated = True

    if timings:
        stats.usage_source = "usage+timings" if usage else "timings"
    elif usage:
        stats.usage_source = "usage"
    stats.raw = {"usage": dict(usage), "timings": dict(timings)} if (usage or timings) else {}


# ─── Collection ────────────────────────────────────────────────────

_collectors: contextvars.ContextVar[tuple[list[CallStats], ...]] = contextvars.ContextVar(
    "avf_call_stat_collectors", default=()
)


@contextlib.contextmanager
def collect_call_stats() -> Iterator[list[CallStats]]:
    """Collect every ``record_call`` made in this context (nestable; inner calls reach all)."""
    sink: list[CallStats] = []
    token = _collectors.set(_collectors.get() + (sink,))
    try:
        yield sink
    finally:
        _collectors.reset(token)


# Cross-check legs may run in worker threads (§ 27.6). A worker started with
# ``contextvars.copy_context().run`` sees the same sink lists, so appends from several threads
# are serialised here.
_record_lock = threading.Lock()


def record_call(stats: CallStats) -> None:
    with _record_lock:
        for sink in _collectors.get():
            sink.append(stats)


# ─── Formatting / totals ───────────────────────────────────────────


def _fmt_tokens(n: int | None) -> str:
    return "?" if n is None else str(n)


def _fmt_rate(rate: float | None, estimated: bool = False) -> str:
    if rate is None:
        return "?"
    return f"{rate:.1f}{'~' if estimated else ''}"


def format_call_line(stats: CallStats | Mapping[str, Any]) -> str:
    """One-line human summary of a call (INFO log line and the narrow-console fallback)."""
    d = stats.to_dict() if isinstance(stats, CallStats) else dict(stats)
    where = f"{d.get('purpose') or 'call'}"
    if d.get("detail"):
        where += f" {d['detail']}"
    head = (
        f"VLM call {where} endpoint={d.get('endpoint') or '-'} provider={d.get('provider')} "
        f"model={d.get('model')}: {d.get('wall_sec', 0.0):.2f}s"
    )
    if not d.get("success", True):
        return f"{head}, FAILED ({d.get('error_class')})"
    if d.get("truncated"):
        head += ", TRUNCATED at max_tokens"
    reasoning = d.get("reasoning_tokens")
    out = _fmt_tokens(d.get("output_tokens")) + (f" (reasoning {reasoning})" if reasoning else "")
    est = d.get("est_vision_tokens")
    return (
        f"{head}, tokens in={_fmt_tokens(d.get('input_tokens'))} out={out} "
        f"total={_fmt_tokens(d.get('total_tokens'))}, prompt {_fmt_rate(d.get('prompt_tps'))} "
        f"tok/s, gen {_fmt_rate(d.get('gen_tps'), bool(d.get('tps_estimated')))} tok/s"
        f"{' (estimated)' if d.get('tps_estimated') else ''}, images={d.get('images', 0)}"
        f"{f' est_vision_tokens={est}' if est is not None else ''}"
    )


def summarize_calls(calls: list[CallStats] | list[Mapping[str, Any]]) -> dict[str, Any]:
    """Run totals over a list of calls (``CallStats`` or their ``to_dict()`` form)."""
    dicts = [c.to_dict() if isinstance(c, CallStats) else dict(c) for c in calls]

    def total(key: str) -> int:
        return sum(int(d[key]) for d in dicts if d.get(key) is not None)

    return {
        "calls": len(dicts),
        "succeeded": sum(1 for d in dicts if d.get("success", True)),
        "failed": sum(1 for d in dicts if not d.get("success", True)),
        "wall_sec": round(sum(float(d.get("wall_sec") or 0.0) for d in dicts), 3),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "total_tokens": total("total_tokens"),
        "images": total("images"),
        "est_vision_tokens": total("est_vision_tokens"),
    }
