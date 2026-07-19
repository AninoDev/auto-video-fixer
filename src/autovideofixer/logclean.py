"""Auto Video Fixer - PII-clean log substitution (REQUIREMENTS.md § 6.7).

``PIICleaner`` holds a per-run, per-instance mapping of KNOWN real values
(actual input/output/config paths, configured VLM/LLM endpoint hosts, embedded
video titles) to stable, human-readable placeholders, and a ``clean()``
method that substitutes every registered real value it finds inside a string.

This is substitution, not guesswork regexes -- that's the § 6.7 reliability
contract: only values some code path explicitly registered ever get replaced,
so a cleaned log can be trusted not to have silently mangled unrelated text.
Free-text PII embedded in e.g. a DEBUG-logged VLM response can't be caught
this way and is out of scope for v1 (documented in AGENTS.md).

Registration is cheap and idempotent (the same real value registered twice
keeps its first-assigned placeholder/number -- "numbered by first
appearance"), so call sites are free to register defensively/redundantly
(e.g. both the CLI and ``Pipeline.add_job()`` register the same input path)
without it affecting numbering.
"""

from __future__ import annotations

import os
import threading
from urllib.parse import urlsplit

_VALID_DIRECTORY_ROLES = ("input", "output", "config", "logs")


class PIICleaner:
    """Per-run KNOWN-value -> placeholder substitution mapping.

    Thread-safe: registration happens from worker threads (stage execution,
    scene-mode workers, ...) as well as the main CLI thread, and ``clean()``
    itself runs on whatever thread emits a log record through a clean file
    handler -- a single lock serializes both.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # real path (verbatim string as registered) -> placeholder filename
        # (e.g. "input_video_01.mkv"), insertion-ordered so "numbered by
        # first appearance" falls out of dict iteration order for free.
        self._input_files: dict[str, str] = {}
        self._output_files: dict[str, str] = {}
        # real directory path (verbatim) -> "/path/to/<role>/".
        self._dirs: dict[str, str] = {}
        # "scheme://host[:port]" (verbatim) -> "http://vlm-endpoint" (fixed,
        # not numbered -- see register_endpoint()'s docstring).
        self._endpoints: dict[str, str] = {}
        # real title text -> "video_title_01".
        self._titles: dict[str, str] = {}

    # --- registration -----------------------------------------------------

    def register_input(self, path: str | None) -> None:
        """Register a real input file path, e.g. from a resolved CLI arg or
        ``Job.input_path``. No-op for a falsy/already-registered path."""
        if not path:
            return
        with self._lock:
            if path in self._input_files:
                return
            n = len(self._input_files) + 1
            ext = os.path.splitext(path)[1]
            self._input_files[path] = f"input_video_{n:02d}{ext}"

    def register_output(self, path: str | None) -> None:
        """Register a real output file path (including a mismatch-renamed
        output -- see ``Pipeline._rename_mismatched_output()``). No-op for a
        falsy/already-registered path."""
        if not path:
            return
        with self._lock:
            if path in self._output_files:
                return
            n = len(self._output_files) + 1
            ext = os.path.splitext(path)[1]
            self._output_files[path] = f"output_video_{n:02d}{ext}"

    def register_directory(self, path: str | None, role: str) -> None:
        """Register a real directory path under a role ("input"/"output"/
        "config"), replaced with the fixed ``/path/to/<role>/`` placeholder
        (not numbered -- directories are role-based, not per-instance).
        No-op for a falsy/already-registered path.
        """
        if not path:
            return
        if role not in _VALID_DIRECTORY_ROLES:
            raise ValueError(
                f"Invalid directory role: {role!r} (must be one of {_VALID_DIRECTORY_ROLES!r})"
            )
        with self._lock:
            if path in self._dirs:
                return
            self._dirs[path] = f"/path/to/{role}/"

    def register_endpoint(self, url: str | None) -> None:
        """Register a real network endpoint (e.g. ``analysis.vlm.api_url``).

        Only the ``scheme://host[:port]`` prefix is replaced -- the path/
        query is kept intact so a cleaned log still shows WHICH API route was
        hit (e.g. ``http://10.0.0.5:11434/api/generate`` ->
        ``http://vlm-endpoint/api/generate``). Every registered endpoint maps
        to the SAME fixed placeholder (not numbered, unlike inputs/outputs/
        titles) -- per REQUIREMENTS.md § 6.7's literal example. No-op for a
        falsy/unparsable/already-registered URL.
        """
        if not url:
            return
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return
        prefix = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            if prefix in self._endpoints:
                return
            self._endpoints[prefix] = "http://vlm-endpoint"

    def register_title(self, text: str | None) -> None:
        """Register an embedded video title (e.g. ``ProbeResult.title``).
        No-op for a falsy/already-registered title."""
        if not text:
            return
        with self._lock:
            if text in self._titles:
                return
            n = len(self._titles) + 1
            self._titles[text] = f"video_title_{n:02d}"

    # --- substitution -------------------------------------------------

    def clean(self, text: str) -> str:
        """Return `text` with every registered real value substituted.

        Substitution order: longest registered string first, so a full path
        is replaced before its bare directory, and a directory before a bare
        basename -- e.g. ``/data/in/vid.mkv`` (input file registered, with
        ``/data/in`` also registered as an "input"-role directory) cleans to
        ``/path/to/input/input_video_01.mkv``: the full-path entry below
        composes the directory's placeholder with the file's placeholder
        filename, so it wins over the shorter bare-directory and bare-
        basename entries competing for the same substring.
        """
        with self._lock:
            mapping: dict[str, str] = {}

            for path, filename_ph in self._input_files.items():
                dirname = os.path.dirname(path)
                dir_ph = self._dirs.get(dirname)
                if dir_ph is not None:
                    mapping[path] = dir_ph + filename_ph
                basename = os.path.basename(path)
                if basename:
                    mapping.setdefault(basename, filename_ph)

            for path, filename_ph in self._output_files.items():
                dirname = os.path.dirname(path)
                dir_ph = self._dirs.get(dirname)
                if dir_ph is not None:
                    mapping[path] = dir_ph + filename_ph
                basename = os.path.basename(path)
                if basename:
                    mapping.setdefault(basename, filename_ph)

            for path, dir_ph in self._dirs.items():
                # The trailing-slash variant first (longest-match-first makes
                # it win): the placeholder already ends in "/", so replacing
                # only the bare directory would leave the original separator
                # behind and render doubled slashes ("/path/to/logs//run.log").
                mapping.setdefault(path.rstrip("/") + "/", dir_ph)
                mapping.setdefault(path, dir_ph)

            for prefix, placeholder in self._endpoints.items():
                mapping.setdefault(prefix, placeholder)

            for title, placeholder in self._titles.items():
                mapping.setdefault(title, placeholder)

        for real in sorted(mapping, key=len, reverse=True):
            if real and real in text:
                text = text.replace(real, mapping[real])
        return text


_cleaner_lock = threading.Lock()
_singleton: PIICleaner | None = None


def get_pii_cleaner() -> PIICleaner:
    """Return the process-wide singleton ``PIICleaner`` (lazily created)."""
    global _singleton
    with _cleaner_lock:
        if _singleton is None:
            _singleton = PIICleaner()
        return _singleton


def reset_pii_cleaner() -> None:
    """Replace the singleton with a fresh, empty ``PIICleaner``. Test-only."""
    global _singleton
    with _cleaner_lock:
        _singleton = PIICleaner()
