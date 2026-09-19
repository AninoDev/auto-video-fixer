"""Tests for the `general.symlinks` directory-scan policy (REQUIREMENTS.md § 21).

``core.analysis.scan_directory()`` used to treat symlinks inconsistently and
undocumented: a symlinked FILE was picked up (via ``os.walk``'s own `files`
list + `is_video_file()` transparently following it), but a symlinked
DIRECTORY was never descended into (``os.walk``'s ``followlinks=False``
default). ``symlinks="files_only"`` now documents that exact behavior;
``"follow"`` also descends into symlinked directories (with a loop guard);
``"skip"`` ignores every symlink, file or directory.

Real filesystem symlinks are used throughout (no ffmpeg/subprocess mocking
needed -- this is pure path-scanning logic).
"""

from __future__ import annotations

import io
import os

import pytest
from rich.console import Console

from autovideofixer.cli.cli import _plan_resolved_inputs
from autovideofixer.config import Config
from autovideofixer.core.analysis import scan_directory
from autovideofixer.core.pipeline import Pipeline


def _video(path, content=b"fake video content"):
    with open(path, "wb") as f:
        f.write(content)
    return str(path)


def _symlinks_supported(tmp_path) -> bool:
    """Some sandboxed/Windows-without-privilege environments can't create
    symlinks -- skip rather than fail in that case."""
    try:
        target = tmp_path / "_symlink_probe_target"
        target.write_text("x")
        link = tmp_path / "_symlink_probe_link"
        os.symlink(target, link)
        return True
    except OSError, NotImplementedError:
        return False


@pytest.fixture(autouse=True)
def _require_symlinks(tmp_path):
    if not _symlinks_supported(tmp_path):
        pytest.skip("symlinks not supported in this environment")


def _build_tree(tmp_path):
    """
    tmp_path/
      real/
        a.mp4                (real file)
      linked_dir -> real     (symlinked directory, contains a.mp4)
      b_link.mp4 -> real/a.mp4   (symlinked file, points at a valid video)
    """
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_video = _video(real_dir / "a.mp4")

    linked_dir = tmp_path / "linked_dir"
    os.symlink(real_dir, linked_dir)

    linked_file = tmp_path / "b_link.mp4"
    os.symlink(real_video, linked_file)

    return str(real_dir), real_video, str(linked_dir), str(linked_file)


class TestFilesOnlyPolicy:
    """Default policy: symlinked files followed, symlinked dirs not descended into."""

    def test_symlinked_file_at_top_level_is_picked_up(self, tmp_path):
        _real_dir, _real_video, _linked_dir, linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="files_only")
        assert linked_file in videos

    def test_symlinked_directory_is_not_descended_into(self, tmp_path):
        real_dir, real_video, linked_dir, _linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="files_only")
        # The REAL video (via real/a.mp4) is found through the real directory
        # walk regardless of the symlink.
        assert real_video in videos
        # But nothing is reached ONLY via linked_dir (no distinct path
        # through linked_dir/a.mp4 appears).
        assert not any(v.startswith(str(linked_dir) + os.sep) for v in videos)

    def test_is_the_default(self, tmp_path):
        _real_dir, _real_video, _linked_dir, linked_file = _build_tree(tmp_path)
        default_videos = scan_directory(str(tmp_path), recursive=True)
        explicit_videos = scan_directory(str(tmp_path), recursive=True, symlinks="files_only")
        assert sorted(default_videos) == sorted(explicit_videos)
        assert linked_file in default_videos


class TestFollowPolicy:
    def test_descends_into_symlinked_directory(self, tmp_path):
        real_dir, real_video, linked_dir, linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="follow")
        via_link = os.path.join(str(linked_dir), "a.mp4")
        assert via_link in videos
        assert real_video in videos
        assert linked_file in videos

    def test_surfaces_same_file_via_link_and_real_path(self, tmp_path):
        """`follow` can legitimately list the same file twice -- once via the
        real path, once via the symlinked directory -- REQUIREMENTS.md § 21
        says this is expected and left to the existing realpath-keyed dedup."""
        real_dir, real_video, linked_dir, _linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="follow")
        via_link = os.path.join(str(linked_dir), "a.mp4")
        assert real_video in videos
        assert via_link in videos
        assert os.path.realpath(real_video) == os.path.realpath(via_link)

    def test_symlink_loop_terminates(self, tmp_path):
        """a -> b -> a must not hang os.walk(followlinks=True)."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_video = _video(real_dir / "a.mp4")

        loop_a = tmp_path / "loop_a"
        loop_b = tmp_path / "loop_b"
        loop_a.mkdir()
        # loop_a/into_b -> loop_b, loop_b/into_a -> loop_a: a genuine cycle.
        os.symlink(loop_b, loop_a / "into_b")
        loop_b.mkdir()
        os.symlink(loop_a, loop_b / "into_a")

        # Must return promptly (no hang) and not raise.
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="follow")
        assert real_video in videos

    def test_self_referential_loop_terminates(self, tmp_path):
        """A directory that symlinks to itself is the simplest possible loop."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_video = _video(real_dir / "a.mp4")
        self_loop = tmp_path / "self_loop"
        self_loop.mkdir()
        os.symlink(self_loop, self_loop / "self")

        videos = scan_directory(str(tmp_path), recursive=True, symlinks="follow")
        assert real_video in videos


class TestSkipPolicy:
    def test_ignores_symlinked_file_even_pointing_at_valid_video(self, tmp_path):
        _real_dir, real_video, _linked_dir, linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="skip")
        assert linked_file not in videos
        # The real file (not itself a symlink) is still found.
        assert real_video in videos

    def test_ignores_symlinked_directory_entirely(self, tmp_path):
        real_dir, real_video, linked_dir, _linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="skip")
        via_link = os.path.join(str(linked_dir), "a.mp4")
        assert via_link not in videos
        assert real_video in videos

    def test_tests_islink_on_entry_not_resolved_target(self, tmp_path):
        """A symlink to a directory containing a real (non-symlink) file
        must still be skipped -- the target being "fine" doesn't matter,
        only whether the entry ITSELF is a symlink."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        _video(real_dir / "a.mp4")
        linked_dir = tmp_path / "linked_dir"
        os.symlink(real_dir, linked_dir)

        videos = scan_directory(str(tmp_path), recursive=True, symlinks="skip")
        assert not any("linked_dir" in v for v in videos)


class TestNonRecursiveTopLevelSymlinks:
    def test_skip_ignores_top_level_symlinked_file(self, tmp_path):
        _real_dir, _real_video, _linked_dir, linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=False, symlinks="skip")
        assert linked_file not in videos

    def test_files_only_picks_up_top_level_symlinked_file(self, tmp_path):
        _real_dir, _real_video, _linked_dir, linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=False, symlinks="files_only")
        assert linked_file in videos


class TestInvalidPolicy:
    def test_unknown_policy_raises(self, tmp_path):
        (tmp_path / "real").mkdir()
        with pytest.raises(ValueError, match="Invalid symlinks policy"):
            scan_directory(str(tmp_path), symlinks="bogus")


class TestFollowDedupInteraction:
    """REQUIREMENTS.md § 21: `follow` surfacing the same file twice must be
    collapsed by the existing per-run content dedup (realpath-keyed), not by
    scan_directory() itself -- covered end-to-end so the interaction can't
    silently regress."""

    def test_plan_resolved_inputs_collapses_the_follow_duplicate(self, tmp_path):
        _real_dir, real_video, linked_dir, _linked_file = _build_tree(tmp_path)
        videos = scan_directory(str(tmp_path), recursive=True, symlinks="follow")
        via_link = os.path.join(str(linked_dir), "a.mp4")
        # Sanity: pre-dedup, the same content is reachable via more than one
        # distinct path (real dir, symlinked dir, and the top-level
        # symlinked file all resolve to the same realpath).
        assert real_video in videos and via_link in videos
        assert len(videos) >= 2

        resolved = [(v, None) for v in videos]
        console = Console(file=io.StringIO(), no_color=True)
        pipeline = Pipeline(Config(tmp_path / "nonexistent.yaml"))

        to_process, fanout = _plan_resolved_inputs(resolved, pipeline, console)

        # Every realpath-identical entry collapses onto exactly ONE
        # representative job, regardless of which of the (>= 2) distinct
        # paths it happened to be scanned through.
        processed_paths = [p for p, _ in to_process]
        assert len(processed_paths) == 1
        assert os.path.realpath(processed_paths[0]) == os.path.realpath(real_video)
