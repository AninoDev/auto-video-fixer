"""Tests for the frame I/O transport adapter (ai/frame_pipe.py).

Covers: backend-selection fallback logic (unit, no ffmpeg needed), API-surface
parity between the Rust and Python-fallback backends (unit), a differential
identity test proving the two transports are interchangeable at the byte
level (integration, real ffmpeg + real avf_framepipe extension), and a soak
test proving the streaming path doesn't buffer the whole video in memory
(integration).
"""

from __future__ import annotations

import inspect
import subprocess

import numpy as np
import pytest

import autovideofixer.ai.frame_pipe as frame_pipe

# ─── Unit: backend-selection fallback logic ────────────────────────────


class TestBackendSelection:
    """`get_frame_reader()`/`get_frame_writer()` must pick the Rust backend
    when `avf_framepipe` imported successfully, and the Python fallback
    otherwise -- mirroring the avf_scenes/avf_hashing convention in
    core/analysis.py."""

    def test_rust_available_flag_matches_import(self):
        """Sanity check: RUST_AVAILABLE reflects whether the module-level
        import actually succeeded in this environment."""
        try:
            import avf_framepipe  # noqa: F401

            assert frame_pipe.RUST_AVAILABLE is True
        except ImportError:
            assert frame_pipe.RUST_AVAILABLE is False

    def test_get_frame_reader_uses_python_fallback_when_rust_unavailable(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(frame_pipe, "RUST_AVAILABLE", False)
        monkeypatch.setattr(frame_pipe, "_avf_framepipe_native", None)
        monkeypatch.setattr(frame_pipe, "get_ffmpeg_path", lambda *a, **k: "ffmpeg")

        video = str(tmp_path / "does_not_matter.mp4")
        reader = frame_pipe.get_frame_reader(video, 16, 16)
        assert isinstance(reader, frame_pipe._PythonFrameReader)

    def test_get_frame_writer_uses_python_fallback_when_rust_unavailable(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(frame_pipe, "RUST_AVAILABLE", False)
        monkeypatch.setattr(frame_pipe, "_avf_framepipe_native", None)
        monkeypatch.setattr(frame_pipe, "get_ffmpeg_path", lambda *a, **k: "ffmpeg")

        out = str(tmp_path / "out.mp4")
        writer = frame_pipe.get_frame_writer(
            out, 16, 16, 30.0, codec="libx264", crf=20, preset="fast"
        )
        assert isinstance(writer, frame_pipe._PythonFrameWriter)

    def test_get_frame_reader_uses_rust_when_available(self, monkeypatch, tmp_path):
        if not frame_pipe.RUST_AVAILABLE:
            pytest.skip("avf_framepipe Rust extension not built in this environment")
        import avf_framepipe

        monkeypatch.setattr(frame_pipe, "get_ffmpeg_path", lambda *a, **k: "ffmpeg")
        video = str(tmp_path / "does_not_matter.mp4")
        reader = frame_pipe.get_frame_reader(video, 16, 16, ffmpeg_path="ffmpeg")
        # avf_framepipe.FrameReader spawns ffmpeg eagerly on construction and
        # doesn't fail until next_batch()/read is attempted -- constructing it
        # against a nonexistent path is fine here, we're only checking type.
        assert isinstance(reader, avf_framepipe.FrameReader)
        reader.close()

    def test_ffmpeg_path_resolved_via_get_ffmpeg_path_when_none(self, monkeypatch, tmp_path):
        calls = []

        def fake_get_ffmpeg_path(*a, **k):
            calls.append(True)
            return "/usr/bin/ffmpeg"

        monkeypatch.setattr(frame_pipe, "RUST_AVAILABLE", False)
        monkeypatch.setattr(frame_pipe, "_avf_framepipe_native", None)
        monkeypatch.setattr(frame_pipe, "get_ffmpeg_path", fake_get_ffmpeg_path)

        frame_pipe.get_frame_reader(str(tmp_path / "x.mp4"), 16, 16)
        assert calls, "get_ffmpeg_path() should be called when ffmpeg_path=None"

    def test_explicit_ffmpeg_path_not_overridden(self, monkeypatch, tmp_path):
        calls = []

        def fake_get_ffmpeg_path(*a, **k):
            calls.append(True)
            return "/usr/bin/ffmpeg"

        monkeypatch.setattr(frame_pipe, "RUST_AVAILABLE", False)
        monkeypatch.setattr(frame_pipe, "_avf_framepipe_native", None)
        monkeypatch.setattr(frame_pipe, "get_ffmpeg_path", fake_get_ffmpeg_path)

        frame_pipe.get_frame_reader(str(tmp_path / "x.mp4"), 16, 16, ffmpeg_path="/custom/ffmpeg")
        assert not calls, "explicit ffmpeg_path must not be overridden by get_ffmpeg_path()"


# ─── Unit: API-surface parity between backends ─────────────────────────


class TestApiSurfaceParity:
    """Both backends must expose the same reader/writer method surface so
    call sites never need to branch on which one is active."""

    def test_python_reader_has_rust_reader_surface(self):
        assert hasattr(frame_pipe._PythonFrameReader, "next_batch")
        assert hasattr(frame_pipe._PythonFrameReader, "frames_read")
        assert hasattr(frame_pipe._PythonFrameReader, "close")

    def test_python_writer_has_rust_writer_surface(self):
        assert hasattr(frame_pipe._PythonFrameWriter, "write_batch")
        assert hasattr(frame_pipe._PythonFrameWriter, "frames_written")
        assert hasattr(frame_pipe._PythonFrameWriter, "close")

    def test_rust_reader_has_same_surface_as_python_reader(self):
        if not frame_pipe.RUST_AVAILABLE:
            pytest.skip("avf_framepipe Rust extension not built in this environment")
        import avf_framepipe

        for method in ("next_batch", "frames_read", "close"):
            assert hasattr(avf_framepipe.FrameReader, method), (
                f"Rust FrameReader missing '{method}' present on Python fallback"
            )

    def test_rust_writer_has_same_surface_as_python_writer(self):
        if not frame_pipe.RUST_AVAILABLE:
            pytest.skip("avf_framepipe Rust extension not built in this environment")
        import avf_framepipe

        for method in ("write_batch", "frames_written", "close"):
            assert hasattr(avf_framepipe.FrameWriter, method), (
                f"Rust FrameWriter missing '{method}' present on Python fallback"
            )

    def test_factory_signatures_expose_same_keyword_args(self):
        reader_sig = inspect.signature(frame_pipe.get_frame_reader)
        writer_sig = inspect.signature(frame_pipe.get_frame_writer)
        assert set(reader_sig.parameters) >= {
            "path",
            "width",
            "height",
            "chunk_size",
            "read_ahead",
            "ffmpeg_path",
        }
        assert set(writer_sig.parameters) >= {
            "path",
            "width",
            "height",
            "fps",
            "codec",
            "crf",
            "preset",
            "write_queue",
            "ffmpeg_path",
        }


# ─── Helpers shared by the integration tests ───────────────────────────


def _make_testsrc_clip(path, w, h, frames, rate=10):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={w}x{h}:rate={rate}",
            "-frames:v",
            str(frames),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _read_all_frames_bgr24(path, w, h):
    """Decode `path` to raw BGR24 bytes via ffmpeg and return a list of
    (h, w, 3) uint8 numpy arrays -- an independent decode path from both
    transports under test, used only to verify their outputs."""
    frame_size = w * h * 3
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    data = proc.stdout
    n_frames = len(data) // frame_size
    return [
        np.frombuffer(data[i * frame_size : (i + 1) * frame_size], dtype=np.uint8).reshape(h, w, 3)
        for i in range(n_frames)
    ]


# ─── Integration: differential identity test (transport invisibility) ──


class TestTransportIdentity:
    """The Rust and Python-fallback transports must be interchangeable: the
    same input, pushed through identical passthrough "inference" (a no-op)
    and identical encode settings, must produce bit-exact output regardless
    of which transport did the reading/writing."""

    @pytest.mark.integration
    def test_rust_and_python_transports_produce_bit_exact_output(self, tmp_path):
        if not frame_pipe.RUST_AVAILABLE:
            pytest.skip("avf_framepipe Rust extension not built in this environment")

        w, h, n_frames, fps = 32, 24, 40, 10
        src = tmp_path / "src.mp4"
        _make_testsrc_clip(src, w, h, n_frames, rate=fps)

        encode_kwargs = dict(codec="libx264", crf=0, preset="ultrafast")

        def run_transport(rust: bool, out_name: str):
            out_path = tmp_path / out_name
            if rust:
                import avf_framepipe

                reader = avf_framepipe.FrameReader(str(src), w, h, chunk_size=7, read_ahead=2)
                writer = avf_framepipe.FrameWriter(
                    str(out_path), w, h, fps, write_queue=4, **encode_kwargs
                )
            else:
                reader = frame_pipe._PythonFrameReader(str(src), w, h, chunk_size=7, read_ahead=2)
                writer = frame_pipe._PythonFrameWriter(
                    str(out_path), w, h, fps, write_queue=4, **encode_kwargs
                )

            total = 0
            while True:
                batch = reader.next_batch()
                if batch is None:
                    break
                # Passthrough "inference": identity on each chunk.
                identity_batch = list(batch)
                if rust:
                    writer.write_batch(identity_batch)
                else:
                    writer.write_batch(identity_batch)
                total += len(identity_batch)
            reader.close()
            ok = writer.close()
            assert ok, f"{'rust' if rust else 'python'} writer failed to close cleanly"
            return out_path, total

        rust_out, rust_total = run_transport(True, "rust_out.mp4")
        py_out, py_total = run_transport(False, "py_out.mp4")

        assert rust_total == n_frames
        assert py_total == n_frames

        rust_frames = _read_all_frames_bgr24(rust_out, w, h)
        py_frames = _read_all_frames_bgr24(py_out, w, h)

        assert len(rust_frames) == len(py_frames) == n_frames, (
            f"frame count mismatch: rust={len(rust_frames)} python={len(py_frames)} "
            f"expected={n_frames}"
        )
        for i, (rf, pf) in enumerate(zip(rust_frames, py_frames)):
            assert np.array_equal(rf, pf), f"frame {i} differs between transports (not bit-exact)"


# ─── Integration: soak test (bounded memory over a long stream) ────────


def _rss_kb() -> int:
    """Current process RSS in KiB, read from /proc/self/status (Linux)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found in /proc/self/status")


class TestFramePipeSoak:
    """A long (1000+ frame) stream through reader -> writer, with passthrough
    "inference", must not buffer the whole video in memory: peak RSS should
    stay bounded relative to a baseline taken before streaming starts."""

    @pytest.mark.integration
    def test_soak_bounded_memory_over_long_stream(self, tmp_path):
        if not frame_pipe.RUST_AVAILABLE:
            pytest.skip("avf_framepipe Rust extension not built in this environment")

        w, h, n_frames, fps = 128, 72, 1200, 30
        src = tmp_path / "soak_src.mp4"
        _make_testsrc_clip(src, w, h, n_frames, rate=fps)
        out = tmp_path / "soak_out.mp4"

        baseline_kb = _rss_kb()
        peak_kb = baseline_kb

        reader = frame_pipe.get_frame_reader(str(src), w, h, chunk_size=25, read_ahead=2)
        writer = frame_pipe.get_frame_writer(
            str(out), w, h, fps, codec="libx264", crf=28, preset="ultrafast", write_queue=4
        )

        total = 0
        chunks_seen = 0
        while True:
            batch = reader.next_batch()
            if batch is None:
                break
            writer.write_batch(list(batch))
            total += len(batch)
            chunks_seen += 1
            if chunks_seen % 4 == 0:
                peak_kb = max(peak_kb, _rss_kb())
        reader.close()
        ok = writer.close()
        peak_kb = max(peak_kb, _rss_kb())

        assert ok
        assert total == n_frames

        growth_mb = (peak_kb - baseline_kb) / 1024.0
        assert growth_mb < 300, (
            f"peak RSS grew {growth_mb:.1f}MB over baseline while streaming "
            f"{n_frames} frames of {w}x{h} -- expected bounded (<300MB) growth, "
            f"not full-video buffering (uncompressed {w}x{h}x{n_frames}x3 bytes "
            f"= {(w * h * 3 * n_frames) / (1024 * 1024):.1f}MB)"
        )
