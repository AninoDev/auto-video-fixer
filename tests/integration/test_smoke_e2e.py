"""Real-FFmpeg end-to-end smoke tests.

These exercise the pipeline against tiny, real (not mocked) media files produced
by FFmpeg's ``testsrc2`` lavfi source, in both a WebM (VP9/Opus) and an MP4
(H.264/AAC) container. The WebM case is a regression test for a bug where
intermediate temp files inherited the *input's* container extension
(``generate_temp_path``) while stages hardcoded H.264 -- writing H.264 into a
``.webm``-suffixed temp file made ffmpeg reject the container outright, which
surfaced as a "Broken pipe" in the stabilize stage's decode->transform pipe.

Run with: pytest tests/integration/test_smoke_e2e.py -v -m integration
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_video(path: str, container: str) -> None:
    """Generate a tiny 2s/320x180/30fps real video with FFmpeg.

    container: "webm" (libvpx-vp9 + libopus) or "mp4" (libx264 + aac).
    """
    if container == "webm":
        video_args = ["-c:v", "libvpx-vp9", "-c:a", "libopus"]
    elif container == "mp4":
        video_args = ["-c:v", "libx264", "-c:a", "aac"]
    else:
        raise ValueError(container)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=2:size=320x180:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=duration=2",
        "-shortest",
        *video_args,
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _no_orphaned_temp_files(*directories: str) -> bool:
    for d in directories:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith(".avf_"):
                return False
    return True


def _ffprobe_video_codec(path: str) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    assert streams, f"no video stream found in {path}"
    return streams[0]["codec_name"]


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not on PATH")
class TestSmokeE2E:
    """Full pipeline run against real, tiny videos in both webm and mp4 containers."""

    def _run_job(self, tmp_path, container: str):
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_path = str(input_dir / f"source.{container}")
        _make_video(input_path, container)

        config = Config(tmp_path / "config.yaml")
        config.set(False, "general", "use_ai")
        config.set(str(output_dir), "general", "output_dir")

        pipeline = Pipeline(config)
        job = pipeline.add_job(input_path, stage_names=["stabilize", "encode"])
        result = pipeline.execute_job(job)

        return result, str(input_dir), str(output_dir)

    def test_webm_input_succeeds_and_produces_h264_output(self, tmp_path):
        """Regression test: .webm input must not die at the first re-encoding
        stage because its intermediate temp file inherited a .webm extension
        while the stage wrote H.264 into it."""
        result, input_dir, output_dir = self._run_job(tmp_path, "webm")

        assert result.success, f"job failed: {result.errors}"
        assert result.output_path is not None
        assert os.path.exists(result.output_path)
        assert _ffprobe_video_codec(result.output_path) == "h264"
        assert _no_orphaned_temp_files(input_dir, output_dir)

    def test_mp4_input_succeeds_and_produces_h264_output(self, tmp_path):
        result, input_dir, output_dir = self._run_job(tmp_path, "mp4")

        assert result.success, f"job failed: {result.errors}"
        assert result.output_path is not None
        assert os.path.exists(result.output_path)
        assert _ffprobe_video_codec(result.output_path) == "h264"
        assert _no_orphaned_temp_files(input_dir, output_dir)

    def test_failed_job_leaves_no_orphaned_temp_files(self, tmp_path):
        """An unreadable/corrupt input should fail the job cleanly, without
        leaving any .avf_* intermediate temp files behind.

        Probing an unreadable file raises before any temp path is generated, so
        this mainly guards the invariant end-to-end (via the same execute_all()
        entrypoint the CLI uses, which catches the probe failure and reports it
        as a failed JobResult) rather than exercising the new mid-stage cleanup
        path directly.
        """
        from autovideofixer.config import Config
        from autovideofixer.core.pipeline import Pipeline

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Not a real video: probing/decoding will fail, but the file exists so
        # add_job()'s existence check passes and execute_job() actually runs.
        bad_input = input_dir / "broken.mp4"
        bad_input.write_bytes(b"not a real video file")

        config = Config(tmp_path / "config2.yaml")
        config.set(False, "general", "use_ai")

        pipeline = Pipeline(config)
        pipeline.add_job(str(bad_input), stage_names=["stabilize", "encode"])
        results = pipeline.execute_all()

        assert len(results) == 1
        assert not results[0].success
        assert _no_orphaned_temp_files(str(input_dir))
